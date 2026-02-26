"""Elastic Weight Consolidation (EWC) learner compatible with ``main.py``.

This version mirrors the behaviour of the original script—keeping the Fisher
information based regulariser and the per-task parameter snapshots—while
adapting it to the common ``Net`` interface used throughout the repository.  A
``ResNet1D`` backbone supplies task-agnostic features, and all interaction with
training happens through ``observe`` so it plugs directly into
``life_experience``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class EwcConfig:
    """Hyper-parameters pulled from ``args`` with sensible fallbacks."""

    lr: float = 0.03
    optimizer: str = "sgd"
    lamb: float = 1.0
    clipgrad: float = 5.0
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 32

    @staticmethod
    def from_args(args: object) -> "EwcConfig":
        cfg = EwcConfig()
        # Override defaults with any args attributes that match
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        if hasattr(args, "clipgrad") and not hasattr(args, "clipgrad_norm"):
            cfg.clipgrad = getattr(args, "clipgrad")
        if hasattr(args, "lamb"):
            cfg.lamb = getattr(args, "lamb")
        return cfg


class Net(DetectionReplayMixin, nn.Module):
    """EWC continual learner built on top of ``ResNet1D``."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()

        assert n_tasks > 0, "EWC requires a positive number of tasks"

        self.cfg = EwcConfig.from_args(args)
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

        self.net = ResNet1D(n_outputs, args)

        self.opt = self._build_optimizer()
        self.ce = nn.CrossEntropyLoss()

        self.lamb = float(self.cfg.lamb)
        self.clipgrad = float(self.cfg.clipgrad) if self.cfg.clipgrad > 0 else None
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.current_task: Optional[int] = None
        self._tasks_consolidated = 0

        self.fisher: Dict[str, torch.Tensor] = {}
        self.param_star: Dict[str, torch.Tensor] = {}

        self._fisher_accum: Optional[Dict[str, torch.Tensor]] = None
        self._fisher_count: int = 0
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, return_det: bool = False) -> torch.Tensor:
        det_logits, cls_logits = self._forward_heads(x)
        offset1, offset2 = self._compute_offsets(t)
        masked = cls_logits.clone()
        if offset1 > 0:
            masked[:, :offset1] = masked[:, :offset1].new_full(masked[:, :offset1].shape, -1e9)
        if offset2 < self.n_outputs:
            masked[:, offset2:] = masked[:, offset2:].new_full(masked[:, offset2:].shape, -1e9)
        if return_det:
            return det_logits, masked
        return masked

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> Tuple[float, float]:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self._consolidate_current_task()
            self.current_task = t

        self.net.train()

        class_counts = getattr(self, "classes_per_task", None)
        noise_label = None
        if class_counts is not None:
            _, offset2 = misc_utils.compute_offsets(t, class_counts)
            noise_label = offset2 - 1
        y_cls, y_det = self._unpack_labels(
            y,
            noise_label=noise_label,
            use_detector_arch=bool(getattr(self, "det_enabled", False)),
        )
        if y_det is None: print("Warning: y_det is None in Observe().")
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        det_logits, cls_logits = self._forward_heads(x)
        offset1, offset2 = self._compute_offsets(t)
        valid_mask = (y_det == 1) & (y_cls >= 0)
        if valid_mask.any():
            logits_task = cls_logits[valid_mask][:, offset1:offset2]
            targets = (y_cls[valid_mask] - offset1).long()
            preds = torch.argmax(logits_task, dim=1)
            tr_acc = macro_recall(preds, targets)
            loss_ce = self.ce(logits_task, targets)
        else:
            logits_task = cls_logits.new_empty((0, offset2 - offset1))
            targets = y_cls.new_empty((0,), dtype=torch.long)
            tr_acc = 0.0
            loss_ce = cls_logits.new_zeros(1)

        self.opt.zero_grad()
        if valid_mask.any():
            torch.autograd.set_detect_anomaly(True)
            loss_ce.backward(retain_graph=True)
            self._accumulate_fisher(int(valid_mask.sum().item()))

        self.opt.zero_grad()
        det_loss = self.det_loss(det_logits, y_det.float())
        det_replay = self._sample_det_memory()
        if det_replay is not None:
            print("det_replay:", det_replay[0].shape, det_replay[1].shape)
            mem_x, mem_y = det_replay
            mem_det_logits, _ = self._forward_heads(mem_x)
            mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            det_loss = 0.5 * (det_loss + mem_loss)
        loss = (self.cls_lambda * loss_ce
                + self.det_lambda * det_loss
                + 0.5 * self.lamb * self._ewc_penalty())
        loss.backward()

        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipgrad)

        self.opt.step()

        return float(loss.item()), tr_acc

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = list(self.net.parameters())
        optim = self.cfg.optimizer.lower()
        lr = float(self.cfg.lr)

        if optim == "adam":
            return torch.optim.Adam(params, lr=lr)

        return torch.optim.SGD(params, lr=lr, momentum=0.9)

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device

    def _forward_heads(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net.forward_heads(x)


    # ------------------------------------------------------------------
    def _accumulate_fisher(self, batch_size: int) -> None:
        if self._fisher_accum is None:
            self._fisher_accum = {
                name: torch.zeros_like(param, device=param.device)
                for name, param in self.net.named_parameters()
                if param.requires_grad and not name.startswith("det_head")
            }
            self._fisher_count = 0

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            grad = param.grad
            if grad is None:
                continue
            self._fisher_accum[name] += grad.detach().clone().pow(2) * batch_size
        self._fisher_count += batch_size

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        if self._fisher_accum is None or self._fisher_count == 0:
            self._reset_fisher_accum()
            return

        scale = 1.0 / float(self._fisher_count)
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            fisher_est = self._fisher_accum.get(name)
            if fisher_est is None:
                continue
            fisher_est = fisher_est * scale
            if name in self.fisher:
                prev = self.fisher[name]
                merged = (prev * self._tasks_consolidated + fisher_est) / (self._tasks_consolidated + 1)
                self.fisher[name] = merged
            else:
                self.fisher[name] = fisher_est
            self.param_star[name] = param.detach().clone()

        self._tasks_consolidated += 1
        self._reset_fisher_accum()

    # ------------------------------------------------------------------
    def _ewc_penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros(1, device=self._device())
        penalty = torch.zeros(1, device=self._device())
        for name, param in self.net.named_parameters():
            if name not in self.fisher:
                continue
            penalty += (self.fisher[name] * (param - self.param_star[name]).pow(2)).sum()
        return penalty

    # ------------------------------------------------------------------
    def _reset_fisher_accum(self) -> None:
        self._fisher_accum = None
        self._fisher_count = 0

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook for the training harness (called if available)."""
        self._consolidate_current_task()
