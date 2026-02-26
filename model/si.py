"""Synaptic Intelligence learner compatible with the repo training loop.

This rewrite mirrors the original SI implementation while exposing the same
``Net`` interface used elsewhere.  A ResNet1D backbone provides the shared
feature extractor and the SI path-integral bookkeeping lives inside the class so
``main.py`` can drive it through ``forward``/``observe``.
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
class SiConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``."""

    lr: float = 0.001
    si_c: float = 0.1
    si_epsilon: float = 0.01
    
    optimizer: str = "sgd"
    clipgrad: Optional[float] = 100.0
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "SiConfig":
        cfg = SiConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class Net(DetectionReplayMixin, nn.Module):
    """Synaptic Intelligence continual learner built on ``ResNet1D``."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()
        del n_inputs  # ResNet1D fixes its own receptive field

        assert n_tasks > 0, "SI requires at least one task"

        self.cfg = SiConfig.from_args(args)
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.is_task_incremental = True

        self.net = ResNet1D(n_outputs, args)
        self.ce = nn.CrossEntropyLoss()
        self.opt = self._build_optimizer()

        self.si_c = float(self.cfg.si_c)
        self.epsilon = float(self.cfg.si_epsilon)
        self.clipgrad = self.cfg.clipgrad
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.current_task: Optional[int] = None
        self._param_to_key: Dict[str, str] = {}
        self._initialise_si_state()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:
        logits = self.net(x)
        if not self.is_task_incremental:
            return logits
        offset1, offset2 = self._compute_offsets(t)
        masked = logits.clone()
        if offset1 > 0:
            masked[:, :offset1] = -1e9
        if offset2 < self.n_outputs:
            masked[:, offset2:] = -1e9
        return masked

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> Tuple[float, float]:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self._consolidate_current_task()
            self.current_task = t

        self.net.train()
        self.opt.zero_grad()

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
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        det_logits, cls_logits = self.net.forward_heads(x)
        offset1, offset2 = self._compute_offsets(t) if self.is_task_incremental else (0, self.n_outputs)
        valid_mask = (y_det == 1) & (y_cls >= 0)
        if valid_mask.any():
            logits = cls_logits[valid_mask]
            targets = y_cls[valid_mask].long()
            if self.is_task_incremental:
                logits = logits[:, offset1:offset2]
                targets = (targets - offset1).long()
            loss_ce = self.ce(logits, targets)
            preds = torch.argmax(logits, dim=1)
            tr_acc = macro_recall(preds, targets)
        else:
            loss_ce = cls_logits.new_zeros(1)
            tr_acc = 0.0
        det_loss = self.det_loss(det_logits, y_det.float())
        det_replay = self._sample_det_memory()
        if det_replay is not None:
            mem_x, mem_y = det_replay
            mem_det_logits, _ = self.net.forward_heads(mem_x)
            mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            det_loss = 0.5 * (det_loss + mem_loss)

        loss = (self.cls_lambda * loss_ce
                + self.det_lambda * det_loss
                + self.si_c * self._surrogate_loss())

        loss.backward()
        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        self.opt.step()
        self._update_path_integral()

        return float(loss.item()), tr_acc

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook to consolidate the final task."""
        self._consolidate_current_task()

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim = (self.cfg.optimizer or "adam").lower()
        lr = float(self.cfg.lr)

        if optim in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr)
        if optim == "adagrad":
            return torch.optim.Adagrad(params, lr=lr)
        if optim in {"sgd", "sgd_momentum_decay"}:
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        return torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    def _initialise_si_state(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            key = name.replace('.', '__')
            self._param_to_key[name] = key
            initial = param.detach().clone()
            self.register_buffer(f"{key}_si_prev", initial.clone())
            self.register_buffer(f"{key}_si_omega", torch.zeros_like(param))
            self.register_buffer(f"{key}_si_W", torch.zeros_like(param))
            self.register_buffer(f"{key}_si_p_old", initial.clone())

    # ------------------------------------------------------------------
    def _update_path_integral(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            grad = param.grad
            if grad is None:
                continue
            key = self._param_to_key[name]
            W_buf = getattr(self, f"{key}_si_W")
            p_old_buf = getattr(self, f"{key}_si_p_old")
            W_buf.add_(-grad * (param.detach() - p_old_buf))
            p_old_buf.copy_(param.detach())

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        if self.current_task is None:
            return
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            key = self._param_to_key[name]
            prev = getattr(self, f"{key}_si_prev")
            omega = getattr(self, f"{key}_si_omega")
            W_buf = getattr(self, f"{key}_si_W")
            delta = param.detach() - prev
            omega.add_(W_buf / (delta.pow(2) + self.epsilon))
            prev.copy_(param.detach())
            W_buf.zero_()
            getattr(self, f"{key}_si_p_old").copy_(param.detach())

    # ------------------------------------------------------------------
    def _surrogate_loss(self) -> torch.Tensor:
        device = self._device()
        loss = torch.zeros(1, device=device)
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_si_omega")
            prev = getattr(self, f"{key}_si_prev")
            loss = loss + (omega * (param - prev).pow(2)).sum()
        return loss

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device



__all__ = ["Net"]
