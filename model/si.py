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
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class SiConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``."""

    lr: float = 0.001
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    clipgrad: Optional[float] = 100.0
    si_c: float = 0.1
    si_epsilon: float = 0.01

    @staticmethod
    def from_args(args: object) -> "SiConfig":
        cfg = SiConfig()
        # for field in ("lr", "optimizer", "momentum", "weight_decay", "clipgrad"):
        #     if hasattr(args, field):
        #         value = getattr(args, field)
        #         if value is not None:
        #             setattr(cfg, field, value)
        # if hasattr(args, "clipgrad_norm") and not hasattr(args, "clipgrad"):
        #     cfg.clipgrad = getattr(args, "clipgrad_norm")
        # if hasattr(args, "c"):
        #     cfg.si_c = getattr(args, "c")
        # if hasattr(args, "si_c"):
        #     cfg.si_c = getattr(args, "si_c")
        # if hasattr(args, "epsilon"):
        #     cfg.si_epsilon = getattr(args, "epsilon")
        # if getattr(args, "experiment", None) == "split_notmnist" and not hasattr(args, "epsilon"):
        #     cfg.si_epsilon = 0.001
        return cfg


class Net(nn.Module):
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

        logits = self.net(x)
        offset1, offset2 = self._compute_offsets(t) if self.is_task_incremental else (0, self.n_outputs)
        targets = y.long()
        if self.is_task_incremental:
            logits = logits[:, offset1:offset2]
            targets = (targets - offset1).long()

        loss = self.ce(logits, targets)
        preds = torch.argmax(logits, dim=1)
        tr_acc = macro_recall(preds, targets)
        loss = loss + self.si_c * self._surrogate_loss()

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
            momentum = float(self.cfg.momentum)
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        return torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    def _initialise_si_state(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
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
