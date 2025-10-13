"""Riemannian Walk (RWalk) learner wired for the repo training loop.

The original script expected to orchestrate its own epochs and data loading.
This version keeps the same Fisher and path-integral bookkeeping but exposes the
``Net``/``observe`` interface so ``main.py`` can drive it batch-by-batch.  A
``ResNet1D`` backbone supplies the feature extractor to stay consistent with the
rest of the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D


@dataclass
class RWalkConfig:
    """Hyper-parameters harvested from ``args`` with safe fallbacks."""

    lr: float = 0.001
    optimizer: str = "adam"
    momentum: float = 0.0
    weight_decay: float = 0.0
    clipgrad: Optional[float] = 100.0
    lamb: float = 1.0
    alpha: float = 0.9
    eps: float = 0.01

    @staticmethod
    def from_args(args: object | None) -> "RWalkConfig":
        cfg = RWalkConfig()
        if args is None:
            return cfg

        for field in ("lr", "optimizer", "momentum", "weight_decay", "lamb", "alpha", "eps"):
            if hasattr(args, field):
                value = getattr(args, field)
                if value is not None:
                    setattr(cfg, field, value)
        if hasattr(args, "clipgrad") and getattr(args, "clipgrad") is not None:
            cfg.clipgrad = getattr(args, "clipgrad")
        elif hasattr(args, "clipgrad_norm") and getattr(args, "clipgrad_norm") is not None:
            cfg.clipgrad = getattr(args, "clipgrad_norm")

        parameter = getattr(args, "parameter", "")
        if isinstance(parameter, str) and parameter:
            try:
                cfg.lamb = float(parameter.split(",")[0])
            except (ValueError, IndexError):
                pass

        return cfg


class Net(nn.Module):
    """RWalk continual learner built on top of ``ResNet1D``."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object | None) -> None:
        super().__init__()
        del n_inputs  # The ResNet1D front-end dictates its own receptive field

        assert n_tasks > 0, "RWalk requires at least one task"
        assert n_outputs % n_tasks == 0, "RWalk assumes balanced classes per task"

        self.cfg = RWalkConfig.from_args(args)
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.nc_per_task = n_outputs // n_tasks
        self.is_task_incremental = getattr(args, "class_incremental", True)

        self.net = ResNet1D(n_outputs, args)
        self.ce = nn.CrossEntropyLoss()
        self.opt = self._build_optimizer()

        self.lamb = float(self.cfg.lamb)
        self.alpha = float(self.cfg.alpha)
        self.eps = float(self.cfg.eps)
        self.clipgrad = self.cfg.clipgrad

        self.current_task: Optional[int] = None
        self.tasks_trained: int = 0

        self.fisher: Dict[str, torch.Tensor] = {}
        self.s: Dict[str, torch.Tensor] = {}
        self.fisher_running: Dict[str, torch.Tensor] = {}
        self.s_running: Dict[str, torch.Tensor] = {}
        self.p_old: Dict[str, torch.Tensor] = {}
        self.param_star: Dict[str, torch.Tensor] = {}

        self._initialise_state()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:  # pragma: no cover - thin wrapper
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
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> float:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self._consolidate_current_task()
            self.current_task = t

        self.net.train()
        self.opt.zero_grad()

        logits = self.net(x)
        offset1, offset2 = (0, self.n_outputs)
        targets = y.long()
        if self.is_task_incremental:
            offset1, offset2 = self._compute_offsets(t)
            logits = logits[:, offset1:offset2]
            targets = (targets - offset1).long()

        loss_ce = self.ce(logits, targets)
        loss = loss_ce + self.lamb * self._regulariser()
        loss.backward()

        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        self.opt.step()
        self._update_running_statistics()

        return float(loss.item())

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook so callers can flush the last task explicitly."""
        self._consolidate_current_task()

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim = (self.cfg.optimizer or "adam").lower()
        lr = float(self.cfg.lr)
        wd = float(self.cfg.weight_decay)

        if optim in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr, weight_decay=wd)
        if optim == "adagrad":
            return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
        if optim in {"sgd", "sgd_momentum_decay"}:
            momentum = float(self.cfg.momentum)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # ------------------------------------------------------------------
    def _initialise_state(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            zero = torch.zeros_like(param)
            device = param.device
            self.fisher[name] = zero.clone().to(device)
            self.s[name] = zero.clone().to(device)
            self.fisher_running[name] = zero.clone().to(device)
            self.s_running[name] = zero.clone().to(device)
            self.p_old[name] = param.detach().clone().to(device)
            self.param_star[name] = param.detach().clone().to(device)

    # ------------------------------------------------------------------
    def _regulariser(self) -> torch.Tensor:
        if self.tasks_trained == 0:
            return torch.zeros(1, device=self._device())

        penalty = torch.zeros(1, device=self._device())
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            self._ensure_state_device(name, param)
            fisher = self.fisher.get(name)
            s_term = self.s.get(name)
            star = self.param_star.get(name)
            if fisher is None or s_term is None or star is None:
                continue
            diff = param - star
            penalty = penalty + ((fisher + s_term) * diff.pow(2)).sum()
        return penalty

    # ------------------------------------------------------------------
    def _update_running_statistics(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            self._ensure_state_device(name, param)
            grad = param.grad
            if grad is None:
                continue

            fisher_current = grad.detach().pow(2)
            prev_running = self.fisher_running[name]
            self.fisher_running[name] = self.alpha * fisher_current + (1.0 - self.alpha) * prev_running

            delta = param.detach() - self.p_old[name]
            fisher_distance = 0.5 * self.fisher_running[name] * delta.pow(2)
            loss_diff = -grad.detach() * delta
            s_update = loss_diff / (fisher_distance + self.eps)
            self.s_running[name] = self.s_running[name] + s_update.detach()

            self.p_old[name] = param.detach().clone()

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        if self.current_task is None:
            return
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            self._ensure_state_device(name, param)
            self.fisher[name] = self.fisher_running[name].detach().clone()
            s_clone = 0.5 * self.s_running[name].detach().clone()
            self.s[name] = s_clone
            self.s_running[name] = s_clone.clone()
            self.param_star[name] = param.detach().clone()
            self.p_old[name] = param.detach().clone()
        self.tasks_trained += 1

    # ------------------------------------------------------------------
    def _ensure_state_device(self, name: str, param: torch.nn.Parameter) -> None:
        """Make sure cached tensors follow the parameter device."""
        device = param.device

        def move_if_needed(t: torch.Tensor) -> torch.Tensor:
            return t.to(device) if t.device != device else t

        if name in self.fisher:
            self.fisher[name] = move_if_needed(self.fisher[name])
        if name in self.s:
            self.s[name] = move_if_needed(self.s[name])
        if name in self.fisher_running:
            self.fisher_running[name] = move_if_needed(self.fisher_running[name])
        if name in self.s_running:
            self.s_running[name] = move_if_needed(self.s_running[name])
        if name in self.p_old:
            self.p_old[name] = move_if_needed(self.p_old[name])
        if name in self.param_star:
            self.param_star[name] = move_if_needed(self.param_star[name])

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1 = task * self.nc_per_task
        offset2 = min(self.n_outputs, (task + 1) * self.nc_per_task)
        return offset1, offset2

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = ["Net"]
