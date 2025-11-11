"""Learning without Forgetting (LwF) adapted to this repository.

This implementation mirrors the behaviour of the original LwF algorithm while
complying with the ``Net`` interface expected by ``life_experience`` in
``main.py``.  A shared backbone (ResNet18/ResNet1D/MLP) produces logits for all
classes, ``observe`` performs the usual supervised update on the current task,
and a temperature-scaled distillation loss preserves knowledge about previously
seen tasks via a frozen teacher snapshot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import model.meta.learner as Learner
import model.meta.modelfactory as mf
from model.resnet import ResNet18
from model.resnet1d import ResNet1D


@dataclass
class LwfConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``."""

    lr: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.0
    weight_decay: float = 0.0
    clipgrad: Optional[float] = 100.0
    temperature: float = 2.0
    distill_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "LwfConfig":
        cfg = LwfConfig()
        for field in ("lr", "optimizer", "momentum", "weight_decay", "clipgrad"):
            if hasattr(args, field):
                value = getattr(args, field)
                if value is not None:
                    setattr(cfg, field, value)
        if getattr(cfg, "clipgrad", None) is None and hasattr(args, "grad_clip_norm"):
            cfg.clipgrad = getattr(args, "grad_clip_norm")
        # Allow multiple naming variants for the LwF knobs.
        temperature_fields = ("temperature", "distill_temperature", "lwf_temperature")
        lambda_fields = ("distill_lambda", "lwf_lambda", "lwf_distill_lambda")
        for field in temperature_fields:
            if hasattr(args, field):
                value = getattr(args, field)
                if value is not None:
                    cfg.temperature = float(value)
                    break
        for field in lambda_fields:
            if hasattr(args, field):
                value = getattr(args, field)
                if value is not None:
                    cfg.distill_lambda = float(value)
                    break
        return cfg


class Net(nn.Module):
    """LwF learner that plugs directly into the repository training loop."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()
        if n_tasks <= 0:
            raise ValueError("LwF requires a positive number of tasks")
        self.args = args
        self.cfg = LwfConfig.from_args(args)

        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        default_nc = n_outputs // n_tasks if n_tasks > 0 else n_outputs
        self.nc_per_task = getattr(args, "nc_per_task", default_nc)
        self.is_task_incremental = True

        self.net = self._build_backbone(n_inputs, n_outputs, args)
        self.opt = self._build_optimizer()

        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.temperature = float(self.cfg.temperature)
        self.distill_lambda = float(self.cfg.distill_lambda)
        self.clipgrad = self.cfg.clipgrad

        self.current_task: Optional[int] = None
        self.teacher: Optional[nn.Module] = None

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
            self._update_teacher()
            self.current_task = t

        self.net.train()
        logits = self.net(x)
        offset1, offset2 = self._compute_offsets(t)
        current_logits = logits[:, offset1:offset2]
        targets = (y - offset1).long()

        preds = torch.argmax(current_logits, dim=1)
        tr_acc = (preds == targets).float().mean().item()

        loss_ce = self.ce(current_logits, targets)
        distill_loss = self._distillation_loss(logits, x, offset1)
        loss = loss_ce + self.distill_lambda * distill_loss

        self.opt.zero_grad()
        loss.backward()
        if self.clipgrad is not None and self.clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        self.opt.step()

        return float(loss.item()), tr_acc

    # ------------------------------------------------------------------
    def _build_backbone(self, n_inputs: int, n_outputs: int, args: object) -> nn.Module:
        arch = getattr(args, "arch", "resnet1d")
        arch = arch.lower() if isinstance(arch, str) else "resnet1d"
        if arch == "resnet18":
            return ResNet18(n_outputs, args)
        if arch == "resnet1d":
            return ResNet1D(n_outputs, args)

        nl = getattr(args, "n_layers", 2)
        nh = getattr(args, "n_hiddens", 100)
        config = mf.ModelFactory.get_model(
            model_type=arch,
            sizes=[n_inputs] + [nh] * nl + [n_outputs],
            dataset=getattr(args, "dataset", ""),
            args=args,
        )
        return Learner.Learner(config, args)

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim_name = (self.cfg.optimizer or "adam").lower()
        lr = float(self.cfg.lr)
        wd = float(self.cfg.weight_decay)

        if optim_name in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim_name == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr, weight_decay=wd)
        if optim_name == "adagrad":
            return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
        momentum = float(self.cfg.momentum)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1 = task * self.nc_per_task
        offset2 = min(self.n_outputs, offset1 + self.nc_per_task)
        return offset1, offset2

    # ------------------------------------------------------------------
    def _distillation_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor, previous_classes: int
    ) -> torch.Tensor:
        if self.teacher is None or previous_classes == 0:
            return torch.zeros(1, device=student_logits.device)

        with torch.no_grad():
            teacher_logits = self.teacher(x)
            teacher_logits = teacher_logits[:, :previous_classes]
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        student_log_probs = F.log_softmax(
            student_logits[:, :previous_classes] / self.temperature, dim=1
        )
        loss = self.kl(student_log_probs, teacher_probs) * (self.temperature ** 2)
        return loss

    # ------------------------------------------------------------------
    def _update_teacher(self) -> None:
        device = self._device()
        self.teacher = copy.deepcopy(self.net)
        self.teacher.to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = ["Net"]
