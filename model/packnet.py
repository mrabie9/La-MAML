"""PackNet continual learner compatible with ``main.py``'s training loop.

This rewrite keeps the core idea of PackNet—iterative pruning and weight
freezing between tasks—while exposing the same ``Net`` API as the other models
in the repository.  After each task finishes, important weights are "packed"
and frozen via magnitude-based masking and future tasks reuse only the
remaining free parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import model.meta.learner as Learner
import model.meta.modelfactory as mf
from model.resnet import ResNet18
from model.resnet1d import ResNet1D


@dataclass
class PackNetConfig:
    """Hyper-parameters pulled from ``args`` with sensible defaults."""

    lr: float = 0.01
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    clipgrad: Optional[float] = 100.0
    prune_perc: float = 0.5  # fraction of currently used weights to prune

    @staticmethod
    def from_args(args: object) -> "PackNetConfig":
        cfg = PackNetConfig()
        for field in ("lr", "optimizer", "momentum", "weight_decay", "clipgrad"):
            if hasattr(args, field):
                value = getattr(args, field)
                if value is not None:
                    setattr(cfg, field, value)
        prune_fields = ("packnet_prune_perc", "prune_perc_per_layer", "packnet_prune_frac")
        for field in prune_fields:
            if hasattr(args, field):
                value = getattr(args, field)
                if value is not None:
                    cfg.prune_perc = float(value)
                    break
        cfg.prune_perc = float(max(0.0, min(1.0, cfg.prune_perc)))
        return cfg


class Net(nn.Module):
    """PackNet learner with ResNet/MLP backbones and task-aware pruning."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()

        if n_tasks <= 0:
            raise ValueError("PackNet requires at least one task")
        if n_outputs % n_tasks != 0:
            raise ValueError("PackNet assumes a balanced number of classes per task")

        self.args = args
        self.cfg = PackNetConfig.from_args(args)
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.nc_per_task = n_outputs // n_tasks
        self.is_task_incremental = True

        self.net = self._build_backbone(n_inputs, n_outputs, args)
        self.ce = nn.CrossEntropyLoss()
        self.opt = self._build_optimizer()

        self.clipgrad = self.cfg.clipgrad
        self.prune_perc = self.cfg.prune_perc

        self.current_task: Optional[int] = None
        self._param_to_buffers: Dict[str, Tuple[str, str]] = {}
        self._init_masks_and_frozen()

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
            self._pack_current_task()
            self.current_task = t

        self.net.train()
        logits = self.net(x)
        offset1, offset2 = self._compute_offsets(t)
        logits = logits[:, offset1:offset2]
        targets = (y - offset1).long()

        loss = self.ce(logits, targets)
        preds = torch.argmax(logits, dim=1)
        tr_acc = (preds == targets).float().mean().item()

        self.opt.zero_grad()
        loss.backward()
        self._zero_frozen_grads()
        if self.clipgrad is not None and self.clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        self.opt.step()
        self._restore_frozen_weights()

        return float(loss.item()), tr_acc

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook if the training loop signals explicit task boundaries."""
        self._pack_current_task()

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
        optim_name = (self.cfg.optimizer or "sgd").lower()
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
    def _init_masks_and_frozen(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            key = name.replace(".", "__")
            mask_name = f"{key}_mask"
            frozen_name = f"{key}_frozen"
            self.register_buffer(mask_name, torch.zeros_like(param, dtype=torch.bool))
            self.register_buffer(frozen_name, param.detach().clone())
            self._param_to_buffers[name] = (mask_name, frozen_name)

    # ------------------------------------------------------------------
    def _get_mask_and_frozen(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_name, frozen_name = self._param_to_buffers[name]
        return getattr(self, mask_name), getattr(self, frozen_name)

    # ------------------------------------------------------------------
    def _zero_frozen_grads(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            mask, _ = self._get_mask_and_frozen(name)
            if mask.any():
                param.grad.data.masked_fill_(mask, 0.0)

    # ------------------------------------------------------------------
    def _restore_frozen_weights(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            mask, frozen = self._get_mask_and_frozen(name)
            if mask.any():
                param.data[mask] = frozen.data[mask]

    # ------------------------------------------------------------------
    def _pack_current_task(self) -> None:
        if self.current_task is None:
            return
        keep_ratio = 1.0 - float(self.prune_perc)
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        if keep_ratio == 0.0:
            # Nothing to keep; simply free all currently available weights.
            for name, param in self.net.named_parameters():
                if not param.requires_grad:
                    continue
                mask, _ = self._get_mask_and_frozen(name)
                free_mask = ~mask
                param.data[free_mask] = 0.0
            return

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            mask, frozen = self._get_mask_and_frozen(name)
            free_mask = ~mask
            free_count = int(free_mask.sum().item())
            if free_count == 0:
                continue
            keep_count = int(math.ceil(keep_ratio * free_count))
            keep_count = max(1, keep_count)

            flat_abs = param.detach().abs()[free_mask]
            if keep_count >= flat_abs.numel():
                keep_mask = free_mask.clone()
            else:
                threshold = torch.topk(flat_abs, keep_count, largest=True).values.min()
                keep_mask = free_mask & (param.detach().abs() >= threshold)

            # Lock kept weights and remember their values.
            mask |= keep_mask
            frozen.data[keep_mask] = param.detach()[keep_mask]

            # Zero out the newly freed weights for the next task.
            free_to_reset = free_mask & (~keep_mask)
            if free_to_reset.any():
                param.data[free_to_reset] = 0.0

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1 = task * self.nc_per_task
        offset2 = min(self.n_outputs, (task + 1) * self.nc_per_task)
        return offset1, offset2

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = ["Net"]
