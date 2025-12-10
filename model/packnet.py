"""PackNet continual learner compatible with ``main.py``'s training loop.

This rewrite keeps the core idea of PackNet—iterative pruning and weight
freezing between tasks—while exposing the same ``Net`` API as the other models
in the repository.  After each task finishes, important weights are "packed"
and frozen via magnitude-based masking and future tasks reuse only the
remaining free parameters.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall


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
        # Override defaults with any args attributes that match
        # for field in ("lr", "optimizer", "momentum", "weight_decay", "clipgrad"):
        #     if hasattr(args, field):
        #         value = getattr(args, field)
        #         if value is not None:
        #             setattr(cfg, field, value)
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
        self._named_modules = dict(self.net.named_modules())
        self._non_prunable_params = self._compute_non_prunable_params()
        self.ce = nn.CrossEntropyLoss()
        self.opt = self._build_optimizer()

        self.clipgrad = self.cfg.clipgrad
        self.prune_perc = self.cfg.prune_perc

        self.current_task: Optional[int] = None
        self._param_to_buffers: Dict[str, Tuple[str, str]] = {}
        self._init_masks_and_frozen()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:
        allow_free = self.current_task is None or t >= self.current_task
        with self._apply_task_mask(t, allow_free):
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
        tr_acc = macro_recall(preds, targets)

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
        if arch != "resnet1d":
            raise ValueError(f"Unsupported arch {arch}; only resnet1d is available now.")
        return ResNet1D(n_outputs, args)

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
    def _compute_non_prunable_params(self) -> set[str]:
        names: set[str] = set()
        module_dict = dict(self.net.named_modules())
        for module_name, module in module_dict.items():
            if isinstance(module, (_BatchNorm, nn.GroupNorm)):
                for param_name, _ in module.named_parameters(recurse=False):
                    full = f"{module_name}.{param_name}" if module_name else param_name
                    names.add(full)
        for name, param in self.net.named_parameters():
            if self._is_classifier_param(name, param, module_dict):
                names.add(name)
        return names

    # ------------------------------------------------------------------
    def _is_classifier_param(
        self, name: str, param: torch.nn.Parameter, module_dict: Dict[str, nn.Module]
    ) -> bool:
        module_name, _, _ = name.rpartition(".")
        module = module_dict.get(module_name, None)
        if isinstance(module, nn.Linear):
            if param.dim() == 2 and param.size(0) == self.n_outputs:
                return True
            if param.dim() == 1 and param.size(0) == self.n_outputs:
                return True
        if param.dim() == 2 and param.size(0) == self.n_outputs:
            return True
        if param.dim() == 1 and param.size(0) == self.n_outputs:
            return True
        return False

    # ------------------------------------------------------------------
    def _init_masks_and_frozen(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._non_prunable_params:
                continue
            key = name.replace(".", "__") 
            owner_name = f"{key}_owner" # Name of buffer tracking which task owns each weight
            frozen_name = f"{key}_frozen" # Name of buffer ...
            self.register_buffer(owner_name, torch.full_like(param, fill_value=-1, dtype=torch.long))
            self.register_buffer(frozen_name, param.detach().clone())
            self._param_to_buffers[name] = (owner_name, frozen_name) # Map param name to its buffers

    # ------------------------------------------------------------------
    def _get_owner_and_frozen(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        owner_name, frozen_name = self._param_to_buffers[name]
        return getattr(self, owner_name), getattr(self, frozen_name)

    # ------------------------------------------------------------------
    def _prunable_named_parameters(self):
        for name, param in self.net.named_parameters():
            if name in self._param_to_buffers:
                yield name, param

    # ------------------------------------------------------------------
    def _zero_frozen_grads(self) -> None:
        for name, param in self._prunable_named_parameters():
            if param.grad is None:
                continue
            owner, _ = self._get_owner_and_frozen(name)
            frozen_positions = owner >= 0
            if frozen_positions.any():
                param.grad.data.masked_fill_(frozen_positions, 0.0)

    # ------------------------------------------------------------------
    def _restore_frozen_weights(self) -> None:
        for name, param in self._prunable_named_parameters():
            owner, frozen = self._get_owner_and_frozen(name)
            frozen_positions = owner >= 0 # Frozen if param has an owner
            if frozen_positions.any():
                param.data[frozen_positions] = frozen.data[frozen_positions] # Restore frozen weights

    # ------------------------------------------------------------------
    def _pack_current_task(self) -> None:
        if self.current_task is None:
            return
        keep_ratio = 1.0 - float(self.prune_perc)
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        if keep_ratio == 0.0:
            # Nothing to keep; simply free all currently available weights.
            for name, param in self._prunable_named_parameters():
                owner, _ = self._get_owner_and_frozen(name)
                free_mask = owner < 0
                if free_mask.any():
                    param.data[free_mask] = 0.0
            return

        for name, param in self._prunable_named_parameters():
            owner, frozen = self._get_owner_and_frozen(name)
            free_mask = owner < 0 # Weights not yet owned by any task
            free_count = int(free_mask.sum().item())
            if free_count == 0:
                continue
            keep_count = int(math.ceil(keep_ratio * free_count))
            keep_count = max(1, keep_count)

            flat_abs = param.detach().abs()[free_mask]
            if keep_count >= flat_abs.numel():
                keep_mask = free_mask.clone()
            else:
                threshold = torch.topk(flat_abs, keep_count, largest=True).values.min() # Params to keep
                keep_mask = free_mask & (param.detach().abs() >= threshold) # Mask of params to keep

            # Lock kept weights and remember their values.
            owner[keep_mask] = self.current_task
            frozen.data[keep_mask] = param.detach()[keep_mask]

            # Zero out the newly freed weights for the next task.
            free_to_reset = free_mask & (~keep_mask)
            if free_to_reset.any():
                param.data[free_to_reset] = 0.0
                owner[free_to_reset] = -1
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _apply_task_mask(self, task: int, allow_free: bool):
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
        for name, param in self._prunable_named_parameters():
            owner, _ = self._get_owner_and_frozen(name)
            allowed = torch.zeros_like(owner, dtype=torch.bool)
            if allow_free:
                allowed |= owner < 0
            allowed |= (owner >= 0) & (owner <= task) # Allow owned by current or previous tasks
            disallowed = ~allowed
            if not disallowed.any():
                continue
            backup_vals = param.data[disallowed].clone()
            backups.append((param, disallowed, backup_vals))
            param.data[disallowed] = 0.0
        try:
            yield
        finally:
            for param, disallowed, backup_vals in reversed(backups):
                param.data[disallowed] = backup_vals

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1 = task * self.nc_per_task
        offset2 = min(self.n_outputs, (task + 1) * self.nc_per_task)
        return offset1, offset2

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = ["Net"]
