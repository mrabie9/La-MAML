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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from model.detection_replay import (
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class PackNetConfig:
    """Hyper-parameters pulled from ``args`` with sensible defaults."""

    inner_steps: int = 1
    lr: float = 0.01
    optimizer: str = "sgd"
    n_tasks: int = 3
    prune_perc: float = 0.75  # fraction of currently used weights to prune
    clipgrad: Optional[float] = 100.0
    # Extra SGD passes on task data after packing; gradients only on owner==task.
    post_prune_epochs: int = 0

    @staticmethod
    def from_args(args: object) -> "PackNetConfig":
        cfg = PackNetConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(nn.Module):
    """PackNet learner with ResNet/MLP backbones and task-aware pruning."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()

        if n_tasks <= 0:
            raise ValueError("PackNet requires at least one task")

        self.args = args
        self.cfg = PackNetConfig.from_args(args)
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.is_task_incremental = True

        self.net = self._build_backbone(n_inputs, n_outputs, args)
        self._named_modules = dict(self.net.named_modules())
        self._non_prunable_params = self._compute_non_prunable_params()
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.noise_label: int | None = noise_label_from_args(args)
        self.incremental_loader_name = getattr(args, "loader", None)
        self.opt = self._build_optimizer()
        self.clipgrad = self.cfg.clipgrad
        self.prune_perc = self.cfg.prune_perc  # float(1 / self.n_tasks)
        print(
            f"PackNet will prune {self.prune_perc*100:.1f}% of currently used weights after each task."
        )

        self.current_task: Optional[int] = None
        self._finalized_tasks: set[int] = set()
        self._param_to_buffers: Dict[str, Tuple[str, str]] = {}
        self._init_masks_and_frozen()

        # BatchNorm handling: keep separate running stats per task to avoid
        # cross-task interference while allowing task-specific affine params.
        self._bn_modules: List[_BatchNorm] = [
            m for m in self.net.modules() if isinstance(m, _BatchNorm)
        ]
        # print(f"Found {len(self._bn_modules)} BN modules in the model.")
        # Mapping: task_id -> list of (running_mean, running_var, num_batches_tracked)
        self._bn_task_stats: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = (
            {}
        )
        # Mapping: task_id -> list of (weight, bias) snapshots (None when affine=False).
        self._bn_task_affine: Dict[
            int, List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ] = {}

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:
        # Ensure BN stats for the queried task are active (important for eval).
        if self._bn_modules and t in self._bn_task_stats:
            self._restore_bn_stats(t)

        allow_free = self.current_task is None or t >= self.current_task
        with self._apply_task_mask(t, allow_free):
            logits = self.net(x)

        if not self.is_task_incremental:
            return logits
        cil = kwargs.get("cil_all_seen_upto_task")
        return misc_utils.apply_task_incremental_logit_mask(
            logits,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil,
            global_noise_label=self.noise_label,
            loader=self.incremental_loader_name,
        )

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> Tuple[float, float]:
        if self.current_task is None:
            self.current_task = t
            self._restore_bn_stats(t)
        elif t != self.current_task:
            # Packing and BN snapshot for the previous task run in
            # ``finalize_task_after_training`` at the end of that task.
            self.current_task = t
            self._restore_bn_stats(t)

        self.net.train()
        for _ in range(self.cfg.inner_steps):
            logits = self.net(x)
            y_cls = unpack_y_to_class_labels(y)
            signal_mask = signal_mask_exclude_noise(y_cls, self.noise_label)
            logits_for_loss = logits
            if self.is_task_incremental:
                logits_for_loss = misc_utils.apply_task_incremental_logit_mask(
                    logits,
                    t,
                    self.classes_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    global_noise_label=self.noise_label,
                    loader=self.incremental_loader_name,
                )
            targets_for_loss = y_cls.long()
            loss = classification_cross_entropy(
                logits_for_loss,
                targets_for_loss,
                class_weighted_ce=self.class_weighted_ce,
            )
            if signal_mask.any():
                preds = torch.argmax(logits_for_loss[signal_mask], dim=1)
                cls_tr_rec = macro_recall(preds, y_cls[signal_mask].long())
            else:
                cls_tr_rec = 0.0

            self.opt.zero_grad()
            loss.backward()
            self._zero_frozen_grads()
            if self.clipgrad is not None and self.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
            self.opt.step()
            self._restore_frozen_weights()

        return float(loss.item()), cls_tr_rec

    # ------------------------------------------------------------------
    def finalize_task_after_training(
        self,
        train_loader: Optional[Iterable] = None,
        *,
        completed_task_index: Optional[int] = None,
    ) -> None:
        """Pack weights for the current task, snapshot BN, optional post-prune finetune.

        Call once per task after normal training for that task finishes and
        before end-of-task validation. Idempotent per ``completed_task_index``.

        Args:
            train_loader: Batches ``(x, y)`` or ``(x, y, ...)`` for this task;
                used only when ``post_prune_epochs > 0``. May be ``None`` to skip
                the finetune loop (pack + BN snapshot still run).
            completed_task_index: Task that just finished; defaults to
                ``self.current_task``.

        Usage:
            >>> # After the epoch loop for task ``t`` (``model.current_task == t``):
            >>> model.finalize_task_after_training(train_loader)  # doctest: +SKIP
        """
        task_id = (
            completed_task_index
            if completed_task_index is not None
            else self.current_task
        )
        if task_id is None:
            raise RuntimeError("finalize_task_after_training requires a current task.")
        if task_id in self._finalized_tasks:
            return
        if self.current_task != task_id:
            raise RuntimeError(
                f"finalize_task_after_training expected current_task=={task_id}, "
                f"got {self.current_task}."
            )

        self._pack_current_task()
        if self._bn_modules:
            self._snapshot_bn_stats(task_id)

        epochs = int(self.cfg.post_prune_epochs)
        if epochs > 0 and train_loader is not None:
            self.net.train()
            for _ in range(epochs):
                for batch in train_loader:
                    x, y = self._unpack_batch_for_observe(batch)
                    x, y = self._move_batch_to_model_device(x, y)
                    for _ in range(self.cfg.inner_steps):
                        with self._apply_task_mask(task_id, allow_free=False):
                            logits = self.net(x)
                        y_cls = unpack_y_to_class_labels(y)
                        logits_for_loss = logits
                        if self.is_task_incremental:
                            logits_for_loss = (
                                misc_utils.apply_task_incremental_logit_mask(
                                    logits,
                                    task_id,
                                    self.classes_per_task,
                                    self.n_outputs,
                                    cil_all_seen_upto_task=task_id,
                                    global_noise_label=self.noise_label,
                                    loader=self.incremental_loader_name,
                                )
                            )
                        loss = classification_cross_entropy(
                            logits_for_loss,
                            y_cls.long(),
                            class_weighted_ce=self.class_weighted_ce,
                        )
                        self.opt.zero_grad()
                        loss.backward()
                        self._zero_grads_keep_only_task(task_id)
                        self._zero_grads_non_prunable()
                        if self.clipgrad is not None and self.clipgrad > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.net.parameters(), self.clipgrad
                            )
                        self.opt.step()
                        self._restore_frozen_weights_older_tasks_only(task_id)

            for name, param in self._prunable_named_parameters():
                owner, frozen = self._get_owner_and_frozen(name)
                mask = owner == task_id
                if mask.any():
                    frozen.data[mask] = param.data[mask].detach()

        self._finalized_tasks.add(task_id)

    def on_task_end(self) -> None:
        """Thin alias for explicit task boundaries (pack + snapshot; no finetune)."""
        self.finalize_task_after_training(train_loader=None)

    # ------------------------------------------------------------------
    def _build_backbone(self, n_inputs: int, n_outputs: int, args: object) -> nn.Module:
        arch = getattr(args, "arch", "resnet1d")
        arch = arch.lower() if isinstance(arch, str) else "resnet1d"
        if arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {arch}; only resnet1d is available now."
            )
        return ResNet1D(n_outputs, args)

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim_name = (self.cfg.optimizer or "sgd").lower()
        lr = float(self.cfg.lr)

        if optim_name in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim_name == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr)
        if optim_name == "adagrad":
            return torch.optim.Adagrad(params, lr=lr)

        return torch.optim.SGD(params, lr=lr, momentum=0.9)

    # ------------------------------------------------------------------
    def _compute_non_prunable_params(self) -> set[str]:
        """Return the set of parameter names that should never be pruned.

        This includes:
        - All BatchNorm / GroupNorm affine parameters
        - Final classifier parameters
        - Parameters of any input adapters (e.g., ADC IQ adapters)
        """
        names: set[str] = set()
        module_dict = dict(self.net.named_modules())

        # Normalization layers: keep all affine params trainable and unpruned.
        for module_name, module in module_dict.items():
            if isinstance(module, (_BatchNorm, nn.GroupNorm)):
                for param_name, _ in module.named_parameters(recurse=False):
                    full = f"{module_name}.{param_name}" if module_name else param_name
                    names.add(full)

        # Final classifier layer(s): never prune.
        for name, param in self.net.named_parameters():
            if self._is_classifier_param(name, param, module_dict):
                names.add(name)

        # Input adapters (e.g., ADC IQ adapter): do not prune or partition.
        for module_name, module in module_dict.items():
            if module_name.endswith("input_adapter"):
                for param_name, _ in module.named_parameters(recurse=False):
                    full = f"{module_name}.{param_name}" if module_name else param_name
                    names.add(full)

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
            owner_name = (
                f"{key}_owner"  # Name of buffer tracking which task owns each weight
            )
            frozen_name = f"{key}_frozen"  # Name of buffer ...
            self.register_buffer(
                owner_name, torch.full_like(param, fill_value=-1, dtype=torch.long)
            )
            self.register_buffer(frozen_name, param.detach().clone())
            self._param_to_buffers[name] = (
                owner_name,
                frozen_name,
            )  # Map param name to its buffers

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
            frozen_positions = owner >= 0  # Frozen if param has an owner
            if frozen_positions.any():
                param.data[frozen_positions] = frozen.data[
                    frozen_positions
                ]  # Restore frozen weights

    def _restore_frozen_weights_older_tasks_only(self, task_id: int) -> None:
        """Restore packed values for tasks ``< task_id`` only (keeps task_id slots trainable)."""
        for name, param in self._prunable_named_parameters():
            owner, frozen = self._get_owner_and_frozen(name)
            restore_mask = (owner >= 0) & (owner < task_id)
            if restore_mask.any():
                param.data[restore_mask] = frozen.data[restore_mask]

    def _zero_grads_keep_only_task(self, task_id: int) -> None:
        """Zero prunable gradients except at positions owned by ``task_id``."""
        for name, param in self._prunable_named_parameters():
            if param.grad is None:
                continue
            owner, _ = self._get_owner_and_frozen(name)
            keep = owner == task_id
            if keep.all():
                continue
            param.grad.data.masked_fill_(~keep, 0.0)

    def _zero_grads_non_prunable(self) -> None:
        """Zero gradients on BN/classifier/adapter params (post-prune finetune)."""
        for name, param in self.net.named_parameters():
            if name in self._non_prunable_params and param.grad is not None:
                param.grad.zero_()

    @staticmethod
    def _unpack_batch_for_observe(batch: object) -> Tuple[torch.Tensor, object]:
        """Split a dataloader batch into ``(x, y)`` (tensors may still be on CPU)."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch[0], batch[1]
            elif len(batch) >= 3:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("Batch must have at least two elements.")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if isinstance(y, (tuple, list)) and len(y) == 2:
            cls_part, det_part = y[0], y[1]
            if not torch.is_tensor(cls_part):
                cls_part = torch.as_tensor(cls_part)
            if not torch.is_tensor(det_part):
                det_part = torch.as_tensor(det_part)
            y = (cls_part, det_part)
        else:
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
        return x, y

    def _move_batch_to_model_device(
        self, x: torch.Tensor, y: object
    ) -> Tuple[torch.Tensor, object]:
        """Place ``x`` and ``y`` on the same device as ``self.net``."""
        device = self._device()
        x = x.to(device)
        if isinstance(y, tuple) and len(y) == 2:
            y = (y[0].to(device), y[1].to(device))
        elif torch.is_tensor(y):
            y = y.to(device)
        return x, y

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
            free_mask = owner < 0  # Weights not yet owned by any task
            free_count = int(free_mask.sum().item())
            if free_count == 0:
                continue
            keep_count = int(math.ceil(keep_ratio * free_count))
            keep_count = max(1, keep_count)

            flat_abs = param.detach().abs()[free_mask]
            if keep_count >= flat_abs.numel():
                keep_mask = free_mask.clone()
            else:
                threshold = torch.topk(
                    flat_abs, keep_count, largest=True
                ).values.min()  # Params to keep
                keep_mask = free_mask & (
                    param.detach().abs() >= threshold
                )  # Mask of params to keep

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
            allowed |= (owner >= 0) & (
                owner <= task
            )  # Allow owned by current or previous tasks
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
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device

    # ------------------------------------------------------------------
    # BatchNorm per-task statistics helpers
    # ------------------------------------------------------------------
    def _snapshot_bn_stats(self, task: int) -> None:
        """Store running stats and affine params for all BN layers for the given task."""
        stats: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        affine: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        for bn in self._bn_modules:
            running_mean = bn.running_mean.detach().clone()
            running_var = bn.running_var.detach().clone()
            num_batches = int(bn.num_batches_tracked.item())
            stats.append((running_mean, running_var, num_batches))
        self._bn_task_stats[task] = stats
        for bn in self._bn_modules:
            if bn.affine:
                w = bn.weight.detach().clone()
                b = bn.bias.detach().clone()
            else:
                w = None
                b = None
            affine.append((w, b))
        self._bn_task_affine[task] = affine
        # print(f"Snapshot BN stats and affine params for task {task}, stats: {stats}, affine: {affine}")

    def _restore_bn_stats(self, task: int) -> None:
        """Load BN running stats and affine params for ``task`` or reset if unseen."""
        if not self._bn_modules:
            return

        stats = self._bn_task_stats.get(task)
        affine = self._bn_task_affine.get(task)
        if stats is None:
            # Fresh task: reset stats to defaults. Affine params are left as-is
            # so new tasks can initialize from the most recently trained state.
            for bn in self._bn_modules:
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0)
                bn.num_batches_tracked.zero_()
            return

        if affine is None:
            raise RuntimeError(
                f"BatchNorm affine snapshot missing for task {task} despite saved stats."
            )
        if len(stats) != len(self._bn_modules) or len(affine) != len(self._bn_modules):
            raise RuntimeError(
                "BatchNorm state snapshot is out of sync with model BatchNorm modules."
            )
        for bn, (running_mean, running_var, num_batches) in zip(
            self._bn_modules, stats
        ):
            bn.running_mean.data.copy_(running_mean)
            bn.running_var.data.copy_(running_var)
            bn.num_batches_tracked.data.fill_(num_batches)
        for bn, (w, b) in zip(self._bn_modules, affine):
            if w is not None and bn.affine:
                bn.weight.data.copy_(w)
                bn.bias.data.copy_(b)


__all__ = ["Net"]
