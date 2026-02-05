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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class LwfConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``."""

    lr: float = 1e-3
    temperature: float = 5.0
    distill_lambda: float = 1.0

    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    clipgrad: Optional[float] = 100.0
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "LwfConfig":
        cfg = LwfConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class Net(DetectionReplayMixin, nn.Module):
    """LwF learner that plugs directly into the repository training loop."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()
        if n_tasks <= 0:
            raise ValueError("LwF requires a positive number of tasks")
        self.args = args
        self.cfg = LwfConfig.from_args(args)

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

        self.net = self._build_backbone(n_inputs, n_outputs, args)
        self.opt = self._build_optimizer()

        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.temperature = float(self.cfg.temperature)
        self.distill_lambda = float(self.cfg.distill_lambda)
        self.clipgrad = self.cfg.clipgrad
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(self.cfg.det_memories, self.cfg.det_replay_batch)

        self.current_task: Optional[int] = None
        self.teacher: Optional[nn.Module] = None
        # Track the global class ids that belong to each task so we can
        # correctly slice logits even when the dataset does not provide
        # contiguous class blocks per task.
        self.task_class_ids: Dict[int, List[int]] = {}
        self.task_label_maps: Dict[int, Dict[int, int]] = {}

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:
        logits = self.net(x)
        if not self.is_task_incremental:
            return logits
        class_ids = self.task_class_ids.get(t)
        if not class_ids:
            offset1, offset2 = self._compute_offsets(t)
            masked = logits.clone()
            if offset1 > 0:
                masked[:, :offset1] = -1e9
            if offset2 < self.n_outputs:
                masked[:, offset2:] = -1e9
            return masked
        masked = logits.new_full(logits.shape, -1e9)
        idx = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
        masked.index_copy_(1, idx, logits.index_select(1, idx))
        return masked

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> Tuple[float, float]:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self._update_teacher()
            self.current_task = t

        self.net.train()
        y_cls, y_det = self._unpack_labels(y)
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        det_logits, cls_logits = self.net.forward_heads(x)

        valid_mask = (y_det == 1) & (y_cls >= 0)
        if valid_mask.any():
            cls_labels = y_cls[valid_mask]
            class_ids = self._update_task_classes(t, cls_labels)
            current_logits = self._select_task_logits(cls_logits[valid_mask], class_ids)
            targets = self._map_labels_to_local(cls_labels, t)
            preds = torch.argmax(current_logits, dim=1)
            tr_acc = macro_recall(preds, targets)
            loss_ce = self.ce(current_logits, targets)
        else:
            loss_ce = cls_logits.new_zeros(1)
            tr_acc = 0.0

        prev_class_ids = self._collect_previous_class_ids(t)
        distill_loss = self._distillation_loss(cls_logits, x, prev_class_ids)

        det_loss = self.det_loss(det_logits, y_det.float())
        det_replay = self._sample_det_memory()
        if det_replay is not None:
            mem_x, mem_y = det_replay
            mem_det_logits, _ = self.net.forward_heads(mem_x)
            mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            det_loss = 0.5 * (det_loss + mem_loss)

        loss = (self.cls_lambda * loss_ce
                + self.det_lambda * det_loss
                + self.distill_lambda * distill_loss)

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
        if arch != "resnet1d":
            raise ValueError(f"Unsupported arch {arch}; only resnet1d is available now.")
        return ResNet1D(n_outputs, args)

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim_name = (self.cfg.optimizer or "adam").lower()
        lr = float(self.cfg.lr)

        if optim_name in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim_name == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr)
        if optim_name == "adagrad":
            return torch.optim.Adagrad(params, lr=lr)
        
        return torch.optim.SGD(params, lr=lr, momentum=0.9)

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _distillation_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor, prev_class_ids: List[int]
    ) -> torch.Tensor:
        if self.teacher is None or not prev_class_ids:
            return torch.zeros(1, device=student_logits.device)

        idx = torch.as_tensor(prev_class_ids, dtype=torch.long, device=student_logits.device)
        student_prev = student_logits.index_select(1, idx)
        with torch.no_grad():
            teacher_logits = self.teacher(x).index_select(1, idx)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        student_log_probs = F.log_softmax(student_prev / self.temperature, dim=1)
        loss = self.kl(student_log_probs, teacher_probs) * (self.temperature ** 2)
        return loss

    def _collect_previous_class_ids(self, current_task: int) -> List[int]:
        """Return the ordered list of class ids belonging to completed tasks."""
        class_ids: List[int] = []
        for task_id in range(current_task):
            class_ids.extend(self.task_class_ids.get(task_id, []))
        return class_ids

    def _update_task_classes(self, task: int, labels: torch.Tensor) -> List[int]:
        label_map = self.task_label_maps.setdefault(task, {})
        class_ids = self.task_class_ids.setdefault(task, [])
        labels_np = labels.detach().cpu()
        labels_np = labels_np[labels_np >= 0]
        unique_labels = torch.unique(labels_np).tolist()
        for label in unique_labels:
            label = int(label)
            if label not in label_map:
                label_map[label] = len(class_ids)
                class_ids.append(label)
        return class_ids

    def _map_labels_to_local(self, labels: torch.Tensor, task: int) -> torch.Tensor:
        label_map = self.task_label_maps[task]
        mapped = [label_map[int(lbl)] for lbl in labels.detach().cpu().tolist()]
        return torch.tensor(mapped, dtype=torch.long, device=labels.device)

    def _select_task_logits(self, logits: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
        idx = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
        return logits.index_select(1, idx)

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
