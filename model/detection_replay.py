from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class DetectionReplayMixin:
    """Shared detection replay buffer utilities for continual learners."""

    def _init_det_replay(self, det_memories: int, det_replay_batch: int) -> None:
        self.det_loss = nn.BCEWithLogitsLoss()
        self.det_memories = int(det_memories)
        self.det_replay_batch = int(det_replay_batch)
        self._det_mem_x: Optional[torch.Tensor] = None
        self._det_mem_y: Optional[torch.Tensor] = None
        self._det_mem_seen: int = 0
        self._det_mem_count: int = 0

    def _unpack_labels(
        self,
        y: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(y, (tuple, list)) and len(y) == 2:
            y_cls, y_det = y
        elif isinstance(y, dict):
            y_cls = y.get("y_cls", y.get("y"))
            y_det = y.get("y_det")
        else:
            y_cls = y
            y_det = None

        if not torch.is_tensor(y_cls):
            y_cls = torch.as_tensor(y_cls)
        if y_det is None:
            y_det = torch.ones_like(y_cls, dtype=torch.float)
        elif not torch.is_tensor(y_det):
            y_det = torch.as_tensor(y_det)

        return y_cls, y_det

    def _init_det_memory(self, sample_x: torch.Tensor) -> None:
        if self.det_memories <= 0:
            return
        sample_x = sample_x.detach().cpu()
        self._det_mem_x = torch.zeros(
            (self.det_memories,) + sample_x.shape[1:],
            dtype=sample_x.dtype,
        )
        self._det_mem_y = torch.zeros((self.det_memories,), dtype=torch.long)
        self._det_mem_seen = 0
        self._det_mem_count = 0

    def _update_det_memory(self, x: torch.Tensor, y_det: torch.Tensor) -> None:
        if self.det_memories <= 0:
            return
        if self._det_mem_x is None or self._det_mem_y is None:
            self._init_det_memory(x)
        x_cpu = x.detach().cpu()
        y_cpu = y_det.detach().cpu().long()
        batch_size = x_cpu.size(0)
        for i in range(batch_size):
            self._det_mem_seen += 1
            if self._det_mem_count < self.det_memories:
                idx = self._det_mem_count
                self._det_mem_count += 1
            else:
                j = random.randint(0, self._det_mem_seen - 1)
                if j >= self.det_memories:
                    continue
                idx = j
            self._det_mem_x[idx].copy_(x_cpu[i])
            self._det_mem_y[idx] = y_cpu[i]

    def _sample_det_memory(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._det_mem_x is None or self._det_mem_count == 0:
            return None
        batch_size = min(self.det_replay_batch, self._det_mem_count)
        indices = torch.randint(0, self._det_mem_count, (batch_size,))
        device = self._det_device()
        mem_x = self._det_mem_x.index_select(0, indices).to(device)
        mem_y = self._det_mem_y.index_select(0, indices).to(device)
        mem_x = self._prepare_det_input(mem_x)
        return mem_x, mem_y

    def _det_device(self) -> torch.device:
        if hasattr(self, "_device"):
            try:
                return self._device()
            except TypeError:
                pass
        return next(self.parameters()).device

    def _prepare_det_input(self, x: torch.Tensor) -> torch.Tensor:
        return x
