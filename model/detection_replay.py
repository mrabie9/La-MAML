from __future__ import annotations

import random
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def noise_label_from_args(args: object | None) -> int | None:
    """Return the global IQ noise class id from training args, if configured."""
    if args is None:
        return None
    raw = getattr(args, "noise_label", None)
    if raw is None:
        return None
    return int(raw)


def signal_mask_exclude_noise(
    y_cls: torch.Tensor,
    noise_label: int | None,
) -> torch.Tensor:
    """Boolean mask of samples that participate in task-local classification CE."""
    mask = torch.ones(y_cls.shape[0], dtype=torch.bool, device=y_cls.device)
    if noise_label is not None:
        mask = y_cls != noise_label
    return mask


def classification_loss_zero_stub(cls_logits: torch.Tensor) -> torch.Tensor:
    """Scalar zero loss tied to logits (keeps autograd on an empty CE minibatch)."""
    return cls_logits.sum() * 0.0


def unpack_y_to_class_labels(
    y: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Extract 1D class labels (IQ ``[N, 2]`` tensors and tuple payloads use column 0)."""
    if isinstance(y, (tuple, list)) and len(y) == 2:
        y_cls = y[0]
    elif isinstance(y, dict):
        y_cls = y.get("y_cls", y.get("y"))
    elif torch.is_tensor(y) and y.dim() == 2 and y.size(1) == 2:
        y_cls = y[:, 0]
    else:
        y_cls = y
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    return y_cls


class _ZeroLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return logits.new_zeros(1)


class DetectionReplayMixin:
    """Shared detection replay buffer utilities for continual learners."""

    def _init_det_replay(
        self, det_memories: int, det_replay_batch: int, enabled: bool | None = None
    ) -> None:
        self.det_enabled = True if enabled is None else bool(enabled)
        self.det_loss = nn.BCEWithLogitsLoss()
        self.det_memories = int(det_memories)
        self.det_replay_batch = int(det_replay_batch)
        self._det_mem_x: Optional[torch.Tensor] = None
        self._det_mem_y: Optional[torch.Tensor] = None
        self._det_mem_seen: int = 0
        self._det_mem_count: int = 0
        if not self.det_enabled:
            self.det_memories = 0
            self.det_replay_batch = 0
            self.det_loss = _ZeroLoss()

    def _unpack_labels(
        self,
        y: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor],
        noise_label: int | None = None,
        use_detector_arch: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return classification and detection labels with sensible defaults.

        Args:
            y: Labels provided by the dataloader.
            noise_label: Optional label used to derive detection targets when
                detector labels are not provided.
            use_detector_arch: Whether the model is using the detector
                architecture.

        Returns:
            Tuple of classification labels and detection labels.
        """
        if isinstance(y, (tuple, list)) and len(y) == 2:
            y_cls, y_det = y
        elif isinstance(y, dict):
            y_cls = y.get("y_cls", y.get("y"))
            y_det = y.get("y_det")
        elif torch.is_tensor(y) and y.dim() == 2 and y.size(1) == 2:
            y_cls = y[:, 0]
            y_det = y[:, 1]
        else:
            y_cls = y
            y_det = None

        if not torch.is_tensor(y_cls):
            y_cls = torch.as_tensor(y_cls)
        if y_det is None:
            if noise_label is not None and use_detector_arch is False:
                y_det = (y_cls != noise_label).long()
            else:
                y_det = torch.ones_like(y_cls, dtype=torch.float)
        elif not torch.is_tensor(y_det):
            y_det = torch.as_tensor(y_det)
        # print(f"Unpacked labels: y_cls shape {y_cls.shape}, y_det shape {y_det.shape}")
        return y_cls, y_det

    def _canonicalize_input(
        self,
        x: torch.Tensor,
        *,
        detach: bool,
    ) -> torch.Tensor:
        """Convert inputs to canonical replay/training shape.

        Canonical shape is typically `(B, 2, 512)` for IQ inputs after optional
        3-ADC adaptation.

        Args:
            x: Input batch in one of the supported formats.
            detach: Whether to detach from graph and disable gradient flow.

        Returns:
            Canonicalized tensor suitable for replay buffers or training.
        """
        if detach:
            x = x.detach()

        if x.dim() == 2:
            batch, features = x.shape
            if features % 2 == 0 and features % 3 != 0:
                x = x.view(batch, 2, features // 2)
            return x

        if x.dim() == 3 and x.size(1) == 3:
            # (B, 3, 1024) -> (B, 3, 2, 512) then adapter
            batch, _, sequence_length = x.shape
            if sequence_length % 2 != 0:
                return x
            x = x.view(batch, 3, 2, sequence_length // 2)
            # fall through to 4D adapter path

        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            adapter = getattr(self, "net", None) and (
                getattr(self.net, "input_adapter", None)
                or getattr(getattr(self.net, "model", None), "input_adapter", None)
            )
            if adapter is not None and not isinstance(adapter, nn.Identity):
                sequence_length = x.size(3)
                if sequence_length > 512:
                    x = x[:, :, :, :512].contiguous()
                return adapter(x)
        return x

    def _input_for_replay(self, x: torch.Tensor) -> torch.Tensor:
        """Return detached canonical input for replay-buffer storage."""
        with torch.no_grad():
            return self._canonicalize_input(x, detach=True)

    def _init_det_memory(self, sample_x: torch.Tensor) -> None:
        if not getattr(self, "det_enabled", True):
            return
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
        if not getattr(self, "det_enabled", True):
            return
        if self.det_memories <= 0:
            return
        x = self._input_for_replay(x)
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
        if not getattr(self, "det_enabled", True):
            return None
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
