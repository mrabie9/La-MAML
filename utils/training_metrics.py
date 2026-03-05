from typing import Union

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


TensorLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(preds: TensorLike, targets: TensorLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert preds and targets to numpy arrays."""
    preds_np = (
        preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    )
    targets_np = (
        targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    )
    return preds_np, targets_np


def macro_precision_signal_only(
    preds: TensorLike,
    targets: TensorLike,
    noise_label: int | None,
) -> float:
    """Macro precision over signal classes only (excludes noise)."""
    preds_np, targets_np = _to_numpy(preds, targets)
    if preds_np.size == 0 or targets_np.size == 0:
        return 0.0
    if noise_label is not None:
        mask = targets_np != noise_label
        if not np.any(mask):
            return 0.0
        preds_np = preds_np[mask]
        targets_np = targets_np[mask]
    return precision_score(
        targets_np, preds_np, average="macro", zero_division=0
    )


def macro_f1_including_noise(preds: TensorLike, targets: TensorLike) -> float:
    """Macro F1 over all classes (including noise)."""
    preds_np, targets_np = _to_numpy(preds, targets)
    if preds_np.size == 0 or targets_np.size == 0:
        return 0.0
    return f1_score(targets_np, preds_np, average="macro", zero_division=0)


def macro_recall(preds: TensorLike, targets: TensorLike) -> float:
    """Compute macro-averaged recall for a batch of predictions."""
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    if preds_np.size == 0 or targets_np.size == 0:
        return 0.0

    return recall_score(targets_np, preds_np, average="macro", zero_division=0)
