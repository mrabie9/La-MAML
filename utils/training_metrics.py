from typing import Union

import numpy as np
import torch
from sklearn.metrics import recall_score


TensorLike = Union[torch.Tensor, np.ndarray]


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
