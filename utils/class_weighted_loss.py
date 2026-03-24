"""Minibatch class-weighted cross-entropy used across continual learners."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_inverse_frequency_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build inverse-frequency CE class weights for labels in a minibatch.

    Args:
        labels: Class indices in ``[0, num_classes - 1]`` (any integer dtype).
        num_classes: Logit width ``C`` for the slice used in CE.
        device: Device for the returned weight vector.

    Returns:
        Per-class weights of shape ``(num_classes,)``, mean 1.0 over classes
        that appear in the batch; unseen classes keep weight 1.0.

    Usage:
        >>> w = compute_inverse_frequency_class_weights(y.long(), 6, logits.device)
        >>> loss = F.cross_entropy(logits, y.long(), weight=w)
    """
    labels_long = labels.long().flatten()
    label_counts = torch.bincount(labels_long, minlength=num_classes).float().to(device)
    observed_mask = label_counts > 0
    weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    if observed_mask.any():
        observed_counts = label_counts[observed_mask]
        inverse_frequency = observed_counts.sum() / observed_counts.clamp_min(1.0)
        inverse_frequency = inverse_frequency / inverse_frequency.mean().clamp_min(1e-8)
        weights[observed_mask] = inverse_frequency
    return weights


def classification_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weighted_ce: bool = True,
) -> torch.Tensor:
    """Standard or inverse-frequency-weighted cross-entropy for one logit slice.

    Args:
        logits: Class logits ``(N, C)``.
        targets: Class indices ``(N,)`` in ``[0, C - 1]``.
        class_weighted_ce: If True, weight CE by inverse class frequency in
            this minibatch (same scheme as ``ucl_bresnet``); else unweighted CE.

    Returns:
        Scalar mean cross-entropy loss.

    Usage:
        >>> loss = classification_cross_entropy(logits, y, class_weighted_ce=True)
    """
    targets_long = targets.long()
    if not class_weighted_ce:
        return F.cross_entropy(logits, targets_long)
    num_classes = int(logits.size(-1))
    class_weights = compute_inverse_frequency_class_weights(
        targets_long, num_classes, logits.device
    )
    return F.cross_entropy(logits, targets_long, weight=class_weights)
