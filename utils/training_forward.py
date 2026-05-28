"""Training-time forward helpers shared by ``main`` and continual-learning models."""

from __future__ import annotations

from typing import Any, Tuple

import torch

from utils import misc_utils


def model_forward_for_metric_loop(
    model: object, x: torch.Tensor, task_index: int, args: object
) -> torch.Tensor:
    """Run ``model`` forward for metric computation (validation, test, or train probe).

    For ``class_incremental_loader``, passes ``cil_all_seen_upto_task`` so
    :func:`utils.misc_utils.apply_task_incremental_logit_mask` is applied in
    model code with the **cumulative** class boundary (true CIL inference). For
    task-incremental loaders, no extra keyword is passed (per-task masking
    only).

    **iCaRL:** Metrics use ``netforward`` logits plus the same
    :func:`utils.misc_utils.apply_task_incremental_logit_mask` call as
    :meth:`model.icarl.Net.observe` (``cil_all_seen_upto_task=task_index`` and
    ``global_noise_label``), not nearest-mean ``forward`` (which omits noise
    before / without exemplars). This matches training for TIL and CIL runs,
    because ``observe`` always passes ``cil_all_seen_upto_task=task_index``.

    Args:
        model: Continual-learning module with ``forward(x, task_index, ...)``.
        x: Input batch on the correct device.
        task_index: Zero-based continual task id (same as ``task_info['task']``).
        args: Experiment arguments (``loader``, ``model`` id).

    Returns:
        Classifier logits tensor.

    Usage:
        logits = model_forward_for_metric_loop(model, batch_x, task_id, args)
    """
    forward_kw: dict[str, Any] = {}
    if getattr(args, "loader", "") == "class_incremental_loader":
        forward_kw["cil_all_seen_upto_task"] = task_index
    if getattr(args, "model", "") == "anml":
        return model(x, fast_weights=None)  # type: ignore[operator]
    if getattr(args, "model", "") == "icarl":
        raw_logits = model.netforward(x)  # type: ignore[attr-defined]
        return misc_utils.apply_task_incremental_logit_mask(
            raw_logits,
            task_index,
            model.classes_per_task,  # type: ignore[attr-defined]
            model.n_classes,  # type: ignore[attr-defined]
            cil_all_seen_upto_task=task_index,
            global_noise_label=getattr(model, "noise_label", None),
            loader=getattr(args, "loader", None),
        )
    try:
        return model(x, task_index, **forward_kw)  # type: ignore[operator]
    except TypeError:
        return model(x, task_index)  # type: ignore[operator]


def unpack_observe_result(
    result: Tuple[float, float] | Tuple[float, float, torch.Tensor | None],
) -> Tuple[float, float, torch.Tensor | None]:
    """Unpack ``observe()`` return value (2- or 3-tuple).

    Args:
        result: ``(loss, cls_tr_rec)`` or ``(loss, cls_tr_rec, metric_logits)``.

    Returns:
        Loss scalar, training recall scalar, and optional detached logits for
        progress-bar metrics (``None`` when the model did not provide them).

    Usage:
        loss, rec, logits = unpack_observe_result(model.observe(x, y, t))
    """
    if len(result) == 3:
        loss, cls_tr_rec, metric_logits = result
        return float(loss), float(cls_tr_rec), metric_logits
    loss, cls_tr_rec = result
    return float(loss), float(cls_tr_rec), None
