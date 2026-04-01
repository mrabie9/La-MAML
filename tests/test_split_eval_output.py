"""Guard ``_split_eval_output`` against ambiguous per-task metric lists."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import _split_eval_output


def test_five_tuple_from_eval_class_tasks_yields_list_as_cls_only() -> None:
    """``eval_class_tasks`` returns the same shape as ``eval_tasks`` (5-tuple)."""
    cls_list, prec, f1, det, fa = _split_eval_output(
        ([0.18, 0.2045], None, None, None, None)
    )
    assert cls_list == [0.18, 0.2045]
    assert prec is None and f1 is None and det is None and fa is None


def test_len_two_list_still_misinterpreted_documented() -> None:
    """Bare length-2 list is ambiguous; callers must wrap as 5-tuple."""
    cls_or_scalar, _, _, det_guess, _ = _split_eval_output([0.18, 0.20])
    assert cls_or_scalar == 0.18
    assert det_guess == 0.20
