"""Tests for deterministic task-order permutation."""

import numpy as np

from dataloaders.task_incremental_loader import permute_task_sequence


def test_permute_task_sequence_none_is_identity() -> None:
    """When seed is None, order is unchanged and perm is None."""

    base = ["a", "b", "c"]
    out, perm = permute_task_sequence(base, None)
    assert out == list(base)
    assert perm is None


def test_permute_task_sequence_single_task_no_perm_array() -> None:
    """One task does not require permutation."""

    out, perm = permute_task_sequence(("only",), 123)
    assert out == ["only"]
    assert perm is None


def test_permute_task_sequence_fixed_seed_stable() -> None:
    """Same seed yields the same permutation."""

    base = list(range(5))
    first, perm_a = permute_task_sequence(base, 42)
    second, perm_b = permute_task_sequence(base, 42)
    assert first == second
    assert np.array_equal(perm_a, perm_b)


def test_permute_task_sequence_maps_slots_to_base_indices() -> None:
    """Result slot t equals base[int(perm[t])]."""

    base = ["x", "y", "z"]
    reordered, perm = permute_task_sequence(base, 7)
    assert perm is not None
    for slot in range(len(base)):
        assert reordered[slot] == base[int(perm[slot])]
