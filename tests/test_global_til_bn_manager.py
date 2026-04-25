"""Tests for universal TIL BatchNorm task-state manager in ``main.py``."""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import (  # noqa: E402
    _GlobalTilBatchNormStateManager,
    _disable_model_specific_bn_task_state,
    _should_use_global_til_bn_manager,
)


class _TinyModel(nn.Module):
    """Small model with one BatchNorm layer for manager tests."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)


class _Args:
    """Simple args object with only ``loader``."""

    def __init__(self, loader: str, model: str = "icarl") -> None:
        self.loader = loader
        self.model = model


def test_global_til_bn_manager_restores_task_state() -> None:
    """Task snapshots restore the right BN state and keep eval isolated."""
    model = _TinyModel()
    args = _Args("task_incremental_loader")
    manager = _GlobalTilBatchNormStateManager(model, args)
    assert manager.enabled

    manager.on_task_start(0)
    model.bn.running_mean.fill_(5.0)
    model.bn.running_var.fill_(2.0)
    model.bn.num_batches_tracked.fill_(10)
    manager.on_task_end(0)

    manager.on_task_start(1)
    assert torch.equal(model.bn.running_mean, torch.zeros_like(model.bn.running_mean))
    assert torch.equal(model.bn.running_var, torch.ones_like(model.bn.running_var))
    assert int(model.bn.num_batches_tracked.item()) == 0

    model.bn.running_mean.fill_(3.0)
    model.bn.running_var.fill_(4.0)
    model.bn.num_batches_tracked.fill_(7)

    with manager.eval_task(0):
        assert torch.allclose(
            model.bn.running_mean, torch.full_like(model.bn.running_mean, 5.0)
        )
        assert torch.allclose(
            model.bn.running_var, torch.full_like(model.bn.running_var, 2.0)
        )
        assert int(model.bn.num_batches_tracked.item()) == 10

    assert torch.allclose(
        model.bn.running_mean, torch.full_like(model.bn.running_mean, 3.0)
    )
    assert torch.allclose(
        model.bn.running_var, torch.full_like(model.bn.running_var, 4.0)
    )
    assert int(model.bn.num_batches_tracked.item()) == 7


def test_global_til_bn_manager_disabled_for_non_til() -> None:
    """Class-incremental mode should not activate global BN task-state manager."""
    model = _TinyModel()
    args = _Args("class_incremental_loader")
    manager = _GlobalTilBatchNormStateManager(model, args)
    assert not manager.enabled


def test_disable_model_specific_bn_task_state_keeps_finalize_hook() -> None:
    """Universal mode should neutralize BN hooks but preserve task finalization."""

    class _ModelWithHooks:
        def _reset_bn_stats(self):
            raise RuntimeError("should not run")

        def _snapshot_bn_stats(self, task: int):
            raise RuntimeError(task)

        def _restore_bn_stats(self, task: int):
            raise RuntimeError(task)

        def _apply_bn_state(self, stats, affine):
            raise RuntimeError("should not run")

        def _capture_bn_state(self):
            raise RuntimeError("should not run")

        def __init__(self) -> None:
            self.finalize_called = False

        def finalize_task_after_training(self, train_loader=None):
            self.finalize_called = True

    model = _ModelWithHooks()
    _disable_model_specific_bn_task_state(model, _Args("task_incremental_loader"))

    model._reset_bn_stats()
    model._snapshot_bn_stats(0)
    model._restore_bn_stats(0)
    model._apply_bn_state([], [])
    model.finalize_task_after_training(None)
    assert model.finalize_called is True
    captured = model._capture_bn_state()
    assert captured == ([], [])


def test_disable_model_specific_bn_task_state_skips_hat_and_packnet() -> None:
    """HAT/PackNet should keep their model-native BN hooks untouched."""

    class _ModelWithHooks:
        def _restore_bn_stats(self, task: int):
            return task

    for model_name in ("hat", "packnet"):
        model = _ModelWithHooks()
        original_function = model._restore_bn_stats.__func__
        _disable_model_specific_bn_task_state(
            model, _Args("task_incremental_loader", model=model_name)
        )
        assert model._restore_bn_stats.__func__ is original_function


def test_should_use_global_til_bn_manager_excludes_native_bn_models() -> None:
    """Global BN manager should be disabled for models with native BN logic."""
    assert _should_use_global_til_bn_manager(_Args("task_incremental_loader", "icarl"))
    assert not _should_use_global_til_bn_manager(
        _Args("task_incremental_loader", "hat")
    )
    assert not _should_use_global_til_bn_manager(
        _Args("task_incremental_loader", "packnet")
    )
