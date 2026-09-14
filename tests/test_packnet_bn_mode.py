"""PackNet bn_mode option: shared vs task_specific BatchNorm handling."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from model.packnet import Net  # noqa: E402


def _tiny_args(bn_mode: str) -> object:
    chain = [
        str(ROOT / "configs" / "base.yaml"),
        str(ROOT / "configs" / "models" / "til" / "packnet.yaml"),
    ]
    args = file_parser.parse_args_from_yaml(chain)
    args.cuda = False
    args.model = "packnet"
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.classes_per_task = [2, 2]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.batch_size = 8
    args.inner_steps = 1
    args.lr = 0.01
    args.optimizer = "sgd"
    args.post_prune_epochs = 0
    args.prune_perc = 0.5
    args.class_weighted_ce = False
    args.loader = "task_incremental_loader"
    args.bn_mode = bn_mode
    return args


def _build_model(bn_mode: str) -> Net:
    torch.manual_seed(0)
    args = _tiny_args(bn_mode)
    return Net(2 * 32, 4, 2, args)


def test_invalid_bn_mode_rejected() -> None:
    args = _tiny_args("bogus")
    with pytest.raises(ValueError):
        Net(2 * 32, 4, 2, args)


def test_task_specific_snapshots_and_restores_bn_per_task() -> None:
    model = _build_model("task_specific")
    assert model.bn_mode == "task_specific"

    x0 = torch.randn(4, 2, 32)
    y0 = torch.randint(0, 2, (4,))
    for _ in range(3):
        model.observe(x0, y0, 0)

    model.finalize_task_after_training(train_loader=None)
    assert 0 in model._bn_task_stats
    assert 0 in model._bn_task_affine

    # Switching to an unseen task resets BN running stats to defaults.
    model._restore_bn_stats(1)
    bn = model._bn_modules[0]
    assert torch.allclose(bn.running_mean, torch.zeros_like(bn.running_mean))
    assert torch.allclose(bn.running_var, torch.ones_like(bn.running_var))

    # Restoring task 0 brings back its snapshot exactly.
    snapshot_mean, snapshot_var, _ = model._bn_task_stats[0][0]
    model._restore_bn_stats(0)
    assert torch.allclose(bn.running_mean, snapshot_mean)
    assert torch.allclose(bn.running_var, snapshot_var)

    # observe() and finalize() actively call the snapshot/restore helpers.
    model._restore_bn_stats = MagicMock(wraps=model._restore_bn_stats)
    model.observe(x0, y0, 1)
    assert model._restore_bn_stats.called


def test_shared_bn_never_snapshots_or_restores() -> None:
    model = _build_model("shared")
    assert model.bn_mode == "shared"
    model._restore_bn_stats = MagicMock(wraps=model._restore_bn_stats)
    model._snapshot_bn_stats = MagicMock(wraps=model._snapshot_bn_stats)

    x0 = torch.randn(4, 2, 32)
    y0 = torch.randint(0, 2, (4,))
    for _ in range(3):
        model.observe(x0, y0, 0)
    model.finalize_task_after_training(train_loader=None)

    x1 = torch.randn(4, 2, 32)
    y1 = torch.randint(0, 2, (4,))
    model.observe(x1, y1, 1)
    model.forward(x0, 0)

    model._restore_bn_stats.assert_not_called()
    model._snapshot_bn_stats.assert_not_called()
    assert model._bn_task_stats == {}
    assert model._bn_task_affine == {}
