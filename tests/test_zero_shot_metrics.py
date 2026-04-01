"""Tests for zero-shot pre-train metrics in ``life_experience`` and NPZ compatibility."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Tuple
from unittest.mock import patch

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import (  # noqa: E402
    _mean_metric_across_tasks,
    _per_task_metric_array,
    _scalar_metric_at_task_index,
    life_experience,
)
from scripts.plot_metrics import load_metrics  # noqa: E402


class _TqdmPassthrough:
    """Minimal tqdm stand-in: iterable plus ``set_description`` (used in ``main``)."""

    def __init__(self, iterable: Iterable[Any], **kwargs: Any) -> None:
        del kwargs
        self._iterable = iterable

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _TinyContinualModel(torch.nn.Module):
    """Minimal module compatible with ``life_experience`` train+eval loops."""

    split = False

    def __init__(self, n_inputs: int = 8, n_outputs: int = 6) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(n_inputs, n_outputs)
        self.real_epoch = 0

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        del task_id
        return self.fc(x)

    def observe(
        self, v_x: torch.Tensor, v_y: torch.Tensor, task_id: int
    ) -> Tuple[float, torch.Tensor]:
        del task_id
        x_tensor = v_x if isinstance(v_x, torch.Tensor) else v_x.data  # type: ignore[union-attr]
        logits = self.forward(x_tensor, 0)
        # Scalar float so ``main`` can ``round(loss, 3)`` (tensors break ``round`` in Py3).
        loss_value = float(
            torch.nn.functional.cross_entropy(logits, v_y.long()).detach()
        )
        recall = torch.tensor(0.5, device=logits.device, dtype=torch.float32)
        return loss_value, recall


class _MockIncrementalLoader:
    """Two-task loader with disjoint label ranges per task (task-incremental style)."""

    def __init__(self) -> None:
        self.n_tasks = 2
        self._next_task_index = 0
        torch.manual_seed(42)

    def new_task(self) -> Tuple[dict[str, Any], DataLoader, None, DataLoader]:
        task_index = self._next_task_index
        self._next_task_index += 1
        if task_index == 0:
            labels = torch.randint(0, 3, (32,), dtype=torch.long)
        elif task_index == 1:
            labels = torch.randint(3, 6, (32,), dtype=torch.long)
        else:
            raise RuntimeError("Mock loader only supports two tasks.")
        features = torch.randn(32, 8, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(features, labels), batch_size=16, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(features.clone(), labels.clone()),
            batch_size=16,
            shuffle=False,
        )
        info = {"task": task_index, "task_name": f"mock_task_{task_index}"}
        return info, train_loader, None, test_loader


def test_zero_shot_metric_helpers() -> None:
    """Unit-test extraction helpers used for zero-shot NPZ fields."""
    metrics_list = [0.1, torch.tensor(0.2), 0.3]
    assert abs(_scalar_metric_at_task_index(metrics_list, 1) - 0.2) < 1e-6
    assert np.isnan(_scalar_metric_at_task_index(metrics_list, 9))
    assert np.isnan(_scalar_metric_at_task_index(None, 0))
    mean_f1 = _mean_metric_across_tasks([torch.tensor(0.8), torch.tensor(0.6), 0.4])
    assert abs(mean_f1 - (0.8 + 0.6 + 0.4) / 3.0) < 1e-6
    row = _per_task_metric_array([1.0, torch.tensor(2.0)], 3)
    assert row.shape == (3,)
    assert row[0] == 1.0 and row[1] == 2.0 and np.isnan(row[2])


def test_load_metrics_accepts_legacy_plus_zero_shot_keys() -> None:
    """``plot_metrics.load_metrics`` must still load npz with new zero-shot arrays."""
    with tempfile.TemporaryDirectory() as tmp:
        metrics_dir = os.path.join(tmp, "metrics")
        os.makedirs(metrics_dir)
        np.savez(
            os.path.join(metrics_dir, "task0.npz"),
            losses=np.array([0.1, 0.2]),
            cls_tr_rec=np.array([0.5, 0.6]),
            val_acc=np.array([0.7]),
            zero_shot_rec_cls=np.float64(0.11),
            zero_shot_prec_cls=np.float64(0.22),
            zero_shot_f1_cls=np.float64(0.33),
            zero_shot_det=np.float64(0.44),
            zero_shot_pfa=np.float64(0.55),
            zero_shot_total_f1=np.float64(0.66),
            zero_shot_per_task_rec_cls=np.array([0.11]),
            zero_shot_per_task_prec_cls=np.array([0.22]),
            zero_shot_per_task_f1_cls=np.array([0.33]),
            zero_shot_per_task_det=np.array([0.44]),
            zero_shot_per_task_pfa=np.array([0.55]),
        )
        tasks = load_metrics(Path(metrics_dir))
        assert len(tasks) == 1
        task0 = tasks[0]
        assert "losses" in task0
        assert "val_acc" in task0
        assert task0["zero_shot_rec_cls"].shape == ()
        assert task0["zero_shot_per_task_rec_cls"].shape == (1,)


def test_life_experience_npz_has_zero_shot_and_legacy_keys() -> None:
    """Integration: ``life_experience`` writes zero-shot fields without dropping legacy keys."""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = os.path.join(tmp, "run0")
        os.makedirs(log_dir)
        args = type("Args", (), {})()
        args.state_logging = False
        args.n_epochs = 1
        args.val_rate = 1
        args.loader = "task_incremental_loader"
        args.cuda = False
        args.arch = ""
        args.model = "stub_model_name"
        args.use_detector_arch = False
        args.calc_test_accuracy = False
        args.log_dir = log_dir

        model = _TinyContinualModel(n_inputs=8, n_outputs=6)
        loader = _MockIncrementalLoader()

        with patch("main.tqdm", _TqdmPassthrough):
            life_experience(model, loader, args)

        metrics_path = os.path.join(log_dir, "metrics")
        assert os.path.isdir(metrics_path)

        for task_i, expected_len in enumerate([1, 2]):
            path_npz = os.path.join(metrics_path, f"task{task_i}.npz")
            assert os.path.isfile(path_npz)
            data = np.load(path_npz, allow_pickle=True)

            for key in (
                "losses",
                "cls_tr_rec",
                "val_acc",
                "zero_shot_rec_cls",
                "zero_shot_prec_cls",
                "zero_shot_f1_cls",
                "zero_shot_det",
                "zero_shot_pfa",
                "zero_shot_total_f1",
                "zero_shot_per_task_rec_cls",
                "zero_shot_per_task_prec_cls",
                "zero_shot_per_task_f1_cls",
                "zero_shot_per_task_det",
                "zero_shot_per_task_pfa",
            ):
                assert key in data.files, f"missing {key} in task{task_i}.npz"

            assert data["zero_shot_per_task_rec_cls"].shape == (expected_len,)
            assert not np.isnan(data["zero_shot_rec_cls"].item())
            assert not np.isnan(data["zero_shot_total_f1"].item())

        plot_tasks = load_metrics(Path(metrics_path))
        assert len(plot_tasks) == 2
        assert "zero_shot_total_f1" in plot_tasks[1]


def test_eval_tasks_runs_with_variable_wrapped_batch() -> None:
    """Guard: training passes ``Variable`` batch into ``observe`` (older La-MAML style)."""
    model = _TinyContinualModel()
    x = torch.randn(4, 8)
    y = torch.randint(0, 3, (4,), dtype=torch.long)
    loss, rec = model.observe(Variable(x), y, 0)
    assert isinstance(loss, float)
    assert rec.ndim == 0
