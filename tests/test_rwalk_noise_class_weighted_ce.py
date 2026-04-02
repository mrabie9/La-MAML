"""Regression: RWalk + inverse-frequency CE with global noise labels (CIL IQ).

Batches can carry a global ``noise_label`` (e.g. 54) while the task logit slice
only spans signal classes (e.g. width 5). Out-of-range targets must not reach
``torch.bincount`` with a small ``minlength``, and RWalk must mask noise from CE.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.rwalk import Net as RWalkNet
from utils.class_weighted_loss import compute_inverse_frequency_class_weights


def _minimal_rwalk_args(
    *,
    classes_per_task: list[int],
    noise_label: int | None,
    class_weighted_ce: bool = True,
) -> object:
    """Build a plain namespace with fields ``ResNet1D`` / ``RWalk`` expect."""
    o = type("Args", (), {})()
    o.class_incremental = True
    o.classes_per_task = classes_per_task
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.noise_label = noise_label
    o.class_weighted_ce = class_weighted_ce
    o.use_detector_arch = False
    o.use_iq_aug_features = False
    o.data_scaling = "none"
    o.iq_aug_feature_type = "power"
    o.lr = 0.01
    o.optimizer = "adam"
    o.lamb = 1.0
    o.alpha = 0.9
    o.eps = 0.01
    o.clipgrad = 100.0
    o.det_lambda = 1.0
    o.cls_lambda = 1.0
    o.det_memories = 0
    o.det_replay_batch = 64
    o.norm_track_stats = True
    o.alpha_init = 1e-3
    return o


def test_compute_inverse_frequency_ignores_labels_outside_logit_range() -> None:
    device = torch.device("cpu")
    labels = torch.tensor([0, 1, 54, 3, 4], dtype=torch.long)
    weights = compute_inverse_frequency_class_weights(
        labels, num_classes=5, device=device
    )
    assert weights.shape == (5,)
    assert torch.isfinite(weights).all()


def test_rwalk_observe_masks_global_noise_for_weighted_ce() -> None:
    """Simulates task 0: five signal classes (0–4) and global noise id 54."""
    n_outputs = 55
    n_tasks = 10
    args = _minimal_rwalk_args(
        classes_per_task=[5, 6, 5, 6, 6, 5, 5, 5, 6, 5],
        noise_label=54,
        class_weighted_ce=True,
    )
    model = RWalkNet(128, n_outputs, n_tasks, args)
    model.eval()
    torch.manual_seed(0)
    batch = 16
    x = torch.randn(batch, 2, 64)
    y_cls = torch.tensor([0, 1, 2, 3, 4, 54, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    loss, rec = model.observe(x, y_cls, 0)
    assert loss == loss and loss >= 0.0
    assert 0.0 <= rec <= 1.0


def test_rwalk_observe_accepts_two_column_iq_labels() -> None:
    """Dataloader-style ``(N, 2)`` tensor: class + detector columns."""
    n_outputs = 12
    n_tasks = 2
    args = _minimal_rwalk_args(
        classes_per_task=[5, 6],
        noise_label=11,
        class_weighted_ce=True,
    )
    model = RWalkNet(128, n_outputs, n_tasks, args)
    model.eval()
    b = 8
    x = torch.randn(b, 2, 32)
    y = torch.zeros(b, 2, dtype=torch.long)
    y[:, 0] = torch.tensor([0, 1, 2, 11, 3, 4, 11, 0])
    y[:, 1] = 1
    loss, rec = model.observe(x, y, 0)
    assert loss == loss and loss >= 0.0
    assert 0.0 <= rec <= 1.0


def test_rwalk_all_noise_batch_trains_noise_class() -> None:
    """Noise rows contribute to CE; loss should be positive and finite."""
    args = _minimal_rwalk_args(
        classes_per_task=[5], noise_label=9, class_weighted_ce=True
    )
    n_outputs = 10
    model = RWalkNet(64, n_outputs, 1, args)
    model.train()
    x = torch.randn(4, 2, 32)
    y = torch.full((4,), 9, dtype=torch.long)
    loss, rec = model.observe(x, y, 0)
    assert loss > 0.0
    assert loss == loss
    assert rec == 0.0
