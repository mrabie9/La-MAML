"""Tests for UCL model input adapter: 2-channel and 3-channel inputs.

The BayesianClassifier uses the same AdcIqAdapter as ResNet1D. These tests
verify that 2-channel input passes through and 3-channel (3D and 4D) is
converted to 2-channel before the backbone.
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.ucl_bresnet import BayesianClassifier, UCLConfig


def _make_args(n_tasks=2, classes_per_task=None):
    """Minimal args-like object for BayesianClassifier."""
    if classes_per_task is None:
        classes_per_task = [6, 6]
    o = type("Args", (), {})()
    o.classes_per_task = classes_per_task
    o.nc_per_task_list = ""
    o.nc_per_task = None
    return o


def test_bayesian_classifier_2channel():
    """2-channel input (B, 2, L) bypasses adapter and reaches backbone."""
    cfg = UCLConfig()
    args = _make_args()
    model = BayesianClassifier(
        n_outputs=12,
        n_tasks=2,
        cfg=cfg,
        args=args,
        classes_per_task=[6, 6],
    )
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 2, 128)
        out = model(x, sample=False)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (4, 6)
    assert out[1].shape == (4, 6)


def test_bayesian_classifier_3channel_3d():
    """3-channel 3D input (B, 3, L) is adapted to 2-channel then forwarded."""
    cfg = UCLConfig()
    args = _make_args()
    model = BayesianClassifier(
        n_outputs=12,
        n_tasks=2,
        cfg=cfg,
        args=args,
        classes_per_task=[6, 6],
    )
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 1024)
        out = model(x, sample=False)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (4, 6)
    assert out[1].shape == (4, 6)


def test_bayesian_classifier_3channel_4d():
    """3-channel 4D input (B, 3, 2, L) is adapted to 2-channel then forwarded."""
    cfg = UCLConfig()
    args = _make_args()
    model = BayesianClassifier(
        n_outputs=12,
        n_tasks=2,
        cfg=cfg,
        args=args,
        classes_per_task=[6, 6],
    )
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 2, 512)
        out = model(x, sample=False)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (4, 6)
    assert out[1].shape == (4, 6)


def test_bayesian_classifier_sample_mode():
    """Forward with sample=True runs without error for 2- and 3-channel."""
    cfg = UCLConfig()
    args = _make_args()
    model = BayesianClassifier(
        n_outputs=12,
        n_tasks=2,
        cfg=cfg,
        args=args,
        classes_per_task=[6, 6],
    )
    model.eval()
    with torch.no_grad():
        x2 = torch.randn(2, 2, 128)
        out2 = model(x2, sample=True)
        x3 = torch.randn(2, 3, 256)
        out3 = model(x3, sample=True)
    assert len(out2) == 2 and out2[0].shape == (2, 6)
    assert len(out3) == 2 and out3[0].shape == (2, 6)


def test_bayesian_classifier_grad_flow_3channel():
    """Gradients flow through the adapter for 3-channel input."""
    cfg = UCLConfig()
    args = _make_args()
    model = BayesianClassifier(
        n_outputs=12,
        n_tasks=2,
        cfg=cfg,
        args=args,
        classes_per_task=[6, 6],
    )
    model.train()
    x = torch.randn(2, 3, 128, requires_grad=True)
    out = model(x, sample=False)
    loss = out[0].sum() + out[1].sum()
    loss.backward()
    assert model.input_adapter.proj_3ch.weight.grad is not None
    assert x.grad is not None


if __name__ == "__main__":
    test_bayesian_classifier_2channel()
    test_bayesian_classifier_3channel_3d()
    test_bayesian_classifier_3channel_4d()
    test_bayesian_classifier_sample_mode()
    test_bayesian_classifier_grad_flow_3channel()
    print("All UCL input adapter tests passed.")
