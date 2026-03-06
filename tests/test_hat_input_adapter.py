"""Tests for HAT model input adapter: 2-channel and 3-channel inputs.

The HAT backbone uses HatInputAdapter (wrapping AdcIqAdapter from resnet1d).
These tests verify that 2-channel input passes through and 3-channel is
converted to 2-channel before the gated ResNet.
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.hat import Net as HatNet


def _make_args(n_tasks=2, classes_per_task=None):
    """Minimal args-like object for HAT Net."""
    if classes_per_task is None:
        classes_per_task = [6, 6]
    o = type("Args", (), {})()
    o.classes_per_task = classes_per_task
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.get_samples_per_task = None
    o.samples_per_task = -1
    o.batch_size = 128
    o.clipgrad = 10.0
    o.grad_clip_norm = 10.0
    o.dataset = "iq"
    o.arch = "resnet1d"
    o.input_channels = 2
    return o


def test_hat_2channel_3d():
    """2-channel 3D input (B, 2, L) bypasses adapter and reaches backbone."""
    args = _make_args()
    # n_inputs=1024 -> IQ seq_len=512, 2 channels
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 2, 512)
        logits = model(x, t=0)
    assert logits.shape == (4, 13)


def test_hat_2channel_2d():
    """2D flat input (B, 1024) is reshaped to (B, 2, 512) then forwarded."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 1024)
        logits = model(x, t=0)
    assert logits.shape == (4, 13)


def test_hat_3channel_3d():
    """3-channel 3D input (B, 3, L) is adapted to 2-channel then forwarded."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 1024)
        logits = model(x, t=0)
    assert logits.shape == (4, 13)


def test_hat_2channel_and_3channel_forward():
    """Forward runs without error for both 2- and 3-channel inputs."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x2 = torch.randn(2, 2, 512)
        out2 = model(x2, t=0)
        x3 = torch.randn(2, 3, 1024)
        out3 = model(x3, t=0)
    assert out2.shape == (2, 13)
    assert out3.shape == (2, 13)


def test_hat_grad_flow_3channel():
    """Gradients flow through the adapter for 3-channel input."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.train()
    x = torch.randn(2, 3, 1024, requires_grad=True)
    logits = model(x, t=0)
    loss = logits.sum()
    loss.backward()
    proj = model.bridge.input_adapter._adapter.proj_3ch
    assert proj.weight.grad is not None
    assert x.grad is not None


def test_hat_grad_flow_2channel():
    """Gradients flow for 2-channel input (adapter is identity)."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.train()
    x = torch.randn(2, 2, 512, requires_grad=True)
    logits = model(x, t=0)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None


if __name__ == "__main__":
    test_hat_2channel_3d()
    test_hat_2channel_2d()
    test_hat_3channel_3d()
    test_hat_2channel_and_3channel_forward()
    test_hat_grad_flow_3channel()
    test_hat_grad_flow_2channel()
    print("All HAT input adapter tests passed.")
