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

from model.hat import Net as HatNet  # noqa: E402


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
    """Gradients flow through the adapter for 3-channel input.

    ``HatInputAdapter`` reshapes ``(B, 3, L)`` to ``(B, 3, 2, L/2)``, so
    ``AdcIqAdapter`` uses the 4D einsum path (``weight``), not ``proj_3ch``.
    """
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.train()
    x = torch.randn(2, 3, 1024, requires_grad=True)
    logits = model(x, t=0)
    loss = logits.sum()
    loss.backward()
    adapter = model.bridge.input_adapter._adapter
    assert adapter.weight.grad is not None
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


def test_hat_forward_uses_live_bn_for_unfinalized_current_task():
    """Unfinalized current-task validation uses accumulated live BN statistics."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    model.current_task = 0
    model._bn_initialized_tasks.add(0)

    live_running_means = []
    stale_stats = []
    stale_affine = []
    for batch_norm_module in model._bn_modules:
        batch_norm_module.running_mean.fill_(5.0)
        batch_norm_module.running_var.fill_(2.0)
        batch_norm_module.num_batches_tracked.fill_(10)
        live_running_means.append(batch_norm_module.running_mean.detach().clone())
        stale_stats.append(
            (
                torch.zeros_like(batch_norm_module.running_mean),
                torch.ones_like(batch_norm_module.running_var),
                0,
            )
        )
        if batch_norm_module.affine:
            stale_affine.append(
                (
                    batch_norm_module.weight.detach().clone(),
                    batch_norm_module.bias.detach().clone(),
                )
            )
        else:
            stale_affine.append((None, None))

    model._bn_task_stats[0] = stale_stats
    model._bn_task_affine[0] = stale_affine
    captured_running_means = []

    def fake_bridge_forward(task, x, s, return_masks=False):
        """Capture the active BN state used by ``Net.forward``."""
        del task, s, return_masks
        captured_running_means.extend(
            batch_norm_module.running_mean.detach().clone()
            for batch_norm_module in model._bn_modules
        )
        return torch.zeros(x.size(0), model.n_outputs, device=x.device)

    model.bridge.forward = fake_bridge_forward

    with torch.no_grad():
        logits = model(torch.randn(2, 2, 512), t=0)

    assert logits.shape == (2, 13)
    assert captured_running_means
    for captured_running_mean, live_running_mean in zip(
        captured_running_means, live_running_means
    ):
        assert torch.equal(captured_running_mean, live_running_mean)


def test_hat_forward_leaves_bn_unchanged_for_uninitialized_current_task():
    """Per-task BN is disabled: forward does not reset running stats before bridge."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    assert model._use_task_bn_state is False
    model.eval()
    model.current_task = 0

    for batch_norm_module in model._bn_modules:
        batch_norm_module.running_mean.fill_(5.0)
        batch_norm_module.running_var.fill_(2.0)
        batch_norm_module.num_batches_tracked.fill_(10)

    captured_running_means = []
    captured_running_vars = []

    def fake_bridge_forward(task, x, s, return_masks=False):
        """Capture the BN state seen inside the backbone forward."""
        del task, s, return_masks
        captured_running_means.extend(
            batch_norm_module.running_mean.detach().clone()
            for batch_norm_module in model._bn_modules
        )
        captured_running_vars.extend(
            batch_norm_module.running_var.detach().clone()
            for batch_norm_module in model._bn_modules
        )
        return torch.zeros(x.size(0), model.n_outputs, device=x.device)

    model.bridge.forward = fake_bridge_forward

    with torch.no_grad():
        logits = model(torch.randn(2, 2, 512), t=0)

    assert logits.shape == (2, 13)
    assert 0 not in model._bn_initialized_tasks
    assert captured_running_means
    for captured_running_mean, captured_running_var in zip(
        captured_running_means, captured_running_vars
    ):
        assert torch.equal(
            captured_running_mean, torch.full_like(captured_running_mean, 5.0)
        )
        assert torch.equal(
            captured_running_var, torch.full_like(captured_running_var, 2.0)
        )


def test_hat_forward_does_not_initialize_bn_before_first_observe():
    """Pre-observe forward leaves task BN initialization to observe()."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()

    live_running_means = []
    for batch_norm_module in model._bn_modules:
        batch_norm_module.running_mean.fill_(5.0)
        batch_norm_module.running_var.fill_(2.0)
        batch_norm_module.num_batches_tracked.fill_(10)
        live_running_means.append(batch_norm_module.running_mean.detach().clone())

    captured_running_means = []

    def fake_bridge_forward(task, x, s, return_masks=False):
        """Capture the BN state used before task training starts."""
        del task, s, return_masks
        captured_running_means.extend(
            batch_norm_module.running_mean.detach().clone()
            for batch_norm_module in model._bn_modules
        )
        return torch.zeros(x.size(0), model.n_outputs, device=x.device)

    model.bridge.forward = fake_bridge_forward

    with torch.no_grad():
        logits = model(torch.randn(2, 2, 512), t=0)

    assert logits.shape == (2, 13)
    assert model.current_task is None
    assert 0 not in model._bn_initialized_tasks
    assert captured_running_means
    for captured_running_mean, live_running_mean in zip(
        captured_running_means, live_running_means
    ):
        assert torch.equal(captured_running_mean, live_running_mean)


if __name__ == "__main__":
    test_hat_2channel_3d()
    test_hat_2channel_2d()
    test_hat_3channel_3d()
    test_hat_2channel_and_3channel_forward()
    test_hat_grad_flow_3channel()
    test_hat_grad_flow_2channel()
    test_hat_forward_uses_live_bn_for_unfinalized_current_task()
    test_hat_forward_leaves_bn_unchanged_for_uninitialized_current_task()
    test_hat_forward_does_not_initialize_bn_before_first_observe()
    print("All HAT input adapter tests passed.")
