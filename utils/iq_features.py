"""Shared IQ feature augmentation utilities."""

from __future__ import annotations

import torch
from typing import Literal


def _scale_channels_per_sample(
    channels: torch.Tensor,
    scaling_mode: str,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Apply per-sample scaling over the time axis for channel-first tensors."""
    if scaling_mode == "none":
        return channels
    if scaling_mode == "normalize":
        channel_minimum = channels.amin(dim=-1, keepdim=True)
        channel_maximum = channels.amax(dim=-1, keepdim=True)
        channel_scale = torch.clamp(channel_maximum - channel_minimum, min=epsilon)
        return (channels - channel_minimum) / channel_scale
    if scaling_mode == "standardize":
        channel_mean = channels.mean(dim=-1, keepdim=True)
        channel_std = channels.std(dim=-1, keepdim=True, unbiased=False)
        channel_scale = torch.clamp(channel_std, min=epsilon)
        return (channels - channel_mean) / channel_scale
    raise ValueError(
        f"Unsupported scaling_mode '{scaling_mode}' for IQ augmented features."
    )


def append_iq_augmented_features(
    iq_tensor: torch.Tensor,
    enabled: bool,
    scaling_mode: str = "none",
    feature_type: Literal["power", "cross"] = "power",
) -> torch.Tensor:
    """Append derived IQ channels when augmentation is enabled.

    Args:
        iq_tensor: Channel-first IQ tensor with shape ``(..., 2, L)``.
        enabled: Whether to append derived channels.
        scaling_mode: Optional scaling mode for derived channels. Supported
            values are ``none``, ``normalize``, and ``standardize``.
        feature_type: Which derived feature to append. If ``"power"``, the
            appended channel is ``I**2 + Q**2``; if ``"cross"``, it is ``I*Q``.

    Returns:
        Tensor with shape ``(..., 2, L)`` when disabled, or ``(..., 3, L)``
        when enabled by appending exactly one derived feature.

    Raises:
        ValueError: If augmentation is enabled but the input does not have
            exactly two IQ channels on the penultimate axis.
    """
    if not enabled:
        return iq_tensor

    if iq_tensor.dim() < 2 or iq_tensor.size(-2) != 2:
        raise ValueError(
            "IQ feature augmentation expects channel-first input with 2 channels "
            f"on axis -2; got shape {tuple(iq_tensor.shape)}."
        )

    i_channel = iq_tensor.select(dim=-2, index=0)
    q_channel = iq_tensor.select(dim=-2, index=1)
    if feature_type == "power":
        derived_channel = (i_channel * i_channel) + (q_channel * q_channel)
    elif feature_type == "cross":
        derived_channel = i_channel * q_channel
    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported feature_type '{feature_type}'. Expected 'power' or 'cross'."
        )

    derived_channels = derived_channel.unsqueeze(dim=-2)  # (..., 1, L)
    scaled_derived_channels = _scale_channels_per_sample(
        derived_channels, scaling_mode=scaling_mode
    )
    return torch.cat(
        (iq_tensor, scaled_derived_channels.to(dtype=iq_tensor.dtype)), dim=-2
    )
