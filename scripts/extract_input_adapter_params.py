#!/usr/bin/env python3
"""Extract and display input_adapter parameters from a results.pt checkpoint.

The input_adapter (AdcIqAdapter or HatInputAdapter wrapping it) learns a linear
mapping from 3 input channels to 2 IQ channels. This script loads a saved
results.pt, finds the adapter parameters in the state_dict, and prints how
each of the 2 output channels is a linear combination of the 3 input channels.

Usage
-----
    python scripts/extract_input_adapter_params.py path/to/results.pt
    python scripts/extract_input_adapter_params.py logs/my_run/0/results.pt --format csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Repo root for imports if needed
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Allow loading checkpoints that pickle argparse.Namespace
try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None


def load_results_pt(path: Path) -> tuple:
    """Load the results.pt bundle; return (state_dict, args)."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if add_safe_globals is not None:
        try:
            import argparse as _argparse

            add_safe_globals([_argparse.Namespace])
        except Exception:
            pass
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, (tuple, list)) or len(data) < 3:
        raise ValueError(
            f"Expected results.pt to be a tuple (val_t, val_a, state_dict, ...); got {type(data)} len={len(data) if hasattr(data, '__len__') else 'N/A'}"
        )
    state_dict = data[2]
    args = data[5] if len(data) > 5 else None
    return state_dict, args


def find_input_adapter_keys(state_dict: dict) -> list[str]:
    """Return state_dict keys that belong to the input_adapter."""
    return [k for k in state_dict if "input_adapter" in k]


def get_combined_weight_bias(
    state_dict: dict, adapter_keys: list[str]
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """Extract (weight_2x3, bias_2) for 3D path and (weight_4d_2x3, bias_4d_2) for 4D path if present.

    AdcIqAdapter has:
    - For 3D input (B, 3, L): proj_3ch Conv1d(3, 2, 1) -> weight (2, 3, 1), bias (2,)
    - For 4D input (B, 3, 2, L): weight (2, 3), bias (2,)
    Returns (weight_3d, bias_3d, weight_4d, bias_4d). Any can be None if not found.
    """
    weight_3d: torch.Tensor | None = None
    bias_3d: torch.Tensor | None = None
    weight_4d: torch.Tensor | None = None
    bias_4d: torch.Tensor | None = None

    for k in adapter_keys:
        v = state_dict[k]
        if not torch.is_tensor(v):
            continue
        if k.endswith("proj_3ch.weight"):
            # (2, 3, 1) -> (2, 3)
            weight_3d = v.squeeze().clone()
        elif k.endswith("proj_3ch.bias"):
            bias_3d = v.clone()
        elif (
            k.endswith(".weight")
            and "proj_3ch" not in k
            and v.dim() == 2
            and v.shape[0] == 2
            and v.shape[1] == 3
        ):
            weight_4d = v.clone()
        elif k.endswith(".bias") and "proj_3ch" not in k and v.numel() == 2:
            bias_4d = v.clone()

    return weight_3d, bias_3d, weight_4d, bias_4d


def format_linear_combination(
    weight: torch.Tensor, bias: torch.Tensor, channel_names: list[str]
) -> list[str]:
    """Describe each output channel as a linear combination of input channels."""
    weight = weight.detach().float()
    bias = bias.detach().float()
    lines = []
    for i in range(weight.shape[0]):
        terms = [
            f"{weight[i, j].item():+.4f} * {channel_names[j]}"
            for j in range(weight.shape[1])
        ]
        expr = " ".join(terms) + f" {bias[i].item():+.4f}"
        lines.append(f"  out[{i}] (IQ[{i}]): {expr}")
    return lines


def normalize_weight_rows(weight: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize rows so each output channel's coefficients sum to 1.

    Args:
        weight: Tensor of shape (out_channels, in_channels).
        eps: Threshold for treating a row-sum as zero.

    Returns:
        Row-normalized tensor with the same shape.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"Expected rank-2 weight tensor, got shape {tuple(weight.shape)}"
        )

    row_sums = weight.sum(dim=1, keepdim=True)
    zero_row_mask = row_sums.abs() <= eps
    denom = torch.where(zero_row_mask, torch.ones_like(row_sums), row_sums)
    normalized = weight / denom

    uniform = torch.full_like(weight, 1.0 / weight.size(1))
    normalized = torch.where(
        zero_row_mask.expand_as(normalized),
        uniform,
        normalized,
    )
    return normalized


def print_adapter_params(
    path: Path,
    state_dict: dict,
    *,
    channel_names: list[str] | None = None,
    output_format: str = "human",
    normalize_weights: bool = False,
) -> None:
    """Print input_adapter parameters in a readable way."""
    if channel_names is None:
        channel_names = ["ch0", "ch1", "ch2"]
    adapter_keys = find_input_adapter_keys(state_dict)
    if not adapter_keys:
        print(f"No keys containing 'input_adapter' in {path}")
        return

    weight_3d, bias_3d, weight_4d, bias_4d = get_combined_weight_bias(
        state_dict, adapter_keys
    )

    if normalize_weights:
        if weight_3d is not None:
            weight_3d = normalize_weight_rows(weight_3d)
        if weight_4d is not None:
            weight_4d = normalize_weight_rows(weight_4d)

    print(f"Input adapter parameters from: {path}")
    print("Keys found:", adapter_keys)
    print()

    if weight_3d is not None and bias_3d is not None:
        print("3D path (B, 3, L) -> (B, 2, L) via proj_3ch (1x1 conv):")
        print("Weight matrix (2 x 3): rows = output IQ channels, cols = input channels")
        print(weight_3d.numpy())
        print("Bias (2,):", bias_3d.numpy())
        print("Linear combination (out = weight @ channels + bias):")
        for line in format_linear_combination(weight_3d, bias_3d, channel_names):
            print(line)
        if output_format == "csv":
            print(
                "csv_3d_weight:",
                ",".join(f"{x:.6f}" for x in weight_3d.flatten().tolist()),
            )
            print("csv_3d_bias:", ",".join(f"{x:.6f}" for x in bias_3d.tolist()))
        print()

    if weight_4d is not None and bias_4d is not None:
        print("4D path (B, 3, 2, L) -> (B, 2, L) via einsum + weight (2,3) + bias:")
        print("Weight matrix (2 x 3):")
        print(weight_4d.numpy())
        print("Bias (2,):", bias_4d.numpy())
        print("Linear combination:")
        for line in format_linear_combination(weight_4d, bias_4d, channel_names):
            print(line)
        if output_format == "csv":
            print(
                "csv_4d_weight:",
                ",".join(f"{x:.6f}" for x in weight_4d.flatten().tolist()),
            )
            print("csv_4d_bias:", ",".join(f"{x:.6f}" for x in bias_4d.tolist()))
        print()

    if weight_3d is None and weight_4d is None:
        print("No 2x3 weight or proj_3ch found in adapter keys.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_pt",
        type=Path,
        help="Path to results.pt file",
    )
    parser.add_argument(
        "--format",
        choices=("human", "csv"),
        default="human",
        help="Output format: human-readable or csv-style lines",
    )
    parser.add_argument(
        "--channel-names",
        nargs=3,
        default=None,
        metavar=("C0", "C1", "C2"),
        help="Names for the 3 input channels (e.g. I Q ADC)",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Row-normalize adapter weights before printing (each output row sums to 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_dict, _ = load_results_pt(args.results_pt)
    print_adapter_params(
        args.results_pt,
        state_dict,
        channel_names=args.channel_names,
        output_format=args.format,
        normalize_weights=args.normalize_weights,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
