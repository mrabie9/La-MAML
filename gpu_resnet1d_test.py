"""GPU ResNet1D forward-pass benchmark (compare setups on identical workload).

Times ``ResNet1D.forward_heads`` on CUDA with IQ-shaped tensors, matching the
backbone used by continual-learning models in this repo (e.g. EWC).

Usage:
    source la-maml_env/bin/activate
    python gpu_resnet1d_test.py
    python gpu_resnet1d_test.py --dtype bfloat16 --batch-size 512 --seq-len 1024
    python gpu_resnet1d_test.py --input-layout iq_2ch --bench-iters 50
"""

from __future__ import annotations

import argparse
import time
from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D


def _resolve_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    if not torch.cuda.is_available():
        print("CUDA not available; running on CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def _resolve_dtype(name: str) -> torch.dtype:
    """Map CLI dtype name to ``torch.dtype``."""
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype '{name}'.")


def _build_args_namespace(use_groupnorm: bool) -> Namespace:
    """Minimal ``args`` object for ``ResNet1D`` construction."""
    return Namespace(
        use_iq_aug_features=False,
        data_scaling="none",
        iq_aug_feature_type="power",
        use_groupnorm=use_groupnorm,
        norm_type="batchnorm",
        alpha_init=1e-3,
    )


def _make_inputs(
    batch_size: int,
    seq_len: int,
    input_layout: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create random IQ tensors in the layout expected by ``ResNet1D``.

    Args:
        batch_size: Number of samples per batch.
        seq_len: IQ sequence length (per I/Q channel, before 3-ADC interleave).
        input_layout: ``iq_3adc`` for ``(B, 3, 2, L)`` or ``iq_2ch`` for ``(B, 2, L)``.
        device: Target device.
        dtype: Tensor dtype.

    Returns:
        Input batch tensor.

    Usage:
        >>> x = _make_inputs(4, 128, "iq_2ch", torch.device("cpu"), torch.float32)
        >>> tuple(x.shape)
        (4, 2, 128)
    """
    if input_layout == "iq_3adc":
        return torch.randn(batch_size, 3, 2, seq_len, device=device, dtype=dtype)
    if input_layout == "iq_2ch":
        return torch.randn(batch_size, 2, seq_len, device=device, dtype=dtype)
    raise ValueError(f"Unsupported input_layout '{input_layout}'.")


def _synchronize(device: torch.device) -> None:
    """Block until GPU work completes when benchmarking on CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_benchmark(
    model: nn.Module,
    inputs: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Warm up then time ``bench_iters`` ``forward_heads`` calls.

    Args:
        model: ``ResNet1D`` in eval mode.
        inputs: IQ batch tensor.
        warmup_iters: Number of untimed warmup iterations.
        bench_iters: Number of timed iterations.
        device: Execution device.

    Returns:
        Total seconds and seconds per iteration for the timed loop.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model.forward_heads(inputs)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = model.forward_heads(inputs)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    per_iter = elapsed / bench_iters if bench_iters > 0 else float("nan")
    return elapsed, per_iter


def _print_runtime_info(device: torch.device, dtype: torch.dtype) -> None:
    """Print PyTorch/CUDA context useful when comparing two machines."""
    print(f"PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        capability = torch.cuda.get_device_capability(device)
        print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"dtype: {dtype}")


def main() -> None:
    """Parse CLI args, run ResNet1D benchmark, and print timings."""
    parser = argparse.ArgumentParser(
        description="Benchmark ResNet1D forward_heads on GPU."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument(
        "--input-layout",
        type=str,
        default="iq_3adc",
        choices=["iq_3adc", "iq_2ch"],
        help="iq_3adc: (B, 3, 2, L); iq_2ch: (B, 2, L).",
    )
    parser.add_argument(
        "--use-groupnorm",
        action="store_true",
        help="Use GroupNorm instead of BatchNorm in the backbone.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    args = parser.parse_args()

    device = _resolve_device()
    dtype = _resolve_dtype(args.dtype)

    _print_runtime_info(device, dtype)

    model_args = _build_args_namespace(use_groupnorm=args.use_groupnorm)
    model = ResNet1D(args.num_classes, model_args).to(device=device, dtype=dtype)

    inputs = _make_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_layout=args.input_layout,
        device=device,
        dtype=dtype,
    )

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    layout_description = (
        "(batch, 3, 2, seq_len)"
        if args.input_layout == "iq_3adc"
        else "(batch, 2, seq_len)"
    )
    print(
        "Shape: batch={} seq_len={} layout={} {} num_classes={}".format(
            args.batch_size,
            args.seq_len,
            args.input_layout,
            layout_description,
            args.num_classes,
        )
    )
    print(f"Input tensor shape: {tuple(inputs.shape)}")
    print(f"Parameters: {num_parameters:,}")

    total_seconds, seconds_per_iter = _run_benchmark(
        model=model,
        inputs=inputs,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        device=device,
    )

    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.bench_iters}")
    print(f"Time taken: {total_seconds:.6f} seconds")
    print(f"Time per iteration: {seconds_per_iter:.6f} seconds")
    print(f"Throughput: {args.batch_size / seconds_per_iter:.2f} samples/s")


if __name__ == "__main__":
    main()
