"""GPU transformer forward-pass benchmark (compare setups on identical workload).

Runs a fixed ``nn.TransformerEncoder`` forward loop on CUDA and reports wall time
per iteration, similar to ``gpu_test.py`` but for attention/MLP stacks.

Usage:
    source la-maml_env/bin/activate
    python gpu_transformer_test.py
    python gpu_transformer_test.py --dtype bfloat16 --batch-size 128 --seq-len 1024
    python gpu_transformer_test.py --dtype float16 --bench-iters 50
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn


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


def _build_transformer_encoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerEncoder:
    """Build a standard PyTorch transformer encoder stack."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


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
    """Warm up then time ``bench_iters`` forward passes.

    Args:
        model: Module in eval mode.
        inputs: Batch tensor ``(batch, seq_len, d_model)``.
        warmup_iters: Number of untimed warmup iterations.
        bench_iters: Number of timed iterations.
        device: Execution device.

    Returns:
        Total seconds and seconds per iteration for the timed loop.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(inputs)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = model(inputs)
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
    """Parse CLI args, run transformer benchmark, and print timings."""
    parser = argparse.ArgumentParser(
        description="Benchmark TransformerEncoder forward passes on GPU."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
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

    if args.d_model % args.nhead != 0:
        raise SystemExit(
            f"--d-model ({args.d_model}) must be divisible by --nhead ({args.nhead})."
        )

    _print_runtime_info(device, dtype)

    model = _build_transformer_encoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device=device, dtype=dtype)

    inputs = torch.randn(
        args.batch_size,
        args.seq_len,
        args.d_model,
        device=device,
        dtype=dtype,
    )

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Shape: batch={} seq_len={} d_model={} layers={} heads={} ff={}".format(
            args.batch_size,
            args.seq_len,
            args.d_model,
            args.num_layers,
            args.nhead,
            args.dim_feedforward,
        )
    )
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
