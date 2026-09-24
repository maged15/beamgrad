# SPDX-License-Identifier: MIT
"""Benchmark beamgrad against a plain PyTorch beam search.

For each shape the script times, on every available device:
  * beamgrad.final_scores forward,
  * beamgrad forward + backward,
  * a reference beam search written with torch.topk (forward only),
and checks that beamgrad's scores match the reference. Results print as a
Markdown table; --csv also writes them to a file with environment metadata.

    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --device cuda --repeats 50 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from collections.abc import Callable

import torch

import beamgrad

SHAPES = [  # (B, T, K, V)
    (1, 16, 4, 32_000),
    (8, 16, 4, 32_000),
    (8, 32, 8, 32_000),
    (4, 16, 8, 128_000),
    (16, 64, 4, 50_000),
]


def reference_beam_search(x: torch.Tensor, k: int) -> torch.Tensor:
    """Hard beam search with torch.topk: [B, T, K, V] -> [B, K] (no EOS, no penalty)."""
    B, T, K, V = x.shape
    scores = torch.full((B, K), float("-inf"), device=x.device)
    scores[:, 0] = 0.0
    for t in range(T):
        candidates = (scores[:, :, None] + x[:, t]).reshape(B, K * V)
        scores = candidates.topk(k, dim=1).values
    return scores


def timed(fn: Callable[[], object], device: str, repeats: int) -> float:
    """Median wall time in milliseconds."""
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def run(device: str, shape: tuple[int, int, int, int], repeats: int) -> dict:
    B, T, K, V = shape
    torch.manual_seed(0)
    x = torch.randn(B, T, K, V, device=device).log_softmax(-1)
    options = beamgrad.BeamOptions(beam_size=K, validate_inputs=False)
    grad = torch.ones(B, K, device=device)

    def forward():
        return beamgrad.final_scores(x, options)

    def forward_backward():
        leaf = x.detach().requires_grad_(True)
        beamgrad.final_scores(leaf, options).backward(grad)

    ours = forward()
    reference = reference_beam_search(x, K)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-4)
    return {
        "device": device,
        "B": B,
        "T": T,
        "K": K,
        "V": V,
        "forward_ms": timed(forward, device, repeats),
        "forward_backward_ms": timed(forward_backward, device, repeats),
        "torch_topk_ms": timed(lambda: reference_beam_search(x, K), device, repeats),
    }


def environment(device: str) -> str:
    parts = [f"beamgrad {beamgrad.__version__}", f"torch {torch.__version__}", f"{torch.get_num_threads()} CPU threads"]
    if device == "cuda":
        parts.append(torch.cuda.get_device_name())
    else:
        parts.append(platform.processor() or platform.machine())
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--csv", help="also write results to this CSV file")
    args = parser.parse_args()

    devices = ["cpu", "cuda"] if args.device == "all" else [args.device]
    if "cuda" in devices and not beamgrad.cuda_available():
        if args.device == "cuda":
            raise SystemExit("CUDA requested but beamgrad's CUDA operators or a GPU are unavailable")
        devices.remove("cuda")

    rows = []
    for device in devices:
        print(f"\n{environment(device)}\n")
        print("| B | T | K | V | forward (ms) | forward + backward (ms) | torch.topk reference (ms) |")
        print("|--:|--:|--:|--:|--:|--:|--:|")
        for shape in SHAPES:
            row = run(device, shape, args.repeats)
            rows.append(row)
            print(
                f"| {row['B']} | {row['T']} | {row['K']} | {row['V']:,} | {row['forward_ms']:.2f} "
                f"| {row['forward_backward_ms']:.2f} | {row['torch_topk_ms']:.2f} |"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            f.write(f"# {environment(devices[0])}\n")
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
