#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run beamgrad's tests on a real NVIDIA GPU and write a Markdown report.

GitHub-hosted runners have no GPU: CI compiles the CUDA code and runs the
kernels under a CPU emulation layer, but only a machine with a GPU runs them
on the device. This script does that and records what was tested where:

1. the environment (GPU, driver, nvcc, PyTorch, commit, and whether the
   library sources equal a release tag's);
2. the C/C++ suites with the native CUDA backend (`dbs_cuda_device_tests`
   runs the compiled kernels on the GPU against the CPU decoder, bit for bit;
   `dbs_cuda_emulation_tests` runs the same kernels on the CPU);
3. the Python suite with the GPU visible (CUDA-vs-CPU parity, autocast,
   estimators on CUDA), against the installed beamgrad;
4. optionally, `benchmarks/benchmark.py --device cuda`.

    python scripts/gpu_report.py --output docs/gpu-report.md --tag v2.1.0 --benchmark

Build settings for CMake can be passed through CMAKE_ARGS, e.g.
`CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=g++-13"` when nvcc does not accept the
default host compiler.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATHS = ["src", "include", "cuda", "python/beamgrad", "python/csrc", "CMakeLists.txt", "setup.py"]


def run(command: list[str], cwd: Path = ROOT, env: dict | None = None) -> tuple[int, str]:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def first_line(command: list[str]) -> str:
    try:
        return run(command)[1].strip().splitlines()[0]
    except (OSError, IndexError):
        return "unavailable"


def environment(tag: str | None) -> list[str]:
    import torch

    import beamgrad

    rows = [
        ("GPU", first_line(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])),
        ("Driver", first_line(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])),
        (
            "nvcc",
            next((line for line in run(["nvcc", "--version"])[1].splitlines() if "release" in line), "unavailable"),
        ),
        ("PyTorch", f"{torch.__version__} (CUDA {torch.version.cuda})"),
        ("beamgrad", f"{beamgrad.__version__}, CUDA operators: {beamgrad.cuda_available()}"),
        ("Python", sys.version.split()[0]),
        ("Commit", first_line(["git", "describe", "--always", "--dirty", "--tags"])),
    ]
    if tag:
        code, diff = run(["git", "diff", "--stat", tag, "--", *LIBRARY_PATHS])
        same = code == 0 and not diff.strip()
        rows.append((f"Library sources vs {tag}", "identical" if same else "differ:\n" + diff.strip()))
    return ["| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]]


def cpp_suites() -> tuple[bool, list[str]]:
    build = ROOT / "build-gpu-report"
    extra = shlex.split(os.environ.get("CMAKE_ARGS", ""))
    code, out = run(
        [
            "cmake",
            "-S",
            ".",
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DDBS_ENABLE_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=native",
            *extra,
        ]
    )
    if code == 0:
        code, out = run(["cmake", "--build", str(build), "--parallel"])
    if code != 0:
        return False, ["Build failed:", "", "```", out[-3000:], "```"]
    code, out = run(["ctest", "--test-dir", str(build), "--output-on-failure"])
    results = [line.strip() for line in out.splitlines() if re.search(r"Test +#\d+:", line)]
    summary = [line.strip() for line in out.splitlines() if "tests passed" in line or "tests failed" in line]
    device_ran = any("dbs_cuda_device_tests" in line and "Passed" in line for line in results)
    lines = ["```", *results, *summary, "```"]
    if not device_ran:
        lines.append("\n**`dbs_cuda_device_tests` did not pass on the device** (skipped means no GPU was usable).")
    return code == 0 and device_ran, lines


def python_suite() -> tuple[bool, list[str]]:
    code, out = run(
        [sys.executable, "-m", "pytest", str(ROOT / "python" / "tests"), "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=Path(os.environ.get("TMPDIR", "/tmp")),
    )
    tail = [line for line in out.splitlines() if " passed" in line or " failed" in line or line.startswith("SKIPPED")]
    code_cuda, out_cuda = run(
        [sys.executable, "-m", "pytest", str(ROOT / "python" / "tests"), "-q", "-p", "no:cacheprovider", "-k", "cuda"],
        cwd=Path(os.environ.get("TMPDIR", "/tmp")),
    )
    cuda_line = next((line for line in out_cuda.splitlines() if " passed" in line or " failed" in line), "")
    lines = ["```", *tail, "```", "", f"Of which CUDA tests (`-k cuda`): `{cuda_line.strip()}`"]
    return code == 0 and code_cuda == 0, lines


def benchmark(repeats: int) -> list[str]:
    code, out = run(
        [sys.executable, str(ROOT / "benchmarks" / "benchmark.py"), "--device", "cuda", "--repeats", str(repeats)]
    )
    table = [line for line in out.splitlines() if line.startswith("|")]
    return table if code == 0 and table else ["```", out[-3000:], "```"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", help="release tag to compare the library sources with, e.g. v2.1.0")
    parser.add_argument("--benchmark", action="store_true", help="also run benchmarks/benchmark.py on the GPU")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cpp_ok, cpp = cpp_suites()
    py_ok, py = python_suite()
    status = "passed" if cpp_ok and py_ok else "FAILED"
    lines = [
        f"# GPU test report: {status}",
        "",
        f"Generated by `scripts/gpu_report.py` on {started}. GitHub-hosted CI has no GPU; this report is what a",
        "real device ran. See [cuda.md](cuda.md) for what each suite checks.",
        "",
        "## Environment",
        "",
        *environment(args.tag),
        "",
        "## C/C++ suites with the native CUDA backend",
        "",
        "`dbs_cuda_device_tests` runs the compiled kernels on the GPU and compares decode traces, final scores",
        "and gradients with the CPU decoder bit for bit; `dbs_cuda_emulation_tests` runs the same kernel source",
        "on the CPU under three thread schedules.",
        "",
        *cpp,
        "",
        "## Python suite with the GPU visible",
        "",
        *py,
    ]
    if args.benchmark:
        lines += ["", "## CUDA benchmark (`benchmarks/benchmark.py --device cuda`)", "", *benchmark(args.repeats)]
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}: {status}")
    sys.exit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
