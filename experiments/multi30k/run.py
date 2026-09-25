# SPDX-License-Identifier: MIT
"""Run the whole Multi30k experiment (or what is missing of it) and summarise it.

    python experiments/multi30k/run.py                 # pretrain, main arms, ablations, summary
    python experiments/multi30k/run.py --summary-only  # just rebuild results.md from runs/

Completed runs (a metrics.json in their directory) are skipped, so the script
can be interrupted and restarted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> finetune arguments. The first group is the main comparison; the
# rest vary one setting of an arm.
MAIN = {
    "mle": ["--arm", "mle"],
    "mrt": ["--arm", "mrt"],
    "margin": ["--arm", "margin"],
    "retain": ["--arm", "retain"],
}
ABLATIONS = {
    "mrt-beam2": ["--arm", "mrt", "--beam", "2"],
    "mrt-beam8": ["--arm", "mrt", "--beam", "8"],
    "mrt-alpha0": ["--arm", "mrt", "--alpha", "0"],
    "margin-rescore": ["--arm", "margin", "--gradient", "rescore"],
}
# Cost: measured separately (seed 0, no evaluation, nothing else running), so
# that time per step is not affected by evaluation or by other jobs.
TIMING = {
    "mle": ["--arm", "mle"],
    "mrt": ["--arm", "mrt"],
    "mrt-rescore": ["--arm", "mrt", "--gradient", "rescore"],
    "margin": ["--arm", "margin"],
    "margin-rescore": ["--arm", "margin", "--gradient", "rescore"],
    "retain": ["--arm", "retain"],
    "mrt-beam8": ["--arm", "mrt", "--beam", "8"],
}
TIMING_STEPS = 110


def run(args, extra: list[str]) -> None:
    command = [sys.executable, str(HERE / "train.py"), *extra, "--data", str(args.data), "--out", str(args.out)]
    print("+", " ".join(command[1:]), flush=True)
    subprocess.run(command, check=True, cwd=HERE)


def load(out: Path, name: str, seeds) -> list[dict]:
    found = []
    for seed in seeds:
        path = out / f"{name}-s{seed}" / "metrics.json"
        if path.exists():
            found.append(json.loads(path.read_text()))
    return found


def mean_std(values: list[float]) -> str:
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def summarise(args) -> str:
    seeds = range(args.seeds)
    pre = load(args.out, "pretrain", seeds)
    baseline = {m["seed"]: m["test_bleu"] for m in load(args.out, "mle", seeds)}
    lines = [
        f"Test BLEU (sacreBLEU, test2016, beam {args.eval_beam}, alpha {args.eval_alpha}), "
        f"mean ± std over {args.seeds} seeds; Δ is the paired difference from `mle`.",
        "",
        "| run | test BLEU | Δ vs mle | best valid BLEU | valid BLEU at the last step |",
        "|---|---|---|---|---|",
        f"| pretrained (MLE, start of fine-tuning) | {mean_std([m['test_bleu'] for m in pre])} | | "
        f"{mean_std([m['valid_bleu'] for m in pre])} | |",
    ]
    table = {}
    for name in [*MAIN, *ABLATIONS]:
        ms = load(args.out, name, seeds)
        if not ms:
            continue
        deltas = [m["test_bleu"] - baseline[m["seed"]] for m in ms if m["seed"] in baseline]
        delta = ""
        if name != "mle" and deltas:
            delta = ("+" if statistics.mean(deltas) >= 0 else "") + mean_std(deltas)
        final = [m["history"][-1]["valid_bleu"] for m in ms]
        lines.append(
            f"| `{name}` | {mean_std([m['test_bleu'] for m in ms])} | {delta} | "
            f"{mean_std([m['valid_bleu'] for m in ms])} | {mean_std(final)} |"
        )
        table[name] = {
            "seeds": [m["seed"] for m in ms],
            "test_bleu": [m["test_bleu"] for m in ms],
            "valid_bleu": [m["valid_bleu"] for m in ms],
            "final_valid_bleu": final,
            "delta_vs_mle": deltas,
        }
        if name == list(MAIN)[-1]:
            lines.append("| *ablations* | | | | |")
    timing = {name: load(args.out, f"timing-{name}", [0]) for name in TIMING}
    if any(timing.values()):
        lines += [
            "",
            f"Cost per fine-tuning step (batch 32; mean of steps 11–{TIMING_STEPS}, seed 0, without evaluation):",
            "",
            "| run | s / step | peak GiB |",
            "|---|---|---|",
        ]
        for name, ms in timing.items():
            if ms:
                m = ms[0]
                lines.append(f"| `{name}` | {m['seconds_per_step']:.3f} | {m['peak_gib']:.2f} |")
                table.setdefault(name, {})["timing"] = {k: m[k] for k in ("seconds_per_step", "peak_gib")}
    (args.out / "results.json").write_text(
        json.dumps({"pretrained": [m["test_bleu"] for m in pre], "runs": table}, indent=2)
    )
    text = "\n".join(lines) + "\n"
    (args.out / "results.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--eval-beam", type=int, default=4)
    parser.add_argument("--eval-alpha", type=float, default=0.6)
    args = parser.parse_args()
    if not args.summary_only:
        groups = [MAIN] if args.no_ablations else [MAIN, ABLATIONS]
        for seed in range(args.seeds):
            if not (args.out / f"pretrain-s{seed}" / "metrics.json").exists():
                run(args, ["pretrain", "--seed", str(seed)])
        for group in groups:
            for seed in range(args.seeds):
                for name, extra in group.items():
                    if not (args.out / f"{name}-s{seed}" / "metrics.json").exists():
                        run(args, ["finetune", *extra, "--name", name, "--seed", str(seed)])
        if not args.no_timing:
            for name, extra in TIMING.items():
                if not (args.out / f"timing-{name}-s0" / "metrics.json").exists():
                    run(
                        args,
                        ["finetune", *extra, "--name", f"timing-{name}", "--seed", "0"]
                        + ["--steps", str(TIMING_STEPS), "--time-only"],
                    )
    print(summarise(args))


if __name__ == "__main__":
    main()
