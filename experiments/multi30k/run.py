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
import math
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
    "mrt-rescore": ["--arm", "mrt", "--gradient", "rescore"],
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


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def paired_t_p(deltas: list[float]) -> float:
    """Two-sided p-value of a paired t-test (a one-sample t-test on the differences)."""
    n = len(deltas)
    if n < 2:
        return math.nan
    mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
    if sd == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def sign_test_p(deltas: list[float]) -> float:
    """Two-sided exact sign test: how often a fair coin splits the non-zero differences this unevenly."""
    signs = [d > 0 for d in deltas if d != 0]
    n, wins = len(signs), sum(signs)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, n - wins) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def summarise(args) -> str:
    seeds = range(args.seeds)
    pre = load(args.out, "pretrain", seeds)
    baseline = {m["seed"]: m["test_bleu"] for m in load(args.out, "mle", seeds)}
    lines = [
        f"Test BLEU (sacreBLEU, test2016, beam {args.eval_beam}, alpha {args.eval_alpha}), mean ± std over the "
        "`n` seeds of each run. Δ is the paired difference from `mle` on the same seeds; `wins` counts the "
        "seeds where the run beat `mle`, with the two-sided exact sign test and paired t-test p-values.",
        "",
        "| run | n | test BLEU | Δ vs mle | wins | sign test p | t-test p | best valid BLEU "
        "| valid BLEU at the last step |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| pretrained (MLE, start of fine-tuning) | {len(pre)} | {mean_std([m['test_bleu'] for m in pre])} | | | | | "
        f"{mean_std([m['valid_bleu'] for m in pre])} | |",
    ]
    table = {}
    for name in [*MAIN, *ABLATIONS]:
        ms = load(args.out, name, seeds)
        if not ms:
            continue
        deltas = [m["test_bleu"] - baseline[m["seed"]] for m in ms if m["seed"] in baseline]
        delta = wins = sign_p = t_p = ""
        if name != "mle" and deltas:
            delta = ("+" if statistics.mean(deltas) >= 0 else "") + mean_std(deltas)
            wins = f"{sum(d > 0 for d in deltas)}/{len(deltas)}"
            sign_p = f"{sign_test_p(deltas):.3g}"
            t_p = f"{paired_t_p(deltas):.3g}" if len(deltas) > 1 else "–"
        final = [m["history"][-1]["valid_bleu"] for m in ms]
        lines.append(
            f"| `{name}` | {len(ms)} | {mean_std([m['test_bleu'] for m in ms])} | {delta} | {wins} | {sign_p} "
            f"| {t_p} | {mean_std([m['valid_bleu'] for m in ms])} | {mean_std(final)} |"
        )
        table[name] = {
            "seeds": [m["seed"] for m in ms],
            "test_bleu": [m["test_bleu"] for m in ms],
            "valid_bleu": [m["valid_bleu"] for m in ms],
            "final_valid_bleu": final,
            "delta_vs_mle": deltas,
        }
        if name != "mle" and len(deltas) > 1:
            table[name]["sign_test_p"] = sign_test_p(deltas)
            table[name]["paired_t_p"] = paired_t_p(deltas)
        if name == list(MAIN)[-1]:
            lines.append("| *ablations* | | | | | | | | |")
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
    parser.add_argument(
        "--arms", help="comma-separated runs to train (default: all); the summary shows every run found"
    )
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--eval-beam", type=int, default=4)
    parser.add_argument("--eval-alpha", type=float, default=0.6)
    args = parser.parse_args()
    if not args.summary_only:
        groups = [MAIN] if args.no_ablations else [MAIN, ABLATIONS]
        if args.arms:
            wanted = set(args.arms.split(","))
            unknown = wanted - set(MAIN) - set(ABLATIONS)
            if unknown:
                parser.error(f"unknown runs: {', '.join(sorted(unknown))}")
            groups = [{k: v for k, v in group.items() if k in wanted} for group in (MAIN, ABLATIONS)]
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
