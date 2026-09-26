# SPDX-License-Identifier: MIT
"""MRT vs continued SFT test BLEU: paired bootstrap per seed (Koehn 2004), then across seeds.

python significance.py runs/results.json [runs/results_seeds3-7.json ...]
"""

import itertools
import json
import math
import sys

import numpy as np
from sacrebleu.metrics import BLEU

files = sys.argv[1:] or ["runs/results.json"]
runs, n_test = [], None
for f in files:
    res = json.load(open(f))
    n_test = res["args"]["test"]
    for run in res["runs"]:
        run["sft_start"] = res["sft"]
        runs.append(run)
refs = [json.loads(line)["de"] for line in open("data/test.jsonl", encoding="utf-8")][:n_test]
bleu = BLEU()


def stats(hyps):
    return np.array(bleu._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)  # [N, 2 + 2*4]


def score(total):
    return bleu._compute_score_from_stats(list(total)).score


rng = np.random.default_rng(0)
N, R = len(refs), 2000
counts = np.stack([np.bincount(i, minlength=N) for i in rng.integers(0, N, (R, N))])  # [R, N]
deltas = []
print("per seed (paired bootstrap over test sentences, 2000 resamples)")
for run in sorted(runs, key=lambda r: r["seed"]):
    a, b = stats(run["sft_hyps"]), stats(run["mrt_hyps"])
    boot = np.array([score(c @ b) - score(c @ a) for c in counts])
    d = score(b.sum(0)) - score(a.sum(0))
    deltas.append(d)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"  seed {run['seed']}: SFT {score(a.sum(0)):.2f}  MRT {score(b.sum(0)):.2f}  delta {d:+.2f}  "
        f"95% CI [{lo:+.2f}, {hi:+.2f}]  P(MRT <= SFT) {(boot <= 0).mean():.3f}   (SFT start {run['sft_start']:.2f})"
    )

d = np.array(deltas)
n = len(d)
mean, sd = d.mean(), d.std(ddof=1)
t = mean / (sd / math.sqrt(n))
try:
    from scipy import stats as st

    p_t = 2 * st.t.sf(abs(t), n - 1)
except ImportError:
    p_t = float("nan")
# Exact sign-flip permutation test on the per-seed deltas (two-sided).
flips = np.array(list(itertools.product([1, -1], repeat=n)))
p_perm = (np.abs((flips * d).mean(1)) >= abs(mean) - 1e-12).mean()
print(f"\nacross {n} seeds: mean delta {mean:+.2f} +/- {sd:.2f} (sd)  MRT wins {int((d > 0).sum())}/{n}")
print(f"  paired t-test t = {t:.2f}, p = {p_t:.3f};  exact sign-flip permutation p = {p_perm:.3f}")

# Pooled: all seeds' test outputs as one big paired corpus.
A = sum(stats(r["sft_hyps"]) for r in runs)
B = sum(stats(r["mrt_hyps"]) for r in runs)
boot = np.array([score(c @ B) - score(c @ A) for c in counts])
print(
    f"  pooled over seeds, bootstrap over sentences: delta {score(B.sum(0)) - score(A.sum(0)):+.2f}  "
    f"95% CI [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]  "
    f"P(MRT <= SFT) {(boot <= 0).mean():.3f}"
)
