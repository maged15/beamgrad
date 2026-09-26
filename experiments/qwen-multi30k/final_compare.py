# SPDX-License-Identifier: MIT
"""beamgrad MRT vs DIY MRT (generate + re-scoring), same SFT checkpoint and batches.

Decoder A = beamgrad search (seeds 3-13), decoder B = HF generate() (seeds 8-13).
"""

import itertools
import json
import math

import numpy as np
from sacrebleu.metrics import BLEU
from scipy import stats as st

refs = [json.loads(line)["de"] for line in open("data/test.jsonl", encoding="utf-8")]
bgdec, hfdec = {}, {}  # seed -> (beamgrad hyps, diy hyps)
first = {r["seed"]: r for r in json.load(open("runs/results_seeds3-7.json"))["runs"]}
for r in json.load(open("runs/results_diy.json"))["runs"]:
    bgdec[r["seed"]] = (first[r["seed"]]["mrt_hyps"], r["diy_hyps"])
for r in json.load(open("runs/results_pair.json"))["runs"]:
    bgdec[r["seed"]] = (r["beamgrad_bgdec_hyps"], r["diy_bgdec_hyps"])
    hfdec[r["seed"]] = (r["beamgrad_hfdec_hyps"], r["diy_hfdec_hyps"])

bleu = BLEU()
rng = np.random.default_rng(0)
N, R = len(refs), 2000
counts = np.stack([np.bincount(i, minlength=N) for i in rng.integers(0, N, (R, N))])


def stats(hyps):
    return np.array(bleu._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)


def score(total):
    return bleu._compute_score_from_stats(list(total)).score


def report(title, runs):
    print(f"\n{title}")
    print("  seed  DIY     beamgrad  delta   95% CI (bootstrap over sentences)")
    deltas, A, B = [], 0, 0
    for s in sorted(runs):
        b, a = stats(runs[s][0]), stats(runs[s][1])
        A, B = A + a, B + b
        boot = np.array([score(c @ b) - score(c @ a) for c in counts])
        d = score(b.sum(0)) - score(a.sum(0))
        deltas.append(d)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {s:4d}  {score(a.sum(0)):.2f}   {score(b.sum(0)):.2f}    {d:+.2f}   [{lo:+.2f}, {hi:+.2f}]")
    d = np.array(deltas)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    flips = np.array(list(itertools.product([1, -1], repeat=n)))
    p_perm = (np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()
    boot = np.array([score(c @ B) - score(c @ A) for c in counts])
    print(
        f"  mean  {np.mean([score(stats(runs[s][1]).sum(0)) for s in runs]):.2f}   "
        f"{np.mean([score(stats(runs[s][0]).sum(0)) for s in runs]):.2f}    {d.mean():+.2f}"
    )
    print(
        f"  beamgrad ahead on {int((d > 0).sum())}/{n} seeds;  mean delta {d.mean():+.2f} +/- {d.std(ddof=1):.2f} (sd)"
    )
    print(
        f"  paired t-test p = {2 * st.t.sf(abs(t), n - 1):.3f};  exact sign-flip p = {p_perm:.3f};  "
        f"pooled bootstrap 95% CI [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]"
    )


report(f"decoder: beamgrad search ({len(bgdec)} seeds)", bgdec)
report(f"decoder: HF generate() ({len(hfdec)} seeds)", hfdec)
if hfdec:
    diff = [sum(x != y for x, y in zip(bgdec[s][i], hfdec[s][i], strict=True)) for s in hfdec for i in (0, 1)]
    print(f"\ntest outputs that differ between the two decoders, per model: {diff} (of {N})")
