# SPDX-License-Identifier: MIT
"""Seeds 3-7 from the same SFT checkpoint: continued SFT vs beamgrad MRT vs DIY MRT (generate + re-scoring)."""

import itertools
import json
import math

import numpy as np
from sacrebleu.metrics import BLEU
from scipy import stats as st

bg = {r["seed"]: r for r in json.load(open("runs/results_seeds3-7.json"))["runs"]}
diy = {r["seed"]: r for r in json.load(open("runs/results_diy.json"))["runs"]}
seeds = sorted(set(bg) & set(diy))
refs = [json.loads(line)["de"] for line in open("data/test.jsonl", encoding="utf-8")]
bleu = BLEU()
rng = np.random.default_rng(0)
N, R = len(refs), 2000
counts = np.stack([np.bincount(i, minlength=N) for i in rng.integers(0, N, (R, N))])


def stats(hyps):
    return np.array(bleu._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)


def score(total):
    return bleu._compute_score_from_stats(list(total)).score


def compare(name, a_key, b_key):
    """b minus a, per seed (paired bootstrap) and across seeds."""
    print(f"\n{name}")
    deltas = []
    for s in seeds:
        a = stats((bg if a_key != "diy_hyps" else diy)[s][a_key])
        b = stats((bg if b_key != "diy_hyps" else diy)[s][b_key])
        boot = np.array([score(c @ b) - score(c @ a) for c in counts])
        d = score(b.sum(0)) - score(a.sum(0))
        deltas.append(d)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(
            f"  seed {s}: {score(a.sum(0)):.2f} -> {score(b.sum(0)):.2f}  delta {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
        )
    d = np.array(deltas)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    flips = np.array(list(itertools.product([1, -1], repeat=n)))
    p_perm = (np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()
    print(
        f"  across {n} seeds: mean {d.mean():+.2f} +/- {d.std(ddof=1):.2f}  wins {int((d > 0).sum())}/{n}  "
        f"t-test p = {2 * st.t.sf(abs(t), n - 1):.3f}  sign-flip p = {p_perm:.3f}"
    )


print("test BLEU per seed (same SFT checkpoint, same batches)")
print("  seed  cont.SFT  beamgrad-MRT  DIY-MRT")
for s in seeds:
    print(f"  {s:4d}  {bg[s]['sft']:8.2f}  {bg[s]['mrt']:12.2f}  {diy[s]['diy']:7.2f}")
print(
    f"  mean  {np.mean([bg[s]['sft'] for s in seeds]):8.2f}  {np.mean([bg[s]['mrt'] for s in seeds]):12.2f}  "
    f"{np.mean([diy[s]['diy'] for s in seeds]):7.2f}"
)
compare("DIY MRT vs continued SFT", "sft_hyps", "diy_hyps")
compare("beamgrad MRT vs DIY MRT", "diy_hyps", "mrt_hyps")
