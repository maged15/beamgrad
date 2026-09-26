Test BLEU (sacreBLEU, test2016, beam 4, alpha 0.6), mean ± std over the `n` seeds of each run. Δ is the paired difference from `mle` on the same seeds; `wins` counts the seeds where the run beat `mle`, with the two-sided exact sign test and paired t-test p-values.

| run | n | test BLEU | Δ vs mle | wins | sign test p | t-test p | best valid BLEU | valid BLEU at the last step |
|---|---|---|---|---|---|---|---|---|
| pretrained (MLE, start of fine-tuning) | 8 | 37.09 ± 0.27 | | | | | 37.23 ± 0.25 | |
| `mle` | 8 | 37.63 ± 0.34 |  |  |  |  | 38.08 ± 0.36 | 37.86 ± 0.43 |
| `mrt` | 8 | 38.02 ± 0.32 | +0.39 ± 0.31 | 7/8 | 0.0703 | 0.00909 | 38.34 ± 0.29 | 38.03 ± 0.40 |
| `margin` | 3 | 36.99 ± 0.24 | -0.46 ± 0.48 | 1/3 | 1 | 0.242 | 37.28 ± 0.31 | 34.54 ± 0.74 |
| `retain` | 3 | 37.70 ± 0.38 | +0.25 ± 0.60 | 2/3 | 1 | 0.54 | 38.55 ± 0.23 | 37.99 ± 0.36 |
| *ablations* | | | | | | | | |
| `mrt-beam2` | 3 | 37.86 ± 0.12 | +0.42 ± 0.38 | 3/3 | 0.25 | 0.199 | 38.24 ± 0.34 | 37.88 ± 0.39 |
| `mrt-beam8` | 3 | 38.04 ± 0.11 | +0.59 ± 0.48 | 3/3 | 0.25 | 0.164 | 38.65 ± 0.09 | 38.36 ± 0.32 |
| `mrt-alpha0` | 3 | 37.62 ± 0.36 | +0.18 ± 0.25 | 2/3 | 1 | 0.342 | 38.48 ± 0.35 | 38.08 ± 0.58 |
| `margin-rescore` | 3 | 36.99 ± 0.24 | -0.46 ± 0.48 | 1/3 | 1 | 0.242 | 37.28 ± 0.31 | 34.76 ± 0.60 |
| `mrt-rescore` | 8 | 37.74 ± 0.23 | +0.11 ± 0.30 | 5/8 | 0.727 | 0.327 | 38.33 ± 0.34 | 38.02 ± 0.26 |

Cost per fine-tuning step (batch 32; mean of steps 11–110, seed 0, without evaluation):

| run | s / step | peak GiB |
|---|---|---|
| `mle` | 0.007 | 0.43 |
| `mrt` | 0.123 | 3.05 |
| `mrt-rescore` | 0.063 | 1.79 |
| `margin` | 0.137 | 3.44 |
| `margin-rescore` | 0.069 | 2.13 |
| `retain` | 0.151 | 3.30 |
| `mrt-beam8` | 0.154 | 4.30 |
