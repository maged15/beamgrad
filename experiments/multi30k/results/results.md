Test BLEU (sacreBLEU, test2016, beam 4, alpha 0.6), mean ± std over 3 seeds; Δ is the paired difference from `mle`.

| run | test BLEU | Δ vs mle | best valid BLEU | valid BLEU at the last step |
|---|---|---|---|---|
| pretrained (MLE, start of fine-tuning) | 36.99 ± 0.24 | | 37.28 ± 0.31 | |
| `mle` | 37.44 ± 0.37 |  | 38.16 ± 0.30 | 37.80 ± 0.31 |
| `mrt` | 38.00 ± 0.03 | +0.55 ± 0.34 | 38.27 ± 0.39 | 38.01 ± 0.49 |
| `margin` | 36.99 ± 0.24 | -0.46 ± 0.48 | 37.28 ± 0.31 | 34.54 ± 0.74 |
| `retain` | 37.70 ± 0.38 | +0.25 ± 0.60 | 38.55 ± 0.23 | 37.99 ± 0.36 |
| *ablations* | | | | |
| `mrt-beam2` | 37.86 ± 0.12 | +0.42 ± 0.38 | 38.24 ± 0.34 | 37.88 ± 0.39 |
| `mrt-beam8` | 38.04 ± 0.11 | +0.59 ± 0.48 | 38.65 ± 0.09 | 38.36 ± 0.32 |
| `mrt-alpha0` | 37.62 ± 0.36 | +0.18 ± 0.25 | 38.48 ± 0.35 | 38.08 ± 0.58 |
| `margin-rescore` | 36.99 ± 0.24 | -0.46 ± 0.48 | 37.28 ± 0.31 | 34.76 ± 0.60 |

Cost per fine-tuning step (batch 32; mean of steps 11–110, seed 0, without evaluation):

| run | s / step | peak GiB |
|---|---|---|
| `mle` | 0.007 | 0.43 |
| `mrt` | 0.123 | 3.05 |
| `mrt-rescore` | 0.063 | 1.79 |
| `margin` | 0.137 | 3.44 |
| `margin-rescore` | 0.069 | 2.13 |
| `retain` | 0.248 | 3.30 |
| `mrt-beam8` | 0.154 | 4.30 |
