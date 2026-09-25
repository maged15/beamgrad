# Multi30k En→De: does training through beam search help?

This is a controlled comparison of sequence-level objectives computed on
beamgrad's beam search. Each objective fine-tunes the same pretrained
translation model, and continued MLE is the control. Everything else is the
same: data order, steps, optimizer, dropout and evaluation.

## Protocol

**Data.** Multi30k En→De (Elliott et al., 2016), from
[`bentrevett/multi30k`](https://huggingface.co/datasets/bentrevett/multi30k):
29,000 training pairs, 1,014 validation pairs, and test2016 (1,000 pairs).
Joint BPE with 8,000 tokens (Hugging Face `tokenizers`, NFKC, Metaspace),
learned on the training sentences.

**Model.** A pre-LN transformer with 3 encoder and 3 decoder layers,
d = 256, 4 heads, feed-forward 1,024, tied source, target and output
embeddings, learned positions: 7.6M parameters. The decoder keeps a
key/value cache that `beams.parents` reorders during the search
([`seq2seq.py`](seq2seq.py)).

**Pretraining (MLE).** One model per seed. 30 epochs, batch 128
sentences, Adam (β = 0.9, 0.98), learning rate 1e-3 with 1,000 warm-up steps
and inverse square-root decay, dropout 0.3, label smoothing 0.1, gradient
clipping 1.0. Checkpoint with the best validation BLEU (evaluated every
5 epochs). Takes 1.8 minutes on the GPU below.

**Fine-tuning.** Every arm starts from its seed's checkpoint: 1,000 steps
of batch 32 (about 1.1 epochs), Adam at 1e-4, dropout 0.1, clipping 1.0.
Validation BLEU is measured at step 0 and every 200 steps. The
checkpoint with the best validation BLEU is evaluated on test; step 0, the
unchanged pretrained model, is a candidate too. During training the search
uses beam 4, length penalty α = 0.6, EOS, and bans padding, BOS and unknown
tokens. It runs through the steps, as in `beamgrad.beam_search`'s default.

**Arms.** The sequence-level losses add 0.3 × the label-smoothed token NLL
(Edunov et al., 2018).

| arm | loss |
|---|---|
| `mle` | label-smoothed token NLL (the control) |
| `mrt` | `losses.minimum_risk` over the 4 final beams: expected (1 − smoothed sentence BLEU on BPE ids), temperature 1 |
| `margin` | `losses.structured_margin`: the reference, scored teacher-forced with `sequence_scores`, must beat the best non-reference beam by 1 |
| `retain` | `estimators.relaxed_topk` (pool 16, temperature 1): −log of the relaxed top-k weight of the candidate that extends the reference prefix, at every step while the prefix is still in the beam |

Ablations, each run with 3 seeds:
- `mrt-beam2` and `mrt-beam8`: MRT with 2 and 8 training beams.
- `mrt-alpha0`: MRT without the length penalty in training (evaluation
  keeps α = 0.6).
- `margin-rescore`: the margin loss with gradients by re-scoring
  (`rescore_fn`). This is the conventional pipeline: generate without
  gradients, then score the hypotheses teacher-forced.

**Evaluation.** Beam 4, α = 0.6, with the same search, via
`beamgrad.beam_search`. sacreBLEU (default `13a` tokenization) of the
detokenized output against the raw references.

**Seeds.** 0, 1 and 2. Each seed has its own pretrained model, its own
fine-tuning data order and dropout, and all arms share both.

## Results

Test BLEU, mean ± standard deviation over 3 seeds. Δ is the per-seed
difference from `mle`, averaged.

| run | test BLEU | Δ vs `mle` | Δ per seed | best valid BLEU | valid BLEU at the last step |
|---|---|---|---|---|---|
| pretrained (start of fine-tuning) | 36.99 ± 0.24 | | | 37.28 ± 0.31 | |
| `mle` | 37.44 ± 0.37 | | | 38.16 ± 0.30 | 37.80 ± 0.31 |
| `mrt` | **38.00 ± 0.03** | **+0.55 ± 0.34** | +0.93, +0.47, +0.26 | 38.27 ± 0.39 | 38.01 ± 0.49 |
| `margin` | 36.99 ± 0.24 | −0.46 ± 0.48 | +0.09, −0.79, −0.67 | 37.28 ± 0.31 | 34.54 ± 0.74 |
| `retain` | 37.70 ± 0.38 | +0.25 ± 0.60 | +0.60, +0.59, −0.44 | 38.55 ± 0.23 | 37.99 ± 0.36 |
| *ablations* | | | | | |
| `mrt-beam2` | 37.86 ± 0.12 | +0.42 ± 0.38 | +0.86, +0.21, +0.18 | 38.24 ± 0.34 | 37.88 ± 0.39 |
| `mrt-beam8` | 38.04 ± 0.11 | +0.59 ± 0.48 | +1.10, +0.52, +0.16 | 38.65 ± 0.09 | 38.36 ± 0.32 |
| `mrt-alpha0` | 37.62 ± 0.36 | +0.18 ± 0.25 | +0.17, +0.42, −0.07 | 38.48 ± 0.35 | 38.08 ± 0.58 |
| `margin-rescore` | 36.99 ± 0.24 | −0.46 ± 0.48 | +0.09, −0.79, −0.67 | 37.28 ± 0.31 | 34.76 ± 0.60 |

Cost of one fine-tuning step (batch 32; mean of steps 11–110 on seed 0, no
evaluation, nothing else running; RTX 4080 SUPER, PyTorch 2.14, CUDA 12.6):

| run | s / step | peak GiB |
|---|---|---|
| `mle` | 0.007 | 0.43 |
| `mrt` (through the steps) | 0.123 | 3.05 |
| `mrt` (re-scoring) | 0.063 | 1.79 |
| `mrt-beam8` | 0.154 | 4.30 |
| `margin` (through the steps) | 0.137 | 3.44 |
| `margin` (re-scoring) | 0.069 | 2.13 |
| `retain` | 0.248 | 3.30 |

## What this shows

- **Minimum-risk training on the search's beams helps.** It beat
  continued MLE on every seed, with a mean of +0.55 test BLEU. Its test BLEU
  varies less across seeds than the control's (±0.03 vs ±0.37), and the
  gain holds with 2, 4 and 8 training beams. With three seeds this is
  consistent but not statistically established. A paired t-test gives
  p ≈ 0.1.
- **Train with the length penalty you decode with.** MRT trained with α = 0
  and evaluated with α = 0.6 gained +0.18, against +0.55 when the training
  search matched the test search.
- **A margin against the reference hurts here.** In every seed, validation
  BLEU fell 1–3 points within the first 200 steps and ended 1.8–3.7 points
  below where it started. The
  best checkpoint was always the starting model, which is why `margin`'s
  test BLEU equals the pretrained model's. The loss pushes down whatever
  beats the reference, and in translation that is usually a good paraphrase,
  since the exact reference was among the 4 beams only about 15% of the
  time. The README example uses this loss because it is the simplest to
  explain. For translation quality, prefer `minimum_risk` or a cost-aware
  objective.
- **Re-scoring gives the same training at half the cost.** Re-scoring and
  going through the steps compute the same gradient. They differ only in
  dropout masks, because the re-scoring pass draws new ones. The margin arm
  follows the same trajectory both ways, and re-scoring halves the time and
  cuts peak memory by 40%.
- **Keeping the reference in the beam (`retain`) is inconclusive.** The
  loss uses the relaxed top-k estimator, so it gives gradient to pruned
  candidates, which the final-score losses cannot do. It improved two seeds
  by 0.6 and lost 0.44 on the third. It also costs twice as much per step as
  MRT, because it ranks every candidate at every step in Python.

Beyond these numbers:
- The model is small, and 1,000 steps is short. Larger models and longer
  fine-tuning may behave differently.
- Evaluation uses beam 4 only.
- MRT's cost uses sentence BLEU on BPE ids, not detokenized sacreBLEU.
- No hyperparameters were tuned per arm: the NLL weight, temperature, margin
  and learning rate are the same fixed values everywhere.

## Reproducing

```bash
pip install sacrebleu tokenizers        # plus beamgrad with its CUDA operators
python experiments/multi30k/run.py      # about an hour on one RTX 4080 SUPER
```

`run.py` downloads the data into `data/`, writes every run to `runs/` (its
`metrics.json` has the full configuration and validation history), skips
completed runs, and writes `runs/results.md` and `runs/results.json`.
`python train.py finetune --arm mrt --seed 0 ...` runs a single arm.
[`results/`](results) holds the numbers above: `results.md`, `results.json`
and each run's metrics (`results/runs/*.json`).
