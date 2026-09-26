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

Ablations:
- `mrt-rescore`: MRT with gradients by re-scoring (`rescore_fn`), 8 seeds.
- `mrt-beam2` and `mrt-beam8`: MRT with 2 and 8 training beams.
- `mrt-alpha0`: MRT without the length penalty in training (evaluation
  keeps α = 0.6).
- `margin-rescore`: the margin loss with gradients by re-scoring.

Re-scoring is the conventional pipeline: generate without gradients, then
score the hypotheses teacher-forced.

**Evaluation.** Beam 4, α = 0.6, with the same search, via
`beamgrad.beam_search`. sacreBLEU (default `13a` tokenization) of the
detokenized output against the raw references.

**Seeds.** `mle`, `mrt` and `mrt-rescore` ran with seeds 0–7; the other
arms with seeds 0–2. Each seed has its own pretrained model, its own
fine-tuning data order and dropout, and all arms share both.

## Results

Test BLEU, mean ± standard deviation over the `n` seeds of each run. Δ is the
difference from `mle` on the same seeds. `wins` counts the seeds where the run
beat `mle`. The p-values are two-sided: an exact sign test and a paired
t-test.

| run | n | test BLEU | Δ vs `mle` | wins | sign test p | t-test p | best valid BLEU | valid BLEU at the last step |
|---|---|---|---|---|---|---|---|---|
| pretrained (start of fine-tuning) | 8 | 37.09 ± 0.27 | | | | | 37.23 ± 0.25 | |
| `mle` | 8 | 37.63 ± 0.34 | | | | | 38.08 ± 0.36 | 37.86 ± 0.43 |
| `mrt` | 8 | **38.02 ± 0.32** | **+0.39 ± 0.31** | **7/8** | 0.070 | **0.009** | 38.34 ± 0.29 | 38.03 ± 0.40 |
| `margin` | 3 | 36.99 ± 0.24 | −0.46 ± 0.48 | 1/3 | 1 | 0.24 | 37.28 ± 0.31 | 34.54 ± 0.74 |
| `retain` | 3 | 37.70 ± 0.38 | +0.25 ± 0.60 | 2/3 | 1 | 0.54 | 38.55 ± 0.23 | 37.99 ± 0.36 |
| *ablations* | | | | | | | | |
| `mrt-rescore` | 8 | 37.74 ± 0.23 | +0.11 ± 0.30 | 5/8 | 0.73 | 0.33 | 38.33 ± 0.34 | 38.02 ± 0.26 |
| `mrt-beam2` | 3 | 37.86 ± 0.12 | +0.42 ± 0.38 | 3/3 | 0.25 | 0.20 | 38.24 ± 0.34 | 37.88 ± 0.39 |
| `mrt-beam8` | 3 | 38.04 ± 0.11 | +0.59 ± 0.48 | 3/3 | 0.25 | 0.16 | 38.65 ± 0.09 | 38.36 ± 0.32 |
| `mrt-alpha0` | 3 | 37.62 ± 0.36 | +0.18 ± 0.25 | 2/3 | 1 | 0.34 | 38.48 ± 0.35 | 38.08 ± 0.58 |
| `margin-rescore` | 3 | 36.99 ± 0.24 | −0.46 ± 0.48 | 1/3 | 1 | 0.24 | 37.28 ± 0.31 | 34.76 ± 0.60 |

Per seed, Δ vs `mle`:
- `mrt`: +0.93, +0.47, +0.26, +0.54, +0.21, +0.29, −0.12, +0.53.
- `mrt-rescore`: +0.68, +0.19, +0.28, −0.30, +0.09, +0.10, −0.03, −0.13.

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
| `retain` | 0.151 | 3.30 |

`retain` took 0.248 s per step before beamgrad 2.1.1, which runs the relaxed
top-k's bisection once for all steps; its results are bit for bit the same.

## What this shows

- **Minimum-risk training on the search's beams helps.** Over 8 seeds it
  beat continued MLE on 7, by +0.39 test BLEU on average. The paired t-test
  gives p = 0.009 and the sign test p = 0.07. The gain holds with 2, 4 and 8
  training beams (3 seeds each). The first 3 seeds alone suggested +0.55
  (p ≈ 0.1); the extra seeds confirmed the effect and made it smaller.
- **With dropout, re-scoring trained less well than going through the
  steps.** MRT through re-scoring gained only +0.11 (5 of 8 seeds, p = 0.33).
  Paired on the same seeds, it trailed MRT through the steps by 0.28 BLEU (6
  of 8 seeds, t-test p = 0.044). The two compute the same gradient function
  but not the same sample of it. Fine-tuning ran with dropout 0.1, and the
  re-scoring pass draws new dropout masks. Its gradient is therefore that of
  a different random model than the one whose beams were scored. That is the
  likely cause; it was not isolated here, for example by a run without
  dropout. Re-scoring still halves the time per step and cuts peak memory by
  about 40%, so it remains the way to fit large models. With dropout, prefer
  going through the steps when memory allows.
- **Train with the length penalty you decode with.** MRT trained with α = 0
  and evaluated with α = 0.6 gained +0.18, against +0.55 on the same 3 seeds
  when the training search matched the test search.
- **A margin against the reference hurts here.** In every seed, validation
  BLEU fell 1–3 points within the first 200 steps and ended 1.8–3.7 points
  below where it started. The best checkpoint was always the starting model,
  which is why `margin`'s test BLEU equals the pretrained model's. The loss
  pushes down whatever beats the reference, and in translation that is
  usually a good paraphrase, since the exact reference was among the 4 beams
  only about 15% of the time. For translation quality, prefer
  `minimum_risk` or a cost-aware objective.
- **Keeping the reference in the beam (`retain`) is inconclusive.** The
  loss uses the relaxed top-k estimator, so it gives gradient to pruned
  candidates, which the final-score losses cannot do. It improved two seeds
  by 0.6 and lost 0.44 on the third.

Beyond these numbers:
- The model is small, 1,000 steps is short, and there is one dataset. Larger
  models, longer fine-tuning and other tasks may behave differently.
- Evaluation uses beam 4 only.
- MRT's cost uses sentence BLEU on BPE ids, not detokenized sacreBLEU.
- No hyperparameters were tuned per arm: the NLL weight, temperature, margin
  and learning rate are the same fixed values everywhere.

## Reproducing

```bash
pip install sacrebleu tokenizers        # plus beamgrad with its CUDA operators
python experiments/multi30k/run.py      # 3 seeds, every run: about an hour on one RTX 4080 SUPER
python experiments/multi30k/run.py --seeds 8 --arms mle,mrt,mrt-rescore   # seeds 3-7: about 50 more minutes
```

`run.py` downloads the data into `data/`, writes every run to `runs/` (its
`metrics.json` has the full configuration and validation history), skips
completed runs, and writes `runs/results.md` and `runs/results.json`.
`python train.py finetune --arm mrt --seed 0 ...` runs a single arm.
[`results/`](results) holds the numbers above: `results.md`, `results.json`
and each run's metrics (`results/runs/*.json`).
