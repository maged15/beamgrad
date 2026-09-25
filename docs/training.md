# Training through beam search

This guide covers what beamgrad differentiates when a model runs inside the
search, how to keep that affordable, which losses and gradient estimators are
available, and what they did in a controlled experiment. The API reference
is [python.md](python.md); the search itself is defined in
[algorithm.md](algorithm.md).

## What the gradient is

`beamgrad.beam_search(step_fn, options, max_steps)` calls the model once per
step. At step `t` it asks for the rows `x_t = log p_θ(· | prefix of beam k)`,
`[B, K, V]`, and the engine selects the next beams from them. Final beam `k`
followed a path `(p_0, v_0), (p_1, v_1), ...` (parent slot and token at each
step), and its score is

```
s_k = ( x_0[p_0, v_0] + x_1[p_1, v_1] + ... ) / lp(L_k),     lp(L) = ((5 + L) / 6) ** alpha
```

Beam selection is piecewise constant in `θ`: a small change of the model
either leaves the chosen beams unchanged or swaps one for another. beamgrad
differentiates `s_k` with the selection held fixed:

```
d s_k / d θ = (1 / lp(L_k)) · Σ_t d x_t[p_t, v_t] / d θ
```

The sum runs over the steps on beam `k`'s path; carried-forward steps after
EOS add nothing. `x_t[p_t, v_t]` is the log-probability of the beam's `t`-th
token given its own prefix. So this is exactly the gradient of the
teacher-forced sequence log-probability of the beams the search found. It
equals the model's true gradient wherever the selection does not change, and
it ignores how the selection would change. (Beams that were not selected get
no gradient through `s`. The estimators below differ here.)

Two consequences follow:

- **A loss on `result.scores` trains the model on its own search output.** It
  gets the sequence-level gradient of whatever the beams were, whether the
  loss is a margin against a reference, an expected cost or a distillation
  target. How the beams were found does not enter the gradient.
- **It can be computed after the search.** Re-running the model
  teacher-forced on the final beams gives the same `d s_k / d θ`. That makes
  the memory-saving mode below exact, not an approximation.
  `python/tests/test_search.py` and `test_models.py` check that the two agree
  to float32 rounding, and `benchmarks/hf_training_memory.py --check` checks
  it on Qwen2.5-0.5B (cosine similarity 1.000000, relative difference
  1.4e-5).

## Memory: three ways to take the gradient

**Through the steps (default).** Each step's autograd graph is kept. The
gradient reaches the model through the rows the search actually read. The
engine's backward hands every step only its own `[B, K]` path gradient, which
is scattered into that step's rows when autograd reaches them. So no dense
`[B, T, K, V]` gradient is ever built, and each step's gradient is freed as
autograd moves on. What remains is the model's own activations for `T`
incremental steps.

**By re-scoring** (`rescore_fn`). The search runs under `torch.no_grad()`,
with inference memory. Then one teacher-forced pass over the final beams
supplies the gradient. The returned scores keep the search's exact values;
only their gradient comes from the re-scoring. It costs one extra forward
pass over `B × K` sequences.

**Re-scoring with activation checkpointing.** Incremental decoding with a
key/value cache cannot use a model's gradient checkpointing. The
teacher-forced pass can, and `beamgrad.hf.CausalLMRescorer` also projects to
the vocabulary in checkpointed chunks, so the `[B·K, T, V]` logits never
exist at once.

Peak memory of one training step (search plus backward, all parameters
trainable) on Qwen2.5-0.5B in bfloat16, RTX 4080 SUPER 16 GB, measured by
`benchmarks/hf_training_memory.py` (memory above the weights):

| B × K × T | through the steps | re-scoring | re-scoring + checkpointing |
|---|--:|--:|--:|
| 4 × 4 × 32 | 3.61 GiB | 1.76 GiB | 1.20 GiB |
| 8 × 8 × 32 | 10.06 GiB | 4.53 GiB | 1.22 GiB |
| 8 × 8 × 64 | out of memory | 7.35 GiB | 1.33 GiB |
| 8 × 8 × 128 | out of memory | out of memory | 1.71 GiB |

Which to use:

- Use **through the steps** when the step function is cheap. You also need
  it when a loss uses the rows themselves: the estimators below read
  `result.step_log_probs`, which re-scoring does not keep.
- Use **re-scoring** for large models. Write a `rescore_fn(sequences,
  lengths) -> [B, K, T]` that returns each beam's teacher-forced token
  log-probabilities, or use `CausalLMRescorer` for Hugging Face causal LMs.
- With **dropout** on, the two modes differ. The re-scoring pass draws new
  dropout masks, so its gradient belongs to a different random function than
  the one the search saw. Both are valid stochastic gradients, but they are
  not identical, as they are without dropout.

## Losses

`beamgrad.losses` works on any `BeamSearchResult`:

- **`structured_margin(result, reference, reference_scores, margin)`**
  computes `max(0, margin + best rival − reference score)`, where the rival is
  the best final beam that is not the reference. It is zero once the
  reference wins by the margin, whether or not the search found it.
  Beam-search optimisation (Wiseman & Rush, 2016) uses this form. Put
  `reference_scores` on the beams' scale with
  `beamgrad.sequence_scores(teacher_forced_token_log_probs, lengths,
  options)`, which applies the same length penalty. With an EOS token, the
  references must end with it and their lengths must count it, as the
  beams' do. A reference without it never matches a beam, so the loss
  cannot reach zero; pass `eos_token=options.eos_token` to be warned.
- **`minimum_risk(result, costs, temperature)`** is the expected cost under
  `softmax(scores / temperature)` over the final beams. This is minimum-risk
  training (Shen et al., 2016). For translation, use 1 − sentence BLEU
  against the reference. Examples whose beams are all dead contribute zero,
  not NaN.
- **`matches(sequences, reference)`** returns `[B, K]` flags that mark which
  beams equal the reference.

Mixing in the token-level NLL keeps sequence-level fine-tuning stable
(Edunov et al., 2018); the experiment below adds 0.3 × NLL.

## Gradient estimators

`beamgrad.estimators` computes smooth quantities from a search's trace and the
rows it read, in plain PyTorch. They run on CPU and CUDA and match the C
library's surrogates.

| estimator | value | gradient reaches |
|---|---|---|
| `path_scores(result, options)` | `[B, T, K]` score of every selected beam at every step (= `trace.scores`) | each beam's path up to step `t` |
| `selected_softmax(result, options, temperature)` | `[B, T, K]` `softmax(scores[t] / τ)` over the selected beams | every selected beam's path, weighted |
| `relaxed_topk(result, options, pool_multiplier, temperature)` | `[B, T, P]` sigmoid k-hot membership of the best `P = K·m` candidates | candidates the search **did not** keep too |

`relaxed_topk` is the one that differentiates the selection itself. At each
step it ranks all candidates as the search does (each live beam extended by
every allowed token, plus finished beams carried forward) and keeps the best
`P`. It then assigns `r_i = sigmoid((score_i − θ) / τ)` with `θ` found by
bisection so that `Σ r_i = K`, and returns them with each candidate's
`parents`, `tokens` and `from_logprob`. The gradient comes from implicit
differentiation through `θ`:

```
d r_i = a_i / τ · (g_i − Σ_j g_j a_j / Σ_j a_j),     a_i = r_i (1 − r_i)
```

This formula pushes a candidate toward or away from the top `K` relative to
its competitors. A loss can use it to keep the reference prefix inside the
beam, which a loss on final scores cannot express once the reference has
been pruned. `relaxed_topk` does not support n-gram blocking or repetition
penalties. It needs the rows, so run `beam_search` through the steps or use
`beamgrad.search`.

## A training loop

A sequence-to-sequence model whose decoder keeps a key/value cache.
[`experiments/multi30k/seq2seq.py`](../experiments/multi30k/seq2seq.py) has
the complete `BeamStep` and `Rescorer`.

```python
import beamgrad
from beamgrad import losses

options = beamgrad.BeamOptions(beam_size=4, eos_token=EOS, length_penalty_alpha=0.6)

for batch in data:
    gold_lp = model.teacher_forced_log_probs(batch.src, batch.gold)     # [B, L], 0 past the end
    nll = -gold_lp.sum() / batch.gold_lengths.sum()

    step = BeamStep(model, batch.src, K=options.beam_size)            # encoder once; cache follows parents
    result = beamgrad.beam_search(step, options, max_steps=batch.max_len, batch_size=B)
    # or, for a large model: rescore_fn=Rescorer(model, batch.src)

    costs = 1 - sentence_bleu(result.sequences, batch.gold)          # [B, K], no gradient
    loss = losses.minimum_risk(result, costs) + 0.3 * nll
    # margin instead:
    #   ref = beamgrad.sequence_scores(gold_lp, batch.gold_lengths, options)
    #   loss = losses.structured_margin(result, batch.gold, ref, margin=1.0) + 0.3 * nll
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Reorder any per-beam state by `beams.parents` (clamped at 0) at the start of
every step after the first. `beams.sequences` holds each beam's prefix if the
model recomputes instead of caching. Test the cache handling the way
`python/tests/test_models.py` does: decoding with the cache must give the
same beams as recomputing every prefix.

## Experiment: Multi30k En→De

A 7.6M-parameter transformer was pretrained with MLE, then fine-tuned for
1,000 steps with each objective, using 3 seeds and the same data order and
settings everywhere. Test BLEU (test2016, beam 4) compared with continued
MLE:

| objective | test BLEU | Δ vs continued MLE (per seed) |
|---|---|---|
| continued MLE (control) | 37.44 ± 0.37 | |
| `minimum_risk` (MRT) | 38.00 ± 0.03 | +0.55 (+0.93, +0.47, +0.26) |
| `structured_margin` against the reference | 36.99 ± 0.24 | −0.46; validation BLEU fell 1.8–3.7 points |
| `relaxed_topk` keep-the-reference-in-the-beam | 37.70 ± 0.38 | +0.25 (+0.60, +0.59, −0.44) |

Minimum-risk training on the search's beams improved every seed, with 2, 4
and 8 training beams. It helped less when the training search omitted the
length penalty that decoding used. The margin against the reference
consistently hurt: most of the time the beams that beat the reference were
good paraphrases, and the loss pushed them down. Re-scoring halved the cost
per step and trained the same way. Three seeds make these results
directional rather than established.
[`experiments/multi30k`](../experiments/multi30k) has the full protocol,
the ablations, cost and memory, and the code to reproduce it.
