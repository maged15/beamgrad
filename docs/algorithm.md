# How beamgrad works

beamgrad runs **exact, hard beam search** in the forward pass and returns
**surrogate gradients** in the backward pass. This page defines both precisely.

## Inputs

Decoding consumes per-step log-probabilities `log_probs[t, k, v]` of shape
`[T, K, V]` (or `[B, T, K, V]` for a batch):

- `t` is the decoding step, `k` the beam, `v` the vocabulary token;
- row `log_probs[t, k]` is the next-token distribution *conditioned on beam
  `k`'s prefix* after step `t - 1`;
- at `t = 0` only beam 0 is live (all beams share the empty prefix);
- `-inf` marks an impossible token; NaN and `+inf` are rejected when input
  validation is on, and otherwise never selected. Validation covers exactly
  the rows the search reads: those of live, unfinished beams (every element,
  including banned or masked tokens). Rows of beams that are not live, such as
  beams 1..K-1 at `t = 0`, and of finished beams are never read.

The tensor can come from a model evaluated on each beam's prefix, or from any
scoring function that produces one distribution per beam and step.

## Forward: hard beam search

Each beam carries a cumulative log-probability (`raw` score), a length, and a
finished flag. At every step `t`:

1. **Expand.** Every live, unfinished beam `p` proposes every token `v`:

   ```
   raw(p, v)   = raw(p) + log_probs[t, p, v]
   length      = length(p) + 1
   score(p, v) = raw(p, v) / penalty(length)
   ```

   where `penalty(l) = ((5 + max(l, 1)) / 6) ** alpha` is the GNMT length
   penalty (`alpha = 0` disables it, so `score == raw`).

2. **Carry finished beams.** A beam that has emitted EOS is finished. Instead
   of expanding it, it proposes one candidate that repeats EOS with its score,
   raw score and length unchanged. Finished hypotheses therefore compete
   fairly with longer ones at every later step.

3. **Mask early EOS and apply constraints.** While a hypothesis is shorter
   than `min_length` tokens (counting the EOS itself), EOS is not proposed.
   Banned tokens are never proposed. With `no_repeat_ngram_size = n`, a beam
   does not propose a token that would complete an n-gram already present in
   its own prefix. With `repetition_penalty = r > 1`, a token already in the
   beam's prefix is proposed with `log_probs[t, p, v] - log(r)`.

4. **Select.** The `K` best candidates become the next beams. Candidates are
   ranked by a strict total order, so selection is deterministic:

   | key | direction |
   |---|---|
   | score | higher first |
   | raw score | higher first |
   | parent beam | lower first |
   | token | lower first |
   | length | shorter first |
   | origin | expanded before carried forward |

The final scores are the `K` surviving scores after step `T - 1`, best first.

All backends implement this order exactly: the scalar, AVX-512, AVX2, SSE4.2
and NEON CPU kernels and the CUDA engine select the same beams with
bitwise-identical scores and gradients. No operation is fused into a
multiply-add, and the length penalty is evaluated in double precision from
basic IEEE operations only and rounded once, so every CPU and the GPU compute
the same bits. (The C-level softmax and sigmoid surrogates below use the
platform's `expf`, so they are identical across kernel paths on one machine
but may differ in the last bit between C libraries.) Tests enforce this (see
[development.md](development.md#testing)).

## Backward: surrogate gradients

Top-k selection is piecewise constant, so its true derivative is zero almost
everywhere. beamgrad instead differentiates the **scores of the selected
hypotheses with the selection held fixed**.

### Final scores (PyTorch, JAX, C, CUDA)

Final beam `k`'s score is a sum of the log-probabilities along its path,
divided by its length penalty:

```
final[k] = ( log_probs[0, p_0, v_0] + log_probs[1, p_1, v_1] + ... ) / penalty(length_k)
```

where `(p_t, v_t)` is the parent beam and token the path took at step `t`
(carry-forward steps add nothing). The backward pass returns exactly this
function's gradient:

```
d final[k] / d log_probs[t, p_t, v_t] = 1 / penalty(length_k)
```

for every step on beam `k`'s path, and zero elsewhere. Upstream gradients of
all final beams are accumulated, including where paths share a prefix. This is
the true derivative wherever the selection does not change under a small
perturbation of `log_probs`, which the test suite checks against finite
differences.

This is what `beamgrad.final_scores` differentiates. Typical uses: margin
losses that separate a reference sequence from the best competing beam
(see [`examples/train.py`](../examples/train.py)), expected-risk objectives
over the beam, and sequence-level distillation.

### Additional surrogates (C ABI and `beamgrad.estimators`)

The C library (`dbs_decode` / `dbs_backward`) additionally exposes two smooth
relaxations of the selection itself, and accepts upstream gradients for them.
In Python, `beamgrad.estimators.selected_softmax` and `relaxed_topk` compute
the same quantities in PyTorch, on CPU or CUDA, with the same gradients (the
test suite checks them against the C library):

- **Selected-beam weights.** At each step, `weights[t] = softmax(scores[t] /
  selected_temperature)` over the `K` selected beams. The gradient
  `w * (g - <w, g>) / temperature` flows to each beam's score and from there,
  through its path, to `log_probs`.
- **Relaxed top-k pool** (opt-in: set `relaxed_pool_multiplier >= 1`). Each
  step also keeps the best `P = K * relaxed_pool_multiplier` candidates and a
  sigmoid "k-hot"
  relaxation of membership in the top `K`:
  `r_i = sigmoid((score_i - theta) / soft_topk_temperature)`, with `theta`
  found by bisection so that `sum_i r_i = K`. The bisection stops once the
  sum is within `soft_topk_tolerance` of `K` or `theta` is bracketed to
  `soft_topk_tolerance × soft_topk_temperature`. It bisects `theta`'s offset
  from the best candidate's score, so this accuracy holds for scores in the
  thousands too. Its gradient is obtained by
  implicit differentiation through `theta`:
  `d r_i = a_i / tau * (g_i - sum_j g_j a_j / sum_j a_j)` with
  `a_i = r_i (1 - r_i)`.

The pool never changes which beams are selected, so it does not affect the
final scores or the final-score gradient.

## Guarantees and limits

- **Deterministic.** Same input, same output: no randomness, a total order on
  candidates, and fixed accumulation order in every backward pass.
- **Batch invariant.** Each example of a batch is decoded independently.
- **Surrogate, not exact.** Gradients do not account for how selection would
  change; a beam that is not selected receives no gradient, except through
  the relaxed top-k pool, which differentiates a smooth version of the
  selection instead.
- **Rows conditioned on the search.** The search consumes `[T, K, V]` rows.
  When each row depends on its beam's prefix (an autoregressive model), the
  model must produce the rows of step `t` after the beams of step `t - 1` are
  known. `beamgrad.beam_search` (Python) and `dbs_decode_model_steps_ex` (C)
  run the search step by step and ask the model for each step's rows, telling
  it which previous beam each row continues. The scores are then those of the
  stacked rows, so the gradient above applies unchanged, and reaches the
  model through every row a final beam's path used.

## What beamgrad is and isn't

**The default gradient is not new.** `final_scores`, `search` and
`beam_search` differentiate each selected beam's length-penalised
log-probability, `s_k = (sum_t log p(y_t | y_<t)) / penalty(length_k)`, with
the selection held fixed. That is the gradient you get by scoring the
returned beams again under teacher forcing and differentiating that score.
`beam_search(..., rescore_fn=...)` computes it exactly that way, and the
tests check that both routes agree, up to floating-point differences between
incremental and parallel evaluation. With dropout the two passes draw
different masks.

What beamgrad adds is engineering:
- an exact, deterministic beam search that selects the same beams with
  bit-identical scores on every CPU kernel and on the GPU, fast enough to
  run inside training;
- training through it in bounded memory: per-step gradients, or re-scoring
  with activation checkpointing (see [training.md](training.md));
- operators that work with `torch.compile`, `torch.export`, `torch.vmap` and
  `torch.func`.

**Only the relaxed top-k differentiates the selection.**
`beamgrad.estimators.relaxed_topk`, and the C library's relaxed pool
(`relaxed_pool_multiplier`), replace "kept by the search" with a standard
sigmoid top-k relaxation: `r_i = sigmoid((s_i - theta) / tau)` with
`sum_i r_i = K`, differentiated implicitly through `theta`. That sends
gradient to candidates the search pruned. Everything else treats the
selection as fixed: final scores, `path_scores` and `selected_softmax`.
The search itself always stays hard, so the model only ever sees the beams
that were actually selected.

**Related work.**

- *Beam-search optimisation.* Wiseman and Rush, "Sequence-to-Sequence
  Learning as Beam-Search Optimization", EMNLP 2016. It trains with a
  margin between the gold sequence and the beam, and restarts the search
  from the gold prefix when that prefix falls off the beam (a LaSO-style
  update). `losses.structured_margin` is the sequence-level margin against
  the final beams, without restarts.
- *Minimum risk training.* Shen et al., "Minimum Risk Training for Neural
  Machine Translation", ACL 2016. It minimises the expected cost of
  candidate outputs under the model's distribution renormalised over a
  candidate set. `losses.minimum_risk` does this over the final beams of
  the search. Edunov et al., "Classical Structured Prediction Losses for
  Sequence to Sequence Learning", NAACL 2018, compares these sequence-level
  losses, including mixing them with the token-level loss.
- *Continuous relaxations of beam search.* Goyal, Neubig, Dyer and
  Berg-Kirkpatrick, "A Continuous Relaxation of Beam Search for End-to-end
  Training of Neural Sequence Models", AAAI 2018. It replaces the hard
  top-k inside the search with a soft one, so decoding itself becomes
  differentiable and soft beams feed the next step. beamgrad does not relax
  the search. `relaxed_topk` relaxes only the membership weights that a loss
  sees.
