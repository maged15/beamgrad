# Python API

```python
import beamgrad
```

All functions accept CPU or CUDA tensors of any floating dtype. float16 and
bfloat16 rows are read as they are (converted exactly, row by row, without a
float32 copy of the input); other dtypes are converted to float32. Computation
runs in float32 on the tensor's device; gradients are returned in the input's
dtype. There is no silent device fallback: CUDA tensors run the native CUDA
engine, forward and backward.

`final_scores`, `decode` and `search` also take logits (`from_logits=True`):
each row the search reads is normalised on the fly, `x - logsumexp(row)`, so
the search is the one over `log_softmax(logits)` without materialising a
`[B, T, K, V]` tensor of log-probabilities, and gradients flow through the
log-softmax. The logsumexp is the library's own (a fixed summation order and
`exp` polynomial), so CPU and CUDA agree bit for bit; it can differ from
`torch.logsumexp` in the last bits, which only matters for exact ties.

The work is done by operators registered with `torch.library`
(`torch.ops.beamgrad.decode_ex`, `decode_backward`, `decode_step`,
`final_scores_path_gradient` and `length_penalty`, and the float32-only
`decode` and `final_scores_backward` they extend)
with fake-tensor implementations, an autograd formula and a vmap rule. So
`final_scores`, `decode`, `search` and `sequence_scores` work inside
`torch.compile` (including `fullgraph=True`), with `torch.export` and
fake-tensor tracing, and under `torch.vmap` (PyTorch 2.5+). `final_scores` and
`decode` also work under the `torch.func` transforms (`grad`, `vjp`,
`jacrev`, and `vmap` of them, for example per-example gradients).

### torch.compile and `beam_search`

`beam_search` is different: it is a Python loop that calls your step function
once per step and stops as soon as every beam has finished. That stop is a
data-dependent condition, so `torch.compile(..., fullgraph=True)` cannot
capture the whole search as one graph (Dynamo reports that it "could not guard
on data-dependent expression"). Without `fullgraph` it falls back to graph
breaks. Compile the model inside the step function instead; that is where
nearly all of the time goes, and the search is unchanged:

```python
compiled = torch.compile(model)

def step(beams):
    return compiled(...).log_softmax(-1)      # compiled once, reused every step

result = beamgrad.beam_search(step, options, max_steps=T, batch_size=B)
```

## `BeamOptions`

```python
beamgrad.BeamOptions(
    beam_size: int,                    # K; must equal the beam dimension of log_probs
    eos_token: int | Sequence[int] = -1,  # EOS token id(s); -1 disables EOS handling
    min_length: int = 0,               # EOS is masked until a hypothesis has this many tokens
    length_penalty_alpha: float = 0.0, # GNMT length penalty exponent
    validate_inputs: bool = True,      # reject NaN / +inf in the rows the search reads
    banned_tokens: Sequence[int] | None = None,  # token ids never selected
    no_repeat_ngram_size: int = 0,     # n > 0 blocks repeating any n-gram of a beam's prefix
    repetition_penalty: float = 1.0,   # > 1 penalises tokens a beam already emitted
)
```

A frozen dataclass; invalid values raise `ValueError` on construction. NumPy
integer and float scalars are accepted, as are arrays or tensors of
`banned_tokens`; every field is stored as a plain Python `int`, `float` or
tuple of `int`s.

`eos_token` can list several tokens, as Hugging Face's `eos_token_id` does:
Qwen ends with `<|im_end|>` or `<|endoftext|>`, and Llama 3 has three. So
`BeamOptions(beam_size=4, eos_token=model.generation_config.eos_token_id)`
works as it is. Any of them finishes a hypothesis, `min_length` masks all of
them, and a finished beam is carried forward with the token it emitted.
`options.eos_tokens` lists them (`()` when EOS handling is off). CUDA supports
up to 16 besides the first.

`validate_inputs` raises `ValueError` if a row the search reads (the row of a
live, unfinished beam, including banned or masked tokens) contains NaN or
`+inf`. The check is part of the decode kernels, so it costs no extra pass
over the tensor; on CUDA it reads one flag per example back from the device,
which synchronizes the stream, so turn it off in hot loops once inputs are
known to be clean. `-inf` is always allowed and marks impossible tokens.

The constraints work the same on CPU, CUDA and JAX.
`repetition_penalty` subtracts `log(repetition_penalty)` from a repeated
token's log-probability; that shift is a constant, so gradients are unchanged.
Values in `(0, 1]` disable it: unlike `transformers`' `repetition_penalty`,
values below 1 do not favour repeated tokens.

## `final_scores(log_probs, options, steps=None, *, from_logits=False)`

Runs beam search and returns the final scores, best beam first, with
gradients.

| input | output |
|---|---|
| `[T, K, V]` | `[K]` |
| `[B, T, K, V]` | `[B, K]` |

`steps` (batched input only) is an optional `[B]` tensor or sequence giving
the number of steps to decode for each example, for variable-length batches;
each entry must be in `[1, T]` (checked by the operator, which raises
`ValueError`). `from_logits=True` takes logits instead of log-probabilities
(see above).

The backward pass returns the gradient of each final score along the path
that produced it; see [algorithm.md](algorithm.md#backward-surrogate-gradients).
From logits it continues through the log-softmax: a row whose path gradient
is `g` gets `g - softmax(row) * sum(g)`. Double backward is not supported.

```python
options = beamgrad.BeamOptions(beam_size=4, eos_token=2)
x = model(...).log_softmax(-1)                                   # [B, T, K, V], requires grad
scores = beamgrad.final_scores(x, options)                       # [B, K]
paths = beamgrad.backtrack(beamgrad.decode(x.detach(), options))  # [B, K, T]
# Structured margin: the reference must beat the best beam that is not the reference.
is_gold = (paths == gold[:, None]).all(-1)                       # gold: [B, T] reference tokens
rival = scores.masked_fill(is_gold, float("-inf")).amax(1)
loss = torch.relu(rival - gold_score + 1.0).mean()               # gold_score: [B], its own score
loss.backward()
```

With a model whose rows depend on the beams chosen so far, use
[`beam_search`](#beam_searchstep_fn-options-max_steps--batch_size1-devicenone---beamsearchresult)
instead.

## `decode(log_probs, options, steps=None, *, from_logits=False) -> BeamSearchOutput`

The same search, returning the whole trace as a named tuple.
The leading `[B]` dimension is absent for unbatched input.

When `log_probs` requires grad, the four score fields (`final_scores`,
`final_raw_scores`, `scores` and `raw_scores`) are differentiable: each
score's gradient flows along the path of tokens that produced it, holding the
selection fixed, as for `final_scores`. A loss can combine any of them, for
example per-step scores.

| field | shape | dtype | meaning |
|---|---|---|---|
| `final_scores` | `[B, K]` | float32 | length-penalised final scores, best first |
| `final_raw_scores` | `[B, K]` | float32 | cumulative log-probabilities |
| `final_lengths` | `[B, K]` | int64 | hypothesis lengths |
| `tokens` | `[B, T, K]` | int64 | token chosen at each step |
| `parents` | `[B, T, K]` | int64 | parent beam at the previous step |
| `lengths` | `[B, T, K]` | int64 | hypothesis length after each step |
| `scores` | `[B, T, K]` | float32 | ranking score after each step |
| `raw_scores` | `[B, T, K]` | float32 | cumulative log-probability after each step |
| `from_logprob` | `[B, T, K]` | bool | `False` for EOS carry-forward and padding slots |
| `steps` | `[B]` | int64 | number of decoded steps per example |

Slot `k` at step `t` is the `k`-th best hypothesis after step `t`. Steps past
an example's `steps` entry, and beams that never became live, hold token and
parent `-1`, length `0`, and `-inf` scores.

## `backtrack(output) -> Tensor`

Recovers each final beam's token sequence, `[B, K, T]` (or `[K, T]`), by
following parent pointers back from the last decoded step. A finished beam
repeats its EOS token after first producing it; padding steps and dead beams
hold `-1`.

```python
trace = beamgrad.decode(x, options)
best = beamgrad.backtrack(trace)[:, 0]    # [B, T] best hypothesis per example
```

## `beam_search(step_fn, options, max_steps, *, batch_size=1, device=None, rescore_fn=None, return_log_probs=None) -> BeamSearchResult`

Beam search over a model's next-token distributions, for autoregressive models
whose rows at step `t` depend on the beams chosen at step `t - 1`. Each step
calls `step_fn(beams)` with the current `BeamState` and expects `[B, K, V]`
log-probabilities (row `k` follows beam `k`'s hypothesis). The next beams are
chosen by the native engine on the rows' device, exactly as `decode` chooses
them from the stacked rows. The search stops after `max_steps` steps, or once
every beam has finished.

`BeamState` fields (`t` is `beams.step`):

| field | shape | meaning |
|---|---|---|
| `sequences` | `[B, K, t]` int64 | tokens of each hypothesis, `-1` past its length and for dead slots |
| `parents` | `[B, K]` int64 | slot at step `t - 1` that beam `k` continues (`-1` at step 0 and for dead slots) |
| `tokens` | `[B, K]` int64 | token beam `k` emitted at step `t - 1` (`-1` at step 0 and for dead slots) |
| `lengths` | `[B, K]` int64 | hypothesis lengths |
| `scores` | `[B, K]` float32 | length-penalised scores (no gradient) |
| `finished` | `[B, K]` bool | the hypothesis ended with EOS |
| `active` | `[B, K]` bool | live and unfinished: the rows the search reads |

Beams are re-ranked every step, so slot `k` usually continues a different
hypothesis than slot `k` did before: reorder any per-beam state (a
transformer's key/value cache, an RNN's hidden state) by `parents` (clamped at
0; dead slots' rows are never read). At step 0 all slots hold the empty
hypothesis and only slot 0 is read.

`BeamSearchResult` fields:

| field | shape | meaning |
|---|---|---|
| `scores` | `[B, K]` | final length-penalised scores, best first, **differentiable** |
| `sequences` | `[B, K, max_steps]` int64 | each final beam's tokens, `-1` past its length |
| `lengths`, `raw_scores` | `[B, K]` | hypothesis lengths and cumulative log-probabilities |
| `trace` | `BeamSearchOutput` | the per-step trace, as `decode(torch.stack(step_log_probs, 1), options)` returns it |
| `step_log_probs` | `T × [B, K, V]` | the tensors `step_fn` returned (not copied); empty when not kept |

`scores` equals `final_scores(torch.stack(step_log_probs, 1), options)`, with
the same gradient. Each beam is differentiated along its own path with the
selection held fixed, and the gradient reaches whatever the rows were
computed from. The rows are never stacked or copied. The engine's backward
gives each step only its `[B, K]` path gradient, which autograd scatters into
that step's rows when it reaches them, so no dense `[B, T, K, V]` gradient
exists. `return_log_probs` (by default: whether gradients are enabled)
controls whether the rows are kept in `step_log_probs`. Under
`torch.no_grad()` they are not, so each step's rows are freed once the next
step starts.

With `rescore_fn`, the search runs without an autograd graph (inference
memory), and the gradient comes from one teacher-forced pass instead:
`rescore_fn(sequences [B, K, T], lengths [B, K])` must return each final
beam's token log-probabilities, `[B, K, T]` (entries at or past a beam's
length are ignored). `scores` keep the search's values; their gradient is the
re-scoring's, which is the same function's gradient recomputed (no dropout),
and the pass can use activation checkpointing. See
[training.md](training.md#memory-three-ways-to-take-the-gradient) for
measurements and trade-offs.

```python
def step(beams):                                   # an RNN whose hidden states follow the beams
    global hidden
    if beams.step == 0:
        hidden = encoder(src).repeat_interleave(K, 0)            # [B*K, H]
    else:
        order = (torch.arange(B)[:, None] * K + beams.parents.clamp(min=0)).flatten()
        hidden = cell(embed(beams.tokens.clamp(min=0).flatten()), hidden[order])
    return head(hidden).log_softmax(-1).view(B, K, -1)

result = beamgrad.beam_search(step, options, max_steps=20, batch_size=B)
```

[`examples/train_lm.py`](../examples/train_lm.py) trains such a model so that
reference sequences win the search by a margin.

## `search(log_probs, options, steps=None, *, from_logits=False) -> BeamSearchResult`

`final_scores` and `decode` in one decode, returned as a `BeamSearchResult`,
for rows that are already a `[B, T, K, V]` (or `[T, K, V]`) tensor.
`scores` carry the path gradient, `sequences` are `-1`-padded after each
beam's length, and `step_log_probs` are views of the input's steps. The
losses and estimators below accept it like `beam_search`'s result.
With `from_logits=True`, `step_log_probs` is empty (the log-probabilities are
never materialised); the estimators, which read them, then need
`log_softmax(logits)` passed without `from_logits`.

## `sequence_scores(token_log_probs, lengths, options) -> Tensor`

Summed token log-probabilities (`[..., T]`, entries at or past `lengths`
ignored) divided by the search's length penalty. This puts a
teacher-forced reference sequence on the same scale as the final beam
scores. Differentiable.

`lengths` must be in `[0, T]`. A longer length would silently score a
truncated sequence, so it raises `ValueError`. The check reads the lengths
(one flag from the GPU) and is skipped while compiling or tracing and for
fake tensors.

## `length_penalty(lengths, alpha) -> Tensor`

`((5 + max(lengths, 1)) / 6) ** alpha` in float32, bit-identical to the
penalty the CPU and CUDA engines use.

## `beamgrad.losses`

| function | loss |
|---|---|
| `structured_margin(result, reference, reference_scores, margin=1.0, reduction="mean", *, eos_token=None)` | `max(0, margin + best non-reference beam's score − reference_scores)` with `reference_scores` of shape `[B]`; zero when the reference wins by the margin. References must end with the EOS the search emits (and `sequence_scores` lengths must count it), or they never match a beam; `eos_token=options.eos_token` warns about rows that do not |
| `minimum_risk(result, costs, temperature=1.0, reduction="mean")` | `Σ_k softmax(scores / temperature)_k · costs[:, k]`; dead beams get no probability and their costs are ignored (even NaN), examples without live beams give 0 |
| `matches(sequences, reference)` | `[B, K]` bool, beam `k` equals the `-1`-padded reference `[B, T']` |

## `beamgrad.estimators`

Smooth quantities of a search, computed from its trace and the rows it read
(`result.step_log_probs`): use `search()`, or `beam_search()` through the
steps. They match the C library's surrogates and run on the rows' device.

| function | returns |
|---|---|
| `path_scores(result, options)` | `[B, T, K]` every step's selected-beam scores (`trace.scores`), differentiable along each path |
| `selected_softmax(result, options, temperature=1.0)` | `[B, T, K]` `softmax(scores[t] / temperature)` over the selected beams; dead slots 0 |
| `relaxed_topk(result, options, pool_multiplier=8, temperature=0.25, tolerance=1e-4, max_iters=48)` | `RelaxedTopK(scores, weights, parents, tokens, from_logprob)`, each `[B, T, P]` with `P = K · pool_multiplier`: the best `P` candidates per step, their sigmoid k-hot membership (summing to `K`, gradient by implicit differentiation through the bisection threshold), and which candidate each is |

`relaxed_topk` gives gradient to candidates the search did not keep, so it
can train the model to keep a reference prefix in the beam. It does not
support `no_repeat_ngram_size` or `repetition_penalty`. The formulas are in
[training.md](training.md#gradient-estimators).

## `beamgrad.hf`

Adapters for Hugging Face `transformers` causal LMs (the module does not
import `transformers`):

- `CausalLMStep(model, input_ids, attention_mask, beam_size, *,
  share_prompt=True)`: a step function. It runs the left-padded prompts, then
  one token per beam per step, reordering the key/value cache by
  `beams.parents`. By default each prompt runs once and its cache is copied
  to the example's beam slots. `generate(num_beams=K)` runs every prompt `K`
  times, so on long prompts this is much cheaper: with 2,048-token prompts
  (Qwen2.5-0.5B, batch 8, 4 beams, 8 new tokens) a search takes 916 ms and
  3.4 GiB instead of `generate()`'s 2,759 ms and 7.8 GiB. It returns the same
  beams as `generate()`, with float32 scores within about 1e-4 (the prompt
  runs at another batch size, which rounds differently). `share_prompt=False`
  runs the prompts once per beam as `generate()` does, and then the scores are
  bit-identical too. Each search restarts from the prompts, so one instance
  can drive several searches.
- `CausalLMRescorer(model, input_ids, attention_mask, chunk_size=256,
  gradient_checkpointing=False)`: a `rescore_fn`. It runs one teacher-forced
  pass over prompt + beam and projects to the vocabulary in checkpointed
  chunks. With `gradient_checkpointing=True` it enables the model's
  checkpointing and training mode for its own pass only. Run the search in
  eval mode, because `transformers` disables the cache for checkpointing
  models in training mode. Full fine-tuning of a 0.5B model through the
  search fits in 16 GB only with it: see
  [training.md](training.md#memory-three-ways-to-take-the-gradient).

Pass `eos_token=model.generation_config.eos_token_id` to `BeamOptions`: it is
a list for models with several EOS tokens (Qwen, Llama 3). When comparing with
`generate()`, note that instruct models' generation configs also set sampling
options and a repetition penalty, which `generate()` applies even to beam
search; pass `repetition_penalty=1.0` to it for the plain search.

## `cuda_available() -> bool`

`True` when beamgrad was built with its CUDA operators and PyTorch sees a CUDA
device. CUDA supports beam sizes up to `beamgrad.CUDA_MAX_BEAM` (1024); the
CPU backend has no fixed limit.

## Installation and the PyTorch version

The compiled operators only work with the PyTorch they were built against.
Prebuilt wheels exist for some combinations (see
[installation.md](installation.md)). Otherwise, build without isolation so
they are compiled against the torch you use:

```bash
pip install --no-build-isolation beamgrad
```

A plain `pip install beamgrad` builds in an isolated environment with the
newest torch; if that differs from the installed one, `import beamgrad` raises
an `ImportError` naming both versions and the command to rebuild, instead of
failing later with undefined symbols.

## JAX

```python
import beamgrad.jax
scores = beamgrad.jax.final_scores(log_probs, beamgrad.BeamOptions(beam_size=4), steps=None)
scores = beamgrad.jax.final_scores(logits, beamgrad.BeamOptions(beam_size=4), from_logits=True)
```

Same values, options and gradients as the PyTorch function (including
`from_logits`), via a custom VJP.
Decoding runs on the host through the bundled C library (`jax.pure_callback`):
the whole batch is decoded in one multi-threaded call, and the forward pass
keeps the decode trace so the backward pass does not decode again. It composes
with `jit`, `grad` and `vmap` on any backend but copies data to the host.
With `validate_inputs`, NaN or `+inf` fails the call (under `jit`, JAX reports
the host callback's error). Requires the `jax` extra
(`pip install "beamgrad[jax]"`).

## The C library from Python

The wheel bundles the C library as `beamgrad._libdbs`.
`beamgrad._ctypes.load_library()` returns it as a `ctypes.CDLL` with the
common signatures bound, for the parts of the [C API](c-api.md) that the
PyTorch functions do not expose (selected-beam and relaxed-pool gradients,
forced tokens and token filters, model-step decoding, typed fp16/bf16 input).
Set `DBS_LIBRARY` to load a different build.
