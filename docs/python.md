# Python API

```python
import beamgrad
```

All functions accept CPU or CUDA tensors of any floating dtype. Computation
runs in float32 on the tensor's device; gradients are returned in the input's
dtype. There is no silent device fallback: CUDA tensors run the native CUDA
engine, forward and backward.

The work is done by operators registered with `torch.library`
(`torch.ops.beamgrad.decode`, `decode_step` and `final_scores_backward`)
with fake-tensor implementations, an autograd formula and a vmap rule, so
beamgrad works inside `torch.compile` (including `fullgraph=True`), with
`torch.export` and fake-tensor tracing, and under `torch.vmap` (PyTorch 2.5+).
`final_scores` also works under the `torch.func` transforms (`grad`, `vjp`,
`jacrev`, and `vmap` of them, for example per-example gradients).

## `BeamOptions`

```python
beamgrad.BeamOptions(
    beam_size: int,                    # K; must equal the beam dimension of log_probs
    eos_token: int = -1,               # -1 disables EOS handling
    min_length: int = 0,               # EOS is masked until a hypothesis has this many tokens
    length_penalty_alpha: float = 0.0, # GNMT length penalty exponent
    validate_inputs: bool = True,      # reject NaN / +inf in the rows the search reads
    banned_tokens: Sequence[int] | None = None,  # token ids never selected
    no_repeat_ngram_size: int = 0,     # n > 0 blocks repeating any n-gram of a beam's prefix
    repetition_penalty: float = 1.0,   # > 1 penalises tokens a beam already emitted
)
```

A frozen dataclass; invalid values raise `ValueError` on construction.

`validate_inputs` raises `ValueError` if a row the search reads (the row of a
live, unfinished beam, including banned or masked tokens) contains NaN or
`+inf`. The check is part of the decode kernels, so it costs no extra pass
over the tensor; on CUDA it reads one flag per example back from the device,
which synchronizes the stream, so turn it off in hot loops once inputs are
known to be clean. `-inf` is always allowed and marks impossible tokens.

The constraints work the same on CPU, CUDA and JAX.
`repetition_penalty` subtracts `log(repetition_penalty)` from a repeated
token's log-probability; that shift is a constant, so gradients are unchanged.

## `final_scores(log_probs, options, steps=None)`

Runs beam search and returns the final scores, best beam first, with
gradients.

| input | output |
|---|---|
| `[T, K, V]` | `[K]` |
| `[B, T, K, V]` | `[B, K]` |

`steps` (batched input only) is an optional `[B]` tensor or sequence giving
the number of steps to decode for each example, for variable-length batches;
each entry must be in `[1, T]` (checked by the operator, which raises
`ValueError`).

The backward pass returns the gradient of each final score along the path
that produced it; see [algorithm.md](algorithm.md#backward-surrogate-gradients).
Double backward is not supported.

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

## `decode(log_probs, options, steps=None) -> BeamSearchOutput`

The same search without gradients, returning the whole trace as a named tuple.
The leading `[B]` dimension is absent for unbatched input.

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

## `beam_search(step_fn, options, max_steps, *, batch_size=1, device=None) -> BeamSearchResult`

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
| `step_log_probs` | `T × [B, K, V]` | the tensors `step_fn` returned (not copied) |

`scores` equals `final_scores(torch.stack(step_log_probs, 1), options)`, with
the same gradient: each beam is differentiated along its own path with the
selection held fixed, and the gradient reaches whatever the rows were computed
from. The rows are never stacked or copied, so with a large vocabulary the
search itself adds only `[B, K]` state per step; under `torch.no_grad()` a
step's rows can be freed once the next step starts.

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

## `cuda_available() -> bool`

`True` when beamgrad was built with its CUDA operators and PyTorch sees a CUDA
device. CUDA supports beam sizes up to `beamgrad.CUDA_MAX_BEAM` (1024); the
CPU backend has no fixed limit.

## Installation and the PyTorch version

The compiled operators only work with the PyTorch they were built against.
Build without isolation so they are compiled against the torch you use:

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
```

Same values, options and gradients as the PyTorch function, via a custom VJP.
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
