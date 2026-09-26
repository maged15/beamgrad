# beamgrad

**Differentiable beam search for PyTorch.** Exact beam search forward,
surrogate gradients backward, native on CPU and CUDA.

[![CI](https://github.com/maged15/beamgrad/actions/workflows/ci.yml/badge.svg)](https://github.com/maged15/beamgrad/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/maged15/beamgrad)](https://github.com/maged15/beamgrad/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)

Beam search is how sequence models decode, but it is discrete: top-k selection
has no useful gradient. So models are usually trained with teacher forcing,
then decoded with a search they never saw during training. beamgrad puts the
search inside the computation graph. It runs exact, deterministic beam search,
then backpropagates through the hypotheses it selected, so you can write losses
on what the decoder actually produces.

```python
import torch
import beamgrad

# An autoregressive model: next-token log-probabilities for each beam's prefix.
def step(beams):                                                # beams.sequences: [B, K, t] tokens so far
    return model(src, beams.sequences).log_softmax(-1)         # [B, K, V]

options = beamgrad.BeamOptions(beam_size=4, eos_token=EOS)
result = beamgrad.beam_search(step, options, max_steps=T, batch_size=B)
# result.scores: [B, K], best first, differentiable w.r.t. the model
# result.sequences: [B, K, T] tokens of each beam, -1 after it ends

# Minimum-risk training: move probability toward the beams with the lowest cost.
costs = 1 - sentence_bleu(result.sequences, references)       # [B, K], any per-beam cost, no gradient needed
loss = beamgrad.losses.minimum_risk(result, costs)
loss.backward()                                                 # through the search, into the model
```

The loss is defined on what beam search actually returns. It is the expected
cost of the beams under a softmax over their scores, so its gradient raises
the scores of good beams and lowers those of bad ones.
[`experiments/multi30k`](experiments/multi30k) is a runnable version for
translation, with sentence BLEU as the cost and a transformer whose
key/value cache follows the beams (`beams.parents` reorders it). It also
compares the alternatives on the same task: a structured margin against
the reference (`losses.structured_margin`, runnable in
[`examples/train_lm.py`](examples/train_lm.py)) and the relaxed top-k
estimator. There, minimum-risk training beat continued MLE on 7 of 8 seeds
(+0.39 BLEU, paired t-test p = 0.009) and the margin against the reference
did not help. [docs/training.md](docs/training.md) explains which loss and
which gradient mode (through the steps, or re-scoring for large models) to
use.

If the next-token distributions of every beam are already in a
`[B, T, K, V]` tensor, score and decode it directly:

```python
scores = beamgrad.final_scores(log_probs, options)             # [B, K], differentiable
best = beamgrad.backtrack(beamgrad.decode(log_probs, options))[:, 0]   # [B, T] best sequence
```

## Features

- **Exact beam search.** GNMT length penalty, EOS handling (finished beams are
  carried forward and keep competing), minimum length, variable-length
  batches, banned tokens, n-gram blocking and a repetition penalty. A strict
  total order on candidates makes results deterministic.
- **Gradients through the search.** Each final score is differentiated along
  the path that produced it (see [how it works](docs/algorithm.md)). The
  gradients match the C reference bit for bit and agree with finite
  differences.
- **Trains in bounded memory.** The backward pass hands each step only its
  own path gradient, so no dense `[B, T, K, V]` gradient is ever built. With
  `rescore_fn`, the search runs with inference memory and the gradient comes
  from one teacher-forced pass, which can use activation checkpointing. A
  Qwen2.5-0.5B training step at 8 beams × 64 steps × batch 8 drops from out
  of memory on 16 GB to 1.3 GiB, with the same gradient
  ([docs/training.md](docs/training.md)).
- **Losses and estimators.** `beamgrad.losses` has a structured margin and
  minimum-risk training. `beamgrad.estimators` has smoother surrogates:
  softmax over the selected beams, and a relaxed top-k that also sends
  gradient to candidates the search pruned.
- **Drives real models.** `beam_search` runs an autoregressive model inside
  the search, one step at a time, reordering its cache by each beam's parent.
  On Qwen2.5-0.5B and Qwen3-0.6B, with EOS suppressed and
  `length_penalty=0`, it returns all `K` beams of `transformers`'
  `generate(num_beams=K)` with bit-identical scores in float32, at the same
  speed. With EOS enabled the two differ by design: beamgrad keeps finished
  hypotheses in their beam slots, and `transformers` keeps them in a separate
  pool. So there only the best beam is compared
  ([benchmarks/hf_beam_search.py](benchmarks/hf_beam_search.py),
  [docs/benchmarks.md](docs/benchmarks.md)). It also fine-tunes the model
  through the search.
- **Native everywhere.** CPU kernels are multi-threaded across the batch and
  pick AVX-512, AVX2, SSE4.2 or NEON at runtime. A CUDA engine runs forward
  and backward on the GPU for beams up to 1024, on PyTorch's stream and
  allocator. Every backend selects the same beams with the same scores, bit
  for bit.
- **A good PyTorch citizen.** The operators are registered with
  `torch.library`, with fake-tensor, autograd and vmap rules. So
  `final_scores`, `decode` and `search` work under `torch.compile` (even
  `fullgraph=True`), `torch.export`, `torch.vmap` and `torch.func` (`grad`,
  `vjp`, `jacrev`, per-example gradients). `beam_search` is a Python loop
  that stops once every beam has finished, so it is not captured as one
  graph: compile the model it calls instead
  ([details](docs/python.md#torchcompile-and-beam_search)).
- **A stable C ABI.** `libdbs` works from C, C++ or any FFI. It adds forced
  tokens and token-filter callbacks, fp16/bf16 input, incremental
  model-callback decoding (with each beam's parent, for KV-cache reordering),
  and two extra smooth surrogates: selected-beam softmax weights and a
  relaxed top-k pool.
- **Also in JAX**, through a custom VJP (`beamgrad.jax.final_scores`), with
  `jit`, `grad` and `vmap`.

## Installation

Releases include prebuilt wheels for PyTorch 2.13 and 2.14: Linux (CPU,
CUDA 12.6, CUDA 13.0), macOS and Windows, each for every Python from 3.10.
With PyTorch installed, this picks the wheel that matches it:

```bash
pip install beamgrad -f "https://maged15.github.io/beamgrad/whl/$(python -c "import torch; v = torch.__version__.split('+')[0].split('.'); c = torch.version.cuda; print(f'pt{v[0]}{v[1]}' + ('cu' + c.replace('.', '') if c else 'cpu'))").html"
```

For other PyTorch versions, beamgrad compiles against your installed PyTorch:

```bash
pip install torch "setuptools>=77" wheel "packaging>=24.2"
pip install --no-build-isolation beamgrad
```

[docs/installation.md](docs/installation.md) has the compatibility matrix and
the details.

`--no-build-isolation` matters: the compiled operators only work with the
PyTorch they were built against, and if the two differ, `import beamgrad`
says so and gives the fix. If a CUDA toolkit (`nvcc`) is available, the CUDA
operators are built automatically; `BEAMGRAD_CUDA=1` makes them required and
`BEAMGRAD_CUDA=0` skips them. `beamgrad.cuda_available()` reports what you
got. For the C library alone, use CMake (see [the C API](docs/c-api.md)).

## How it works

At each step, every live beam proposes every token. Its cumulative
log-probability is ranked by `raw / ((5 + length) / 6) ** alpha`, and the `K`
best candidates survive. Beams that emitted EOS are carried forward
unchanged. Backward holds this selection fixed. The gradient of final beam
`k`'s score with respect to `log_probs[t, p, v]` is `1 / penalty(length_k)`
for every `(t, p, v)` on its path, and zero elsewhere. That is the exact
derivative wherever a small perturbation would not change the selection.
[docs/algorithm.md](docs/algorithm.md) gives the full definitions, including
the additional C-level surrogates.
[What beamgrad is and isn't](docs/algorithm.md#what-beamgrad-is-and-isnt)
explains two things. The default gradient is the one teacher-forced
re-scoring of the beams gives, so what is new is the engineering. And it
covers how beamgrad relates to beam-search optimisation, minimum risk
training and continuous relaxations of beam search.

## Performance

`python benchmarks/benchmark.py` times the PyTorch API against a beam search
written with `torch.topk`, and checks that the scores agree. CPU results on a
4-core Intel Xeon (2.8 GHz, AVX-512) container, median milliseconds:

| B × T × K × V | forward | forward + backward | `torch.topk` beam search (forward) |
|---|--:|--:|--:|
| 1 × 16 × 4 × 32k | 0.52 | 1.12 | 13.2 |
| 8 × 16 × 4 × 32k | 7.3 | 28.3 | 25.0 |
| 8 × 32 × 8 × 32k | 28.0 | 108 | 160 |
| 4 × 16 × 8 × 128k | 26.5 | 101 | 230 |
| 16 × 64 × 4 × 50k | 83.4 | 293 | 259 |

Backward time is dominated by writing the dense `[B, T, K, V]` gradient that
autograd expects, so it is memory-bound. Run the script on your own hardware,
including GPUs, before relying on these numbers; [docs/cuda.md](docs/cuda.md)
describes the CUDA engine.

## C and C++

```c
#include "dbs.h"

DBSOptionsC opt = {0};              /* zero fields select defaults */
opt.beam_size = 4;
opt.eos_token = -1;                 /* no EOS */

DBSDecoderHandle* decoder = NULL;
dbs_create_ex(opt, &decoder);

DBSResultHandle* result = NULL;
dbs_decode(decoder, log_probs, T, V, &result);           /* log_probs: [T, K, V] */
const float* scores = dbs_result_final_scores(result);   /* [K] */

float grad_final[4] = {1, 0, 0, 0};
DBSBackwardHandle* grad = NULL;
dbs_backward(decoder, result, NULL, NULL, grad_final, &grad);  /* sparse d scores[0] / d log_probs */

dbs_free_backward(grad);
dbs_free_result(result);
dbs_destroy(decoder);
```

```cmake
find_package(beamgrad 2 REQUIRED)
target_link_libraries(app PRIVATE beamgrad::dbs)   # or beamgrad::dbs_cuda
```

The complete version, with error handling, is
[`examples/c_api.c`](examples/c_api.c). The test suite builds and runs it.

## Documentation

| | |
|---|---|
| [docs/installation.md](docs/installation.md) | wheels, compatibility matrix, building from source |
| [docs/algorithm.md](docs/algorithm.md) | what the forward and backward passes compute |
| [docs/training.md](docs/training.md) | training through the search: gradient, memory modes, losses, estimators, experiment |
| [docs/python.md](docs/python.md) | Python API reference |
| [docs/c-api.md](docs/c-api.md) | C API reference, CUDA C API, ABI policy |
| [docs/cuda.md](docs/cuda.md) | CUDA engine design, limits, testing without a GPU |
| [docs/benchmarks.md](docs/benchmarks.md) | what each benchmark measures (kernel or end to end) |
| [docs/development.md](docs/development.md) | building, testing, releasing |
| [examples/](examples) | quickstart, training a model through beam search, C usage |
| [experiments/multi30k](experiments/multi30k) | a controlled training experiment (En→De translation) |

## Scope and limitations

- Gradients are **surrogate** gradients. They are exact for a fixed beam
  selection and do not model how the selection itself would change; only
  `estimators.relaxed_topk` relaxes the selection
  ([what beamgrad is and isn't](docs/algorithm.md#what-beamgrad-is-and-isnt)).
- Through the steps (the default), the model's graph for every step is kept
  until the backward pass, as with any backpropagation through generation.
  `rescore_fn` avoids this at the cost of one teacher-forced pass, and the
  estimators, which need the rows, are only available through the steps.
- CUDA supports beams up to 1024 and about 268M candidates (`K × V`) per step.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
`make test` and `make python-test` run the suites locally.

## Citing

If beamgrad helps your research, please cite it; [CITATION.cff](CITATION.cff)
has the details.

## License

[MIT](LICENSE) © Maged Amr
