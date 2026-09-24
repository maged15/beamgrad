# beamgrad

**Differentiable beam search for PyTorch.** Exact beam search forward,
surrogate gradients backward, native on CPU and CUDA.

[![CI](https://github.com/maged15/beamgrad/actions/workflows/ci.yml/badge.svg)](https://github.com/maged15/beamgrad/actions/workflows/ci.yml)
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

# log_probs[b, t, k] is the next-token distribution for beam k at step t.
log_probs = model(...).log_softmax(-1)                  # [B, T, K, V], CPU or CUDA

options = beamgrad.BeamOptions(beam_size=4, eos_token=2, length_penalty_alpha=0.6)
scores = beamgrad.final_scores(log_probs, options)     # [B, K], best beam first
loss = torch.relu(scores[:, 1] - scores[:, 0] + 1.0).mean()
loss.backward()                                         # gradients reach log_probs and the model

trace = beamgrad.decode(log_probs, options)             # tokens, parents, lengths, scores, ...
best = beamgrad.backtrack(trace)[:, 0]                  # [B, T] best sequence per example
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
- **Native everywhere.** CPU kernels are multi-threaded across the batch and
  pick AVX-512, AVX2, SSE4.2 or NEON at runtime. A CUDA engine runs forward
  and backward on the GPU for beams up to 1024, on PyTorch's stream and
  allocator. Every backend selects the same beams with the same scores, bit
  for bit.
- **A good PyTorch citizen.** The operators are registered with
  `torch.library`, with fake-tensor, autograd and vmap rules: `torch.compile`
  (even `fullgraph=True`), `torch.export`, `torch.vmap` and `torch.func`
  (`grad`, `vjp`, `jacrev`, per-example gradients) work.
- **A stable C ABI.** `libdbs` works from C, C++ or any FFI. It adds forced
  tokens and token-filter callbacks, fp16/bf16 input, incremental
  model-callback decoding (with each beam's parent, for KV-cache reordering),
  and two extra smooth surrogates: selected-beam softmax weights and a
  relaxed top-k pool.
- **Also in JAX**, through a custom VJP (`beamgrad.jax.final_scores`), with
  `jit`, `grad` and `vmap`.

## Installation

beamgrad compiles against your installed PyTorch:

```bash
pip install torch
pip install --no-build-isolation "git+https://github.com/maged15/beamgrad"
```

`--no-build-isolation` matters: the compiled operators only work with the
PyTorch they were built against, and `import beamgrad` says so (with the fix)
if the two differ. If a CUDA toolkit (`nvcc`) is available, the CUDA operators
are built automatically; `BEAMGRAD_CUDA=1` makes them required and
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
| [docs/algorithm.md](docs/algorithm.md) | what the forward and backward passes compute |
| [docs/python.md](docs/python.md) | Python API reference |
| [docs/c-api.md](docs/c-api.md) | C API reference, CUDA C API, ABI policy |
| [docs/cuda.md](docs/cuda.md) | CUDA engine design, limits, testing without a GPU |
| [docs/development.md](docs/development.md) | building, testing, benchmarking, releasing |
| [examples/](examples) | quickstart, training through beam search, C usage |

## Scope and limitations

- Gradients are **surrogate** gradients. They are exact for a fixed beam
  selection and do not model how the selection itself would change.
- `final_scores` consumes a precomputed `[T, K, V]` tensor. When each row
  depends on its beam's prefix, the model has to produce those rows during
  decoding: the C ABI's `dbs_decode_model_steps_ex` asks a callback for each
  step's rows and tells it which beam each row continues.
- CUDA supports beams up to 1024 and about 268M candidates (`K × V`) per step.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
`make test` and `make python-test` run the suites locally.

## Citing

If beamgrad helps your research, please cite it; [CITATION.cff](CITATION.cff)
has the details.

## License

[MIT](LICENSE) © Maged Amr
