# Changelog

## 2.1.1 (2026-09-26)

Closes the gaps left by 2.1.0 and its review:
- the CUDA step's precondition is now checked in the PyTorch operator;
- `length_penalty` has a vmap rule;
- `relaxed_topk` is 14× faster on the GPU;
- the `torch.compile` documentation is accurate;
- wheels install with one command;
- a report of the device tests on a real GPU is published;
- the translation experiment has 8 seeds.

The C ABI is unchanged (version 10).

### Added

- Wheel index pages on GitHub Pages, one per PyTorch minor version and CUDA
  variant: `pip install beamgrad -f
  https://maged15.github.io/beamgrad/whl/pt214cu126.html` installs the
  matching wheel. `docs/installation.md` has a one-line command that picks
  the page for the installed PyTorch. The pages are rebuilt after every
  release (`.github/workflows/wheel-index.yml`, `scripts/wheel_index.py`);
  the wheels stay on the GitHub releases.
- `scripts/gpu_report.py` runs the device tests (C/C++ with the native CUDA
  backend, and the Python suite) on a local GPU and writes a report.
  `docs/gpu-report.md` is the report for 2.1.0. GitHub-hosted CI has no
  GPU; `docs/cuda.md` now separates what CI checks (compilation, emulation)
  from what needs a device.

### Changed

- `estimators.relaxed_topk` runs one bisection for all steps instead of one
  per step. Each iteration reads a convergence flag back from the device, so
  on CUDA this removed about T times as many synchronizations. With B = 32,
  T = 40, K = 4, V = 8000, forward + backward went from 382 ms to 27 ms on an
  RTX 4080 SUPER, and from 136 ms to 106 ms on the CPU. The weights,
  candidates and gradients are bit for bit the same.
- `length_penalty` has a vmap batching rule, so `torch.vmap` of
  `sequence_scores` no longer falls back to a slower loop with a warning.

### Fixed

- `final_scores` and `search` on tensors that do not require grad could not
  be compiled with `torch.compile(fullgraph=True)` on PyTorch 2.4. Its
  Dynamo mis-traces an `autograd.Function` with a separate `setup_context`
  when no input requires grad. They now call the decode operator directly
  when there is nothing to differentiate, as `decode` does; the values are
  the same.
- On CUDA, `torch.ops.beamgrad.decode_step` (and so `beam_search`) silently
  mis-ranked a hand-built state whose live, unfinished beams had different
  lengths when there was a length penalty (the documented precondition of
  `dbs_cuda_decode_step`). With `validate_inputs` on, the default, it now
  raises `ValueError`. The check shares the transfer that already reads the
  NaN flags, so it adds no synchronization. States the search produces were
  never affected.

### Documentation

- `torch.compile`: the operators, `final_scores`, `decode`, `search` and
  `sequence_scores` compile with `fullgraph=True`. `beam_search` is a Python
  loop with a data-dependent stop, so it is not captured as one graph:
  compile the model it calls. Both are now tested.
- The README leads with minimum-risk training, the loss that helped in the
  translation experiment, and links to the guide on which loss and gradient
  mode to use.
- Multi30k experiment: MLE, MRT and MRT through re-scoring now have 8 seeds,
  and the summary reports wins, a sign test and a paired t-test. MRT beat
  continued MLE on 7 of 8 seeds (+0.39 BLEU, t-test p = 0.009; the first 3
  seeds had suggested +0.55 at p ≈ 0.1). Through re-scoring, with dropout
  on, it gained only +0.11 (p = 0.33). The likely cause is that re-scoring
  draws new dropout masks, and `docs/training.md` now says so.

## 2.1.0 (2026-09-26)

Fixes and hardening on top of 2.0.0: an accurate relaxed top-k at any score
magnitude, stricter validation in the Python API, broader fuzzing, and
faster model-step decoding and small batch decodes. The C ABI is unchanged
(version 10).

### Added

- `losses.structured_margin(..., eos_token=)`: when given (and `>= 0`),
  warns with the row numbers of references that do not end with that EOS.
  A finished beam ends with the EOS it emitted, so such a reference never
  matches a beam. Its own copy among the beams then becomes the rival, and
  the loss can never reach zero. The requirement (references and
  `sequence_scores` lengths include the EOS) is now documented. The default
  (`None`) behaves as before.

### Changed

- Model-step decoding (`dbs_decode_model_steps`, `_ex`, `_with_workspace`)
  builds each step's `[K, t]` prefix matrix from the previous step's. It
  copies each beam's parent row and appends one token, where it used to walk
  every path back through the trace. The prefixes are byte for byte the
  same. Best of 30 runs on this machine (`dbs_bench model-steps`, V = 32):
  T = 1024 went from 3.6 ms to 0.55 ms at K = 4, and from 15.1 ms to 2.9 ms
  at K = 16; T = 256, K = 4 went from 268 µs to 88 µs.
- Batch functions with `num_threads <= 0` (automatic) run a batch under 2^19
  candidates (B·T·K·V; B·T·K for `dbs_backward_batch_into`) on the calling
  thread instead of starting a thread per core. For such batches that took
  longer than the work itself. An 8 × 8 × 4 × 1000 decode went from 95 µs
  to 51 µs, a 4 × 4 × 4 × 256 decode from 35 µs to 6 µs; batches of 1M
  candidates and more are unchanged (`dbs_bench batch-threads`). An
  explicit thread count is still honoured, capped at the batch size.
- The libFuzzer harness covers `dbs_decode_constrained_ex`, the relaxed pool
  with both backward passes, `dbs_decode_typed` (F16, BF16),
  `dbs_decode_batch_into` with `dbs_backward_batch_into` on corrupted traces,
  and `dbs_decode_batch_variable`. It checks results as well as crashes:
  status codes, sparse vs dense gradients, typed vs float32 input, and
  rejection of out-of-range traces. `dbs_fuzz` now compiles its own
  coverage-instrumented copy of the library. Before, only the harness was
  instrumented, so libFuzzer got no coverage feedback from the library.

### Documentation

- The README's `generate()` parity claim is qualified as the benchmark
  measures it. All `K` beams are bit-identical in float32 with EOS
  suppressed and `length_penalty=0`. With EOS the two searches differ by
  design (finished hypotheses stay in beamgrad's slots but go to a separate
  pool in `transformers`), and only the best beam is compared.
- `docs/algorithm.md` has a "What beamgrad is and isn't" section, linked
  from the README. It says:
  - the default gradient is that of teacher-forced re-scoring of the
    selected beams;
  - only `relaxed_topk` relaxes the selection;
  - how beamgrad relates to beam-search optimisation, minimum risk training
    and continuous relaxations of beam search.
- CONTRIBUTING.md, `docs/development.md` and `docs/installation.md`: builds
  with `--no-build-isolation` need `setuptools>=77`, `wheel` and
  `packaging>=24.2` installed first.
- `dbs_cuda_decode_step` (`include/dbs_cuda.h`, `docs/cuda.md`) requires every
  live, unfinished beam of an example to have the same length: the CUDA scan
  ranks by raw score. States produced by the search always satisfy this. A
  hand-built state that does not, with a length penalty, can select
  different beams than the CPU. The precondition is documented, not checked,
  since a check would synchronize the stream on every step.

### Fixed

- `beam_search` rejected NumPy integers for `max_steps` and `batch_size`,
  although `BeamOptions` accepts them. Any integer but `bool` is now
  accepted.
- `sequence_scores` accepted lengths past the sequences' last dimension, and
  negative lengths, silently scoring a truncated sequence. It now raises
  `ValueError`. The check is skipped while compiling or tracing and for fake
  tensors, so `sequence_scores` stays traceable.
- The relaxed top-k pool's weights did not sum to `K` when the scores were
  large (a long search's cumulative log-probabilities). The bisection
  stopped once `theta` was known to `soft_topk_tolerance` of its own
  magnitude. For 32 candidates 0.1 apart with `K = 4` and temperature 0.25,
  the weights summed to 4.29 near −1,000 and 4.64 near −5,000; a 64-step
  Python `relaxed_topk` search near −1,500 summed to 1.88 instead of 2. The
  bisection now stops when `theta` is bracketed to `tolerance × temperature`,
  and it bisects `theta` as an offset from the best score, so the sums stay
  within about 1e-4 of `K` at any magnitude. This applies to the C library
  (`relaxed_pool_multiplier`) and to `beamgrad.estimators.relaxed_topk`
  alike. The default of 48 iterations is unchanged; about 21 are needed.

## 2.0.0 (2026-09-25)

The project is renamed **beamgrad** (previously `differentiable-beam-search-cuda`
and the `dbs-torch` Python package). The C library keeps its name, `libdbs`,
and C ABI version 10. Every symbol of the 0.5 baseline is still exported;
three functions are added and one no-op stub is removed (see below).

### Highlights

- **Training a model through beam search.** `beamgrad.beam_search(step_fn,
  options, max_steps)` runs an autoregressive model inside the search: each
  step asks the model for the next-token distributions of the beams chosen so
  far (with each beam's parent slot, to reorder a key/value cache), and the
  returned scores are differentiable with respect to the model. Driving
  Qwen2.5-0.5B and Qwen3-0.6B, it returns the same beams with bit-identical
  scores as `transformers`' `generate(num_beams=...)` in float32, at the same
  speed (`benchmarks/hf_beam_search.py`).
- **Training in bounded memory.** `beam_search`'s backward hands each step
  only its own `[B, K]` path gradient, never a dense `[B, T, K, V]` one, and
  `rescore_fn` takes the same gradient from one teacher-forced pass after a
  search without an autograd graph. With activation checkpointing, a
  Qwen2.5-0.5B training step at batch 8, 8 beams and 128 steps needs 1.7 GiB
  above the weights instead of running out of memory on 16 GB
  (`benchmarks/hf_training_memory.py`).
- **Losses, estimators and a controlled experiment.** `beamgrad.losses`
  (structured margin, minimum risk) and `beamgrad.estimators`
  (selected-beam softmax, relaxed top-k) in PyTorch, and a Multi30k
  translation experiment comparing them with continued MLE over three seeds
  (`experiments/multi30k`): minimum-risk training on the search's beams
  improved test BLEU on every seed (+0.55 on average), and a margin against
  the reference made it worse.
- **Wheels for every Python.** The extensions use only CPython's limited API,
  so one wheel per PyTorch version and CUDA variant serves Python 3.10+.
  Releases build them for PyTorch 2.13 and 2.14 on Linux (CPU, CUDA 12.6,
  CUDA 13.0), macOS and Windows.
- **Native CUDA engine.** Beam search and its backward pass run entirely on
  the GPU for beams up to 1024, with GNMT length penalty, EOS carry-forward,
  `min_length` and variable-length batches. It selects the same beams, with
  the same scores, as the CPU decoder. The PyTorch API no longer falls back
  to the CPU for any option or beam size.
- **`beamgrad` Python package** with one device-agnostic API:
  `final_scores` (autograd), `decode` (full trace, now on CPU too),
  `backtrack`, and `steps=` for variable-length batches.
- **Tests that test.** The C++ tests used `assert()`, which Release builds
  compile out, so CI had been running empty test bodies. They now always run.
  The CUDA kernels are verified without a GPU by running the unmodified kernel
  source through a CPU emulation layer, checked bit for bit against the CPU
  decoder under several thread schedules.
- **Constraints everywhere.** Banned tokens, n-gram blocking and a repetition
  penalty are available from Python on CPU, CUDA and JAX. Their per-beam token
  sets are computed once per step instead of scanning the prefix for every
  candidate (n-gram blocking at T=32, K=4, V=32k: 57 ms to 0.7 ms on CPU).
- **PyTorch integration.** The operators are registered with `torch.library`
  (fake tensors, autograd, vmap), so `torch.compile`, `torch.export` and
  `torch.vmap` work, and `final_scores` works under the `torch.func`
  transforms (`grad`, `vjp`, `jacrev`, per-example gradients). Input
  validation runs inside the decode kernels.

### Added

- `beam_search(..., rescore_fn=, return_log_probs=)`: gradients by
  teacher-forced re-scoring of the final beams, and control over keeping
  the rows. `beamgrad.search` (scores, sequences and trace from one decode),
  `sequence_scores`, `length_penalty`, and `BeamSearchResult.trace`.
- `beamgrad.losses`: `structured_margin`, `minimum_risk`, `matches`.
- `beamgrad.estimators`: `path_scores`, `selected_softmax` and
  `relaxed_topk` (with each pool candidate's parent, token and origin),
  matching the C library's surrogates on CPU and CUDA.
- `beamgrad.hf`: `CausalLMStep` (a step function for Hugging Face causal LMs,
  with a key/value cache that follows the beams) and `CausalLMRescorer`
  (chunked, checkpointed vocabulary projection; optional gradient
  checkpointing).
- Operators `final_scores_path_gradient` and `length_penalty` (CPU and
  CUDA), and their C/CUDA counterparts `dbs_cuda_path_gradient` and
  `dbs_cuda_length_penalty`.
- `experiments/multi30k`, `benchmarks/hf_training_memory.py`, and the
  guides `docs/training.md`, `docs/installation.md` and `docs/benchmarks.md`.
- Wheel builds (`.github/workflows/wheels.yml`), attached to GitHub releases;
  `BEAMGRAD_PIN_TORCH` and `BEAMGRAD_LOCAL_VERSION` for building them.
- CI builds the CUDA operators with CUDA 12.4 (with PyTorch 2.4), 12.6 and
  13.0 (with Blackwell architectures), and tests transformers with a
  key/value cache, 384-step searches and autocast (bfloat16, float16).
- `beamgrad.beam_search`, `BeamState`, `BeamSearchResult`: beam search that
  drives a model step by step, on CPU or CUDA, without stacking or copying the
  per-step rows. `torch.ops.beamgrad.decode_step` (one step from an explicit
  beam state), the C++ `BeamSearchDecoder::step`, and the CUDA C API's
  `dbs_cuda_decode_step` / `DBSCudaBeamState` /
  `dbs_cuda_decode_step_workspace_size` underneath; stepping through the rows
  of a tensor selects exactly what the full decode selects.
- `examples/train_lm.py`: a GRU language model trained so that reference
  sequences win beam search by a margin, and
  `benchmarks/hf_beam_search.py`: parity, speed and training on a Hugging
  Face causal LM against `generate`.
- `dbs_cuda_decode` / `dbs_cuda_backward` with workspace-size queries,
  per-example metadata, constraints, NaN/`+inf` flags, optional trace outputs,
  and asynchronous execution (`DBS_CUDA_SYNC_CHECK=1` synchronizes for
  debugging).
- `dbs_decode_model_steps_ex`: incremental model-step decoding whose callback
  learns each beam's parent slot, length, finished flag and token prefix (so a
  model can reorder per-beam state), with optional constraints. The original
  `dbs_decode_model_steps` also runs incrementally now (`T` callbacks instead
  of re-decoding every prefix).
- `dbs_decode_batch_into` / `dbs_backward_batch_into`: batch decoding and the
  final-score backward on caller-owned arrays, with per-example steps and
  constraints, used by the PyTorch CPU operators and by JAX.
- `DBS_OK` / `DBS_ERROR_*` status constants.
- `BeamOptions.banned_tokens`, `no_repeat_ngram_size`, `repetition_penalty`;
  `steps=` and `vmap` in `beamgrad.jax.final_scores`.
- An import-time check that beamgrad was compiled against the installed
  PyTorch, with the command that fixes a mismatch.
- `beamgrad.BeamOptions`, `final_scores`, `decode`, `backtrack`,
  `BeamSearchOutput`, `cuda_available`, `CUDA_MAX_BEAM`;
  `beamgrad.jax.final_scores`.
- The wheel bundles libdbs as `beamgrad._libdbs` for ctypes and JAX use.
- CMake package `find_package(beamgrad)` with targets `beamgrad::dbs` and
  `beamgrad::dbs_cuda`, a `DBS_WARNINGS_AS_ERRORS` option, and ctest targets
  for the ABI symbol check, a pure-C header test and the C example.
- CI across Linux (GCC, Clang, static), macOS and Windows; sanitizers,
  fuzzing, an nvcc build of the CUDA backend and Python CUDA operators; an
  opt-in self-hosted GPU workflow; a release workflow with provenance
  attestation.
- Documentation in `docs/`: algorithm, Python API, C API, CUDA engine,
  development guide.

### Changed

- The Python extensions are built against CPython's limited API (abi3) and
  no longer use pybind11.
- The CPU core is split into modules (decoder, kernels, dispatch, C ABI)
  instead of one 4,300-line file. `src/c_api.cpp` includes `dbs.h` rather
  than re-declaring it.
- `dbs_create_ex` rejects negative or non-finite options instead of silently
  replacing them with defaults. Zero-initialised fields still select
  defaults.
- The GNMT length penalty is evaluated in double precision from basic IEEE
  operations (no `pow`) and rounded once, in a header the CPU and CUDA code
  share, so the two agree bit for bit. Scores with `length_penalty_alpha != 0`
  can differ from 1.x in the last bit.
- The softmax, dot product and sigmoid of the C-level surrogates are scalar on
  every kernel path (the AVX-512 path used a different `exp`), and the core is
  compiled without floating-point contraction, so results do not depend on
  the SIMD path. The row scan keeps its SIMD paths, now for banned tokens and
  EOS masking too, and a NEON path.
- Input validation (`validate_inputs`) checks every element of every row the
  search reads, during the scan, in C, PyTorch (CPU and CUDA) and JAX.
  Previously the C library sampled 1000 entries, JAX did not check at all, and
  PyTorch made three extra passes over the tensor.
- The relaxed top-k pool is opt-in (`relaxed_pool_multiplier` defaults to 0):
  keeping `8 * K` candidates per step made every decode pay for it
  (K=64: 3.4 ms to 0.9 ms; K=256: 24 ms to 4 ms). `vocab_block` is ignored.
- Invalid arguments and inputs return `-1` as the header documents (they
  returned `-2`), and every failing call records its own error message.
- CUDA: no kernel sorts a whole tile of candidates any more. Beams up to 16
  use a register top-k scan, sized so that a single example still fills the
  GPU; every scan, reduction and selection finds a block's best `K` keys with
  a radix select and only the step's `K` winners are sorted. On an RTX 4080
  SUPER a decode of B=8, T=16, V=32k takes 4.3 ms instead of 28.7 ms at K=64
  (0.26 ms instead of 0.59 ms for B=1, K=4); results are unchanged, bit for
  bit. The backward processes the beams of a step in parallel
  (deterministically) instead of one thread per example.
- `dbs_decode_batch*` run on the calling thread as well and do not start
  threads for a single example; variable batches read smaller beams in place
  instead of copying them.
- PyTorch 2.4 or newer is required (for `torch.library.register_fake` and
  `register_autograd`).
- SIMD scans pass candidates tied with the pool threshold to the exact
  comparator, so every ISA path matches the scalar reference in all tie cases.
- `dbs_last_error` returns a thread-local copy, so the pointer stays valid if
  another thread records an error on the same decoder.
- The CUDA API is reduced to `dbs_cuda_decode` / `dbs_cuda_backward` and is
  asynchronous by default; its header carries export macros.
- Python 3.10 or newer is required.
- The C ABI's relaxed pool, `vocab_block` and status-code changes are listed
  above; code that sets `relaxed_pool_multiplier` explicitly, or only checks
  for a non-zero status, is unaffected.

### Fixed

- `beam_search(..., device="cuda")` rejected rows on `cuda:0`, because
  `torch.device("cuda")` does not compare equal to `torch.device("cuda:0")`.
- Integers beyond 32 bits were truncated instead of rejected: `steps=[4,
  2**32 + 1]` decoded one step for the second example (PyTorch on CPU and
  CUDA, and JAX with 64-bit arrays), and through JAX's ctypes bindings
  `min_length=2**32 + 1` became 1 and `no_repeat_ngram_size=2**32 + 1` became
  1. Step counts are now range-checked before they are narrowed, non-integer
  step counts are rejected, and `BeamOptions` rejects integer options outside
  the 32-bit range.
- `torch.func.grad`, `vjp` and `jacrev` (and `vmap` of them) failed on
  `final_scores`: the autograd formula registered with `torch.library` is an
  `autograd.Function` without a separate `setup_context`, which `torch.func`
  requires.
- The length penalty converted `length_penalty_alpha` to `int` before checking
  its range, which is undefined behaviour for exponents beyond `INT_MAX`.
- `banned_tokens` cost a `[V]` Python list and its conversion to a tensor on
  every PyTorch call (4 ms at V=128k); the mask is now scattered on the
  target device.
- Backward rejected results whose beam size differed from the decoder's, so
  examples of `dbs_decode_batch_variable` with their own beam sizes could not
  be differentiated.
- `dbs_result_validate_deterministic_order` skipped the raw-score tie-break
  and reported the decoder's own output as out of order (with a length
  penalty and EOS).
- `dbs_allocator_counters_reset` zeroed the live byte count, which then went
  negative as allocations were freed; it now resets only the call count.
- Early failures (null handle or output pointer) returned without recording
  an error, so `dbs_last_global_error()` showed the previous call's message.
- `dbs_set_deterministic_seed` wrote the seed without synchronization.
- The CPU-only `libdbs_cuda` stub exported no symbols (hidden visibility and
  no export macros) and lacked five declared functions, so linking against it
  failed.
- The CUDA sparse backward attributed each final beam's gradient to the same
  slot index at every step instead of following the beam's path, and used a
  different estimator from the CPU. It is replaced by the exact path gradient.
- Typed and variable-batch decoding use overflow-checked size arithmetic.
- Batch decoding records errors in stats and reports real elapsed time.
- The JAX custom VJP no longer calls itself from its forward rule.

### Removed

- `dbs_validate_production_gate_manifest`, a stub that always failed.
- The test-only SIMD parity hooks, which were compiled into every build of the
  library; the parity tests now live in `tests/internal_tests.cpp`.
- The `torch_dbs_extension`, `torch_dbs` and `jax_dbs` modules (use
  `beamgrad`, `beamgrad._ctypes` and `beamgrad.jax`).
- `dbs_cuda_decode_forward*`, `dbs_cuda_decode_forward_variable`,
  `dbs_cuda_backward_build_sparse` and `dbs_cuda_sparse_backward_scatter`
  (use `dbs_cuda_decode` / `dbs_cuda_backward`); the
  `DBS_ENABLE_SCORE_ONLY_FAST_PATH` switch and the `DBS_CUDA_USE_FAST_MATH`
  option.
- Placeholder release material: SBOM/provenance templates, fixture manifests
  and platform locks with `TBD` values, and the scripts built around them.

### Migrating from 1.x

| 1.x | 2.0 |
|---|---|
| `pip install dbs-torch` | `pip install --no-build-isolation beamgrad` |
| `from torch_dbs_extension import DBSOptions, final_scores` | `from beamgrad import BeamOptions, final_scores` |
| `DBSOptions(beam_size=K, selected_temperature=..., ...)` | `BeamOptions(beam_size=K, eos_token=..., min_length=..., length_penalty_alpha=...)`; the other fields never affected `final_scores` |
| `torch_dbs_extension.decode(x, opts)` (CUDA only) | `beamgrad.decode(x, opts)` on CPU or CUDA |
| `DBS_BUILD_TORCH_CUDA=1` | automatic; force with `BEAMGRAD_CUDA=1` |
| `find_package(dbs)`, `dbs::dbs` | `find_package(beamgrad)`, `beamgrad::dbs` |

## 1.0.0

First stable release, as `dbs-torch` / `differentiable-beam-search-cuda`.

- C ABI version 10: hard beam search with deterministic ordering, GNMT length
  penalty, EOS handling and minimum length; sparse (default) and capped dense
  surrogate backward over final scores, selected-beam weights and a relaxed
  top-k pool; batch and variable-length batch decoding; banned/forced tokens,
  repetition penalty, n-gram blocking and token-filter callbacks; fp16/bf16
  input; model-callback decoding with reusable workspaces; statistics and
  JSON summaries.
- Runtime-dispatched AVX-512, AVX2, SSE4.2 and NEON kernels.
- PyTorch extension with CPU autograd and a CUDA forward kernel for beams up
  to 32 (other cases ran on the CPU); ctypes and JAX wrappers.

## 0.x

Development releases leading to 1.0: the C ABI, sparse-by-default backward,
batching, constraints, SIMD kernels, and the first CUDA kernels.
