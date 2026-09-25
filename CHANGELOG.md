# Changelog

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
