# Changelog

## 2.0.0 (unreleased)

The project is renamed **beamgrad** (previously `differentiable-beam-search-cuda`
and the `dbs-torch` Python package). The C library keeps its name, `libdbs`,
and its binary interface: C ABI version 10 and the exported symbol set are
unchanged.

### Highlights

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
  decoder.

### Added

- `dbs_cuda_decode` / `dbs_cuda_backward` with workspace-size queries,
  per-example metadata, optional trace outputs, and asynchronous execution
  (`DBS_CUDA_SYNC_CHECK=1` synchronizes for debugging).
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

- The CPU core is split into modules (decoder, kernels, dispatch, C ABI)
  instead of one 4,300-line file. `src/c_api.cpp` includes `dbs.h` rather
  than re-declaring it.
- `dbs_create_ex` rejects negative or non-finite options instead of silently
  replacing them with defaults. Zero-initialised fields still select
  defaults.
- The GNMT length penalty is evaluated in double precision and rounded once,
  so CPU and GPU agree bit for bit. Scores with `length_penalty_alpha != 0`
  can differ from 1.x in the last bit.
- SIMD scans pass candidates tied with the pool threshold to the exact
  comparator, so every ISA path matches the scalar reference in all tie cases.
- `dbs_last_error` returns a thread-local copy, so the pointer stays valid if
  another thread records an error on the same decoder.
- The CUDA API is reduced to `dbs_cuda_decode` / `dbs_cuda_backward` and is
  asynchronous by default; its header carries export macros.
- Python `validate_inputs` checks every element on both devices (CPU used to
  sample 1000).
- Python 3.10 or newer is required.

### Fixed

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
