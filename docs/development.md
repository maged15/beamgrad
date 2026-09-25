# Development

## Repository layout

```
include/            public C headers: dbs.h (CPU ABI), dbs_cuda.h (CUDA)
src/                CPU core: decoder, SIMD kernels, runtime dispatch, C ABI
cuda/               CUDA engine, CPU-only stub, and the CUDA emulation layer
python/beamgrad/    the Python package
python/csrc/        PyTorch operators (CPU, CUDA) and the bundled-libdbs shim
python/tests/       Python tests
tests/              C/C++ tests, emulated CUDA parity tests, fuzz harness
examples/           runnable Python and C examples
benchmarks/         C microbenchmark, PyTorch benchmark, Hugging Face comparison
abi/                exported-symbol manifests checked by ctest
docs/               documentation
```

## Building

### C/C++/CUDA libraries (CMake)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /usr/local        # headers, libs, CMake package, pkg-config
```

| option | default | effect |
|---|---|---|
| `DBS_ENABLE_CUDA` | OFF | build the native CUDA backend (otherwise a stub) |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89;90` | GPU architectures (`native` for the local GPU) |
| `DBS_BUILD_SHARED` | ON | shared libraries (static when OFF) |
| `DBS_BUILD_TESTS` / `DBS_BUILD_BENCHMARKS` | ON when top-level | tests / `dbs_bench` |
| `DBS_BUILD_FUZZER` | OFF | libFuzzer harness (Clang) |
| `DBS_ENABLE_SANITIZERS` | OFF | AddressSanitizer + UndefinedBehaviorSanitizer |
| `DBS_ENABLE_TSAN` | OFF | ThreadSanitizer |
| `DBS_WARNINGS_AS_ERRORS` | OFF | `-Werror` / `/WX` |

The `Makefile` wraps the common invocations: `make test`, `make cuda`,
`make asan`, `make bench`, `make python`, `make python-test`, `make lint`.

### Python package

```bash
pip install torch                                # the build compiles against it
pip install --no-build-isolation -e ".[test]"    # editable, with test extras
pytest python/tests
```

`--no-build-isolation` makes the extension compile against the PyTorch you
will import. On Windows, build from a Visual Studio developer prompt (x64)
with `DISTUTILS_USE_SDK=1` set. `BEAMGRAD_CUDA` controls the CUDA operators: `auto` (default)
builds them when a CUDA toolkit (`nvcc`, `CUDA_HOME`) and a CUDA-enabled
PyTorch are present, `1` requires them, `0` skips them. `TORCH_CUDA_ARCH_LIST`
selects GPU architectures, for example `"8.0;9.0"`.

## Testing

`ctest` runs:

| test | what it checks |
|---|---|
| `dbs_tests` | the C ABI: decoding, constraints, EOS, batching, typed input, backward vs finite differences, options validation, error reporting |
| `dbs_internal_tests` | every SIMD row scan the host supports against the scalar reference, bit for bit, on random rows (ties, `-inf`, NaN/`+inf`, banned and masked tokens); full decode + backward on every path; constrained decoding against a direct reference implementation; input validation; model-step decoding |
| `dbs_cuda_emulation_tests` | the CUDA engine, run through the emulation layer under three thread schedules, against libdbs, bit for bit (Linux) |
| `dbs_cuda_device_tests` | with `DBS_ENABLE_CUDA=ON`: the compiled CUDA engine on a real GPU against libdbs, bit for bit (decode trace, final scores, gradient) on randomized cases, and the NaN/`+inf` flags; skipped without a GPU |
| `dbs_c_api_test` | both headers compile as C and both libraries export their API |
| `dbs_example_c_api` | `examples/c_api.c` builds and runs |
| `dbs_abi_symbols` | exported symbols match `abi/libdbs.symbols` and keep the baseline (Linux, shared builds) |

The C++ tests use an always-on `CHECK` macro, so they check the same things in
Release and Debug builds.

`pytest python/tests` covers the Python API (exact parity with the C ABI,
finite-difference gradients, path/score consistency, EOS/min-length/length
penalty, constraints, variable steps, validation, `torch.compile`, fake
tensors, `torch.vmap`, `torch.func`), the ctypes bindings, JAX (values, gradients, `jit`,
`vmap`, validation), exact CUDA-vs-CPU parity (when a GPU is available), and
`beamgrad.hf` against tiny random `transformers` models (Llama, GPT-2; when
`transformers` is installed): the same beams as `generate()`, re-scoring that
reproduces the search's scores, and the same gradient both ways.

Sanitizers and fuzzing:

```bash
make asan                                                   # ASan + UBSan, all ctest suites
cmake -S . -B build-fuzz -DDBS_BUILD_FUZZER=ON -DDBS_ENABLE_SANITIZERS=ON \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz && ./build-fuzz/dbs_fuzz -max_total_time=300
```

The harness (`tests/fuzz_dbs.cpp`) derives every choice from the input. It
covers `dbs_decode` with a relaxed pool and both backward passes (with
selected-weight, relaxed-pool and final-score gradients),
`dbs_decode_constrained_ex` (banned and forced tokens, `min_length`, n-gram
blocking, repetition penalty), `dbs_decode_typed` (F16, BF16),
`dbs_decode_batch_into` then `dbs_backward_batch_into` on a trace it may
corrupt, and `dbs_decode_batch_variable`.

Besides the sanitizers, it checks that:
- every call returns `DBS_OK` or `DBS_ERROR_INVALID_ARGUMENT`;
- the sparse backward equals the dense one;
- F16/BF16 input decodes exactly like the same values in float32;
- a corrupted trace is rejected exactly when it is out of range.

`dbs_fuzz` compiles its own instrumented copy of the library, so coverage
guides the fuzzer through the library, not just the harness.

## Continuous integration

- **CI** (every push and pull request): C++ on Linux (GCC and Clang with
  warnings as errors, plus a static build), macOS and Windows; ASan/UBSan,
  TSan and a fuzz smoke run; an nvcc build of libdbs_cuda and of the Python
  CUDA operators in a CUDA 12.6 container; the Python package on Linux, macOS
  and Windows with the newest PyTorch (with JAX and `transformers` on Linux,
  Python 3.13), and on Linux with the oldest supported
  versions (Python 3.10, PyTorch 2.4, JAX 0.4.20); lint and version metadata.
- **GPU**: the CUDA tests and benchmark on a self-hosted GPU runner. It is
  enabled by the repository variable `BEAMGRAD_GPU_RUNNER=true` and a runner
  labelled `gpu`.
- **Fuzz**: an hour-long libFuzzer campaign every week.

## Benchmarks

```bash
./build/dbs_bench                               # C ABI microbenchmark matrix
./build/dbs_bench 16 8 32000 4 20               # T K V B repeats
python benchmarks/benchmark.py --device all     # PyTorch API vs a torch.topk beam search
python benchmarks/hf_beam_search.py --train     # beam_search on a Hugging Face LM vs generate()
python benchmarks/hf_training_memory.py --check # training-step memory per gradient mode
```

[benchmarks.md](benchmarks.md) explains what each one measures (kernel or end to
end) and how to read the numbers.

`benchmark.py` also asserts that beamgrad's scores match the reference, and
`--csv` records results with the environment. `hf_beam_search.py` (needs
`transformers` and a GPU) drives an open-weights causal LM (default
`Qwen/Qwen2.5-0.5B`) with `beamgrad.beam_search` and with `transformers`'
`generate(num_beams=...)`, and reports whether they return the same beams,
their speed, and a short fine-tuning run through the search. In float32 the
beams and scores are identical. In bfloat16 the logits often tie exactly, and
the two break ties differently (beamgrad by parent slot then token id,
`torch.topk` in an unspecified order); the script counts those prompts.

## Releasing

1. Update `VERSION`, `DBS_VERSION_*` in `include/dbs.h`, `project(... VERSION)`
   in `CMakeLists.txt`, `python/beamgrad/_version.py`, `version` in
   `CITATION.cff`, and add a `CHANGELOG.md` section.
   `python scripts/check_version_metadata.py` checks that they agree.
2. If the exported symbols changed, update `abi/libdbs.symbols`. If the change
   is binary-incompatible, bump `DBS_ABI_VERSION` in both `include/dbs.h` and
   `CMakeLists.txt`.
3. Tag `vX.Y.Z` and push the tag. The **Release** workflow checks the version,
   builds and tests the source distribution, and creates a GitHub release
   with a build provenance attestation. It publishes to PyPI when
   `PYPI_PUBLISH=true` and PyPI trusted publishing is configured for the
   `pypi` environment.
