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
benchmarks/         C microbenchmark and PyTorch benchmark
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
| `dbs_simd_internal_tests` | every SIMD path the host supports against the scalar reference, bit for bit, on adversarial cases (ties at the pool boundary, `-inf` rows, EOS, constraints) |
| `dbs_cuda_emulation_tests` | the CUDA engine, run through the emulation layer, against libdbs, bit for bit (Linux) |
| `dbs_c_api_test` | both headers compile as C and both libraries export their API |
| `dbs_example_c_api` | `examples/c_api.c` builds and runs |
| `dbs_abi_symbols` | exported symbols match `abi/libdbs.symbols` and keep the baseline (Linux, shared builds) |

The C++ tests use an always-on `CHECK` macro, so they check the same things in
Release and Debug builds.

`pytest python/tests` covers the Python API (exact parity with the C ABI,
finite-difference gradients, path/score consistency, EOS/min-length/length
penalty, variable steps, validation), the ctypes bindings, JAX (when
installed), and CUDA-vs-CPU parity (when a GPU is available).

Sanitizers and fuzzing:

```bash
make asan                                                   # ASan + UBSan, all ctest suites
cmake -S . -B build-fuzz -DDBS_BUILD_FUZZER=ON -DDBS_ENABLE_SANITIZERS=ON \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz && ./build-fuzz/dbs_fuzz -max_total_time=300
```

## Continuous integration

- **CI** (every push and pull request): C++ on Linux (GCC and Clang with
  warnings as errors, plus a static build), macOS and Windows; ASan/UBSan,
  TSan and a fuzz smoke run; an nvcc build of libdbs_cuda and of the Python
  CUDA operators in a CUDA 12.6 container; the Python package on Linux, macOS
  and Windows; lint and version metadata.
- **GPU**: the CUDA tests and benchmark on a self-hosted GPU runner. It is
  enabled by the repository variable `BEAMGRAD_GPU_RUNNER=true` and a runner
  labelled `gpu`.
- **Fuzz**: an hour-long libFuzzer campaign every week.

## Benchmarks

```bash
./build/dbs_bench                               # C ABI microbenchmark matrix
./build/dbs_bench 16 8 32000 4 20               # T K V B repeats
python benchmarks/benchmark.py --device all     # PyTorch API vs a torch.topk beam search
```

`benchmark.py` also asserts that beamgrad's scores match the reference, and
`--csv` records results with the environment.

## Releasing

1. Update `VERSION`, `DBS_VERSION_*` in `include/dbs.h`, `project(... VERSION)`
   in `CMakeLists.txt`, `python/beamgrad/_version.py`, and add a
   `CHANGELOG.md` section. `python scripts/check_version_metadata.py` checks
   that they agree.
2. If the exported symbols changed, update `abi/libdbs.symbols`. If the change
   is binary-incompatible, bump `DBS_ABI_VERSION` in both `include/dbs.h` and
   `CMakeLists.txt`.
3. Tag `vX.Y.Z` and push the tag. The **Release** workflow checks the version,
   builds and tests the source distribution, and creates a GitHub release
   with a build provenance attestation. It publishes to PyPI when
   `PYPI_PUBLISH=true` and PyPI trusted publishing is configured for the
   `pypi` environment.
