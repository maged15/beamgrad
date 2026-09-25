# Contributing to beamgrad

Thanks for helping. Bug reports, fixes, performance work and documentation are
all welcome.

## Setting up

```bash
git clone https://github.com/maged15/beamgrad && cd beamgrad
pip install torch ruff "setuptools>=77" wheel "packaging>=24.2"
make python          # editable install with test extras
make test            # C/C++ suites, including the emulated CUDA parity tests
make python-test     # Python suites
```

`make python` builds with `--no-build-isolation`, so that the extensions are
compiled against the PyTorch you import. The build therefore uses the build
tools already installed rather than fresh ones. `setuptools>=77` needs
`packaging>=24.2`, and an older `packaging` in the environment makes the
build fail, hence the explicit install above.

With a CUDA toolkit installed, `make cuda` builds and tests the native CUDA
backend, and the Python build compiles the CUDA operators automatically.
[docs/development.md](docs/development.md) covers the options.

## Making a change

- **Keep results identical across backends.** The CPU kernels and the CUDA
  engine must select the same beams with the same scores. If you change the
  decoder, update every backend. The SIMD parity tests and the emulated CUDA
  parity tests will tell you if they disagree.
- **Add a test** that fails without your change. C++ tests use the `CHECK`
  macro from `tests/check.hpp` (never `assert`, which Release builds remove).
- **Keep the ABI stable.** Do not change exported `dbs_*` signatures or struct
  layouts in `include/dbs.h`. New functions are fine: add them to
  `abi/libdbs.symbols`. A binary-incompatible change needs a
  `DBS_ABI_VERSION` bump and a changelog entry.
- **Run the checks** before opening a pull request: `make test`,
  `make python-test` and `make lint`. For CUDA changes, run `make cuda` and,
  if you have a GPU, `pytest python/tests/test_cuda.py`.
- **Update the docs and `CHANGELOG.md`** when behaviour changes.

## Reporting bugs

Please include the beamgrad version, platform, compiler or PyTorch/CUDA
versions, and a minimal reproducer. For performance issues, include the output
of `python benchmarks/benchmark.py`.

## Security issues

See [SECURITY.md](SECURITY.md). Please do not open public issues for
vulnerabilities.
