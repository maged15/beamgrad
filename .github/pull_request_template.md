## What does this change?

## How was it tested?

- [ ] `make test` (C/C++ suites, including emulated CUDA parity)
- [ ] `make python-test`
- [ ] `make lint`
- [ ] `make cuda` / GPU tests, if CUDA code changed

## Checklist

- [ ] Backends still agree (CPU kernels and CUDA select the same beams)
- [ ] Exported C symbols unchanged, or `abi/libdbs.symbols` updated
- [ ] Docs and `CHANGELOG.md` updated if behaviour changed
