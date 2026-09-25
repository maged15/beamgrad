# Installation

beamgrad's operators are compiled against PyTorch's C++ API. A build works
with one PyTorch minor version, and for CUDA with that PyTorch's CUDA
variant. It works with any CPython from 3.10 on, because the extensions use
only CPython's limited API.

## Prebuilt wheels

Each GitHub release has wheels for:

| platform | PyTorch | variants | Python |
|---|---|---|---|
| Linux x86_64 (glibc ≥ 2.28) | 2.13, 2.14 | `cpu`, `cu126`, `cu130` | 3.10+ |
| macOS arm64 (11+) | 2.13, 2.14 | `cpu` | 3.10+ |
| Windows x64 | 2.13, 2.14 | `cpu` | 3.10+ |

CUDA wheels contain kernels for these GPU architectures, plus PTX for newer
GPUs:

| variant | compute capability |
|---|---|
| `cu126` | 7.5, 8.0, 8.6, 8.9, 9.0 |
| `cu130` | 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0 |

A wheel's local version label names the PyTorch it needs. For example,
`2.0.0+pt214cu126` requires `torch==2.14.*` built for CUDA 12.x. Install
PyTorch first, then the wheel that matches it. This prints the label for the
installed PyTorch:

```bash
python -c "import torch; v = torch.__version__.split('+')[0].split('.'); c = torch.version.cuda; print(f'pt{v[0]}{v[1]}' + (f'cu{c.replace(\".\", \"\")}' if c else 'cpu'))"
```

Then install that wheel from the release, for example:

```bash
pip install torch==2.14.* --index-url https://download.pytorch.org/whl/cu126
pip install "https://github.com/maged15/beamgrad/releases/download/v2.0.0/beamgrad-2.0.0+pt214cu126-cp310-abi3-linux_x86_64.whl"
```

If there is no wheel for your combination, for example another PyTorch
version, a CUDA 12.8 build or Linux on ARM, build from source.

On a mismatch, `import beamgrad` says what is wrong:

- **Another PyTorch minor version**: an `ImportError` naming both versions.
- **Another CUDA major version**: a `RuntimeWarning` naming both CUDA
  versions. The CPU operators still work.

## From source

Building needs a C++17 compiler. For the CUDA operators it also needs a CUDA
toolkit whose major version matches your PyTorch's CUDA.

```bash
pip install torch "setuptools>=77" wheel "packaging>=24.2"
pip install --no-build-isolation beamgrad                                   # the PyPI sdist
pip install --no-build-isolation "git+https://github.com/maged15/beamgrad"  # or the latest source
```

`--no-build-isolation` compiles against the PyTorch you have. Without it, pip
builds in a temporary environment with the newest PyTorch, and
`import beamgrad` then reports the mismatch. It also means the build uses
the installed build tools, so install them first. `setuptools>=77` needs
`packaging>=24.2`.

Environment variables:

| variable | effect |
|---|---|
| `BEAMGRAD_CUDA` | `auto` (default): build the CUDA operators if `nvcc` and a CUDA-enabled PyTorch are found; `1`: require them; `0`: skip them |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for (default: the GPUs present) |
| `MAX_JOBS` | parallel compile jobs |
| `BEAMGRAD_PIN_TORCH=1` | make the built wheel require the PyTorch minor version it was built against (release wheels) |
| `BEAMGRAD_LOCAL_VERSION` | append a local version label, such as `pt214cu126` (release wheels) |

Tested in CI:

| | versions |
|---|---|
| Python | 3.10, 3.12, 3.13 (Linux, macOS, Windows) |
| PyTorch | 2.4 (oldest supported) to 2.14 |
| CUDA toolkit | 12.4 (with PyTorch 2.4), 12.6, 13.0 |
| Compilers | GCC, Clang, MSVC, Apple Clang; `nvcc` with GCC up to 15 |

The GitHub-hosted runners have no GPU. There, the CUDA kernels run under a
CPU emulation layer (see [cuda.md](cuda.md)). A self-hosted GPU runner runs
the device tests (`.github/workflows/gpu.yml`).

## The C library

`libdbs` (and `libdbs_cuda`) build with CMake and do not need Python or
PyTorch. See [c-api.md](c-api.md).
