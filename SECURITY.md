# Security policy

## Supported versions

Security fixes are made on the latest release line (currently 2.x).

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/maged15/beamgrad/security/advisories/new)
rather than in a public issue. Include the affected version, platform, build
options, and a reproducer if you can share one. You can expect an
acknowledgement within a week.

Areas of particular interest: input validation in the C ABI and the PyTorch
operators (shapes, sizes, integer overflow), memory ownership of C handles,
out-of-bounds access in the CPU or CUDA kernels, and crashes found by the fuzz
harness (`tests/fuzz_dbs.cpp`).
