#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check that every place that states a version agrees with the VERSION file."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    version = read("VERSION").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append(f"VERSION must be MAJOR.MINOR.PATCH, got {version!r}")

    header = read("include/dbs.h")
    parts = [re.search(rf"#define\s+DBS_VERSION_{name}\s+(\d+)", header) for name in ("MAJOR", "MINOR", "PATCH")]
    if not all(parts):
        errors.append("include/dbs.h must define DBS_VERSION_MAJOR/MINOR/PATCH")
    else:
        header_version = ".".join(m.group(1) for m in parts)
        if header_version != version:
            errors.append(f"include/dbs.h version {header_version} != VERSION {version}")

    cmake = read("CMakeLists.txt")
    project = re.search(r"project\(\s*\w+\s+VERSION\s+([\d.]+)", cmake)
    if not project or project.group(1) != version:
        errors.append(f"CMakeLists.txt project VERSION must be {version}")

    abi = re.search(r"#define\s+DBS_ABI_VERSION\s+(\d+)", header)
    cmake_abi = re.search(r"set\(DBS_ABI_VERSION\s+(\d+)", cmake)
    if not abi or not cmake_abi or abi.group(1) != cmake_abi.group(1):
        errors.append("DBS_ABI_VERSION in include/dbs.h and CMakeLists.txt must match")

    pyproject = read("pyproject.toml")
    if not re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', pyproject, flags=re.MULTILINE):
        errors.append('pyproject.toml must list "version" in dynamic (setup.py reads it from VERSION)')
    if 'ROOT / "VERSION"' not in read("setup.py"):
        errors.append("setup.py must read the package version from VERSION")

    py_version = re.search(r'__version__\s*=\s*"([^"]+)"', read("python/beamgrad/_version.py"))
    if not py_version or py_version.group(1) != version:
        errors.append(f"python/beamgrad/_version.py must set __version__ = {version!r}")

    changelog = read("CHANGELOG.md")
    heading = re.search(r"^##\s+\[?(\d+\.\d+\.\d+)", changelog, flags=re.MULTILINE)
    if not heading or heading.group(1) != version:
        errors.append(f"the first CHANGELOG.md release heading must be {version}")

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"version metadata ok: {version} (C ABI {abi.group(1) if abi else '?'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
