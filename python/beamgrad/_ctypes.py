# SPDX-License-Identifier: MIT
"""ctypes bindings for the libdbs C ABI (include/dbs.h).

The wheel ships libdbs as ``beamgrad._libdbs`` so it can be loaded without a
separate install. Set ``DBS_LIBRARY`` to load a different build instead.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

from ._options import BeamOptions

ABI_VERSION = 10


class DBSOptionsC(ctypes.Structure):
    """Mirror of ``DBSOptionsC``; the layout is checked against the ABI below."""

    _fields_ = [
        ("beam_size", ctypes.c_int),
        ("eos_token", ctypes.c_int),
        ("selected_temperature", ctypes.c_float),
        ("soft_topk_temperature", ctypes.c_float),
        ("relaxed_pool_multiplier", ctypes.c_int),
        ("vocab_block", ctypes.c_int),
        ("length_penalty_alpha", ctypes.c_float),
        ("soft_topk_tolerance", ctypes.c_float),
        ("soft_topk_max_iters", ctypes.c_int),
        ("min_length", ctypes.c_int),
        ("validate_inputs", ctypes.c_int),
        ("max_dense_gradient_elements", ctypes.c_int64),
        ("reserved0", ctypes.c_int),
        ("reserved1", ctypes.c_int),
    ]


# Offsets and size of DBSOptionsC on every supported platform (natural alignment).
OPTIONS_LAYOUT = {
    "beam_size": 0,
    "eos_token": 4,
    "selected_temperature": 8,
    "soft_topk_temperature": 12,
    "relaxed_pool_multiplier": 16,
    "vocab_block": 20,
    "length_penalty_alpha": 24,
    "soft_topk_tolerance": 28,
    "soft_topk_max_iters": 32,
    "min_length": 36,
    "validate_inputs": 40,
    "max_dense_gradient_elements": 48,
    "reserved0": 56,
    "reserved1": 60,
}
OPTIONS_SIZE = 64


def options_to_c(options: BeamOptions) -> DBSOptionsC:
    """Convert :class:`BeamOptions`; unused C fields take their documented defaults."""
    c = DBSOptionsC()
    c.beam_size = options.beam_size
    c.eos_token = options.eos_token
    c.min_length = options.min_length
    c.length_penalty_alpha = options.length_penalty_alpha
    c.validate_inputs = 0  # the Python front ends validate the whole tensor themselves
    c.relaxed_pool_multiplier = 1
    return c


def _default_library_path() -> str:
    override = os.environ.get("DBS_LIBRARY")
    if override:
        return override
    package_dir = Path(__file__).resolve().parent
    candidates = (
        sorted(package_dir.glob("_libdbs*.so"))
        + sorted(package_dir.glob("_libdbs*.pyd"))
        + sorted(package_dir.glob("_libdbs*.dylib"))
    )
    if not candidates:
        raise RuntimeError(
            "The bundled libdbs (beamgrad._libdbs) was not found. Reinstall beamgrad, or set "
            "DBS_LIBRARY to the path of a libdbs shared library built with CMake."
        )
    return str(candidates[0])


def _bind(lib: ctypes.CDLL) -> None:
    p, i32, i64, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)
    signatures = {
        "dbs_abi_version": ([], i32),
        "dbs_version_string": ([], ctypes.c_char_p),
        "dbs_last_error": ([p], ctypes.c_char_p),
        "dbs_last_global_error": ([], ctypes.c_char_p),
        "dbs_create_ex": ([DBSOptionsC, ctypes.POINTER(p)], i32),
        "dbs_destroy": ([p], None),
        "dbs_decode": ([p, f32, i32, i32, ctypes.POINTER(p)], i32),
        "dbs_backward": ([p, p, f32, f32, f32, ctypes.POINTER(p)], i32),
        "dbs_free_result": ([p], None),
        "dbs_free_backward": ([p], None),
        "dbs_result_final_scores": ([p], f32),
        "dbs_backward_sparse_logprob_count": ([p], i64),
        "dbs_backward_sparse_logprob_indices": ([p], ctypes.POINTER(ctypes.c_int64)),
        "dbs_backward_sparse_logprob_values": ([p], f32),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype


@cache
def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load libdbs (the bundled copy by default) and check its ABI version."""
    resolved = path or _default_library_path()
    try:
        lib = ctypes.CDLL(resolved)
    except OSError as exc:
        raise RuntimeError(f"Unable to load libdbs from {resolved!r}: {exc}") from exc
    _bind(lib)
    abi = lib.dbs_abi_version()
    if abi != ABI_VERSION:
        raise RuntimeError(f"libdbs at {resolved!r} has ABI version {abi}, expected {ABI_VERSION}")
    return lib


def check(lib: ctypes.CDLL, handle: int | None, status: int) -> None:
    """Raise a RuntimeError carrying libdbs' error message if ``status`` is non-zero."""
    if status == 0:
        return
    message = lib.dbs_last_error(handle) if handle else lib.dbs_last_global_error()
    raise RuntimeError((message or b"libdbs call failed").decode("utf-8", errors="replace"))


def _check_layout() -> None:
    if ctypes.sizeof(DBSOptionsC) != OPTIONS_SIZE:
        raise RuntimeError(f"DBSOptionsC is {ctypes.sizeof(DBSOptionsC)} bytes, expected {OPTIONS_SIZE}")
    for name, offset in OPTIONS_LAYOUT.items():
        if getattr(DBSOptionsC, name).offset != offset:
            raise RuntimeError(
                f"DBSOptionsC.{name} is at offset {getattr(DBSOptionsC, name).offset}, expected {offset}"
            )


_check_layout()
