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


class DBSAdvancedConstraintsC(ctypes.Structure):
    """Mirror of ``DBSAdvancedConstraintsC``."""

    _fields_ = [
        ("banned_tokens", ctypes.c_void_p),
        ("forced_tokens", ctypes.c_void_p),
        ("min_length", ctypes.c_int),
        ("repetition_penalty", ctypes.c_float),
        ("no_repeat_ngram_size", ctypes.c_int),
        ("token_filter", ctypes.c_void_p),
        ("token_filter_user_data", ctypes.c_void_p),
        ("batch_index", ctypes.c_int),
    ]


class DBSDecodeOutputsC(ctypes.Structure):
    """Mirror of ``DBSDecodeOutputsC`` (all fields are pointers)."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "final_scores",
            "final_raw_scores",
            "final_lengths",
            "tokens",
            "parents",
            "lengths",
            "scores",
            "raw_scores",
            "from_logprob",
        )
    ]


class DBSDecodeOutputsExC(ctypes.Structure):
    """Mirror of ``DBSDecodeOutputsExC``: ``base`` plus the pool trace and the rows' logsumexp."""

    _fields_ = [
        ("base", DBSDecodeOutputsC),
        *(
            (name, ctypes.c_void_p)
            for name in (
                "pool_parents",
                "pool_tokens",
                "pool_lengths",
                "pool_scores",
                "pool_raw_scores",
                "pool_from_logprob",
                "row_lse",
            )
        ),
        ("reserved", ctypes.c_void_p * 4),
    ]


class DBSBackwardInputsC(ctypes.Structure):
    """Mirror of ``DBSBackwardInputsC`` (the pointers as ``c_void_p``)."""

    _fields_ = [
        ("batch_size", ctypes.c_int),
        ("steps", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("pool_size", ctypes.c_int),
        *(
            (name, ctypes.c_void_p)
            for name in (
                "steps_per_example",
                "parents",
                "tokens",
                "lengths",
                "from_logprob",
                "pool_parents",
                "pool_tokens",
                "pool_lengths",
                "pool_from_logprob",
                "grad_final_scores",
                "grad_final_raw_scores",
                "grad_scores",
                "grad_raw_scores",
                "grad_pool_scores",
                "grad_pool_raw_scores",
                "logits",
            )
        ),
        ("logits_type", ctypes.c_int),
        ("row_lse", ctypes.c_void_p),
        ("reserved", ctypes.c_void_p * 4),
    ]


# DBSDataTypeC
DTYPE_F32, DTYPE_F16, DTYPE_BF16 = 0, 1, 2


def options_to_c(options: BeamOptions) -> DBSOptionsC:
    """Convert :class:`BeamOptions`; unused C fields take their documented defaults."""
    c = DBSOptionsC()
    c.beam_size = options.beam_size
    c.eos_token = options.eos_token
    c.min_length = options.min_length
    c.length_penalty_alpha = options.length_penalty_alpha
    c.validate_inputs = 1 if options.validate_inputs else 0
    return c


def constraints_to_c(options: BeamOptions, banned_mask) -> DBSAdvancedConstraintsC | None:
    """The options' decoding constraints, or ``None`` when there are none.

    ``banned_mask`` is a contiguous uint8 ``[V]`` array (or ``None``) that must
    outlive the call it is passed to.
    """
    if banned_mask is None and options.no_repeat_ngram_size == 0 and options.repetition_penalty == 1.0:
        return None
    c = DBSAdvancedConstraintsC()
    c.banned_tokens = None if banned_mask is None else banned_mask.ctypes.data
    c.min_length = -1
    c.repetition_penalty = float(options.repetition_penalty)
    c.no_repeat_ngram_size = options.no_repeat_ngram_size
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
        "dbs_decode_batch_into": (
            [p, p, i32, i32, i32, p, ctypes.POINTER(DBSAdvancedConstraintsC), i32, ctypes.POINTER(DBSDecodeOutputsC)],
            i32,
        ),
        "dbs_backward_batch_into": ([p, i32, i32, i32, p, p, p, p, p, p, i32, p], i32),
        "dbs_decode_batch_into_ex": (
            [
                p,
                p,
                i32,
                i32,
                i32,
                i32,
                i32,
                p,
                ctypes.POINTER(DBSAdvancedConstraintsC),
                i32,
                ctypes.POINTER(DBSDecodeOutputsExC),
            ],
            i32,
        ),
        "dbs_backward_batch_into_ex": ([p, ctypes.POINTER(DBSBackwardInputsC), i32, p], i32),
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
    """Raise if ``status`` is non-zero: ValueError for invalid arguments or inputs
    (DBS_ERROR_INVALID_ARGUMENT), RuntimeError otherwise, with libdbs' message."""
    if status == 0:
        return
    # The thread-local message is reliable even when threads share a handle.
    message = lib.dbs_last_global_error() or (lib.dbs_last_error(handle) if handle else None)
    text = (message or b"libdbs call failed").decode("utf-8", errors="replace")
    raise ValueError(text) if status == -1 else RuntimeError(text)


def _check_layout() -> None:
    if ctypes.sizeof(DBSOptionsC) != OPTIONS_SIZE:
        raise RuntimeError(f"DBSOptionsC is {ctypes.sizeof(DBSOptionsC)} bytes, expected {OPTIONS_SIZE}")
    for name, offset in OPTIONS_LAYOUT.items():
        if getattr(DBSOptionsC, name).offset != offset:
            raise RuntimeError(
                f"DBSOptionsC.{name} is at offset {getattr(DBSOptionsC, name).offset}, expected {offset}"
            )


_check_layout()
