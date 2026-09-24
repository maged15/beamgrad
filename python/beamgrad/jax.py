# SPDX-License-Identifier: MIT
"""JAX integration: ``beamgrad.jax.final_scores`` with a custom VJP.

Decoding runs on the host through the bundled libdbs (``jax.pure_callback``),
so it works under ``jit`` and ``grad`` on any JAX backend but moves data to the
host. For device-resident decoding use the PyTorch API.
"""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp
import numpy as np

from ._ctypes import check, load_library, options_to_c
from ._options import BeamOptions

__all__ = ["final_scores"]

_F32P = ctypes.POINTER(ctypes.c_float)


def _decode_one(lib, handle, x: np.ndarray) -> ctypes.c_void_p:
    t, _, v = x.shape
    result = ctypes.c_void_p()
    check(lib, handle, lib.dbs_decode(handle, x.ctypes.data_as(_F32P), t, v, ctypes.byref(result)))
    return result


def _host_forward(x: np.ndarray, options: BeamOptions, lib_path: str | None) -> np.ndarray:
    lib = load_library(lib_path)
    x = np.ascontiguousarray(x, dtype=np.float32)
    k = options.beam_size
    handle = ctypes.c_void_p()
    check(lib, None, lib.dbs_create_ex(options_to_c(options), ctypes.byref(handle)))
    try:
        out = np.empty((x.shape[0], k), dtype=np.float32)
        for b in range(x.shape[0]):
            result = _decode_one(lib, handle, x[b])
            try:
                out[b] = np.ctypeslib.as_array(lib.dbs_result_final_scores(result), shape=(k,))
            finally:
                lib.dbs_free_result(result)
        return out
    finally:
        lib.dbs_destroy(handle)


def _host_backward(x: np.ndarray, g: np.ndarray, options: BeamOptions, lib_path: str | None) -> np.ndarray:
    lib = load_library(lib_path)
    x = np.ascontiguousarray(x, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    handle = ctypes.c_void_p()
    check(lib, None, lib.dbs_create_ex(options_to_c(options), ctypes.byref(handle)))
    try:
        grad = np.zeros(x.shape, dtype=np.float32)
        for b in range(x.shape[0]):
            result = _decode_one(lib, handle, x[b])
            backward = ctypes.c_void_p()
            try:
                check(
                    lib,
                    handle,
                    lib.dbs_backward(handle, result, None, None, g[b].ctypes.data_as(_F32P), ctypes.byref(backward)),
                )
                n = lib.dbs_backward_sparse_logprob_count(backward)
                if n:
                    idx = np.ctypeslib.as_array(lib.dbs_backward_sparse_logprob_indices(backward), shape=(n,))
                    val = np.ctypeslib.as_array(lib.dbs_backward_sparse_logprob_values(backward), shape=(n,))
                    np.add.at(grad[b].reshape(-1), idx, val)
            finally:
                if backward.value:
                    lib.dbs_free_backward(backward)
                lib.dbs_free_result(result)
        return grad
    finally:
        lib.dbs_destroy(handle)


def final_scores(log_probs, options: BeamOptions, lib_path: str | None = None):
    """Final beam scores for ``[T, K, V]`` or ``[B, T, K, V]`` log-probabilities.

    Same semantics and gradients as :func:`beamgrad.final_scores`.
    """
    if log_probs.ndim not in (3, 4):
        raise ValueError(f"log_probs must have shape [T, K, V] or [B, T, K, V], got {log_probs.shape}")
    unbatched = log_probs.ndim == 3
    x = log_probs[None] if unbatched else log_probs
    if x.shape[2] != options.beam_size:
        raise ValueError(f"log_probs has {x.shape[2]} beams but options.beam_size is {options.beam_size}")
    if options.eos_token >= x.shape[3]:
        raise ValueError(f"eos_token {options.eos_token} is outside the vocabulary (size {x.shape[3]})")
    batch = x.shape[0]

    def forward_callback(y):
        return jax.pure_callback(
            lambda a: _host_forward(np.asarray(a), options, lib_path),
            jax.ShapeDtypeStruct((batch, options.beam_size), jnp.float32),
            y,
        )

    @jax.custom_vjp
    def scores(y):
        return forward_callback(y)

    def scores_fwd(y):
        return forward_callback(y), y

    def scores_bwd(y, g):
        grad = jax.pure_callback(
            lambda a, b: _host_backward(np.asarray(a), np.asarray(b), options, lib_path),
            jax.ShapeDtypeStruct(y.shape, jnp.float32),
            y,
            g,
        )
        return (grad.astype(y.dtype),)

    scores.defvjp(scores_fwd, scores_bwd)
    out = scores(x.astype(jnp.float32))
    return out[0] if unbatched else out
