# SPDX-License-Identifier: MIT
"""JAX integration: ``beamgrad.jax.final_scores`` with a custom VJP.

Decoding runs on the host through the bundled libdbs (``jax.pure_callback``),
so it works under ``jit``, ``grad`` and ``vmap`` on any JAX backend but moves
data to the host. The whole batch is decoded in one multi-threaded call, and
the forward pass keeps the decode trace, so the backward pass does not decode
again. For device-resident decoding use the PyTorch API.
"""

from __future__ import annotations

import ctypes
import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np

from ._ctypes import DBSDecodeOutputsC, check, constraints_to_c, load_library, options_to_c
from ._options import BeamOptions

__all__ = ["final_scores"]

# Batched (vmapped) calls reach the host callbacks with extra leading dimensions,
# which they fold into the batch. JAX 0.4.34 renamed `vectorized` to `vmap_method`.
_CALLBACK_BATCHING = (
    {"vmap_method": "broadcast_all"}
    if "vmap_method" in inspect.signature(jax.pure_callback).parameters
    else {"vectorized": True}
)


def _ptr(array: np.ndarray | None):
    return None if array is None else array.ctypes.data


class _Decoder:
    """A libdbs decoder handle for one set of options (freed with the object)."""

    def __init__(self, options: BeamOptions, lib_path: str | None):
        self.lib = load_library(lib_path)
        self.handle = ctypes.c_void_p()
        check(self.lib, None, self.lib.dbs_create_ex(options_to_c(options), ctypes.byref(self.handle)))

    def __del__(self):
        if getattr(self, "handle", None) is not None and self.handle.value:
            self.lib.dbs_destroy(self.handle)


def _fold_steps(steps: np.ndarray | None, lead: tuple[int, ...], n: int) -> np.ndarray | None:
    if steps is None:
        return None
    return np.ascontiguousarray(np.broadcast_to(np.asarray(steps), lead).reshape(n), dtype=np.int32)


def _host_decode(x, steps, options: BeamOptions, lib_path: str | None):
    x = np.asarray(x, dtype=np.float32)
    lead, (T, K, V) = x.shape[:-3], x.shape[-3:]
    n = math.prod(lead)
    x = np.ascontiguousarray(x.reshape(n, T, K, V))
    steps = _fold_steps(steps, lead, n)
    mask = options.banned_mask(V)
    banned = None if mask is None else np.asarray(mask, dtype=np.uint8)
    constraints = constraints_to_c(options, banned)

    final = np.empty((n, K), dtype=np.float32)
    parents = np.empty((n, T, K), dtype=np.int32)
    tokens = np.empty((n, T, K), dtype=np.int32)
    lengths = np.empty((n, T, K), dtype=np.int32)
    from_logprob = np.empty((n, T, K), dtype=np.uint8)
    out = DBSDecodeOutputsC()
    out.final_scores = _ptr(final)
    out.parents = _ptr(parents)
    out.tokens = _ptr(tokens)
    out.lengths = _ptr(lengths)
    out.from_logprob = _ptr(from_logprob)

    decoder = _Decoder(options, lib_path)
    status = decoder.lib.dbs_decode_batch_into(
        decoder.handle,
        _ptr(x),
        n,
        T,
        V,
        _ptr(steps),
        ctypes.byref(constraints) if constraints is not None else None,
        0,
        ctypes.byref(out),
    )
    check(decoder.lib, decoder.handle, status)
    return (
        final.reshape(*lead, K),
        parents.reshape(*lead, T, K),
        tokens.reshape(*lead, T, K),
        lengths.reshape(*lead, T, K),
        from_logprob.reshape(*lead, T, K),
    )


def _host_backward(
    parents, tokens, lengths, from_logprob, steps, g, vocab_size: int, options: BeamOptions, lib_path: str | None
):
    parents = np.asarray(parents, dtype=np.int32)
    lead, (T, K) = parents.shape[:-2], parents.shape[-2:]
    n = math.prod(lead)

    def flat(a, dtype):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=dtype), (*lead, T, K)).reshape(n, T, K))

    trace = [flat(parents, np.int32), flat(tokens, np.int32), flat(lengths, np.int32), flat(from_logprob, np.uint8)]
    g = np.ascontiguousarray(np.broadcast_to(np.asarray(g, dtype=np.float32), (*lead, K)).reshape(n, K))
    steps = _fold_steps(steps, lead, n)
    grad = np.zeros((n, T, K, vocab_size), dtype=np.float32)

    decoder = _Decoder(options, lib_path)
    status = decoder.lib.dbs_backward_batch_into(
        decoder.handle, n, T, vocab_size, _ptr(steps), *(_ptr(a) for a in trace), _ptr(g), 0, _ptr(grad)
    )
    check(decoder.lib, decoder.handle, status)
    return grad.reshape(*lead, T, K, vocab_size)


def final_scores(log_probs, options: BeamOptions, steps=None, lib_path: str | None = None):
    """Final beam scores for ``[T, K, V]`` or ``[B, T, K, V]`` log-probabilities.

    Same semantics, options and gradients as :func:`beamgrad.final_scores`,
    including ``steps`` (``[B]`` steps per example, batched input only). NaN or
    ``+inf`` in a row the search reads fails the call when
    ``options.validate_inputs`` is set (inside ``jit``, JAX reports the error
    raised by the host callback).
    """
    if not isinstance(options, BeamOptions):
        raise TypeError(f"options must be a beamgrad.BeamOptions, got {type(options).__name__}")
    if log_probs.ndim not in (3, 4):
        raise ValueError(f"log_probs must have shape [T, K, V] or [B, T, K, V], got {log_probs.shape}")
    unbatched = log_probs.ndim == 3
    if unbatched and steps is not None:
        raise ValueError("steps is only supported for batched [B, T, K, V] input")
    x = log_probs[None] if unbatched else log_probs
    B, T, K, V = x.shape
    if K != options.beam_size:
        raise ValueError(f"log_probs has {K} beams but options.beam_size is {options.beam_size}")
    if options.eos_token >= V:
        raise ValueError(f"eos_token {options.eos_token} is outside the vocabulary (size {V})")
    options.banned_mask(V)  # checks the banned ids against the vocabulary
    x = x.astype(jnp.float32)  # differentiable: gradients return in the input dtype

    trace_shapes = (
        jax.ShapeDtypeStruct((B, K), jnp.float32),
        jax.ShapeDtypeStruct((B, T, K), jnp.int32),
        jax.ShapeDtypeStruct((B, T, K), jnp.int32),
        jax.ShapeDtypeStruct((B, T, K), jnp.int32),
        jax.ShapeDtypeStruct((B, T, K), jnp.uint8),
    )

    def forward(y, s):
        return jax.pure_callback(
            lambda a, b: _host_decode(a, b, options, lib_path), trace_shapes, y, s, **_CALLBACK_BATCHING
        )

    def backward(parents, tokens, lengths, from_logprob, s, g):
        return jax.pure_callback(
            lambda *args: _host_backward(*args, V, options, lib_path),
            jax.ShapeDtypeStruct((B, T, K, V), jnp.float32),
            parents,
            tokens,
            lengths,
            from_logprob,
            s,
            g,
            **_CALLBACK_BATCHING,
        )

    @jax.custom_vjp
    def scores(y, s):
        return forward(y, s)[0]

    def scores_fwd(y, s):
        final, *trace = forward(y, s)
        return final, (*trace, s)

    def scores_bwd(residuals, g):
        *trace, s = residuals
        return backward(*trace, s, g), None

    scores.defvjp(scores_fwd, scores_bwd)
    # Steps travel as an array (T when not given), so one code path serves both.
    s = jnp.full((B,), T, dtype=jnp.int32) if steps is None else jnp.asarray(steps, dtype=jnp.int32).reshape(B)
    out = scores(x, s)
    return out[0] if unbatched else out
