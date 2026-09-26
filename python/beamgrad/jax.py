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

from ._ctypes import (
    DTYPE_F32,
    DBSBackwardInputsC,
    DBSDecodeOutputsExC,
    check,
    constraints_to_c,
    load_library,
    options_to_c,
)
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


def _host_decode(x, steps, options: BeamOptions, from_logits: bool, lib_path: str | None):
    x = np.asarray(x, dtype=np.float32)
    lead, (T, K, V) = x.shape[:-3], x.shape[-3:]
    n = math.prod(lead)
    x = np.ascontiguousarray(x.reshape(n, T, K, V))
    steps = _fold_steps(steps, lead, n)
    ids = options.banned_ids(V)
    banned = None
    if ids is not None:
        banned = np.zeros(V, dtype=np.uint8)
        banned[list(ids)] = 1
    constraints = constraints_to_c(options, banned)

    final = np.empty((n, K), dtype=np.float32)
    parents = np.empty((n, T, K), dtype=np.int32)
    tokens = np.empty((n, T, K), dtype=np.int32)
    lengths = np.empty((n, T, K), dtype=np.int32)
    from_logprob = np.empty((n, T, K), dtype=np.uint8)
    row_lse = np.empty((n, T, K), dtype=np.float32)
    out = DBSDecodeOutputsExC()
    out.base.final_scores = _ptr(final)
    out.base.parents = _ptr(parents)
    out.base.tokens = _ptr(tokens)
    out.base.lengths = _ptr(lengths)
    out.base.from_logprob = _ptr(from_logprob)
    out.row_lse = _ptr(row_lse)

    decoder = _Decoder(options, lib_path)
    status = decoder.lib.dbs_decode_batch_into_ex(
        decoder.handle,
        _ptr(x),
        DTYPE_F32,
        int(from_logits),
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
        row_lse.reshape(*lead, T, K),
    )


def _host_backward(
    parents,
    tokens,
    lengths,
    from_logprob,
    row_lse,
    logits,
    steps,
    g,
    vocab_size: int,
    options: BeamOptions,
    from_logits: bool,
    lib_path: str | None,
):
    parents = np.asarray(parents, dtype=np.int32)
    lead, (T, K) = parents.shape[:-2], parents.shape[-2:]
    n = math.prod(lead)

    def flat(a, dtype, shape=(T, K)):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=dtype), (*lead, *shape)).reshape(n, *shape))

    trace = [flat(parents, np.int32), flat(tokens, np.int32), flat(lengths, np.int32), flat(from_logprob, np.uint8)]
    g = flat(g, np.float32, (K,))
    steps = _fold_steps(steps, lead, n)
    grad = np.zeros((n, T, K, vocab_size), dtype=np.float32)

    inputs = DBSBackwardInputsC()
    inputs.batch_size, inputs.steps, inputs.vocab_size = n, T, vocab_size
    inputs.steps_per_example = _ptr(steps)
    inputs.parents, inputs.tokens, inputs.lengths, inputs.from_logprob = (_ptr(a) for a in trace)
    inputs.grad_final_scores = _ptr(g)
    if from_logits:  # the log-softmax's part of the gradient needs the logits and their logsumexp
        logits = flat(logits, np.float32, (T, K, vocab_size))
        row_lse = flat(row_lse, np.float32)
        inputs.logits, inputs.logits_type, inputs.row_lse = _ptr(logits), DTYPE_F32, _ptr(row_lse)

    decoder = _Decoder(options, lib_path)
    status = decoder.lib.dbs_backward_batch_into_ex(decoder.handle, ctypes.byref(inputs), 0, _ptr(grad))
    check(decoder.lib, decoder.handle, status)
    return grad.reshape(*lead, T, K, vocab_size)


def _steps_array(steps, batch: int, max_steps: int):
    """``[B]`` int32 step counts, range-checked before they are narrowed to 32 bits.

    Concrete values outside ``[1, T]`` raise here. Traced ones (under ``jit``)
    are clipped to 0 or ``T + 1``, which the library then rejects, so that
    ``2**32 + 1`` cannot wrap around to 1.
    """
    if isinstance(steps, jax.Array):  # includes tracers
        if not jnp.issubdtype(steps.dtype, jnp.integer):
            raise TypeError(f"steps must hold integers, got {steps.dtype}")
        return jnp.clip(steps, 0, max_steps + 1).astype(jnp.int32).reshape(batch)
    host = np.asarray(steps)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError(f"steps must hold integers, got {host.dtype}")
    host = host.reshape(batch)
    bad = np.flatnonzero((host < 1) | (host > max_steps))
    if bad.size:
        raise ValueError(f"steps[{bad[0]}] = {host[bad[0]]} must be in [1, {max_steps}]")
    return jnp.asarray(host.astype(np.int32))


def final_scores(
    log_probs, options: BeamOptions, steps=None, lib_path: str | None = None, *, from_logits: bool = False
):
    """Final beam scores for ``[T, K, V]`` or ``[B, T, K, V]`` log-probabilities.

    Same semantics, options and gradients as :func:`beamgrad.final_scores`,
    including ``steps`` (``[B]`` steps per example, batched input only) and
    ``from_logits`` (the rows are logits, normalised on the fly, with the
    gradient through the log-softmax). NaN or ``+inf`` in a row the search
    reads fails the call when ``options.validate_inputs`` is set (inside
    ``jit``, JAX reports the error raised by the host callback).
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
    options.banned_ids(V)  # checks the banned ids against the vocabulary
    x = x.astype(jnp.float32)  # differentiable: gradients return in the input dtype

    from_logits = bool(from_logits)
    trace_shapes = (
        jax.ShapeDtypeStruct((B, K), jnp.float32),
        jax.ShapeDtypeStruct((B, T, K), jnp.int32),
        jax.ShapeDtypeStruct((B, T, K), jnp.int32),
        jax.ShapeDtypeStruct((B, T, K), jnp.int32),
        jax.ShapeDtypeStruct((B, T, K), jnp.uint8),
        jax.ShapeDtypeStruct((B, T, K), jnp.float32),  # row_lse (zeros without logits)
    )

    def forward(y, s):
        return jax.pure_callback(
            lambda a, b: _host_decode(a, b, options, from_logits, lib_path), trace_shapes, y, s, **_CALLBACK_BATCHING
        )

    def backward(trace, logits, s, g):
        return jax.pure_callback(
            lambda *args: _host_backward(*args, V, options, from_logits, lib_path),
            jax.ShapeDtypeStruct((B, T, K, V), jnp.float32),
            *trace,
            logits,
            s,
            g,
            **_CALLBACK_BATCHING,
        )

    @jax.custom_vjp
    def scores(y, s):
        return forward(y, s)[0]

    def scores_fwd(y, s):
        final, *trace = forward(y, s)
        # The logits are kept only when the backward needs them.
        return final, (tuple(trace), y if from_logits else jnp.zeros((), jnp.float32), s)

    def scores_bwd(residuals, g):
        trace, logits, s = residuals
        return backward(trace, logits, s, g), None

    scores.defvjp(scores_fwd, scores_bwd)
    # Steps travel as an array (T when not given), so one code path serves both.
    s = jnp.full((B,), T, dtype=jnp.int32) if steps is None else _steps_array(steps, B, T)
    out = scores(x, s)
    return out[0] if unbatched else out
