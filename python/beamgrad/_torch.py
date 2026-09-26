# SPDX-License-Identifier: MIT
"""PyTorch front end: autograd-aware beam search on CPU and CUDA.

The work is done by operators registered with ``torch.library``:
``torch.ops.beamgrad.decode_ex`` (float32, float16 or bfloat16 log-probs or
logits) and ``torch.ops.beamgrad.decode_backward`` (the surrogate gradient of
any of its score outputs). ``decode`` and ``final_scores_backward``, their
float32, final-score-only predecessors, remain registered. They have fake
(meta) implementations, an autograd formula and a vmap rule, so they work
under ``torch.compile`` (without graph breaks), ``torch.export``, fake-tensor
tracing and ``torch.vmap``. :func:`final_scores` and :func:`decode`
differentiate through an ``autograd.Function`` with a separate
``setup_context``, which ``torch.func`` transforms (``grad``, ``vjp``,
``jacrev``, ``vmap`` of those) require.
"""

from __future__ import annotations

import importlib.util
import re
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import torch

from ._options import CUDA_MAX_BEAM, BeamOptions


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _check_build() -> None:
    """Fail early, with instructions, if the extensions target another PyTorch."""
    try:
        from ._build_info import TORCH_VERSION as built
    except ImportError:  # pragma: no cover - source checkouts without a build
        return
    if _major_minor(built) != _major_minor(torch.__version__):
        raise ImportError(
            f"beamgrad was compiled against PyTorch {built}, but PyTorch {torch.__version__} is installed. "
            "Its compiled operators only work with the PyTorch they were built with (this usually happens when "
            "pip builds beamgrad in an isolated environment with a different torch). Rebuild it against the "
            "installed PyTorch:\n    pip install --no-build-isolation --no-deps --force-reinstall beamgrad"
        )


_check_build()

try:
    from . import _C  # noqa: F401  defines the operators and their CPU kernels
except ImportError as exc:  # pragma: no cover - broken installs
    raise ImportError(
        f"beamgrad's compiled operators failed to load ({exc}). If PyTorch was upgraded or changed after "
        "beamgrad was installed, rebuild beamgrad against it:\n"
        "    pip install --no-build-isolation --no-deps --force-reinstall beamgrad"
    ) from exc

_C_cuda = None
if importlib.util.find_spec(f"{__package__}._C_cuda") is not None:  # built with a CUDA toolkit
    try:
        from . import _C_cuda  # registers the CUDA kernels
    except ImportError as exc:  # pragma: no cover - depends on the machine
        hint = ""
        try:
            from ._build_info import TORCH_CUDA_VERSION as built_cuda
        except ImportError:
            built_cuda = None
        if built_cuda and _major_minor(built_cuda)[:1] != (_major_minor(torch.version.cuda or "0.0") or (0,))[:1]:
            hint = (
                f" They were built for CUDA {built_cuda}, but PyTorch uses "
                f"{'CUDA ' + torch.version.cuda if torch.version.cuda else 'no CUDA'}: install the beamgrad wheel "
                "for your PyTorch's CUDA variant (docs/installation.md), or build from source."
            )
        warnings.warn(
            f"beamgrad's CUDA operators failed to load ({exc}); CUDA tensors are not supported.{hint}",
            RuntimeWarning,
            stacklevel=2,
        )

StepsLike = torch.Tensor | Sequence[int] | None


class BeamSearchOutput(NamedTuple):
    """Result of :func:`decode`. Leading ``[B]`` is absent for unbatched input.

    Per-step fields have shape ``[B, T, K]``: slot ``k`` at step ``t`` is the
    ``k``-th best hypothesis after step ``t``. Steps past an example's
    ``steps`` entry hold token/parent ``-1``, length ``0`` and ``-inf`` scores.
    """

    final_scores: torch.Tensor  # [B, K] length-penalised scores, best first
    final_raw_scores: torch.Tensor  # [B, K] cumulative log-probabilities
    final_lengths: torch.Tensor  # [B, K] hypothesis lengths (int64)
    tokens: torch.Tensor  # [B, T, K] token chosen at each step (int64)
    parents: torch.Tensor  # [B, T, K] parent beam at the previous step (int64)
    lengths: torch.Tensor  # [B, T, K] hypothesis length after each step (int64)
    scores: torch.Tensor  # [B, T, K] length-penalised ranking scores
    raw_scores: torch.Tensor  # [B, T, K] cumulative log-probabilities
    from_logprob: torch.Tensor  # [B, T, K] False for EOS carry-forward and padding slots
    steps: torch.Tensor  # [B] number of decoded steps per example (int64)


def cuda_available() -> bool:
    """True when the CUDA operators were built and a CUDA device is usable."""
    return _C_cuda is not None and torch.cuda.is_available() and bool(_C_cuda.device_available())


# ---------------------------------------------------------------------------
# Operator registrations: fake implementations, autograd, vmap
# ---------------------------------------------------------------------------


@torch.library.register_fake("beamgrad::decode")
def _decode_fake(
    log_probs,
    steps,
    eos_token,
    min_length,
    length_penalty_alpha,
    banned_tokens,
    no_repeat_ngram_size,
    repetition_penalty,
    validate,
):
    B, T, K, _ = log_probs.shape
    return (
        log_probs.new_empty((B, K), dtype=torch.float32),
        log_probs.new_empty((B, K), dtype=torch.float32),
        log_probs.new_empty((B, K), dtype=torch.int32),
        log_probs.new_empty((B, T, K), dtype=torch.int32),
        log_probs.new_empty((B, T, K), dtype=torch.int32),
        log_probs.new_empty((B, T, K), dtype=torch.int32),
        log_probs.new_empty((B, T, K), dtype=torch.float32),
        log_probs.new_empty((B, T, K), dtype=torch.float32),
        log_probs.new_empty((B, T, K), dtype=torch.uint8),
    )


@torch.library.register_fake("beamgrad::decode_ex")
def _decode_ex_fake(inputs, from_logits, steps, *options):
    B, T, K, _ = inputs.shape
    return (*_decode_fake(inputs, steps, *options), inputs.new_empty((B, T, K), dtype=torch.float32))


@torch.library.register_fake("beamgrad::decode_backward")
def _decode_backward_fake(
    grad_final_scores,
    grad_final_raw_scores,
    grad_scores,
    grad_raw_scores,
    parents,
    tokens,
    lengths,
    from_logprob,
    steps,
    vocab_size,
    length_penalty_alpha,
    logits,
    row_lse,
):
    B, T, K = parents.shape
    return parents.new_empty((B, T, K, vocab_size), dtype=torch.float32)


@torch.library.register_fake("beamgrad::decode_step")
def _decode_step_fake(
    log_probs,
    raw_scores,
    lengths,
    finished,
    prefixes,
    eos_token,
    min_length,
    length_penalty_alpha,
    banned_tokens,
    no_repeat_ngram_size,
    repetition_penalty,
    validate,
):
    B, K, _ = log_probs.shape
    return (
        log_probs.new_empty((B, K), dtype=torch.int32),
        log_probs.new_empty((B, K), dtype=torch.int32),
        log_probs.new_empty((B, K), dtype=torch.int32),
        log_probs.new_empty((B, K), dtype=torch.float32),
        log_probs.new_empty((B, K), dtype=torch.float32),
        log_probs.new_empty((B, K), dtype=torch.uint8),
        log_probs.new_empty((B, K), dtype=torch.uint8),
        log_probs.new_empty((B, K), dtype=torch.float32),
    )


@torch.library.register_fake("beamgrad::length_penalty")
def _length_penalty_fake(lengths, length_penalty_alpha):
    return lengths.new_empty(lengths.shape, dtype=torch.float32)


@torch.library.register_fake("beamgrad::final_scores_path_gradient")
def _final_scores_path_gradient_fake(
    grad_final, parents, tokens, lengths, from_logprob, steps, vocab_size, length_penalty_alpha
):
    return grad_final.new_empty(parents.shape, dtype=torch.float32)


@torch.library.register_fake("beamgrad::final_scores_backward")
def _final_scores_backward_fake(
    grad_final, parents, tokens, lengths, from_logprob, steps, vocab_size, length_penalty_alpha
):
    B, T, K = parents.shape
    return grad_final.new_empty((B, T, K, vocab_size), dtype=torch.float32)


def _decode_setup_context(ctx, inputs, output):
    log_probs, steps, _, _, length_penalty_alpha = inputs[:5]
    _, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob = output
    # Only the final scores carry gradients.
    ctx.mark_non_differentiable(final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob)
    ctx.save_for_backward(parents, tokens, lengths, from_logprob, steps)
    ctx.vocab_size = log_probs.shape[-1]
    ctx.length_penalty_alpha = length_penalty_alpha


def _decode_backward(ctx, grad_final_scores, *unused_grads):
    parents, tokens, lengths, from_logprob, steps = ctx.saved_tensors
    grad = torch.ops.beamgrad.final_scores_backward(
        grad_final_scores.to(torch.float32).contiguous(),
        parents,
        tokens,
        lengths,
        from_logprob,
        steps,
        ctx.vocab_size,
        ctx.length_penalty_alpha,
    )
    return grad, None, None, None, None, None, None, None, None


torch.library.register_autograd("beamgrad::decode", _decode_backward, setup_context=_decode_setup_context)


def _save_decode(ctx, x, from_logits, steps, length_penalty_alpha, output):
    """setup_context of decode_ex: its scores (final and per step, raw and penalised) are differentiable."""
    _, _, final_lengths, tokens, parents, lengths, _, _, from_logprob, row_lse = output
    ctx.mark_non_differentiable(final_lengths, tokens, parents, lengths, from_logprob, row_lse)
    ctx.set_materialize_grads(False)
    logits, lse = (x, row_lse) if from_logits else (None, None)
    ctx.save_for_backward(parents, tokens, lengths, from_logprob, steps, logits, lse)
    ctx.vocab_size = x.shape[-1]
    ctx.length_penalty_alpha = float(length_penalty_alpha)
    ctx.input_dtype = x.dtype


def _decode_grad(ctx, grad_final, grad_final_raw, _l, _t, _p, _n, grad_scores, grad_raw, _f, _r):
    """The input gradient of decode_ex's outputs (see _save_decode)."""
    grads = (grad_final, grad_final_raw, grad_scores, grad_raw)
    if all(g is None for g in grads):
        return None
    parents, tokens, lengths, from_logprob, steps, logits, row_lse = ctx.saved_tensors
    grad = torch.ops.beamgrad.decode_backward(
        *(None if g is None else g.to(torch.float32).contiguous() for g in grads),
        parents,
        tokens,
        lengths,
        from_logprob,
        steps,
        ctx.vocab_size,
        ctx.length_penalty_alpha,
        logits,
        row_lse,
    )
    return grad.to(ctx.input_dtype)


def _decode_ex_setup_context(ctx, inputs, output):
    x, from_logits, steps, _, _, length_penalty_alpha = inputs[:6]
    _save_decode(ctx, x, from_logits, steps, length_penalty_alpha, output)


def _decode_ex_backward(ctx, *grads):
    return (_decode_grad(ctx, *grads),) + (None,) * 9


torch.library.register_autograd("beamgrad::decode_ex", _decode_ex_backward, setup_context=_decode_ex_setup_context)


def _merge_vmap_dim(tensor, dim, size):
    """Moves (or adds) the vmapped dimension to the front and folds it into the batch."""
    tensor = tensor.movedim(dim, 0) if dim is not None else tensor.expand(size, *tensor.shape)
    return tensor.reshape(size * tensor.shape[1], *tensor.shape[2:])


def _decode_vmap(info, in_dims, log_probs, steps, *options):
    if any(dim is not None for dim in in_dims[2:]):
        raise NotImplementedError("beamgrad: vmap over banned_tokens is not supported")
    n = info.batch_size
    x = _merge_vmap_dim(log_probs, in_dims[0], n)
    s = None if steps is None else _merge_vmap_dim(steps, in_dims[1], n)
    outputs = torch.ops.beamgrad.decode(x, s, *options)
    return tuple(o.reshape(n, o.shape[0] // n, *o.shape[1:]) for o in outputs), (0,) * len(outputs)


def _decode_ex_vmap(info, in_dims, inputs, from_logits, steps, *options):
    if any(dim is not None for dim in in_dims[3:]):
        raise NotImplementedError("beamgrad: vmap over banned_tokens is not supported")
    n = info.batch_size
    x = _merge_vmap_dim(inputs, in_dims[0], n)
    s = None if steps is None else _merge_vmap_dim(steps, in_dims[2], n)
    outputs = torch.ops.beamgrad.decode_ex(x, from_logits, s, *options)
    return tuple(o.reshape(n, o.shape[0] // n, *o.shape[1:]) for o in outputs), (0,) * len(outputs)


def _decode_backward_vmap(info, in_dims, *args):
    n = info.batch_size
    # Every tensor argument, batched or not, is folded into the batch dimension.
    merged = [
        a if not isinstance(a, torch.Tensor) else _merge_vmap_dim(a, d, n) for a, d in zip(args, in_dims, strict=True)
    ]
    grad = torch.ops.beamgrad.decode_backward(*merged)
    return grad.reshape(n, grad.shape[0] // n, *grad.shape[1:]), 0


def _final_scores_backward_vmap(
    info, in_dims, grad_final, parents, tokens, lengths, from_logprob, steps, vocab_size, length_penalty_alpha
):
    n = info.batch_size
    tensors = [grad_final, parents, tokens, lengths, from_logprob]
    merged = [_merge_vmap_dim(t, d, n) for t, d in zip(tensors, in_dims[:5], strict=True)]
    s = None if steps is None else _merge_vmap_dim(steps, in_dims[5], n)
    grad = torch.ops.beamgrad.final_scores_backward(*merged, s, vocab_size, length_penalty_alpha)
    return grad.reshape(n, grad.shape[0] // n, *grad.shape[1:]), 0


def _length_penalty_vmap(info, in_dims, lengths, length_penalty_alpha):
    # Elementwise: the batched lengths go through as they are.
    return torch.ops.beamgrad.length_penalty(lengths, length_penalty_alpha), in_dims[0]


if hasattr(torch.library, "register_vmap"):  # PyTorch 2.5+
    torch.library.register_vmap("beamgrad::decode", _decode_vmap)
    torch.library.register_vmap("beamgrad::final_scores_backward", _final_scores_backward_vmap)
    torch.library.register_vmap("beamgrad::length_penalty", _length_penalty_vmap)
    torch.library.register_vmap("beamgrad::decode_ex", _decode_ex_vmap)
    torch.library.register_vmap("beamgrad::decode_backward", _decode_backward_vmap)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_NATIVE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _prepare(
    log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    """Check shapes and return (x [B,T,K,V], steps [B] | None, banned [V] | None, unbatched).

    ``x`` keeps a float16 or bfloat16 dtype (the operators read those rows as
    they are, without a float32 copy); other floating dtypes become float32.

    Only static properties are checked here; values (NaN/+inf, the range of
    steps) are checked inside the operators, which keeps this traceable.
    """
    if not isinstance(options, BeamOptions):
        raise TypeError(f"options must be a beamgrad.BeamOptions, got {type(options).__name__}")
    if not isinstance(log_probs, torch.Tensor):
        raise TypeError("log_probs must be a torch.Tensor")
    if not log_probs.is_floating_point():
        raise TypeError(f"log_probs must be a floating-point tensor, got {log_probs.dtype}")
    if log_probs.dim() == 3:
        unbatched = True
        if steps is not None:
            raise ValueError("steps is only supported for batched [B, T, K, V] input")
        x = log_probs.unsqueeze(0)
    elif log_probs.dim() == 4:
        unbatched = False
        x = log_probs
    else:
        raise ValueError(f"log_probs must have shape [T, K, V] or [B, T, K, V], got {tuple(log_probs.shape)}")

    B, T, K, V = x.shape
    if min(B, T, K, V) == 0:
        raise ValueError(f"log_probs dimensions must be non-empty, got {tuple(log_probs.shape)}")
    if K != options.beam_size:
        raise ValueError(f"log_probs has {K} beams but options.beam_size is {options.beam_size}")
    if options.eos_token >= V:
        raise ValueError(f"eos_token {options.eos_token} is outside the vocabulary (size {V})")
    device = x.device.type
    if device == "cuda":
        if _C_cuda is None:
            raise RuntimeError(
                "beamgrad was installed without its CUDA operators, so CUDA tensors are not supported. "
                "Reinstall on a machine with the CUDA toolkit (nvcc) available, e.g. "
                "`BEAMGRAD_CUDA=1 pip install --no-build-isolation beamgrad`."
            )
        if K > CUDA_MAX_BEAM:
            raise ValueError(f"beam_size {K} exceeds the CUDA backend maximum of {CUDA_MAX_BEAM}")
    elif device != "cpu":
        raise RuntimeError(f"beamgrad supports CPU and CUDA tensors, got device {x.device}")

    if x.dtype not in _NATIVE_DTYPES:
        x = x.to(torch.float32)
    x = x.contiguous()

    steps_t: torch.Tensor | None = None
    if steps is not None:
        steps_t = torch.as_tensor(steps, device=x.device)
        if steps_t.is_floating_point() or steps_t.is_complex() or steps_t.dtype == torch.bool:
            raise TypeError(f"steps must hold integers, got {steps_t.dtype}")
        # Kept at 64 bits: the operators range-check before narrowing, so 2**32 + 1 is an error rather than 1.
        steps_t = steps_t.to(torch.int64)
        if steps_t.dim() != 1 or steps_t.shape[0] != B:
            raise ValueError(f"steps must have shape [B] = [{B}], got {tuple(steps_t.shape)}")

    ids = options.banned_ids(V)
    banned = None
    if ids is not None:
        # Scattered on the target device: a [V] Python list would cost milliseconds per call.
        index = torch.tensor(ids, dtype=torch.long, device=x.device)
        banned = torch.zeros(V, dtype=torch.uint8, device=x.device).index_fill_(0, index, 1)
    return x, steps_t, banned, unbatched


def _decode_op(
    x: torch.Tensor, from_logits: bool, steps_t: torch.Tensor | None, banned: torch.Tensor | None, options: BeamOptions
):
    return torch.ops.beamgrad.decode_ex(
        x,
        bool(from_logits),
        steps_t,
        options.eos_token,
        options.min_length,
        float(options.length_penalty_alpha),
        banned,
        options.no_repeat_ngram_size,
        float(options.repetition_penalty),
        options.validate_inputs,
    )


class _Decode(torch.autograd.Function):
    """The decode operator's outputs, with the surrogate gradients of its scores.

    ``torch.ops.beamgrad.decode_ex`` carries its own autograd formula, but
    ``torch.library`` implements it as an ``autograd.Function`` whose forward
    takes ``ctx``, which ``torch.func`` transforms reject. This function has
    the same backward and a separate ``setup_context``; its vmap rule is
    generated from the operators' own rules.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, from_logits, steps, banned, options):
        # final_scores, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob, row_lse
        return _decode_op(x, from_logits, steps, banned, options)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, from_logits, steps, _, options = inputs
        _save_decode(ctx, x, from_logits, steps, options.length_penalty_alpha, output)

    @staticmethod
    def backward(ctx, *grads):
        return _decode_grad(ctx, *grads), None, None, None, None


def length_penalty(lengths: torch.Tensor, alpha: float) -> torch.Tensor:
    """The GNMT length penalty ``((5 + max(length, 1)) / 6) ** alpha``, as float32.

    Bit for bit the value the search divides raw scores by, on CPU or CUDA, so
    a reference sequence's score can be put on the scale of the beams' scores:
    ``gold_log_prob / length_penalty(gold_length, options.length_penalty_alpha)``.
    """
    alpha = float(alpha)
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths)
    return torch.ops.beamgrad.length_penalty(lengths, alpha)


def _decode_outputs(
    x: torch.Tensor,
    from_logits: bool,
    steps: torch.Tensor | None,
    banned: torch.Tensor | None,
    options: BeamOptions,
):
    """The decode operator's outputs, with the scores' surrogate gradients when ``x`` needs them."""
    if torch.is_grad_enabled() and x.requires_grad:  # also true inside torch.func.grad, vjp and jacrev
        return _Decode.apply(x, from_logits, steps, banned, options)
    # Nothing to differentiate: the operator alone. Same values; and PyTorch 2.4's Dynamo
    # mis-traces an autograd.Function with a separate setup_context when no input requires grad.
    return _decode_op(x, from_logits, steps, banned, options)


def final_scores(
    log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike = None, *, from_logits: bool = False
) -> torch.Tensor:
    """Run beam search and return the final beam scores, with surrogate gradients.

    Args:
        log_probs: Per-step log-probabilities, ``[T, K, V]`` or ``[B, T, K, V]``,
            on CPU or CUDA, in any floating dtype. float16 and bfloat16 are read
            as they are; the computation is float32.
            ``log_probs[b, t, k]`` scores the next token for beam ``k`` at step
            ``t``, i.e. row ``k`` is the distribution conditioned on beam ``k``'s
            prefix. At ``t = 0`` only beam 0 is live.
        options: :class:`BeamOptions`; ``beam_size`` must equal ``K``.
        steps: Optional ``[B]`` number of steps to decode per example (batched
            input only), each in ``[1, T]``. Defaults to ``T`` for every example.
        from_logits: The rows are logits rather than log-probabilities. Each row
            the search reads is normalised on the fly (``x - logsumexp(row)``,
            with the library's deterministic logsumexp), so the search is the
            one over ``log_softmax(logits)`` without materialising it, and the
            gradient flows through the log-softmax to the logits.

    Returns:
        ``[K]`` or ``[B, K]`` final scores (float32), best beam first.

    The forward pass is exact, hard beam search. The backward pass returns the
    gradient of each final score along the path of tokens that produced it,
    holding the beam selection fixed.
    """
    x, steps_t, banned, unbatched = _prepare(log_probs, options, steps)
    scores = _decode_outputs(x, from_logits, steps_t, banned, options)[0]
    return scores.squeeze(0) if unbatched else scores


def decode(
    log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike = None, *, from_logits: bool = False
) -> BeamSearchOutput:
    """Run beam search and return the full trace.

    Takes the same arguments as :func:`final_scores`. See
    :class:`BeamSearchOutput` for the returned fields and
    :func:`backtrack` to recover each final beam's token sequence.

    When ``log_probs`` requires grad, the scores (``final_scores``,
    ``final_raw_scores``, ``scores``, ``raw_scores``) are differentiable: each
    score's gradient flows along the path of tokens that produced it, holding
    the beam selection fixed, as for :func:`final_scores` (through the
    log-softmax with ``from_logits``).
    """
    x, steps_t, banned, unbatched = _prepare(log_probs, options, steps)
    B, T = x.shape[:2]
    out = _decode_outputs(x, from_logits, steps_t, banned, options)
    final_scores_, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob, _ = out
    decoded_steps = torch.full((B,), T, dtype=torch.long, device=x.device) if steps_t is None else steps_t.long()
    result = BeamSearchOutput(
        final_scores=final_scores_,
        final_raw_scores=final_raw,
        final_lengths=final_lengths.long(),
        tokens=tokens.long(),
        parents=parents.long(),
        lengths=lengths.long(),
        scores=scores,
        raw_scores=raw_scores,
        from_logprob=from_logprob.bool(),
        steps=decoded_steps,
    )
    if unbatched:
        result = BeamSearchOutput(*(t.squeeze(0) for t in result))
    return result


def backtrack(output: BeamSearchOutput) -> torch.Tensor:
    """Token sequences of the final beams: ``[K, T]`` or ``[B, K, T]``.

    Entry ``[b, k, t]`` is the token final beam ``k`` emitted at step ``t``,
    following parent pointers back from the last step. Finished beams repeat
    their EOS token after it was first produced (their carry-forward slots);
    steps beyond an example's length, and dead beams, hold ``-1``.
    """
    tokens, parents, steps = output.tokens, output.parents, output.steps
    unbatched = tokens.dim() == 2
    if unbatched:
        tokens, parents, steps = tokens.unsqueeze(0), parents.unsqueeze(0), steps.reshape(1)
    B, T, K = tokens.shape
    sequences = torch.full((B, K, T), -1, dtype=torch.long, device=tokens.device)
    beam = torch.arange(K, device=tokens.device).expand(B, K).clone()
    none = torch.full_like(beam, -1)
    for t in range(T - 1, -1, -1):
        decoded = (t < steps).unsqueeze(1)  # padding steps leave the beam index unchanged
        alive = (beam >= 0) & decoded
        index = beam.clamp(min=0)
        step_tokens = tokens[:, t, :].gather(1, index)
        step_parents = parents[:, t, :].gather(1, index)
        sequences[:, :, t] = torch.where(alive, step_tokens, none)
        beam = torch.where(decoded, torch.where(beam >= 0, step_parents, none), beam)
    return sequences.squeeze(0) if unbatched else sequences
