# SPDX-License-Identifier: MIT
"""PyTorch front end: autograd-aware beam search on CPU and CUDA."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch.autograd.function import once_differentiable

from . import _C  # CPU operators; always built.
from ._options import CUDA_MAX_BEAM, BeamOptions

try:  # CUDA operators; built when a CUDA toolkit is available at install time.
    from . import _C_cuda
except ImportError:  # pragma: no cover - depends on the build
    _C_cuda = None

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


def _ops_for(x: torch.Tensor):
    if x.device.type == "cpu":
        return _C
    if x.device.type == "cuda":
        if _C_cuda is None:
            raise RuntimeError(
                "beamgrad was installed without its CUDA operators, so CUDA tensors are not supported. "
                "Reinstall on a machine with the CUDA toolkit (nvcc) available, e.g. "
                "`BEAMGRAD_CUDA=1 pip install --no-build-isolation beamgrad`."
            )
        return _C_cuda
    raise RuntimeError(f"beamgrad supports CPU and CUDA tensors, got device {x.device}")


def _prepare(
    log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Validate and return (x [B,T,K,V] float32 contiguous, steps [B] int32 | None, unbatched)."""
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
    if x.device.type == "cuda" and K > CUDA_MAX_BEAM:
        raise ValueError(f"beam_size {K} exceeds the CUDA backend maximum of {CUDA_MAX_BEAM}")

    x = x.detach().to(torch.float32).contiguous()
    if options.validate_inputs and bool(((x != x) | (x == float("inf"))).any()):
        raise ValueError("log_probs contains NaN or +inf (pass validate_inputs=False to skip this check)")

    steps_t: torch.Tensor | None = None
    if steps is not None:
        steps_t = torch.as_tensor(steps, device=x.device).to(torch.int32).contiguous()
        if steps_t.dim() != 1 or steps_t.numel() != B:
            raise ValueError(f"steps must have shape [B] = [{B}], got {tuple(steps_t.shape)}")
        if bool((steps_t < 1).any()) or bool((steps_t > T).any()):
            raise ValueError(f"steps entries must be in [1, T] = [1, {T}], got {steps_t.tolist()}")
    return x, steps_t, unbatched


class _FinalScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike):
        x, steps_t, unbatched = _prepare(log_probs, options, steps)
        ops = _ops_for(x)
        alpha = float(options.length_penalty_alpha)
        out = ops.decode(x, steps_t, options.eos_token, options.min_length, alpha)
        final_scores, _, _, tokens, parents, lengths, _, _, from_logprob = out
        ctx.ops = ops
        ctx.alpha = alpha
        ctx.vocab_size = x.size(3)
        ctx.unbatched = unbatched
        ctx.input_dtype = log_probs.dtype
        ctx.save_for_backward(parents, tokens, lengths, from_logprob, steps_t)
        return final_scores.squeeze(0) if unbatched else final_scores

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        parents, tokens, lengths, from_logprob, steps_t = ctx.saved_tensors
        g = grad_output.to(torch.float32)
        if ctx.unbatched:
            g = g.unsqueeze(0)
        grad = ctx.ops.backward(
            g.contiguous(), parents, tokens, lengths, from_logprob, steps_t, ctx.vocab_size, ctx.alpha
        )
        if ctx.unbatched:
            grad = grad.squeeze(0)
        return grad.to(ctx.input_dtype), None, None


def final_scores(log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike = None) -> torch.Tensor:
    """Run beam search and return the final beam scores, with surrogate gradients.

    Args:
        log_probs: Per-step log-probabilities, ``[T, K, V]`` or ``[B, T, K, V]``,
            on CPU or CUDA, in any floating dtype (computation is float32).
            ``log_probs[b, t, k]`` scores the next token for beam ``k`` at step
            ``t``, i.e. row ``k`` is the distribution conditioned on beam ``k``'s
            prefix. At ``t = 0`` only beam 0 is live.
        options: :class:`BeamOptions`; ``beam_size`` must equal ``K``.
        steps: Optional ``[B]`` number of steps to decode per example (batched
            input only). Defaults to ``T`` for every example.

    Returns:
        ``[K]`` or ``[B, K]`` final scores, best beam first.

    The forward pass is exact, hard beam search. The backward pass returns the
    gradient of each final score along the path of tokens that produced it,
    holding the beam selection fixed.
    """
    return _FinalScores.apply(log_probs, options, steps)


@torch.no_grad()
def decode(log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike = None) -> BeamSearchOutput:
    """Run beam search and return the full trace (no gradients).

    Takes the same arguments as :func:`final_scores`. See
    :class:`BeamSearchOutput` for the returned fields and
    :func:`backtrack` to recover each final beam's token sequence.
    """
    x, steps_t, unbatched = _prepare(log_probs, options, steps)
    B, T = x.shape[:2]
    decoded_steps = torch.full((B,), T, dtype=torch.long, device=x.device) if steps_t is None else steps_t.long()
    out = _ops_for(x).decode(x, steps_t, options.eos_token, options.min_length, float(options.length_penalty_alpha))
    final_scores_, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob = out
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
