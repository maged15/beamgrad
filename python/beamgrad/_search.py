# SPDX-License-Identifier: MIT
"""Beam search driving a model: :func:`beam_search`.

:func:`beamgrad.final_scores` scores a precomputed ``[B, T, K, V]`` tensor, but
an autoregressive model can only produce the rows of step ``t`` once it knows
which prefixes survived step ``t - 1``. :func:`beam_search` runs that loop: it
asks a step function for each step's next-token log-probabilities given the
current beams, selects the next beams with the native engine, and keeps the
rows' autograd graph, so the final scores are differentiable with respect to
whatever produced the rows (typically the model's parameters). Each step is one
``torch.ops.beamgrad.decode_step`` call, which advances an explicit beam state
exactly as the full decode would.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from ._options import INT32_MAX, BeamOptions
from ._torch import BeamSearchOutput


class BeamState(NamedTuple):
    """The beams entering a step of :func:`beam_search` (``B`` examples, ``K`` beams).

    Slot ``k`` holds the ``k``-th best hypothesis so far. Beams are re-ranked
    every step, so slot ``k`` usually continues a different hypothesis than slot
    ``k`` did one step earlier: ``parents[b, k]`` is the slot it came from, the
    index to reorder any per-beam cache by (for example a transformer's
    key/value cache). At step 0 every slot holds the empty hypothesis and only
    slot 0 is read.

    Dead slots (fewer than ``K`` hypotheses exist) have parent and token ``-1``
    and a ``-inf`` score; the rows the step function returns for them, and for
    finished beams, are never read, so any values will do there.
    """

    step: int
    sequences: torch.Tensor  # [B, K, step] int64 tokens of each hypothesis; -1 past its length and for dead slots
    parents: torch.Tensor  # [B, K] int64 slot at step - 1 that beam k continues; -1 at step 0 and for dead slots
    tokens: torch.Tensor  # [B, K] int64 token beam k emitted at step - 1; -1 at step 0 and for dead slots
    lengths: torch.Tensor  # [B, K] int64 hypothesis lengths
    scores: torch.Tensor  # [B, K] float32 length-penalised scores (no gradient)
    finished: torch.Tensor  # [B, K] bool: the hypothesis ended with EOS
    active: torch.Tensor  # [B, K] bool: live and not finished (the rows this step reads)


class BeamSearchResult(NamedTuple):
    """Result of :func:`beam_search`; ``T`` is the number of steps run."""

    scores: torch.Tensor  # [B, K] final length-penalised scores, best first; differentiable
    sequences: torch.Tensor  # [B, K, max_steps] each final beam's tokens (int64), -1 past its length
    lengths: torch.Tensor  # [B, K] int64 hypothesis lengths
    raw_scores: torch.Tensor  # [B, K] cumulative log-probabilities (no gradient)
    trace: BeamSearchOutput  # the per-step trace, as decode(torch.stack(step_log_probs, 1), options) returns it
    step_log_probs: tuple[torch.Tensor, ...]  # the T [B, K, V] tensors step_fn returned (not copied)


StepFunction = Callable[[BeamState], torch.Tensor]


class _PathScores(torch.autograd.Function):
    """The final scores of a stepped search, differentiable w.r.t. every step's rows.

    Forward returns the scores the engine computed while stepping; backward is
    the final-score gradient of decode() on the stacked rows, split into steps.
    The rows themselves are never copied or stacked.
    """

    @staticmethod
    def forward(final_scores, parents, tokens, lengths, from_logprob, alpha, *rows):
        return final_scores.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, parents, tokens, lengths, from_logprob, alpha, *rows = inputs
        ctx.save_for_backward(parents, tokens, lengths, from_logprob)
        ctx.alpha = alpha
        ctx.vocab_size = rows[0].shape[-1]
        ctx.dtypes = [row.dtype for row in rows]

    @staticmethod
    def backward(ctx, grad_final_scores):
        parents, tokens, lengths, from_logprob = ctx.saved_tensors
        grad = torch.ops.beamgrad.final_scores_backward(
            grad_final_scores.to(torch.float32).contiguous(),
            parents,
            tokens,
            lengths,
            from_logprob,
            None,
            ctx.vocab_size,
            ctx.alpha,
        )  # [B, T, K, V]
        rows = tuple(grad[:, t].to(dtype) for t, dtype in enumerate(ctx.dtypes))
        return (None, None, None, None, None, None, *rows)


def _check_int(name: str, value, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= INT32_MAX:
        raise ValueError(f"{name} must be an int in [{low}, 2**31 - 1], got {value!r}")


def beam_search(
    step_fn: StepFunction,
    options: BeamOptions,
    max_steps: int,
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> BeamSearchResult:
    """Beam search over a model's next-token distributions, with surrogate gradients.

    Each step calls ``step_fn(state)`` with the current :class:`BeamState` and
    expects ``[B, K, V]`` next-token log-probabilities: row ``k`` is the
    distribution that follows beam ``k``'s hypothesis (``state.sequences[b,
    k]``). The next beams are chosen exactly as :func:`beamgrad.decode` would
    choose them from the stacked rows, on the rows' device (CPU or CUDA). The
    search stops after ``max_steps`` steps, or earlier once every beam has
    finished.

    The returned ``scores`` equal :func:`beamgrad.final_scores` of the stacked
    rows, with the same gradient: each beam's score is differentiated along its
    own path, with the beam selection held fixed, and the gradient reaches every
    tensor the rows were computed from. The rows are never stacked or copied;
    under ``torch.no_grad()`` each step's rows can be freed as soon as the next
    step starts. Typical step function for a Hugging Face causal LM::

        def step(beams):
            nonlocal cache
            if beams.step == 0:
                out = model(prompt.repeat_interleave(K, 0), use_cache=True)
            else:
                order = (torch.arange(B)[:, None] * K + beams.parents.clamp(min=0)).flatten()
                cache.reorder_cache(order)
                out = model(beams.tokens.clamp(min=0).view(B * K, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            return out.logits[:, -1].log_softmax(-1).view(B, K, -1)

    Args:
        step_fn: Maps the beams entering a step to that step's ``[B, K, V]``
            log-probabilities (any floating dtype; a new tensor every step).
        options: :class:`BeamOptions`; ``beam_size`` is ``K``.
        max_steps: Maximum number of decoding steps ``T``.
        batch_size: Number of examples ``B``.
        device: Device of the state tensors passed to ``step_fn``. By default
            they start on the CPU and follow the device of the first rows.

    Returns:
        :class:`BeamSearchResult`.
    """
    if not isinstance(options, BeamOptions):
        raise TypeError(f"options must be a beamgrad.BeamOptions, got {type(options).__name__}")
    _check_int("max_steps", max_steps, 1)
    _check_int("batch_size", batch_size, 1)
    B, K = batch_size, options.beam_size
    fixed_device = torch.device(device) if device is not None else None
    state_device = fixed_device or torch.device("cpu")

    raw = torch.full((B, K), float("-inf"), device=state_device)
    raw[:, 0] = 0.0
    lengths = torch.zeros((B, K), dtype=torch.int32, device=state_device)
    finished = torch.zeros((B, K), dtype=torch.uint8, device=state_device)
    scores = raw.clone()
    sequences = torch.empty((B, K, 0), dtype=torch.int64, device=state_device)
    parents = torch.full((B, K), -1, dtype=torch.int64, device=state_device)
    tokens = parents.clone()

    rows: list[torch.Tensor] = []
    trace: list[tuple[torch.Tensor, ...]] = []
    vocab_size = 0
    banned: torch.Tensor | None = None
    for t in range(max_steps):
        active = torch.isfinite(raw) & (finished == 0)
        if t > 0 and not bool(active.any()):
            break  # every beam has finished: later steps would only carry them forward
        state = BeamState(t, sequences, parents, tokens, lengths.long(), scores, finished.bool(), active)
        lp = step_fn(state)
        if not isinstance(lp, torch.Tensor) or not lp.is_floating_point():
            raise TypeError("step_fn must return a floating-point tensor of log-probabilities")
        if lp.dim() != 3 or lp.shape[0] != B or lp.shape[1] != K:
            raise ValueError(f"step_fn must return [B, K, V] = [{B}, {K}, V] log-probabilities, got {tuple(lp.shape)}")
        if t == 0:
            vocab_size = lp.shape[2]
            if fixed_device is None and lp.device != state_device:
                state_device = lp.device
                raw, lengths, finished, scores = (x.to(state_device) for x in (raw, lengths, finished, scores))
                sequences, parents, tokens = (x.to(state_device) for x in (sequences, parents, tokens))
            if options.eos_token >= vocab_size:
                raise ValueError(f"eos_token {options.eos_token} is outside the vocabulary (size {vocab_size})")
            ids = options.banned_ids(vocab_size)
            if ids is not None:
                index = torch.tensor(ids, dtype=torch.long, device=state_device)
                banned = torch.zeros(vocab_size, dtype=torch.uint8, device=state_device).index_fill_(0, index, 1)
        elif lp.shape[2] != vocab_size:
            raise ValueError(f"step_fn returned {lp.shape[2]} vocabulary entries at step {t}, {vocab_size} before")
        if lp.device != state_device:
            raise ValueError(
                f"step_fn returned log-probs on {lp.device}, but the beam state is on {state_device} (see `device`)"
            )
        rows.append(lp)

        (
            step_tokens,
            step_parents,
            lengths,
            scores,
            raw,
            from_logprob,
            finished,
            final,
        ) = torch.ops.beamgrad.decode_step(
            lp.detach().to(torch.float32).contiguous(),
            raw,
            lengths,
            finished,
            sequences.to(torch.int32),
            options.eos_token,
            options.min_length,
            float(options.length_penalty_alpha),
            banned,
            options.no_repeat_ngram_size,
            float(options.repetition_penalty),
            options.validate_inputs,
        )
        trace.append((step_tokens, step_parents, lengths, scores, raw, from_logprob))

        # Each new hypothesis: its parent's tokens, plus the token it emitted
        # (none for a carried-forward finished beam or a dead slot).
        parents, tokens = step_parents.long(), step_tokens.long()
        source = parents.clamp(min=0)[..., None].expand(B, K, t)
        emitted = torch.where(from_logprob.bool(), tokens, -1)
        sequences = torch.cat([sequences.gather(1, source), emitted[..., None]], dim=2)
        sequences = torch.where((parents >= 0)[..., None], sequences, -1)

    T = len(rows)
    if T < max_steps:  # stopped early: same shape as a search that ran every step
        sequences = torch.nn.functional.pad(sequences, (0, max_steps - T), value=-1)
    step_tokens, step_parents, step_lengths, step_scores, step_raw, step_flp = (
        torch.stack(field, dim=1) for field in zip(*trace, strict=True)
    )
    scores = _PathScores.apply(
        final, step_parents, step_tokens, step_lengths, step_flp, float(options.length_penalty_alpha), *rows
    )
    output = BeamSearchOutput(
        final_scores=final,
        final_raw_scores=raw,
        final_lengths=lengths.long(),
        tokens=step_tokens.long(),
        parents=step_parents.long(),
        lengths=step_lengths.long(),
        scores=step_scores,
        raw_scores=step_raw,
        from_logprob=step_flp.bool(),
        steps=torch.full((B,), T, dtype=torch.long, device=state_device),
    )
    return BeamSearchResult(
        scores=scores,
        sequences=sequences,
        lengths=lengths.long(),
        raw_scores=raw,
        trace=output,
        step_log_probs=tuple(rows),
    )
