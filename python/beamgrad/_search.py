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
from ._torch import BeamSearchOutput, StepsLike, _FinalScores, _prepare, backtrack, length_penalty


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
    step_log_probs: tuple[torch.Tensor, ...]  # the T [B, K, V] tensors step_fn returned, or () (see return_log_probs)


StepFunction = Callable[[BeamState], torch.Tensor]


class _PathScores(torch.autograd.Function):
    """The final scores of a stepped search, differentiable w.r.t. every step's rows.

    The inputs after ``alpha`` are, for each step, the log-probabilities of the
    entries its selected beams used (``[B, K]``, gathered from that step's rows
    right after the step). Forward returns the scores the engine computed while
    stepping. Backward computes the final-score gradient of each of those
    entries (``final_scores_path_gradient``) and returns it to the step's
    gather, whose own backward scatters it into that step's rows. So each
    step's dense ``[B, K, V]`` gradient only exists while autograd processes
    that step, rather than all ``T`` at once, and the result is bit for bit the
    gradient of ``final_scores`` on the stacked rows.
    """

    @staticmethod
    def forward(final_scores, parents, tokens, lengths, from_logprob, alpha, vocab_size, *picked):
        return final_scores.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, parents, tokens, lengths, from_logprob, alpha, vocab_size, *picked = inputs
        ctx.save_for_backward(parents, tokens, lengths, from_logprob)
        ctx.alpha = alpha
        ctx.vocab_size = vocab_size
        ctx.dtypes = [p.dtype for p in picked]

    @staticmethod
    def backward(ctx, grad_final_scores):
        parents, tokens, lengths, from_logprob = ctx.saved_tensors
        draws = torch.ops.beamgrad.final_scores_path_gradient(
            grad_final_scores.to(torch.float32).contiguous(),
            parents,
            tokens,
            lengths,
            from_logprob,
            None,
            ctx.vocab_size,
            ctx.alpha,
        )  # [B, T, K]
        per_step = tuple(draws[:, t].to(dtype) for t, dtype in enumerate(ctx.dtypes))
        return (None, None, None, None, None, None, None, *per_step)


def _check_int(name: str, value, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= INT32_MAX:
        raise ValueError(f"{name} must be an int in [{low}, 2**31 - 1], got {value!r}")


RescoreFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def beam_search(
    step_fn: StepFunction,
    options: BeamOptions,
    max_steps: int,
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    rescore_fn: RescoreFunction | None = None,
    return_log_probs: bool | None = None,
) -> BeamSearchResult:
    """Beam search over a model's next-token distributions, with surrogate gradients.

    Each step calls ``step_fn(state)`` with the current :class:`BeamState` and
    expects ``[B, K, V]`` next-token log-probabilities: row ``k`` is the
    distribution that follows beam ``k``'s hypothesis (``state.sequences[b,
    k]``). The next beams are chosen exactly as :func:`beamgrad.decode` would
    choose them from the stacked rows, on the rows' device (CPU or CUDA). The
    search stops after ``max_steps`` steps, or earlier once every beam has
    finished.

    The returned ``scores`` are the final scores the search computed, and their
    gradient is the path gradient of :func:`beamgrad.final_scores`: each beam's
    score is differentiated along its own path, with the beam selection held
    fixed. It is obtained in one of two ways:

    * **Through the steps** (default): the rows' autograd graphs are kept, and
      the gradient reaches every tensor the rows were computed from, bit for bit
      as ``final_scores`` of the stacked rows would give it. The rows are never
      stacked, and each step's dense gradient only exists while autograd
      processes that step.
    * **By re-scoring** (``rescore_fn``): the search runs without any autograd
      graph, so memory is that of inference, and the gradient comes from
      ``rescore_fn(sequences, lengths)``, which returns the ``[B, K, T]``
      log-probability of each final beam's tokens under teacher forcing (entries
      at or past a beam's length are ignored). Each beam's score is then
      differentiated through that single parallel pass, which, unlike
      incremental decoding with a key/value cache, works with the model's own
      gradient checkpointing. The gradient is the same function's (the search's
      log-probabilities, recomputed), up to floating-point differences between
      the two computations. :class:`beamgrad.hf.CausalLMRescorer` is one for
      Hugging Face causal LMs.

    Typical step function for a Hugging Face causal LM (or use
    :class:`beamgrad.hf.CausalLMStep`)::

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
        rescore_fn: Take gradients by re-scoring the final beams (see above).
        return_log_probs: Keep every step's rows in ``step_log_probs``. By
            default they are kept when the rows carry an autograd graph (which
            holds them anyway) and dropped otherwise, so that inference only
            ever holds one step's rows.

    Returns:
        :class:`BeamSearchResult`.
    """
    if not isinstance(options, BeamOptions):
        raise TypeError(f"options must be a beamgrad.BeamOptions, got {type(options).__name__}")
    _check_int("max_steps", max_steps, 1)
    _check_int("batch_size", batch_size, 1)
    if rescore_fn is None:
        return _search(step_fn, options, max_steps, batch_size, device, return_log_probs)
    with torch.no_grad():
        result = _search(step_fn, options, max_steps, batch_size, device, bool(return_log_probs))
    return _rescore(result, rescore_fn, options)


def sequence_scores(token_log_probs: torch.Tensor, lengths: torch.Tensor, options: BeamOptions) -> torch.Tensor:
    """Scores of given sequences, on the scale of the search's final scores.

    ``token_log_probs[..., t]`` is the log-probability of token ``t`` of a
    sequence given the tokens before it (under teacher forcing), and
    ``lengths[...]`` its length; entries at or past the length are ignored.
    Returns the summed log-probability divided by the length penalty, with the
    penalty the search uses, so for example a reference sequence can be
    compared with the beams in a structured-margin loss. Differentiable.
    """
    if not isinstance(options, BeamOptions):
        raise TypeError(f"options must be a beamgrad.BeamOptions, got {type(options).__name__}")
    T = token_log_probs.shape[-1]
    if tuple(lengths.shape) != tuple(token_log_probs.shape[:-1]):
        raise ValueError(f"lengths must have shape {tuple(token_log_probs.shape[:-1])}, got {tuple(lengths.shape)}")
    lengths = lengths.to(token_log_probs.device)
    on_path = torch.arange(T, device=token_log_probs.device) < lengths[..., None]
    if token_log_probs.dtype != torch.float64:
        token_log_probs = token_log_probs.float()
    raw = token_log_probs.masked_fill(~on_path, 0.0).sum(-1)
    return raw / length_penalty(lengths, options.length_penalty_alpha)


def search(log_probs: torch.Tensor, options: BeamOptions, steps: StepsLike = None) -> BeamSearchResult:
    """Beam search over precomputed ``[B, T, K, V]`` rows: scores and trace from one decode.

    The same search as :func:`beamgrad.final_scores` and :func:`beamgrad.decode`
    together, without decoding twice, returned as a :class:`BeamSearchResult`
    like :func:`beam_search`'s: ``scores`` are differentiable (the path
    gradient of ``final_scores``), and ``step_log_probs`` are the input's steps
    (views, not copies). Unbatched ``[T, K, V]`` input gives unbatched results.
    """
    x, steps_t, banned, unbatched = _prepare(log_probs, options, steps)
    B, T = x.shape[:2]
    final, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob = _FinalScores.apply(
        x, steps_t, banned, options
    )
    trace = BeamSearchOutput(
        final_scores=final.detach(),
        final_raw_scores=final_raw,
        final_lengths=final_lengths.long(),
        tokens=tokens.long(),
        parents=parents.long(),
        lengths=lengths.long(),
        scores=scores,
        raw_scores=raw_scores,
        from_logprob=from_logprob.bool(),
        steps=torch.full((B,), T, dtype=torch.long, device=x.device) if steps_t is None else steps_t.long(),
    )
    paths = backtrack(trace)  # [B, K, T]
    sequences = torch.where(torch.arange(T, device=x.device) < trace.final_lengths[..., None], paths, -1)
    rows = log_probs.unsqueeze(0) if unbatched else log_probs
    result = BeamSearchResult(
        scores=final,
        sequences=sequences,
        lengths=trace.final_lengths,
        raw_scores=final_raw,
        trace=trace,
        step_log_probs=tuple(rows.unbind(1)),
    )
    if unbatched:
        result = BeamSearchResult(
            scores=final.squeeze(0),
            sequences=sequences.squeeze(0),
            lengths=trace.final_lengths.squeeze(0),
            raw_scores=final_raw.squeeze(0),
            trace=BeamSearchOutput(*(t.squeeze(0) for t in trace)),
            step_log_probs=tuple(log_probs.unbind(0)),
        )
    return result


def _rescore(result: BeamSearchResult, rescore_fn: RescoreFunction, options: BeamOptions) -> BeamSearchResult:
    """Attach the gradient of the teacher-forced re-scoring of the final beams."""
    B, K, _ = result.sequences.shape
    T = result.trace.tokens.shape[1]  # steps the search ran
    log_probs = rescore_fn(result.sequences[..., :T], result.lengths)
    if not isinstance(log_probs, torch.Tensor) or tuple(log_probs.shape) != (B, K, T):
        shape = tuple(log_probs.shape) if isinstance(log_probs, torch.Tensor) else type(log_probs).__name__
        raise ValueError(f"rescore_fn must return [B, K, T] = [{B}, {K}, {T}] token log-probabilities, got {shape}")
    rescored = sequence_scores(log_probs, result.lengths, options)
    # The value is the search's own score; the gradient is the re-scoring's.
    delta = (rescored - rescored.detach()).to(result.scores.dtype)
    live = torch.isfinite(result.scores)
    scores = torch.where(live, result.scores + delta, result.scores)
    return result._replace(scores=scores)


def _search(
    step_fn: StepFunction,
    options: BeamOptions,
    max_steps: int,
    batch_size: int,
    device: torch.device | str | None,
    return_log_probs: bool | None,
) -> BeamSearchResult:
    B, K = batch_size, options.beam_size
    keep_rows = torch.is_grad_enabled() if return_log_probs is None else return_log_probs
    fixed_device = None
    if device is not None:
        # The device tensors report ("cuda:0" for "cuda"), so that it compares
        # equal to the rows' device.
        fixed_device = torch.empty(0, device=device).device
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
    picked: list[torch.Tensor] = []
    steps_run = 0
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
        if keep_rows:
            rows.append(lp)
        steps_run += 1

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
        # The entries this step's beams used, gathered now so that autograd
        # reaches (and releases) each step's gradient in turn during backward.
        entry = step_parents.clamp(min=0).long() * vocab_size + step_tokens.clamp(min=0).long()
        picked.append(lp.flatten(1).gather(1, entry))

        # Each new hypothesis: its parent's tokens, plus the token it emitted
        # (none for a carried-forward finished beam or a dead slot).
        parents, tokens = step_parents.long(), step_tokens.long()
        source = parents.clamp(min=0)[..., None].expand(B, K, t)
        emitted = torch.where(from_logprob.bool(), tokens, -1)
        sequences = torch.cat([sequences.gather(1, source), emitted[..., None]], dim=2)
        sequences = torch.where((parents >= 0)[..., None], sequences, -1)

    T = steps_run
    if T < max_steps:  # stopped early: same shape as a search that ran every step
        sequences = torch.nn.functional.pad(sequences, (0, max_steps - T), value=-1)
    step_tokens, step_parents, step_lengths, step_scores, step_raw, step_flp = (
        torch.stack(field, dim=1) for field in zip(*trace, strict=True)
    )
    scores = _PathScores.apply(
        final,
        step_parents,
        step_tokens,
        step_lengths,
        step_flp,
        float(options.length_penalty_alpha),
        vocab_size,
        *picked,
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
