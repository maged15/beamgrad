# SPDX-License-Identifier: MIT
"""Alternative gradient estimators: smooth quantities of an exact beam search.

The search itself is always exact and hard; these functions compute
differentiable quantities from its trace and the rows it read, so a loss can
use a smoother signal than the final scores' path gradient:

* :func:`path_scores`: every step's selected-beam scores, differentiable along
  each beam's path (the per-step version of the final scores).
* :func:`selected_softmax`: per-step softmax weights over the selected beams,
  ``softmax(scores[t] / temperature)``; its gradient reaches every selected
  beam, not only the winner.
* :func:`relaxed_topk`: a sigmoid "k-hot" relaxation of membership in the top
  ``K`` over a larger candidate pool, ``sigmoid((score - theta) / temperature)``
  with ``theta`` such that the weights sum to ``K``; its gradient also reaches
  candidates the search did not keep.

They take a :class:`beamgrad.BeamSearchResult` from :func:`beamgrad.search` or
from :func:`beamgrad.beam_search` through the steps (both keep the rows in
``step_log_probs``), run on the rows' device, and match the C library's
selected-weight and relaxed-pool surrogates (``dbs_backward``).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ._options import BeamOptions
from ._search import BeamSearchResult
from ._torch import length_penalty

__all__ = ["RelaxedTopK", "path_scores", "relaxed_topk", "selected_softmax"]

_NEG_GUARD = -1.0e30  # scores at or below this are padding, as in the C library


def _batched(result: BeamSearchResult):
    trace = result.trace
    rows = result.step_log_probs
    if not rows:
        raise ValueError(
            "the estimators need the log-probabilities of the rows the search read: use beamgrad.search() "
            "(on log_softmax(logits) rather than with from_logits=True), or beam_search() through the steps "
            "(not rescore_fn), with return_log_probs left on"
        )
    if trace.tokens.dim() == 2:  # unbatched search()
        trace = type(trace)(*(t[None] for t in trace))
        rows = tuple(r[None] for r in rows)
    return trace, rows


def _raw_scores(trace, rows) -> list[torch.Tensor]:
    """Selected beams' cumulative log-probabilities per step, [B, K] each, with gradients.

    The same float32 additions as the search, so the values equal trace.raw_scores.
    """
    raws: list[torch.Tensor] = []
    B, _, K = trace.tokens.shape
    for t, row in enumerate(rows):
        parents, tokens = trace.parents[:, t], trace.tokens[:, t]
        V = row.shape[-1]
        entry = row.flatten(1).gather(1, parents.clamp(min=0) * V + tokens.clamp(min=0)).float()
        parent_raw = torch.zeros(B, K, device=entry.device) if t == 0 else raws[-1].gather(1, parents.clamp(min=0))
        raw = torch.where(trace.from_logprob[:, t], parent_raw + entry, parent_raw)
        raws.append(torch.where(parents >= 0, raw, float("-inf")))
    return raws


def path_scores(result: BeamSearchResult, options: BeamOptions) -> torch.Tensor:
    """``[B, T, K]`` scores of the selected beams at every step, differentiable along their paths.

    Equal to ``result.trace.scores``; the gradient of ``[b, t, k]`` flows to the
    rows' entries on that beam's path up to step ``t``, scaled by its length
    penalty.
    """
    trace, rows = _batched(result)
    scores = []
    for t, raw in enumerate(_raw_scores(trace, rows)):
        penalty = length_penalty(trace.lengths[:, t], options.length_penalty_alpha)
        # Expanded beams are ranked by raw * (1 / penalty), carried ones by raw / penalty.
        scores.append(torch.where(trace.from_logprob[:, t], raw * (1.0 / penalty), raw / penalty))
    out = torch.stack(scores, 1)
    return out[0] if result.trace.tokens.dim() == 2 else out


def selected_softmax(result: BeamSearchResult, options: BeamOptions, temperature: float = 1.0) -> torch.Tensor:
    """``[B, T, K]`` weights ``softmax(scores[t] / temperature)`` over each step's selected beams.

    Dead slots get weight 0. The C library's selected-beam weights
    (``selected_temperature``), with the same gradient.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scores = path_scores(result, options)
    live = scores > _NEG_GUARD
    logits = torch.where(live, scores / temperature, float("-inf"))
    any_live = live.any(-1, keepdim=True)
    logits = torch.where(any_live, logits, torch.zeros_like(logits))
    return torch.softmax(logits, -1) * any_live


class _SoftTopK(torch.autograd.Function):
    """Sigmoid k-hot weights with theta found by bisection; implicit gradient through theta."""

    @staticmethod
    def forward(scores, k, temperature, tolerance, max_iters):
        valid = scores > _NEG_GUARD
        active = valid.sum(-1, keepdim=True)
        big = torch.finfo(scores.dtype).max
        # The same bisection as the C library's (soft_topk_inclusion): theta is
        # bisected as an offset from the best score, which keeps its precision
        # when the scores are in the thousands, and it stops once the sum is
        # within tolerance of k or theta is bracketed to tolerance * temperature
        # (or to a few ULPs of the offset, where halving no longer moves it).
        ref = torch.where(valid, scores, -big).amax(-1, keepdim=True)
        shifted = scores - ref
        lo = torch.where(valid, shifted, big).amin(-1, keepdim=True) - 80.0 * temperature
        hi = torch.full_like(lo, 80.0 * temperature)
        bracket = tolerance * temperature
        eps = torch.finfo(scores.dtype).eps
        done = active <= k  # every valid candidate is fully in
        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            s = torch.where(valid, torch.sigmoid((shifted - mid) / temperature), 0.0).sum(-1, keepdim=True)
            err = s - k
            converged = (err.abs() <= tolerance) | (hi - lo <= (4.0 * eps * mid.abs()).clamp(min=bracket))
            newly = converged & ~done
            lo = torch.where(newly, mid, torch.where(done, lo, torch.where(err > 0, mid, lo)))
            hi = torch.where(newly, mid, torch.where(done, hi, torch.where(err > 0, hi, mid)))
            done = done | converged
            if bool(done.all()):
                break
        offset = 0.5 * (lo + hi)
        weights = torch.where(valid, torch.sigmoid((shifted - offset) / temperature), 0.0)
        return torch.where(active <= k, valid.to(scores.dtype), weights)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, temperature, _, _ = inputs
        ctx.save_for_backward(output)
        ctx.temperature = temperature

    @staticmethod
    def backward(ctx, grad):
        (weights,) = ctx.saved_tensors
        a = weights * (1.0 - weights)
        denom = a.sum(-1, keepdim=True)
        center = (grad * a).sum(-1, keepdim=True) / denom.clamp(min=1e-30)
        d = a / ctx.temperature * (grad - center)
        return torch.where(denom > 1e-12, d, 0.0), None, None, None, None


class RelaxedTopK(NamedTuple):
    """Result of :func:`relaxed_topk` (``P`` = pool size)."""

    scores: torch.Tensor  # [B, T, P] the P best candidates' scores per step, best first (differentiable)
    weights: torch.Tensor  # [B, T, P] relaxed top-K membership, summing to K per step (differentiable)
    parents: torch.Tensor  # [B, T, P] int64 slot at step t - 1 each candidate extends; -1 for padding
    tokens: torch.Tensor  # [B, T, P] int64 token it appends (EOS for a finished beam carried forward); -1 for padding
    from_logprob: torch.Tensor  # [B, T, P] bool: an expansion (not a carried beam or padding)


def relaxed_topk(
    result: BeamSearchResult,
    options: BeamOptions,
    pool_multiplier: int = 8,
    temperature: float = 0.25,
    tolerance: float = 1e-4,
    max_iters: int = 48,
) -> RelaxedTopK:
    """Relaxed top-``K`` membership of each step's ``K * pool_multiplier`` best candidates.

    Every step's candidates (each live beam extended by each allowed token,
    plus finished beams carried forward, ranked as the search ranks them) are
    cut to the best ``P``; their weights ``sigmoid((score - theta) /
    temperature)``, with ``theta`` found by bisection so they sum to ``K``,
    are a smooth version of "kept by the search". The gradient comes from
    implicit differentiation through ``theta`` and reaches candidates that were
    not selected. ``parents``, ``tokens`` and ``from_logprob`` say which
    candidate each is, as in the C library's pool (the order of candidates
    with exactly equal scores may differ from it). N-gram blocking and
    repetition penalties are not supported.

    The bisection stops once a step's weights sum to ``K`` within
    ``tolerance``, or ``theta`` is bracketed to ``tolerance * temperature``
    (so each weight is within ``tolerance / 8`` of its value at the exact
    ``theta``), at any magnitude of the scores. From its initial bracket that
    takes at most ``log2(spread / (tolerance * temperature) + 160 /
    tolerance)`` halvings, with ``spread`` the range of the pool's scores:
    21 for the defaults and a pool a few units wide, so ``max_iters=48`` is
    only a safety cap.
    """
    if options.no_repeat_ngram_size > 0 or options.repetition_penalty > 1.0:
        raise NotImplementedError("relaxed_topk does not support no_repeat_ngram_size or repetition_penalty")
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not isinstance(pool_multiplier, int) or pool_multiplier < 1:
        raise ValueError(f"pool_multiplier must be a positive int, got {pool_multiplier!r}")
    trace, rows = _batched(result)
    B, T, K = trace.tokens.shape
    P = K * pool_multiplier
    raws = _raw_scores(trace, rows)
    alpha = options.length_penalty_alpha
    device = trace.tokens.device
    eos = torch.tensor(options.eos_tokens, dtype=trace.tokens.dtype, device=device)  # every EOS token
    banned = None
    pool_scores, pool_index = [], []
    for t, row in enumerate(rows):
        V = row.shape[-1]
        if t == 0:
            options.eos_ids(V)  # checks the EOS tokens against the vocabulary
        if banned is None and options.banned_tokens:
            banned = torch.zeros(V, dtype=torch.bool, device=device)
            banned[list(options.banned_ids(V))] = True
        if t == 0:
            parent_raw = torch.full((B, K), float("-inf"), device=device)
            parent_raw[:, 0] = 0.0
            parent_len = torch.zeros(B, K, dtype=torch.long, device=device)
            parent_done = torch.zeros(B, K, dtype=torch.bool, device=device)
            parent_tokens = None
        else:
            parent_raw = raws[t - 1]
            parent_len = trace.lengths[:, t - 1]
            # A beam is finished when its token is an EOS token: the one it emitted, or carries.
            parent_tokens = trace.tokens[:, t - 1]
            parent_done = torch.isin(parent_tokens, eos) & (trace.parents[:, t - 1] >= 0)
        live = torch.isfinite(parent_raw) & ~parent_done
        new_len = parent_len + 1
        lp = row.float()
        allowed = torch.isfinite(lp) & live[..., None]
        if banned is not None:
            allowed &= ~banned
        if eos.numel():
            allowed[..., eos] &= (new_len >= options.min_length)[..., None]
        inv = 1.0 / length_penalty(new_len, alpha)
        expanded = torch.where(allowed, (parent_raw[..., None] + lp) * inv[..., None], float("-inf"))
        carried = torch.where(
            torch.isfinite(parent_raw) & parent_done,
            parent_raw / length_penalty(parent_len.clamp(min=1), alpha),
            float("-inf"),
        )
        candidates = torch.cat([expanded.flatten(1), carried], dim=1)  # [B, K * V + K]
        best, index = candidates.topk(min(P, candidates.shape[1]), dim=1)
        if best.shape[1] < P:
            best = torch.nn.functional.pad(best, (0, P - best.shape[1]), value=float("-inf"))
            index = torch.nn.functional.pad(index, (0, P - index.shape[1]), value=-1)
        index = torch.where(best > _NEG_GUARD, index, -1)
        pool_scores.append(best)
        pool_index.append((index, K * V, parent_tokens))
    scores = torch.stack(pool_scores, 1)  # [B, T, P]
    # One bisection for every step at once. Each row's is independent, so the
    # weights are the same; but each iteration reads a convergence flag back from
    # the device, so this syncs once per iteration instead of T times.
    weights = _SoftTopK.apply(scores, K, temperature, tolerance, max_iters)
    parents, tokens, from_logprob = [], [], []
    for index, expansions, parent_tokens in pool_index:
        expanded = (index >= 0) & (index < expansions)
        V = expansions // K
        parent = torch.where(expanded, index // V, torch.where(index >= 0, index - expansions, -1))
        parents.append(parent)
        # A carried-forward candidate keeps its beam's EOS token (only possible after step 0).
        carried = parent_tokens.gather(1, parent.clamp(0, K - 1)) if parent_tokens is not None else index
        tokens.append(torch.where(expanded, index % V, torch.where(index >= 0, carried, -1)))
        from_logprob.append(expanded)
    out = RelaxedTopK(
        scores,
        weights,
        torch.stack(parents, 1),
        torch.stack(tokens, 1),
        torch.stack(from_logprob, 1),
    )
    if result.trace.tokens.dim() == 2:
        out = RelaxedTopK(*(x[0] for x in out))
    return out
