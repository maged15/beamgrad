# SPDX-License-Identifier: MIT
"""Sequence-level training objectives on the result of a beam search.

Both take a :class:`beamgrad.BeamSearchResult` (from :func:`beamgrad.beam_search`
or :func:`beamgrad.search`) and are differentiable through its ``scores``, so
their gradient reaches the model along the paths of the beams the search
actually found.

* :func:`structured_margin`: the reference must beat the best beam that is not
  the reference by a margin (beam-search optimisation, Wiseman & Rush 2016).
* :func:`minimum_risk`: expected cost of the beams under a softmax over their
  scores (minimum-risk training, Shen et al. 2016), with any cost, e.g.
  1 - sentence BLEU against the reference.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch

from ._options import BeamOptions
from ._search import BeamSearchResult

__all__ = ["minimum_risk", "structured_margin", "matches"]


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def _batched(result: BeamSearchResult) -> tuple[torch.Tensor, torch.Tensor]:
    scores, sequences = result.scores, result.sequences
    if scores.dim() == 1:  # unbatched search()
        scores, sequences = scores[None], sequences[None]
    return scores, sequences


def matches(sequences: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` bool: beam ``k`` of example ``b`` is exactly the reference.

    ``sequences`` is ``[B, K, T]`` and ``reference`` ``[B, T']``, both padded
    with ``-1`` after the sequence ends; ``T`` and ``T'`` may differ.
    """
    B, K, T = sequences.shape
    if reference.dim() != 2 or reference.shape[0] != B:
        raise ValueError(f"reference must be [B, T'] = [{B}, T'], got {tuple(reference.shape)}")
    reference = reference.to(sequences.device)
    if reference.shape[1] < T:
        reference = torch.nn.functional.pad(reference, (0, T - reference.shape[1]), value=-1)
    fits = torch.ones(B, dtype=torch.bool, device=sequences.device)
    if reference.shape[1] > T:
        fits = (reference[:, T:] < 0).all(1)  # longer references cannot be among the beams
        reference = reference[:, :T]
    return (sequences == reference[:, None]).all(-1) & fits[:, None]


def structured_margin(
    result: BeamSearchResult,
    reference: torch.Tensor,
    reference_scores: torch.Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
    *,
    eos_token: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Hinge loss: the reference must outscore the best non-reference beam by ``margin``.

    ``max(0, margin + best rival score - reference score)`` per example, where
    the rival is the highest-scoring beam that is not the reference. The loss
    is zero once the reference wins by the margin, whether or not the search
    found it; otherwise it raises the reference's score and lowers the rival's.

    A reference is recognised among the beams only if it is token for token
    the sequence the search would return. With ``eos_token >= 0`` a finished
    beam ends with the EOS it emitted, so **references must end with that
    EOS**, and the lengths given to :func:`beamgrad.sequence_scores` for
    ``reference_scores`` must count it. A reference without it never matches:
    the beam that is the reference then counts as its own rival, and the loss
    can never reach zero. Pass ``eos_token`` to be warned about such
    references.

    Args:
        result: The search's result; its ``scores`` carry the gradient.
        reference: ``[B, T']`` reference tokens, ``-1`` after the end.
        reference_scores: ``[B]`` the references' scores on the beams' scale,
            e.g. :func:`beamgrad.sequence_scores` of their teacher-forced
            token log-probabilities (differentiable).
        margin: Required score difference.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        eos_token: The search's ``options.eos_token`` (one token id or
            several). When given and ``>= 0``, a ``UserWarning`` names the
            reference rows that do not end with (one of) them (this reads the
            references on the host). ``None`` (the default) skips the check.
    """
    scores, sequences = _batched(result)
    B = scores.shape[0]
    eos = () if eos_token is None else BeamOptions(beam_size=1, eos_token=eos_token).eos_tokens
    if eos:
        _warn_missing_eos(reference, eos)
    reference_scores = torch.as_tensor(reference_scores)
    # [B, 1] would broadcast against the [B] rivals into a [B, B] loss.
    if tuple(reference_scores.shape) != (B,) and not (result.scores.dim() == 1 and reference_scores.dim() == 0):
        raise ValueError(f"reference_scores must have shape [B] = [{B}], got {tuple(reference_scores.shape)}")
    rivals = scores.masked_fill(matches(sequences, reference), float("-inf"))
    best_rival = rivals.amax(1)
    loss = torch.relu(margin + best_rival - reference_scores)
    # No rival (every beam is the reference or dead): nothing to separate.
    loss = torch.where(torch.isfinite(best_rival), loss, torch.zeros_like(loss))
    return _reduce(loss, reduction)


def _warn_missing_eos(reference: torch.Tensor, eos: tuple[int, ...]) -> None:
    if reference.dim() != 2:
        return  # matches() reports the shape
    lengths = (reference >= 0).sum(1)
    last = reference.gather(1, (lengths - 1).clamp(min=0)[:, None])[:, 0]
    ends = torch.isin(last, torch.tensor(eos, dtype=last.dtype, device=last.device))
    missing = ((lengths == 0) | ~ends).nonzero().flatten().tolist()
    if missing:
        shown = ", ".join(str(r) for r in missing[:10]) + (", ..." if len(missing) > 10 else "")
        named = f"eos_token {eos[0]}" if len(eos) == 1 else f"any of the EOS tokens {list(eos)}"
        warnings.warn(
            f"structured_margin: reference rows [{shown}] do not end with {named}. A finished beam "
            "ends with the EOS it emitted, so these references can never match a beam and their loss cannot "
            "reach zero; append the EOS (and count it in the sequence_scores lengths).",
            UserWarning,
            stacklevel=3,
        )


def minimum_risk(
    result: BeamSearchResult,
    costs: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Expected cost of the beams under ``softmax(scores / temperature)``.

    ``costs[b, k]`` is the cost of beam ``k`` (e.g. 1 - BLEU against the
    reference; no gradient needed). Dead beams (``-inf`` scores) get no
    probability and their costs are ignored, so any value, even NaN, will do
    there. Lower temperatures concentrate the distribution on the best
    beams; Shen et al. (2016) sharpen with ``temperature = 1 / 5e-3`` on
    unnormalised log-probabilities.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scores, _ = _batched(result)
    if tuple(costs.shape) != tuple(scores.shape):
        raise ValueError(f"costs must have the scores' shape {tuple(scores.shape)}, got {tuple(costs.shape)}")
    live = torch.isfinite(scores)
    any_live = live.any(-1, keepdim=True)
    logits = torch.where(live, scores / temperature, float("-inf"))
    # An example without any live beam would softmax all -inf into NaN, forward
    # and backward; give it constant logits instead and no loss.
    logits = torch.where(any_live, logits, torch.zeros_like(logits))
    probs = torch.softmax(logits, dim=-1)
    # Masked rather than multiplied by their zero probability: 0 * NaN is NaN.
    costs = torch.where(live, costs.to(probs), 0.0)
    loss = (probs * costs).sum(-1) * any_live[:, 0]
    return _reduce(loss, reduction)
