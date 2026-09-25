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

import torch

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
) -> torch.Tensor:
    """Hinge loss: the reference must outscore the best non-reference beam by ``margin``.

    ``max(0, margin + best rival score - reference score)`` per example, where
    the rival is the highest-scoring beam that is not the reference. The loss
    is zero once the reference wins by the margin, whether or not the search
    found it; otherwise it raises the reference's score and lowers the rival's.

    Args:
        result: The search's result; its ``scores`` carry the gradient.
        reference: ``[B, T']`` reference tokens, ``-1`` after the end.
        reference_scores: ``[B]`` the references' scores on the beams' scale,
            e.g. :func:`beamgrad.sequence_scores` of their teacher-forced
            token log-probabilities (differentiable).
        margin: Required score difference.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    """
    scores, sequences = _batched(result)
    rivals = scores.masked_fill(matches(sequences, reference), float("-inf"))
    best_rival = rivals.amax(1)
    loss = torch.relu(margin + best_rival - reference_scores)
    # No rival (every beam is the reference or dead): nothing to separate.
    loss = torch.where(torch.isfinite(best_rival), loss, torch.zeros_like(loss))
    return _reduce(loss, reduction)


def minimum_risk(
    result: BeamSearchResult,
    costs: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Expected cost of the beams under ``softmax(scores / temperature)``.

    ``costs[b, k]`` is the cost of beam ``k`` (e.g. 1 - BLEU against the
    reference; no gradient needed). Dead beams (``-inf`` scores) get no
    probability. Lower temperatures concentrate the distribution on the best
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
    loss = (probs * costs.to(probs.dtype)).sum(-1) * any_live[:, 0]
    return _reduce(loss, reduction)
