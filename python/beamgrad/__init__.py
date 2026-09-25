# SPDX-License-Identifier: MIT
"""beamgrad: differentiable beam search for PyTorch, on CPU and CUDA.

Hard beam search forward, surrogate gradients backward::

    import beamgrad

    options = beamgrad.BeamOptions(beam_size=4, eos_token=2, length_penalty_alpha=0.6)
    scores = beamgrad.final_scores(log_probs, options)   # [B, T, K, V] -> [B, K]
    scores[:, 0].sum().backward()                        # gradients w.r.t. log_probs

    trace = beamgrad.decode(log_probs, options)          # tokens, parents, scores, ...
    sequences = beamgrad.backtrack(trace)                # [B, K, T] token paths

With an autoregressive model, whose rows depend on the beams chosen so far,
``beamgrad.beam_search(step_fn, options, max_steps)`` runs the model inside the
search and returns differentiable scores of the beams it found.
``beamgrad.search(log_probs, options)`` returns the same result for rows that
are already computed: scores, sequences and trace from one decode.
``beamgrad.losses`` has training objectives on such a result (structured
margin, minimum risk) and ``beamgrad.estimators`` smoother surrogates of it
(selected-beam softmax, relaxed top-k).
"""

from . import estimators, losses
from ._options import CUDA_MAX_BEAM, BeamOptions
from ._search import BeamSearchResult, BeamState, beam_search, search, sequence_scores
from ._torch import BeamSearchOutput, backtrack, cuda_available, decode, final_scores, length_penalty
from ._version import __version__

__all__ = [
    "BeamOptions",
    "BeamSearchOutput",
    "BeamSearchResult",
    "BeamState",
    "CUDA_MAX_BEAM",
    "__version__",
    "backtrack",
    "beam_search",
    "cuda_available",
    "decode",
    "estimators",
    "final_scores",
    "length_penalty",
    "losses",
    "search",
    "sequence_scores",
]
