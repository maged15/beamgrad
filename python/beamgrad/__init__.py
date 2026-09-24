# SPDX-License-Identifier: MIT
"""beamgrad: differentiable beam search for PyTorch, on CPU and CUDA.

Hard beam search forward, surrogate gradients backward::

    import beamgrad

    options = beamgrad.BeamOptions(beam_size=4, eos_token=2, length_penalty_alpha=0.6)
    scores = beamgrad.final_scores(log_probs, options)   # [B, T, K, V] -> [B, K]
    scores[:, 0].sum().backward()                        # gradients w.r.t. log_probs

    trace = beamgrad.decode(log_probs, options)          # tokens, parents, scores, ...
    sequences = beamgrad.backtrack(trace)                # [B, K, T] token paths
"""

from ._options import CUDA_MAX_BEAM, BeamOptions
from ._torch import BeamSearchOutput, backtrack, cuda_available, decode, final_scores
from ._version import __version__

__all__ = [
    "BeamOptions",
    "BeamSearchOutput",
    "CUDA_MAX_BEAM",
    "__version__",
    "backtrack",
    "cuda_available",
    "decode",
    "final_scores",
]
