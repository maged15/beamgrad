# SPDX-License-Identifier: MIT
"""Decoder options shared by every beamgrad backend."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Mirrors DBS_CUDA_MAX_BEAM in include/dbs_cuda.h.
CUDA_MAX_BEAM = 1024


@dataclass(frozen=True)
class BeamOptions:
    """Beam search configuration.

    Args:
        beam_size: Number of beams ``K``. Must equal the beam dimension of
            ``log_probs``.
        eos_token: End-of-sequence token id, or ``-1`` to disable EOS handling.
            A beam that emits EOS is finished: it keeps its score and is carried
            forward unchanged at every later step.
        min_length: EOS is masked until a hypothesis has at least this many
            tokens (including the EOS itself).
        length_penalty_alpha: GNMT length penalty exponent. Beams are ranked by
            ``raw / ((5 + length) / 6) ** alpha``; ``0`` ranks by the raw
            cumulative log-probability.
        validate_inputs: Reject ``log_probs`` containing NaN or ``+inf``
            (``-inf`` is allowed and marks impossible tokens). The check scans
            the whole tensor, which synchronizes CUDA streams; disable it in
            hot loops once inputs are known to be clean.
    """

    beam_size: int
    eos_token: int = -1
    min_length: int = 0
    length_penalty_alpha: float = 0.0
    validate_inputs: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.beam_size, bool) or not isinstance(self.beam_size, int) or self.beam_size < 1:
            raise ValueError(f"beam_size must be a positive int, got {self.beam_size!r}")
        if isinstance(self.eos_token, bool) or not isinstance(self.eos_token, int) or self.eos_token < -1:
            raise ValueError(f"eos_token must be -1 (disabled) or a token id, got {self.eos_token!r}")
        if isinstance(self.min_length, bool) or not isinstance(self.min_length, int) or self.min_length < 0:
            raise ValueError(f"min_length must be a non-negative int, got {self.min_length!r}")
        alpha = float(self.length_penalty_alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"length_penalty_alpha must be finite and non-negative, got {self.length_penalty_alpha!r}")
        if not isinstance(self.validate_inputs, bool):
            raise ValueError(f"validate_inputs must be a bool, got {self.validate_inputs!r}")
