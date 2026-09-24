# SPDX-License-Identifier: MIT
"""Decoder options shared by every beamgrad backend."""

from __future__ import annotations

import math
from collections.abc import Iterable
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
        validate_inputs: Raise ``ValueError`` if a row the search reads (the
            row of a live, unfinished beam) contains NaN or ``+inf``; ``-inf``
            is allowed and marks impossible tokens. The check is part of the
            decode kernels, so it costs no extra pass; on CUDA it reads one
            flag per example back to the host, which synchronizes the stream.
            Without it, such entries are never selected.
        banned_tokens: Token ids that are never selected.
        no_repeat_ngram_size: ``n > 0`` blocks every token that would repeat an
            n-gram already in the beam's own prefix (``1`` bans every token the
            beam has emitted).
        repetition_penalty: Values above 1 subtract ``log(repetition_penalty)``
            from the log-probability of every token already in the beam's
            prefix. The gradient of such a token's entry is unchanged (the
            penalty is a constant shift).
    """

    beam_size: int
    eos_token: int = -1
    min_length: int = 0
    length_penalty_alpha: float = 0.0
    validate_inputs: bool = True
    banned_tokens: tuple[int, ...] | None = None
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0

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
        if self.banned_tokens is not None:
            if isinstance(self.banned_tokens, (str, bytes)) or not isinstance(self.banned_tokens, Iterable):
                raise ValueError(f"banned_tokens must be a sequence of token ids, got {self.banned_tokens!r}")
            banned = tuple(self.banned_tokens)
            if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in banned):
                raise ValueError(f"banned_tokens must hold non-negative token ids, got {banned!r}")
            object.__setattr__(self, "banned_tokens", banned or None)
        if (
            isinstance(self.no_repeat_ngram_size, bool)
            or not isinstance(self.no_repeat_ngram_size, int)
            or self.no_repeat_ngram_size < 0
        ):
            raise ValueError(f"no_repeat_ngram_size must be a non-negative int, got {self.no_repeat_ngram_size!r}")
        penalty = float(self.repetition_penalty)
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be finite and positive, got {self.repetition_penalty!r}")

    def banned_mask(self, vocab_size: int) -> list[int] | None:
        """The banned tokens as a ``[vocab_size]`` 0/1 list, or ``None``."""
        if not self.banned_tokens:
            return None
        if max(self.banned_tokens) >= vocab_size:
            raise ValueError(
                f"banned_tokens {self.banned_tokens} include ids outside the vocabulary (size {vocab_size})"
            )
        mask = [0] * vocab_size
        for token in self.banned_tokens:
            mask[token] = 1
        return mask
