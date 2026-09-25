# SPDX-License-Identifier: MIT
"""Decoder options shared by every beamgrad backend."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# Mirrors DBS_CUDA_MAX_BEAM in include/dbs_cuda.h.
CUDA_MAX_BEAM = 1024

# Integer options reach the native code as 32-bit ints: larger values are
# rejected here rather than truncated (2**32 + 1 would silently become 1).
INT32_MAX = 2**31 - 1


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        if not _is_int(self.beam_size) or not 1 <= self.beam_size <= INT32_MAX:
            raise ValueError(f"beam_size must be a positive int (at most 2**31 - 1), got {self.beam_size!r}")
        if not _is_int(self.eos_token) or not -1 <= self.eos_token <= INT32_MAX:
            raise ValueError(f"eos_token must be -1 (disabled) or a token id, got {self.eos_token!r}")
        if not _is_int(self.min_length) or not 0 <= self.min_length <= INT32_MAX:
            raise ValueError(f"min_length must be a non-negative int (at most 2**31 - 1), got {self.min_length!r}")
        alpha = float(self.length_penalty_alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"length_penalty_alpha must be finite and non-negative, got {self.length_penalty_alpha!r}")
        if not isinstance(self.validate_inputs, bool):
            raise ValueError(f"validate_inputs must be a bool, got {self.validate_inputs!r}")
        if self.banned_tokens is not None:
            if isinstance(self.banned_tokens, (str, bytes)) or not isinstance(self.banned_tokens, Iterable):
                raise ValueError(f"banned_tokens must be a sequence of token ids, got {self.banned_tokens!r}")
            banned = tuple(self.banned_tokens)
            if any(not _is_int(t) or not 0 <= t <= INT32_MAX for t in banned):
                raise ValueError(f"banned_tokens must hold non-negative token ids, got {banned!r}")
            object.__setattr__(self, "banned_tokens", banned or None)
        if not _is_int(self.no_repeat_ngram_size) or not 0 <= self.no_repeat_ngram_size <= INT32_MAX:
            value = self.no_repeat_ngram_size
            raise ValueError(f"no_repeat_ngram_size must be a non-negative int (at most 2**31 - 1), got {value!r}")
        penalty = float(self.repetition_penalty)
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be finite and positive, got {self.repetition_penalty!r}")

    def banned_ids(self, vocab_size: int) -> tuple[int, ...] | None:
        """The banned token ids, checked against ``vocab_size``, or ``None``."""
        if not self.banned_tokens:
            return None
        if max(self.banned_tokens) >= vocab_size:
            raise ValueError(
                f"banned_tokens {self.banned_tokens} include ids outside the vocabulary (size {vocab_size})"
            )
        return self.banned_tokens

    def banned_mask(self, vocab_size: int) -> list[int] | None:
        """The banned tokens as a ``[vocab_size]`` 0/1 list, or ``None``."""
        ids = self.banned_ids(vocab_size)
        if ids is None:
            return None
        mask = [0] * vocab_size
        for token in ids:
            mask[token] = 1
        return mask
