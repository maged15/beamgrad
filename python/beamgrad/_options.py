# SPDX-License-Identifier: MIT
"""Decoder options shared by every beamgrad backend."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass

# Mirrors DBS_CUDA_MAX_BEAM in include/dbs_cuda.h.
CUDA_MAX_BEAM = 1024

# Integer options reach the native code as 32-bit ints: larger values are
# rejected here rather than truncated (2**32 + 1 would silently become 1).
INT32_MAX = 2**31 - 1


# numbers.Integral and numbers.Real include NumPy's scalar types. bool is an
# int subclass, but True is not a count or an exponent.
def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _finite_float(value) -> float | None:
    """``value`` as a float if it is a finite real number, else ``None``."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except OverflowError:  # an int beyond the float range
        return None
    return result if math.isfinite(result) else None


def _eos_token(value) -> int | tuple[int, ...]:
    """``eos_token`` normalized: an int, or a tuple of two or more distinct token ids."""
    if _is_int(value):
        if not -1 <= value <= INT32_MAX:
            raise ValueError(f"eos_token must be -1 (disabled) or a token id, got {value!r}")
        return int(value)
    tokens = value.tolist() if hasattr(value, "tolist") else value  # a NumPy array or a tensor
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Iterable):
        raise ValueError(f"eos_token must be -1 (disabled), a token id or a sequence of token ids, got {value!r}")
    tokens = tuple(tokens)
    if not tokens or any(not _is_int(t) or not 0 <= t <= INT32_MAX for t in tokens):
        raise ValueError(f"eos_token must be a non-empty sequence of token ids, got {value!r}")
    tokens = tuple(dict.fromkeys(int(t) for t in tokens))  # distinct, in the order given
    return tokens[0] if len(tokens) == 1 else tokens


@dataclass(frozen=True)
class BeamOptions:
    """Beam search configuration.

    Args:
        beam_size: Number of beams ``K``. Must equal the beam dimension of
            ``log_probs``.
        eos_token: End-of-sequence token id, or ``-1`` to disable EOS handling.
            A beam that emits EOS is finished: it keeps its score and is carried
            forward unchanged at every later step. A sequence of ids (for
            example ``model.generation_config.eos_token_id``, which is a list
            for Qwen and Llama 3) makes each of them an EOS token: any of them
            finishes a beam, which is then carried forward with the one it
            emitted. :attr:`eos_tokens` lists them all. Up to 16 extra tokens
            are supported on CUDA.
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
            penalty is a constant shift). Values in ``(0, 1]`` disable it
            (unlike ``transformers``, values below 1 do not favour repeats).
    """

    beam_size: int
    eos_token: int | tuple[int, ...] = -1
    min_length: int = 0
    length_penalty_alpha: float = 0.0
    validate_inputs: bool = True
    banned_tokens: tuple[int, ...] | None = None
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        # Every field is checked, then stored as a plain int, float or tuple of
        # ints, so NumPy scalars are accepted and nothing else leaks through.
        if not _is_int(self.beam_size) or not 1 <= self.beam_size <= INT32_MAX:
            raise ValueError(f"beam_size must be a positive int (at most 2**31 - 1), got {self.beam_size!r}")
        object.__setattr__(self, "eos_token", _eos_token(self.eos_token))
        if not _is_int(self.min_length) or not 0 <= self.min_length <= INT32_MAX:
            raise ValueError(f"min_length must be a non-negative int (at most 2**31 - 1), got {self.min_length!r}")
        alpha = _finite_float(self.length_penalty_alpha)
        if alpha is None or alpha < 0.0:
            raise ValueError(
                f"length_penalty_alpha must be a finite, non-negative number, got {self.length_penalty_alpha!r}"
            )
        if not isinstance(self.validate_inputs, bool):
            raise ValueError(f"validate_inputs must be a bool, got {self.validate_inputs!r}")
        if self.banned_tokens is not None:
            banned = self.banned_tokens
            if hasattr(banned, "tolist"):  # a NumPy array or a tensor
                banned = banned.tolist()
            if isinstance(banned, (str, bytes)) or not isinstance(banned, Iterable):
                raise ValueError(f"banned_tokens must be a sequence of token ids, got {self.banned_tokens!r}")
            banned = tuple(banned)
            if any(not _is_int(t) or not 0 <= t <= INT32_MAX for t in banned):
                raise ValueError(f"banned_tokens must hold non-negative int token ids, got {banned!r}")
            object.__setattr__(self, "banned_tokens", tuple(int(t) for t in banned) or None)
        if not _is_int(self.no_repeat_ngram_size) or not 0 <= self.no_repeat_ngram_size <= INT32_MAX:
            value = self.no_repeat_ngram_size
            raise ValueError(f"no_repeat_ngram_size must be a non-negative int (at most 2**31 - 1), got {value!r}")
        penalty = _finite_float(self.repetition_penalty)
        if penalty is None or penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be a finite, positive number, got {self.repetition_penalty!r}")
        for name in ("beam_size", "min_length", "no_repeat_ngram_size"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "length_penalty_alpha", alpha)
        object.__setattr__(self, "repetition_penalty", penalty)

    @property
    def eos_tokens(self) -> tuple[int, ...]:
        """Every end-of-sequence token id, in the order given; ``()`` when EOS handling is off."""
        if isinstance(self.eos_token, tuple):
            return self.eos_token
        return () if self.eos_token < 0 else (self.eos_token,)

    def eos_ids(self, vocab_size: int) -> tuple[int, ...]:
        """:attr:`eos_tokens`, checked against ``vocab_size``."""
        outside = [t for t in self.eos_tokens if t >= vocab_size]
        if outside:
            raise ValueError(f"eos_token {outside[0]} is outside the vocabulary (size {vocab_size})")
        return self.eos_tokens

    def native_eos(self) -> tuple[int, list[int]]:
        """``(eos_token, extra_eos)`` as the operators take them: the first EOS
        token (``-1`` when off) and the others."""
        tokens = self.eos_tokens
        return (tokens[0], list(tokens[1:])) if tokens else (-1, [])

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
