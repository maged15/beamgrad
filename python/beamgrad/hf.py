# SPDX-License-Identifier: MIT
"""Hugging Face ``transformers`` causal language models in :func:`beamgrad.beam_search`.

:class:`CausalLMStep` is a step function: it runs the model incrementally with
a key/value cache that follows the beams. :class:`CausalLMRescorer` is a
``rescore_fn``: it scores whole beams (or reference sequences) in one
teacher-forced pass, which is what :func:`beamgrad.beam_search` differentiates
in re-scoring mode, and which works with ``model.gradient_checkpointing_enable()``.

This module does not import ``transformers``; any model with the usual causal
LM interface works (``input_ids``, ``attention_mask``, ``position_ids``,
``past_key_values`` with ``reorder_cache``, ``logits_to_keep``,
``base_model`` and ``get_output_embeddings``).
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ._search import BeamState

__all__ = ["CausalLMRescorer", "CausalLMStep"]


def _positions(mask: torch.Tensor) -> torch.Tensor:
    # As generate() computes them: padding positions get 1.
    return (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)


class CausalLMStep:
    """A step function for a causal LM: next-token log-probabilities of every beam.

    Step 0 runs the (left-padded) prompts once per beam slot; each later step
    feeds every beam's last token, after reordering the key/value cache by
    ``beams.parents`` so that each slot continues its own hypothesis. Rows are
    the float32 log-softmax of the logits, as ``generate()`` computes them, and
    the model inputs match ``generate()``'s, so a float32 model gives the same
    beams as ``model.generate(num_beams=K)`` (see ``benchmarks/hf_beam_search.py``).
    Every search starts over from the prompts at step 0, so one instance can
    drive several searches.

    Args:
        model: The causal LM.
        input_ids: ``[B, P]`` prompt tokens, left-padded.
        attention_mask: ``[B, P]``, 0 on padding.
        beam_size: ``K``.
    """

    def __init__(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, beam_size: int):
        self.model = model
        self.batch_size, self.beam_size = input_ids.shape[0], beam_size
        self.input_ids = input_ids.repeat_interleave(beam_size, 0)
        self.prompt_mask = attention_mask.repeat_interleave(beam_size, 0)
        self.mask = self.prompt_mask  # grows by one column per step
        self.cache = None

    def __call__(self, beams: BeamState) -> torch.Tensor:
        B, K = self.batch_size, self.beam_size
        if beams.step == 0:
            self.mask, self.cache = self.prompt_mask, None  # the previous search's cache is freed first
            out = self.model(
                input_ids=self.input_ids,
                attention_mask=self.mask,
                position_ids=_positions(self.mask),
                cache_position=torch.arange(self.mask.shape[1], device=self.mask.device),
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            slots = torch.arange(B, device=beams.parents.device)[:, None] * K
            self.cache.reorder_cache((slots + beams.parents.clamp(min=0)).flatten())
            self.mask = torch.cat([self.mask, self.mask.new_ones(self.mask.shape[0], 1)], dim=1)
            out = self.model(
                input_ids=beams.tokens.clamp(min=0).view(-1, 1),
                attention_mask=self.mask,
                position_ids=self.mask.sum(-1, keepdim=True) - 1,
                cache_position=torch.tensor([self.mask.shape[1] - 1], device=self.mask.device),
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
            )
        self.cache = out.past_key_values
        return out.logits[:, -1].float().log_softmax(-1).view(B, K, -1)


def _token_log_probs(hidden: torch.Tensor, targets: torch.Tensor, head: torch.nn.Module) -> torch.Tensor:
    logits = head(hidden).float()
    return logits.gather(-1, targets[:, None])[:, 0] - logits.logsumexp(-1)


class CausalLMRescorer:
    """Teacher-forced log-probabilities of generated sequences, for ``rescore_fn``.

    Called with ``sequences [B, N, T]`` (``-1`` past each sequence's length) and
    ``lengths [B, N]``, it runs the model once on every prompt followed by each
    of its ``N`` sequences and returns ``[B, N, T]``: the log-probability of
    each token given the prompt and the tokens before it. ``N`` may be the beam
    size (``beam_search(..., rescore_fn=rescorer)``) or anything else, e.g. 1
    to score reference sequences for a structured-margin loss.

    With ``chunk_size`` set (the default), the vocabulary projection runs on
    the base model's last hidden states in chunks of that many positions, under
    activation checkpointing, so the ``[B * N, T, V]`` logits are never held.
    This assumes the logits are ``get_output_embeddings()(hidden)``, true for
    most causal LMs (Llama, Qwen, Mistral, GPT-2, ...); pass ``chunk_size=None``
    for models that rescale or soft-cap their logits.

    With ``gradient_checkpointing=True`` this pass (only) runs with the model's
    gradient checkpointing on, in training mode, so the model's activations are
    recomputed during backward instead of stored. transformers turns the
    key/value cache off whenever a checkpointing model is in training mode, so
    the search itself should run in eval mode; the rescorer restores the
    model's mode and checkpointing setting after its pass.

    Args:
        model: The causal LM.
        input_ids: ``[B, P]`` prompt tokens, left-padded.
        attention_mask: ``[B, P]``, 0 on padding.
        chunk_size: Positions per vocabulary-projection chunk, or ``None``.
        gradient_checkpointing: Recompute the model's activations in backward.
    """

    def __init__(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int | None = 256,
        gradient_checkpointing: bool = False,
    ):
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.gradient_checkpointing:
            return self._score(sequences, lengths)
        model = self.model
        was_training, was_checkpointing = model.training, getattr(model, "is_gradient_checkpointing", False)
        if not was_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        try:
            return self._score(sequences, lengths)
        finally:
            model.train(was_training)
            if not was_checkpointing:
                model.gradient_checkpointing_disable()

    def _score(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, N, T = sequences.shape
        P = self.input_ids.shape[1]
        prompt = self.input_ids.repeat_interleave(N, 0)
        prompt_mask = self.attention_mask.repeat_interleave(N, 0)
        generated = sequences.reshape(B * N, T).clamp(min=0)
        generated_mask = (torch.arange(T, device=sequences.device) < lengths.reshape(B * N, 1)).to(prompt_mask.dtype)
        ids = torch.cat([prompt, generated], dim=1)
        mask = torch.cat([prompt_mask, generated_mask], dim=1)
        if self.chunk_size is None:
            logits = self.model(
                input_ids=ids, attention_mask=mask, position_ids=_positions(mask), use_cache=False, logits_to_keep=T + 1
            ).logits[:, :T]
            log_probs = logits.float().log_softmax(-1).gather(-1, generated[..., None])[..., 0]
            return log_probs.view(B, N, T)
        hidden = self.model.base_model(
            input_ids=ids, attention_mask=mask, position_ids=_positions(mask), use_cache=False
        ).last_hidden_state
        # Position P - 1 + t predicts token t of the sequence.
        hidden = hidden[:, P - 1 : P - 1 + T].reshape(B * N * T, -1)
        targets = generated.reshape(-1)
        head = self.model.get_output_embeddings()
        chunks = [
            checkpoint(
                _token_log_probs,
                hidden[i : i + self.chunk_size],
                targets[i : i + self.chunk_size],
                head,
                use_reentrant=False,
            )
            for i in range(0, hidden.shape[0], self.chunk_size)
        ]
        return torch.cat(chunks).view(B, N, T)
