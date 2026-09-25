# SPDX-License-Identifier: MIT
"""beamgrad.hf: the causal-LM step function and re-scorer, on a stand-in model.

transformers is not needed: ``FakeCausalLM`` has the interface the adapters
use (``past_key_values`` with ``reorder_cache``, ``logits_to_keep``,
``base_model``, ``get_output_embeddings``, gradient checkpointing switches).
Each position attends to the masked mean of the keys up to it, so the cached,
one-token-per-step computation and the teacher-forced one agree.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import beamgrad
from beamgrad import BeamOptions, beam_search
from beamgrad.hf import CausalLMRescorer, CausalLMStep


class Cache:
    def __init__(self, keys):
        self.keys = keys  # [N, L, d]

    def reorder_cache(self, order):
        self.keys = self.keys[order]


class Body(nn.Module):
    def __init__(self, vocab, d, max_positions=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.positions = nn.Embedding(max_positions, d)
        self.mix = nn.Linear(d, d)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values=None, use_cache=False, **unused):
        past = 0 if past_key_values is None else past_key_values.keys.shape[1]
        assert attention_mask.shape == (input_ids.shape[0], past + input_ids.shape[1]), "mask must cover cache + input"
        keys = self.embed(input_ids) + self.positions(position_ids)
        if past_key_values is not None:
            keys = torch.cat([past_key_values.keys, keys], dim=1)
        mask = attention_mask[..., None].to(keys.dtype)
        # Position i attends to the masked keys 0..i.
        ctx = (keys * mask).cumsum(1) / mask.cumsum(1).clamp(min=1e-6)
        hidden = torch.tanh(self.mix(ctx[:, past:]))
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=Cache(keys) if use_cache else None)


class FakeCausalLM(nn.Module):
    def __init__(self, vocab=11, d=12, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.body = Body(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.is_gradient_checkpointing = False

    @property
    def base_model(self):
        return self.body

    def get_output_embeddings(self):
        return self.head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def forward(self, input_ids, attention_mask, position_ids, logits_to_keep=0, **kwargs):
        out = self.body(input_ids, attention_mask, position_ids, **kwargs)
        hidden = out.last_hidden_state[:, -logits_to_keep:] if logits_to_keep else out.last_hidden_state
        return SimpleNamespace(logits=self.head(hidden), past_key_values=out.past_key_values)


def prompts():
    ids = torch.tensor([[3, 1, 4, 1], [0, 0, 5, 9]])  # left-padded
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    return ids, mask


OPTIONS = [BeamOptions(beam_size=3), BeamOptions(beam_size=4, eos_token=2, min_length=2, length_penalty_alpha=0.6)]


def recompute_step(model, ids, mask, K):
    """No cache: every step runs the prompt and each beam's whole prefix."""

    def step(beams):
        B, _, t = beams.sequences.shape
        full_ids = torch.cat([ids.repeat_interleave(K, 0), beams.sequences.reshape(B * K, t).clamp(min=0)], 1)
        full_mask = torch.cat([mask.repeat_interleave(K, 0), mask.new_ones(B * K, t)], 1)
        positions = (full_mask.cumsum(-1) - 1).masked_fill(full_mask == 0, 1)
        logits = model(full_ids, full_mask, positions, logits_to_keep=1).logits[:, -1]
        return logits.float().log_softmax(-1).view(B, K, -1)

    return step


@pytest.mark.parametrize("options", OPTIONS)
def test_step_matches_recomputing_the_prefix(options):
    model = FakeCausalLM().eval()
    ids, mask = prompts()
    K = options.beam_size
    with torch.no_grad():
        cached = beam_search(CausalLMStep(model, ids, mask, K), options, max_steps=6, batch_size=2)
        full = beam_search(recompute_step(model, ids, mask, K), options, max_steps=6, batch_size=2)
    assert torch.equal(cached.sequences, full.sequences)
    torch.testing.assert_close(cached.scores, full.scores)


def test_step_function_can_drive_several_searches():
    model = FakeCausalLM().eval()
    ids, mask = prompts()
    options = OPTIONS[1]
    step = CausalLMStep(model, ids, mask, options.beam_size)
    with torch.no_grad():
        first = beam_search(step, options, max_steps=5, batch_size=2)
        second = beam_search(step, options, max_steps=5, batch_size=2)
    assert torch.equal(first.sequences, second.sequences)
    assert torch.equal(first.scores, second.scores)


@pytest.mark.parametrize("chunk_size", [None, 256, 5])
@pytest.mark.parametrize("options", OPTIONS)
def test_rescorer_reproduces_the_search(options, chunk_size):
    model = FakeCausalLM().eval()
    ids, mask = prompts()
    K = options.beam_size
    with torch.no_grad():
        result = beam_search(CausalLMStep(model, ids, mask, K), options, max_steps=6, batch_size=2)
        T = result.trace.tokens.shape[1]
        log_probs = CausalLMRescorer(model, ids, mask, chunk_size=chunk_size)(result.sequences[..., :T], result.lengths)
    assert log_probs.shape == (2, K, T)
    live = torch.isfinite(result.scores)
    rescored = beamgrad.sequence_scores(log_probs, result.lengths, options)
    torch.testing.assert_close(rescored[live], result.scores[live])


def test_rescoring_gives_the_gradient_through_the_steps():
    ids, mask = prompts()
    options = OPTIONS[1]
    K = options.beam_size
    weights = torch.tensor([1.0, -0.5, 0.25, 2.0])
    grads = []
    for rescore in (False, True):
        model = FakeCausalLM().eval()
        step = CausalLMStep(model, ids, mask, K)
        rescore_fn = CausalLMRescorer(model, ids, mask, chunk_size=4) if rescore else None
        result = beam_search(step, options, max_steps=6, batch_size=2, rescore_fn=rescore_fn)
        live = torch.isfinite(result.scores)
        (torch.where(live, result.scores, 0.0) * weights).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for through_steps, rescored in zip(*grads, strict=True):
        torch.testing.assert_close(rescored, through_steps, rtol=1e-4, atol=1e-5)
    assert any(g.abs().sum() > 0 for g in grads[0])


def test_rescorer_restores_the_model_after_gradient_checkpointing():
    model = FakeCausalLM().eval()
    ids, mask = prompts()
    sequences = torch.tensor([[[4, 2, -1]], [[7, 7, 1]]])
    lengths = torch.tensor([[2], [3]])
    seen = []  # (training, checkpointing) during the rescorer's pass
    original = model.body.forward

    def spy(*args, **kwargs):
        seen.append((model.training, model.is_gradient_checkpointing))
        return original(*args, **kwargs)

    model.body.forward = spy
    rescorer = CausalLMRescorer(model, ids, mask, gradient_checkpointing=True)
    rescorer(sequences, lengths)
    assert seen == [(True, True)]
    assert not model.training and not model.is_gradient_checkpointing

    # Settings the caller chose are kept.
    model.train()
    model.gradient_checkpointing_enable()
    rescorer(sequences, lengths)
    assert model.training and model.is_gradient_checkpointing


def test_rescorer_rejects_a_bad_chunk_size():
    ids, mask = prompts()
    with pytest.raises(ValueError, match="chunk_size"):
        CausalLMRescorer(FakeCausalLM(), ids, mask, chunk_size=0)
