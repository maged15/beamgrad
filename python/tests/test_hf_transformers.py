# SPDX-License-Identifier: MIT
"""beamgrad.hf against real transformers models.

Tiny models with random weights, built from a config (nothing is downloaded):
Llama (rotary positions, grouped-query attention) and GPT-2 (learned
positions). Skipped when transformers is not installed; test_hf.py covers the
adapters with a transformers-free stand-in.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

import beamgrad  # noqa: E402
from beamgrad import BeamOptions  # noqa: E402
from beamgrad.hf import CausalLMRescorer, CausalLMStep  # noqa: E402

B, K, T = 3, 4, 6  # examples, beams, new tokens


def make_model(kind: str):
    torch.manual_seed(0)
    special = dict(pad_token_id=0, bos_token_id=1, eos_token_id=2)
    if kind == "llama":
        config = transformers.LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            **special,
        )
        model = transformers.LlamaForCausalLM(config)
    else:
        config = transformers.GPT2Config(vocab_size=97, n_embd=32, n_layer=2, n_head=4, n_positions=64, **special)
        model = transformers.GPT2LMHeadModel(config)
    return model.eval()


def prompts():
    """Three left-padded prompts of lengths 3, 5 and 2."""
    ids = torch.tensor([[0, 0, 5, 9, 11], [3, 4, 5, 6, 7], [0, 0, 0, 8, 13]])
    mask = (torch.arange(5) >= torch.tensor([2, 0, 3])[:, None]).long()
    return ids, mask


@pytest.fixture(params=["llama", "gpt2"])
def model(request):
    return make_model(request.param)


def test_beam_search_matches_generate(model):
    ids, mask = prompts()
    # Plain beam search on both sides: no EOS, T tokens, total log-probability.
    expected = model.generate(
        ids,
        attention_mask=mask,
        num_beams=K,
        num_return_sequences=K,
        do_sample=False,
        length_penalty=0.0,
        early_stopping=False,
        min_new_tokens=T,
        max_new_tokens=T,
        eos_token_id=None,
        pad_token_id=0,
    )[:, ids.shape[1] :].view(B, K, T)
    with torch.no_grad():
        result = beamgrad.beam_search(CausalLMStep(model, ids, mask, K), BeamOptions(beam_size=K), T, batch_size=B)
    assert torch.equal(result.sequences, expected)


def test_rescorer_reproduces_the_search_scores(model):
    ids, mask = prompts()
    options = BeamOptions(beam_size=K, length_penalty_alpha=0.6)
    with torch.no_grad():
        result = beamgrad.beam_search(CausalLMStep(model, ids, mask, K), options, T, batch_size=B)
        for chunk_size in (256, 5, None):  # chunked, uneven chunks, and the full-logits path
            token_log_probs = CausalLMRescorer(model, ids, mask, chunk_size=chunk_size)(
                result.sequences, result.lengths
            )
            rescored = beamgrad.sequence_scores(token_log_probs, result.lengths, options)
            torch.testing.assert_close(rescored, result.scores, rtol=0, atol=1e-5)


def test_rescore_gradient_matches_the_steps_gradient(model):
    ids, mask = prompts()
    options = BeamOptions(beam_size=K, length_penalty_alpha=0.6)
    weights = torch.linspace(1.0, -1.0, K)

    def gradient(rescore_fn):
        model.zero_grad(set_to_none=True)
        step = CausalLMStep(model, ids, mask, K)
        result = beamgrad.beam_search(step, options, T, batch_size=B, rescore_fn=rescore_fn)
        (result.scores * weights).sum().backward()
        return result.sequences, torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    steps_sequences, steps = gradient(None)
    rescore_sequences, rescored = gradient(CausalLMRescorer(model, ids, mask))
    assert torch.equal(steps_sequences, rescore_sequences)
    assert steps.numel() == rescored.numel() == sum(p.numel() for p in model.parameters())
    cosine = torch.nn.functional.cosine_similarity(steps.double(), rescored.double(), dim=0)
    assert cosine > 0.9999
