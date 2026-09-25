# SPDX-License-Identifier: MIT
"""beam_search driving a transformer with a key/value cache: cache handling,
gradients, long sequences and mixed precision."""

import functools
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import beamgrad
from beamgrad import BeamOptions, beam_search
from beamgrad import estimators as est


class TinyTransformer(nn.Module):
    """One-layer decoder-only LM; position 0 holds an example embedding, the tokens follow."""

    def __init__(self, vocab=13, examples=3, d=16, heads=2, max_positions=1024):
        super().__init__()
        self.heads = heads
        self.examples = nn.Embedding(examples, d)
        self.embed = nn.Embedding(vocab, d)
        self.positions = nn.Embedding(max_positions, d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def start(self, ids):
        return (self.examples(ids) + self.positions.weight[0])[:, None]

    def tokens(self, tokens, offset):
        return self.embed(tokens) + self.positions(torch.arange(offset, offset + tokens.shape[1], device=tokens.device))

    def forward(self, h, past=None):
        """Logits of the new positions h [N, t, d], and the cache after them."""
        N, t, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(N, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=past is None and t > 1)
        h = h + self.proj(a.transpose(1, 2).reshape(N, t, d))
        h = h + self.ff(self.norm2(h))
        return self.out(h), (k, v)


class CachedStep:
    """One new position per step; the key/value cache follows the beams' parents."""

    def __init__(self, model, ids, beam_size, log_softmax_dtype=torch.float32):
        self.model, self.ids, self.K, self.cache = model, ids, beam_size, None
        self.dtype = log_softmax_dtype

    def __call__(self, beams):
        B, K = self.ids.shape[0], self.K
        if beams.step == 0:
            h = self.model.start(self.ids.repeat_interleave(K))
        else:
            order = (torch.arange(B, device=beams.parents.device)[:, None] * K + beams.parents.clamp(min=0)).flatten()
            self.cache = tuple(c[order] for c in self.cache)
            h = self.model.tokens(beams.tokens.clamp(min=0).reshape(B * K, 1), offset=beams.step)
        logits, self.cache = self.model(h, self.cache)
        logits = logits[:, -1].to(self.dtype)
        # Outside autocast, which would run log_softmax in float32 on CUDA.
        with torch.autocast(logits.device.type, enabled=False):
            return logits.log_softmax(-1).view(B, K, -1)


def recompute_step(model, ids, beam_size):
    """No cache: every step runs the whole prefix of every beam."""

    def step(beams):
        B, K = ids.shape[0], beam_size
        h = model.start(ids.repeat_interleave(K))
        if beams.step > 0:
            h = torch.cat([h, model.tokens(beams.sequences.clamp(min=0).reshape(B * K, -1), offset=1)], dim=1)
        logits, _ = model(h)
        return logits[:, -1].float().log_softmax(-1).view(B, K, -1)

    return step


def rescorer(model, ids):
    def rescore(sequences, lengths):
        B, N, T = sequences.shape
        seqs = sequences.clamp(min=0).reshape(B * N, T)
        h = torch.cat([model.start(ids.repeat_interleave(N)), model.tokens(seqs[:, :-1], offset=1)], dim=1)
        logits, _ = model(h)
        return logits.log_softmax(-1).gather(-1, seqs[..., None])[..., 0].view(B, N, T)

    return rescore


def grads(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_cached_decoding_matches_recomputing_the_prefix():
    torch.manual_seed(0)
    model = TinyTransformer().double()
    ids = torch.tensor([0, 2, 1])
    options = BeamOptions(beam_size=4, eos_token=1, length_penalty_alpha=0.6)
    with torch.no_grad():
        cached = beam_search(CachedStep(model, ids, 4), options, max_steps=9, batch_size=3)
        plain = beam_search(recompute_step(model, ids, 4), options, max_steps=9, batch_size=3)
    assert torch.equal(cached.sequences, plain.sequences)
    torch.testing.assert_close(cached.scores, plain.scores)


@pytest.mark.parametrize("alpha", [0.0, 0.8])
def test_gradient_through_the_steps_equals_rescoring(alpha):
    torch.manual_seed(1)
    model = TinyTransformer().double()
    ids = torch.tensor([1, 0])
    options = BeamOptions(beam_size=3, eos_token=2, length_penalty_alpha=alpha)
    weights = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
    stepped = beam_search(CachedStep(model, ids, 3), options, max_steps=8, batch_size=2)
    g_steps = grads(model, (stepped.scores * weights).sum())
    rescored = beam_search(
        CachedStep(model, ids, 3), options, max_steps=8, batch_size=2, rescore_fn=rescorer(model, ids)
    )
    g_rescore = grads(model, (rescored.scores * weights).sum())
    assert torch.equal(stepped.sequences, rescored.sequences)
    torch.testing.assert_close(stepped.scores, rescored.scores)
    torch.testing.assert_close(g_steps, g_rescore, rtol=1e-5, atol=1e-7)


def test_long_sequences():
    torch.manual_seed(2)
    model = TinyTransformer(vocab=40)
    ids = torch.tensor([0, 1])
    T = 384
    options = BeamOptions(beam_size=4, eos_token=-1)
    result = beam_search(CachedStep(model, ids, 4), options, max_steps=T, batch_size=2)
    assert result.sequences.shape == (2, 4, T) and bool((result.lengths == T).all())
    assert bool((result.sequences >= 0).all()) and len(result.step_log_probs) == T
    g = grads(model, result.scores[:, 0].sum())
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    # The final scores are sums of T log-probabilities along each path.
    rows = torch.stack(result.step_log_probs, 1).detach()
    torch.testing.assert_close(result.scores.detach(), beamgrad.final_scores(rows, options), rtol=0, atol=0)
    # Inference holds no rows, and an EOS stops the loop early with the same output shape.
    with torch.no_grad():
        inference = beam_search(CachedStep(model, ids, 4), options, max_steps=T, batch_size=2)
        stops = beam_search(CachedStep(model, ids, 4), BeamOptions(beam_size=4, eos_token=3), max_steps=T, batch_size=2)
    assert inference.step_log_probs == () and torch.equal(inference.sequences, result.sequences)
    assert stops.sequences.shape == (2, 4, T) and stops.trace.tokens.shape[1] <= T


# The test model under CPU bfloat16 autocast, forward and backward, using
# PyTorch's kernels only (no beamgrad operator runs).
_CPU_BFLOAT16_PROBE = """
import sys, torch
sys.path.insert(0, sys.argv[1])
from test_models import TinyTransformer
model = TinyTransformer()
with torch.autocast("cpu", dtype=torch.bfloat16):
    logits, cache = model(model.start(torch.tensor([0, 2]).repeat_interleave(3)))
    for t in range(1, 7):
        logits, cache = model(model.tokens(torch.zeros(6, 1, dtype=torch.long), offset=t), cache)
with torch.autocast("cpu", enabled=False):
    logits[:, -1].to(torch.bfloat16).log_softmax(-1).float().sum().backward()
"""


@functools.cache
def cpu_bfloat16_runs() -> bool:
    """Whether PyTorch's own bfloat16 CPU kernels run on this machine.

    On some virtualised CI hosts they die with an illegal instruction (Windows
    0xC000001D), which would take the whole test process down, so they are
    tried in a subprocess first.
    """
    probe = [sys.executable, "-c", _CPU_BFLOAT16_PROBE, str(Path(__file__).parent)]
    return subprocess.run(probe, capture_output=True, timeout=600).returncode == 0


def autocast_devices():
    devices = ["cpu"]
    if beamgrad.cuda_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", autocast_devices())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast(device, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("CPU autocast is bfloat16")
    if device == "cpu" and not cpu_bfloat16_runs():
        pytest.skip("PyTorch's bfloat16 CPU kernels crash on this machine (illegal instruction)")
    torch.manual_seed(3)
    model = TinyTransformer().to(device)
    ids = torch.tensor([0, 2], device=device)
    options = BeamOptions(beam_size=3, eos_token=1, length_penalty_alpha=0.6)
    with torch.autocast(device, dtype=dtype):
        # Rows straight from the autocast model, without a float32 cast.
        step = CachedStep(model, ids, 3, log_softmax_dtype=dtype)
        result = beam_search(step, options, max_steps=7, batch_size=2, device=device)
    rows = result.step_log_probs
    assert all(r.dtype == dtype for r in rows) and result.scores.dtype == torch.float32
    # The same search as a float32 decode of the low-precision rows, with the same gradient.
    stacked = torch.stack(rows, 1).detach().float().requires_grad_(True)
    reference = beamgrad.search(stacked, options)
    assert torch.equal(result.sequences[..., : reference.sequences.shape[-1]], reference.sequences)
    torch.testing.assert_close(result.scores.detach(), reference.scores.detach(), rtol=0, atol=0)
    weights = torch.tensor([1.0, -0.5, 0.25], device=device)
    g_rows = torch.autograd.grad((result.scores * weights).sum(), rows, retain_graph=True)
    (g_ref,) = torch.autograd.grad((reference.scores * weights).sum(), stacked)
    torch.testing.assert_close(torch.stack(g_rows, 1).float(), g_ref.to(dtype).float())
    # Into the (float32) parameters, and through the estimators.
    g = grads(model, (result.scores * weights).sum() + est.selected_softmax(result, options)[..., 0].sum())
    assert torch.isfinite(g).all() and g.abs().sum() > 0
