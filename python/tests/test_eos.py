# SPDX-License-Identifier: MIT
"""Several end-of-sequence tokens (as Qwen's <|im_end|> and <|endoftext|>)."""

import ctypes
import dataclasses
import warnings

import numpy as np
import pytest
import torch

import beamgrad
from beamgrad import BeamOptions, decode, final_scores
from beamgrad._ctypes import DBSDecodeOutputsC, check, load_library, options_to_c, set_extra_eos


def rows_with_ties(*shape, seed=0):
    """Log-probs with heavy ties, so that every token, EOS tokens included, is often the best."""
    g = torch.Generator().manual_seed(seed)
    return -0.5 * torch.randint(0, 4, shape, generator=g).float()


def test_options():
    assert BeamOptions(beam_size=2, eos_token=5).eos_tokens == (5,)
    assert BeamOptions(beam_size=2).eos_tokens == ()
    two = BeamOptions(beam_size=2, eos_token=[151645, 151643])
    assert two.eos_token == (151645, 151643) and two.eos_tokens == (151645, 151643)
    assert two.native_eos() == (151645, [151643])
    assert BeamOptions(beam_size=2, eos_token=[7]).eos_token == 7  # one token: an int
    assert BeamOptions(beam_size=2, eos_token=(3, 4, 3)).eos_tokens == (3, 4)  # distinct, in order
    assert BeamOptions(beam_size=2, eos_token=np.array([9, 2])).eos_tokens == (9, 2)
    assert BeamOptions(beam_size=2, eos_token=torch.tensor([9, 2])).eos_tokens == (9, 2)
    assert dataclasses.replace(two, beam_size=4).eos_tokens == two.eos_tokens
    for bad in ([], [-1], [1.5], "ab", -2, [True]):
        with pytest.raises(ValueError, match="eos_token"):
            BeamOptions(beam_size=2, eos_token=bad)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        final_scores(torch.zeros(2, 2, 5), BeamOptions(beam_size=2, eos_token=(1, 5)))


def test_either_token_finishes_a_beam():
    # Step 0: token 4 (the second EOS token) is the best; it finishes beam 0,
    # which is then carried forward with token 4 while beam 1 goes on.
    x = torch.full((3, 2, 5), -9.0)
    x[0, 0, 4], x[0, 0, 1] = -0.1, -0.5
    x[1:, :, 2] = -0.2
    out = decode(x, BeamOptions(beam_size=2, eos_token=(3, 4)))
    assert out.tokens[0, 0] == 4 and out.from_logprob[0, 0]
    for t in (1, 2):
        carried = (~out.from_logprob[t]) & (out.parents[t] >= 0)
        assert carried.sum() == 1 and out.tokens[t][carried].item() == 4
    assert 1 in out.final_lengths.tolist()
    # min_length masks both EOS tokens.
    masked = decode(x, BeamOptions(beam_size=2, eos_token=(3, 4), min_length=2))
    assert not bool(torch.isin(masked.tokens[0], torch.tensor([3, 4])).any())
    # With one EOS token, token 4 is an ordinary token.
    one = decode(x, BeamOptions(beam_size=2, eos_token=3))
    assert (one.final_lengths == 3).all()


def c_reference(x, options):
    """dbs_decode_batch_into with dbs_set_extra_eos_tokens."""
    lib = load_library()
    handle = ctypes.c_void_p()
    check(lib, None, lib.dbs_create_ex(options_to_c(options), ctypes.byref(handle)))
    try:
        set_extra_eos(lib, handle, options)
        B, T, K, V = x.shape
        out = {
            "final_scores": np.zeros((B, K), np.float32),
            "tokens": np.zeros((B, T, K), np.int32),
            "parents": np.zeros((B, T, K), np.int32),
            "lengths": np.zeros((B, T, K), np.int32),
            "scores": np.zeros((B, T, K), np.float32),
            "from_logprob": np.zeros((B, T, K), np.uint8),
        }
        c = DBSDecodeOutputsC()
        for name, array in out.items():
            setattr(c, name, array.ctypes.data)
        xc = x.contiguous()
        check(lib, handle, lib.dbs_decode_batch_into(handle, xc.data_ptr(), B, T, V, None, None, 1, ctypes.byref(c)))
        return out
    finally:
        lib.dbs_destroy(handle)


@pytest.mark.parametrize("min_length", [0, 2])
def test_matches_the_c_api(min_length):
    options = BeamOptions(beam_size=3, eos_token=(4, 1, 6), min_length=min_length, length_penalty_alpha=0.6)
    x = rows_with_ties(3, 6, 3, 8, seed=min_length)
    out = decode(x, options)
    for name, expected in c_reference(x, options).items():
        assert torch.equal(getattr(out, name).to(torch.from_numpy(expected).dtype), torch.from_numpy(expected)), name
    assert bool(torch.isin(out.tokens, torch.tensor([1, 6])).any())  # the extra EOS tokens occur


def test_gradients_follow_the_paths():
    # The gradient is the path gradient of each final score, whatever token ended the beam.
    options = BeamOptions(beam_size=3, eos_token=(4, 1), length_penalty_alpha=0.6)
    x = rows_with_ties(2, 5, 3, 8, seed=3).requires_grad_(True)
    trace = decode(x, options)
    trace.final_scores.sum().backward()
    rows = x.detach().clone().requires_grad_(True)
    reference = beamgrad.search(rows, options)
    reference.scores.sum().backward()
    assert torch.equal(x.grad, rows.grad)


def test_structured_margin_accepts_several_eos_tokens():
    options = BeamOptions(beam_size=2, eos_token=(4, 1))
    result = beamgrad.search(rows_with_ties(1, 3, 2, 6, seed=5), options)
    reference = torch.tensor([[2, 1, -1]])  # ends with the second EOS token: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        beamgrad.losses.structured_margin(result, reference, torch.zeros(1), eos_token=options.eos_token)
    with pytest.warns(UserWarning, match=r"any of the EOS tokens \[4, 1\]"):
        beamgrad.losses.structured_margin(result, torch.tensor([[2, 3, -1]]), torch.zeros(1), eos_token=(4, 1))


def test_jax_matches_torch():
    jax = pytest.importorskip("jax")
    import beamgrad.jax as bjax

    options = BeamOptions(beam_size=3, eos_token=(4, 1), min_length=1, length_penalty_alpha=0.4)
    x = rows_with_ties(2, 5, 3, 8, seed=7).numpy()
    value, grad = jax.value_and_grad(lambda y: bjax.final_scores(y, options).sum())(jax.numpy.asarray(x))
    xt = torch.from_numpy(x).requires_grad_(True)
    expected = final_scores(xt, options).sum()
    expected.backward()
    np.testing.assert_array_equal(np.float32(value), np.float32(expected.item()))
    np.testing.assert_array_equal(np.asarray(grad), xt.grad.numpy())


@pytest.mark.skipif(not beamgrad.cuda_available(), reason="beamgrad CUDA operators or GPU unavailable")
@pytest.mark.parametrize("from_logits", [False, True])
def test_cuda_matches_cpu(from_logits):
    options = BeamOptions(beam_size=4, eos_token=(5, 2, 9), min_length=2, length_penalty_alpha=0.6)
    x = rows_with_ties(3, 6, 4, 12, seed=11)
    cpu = x.clone().requires_grad_(True)
    gpu = x.cuda().requires_grad_(True)
    a = decode(cpu, options, from_logits=from_logits)
    b = decode(gpu, options, from_logits=from_logits)
    for name in a._fields:
        assert torch.equal(getattr(b, name).detach().cpu(), getattr(a, name).detach()), name
    a.final_scores.sum().backward()
    b.final_scores.sum().backward()
    assert torch.equal(gpu.grad.cpu(), cpu.grad)
    # The model-in-the-loop search: the CUDA step carries each finished beam's own EOS token.
    rows = x.cuda()
    stepped = beamgrad.beam_search(lambda beams: rows[:, beams.step], options, max_steps=6, batch_size=3)
    expected = decode(rows, options)
    for name in ("tokens", "parents", "lengths", "scores", "from_logprob"):
        assert torch.equal(getattr(stepped.trace, name), getattr(expected, name)), name
    assert bool(torch.isin(expected.tokens.cpu(), torch.tensor([2, 9])).any())
