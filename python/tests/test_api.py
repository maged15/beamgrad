# SPDX-License-Identifier: MIT
"""Tests for the PyTorch API on CPU (CUDA parity lives in test_cuda.py)."""

import ctypes
import math

import pytest
import torch

import beamgrad
from beamgrad import BeamOptions, backtrack, decode, final_scores
from beamgrad._ctypes import check, load_library, options_to_c


def random_log_probs(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape, generator=g) * 2.0, dim=-1).to(dtype)


def c_abi_reference(x, options):
    """Final scores and dense gradient for d(sum(final * w))/dx via the C ABI."""
    lib = load_library()
    handle = ctypes.c_void_p()
    check(lib, None, lib.dbs_create_ex(options_to_c(options), ctypes.byref(handle)))
    try:
        x = x.contiguous()
        T, K, V = x.shape
        result = ctypes.c_void_p()
        f32 = ctypes.POINTER(ctypes.c_float)
        check(lib, handle, lib.dbs_decode(handle, ctypes.cast(x.data_ptr(), f32), T, V, ctypes.byref(result)))
        scores = torch.tensor([lib.dbs_result_final_scores(result)[k] for k in range(K)])
        weights = torch.linspace(-1.0, 1.0, K)
        backward = ctypes.c_void_p()
        check(
            lib,
            handle,
            lib.dbs_backward(handle, result, None, None, ctypes.cast(weights.data_ptr(), f32), ctypes.byref(backward)),
        )
        grad = torch.zeros(T * K * V)
        for i in range(lib.dbs_backward_sparse_logprob_count(backward)):
            grad[lib.dbs_backward_sparse_logprob_indices(backward)[i]] += lib.dbs_backward_sparse_logprob_values(
                backward
            )[i]
        lib.dbs_free_backward(backward)
        lib.dbs_free_result(result)
        return scores, weights, grad.view(T, K, V)
    finally:
        lib.dbs_destroy(handle)


# ---------------------------------------------------------------------------
# Shapes and basic behaviour
# ---------------------------------------------------------------------------


def test_unbatched_and_batched_shapes():
    x = random_log_probs(3, 5, 4, 17)
    opts = BeamOptions(beam_size=4)
    batched = final_scores(x, opts)
    assert batched.shape == (3, 4)
    for b in range(3):
        torch.testing.assert_close(final_scores(x[b], opts), batched[b], rtol=0, atol=0)


def test_scores_are_sorted_best_first():
    y = final_scores(random_log_probs(2, 6, 5, 23), BeamOptions(beam_size=5))
    assert torch.all(y[:, :-1] >= y[:, 1:])


def test_first_step_expands_only_beam_zero():
    x = torch.full((1, 3, 10), -5.0)
    x[0, 0, 7] = -0.1  # beam 0 row
    x[0, 1, 2] = 0.0  # beams 1 and 2 are not live at t = 0
    out = decode(x, BeamOptions(beam_size=3))
    assert out.tokens[0, 0].item() == 7
    assert torch.all(out.parents[0] == 0)


def test_matches_c_abi_exactly():
    for seed, (eos, min_length, alpha) in enumerate([(-1, 0, 0.0), (3, 2, 0.0), (5, 0, 0.7), (0, 3, 1.3)]):
        x = random_log_probs(6, 4, 19, seed=seed)
        opts = BeamOptions(beam_size=4, eos_token=eos, min_length=min_length, length_penalty_alpha=alpha)
        ref_scores, weights, ref_grad = c_abi_reference(x, opts)
        xg = x.clone().requires_grad_(True)
        y = final_scores(xg, opts)
        assert torch.equal(y.detach(), ref_scores)
        (y * weights).sum().backward()
        assert torch.equal(xg.grad, ref_grad)


def test_deterministic_and_batch_invariant():
    x = random_log_probs(4, 7, 3, 31, seed=5)
    opts = BeamOptions(beam_size=3, eos_token=2, length_penalty_alpha=0.6)
    first = decode(x, opts)
    second = decode(x, opts)
    for a, b in zip(first, second, strict=False):
        assert torch.equal(a, b)
    alone = decode(x[2:3], opts)
    assert torch.equal(alone.final_scores[0], first.final_scores[2])
    assert torch.equal(alone.tokens[0], first.tokens[2])


# ---------------------------------------------------------------------------
# Decode trace
# ---------------------------------------------------------------------------


def test_decode_fields_and_dtypes():
    out = decode(random_log_probs(2, 5, 3, 11), BeamOptions(beam_size=3, eos_token=1))
    assert out.final_scores.shape == out.final_raw_scores.shape == out.final_lengths.shape == (2, 3)
    for name in ("tokens", "parents", "lengths", "scores", "raw_scores", "from_logprob"):
        assert getattr(out, name).shape == (2, 5, 3), name
    assert out.tokens.dtype == out.parents.dtype == out.lengths.dtype == torch.int64
    assert out.from_logprob.dtype == torch.bool
    assert out.steps.tolist() == [5, 5]


def test_backtracked_paths_reproduce_raw_scores():
    x = random_log_probs(3, 6, 4, 13, seed=11)
    opts = BeamOptions(beam_size=4, eos_token=3, length_penalty_alpha=0.5)
    out = decode(x, opts)
    paths = backtrack(out)
    for b in range(3):
        for k in range(4):
            total, beam = 0.0, k
            for t in range(5, -1, -1):
                if out.from_logprob[b, t, beam]:
                    total += x[b, t, out.parents[b, t, beam], out.tokens[b, t, beam]].item()
                assert paths[b, k, t] == out.tokens[b, t, beam]
                beam = out.parents[b, t, beam].item()
            assert math.isclose(total, out.final_raw_scores[b, k].item(), rel_tol=1e-5, abs_tol=1e-5)
            assert beam == 0  # every path starts from beam 0


def test_length_penalty_ranking_formula():
    x = random_log_probs(5, 3, 9, seed=2)
    alpha = 0.8
    out = decode(x, BeamOptions(beam_size=3, eos_token=4, length_penalty_alpha=alpha))
    penalty = ((5.0 + out.final_lengths.clamp(min=1).float()) / 6.0) ** alpha
    torch.testing.assert_close(out.final_scores, out.final_raw_scores / penalty)


def test_eos_beams_are_carried_forward_unchanged():
    x = torch.full((4, 2, 5), -3.0)
    x[0, 0, 1] = -0.1  # EOS is the best first token
    x[0, 0, 2] = -0.2
    out = decode(x, BeamOptions(beam_size=2, eos_token=1))
    assert out.tokens[0, 0].item() == 1
    finished = [k for k in range(2) if out.tokens[3, k] == 1]
    assert finished
    k = finished[0]
    assert not out.from_logprob[3, k]
    assert out.final_lengths[k].item() == 1
    assert math.isclose(out.final_raw_scores[k].item(), -0.1, rel_tol=1e-6)


def test_min_length_masks_early_eos():
    x = torch.full((3, 2, 4), -2.0)
    x[:, :, 0] = 0.0  # EOS would always win
    out = decode(x, BeamOptions(beam_size=2, eos_token=0, min_length=3))
    assert 0 not in out.tokens[:2].flatten().tolist()
    assert 0 in out.tokens[2].tolist()


def test_negative_infinity_marks_impossible_tokens():
    x = random_log_probs(3, 2, 6, seed=4)
    x[:, :, 0] = -math.inf
    out = decode(x, BeamOptions(beam_size=2))
    assert 0 not in out.tokens.flatten().tolist()
    assert torch.isfinite(out.final_scores).all()


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradient_matches_finite_differences():
    x = random_log_probs(4, 3, 8, seed=9).double().float()
    opts = BeamOptions(beam_size=3, length_penalty_alpha=0.4)
    weights = torch.tensor([0.7, -0.3, 1.1])
    xg = x.clone().requires_grad_(True)
    (final_scores(xg, opts) * weights).sum().backward()
    eps = 1e-3
    g = torch.Generator().manual_seed(0)
    flat = torch.randperm(x.numel(), generator=g)[:40].tolist()
    nonzero = xg.grad.flatten().nonzero().flatten().tolist()
    for i in flat + nonzero:
        xp, xm = x.clone().flatten(), x.clone().flatten()
        xp[i] += eps
        xm[i] -= eps
        fp = (final_scores(xp.view_as(x), opts) * weights).sum()
        fm = (final_scores(xm.view_as(x), opts) * weights).sum()
        numeric = (fp - fm).item() / (2 * eps)
        assert math.isclose(numeric, xg.grad.flatten()[i].item(), abs_tol=2e-3), i


def test_gradient_dtype_follows_input():
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        x = random_log_probs(3, 2, 8, dtype=dtype).requires_grad_(True)
        y = final_scores(x, BeamOptions(beam_size=2))
        assert y.dtype == torch.float32
        y.sum().backward()
        assert x.grad.dtype == dtype


def test_non_contiguous_input():
    base = random_log_probs(5, 3, 2, 12, seed=8)
    x = base.transpose(0, 1)  # [3, 5, 2, 12], non-contiguous
    assert not x.is_contiguous()
    opts = BeamOptions(beam_size=2)
    torch.testing.assert_close(final_scores(x, opts), final_scores(x.contiguous(), opts), rtol=0, atol=0)


def test_gradients_flow_into_a_model():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 3 * 16)
    h = torch.randn(4, 8)
    log_probs = torch.log_softmax(model(h).view(4, 3, 16), dim=-1)  # [T, K, V]
    loss = -final_scores(log_probs, BeamOptions(beam_size=3))[0]
    loss.backward()
    assert model.weight.grad is not None and model.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Variable-length batches
# ---------------------------------------------------------------------------


def test_steps_per_example():
    x = random_log_probs(3, 6, 2, 10, seed=12)
    opts = BeamOptions(beam_size=2, eos_token=1)
    steps = [6, 2, 4]
    xg = x.clone().requires_grad_(True)
    y = final_scores(xg, opts, steps=steps)
    y.sum().backward()
    for b, s in enumerate(steps):
        ref = x[b, :s].clone().requires_grad_(True)
        yr = final_scores(ref, opts)
        torch.testing.assert_close(y[b], yr, rtol=0, atol=0)
        yr.sum().backward()
        torch.testing.assert_close(xg.grad[b, :s], ref.grad, rtol=0, atol=0)
        assert torch.all(xg.grad[b, s:] == 0)
    out = decode(x, opts, steps=torch.tensor(steps))
    assert out.steps.tolist() == steps
    assert torch.all(out.tokens[1, 2:] == -1)
    paths = backtrack(out)
    assert torch.all(paths[1, :, 2:] == -1)
    assert torch.all(paths[1, :, :2] >= 0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beam_size": 0},
        {"beam_size": 2.0},
        {"beam_size": True},
        {"beam_size": 2, "eos_token": -2},
        {"beam_size": 2, "min_length": -1},
        {"beam_size": 2, "length_penalty_alpha": -0.5},
        {"beam_size": 2, "length_penalty_alpha": float("nan")},
        {"beam_size": 2, "validate_inputs": 1},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        BeamOptions(**kwargs)


def test_invalid_inputs():
    opts = BeamOptions(beam_size=2)
    with pytest.raises(ValueError, match="shape"):
        final_scores(torch.randn(3, 4), opts)
    with pytest.raises(ValueError, match="beams"):
        final_scores(torch.randn(3, 4, 8), opts)
    with pytest.raises(ValueError, match="non-empty"):
        final_scores(torch.randn(0, 2, 8), opts)
    with pytest.raises(ValueError, match="vocabulary"):
        final_scores(torch.randn(3, 2, 8), BeamOptions(beam_size=2, eos_token=8))
    with pytest.raises(TypeError):
        final_scores(torch.zeros(3, 2, 8, dtype=torch.long), opts)
    with pytest.raises(TypeError):
        final_scores(torch.randn(3, 2, 8), {"beam_size": 2})
    with pytest.raises(ValueError, match="steps"):
        final_scores(torch.randn(3, 2, 8), opts, steps=[3])
    with pytest.raises(ValueError, match="steps"):
        final_scores(torch.randn(2, 3, 2, 8), opts, steps=[3])
    with pytest.raises(ValueError, match="steps"):
        final_scores(torch.randn(2, 3, 2, 8), opts, steps=[3, 4])


def test_nan_and_positive_infinity_are_rejected_unless_disabled():
    x = random_log_probs(3, 2, 8)
    for bad in (float("nan"), float("inf")):
        y = x.clone()
        y[1, 1, 3] = bad
        with pytest.raises(ValueError, match="NaN"):
            final_scores(y, BeamOptions(beam_size=2))
        out = final_scores(y, BeamOptions(beam_size=2, validate_inputs=False))
        assert out.shape == (2,)


def test_version_and_capabilities():
    assert beamgrad.__version__.count(".") == 2
    assert isinstance(beamgrad.cuda_available(), bool)
    assert beamgrad.CUDA_MAX_BEAM == 1024
