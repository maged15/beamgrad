# SPDX-License-Identifier: MIT
"""Decoding from logits and 16-bit rows, and the gradients of every decode output (CPU).

CUDA parity for the same features lives in test_cuda.py.
"""

import ctypes
import math

import numpy as np
import pytest
import torch

import beamgrad
from beamgrad import BeamOptions, decode, final_scores, search
from beamgrad._ctypes import (
    DTYPE_BF16,
    DTYPE_F16,
    DTYPE_F32,
    DBSBackwardInputsC,
    DBSDecodeOutputsExC,
    check,
    load_library,
    options_to_c,
)

OPTIONS = BeamOptions(beam_size=3, eos_token=2, min_length=1, length_penalty_alpha=0.6)
SCORE_FIELDS = ("final_scores", "final_raw_scores", "scores", "raw_scores")
C_DTYPES = {torch.float32: DTYPE_F32, torch.float16: DTYPE_F16, torch.bfloat16: DTYPE_BF16}


def random_logits(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 3.0).to(dtype)


def weighted_loss(trace, seed=0):
    """A loss on every score output, with random weights (dead and padding slots, which are -inf, left out)."""
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for name in SCORE_FIELDS:
        value = getattr(trace, name)
        weights = torch.randn(value.shape, generator=g)
        total = total + torch.where(torch.isfinite(value), value, 0.0).mul(weights).sum()
    return total


def c_reference(x, options, from_logits, grads):
    """The C library's dbs_decode_batch_into_ex and dbs_backward_batch_into_ex on x [B, T, K, V]."""
    lib = load_library()
    handle = ctypes.c_void_p()
    check(lib, None, lib.dbs_create_ex(options_to_c(options), ctypes.byref(handle)))
    try:
        B, T, K, V = x.shape
        inputs = x.contiguous()
        f32 = np.float32
        out = {
            "final_scores": np.zeros((B, K), f32),
            "final_raw_scores": np.zeros((B, K), f32),
            "final_lengths": np.zeros((B, K), np.int32),
            "tokens": np.zeros((B, T, K), np.int32),
            "parents": np.zeros((B, T, K), np.int32),
            "lengths": np.zeros((B, T, K), np.int32),
            "scores": np.zeros((B, T, K), f32),
            "raw_scores": np.zeros((B, T, K), f32),
            "from_logprob": np.zeros((B, T, K), np.uint8),
        }
        row_lse = np.zeros((B, T, K), f32)
        ex = DBSDecodeOutputsExC()
        for name, array in out.items():
            setattr(ex.base, name, array.ctypes.data)
        ex.row_lse = row_lse.ctypes.data
        status = lib.dbs_decode_batch_into_ex(
            handle, inputs.data_ptr(), C_DTYPES[x.dtype], int(from_logits), B, T, V, None, None, 1, ctypes.byref(ex)
        )
        check(lib, handle, status)

        grad = np.zeros((B, T, K, V), f32)
        g = {name: np.ascontiguousarray(value.numpy(), f32) for name, value in grads.items()}
        bw = DBSBackwardInputsC()
        bw.batch_size, bw.steps, bw.vocab_size = B, T, V
        for name in ("parents", "tokens", "lengths", "from_logprob"):
            setattr(bw, name, out[name].ctypes.data)
        for name, array in g.items():
            setattr(bw, "grad_" + name, array.ctypes.data)
        if from_logits:
            bw.logits = inputs.data_ptr()
            bw.logits_type = C_DTYPES[x.dtype]
            bw.row_lse = row_lse.ctypes.data
        check(lib, handle, lib.dbs_backward_batch_into_ex(handle, ctypes.byref(bw), 1, grad.ctypes.data))
        return out, row_lse, grad
    finally:
        lib.dbs_destroy(handle)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("from_logits", [False, True])
def test_decode_and_gradients_match_the_c_api_exactly(dtype, from_logits):
    x = random_logits(2, 5, 3, 23, seed=3, dtype=dtype)
    if not from_logits:
        x = torch.log_softmax(x.float(), -1).to(dtype)
    xg = x.clone().requires_grad_(True)
    trace = decode(xg, OPTIONS, from_logits=from_logits)
    loss = weighted_loss(trace, seed=1)
    loss.backward()
    # The loss's gradients with respect to each score output, for the C reference.
    g = torch.Generator().manual_seed(1)
    grads = {}
    for name in SCORE_FIELDS:
        value = getattr(trace, name).detach()
        weights = torch.randn(value.shape, generator=g)
        grads[name] = torch.where(torch.isfinite(value), weights, 0.0)
    out, _, grad = c_reference(x, OPTIONS, from_logits, grads)
    for name, expected in out.items():
        got = getattr(trace, name).detach()
        assert torch.equal(got.to(torch.from_numpy(expected).dtype), torch.from_numpy(expected)), name
    assert xg.grad.dtype == dtype
    assert torch.equal(xg.grad, torch.from_numpy(grad).to(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("from_logits", [False, True])
def test_half_inputs_are_read_exactly(dtype, from_logits):
    # 16-bit rows are converted exactly, so the search is the one over the same values in float32.
    x = random_logits(3, 4, 3, 37, seed=5, dtype=dtype)
    half = x.clone().requires_grad_(True)
    full = x.float().requires_grad_(True)
    a = decode(half, OPTIONS, from_logits=from_logits)
    b = decode(full, OPTIONS, from_logits=from_logits)
    for name in a._fields:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    weighted_loss(a).backward()
    weighted_loss(b).backward()
    assert half.grad.dtype == dtype
    assert torch.equal(half.grad, full.grad.to(dtype))


def test_from_logits_is_the_search_over_log_softmax():
    # Continuous logits (no ties): the same beams as over torch's log_softmax,
    # with scores and gradients equal up to the last bits of the logsumexp.
    x = random_logits(2, 6, 4, 50, seed=7)
    opts = BeamOptions(beam_size=4, eos_token=3, length_penalty_alpha=0.6)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    ta = decode(a, opts, from_logits=True)
    tb = decode(torch.log_softmax(b, -1), opts)
    for name in ("tokens", "parents", "lengths", "from_logprob", "final_lengths"):
        assert torch.equal(getattr(ta, name), getattr(tb, name)), name
    for name in SCORE_FIELDS:
        torch.testing.assert_close(getattr(ta, name), getattr(tb, name), rtol=1e-5, atol=1e-5, equal_nan=True)
    weighted_loss(ta).backward()
    weighted_loss(tb).backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)


def test_row_logsumexp_is_reported_for_the_rows_read():
    x = random_logits(2, 4, 3, 17, seed=9)
    out = torch.ops.beamgrad.decode_ex(x, True, None, -1, 0, 0.0, None, 0, 1.0, True)
    row_lse = out[-1]
    assert row_lse.shape == (2, 4, 3) and row_lse.dtype == torch.float32
    # Step 0 reads only beam 0; afterwards every beam is live (no EOS).
    torch.testing.assert_close(row_lse[:, 0, 0], torch.logsumexp(x[:, 0, 0], -1), rtol=0, atol=1e-5)
    assert torch.all(row_lse[:, 0, 1:] == 0)
    torch.testing.assert_close(row_lse[:, 1:], torch.logsumexp(x[:, 1:], -1), rtol=0, atol=1e-5)
    # Without logits, all zeros.
    assert torch.all(torch.ops.beamgrad.decode_ex(x, False, None, -1, 0, 0.0, None, 0, 1.0, True)[-1] == 0)


def test_rows_without_a_finite_logit_have_no_candidates():
    x = random_logits(1, 3, 2, 6, seed=11)
    x[0, 1, 0] = float("-inf")  # beam 0's row at step 1: nothing to extend it with
    trace = decode(x, BeamOptions(beam_size=2), from_logits=True)
    assert torch.all(trace.parents[0, 1] == 1)


def test_nan_in_logits_is_rejected():
    x = random_logits(1, 2, 2, 5, seed=13)
    x[0, 0, 0, 3] = float("nan")
    with pytest.raises(ValueError, match="logits contain NaN"):
        final_scores(x, BeamOptions(beam_size=2), from_logits=True)
    unchecked = BeamOptions(beam_size=2, validate_inputs=False)
    assert torch.isfinite(final_scores(x, unchecked, from_logits=True)).any()


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("from_logits", [False, True])
def test_gradients_of_every_output_match_finite_differences(from_logits):
    x = random_logits(1, 4, 3, 9, seed=15)
    if not from_logits:
        x = torch.log_softmax(x, -1)
    xg = x.clone().requires_grad_(True)
    weighted_loss(decode(xg, OPTIONS, from_logits=from_logits), seed=2).backward()

    def loss(y):
        with torch.no_grad():
            return weighted_loss(decode(y, OPTIONS, from_logits=from_logits), seed=2).item()

    eps = 1e-3
    g = torch.Generator().manual_seed(0)
    picked = torch.randperm(x.numel(), generator=g)[:30].tolist()
    nonzero = xg.grad.flatten().nonzero().flatten().tolist()[:40]
    for i in picked + nonzero:
        xp, xm = x.clone().flatten(), x.clone().flatten()
        xp[i] += eps
        xm[i] -= eps
        numeric = (loss(xp.view_as(x)) - loss(xm.view_as(x))) / (2 * eps)
        assert math.isclose(numeric, xg.grad.flatten()[i].item(), abs_tol=5e-3), i


def test_final_scores_gradient_is_unchanged():
    # decode_backward with only the final scores' gradient is the old operator's result, bit for bit.
    x = torch.log_softmax(random_logits(3, 5, 4, 21, seed=17), -1)
    opts = BeamOptions(beam_size=4, eos_token=1, length_penalty_alpha=0.8)
    trace = torch.ops.beamgrad.decode(x, None, 1, 0, 0.8, None, 0, 1.0, True)
    g = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    parents, tokens, lengths, flp = trace[4], trace[3], trace[5], trace[8]
    old = torch.ops.beamgrad.final_scores_backward(g, parents, tokens, lengths, flp, None, 21, 0.8)
    new = torch.ops.beamgrad.decode_backward(
        g, None, None, None, parents, tokens, lengths, flp, None, 21, 0.8, None, None
    )
    assert torch.equal(old, new)
    xg = x.clone().requires_grad_(True)
    (final_scores(xg, opts) * g).sum().backward()
    assert torch.equal(xg.grad, old)


def test_logits_gradient_rows_sum_to_zero():
    # Through the log-softmax, each row's gradient sums to (about) zero.
    x = random_logits(2, 5, 3, 31, seed=19).requires_grad_(True)
    weighted_loss(decode(x, OPTIONS, from_logits=True)).backward()
    torch.testing.assert_close(x.grad.sum(-1), torch.zeros(2, 5, 3), rtol=0, atol=1e-5)
    assert (x.grad != 0).any()


def test_decode_without_gradients_is_detached():
    trace = decode(random_logits(1, 3, 2, 5), BeamOptions(beam_size=2), from_logits=True)
    assert not any(t.requires_grad for t in trace)


# ---------------------------------------------------------------------------
# search, estimators, transforms
# ---------------------------------------------------------------------------


def test_search_from_logits():
    x = random_logits(2, 5, 3, 13, seed=21)
    result = search(x, OPTIONS, from_logits=True)
    assert torch.equal(result.scores, final_scores(x, OPTIONS, from_logits=True))
    assert result.step_log_probs == ()
    with pytest.raises(ValueError, match="log_softmax"):
        beamgrad.estimators.selected_softmax(result, OPTIONS)
    xg = x.clone().requires_grad_(True)
    search(xg, OPTIONS, from_logits=True).scores.sum().backward()
    xr = x.clone().requires_grad_(True)
    final_scores(xr, OPTIONS, from_logits=True).sum().backward()
    assert torch.equal(xg.grad, xr.grad)


def test_operators_and_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    assert hasattr(torch.ops.beamgrad, "decode_ex") and hasattr(torch.ops.beamgrad, "decode_backward")
    with FakeTensorMode():
        x = torch.empty(3, 5, 4, 17, dtype=torch.bfloat16)
        out = torch.ops.beamgrad.decode_ex(x, True, None, -1, 0, 0.0, None, 0, 1.0, True)
        assert out[0].shape == (3, 4) and out[0].dtype == torch.float32
        assert out[-1].shape == (3, 5, 4) and out[-1].dtype == torch.float32
        trace = decode(x, BeamOptions(beam_size=4), from_logits=True)
        assert trace.tokens.shape == (3, 5, 4)


def test_torch_compile_fullgraph_from_logits():
    g = torch.Generator().manual_seed(0)
    w_final, w_step = torch.randn(2, 3, generator=g), torch.randn(2, 6, 3, generator=g)

    def loss(x):
        trace = decode(x, OPTIONS, from_logits=True)
        finite = [torch.where(torch.isfinite(getattr(trace, n)), getattr(trace, n), 0.0) for n in SCORE_FIELDS]
        return (finite[0] * w_final).sum() + (finite[1] * w_final).sum() + ((finite[2] + finite[3]) * w_step).sum()

    compiled = torch.compile(loss, fullgraph=True, backend="aot_eager")
    x = random_logits(2, 6, 3, 11, seed=23)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    loss(a).backward()
    compiled(b).backward()
    assert torch.equal(a.grad, b.grad)


def test_torch_func_grad_from_logits():
    x = random_logits(2, 5, 3, 13, seed=25)
    reference = x.clone().requires_grad_(True)
    weighted_loss(decode(reference, OPTIONS, from_logits=True)).backward()
    grad = torch.func.grad(lambda y: weighted_loss(decode(y, OPTIONS, from_logits=True)))(x)
    assert torch.equal(grad, reference.grad)


@pytest.mark.skipif(not hasattr(torch.library, "register_vmap"), reason="torch.library.register_vmap needs PyTorch 2.5")
def test_vmap_from_logits():
    x = random_logits(4, 3, 5, 2, 9, seed=27)  # [N, B, T, K, V]
    options = BeamOptions(beam_size=2, eos_token=1)
    expected = final_scores(x.reshape(12, 5, 2, 9), options, from_logits=True).reshape(4, 3, 2)
    assert torch.equal(torch.vmap(lambda y: final_scores(y, options, from_logits=True))(x), expected)
    per_example = torch.vmap(torch.func.grad(lambda y: final_scores(y, options, from_logits=True).sum()))(x)
    reference = x.clone().requires_grad_(True)
    final_scores(reference.reshape(12, 5, 2, 9), options, from_logits=True).sum().backward()
    assert torch.equal(per_example, reference.grad)
