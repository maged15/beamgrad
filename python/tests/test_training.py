# SPDX-License-Identifier: MIT
"""search(), sequence_scores(), the training losses and the gradient estimators."""

import ctypes
import warnings

import pytest
import torch

import beamgrad
from beamgrad import BeamOptions, decode, final_scores, losses, search, sequence_scores
from beamgrad import estimators as est
from beamgrad._ctypes import check, load_library, options_to_c, set_extra_eos


def random_log_probs(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape, generator=g) * 2.0, dim=-1)


# ---------------------------------------------------------------------------
# search(): scores and trace from one decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [None, [5, 2, 4]])
def test_search_matches_final_scores_and_decode(steps):
    x = random_log_probs(3, 5, 4, 17, seed=1).requires_grad_(True)
    options = BeamOptions(beam_size=4, eos_token=3, length_penalty_alpha=0.6)
    result = search(x, options, steps=steps)
    expected = decode(x.detach(), options, steps=steps)
    for name in expected._fields:
        assert torch.equal(getattr(result.trace, name), getattr(expected, name)), name
    assert torch.equal(result.scores.detach(), final_scores(x.detach(), options, steps=steps))
    weights = torch.linspace(-1.0, 1.0, 4)
    (g1,) = torch.autograd.grad((result.scores * weights).sum(), x)
    (g2,) = torch.autograd.grad((final_scores(x, options, steps=steps) * weights).sum(), x)
    assert torch.equal(g1, g2)
    paths = beamgrad.backtrack(expected)
    positions = torch.arange(5)
    assert torch.equal(result.sequences, torch.where(positions < expected.final_lengths[..., None], paths, -1))
    assert len(result.step_log_probs) == 5 and result.step_log_probs[0].data_ptr() == x.data_ptr()


def test_search_unbatched():
    x = random_log_probs(5, 3, 11, seed=2)
    options = BeamOptions(beam_size=3, eos_token=1)
    result = search(x, options)
    batched = search(x[None], options)
    assert result.scores.shape == (3,) and result.sequences.shape == (3, 5)
    assert torch.equal(result.scores, batched.scores[0])
    assert torch.equal(result.trace.tokens, batched.trace.tokens[0])


def test_sequence_scores_put_references_on_the_beams_scale():
    x = random_log_probs(2, 6, 3, 9, seed=3)
    options = BeamOptions(beam_size=3, eos_token=1, length_penalty_alpha=0.7)
    result = search(x, options)
    # Re-score the final beams from their paths through the rows.
    trace = result.trace
    token_lp = torch.full((2, 3, 6), float("-inf"))
    for b in range(2):
        for k in range(3):
            beam = k
            for t in range(5, -1, -1):
                if trace.from_logprob[b, t, beam]:
                    token_lp[b, k, t] = x[b, t, trace.parents[b, t, beam], trace.tokens[b, t, beam]]
                beam = int(trace.parents[b, t, beam])
    # Only on-path entries (t < length) count; carried steps come after the length.
    rescored = sequence_scores(token_lp.nan_to_num(neginf=0.0), result.lengths, options)
    torch.testing.assert_close(rescored, result.scores, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_structured_margin():
    x = random_log_probs(2, 4, 3, 7, seed=4).requires_grad_(True)
    options = BeamOptions(beam_size=3)
    result = search(x, options)
    best = result.sequences[:, 0]
    # Reference = the best beam of example 0, something else for example 1.
    reference = torch.stack([best[0], torch.tensor([6, 6, 6, 6])])
    # Low reference scores, so both hinges are active.
    reference_scores = torch.tensor([-40.0, -50.0], requires_grad=True)
    loss = losses.structured_margin(result, reference, reference_scores, margin=0.5, reduction="none")
    rival0 = result.scores[0, 1]  # best beam that is not the reference
    rival1 = result.scores[1, 0]
    expected = torch.stack([0.5 + rival0 + 40.0, 0.5 + rival1 + 50.0])
    torch.testing.assert_close(loss, expected)
    loss.sum().backward(retain_graph=True)
    torch.testing.assert_close(reference_scores.grad, torch.tensor([-1.0, -1.0]))
    (g0,) = torch.autograd.grad(result.scores[0, 1] + result.scores[1, 0], x, retain_graph=True)
    torch.testing.assert_close(x.grad, g0)
    # Once the reference wins by the margin, no loss and no gradient.
    assert losses.structured_margin(result, reference, torch.tensor([0.0, 0.0]), margin=0.5) == 0
    assert losses.matches(result.sequences, reference)[0, 0]
    assert not losses.matches(result.sequences, torch.cat([reference, reference[:, :1]], 1))[0, 0]


def test_structured_margin_checks_the_reference_scores_shape():
    result = search(random_log_probs(3, 4, 3, 7, seed=4), BeamOptions(beam_size=3))
    reference = torch.full((3, 4), 6)
    # [B, 1] (e.g. a keepdim sum) would broadcast into a [B, B] loss.
    with pytest.raises(ValueError, match="reference_scores"):
        losses.structured_margin(result, reference, torch.zeros(3, 1))
    with pytest.raises(ValueError, match="reference_scores"):
        losses.structured_margin(result, reference, torch.zeros(()))
    unbatched = search(random_log_probs(4, 3, 7, seed=4), BeamOptions(beam_size=3))
    loss = losses.structured_margin(unbatched, reference[:1], torch.tensor(-30.0), reduction="none")
    assert loss.shape == (1,) and loss.item() > 0


def test_minimum_risk_and_dead_examples():
    x = random_log_probs(2, 3, 4, 5, seed=5).requires_grad_(True)
    result = search(x, BeamOptions(beam_size=4))
    costs = torch.tensor([[0.0, 1.0, 0.5, 0.2], [1.0, 0.0, 0.3, 0.3]])
    loss = losses.minimum_risk(result, costs, temperature=0.5)
    expected = (torch.softmax(result.scores / 0.5, -1) * costs).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    # An example whose beams are all dead contributes no loss and no NaN.
    dead = result._replace(scores=result.scores.detach().clone().index_fill_(0, torch.tensor([1]), float("-inf")))
    y = losses.minimum_risk(dead, costs, reduction="none")
    assert y[1] == 0 and torch.isfinite(y).all()


def test_minimum_risk_ignores_the_costs_of_dead_beams():
    # One step over 3 tokens leaves 3 of 6 beams live.
    x = random_log_probs(2, 1, 6, 3, seed=6).requires_grad_(True)
    result = search(x, BeamOptions(beam_size=6))
    live = torch.isfinite(result.scores)
    assert live.sum(-1).tolist() == [3, 3]
    costs = torch.tensor([[0.1, 0.5, 0.9], [0.3, 0.2, 0.7]])
    # A cost function may give anything for a dead beam's empty hypothesis.
    padded = torch.cat([costs, torch.tensor([[float("nan"), float("inf"), -float("inf")]] * 2)], 1)
    loss = losses.minimum_risk(result, padded)
    expected = (torch.softmax(result.scores[:, :3], -1) * costs).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    # And for an example without any live beam.
    dead = result._replace(scores=result.scores.detach().clone().index_fill_(0, torch.tensor([1]), float("-inf")))
    assert losses.minimum_risk(dead, padded, reduction="none")[1] == 0


# ---------------------------------------------------------------------------
# Estimators, against the C library's surrogates
# ---------------------------------------------------------------------------


def c_library_surrogates(x, options, pool_multiplier, selected_temperature, soft_topk_temperature, g_sel, g_rel):
    """Selected weights, relaxed weights, the pool (scores, parents, tokens) and the dense gradient from libdbs."""
    lib = load_library()
    f32p = ctypes.POINTER(ctypes.c_float)
    for name in (
        "dbs_result_weights",
        "dbs_result_relaxed_weights",
        "dbs_result_pool_scores",
        "dbs_backward_grad_log_probs",
    ):
        getattr(lib, name).argtypes, getattr(lib, name).restype = [ctypes.c_void_p], f32p
    for name in ("dbs_result_pool_parents", "dbs_result_pool_tokens"):
        getattr(lib, name).argtypes, getattr(lib, name).restype = [ctypes.c_void_p], ctypes.POINTER(ctypes.c_int32)
    lib.dbs_backward_dense.argtypes = [ctypes.c_void_p] * 2 + [f32p] * 3 + [ctypes.POINTER(ctypes.c_void_p)]
    lib.dbs_backward_dense.restype = ctypes.c_int
    c = options_to_c(options)
    c.relaxed_pool_multiplier = pool_multiplier
    c.selected_temperature = selected_temperature
    c.soft_topk_temperature = soft_topk_temperature
    handle = ctypes.c_void_p()
    check(lib, None, lib.dbs_create_ex(c, ctypes.byref(handle)))
    set_extra_eos(lib, handle, options)
    T, K, V = x.shape
    P = K * pool_multiplier
    result = ctypes.c_void_p()
    xc = x.detach().contiguous()
    check(lib, handle, lib.dbs_decode(handle, ctypes.cast(xc.data_ptr(), f32p), T, V, ctypes.byref(result)))
    weights = torch.tensor([lib.dbs_result_weights(result)[i] for i in range(T * K)]).view(T, K)
    relaxed = torch.tensor([lib.dbs_result_relaxed_weights(result)[i] for i in range(T * P)]).view(T, P)
    pool = torch.tensor([lib.dbs_result_pool_scores(result)[i] for i in range(T * P)]).view(T, P)
    pool_parents = torch.tensor([lib.dbs_result_pool_parents(result)[i] for i in range(T * P)]).view(T, P)
    pool_tokens = torch.tensor([lib.dbs_result_pool_tokens(result)[i] for i in range(T * P)]).view(T, P)
    backward = ctypes.c_void_p()
    gs, gr = g_sel.contiguous(), g_rel.contiguous()
    check(
        lib,
        handle,
        lib.dbs_backward_dense(
            handle,
            result,
            ctypes.cast(gs.data_ptr(), f32p),
            ctypes.cast(gr.data_ptr(), f32p),
            None,
            ctypes.byref(backward),
        ),
    )
    grad = torch.tensor([lib.dbs_backward_grad_log_probs(backward)[i] for i in range(T * K * V)]).view(T, K, V)
    lib.dbs_free_backward(backward)
    lib.dbs_free_result(result)
    lib.dbs_destroy(handle)
    return weights, relaxed, (pool, pool_parents, pool_tokens), grad


@pytest.mark.parametrize(
    "options",
    [
        BeamOptions(beam_size=3),
        BeamOptions(beam_size=4, eos_token=2, min_length=2, length_penalty_alpha=0.6),
        BeamOptions(beam_size=4, eos_token=(2, 5, 7), min_length=2, length_penalty_alpha=0.6),
    ],
)
def test_estimators_match_the_c_library(options):
    torch.manual_seed(7)
    T, K, V, m = 5, options.beam_size, 9, 3
    x = random_log_probs(T, K, V, seed=8).requires_grad_(True)
    result = search(x, options)
    torch.testing.assert_close(est.path_scores(result, options).detach(), result.trace.scores, rtol=0, atol=0)
    weights = est.selected_softmax(result, options, temperature=0.7)
    relaxed = est.relaxed_topk(result, options, pool_multiplier=m, temperature=0.3)
    g_sel, g_rel = torch.randn(T, K), torch.randn(T, K * m)
    c_weights, c_relaxed, c_pool, c_grad = c_library_surrogates(x, options, m, 0.7, 0.3, g_sel, g_rel)
    torch.testing.assert_close(weights.detach(), c_weights, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(relaxed.scores.detach(), c_pool[0], rtol=0, atol=0)
    assert torch.equal(relaxed.parents, c_pool[1].long()) and torch.equal(relaxed.tokens, c_pool[2].long())
    # Each expansion's score is its parent's raw score plus its entry, length-penalised.
    trace = result.trace
    for t, p in relaxed.from_logprob.nonzero().tolist():
        parent, token = int(relaxed.parents[t, p]), int(relaxed.tokens[t, p])
        raw = (0.0 if t == 0 else trace.raw_scores[t - 1, parent]) + x[t, parent, token].detach()
        length = 1 if t == 0 else int(trace.lengths[t - 1, parent]) + 1
        penalty = beamgrad.length_penalty(torch.tensor(length), options.length_penalty_alpha)
        torch.testing.assert_close(relaxed.scores[t, p].detach(), raw / penalty)
    torch.testing.assert_close(relaxed.weights.detach(), c_relaxed, rtol=1e-4, atol=1e-5)
    ((weights * g_sel).sum() + (relaxed.weights * g_rel).sum()).backward()
    torch.testing.assert_close(x.grad, c_grad, rtol=1e-4, atol=1e-5)


def test_estimators_need_the_rows():
    model_rows = random_log_probs(1, 3, 2, 6, seed=9)
    result = beamgrad.beam_search(lambda beams: model_rows[:, beams.step], BeamOptions(beam_size=2), max_steps=3)
    est.selected_softmax(result, BeamOptions(beam_size=2))  # through the steps: rows kept
    rescored = beamgrad.beam_search(
        lambda beams: model_rows[:, beams.step],
        BeamOptions(beam_size=2),
        max_steps=3,
        rescore_fn=lambda seqs, lengths: torch.zeros(seqs.shape, requires_grad=True),
    )
    with pytest.raises(ValueError, match="rows"):
        est.selected_softmax(rescored, BeamOptions(beam_size=2))
    with pytest.raises(NotImplementedError):
        est.relaxed_topk(result, BeamOptions(beam_size=2, no_repeat_ngram_size=2))


def test_estimators_agree_between_search_and_beam_search():
    # A step function that ignores the beams reads the same rows as search().
    x = random_log_probs(2, 4, 3, 10, seed=10).requires_grad_(True)
    options = BeamOptions(beam_size=3, eos_token=4, length_penalty_alpha=0.8)
    stepped = beamgrad.beam_search(lambda beams: x[:, beams.step], options, max_steps=4, batch_size=2)
    direct = search(x, options)
    assert torch.equal(stepped.sequences, direct.sequences)
    for a, b in [
        (est.path_scores(stepped, options), est.path_scores(direct, options)),
        (est.selected_softmax(stepped, options, 0.5), est.selected_softmax(direct, options, 0.5)),
        (est.relaxed_topk(stepped, options, 2).weights, est.relaxed_topk(direct, options, 2).weights),
    ]:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        g = torch.randn_like(a)
        (ga,) = torch.autograd.grad((a * g).sum(), x, retain_graph=True)
        (gb,) = torch.autograd.grad((b * g).sum(), x, retain_graph=True)
        torch.testing.assert_close(ga, gb)


@pytest.mark.skipif(not beamgrad.cuda_available(), reason="beamgrad CUDA operators or GPU unavailable")
def test_estimators_and_losses_on_cuda():
    options = BeamOptions(beam_size=4, eos_token=2, min_length=2, length_penalty_alpha=0.6)
    x = random_log_probs(3, 6, 4, 33, seed=11)
    outs = {}
    for device in ("cpu", "cuda"):
        xd = x.to(device).clone().requires_grad_(True)
        result = search(xd, options, steps=[6, 3, 5])
        relaxed = est.relaxed_topk(result, options, pool_multiplier=2)
        costs = torch.linspace(0, 1, 4, device=device).expand(3, 4)
        loss = (
            losses.minimum_risk(result, costs)
            + est.selected_softmax(result, options)[..., 0].sum()
            + (relaxed.weights * relaxed.scores.detach().exp()).sum()
        )
        loss.backward()
        outs[device] = (result.sequences.cpu(), loss.detach().cpu(), xd.grad.cpu())
    assert torch.equal(outs["cpu"][0], outs["cuda"][0])
    torch.testing.assert_close(outs["cpu"][1], outs["cuda"][1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs["cpu"][2], outs["cuda"][2], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("offset", [0.0, -1e3, -1e4])
def test_relaxed_topk_weights_sum_to_k_at_any_score_magnitude(offset):
    # A long search has cumulative scores in the thousands; the bisection must
    # still place theta so that each step's weights sum to K.
    spaced = offset + 0.1 * torch.arange(32, dtype=torch.float32)
    for scores in (spaced, spaced.flip(0)):
        weights = est._SoftTopK.apply(scores[None], 4, 0.25, 1e-4, 48)
        assert abs(float(weights.sum()) - 4) < 1e-2
    x = random_log_probs(2, 6, 4, 50, seed=12)
    x[:, 0] += offset  # every hypothesis's score moves by the offset
    options = BeamOptions(beam_size=4)
    result = search(x, options)
    assert float(result.scores.max()) < offset + 1
    relaxed = est.relaxed_topk(result, options, pool_multiplier=8, temperature=0.25)
    assert float((relaxed.weights.sum(-1) - 4).abs().max()) < 1e-2


def test_structured_margin_warns_about_references_without_eos():
    # Rows whose best beam is [1, EOS]: the reference must include that EOS.
    eos = 2
    x = torch.full((1, 3, 2, 5), -9.0)
    x[0, 0, 0, 1] = -0.1
    x[0, 1, :, eos] = -0.1
    options = BeamOptions(beam_size=2, eos_token=eos)
    result = search(x, options)
    assert result.sequences[0, 0].tolist() == [1, eos, -1]
    with_eos, without_eos = torch.tensor([[1, eos]]), torch.tensor([[1]])
    assert losses.matches(result.sequences, with_eos)[0, 0]
    assert not losses.matches(result.sequences, without_eos).any()
    # Scored as the search scored it, the reference with EOS beats the other
    # beam by the margin; without EOS, the beam that is the reference is its
    # own rival and the loss is stuck at the margin.
    best = result.scores[:, 0].detach()
    assert losses.structured_margin(result, with_eos, best, eos_token=eos) == 0
    with pytest.warns(UserWarning, match=r"reference rows \[0\] do not end with eos_token 2"):
        assert losses.structured_margin(result, without_eos, best, eos_token=eos) == 1.0
    # Only the offending rows are named; correct references and the default stay silent.
    two = search(x.expand(2, -1, -1, -1), options)
    with pytest.warns(UserWarning, match=r"reference rows \[1\] "):
        losses.structured_margin(two, torch.tensor([[1, eos], [1, -1]]), torch.zeros(2), eos_token=eos)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        losses.structured_margin(two, torch.tensor([[1, eos], [3, eos]]), torch.zeros(2), eos_token=eos)
        losses.structured_margin(two, torch.tensor([[1, -1], [1, -1]]), torch.zeros(2))  # no eos_token: no check
        losses.structured_margin(two, torch.tensor([[1, -1], [1, -1]]), torch.zeros(2), eos_token=-1)
