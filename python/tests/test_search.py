# SPDX-License-Identifier: MIT
"""beamgrad.beam_search: beam search that drives an autoregressive model."""

import pytest
import torch
from torch import nn

import beamgrad
from beamgrad import BeamOptions, backtrack, beam_search, decode, final_scores


class TinyLM(nn.Module):
    """A GRU language model: the rows of a step depend on each beam's prefix."""

    def __init__(self, vocab: int = 13, hidden: int = 16, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab + 1, hidden)  # the last id is BOS
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)
        self.bos = vocab

    def step_fn(self, batch_size: int):
        """A step function that runs the model on each beam's full prefix."""

        def step(beams: beamgrad.BeamState) -> torch.Tensor:
            B, K, t = beams.sequences.shape
            h = torch.zeros(B * K, self.gru.hidden_size, dtype=self.out.weight.dtype)
            tokens = beams.sequences.reshape(B * K, t).clamp(min=0)
            h = self.gru(self.embed(torch.full((B * K,), self.bos)), h)
            for i in range(t):
                h = self.gru(self.embed(tokens[:, i]), h)
            return self.out(h).log_softmax(-1).view(B, K, -1)

        return step


def expected_paths(trace):
    """[B, K, T] tokens of each final beam, -1 past its length."""
    paths = backtrack(trace)
    positions = torch.arange(paths.shape[-1])
    return torch.where(positions < trace.final_lengths[..., None], paths, -1)


@pytest.mark.parametrize(
    "options",
    [
        BeamOptions(beam_size=3),
        BeamOptions(beam_size=4, eos_token=2, min_length=2, length_penalty_alpha=0.6),
        BeamOptions(beam_size=3, eos_token=1, banned_tokens=[4, 5], no_repeat_ngram_size=2, repetition_penalty=1.5),
    ],
)
def test_matches_decode_of_the_rows_it_produced(options):
    model = TinyLM()
    result = beam_search(model.step_fn(2), options, max_steps=6, batch_size=2)
    rows = torch.stack(result.step_log_probs, 1).detach()
    expected = decode(rows, options)
    for name in ("tokens", "parents", "lengths", "scores", "raw_scores", "from_logprob"):
        assert torch.equal(getattr(result.trace, name), getattr(expected, name)), name
    assert torch.equal(result.scores.detach(), expected.final_scores)
    assert torch.equal(result.scores.detach(), final_scores(rows, options))
    assert torch.equal(result.raw_scores, expected.final_raw_scores)
    assert torch.equal(result.lengths, expected.final_lengths)
    T = rows.shape[1]
    assert result.sequences.shape == (2, options.beam_size, 6)
    assert torch.equal(result.sequences[..., :T], expected_paths(expected))
    assert bool((result.sequences[..., T:] == -1).all())


def test_step_function_sees_each_beams_prefix():
    model = TinyLM(seed=1)
    options = BeamOptions(beam_size=3, eos_token=0, length_penalty_alpha=0.5)
    seen = []
    inner = model.step_fn(2)

    def step(beams):
        seen.append(beams)
        return inner(beams)

    result = beam_search(step, options, max_steps=5, batch_size=2)
    trace = result.trace
    assert seen[0].step == 0 and seen[0].sequences.shape == (2, 3, 0)
    assert torch.equal(seen[0].active, torch.tensor([[True, False, False]] * 2))
    for beams in seen[1:]:
        t = beams.step
        # The prefix of slot k after step t - 1, from the trace.
        prefix = decode(torch.stack(result.step_log_probs[:t], 1).detach(), options)
        assert torch.equal(beams.sequences, expected_paths(prefix))
        assert torch.equal(beams.parents, trace.parents[:, t - 1])
        assert torch.equal(beams.tokens, trace.tokens[:, t - 1])
        assert torch.equal(beams.lengths, trace.lengths[:, t - 1])
        assert torch.equal(beams.scores, trace.scores[:, t - 1])


def test_gradients_reach_the_model():
    model = TinyLM(seed=2)
    options = BeamOptions(beam_size=3, eos_token=3, length_penalty_alpha=0.7)
    weights = torch.tensor([[1.0, -0.5, 0.25]])
    result = beam_search(model.step_fn(1), options, max_steps=5)
    (result.scores * weights).sum().backward(retain_graph=True)  # the reference below reuses the graph
    got = [p.grad.clone() for p in model.parameters()]

    # The same loss written directly: each final beam's path through the rows.
    model.zero_grad()
    rows = torch.stack(result.step_log_probs, 1)
    trace = result.trace
    penalty = ((5.0 + trace.final_lengths.clamp(min=1).double()) / 6.0) ** 0.7
    loss = rows.new_zeros(())
    for k in range(3):
        beam, total = k, rows.new_zeros(())
        for t in range(rows.shape[1] - 1, -1, -1):
            parent, token = int(trace.parents[0, t, beam]), int(trace.tokens[0, t, beam])
            if trace.from_logprob[0, t, beam]:
                total = total + rows[0, t, parent, token]
            beam = parent
        loss = loss + weights[0, k] * total / penalty[0, k].float()
    loss.backward()
    for g, p in zip(got, model.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad, rtol=1e-5, atol=1e-6)
    assert any(g.abs().sum() > 0 for g in got)


def test_stops_once_every_beam_has_finished():
    options = BeamOptions(beam_size=2, eos_token=0)
    calls = []

    def step(beams):
        calls.append(beams.step)
        lp = torch.full((1, 2, 5), -5.0)
        lp[..., 0] = -0.01  # EOS dominates
        return lp.log_softmax(-1)

    result = beam_search(step, options, max_steps=10)
    assert len(result.step_log_probs) == len(calls) < 10
    assert result.sequences.shape == (1, 2, 10)
    assert bool(result.trace.lengths[:, -1].gt(0).all())
    assert torch.equal(result.scores, final_scores(torch.stack(result.step_log_probs, 1), options))


def test_gradient_equals_final_scores_of_the_stacked_rows():
    model = TinyLM(seed=3)
    options = BeamOptions(beam_size=4, eos_token=1, length_penalty_alpha=0.6, no_repeat_ngram_size=2)
    result = beam_search(model.step_fn(3), options, max_steps=6, batch_size=3)
    rows = result.step_log_probs
    weights = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    got = torch.autograd.grad((result.scores * weights).sum(), rows, retain_graph=True)
    stacked = torch.stack(rows, 1)
    expected = torch.autograd.grad((final_scores(stacked, options) * weights).sum(), rows)
    for g, e in zip(got, expected, strict=True):
        assert torch.equal(g, e)


def test_no_grad_keeps_no_graph_and_no_rows():
    model = TinyLM(seed=4)
    with torch.no_grad():
        result = beam_search(model.step_fn(2), BeamOptions(beam_size=3), max_steps=4, batch_size=2)
        kept = beam_search(model.step_fn(2), BeamOptions(beam_size=3), max_steps=4, batch_size=2, return_log_probs=True)
    assert not result.scores.requires_grad
    assert result.step_log_probs == ()  # inference holds one step's rows at a time
    assert torch.equal(result.scores, kept.scores)
    assert torch.equal(kept.scores, final_scores(torch.stack(kept.step_log_probs, 1), BeamOptions(beam_size=3)))


class TinyLMRescorer:
    """Teacher-forced token log-probabilities of TinyLM, for rescore_fn."""

    def __init__(self, model):
        self.model = model

    def __call__(self, sequences, lengths):
        B, K, T = sequences.shape
        tokens = sequences.reshape(B * K, T).clamp(min=0)
        h0 = torch.zeros(B * K, 16, dtype=self.model.out.weight.dtype)
        h = self.model.gru(self.model.embed(torch.full((B * K,), self.model.bos)), h0)
        out = []
        for t in range(T):
            out.append(self.model.out(h).log_softmax(-1).gather(1, tokens[:, t : t + 1])[:, 0])
            h = self.model.gru(self.model.embed(tokens[:, t]), h)
        return torch.stack(out, 1).view(B, K, T)


@pytest.mark.parametrize(
    "options", [BeamOptions(beam_size=3), BeamOptions(beam_size=4, eos_token=1, length_penalty_alpha=0.8)]
)
def test_rescoring_gives_the_same_gradient(options):
    model = TinyLM(seed=5).double()
    step = model.step_fn(2)

    def step64(beams):
        return step(beams).double()

    through_steps = beam_search(step64, options, max_steps=6, batch_size=2)
    rescored = beam_search(step64, options, max_steps=6, batch_size=2, rescore_fn=TinyLMRescorer(model))
    assert rescored.step_log_probs == ()
    assert torch.equal(rescored.scores.detach(), through_steps.scores.detach())  # the search's own scores
    assert torch.equal(rescored.sequences, through_steps.sequences)
    weights = torch.linspace(-1.0, 1.0, options.beam_size).double()
    live = torch.isfinite(through_steps.scores)
    a = torch.autograd.grad((through_steps.scores[live] * weights.expand(2, -1)[live]).sum(), list(model.parameters()))
    b = torch.autograd.grad((rescored.scores[live] * weights.expand(2, -1)[live]).sum(), list(model.parameters()))
    # Through the steps, the engine computes each entry's gradient in float32.
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_rescore_fn_shape_is_checked():
    model = TinyLM(seed=6)
    with pytest.raises(ValueError, match="rescore_fn"):
        beam_search(model.step_fn(1), BeamOptions(beam_size=2), max_steps=3, rescore_fn=lambda s, n: torch.zeros(1, 2))


def test_length_penalty_matches_the_search():
    trace = decode(
        torch.randn(2, 6, 3, 9).log_softmax(-1), BeamOptions(beam_size=3, eos_token=1, length_penalty_alpha=0.7)
    )
    penalty = beamgrad.length_penalty(trace.final_lengths, 0.7)
    assert penalty.dtype == torch.float32
    assert torch.equal(trace.final_raw_scores / penalty, trace.final_scores)
    assert torch.equal(beamgrad.length_penalty(torch.tensor([0, 1, 7]), 0.0), torch.ones(3))


def test_argument_errors():
    options = BeamOptions(beam_size=2)
    good = lambda beams: torch.zeros(1, 2, 4).log_softmax(-1)  # noqa: E731
    with pytest.raises(ValueError, match="max_steps"):
        beam_search(good, options, max_steps=0)
    with pytest.raises(ValueError, match="batch_size"):
        beam_search(good, options, max_steps=2, batch_size=2**32 + 1)
    with pytest.raises(ValueError, match=r"\[B, K, V\]"):
        beam_search(lambda beams: torch.zeros(1, 3, 4), options, max_steps=2)
    with pytest.raises(TypeError):
        beam_search(lambda beams: torch.zeros(1, 2, 4, dtype=torch.long), options, max_steps=2)
    with pytest.raises(ValueError, match="vocabulary"):
        beam_search(lambda beams: torch.zeros(1, 2, 4 + beams.step).log_softmax(-1), options, max_steps=3)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        beam_search(good, BeamOptions(beam_size=2, eos_token=4), max_steps=2)


def test_integer_arguments_accept_numpy_integers():
    # Like BeamOptions: any integer type but bool.
    np = pytest.importorskip("numpy")
    model = TinyLM(seed=7)
    options = BeamOptions(beam_size=2)
    reference = beam_search(model.step_fn(2), options, max_steps=4, batch_size=2)
    result = beam_search(model.step_fn(2), options, max_steps=np.int64(4), batch_size=np.int32(2))
    assert torch.equal(result.sequences, reference.sequences)
    assert result.sequences.shape == (2, 2, 4)
    for bad in (True, 4.0, np.float64(4), "4"):
        with pytest.raises(ValueError, match="max_steps"):
            beam_search(model.step_fn(2), options, max_steps=bad, batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        beam_search(model.step_fn(1), options, max_steps=4, batch_size=np.bool_(True))


def test_sequence_scores_rejects_lengths_outside_the_sequences():
    lp = torch.randn(3, 5)
    options = BeamOptions(beam_size=1, length_penalty_alpha=0.6)
    beamgrad.sequence_scores(lp, torch.tensor([5, 0, 2]), options)  # the full length and empty are fine
    for lengths in ([6, 0, 2], [5, -1, 2]):
        with pytest.raises(ValueError, match=r"lengths must be in \[0, 5\]"):
            beamgrad.sequence_scores(lp, torch.tensor(lengths), options)


def test_sequence_scores_length_check_stays_out_of_traced_code():
    from torch._subclasses.fake_tensor import FakeTensorMode

    lp, lengths = torch.randn(3, 5), torch.tensor([5, 0, 2])
    options = BeamOptions(beam_size=1, length_penalty_alpha=0.6)
    expected = beamgrad.sequence_scores(lp, lengths, options)
    compiled = torch.compile(beamgrad.sequence_scores, fullgraph=True, backend="aot_eager")
    assert torch.equal(compiled(lp, lengths, options), expected)
    with FakeTensorMode():  # no values to check
        assert beamgrad.sequence_scores(torch.randn(3, 5), torch.tensor([9, 0, 2]), options).shape == (3,)
    batched = torch.vmap(lambda x, n: beamgrad.sequence_scores(x, n, options))(lp[:, None], lengths[:, None])
    torch.testing.assert_close(batched[:, 0], expected)


# ---------------------------------------------------------------------------
# 32-bit limits: out-of-range integers are errors, not truncated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [[4, 2**32 + 1], torch.tensor([4, 2**32 + 1]), [4, -(2**32) + 3]])
def test_step_counts_beyond_int32_are_rejected(steps):
    x = torch.randn(2, 4, 3, 10).log_softmax(-1)
    with pytest.raises(ValueError, match="steps"):
        decode(x, BeamOptions(beam_size=3), steps=steps)
    with pytest.raises(ValueError, match="steps"):
        final_scores(x, BeamOptions(beam_size=3), steps=steps)


def test_fractional_step_counts_are_rejected():
    x = torch.randn(2, 4, 3, 10).log_softmax(-1)
    with pytest.raises(TypeError, match="integers"):
        decode(x, BeamOptions(beam_size=3), steps=torch.tensor([2.5, 3.0]))


@pytest.mark.parametrize(
    "field,value",
    [("beam_size", 2**32 + 1), ("eos_token", 2**32 + 2), ("min_length", 2**32 + 1), ("no_repeat_ngram_size", 2**31)],
)
def test_options_beyond_int32_are_rejected(field, value):
    kwargs = {"beam_size": 2, field: value}
    with pytest.raises(ValueError, match=field):
        BeamOptions(**kwargs)
    with pytest.raises(ValueError, match="banned_tokens"):
        BeamOptions(beam_size=2, banned_tokens=[2**32 + 1])
