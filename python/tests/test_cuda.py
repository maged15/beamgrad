# SPDX-License-Identifier: MIT
"""CUDA backend tests: native results must match the CPU backend.

Skipped unless the CUDA operators are built and a GPU is present. The CUDA
engine is additionally exercised without a GPU by the C++ emulation tests
(tests/cuda_emulation_tests.cpp).
"""

import pytest
import torch

import beamgrad
from beamgrad import BeamOptions, decode, final_scores

pytestmark = pytest.mark.skipif(not beamgrad.cuda_available(), reason="beamgrad CUDA operators or GPU unavailable")


def random_log_probs(*shape, seed=0, ties=False):
    g = torch.Generator().manual_seed(seed)
    if ties:
        return -0.5 * torch.randint(0, 4, shape, generator=g).float()
    return torch.log_softmax(torch.randn(*shape, generator=g) * 2.0, dim=-1)


CASES = [
    # (shape [B, T, K, V], options)
    ((1, 4, 2, 128), BeamOptions(beam_size=2)),
    ((3, 6, 4, 1024), BeamOptions(beam_size=4, eos_token=2)),
    ((2, 8, 8, 32000), BeamOptions(beam_size=8, eos_token=7, min_length=3, length_penalty_alpha=0.6)),
    ((2, 5, 33, 300), BeamOptions(beam_size=33, eos_token=1, length_penalty_alpha=1.0)),
    ((1, 3, 64, 5000), BeamOptions(beam_size=64, eos_token=0)),
    ((1, 2, 1024, 7), BeamOptions(beam_size=1024)),
]


def assert_same_decode(cuda_out, cpu_out):
    # Bit for bit: the backends share the length penalty and use no fused multiply-add.
    for name in cuda_out._fields:
        assert torch.equal(getattr(cuda_out, name).cpu(), getattr(cpu_out, name)), name


@pytest.mark.parametrize("shape,options", CASES)
@pytest.mark.parametrize("ties", [False, True])
def test_decode_matches_cpu(shape, options, ties):
    x = random_log_probs(*shape, seed=sum(shape), ties=ties)
    assert_same_decode(decode(x.cuda(), options), decode(x, options))


@pytest.mark.parametrize("shape,options", CASES)
def test_gradients_match_cpu(shape, options):
    x = random_log_probs(*shape, seed=7)
    weights = torch.linspace(-1.0, 1.0, shape[2])
    cpu = x.clone().requires_grad_(True)
    (final_scores(cpu, options) * weights).sum().backward()
    gpu = x.cuda().requires_grad_(True)
    y = final_scores(gpu, options)
    assert y.is_cuda
    (y * weights.cuda()).sum().backward()
    assert gpu.grad.is_cuda
    assert torch.equal(gpu.grad.cpu(), cpu.grad)


def test_variable_steps_match_cpu():
    x = random_log_probs(4, 7, 3, 50, seed=3)
    options = BeamOptions(beam_size=3, eos_token=5, length_penalty_alpha=0.3)
    steps = [7, 1, 4, 6]
    assert_same_decode(decode(x.cuda(), options, steps=steps), decode(x, options, steps=steps))
    cpu = x.clone().requires_grad_(True)
    final_scores(cpu, options, steps=steps).sum().backward()
    gpu = x.cuda().requires_grad_(True)
    final_scores(gpu, options, steps=torch.tensor(steps, device="cuda")).sum().backward()
    assert torch.equal(gpu.grad.cpu(), cpu.grad)


def test_unbatched_and_half_precision():
    x = random_log_probs(5, 3, 257, seed=4)
    options = BeamOptions(beam_size=3)
    assert torch.equal(final_scores(x.cuda(), options).cpu(), final_scores(x, options))
    for dtype in (torch.float16, torch.bfloat16):
        xh = x.cuda().to(dtype).requires_grad_(True)
        final_scores(xh, options).sum().backward()
        assert xh.grad.dtype == dtype


def test_runs_on_the_current_stream():
    x = random_log_probs(2, 5, 3, 257, seed=5).cuda()
    options = BeamOptions(beam_size=3, eos_token=7)
    expected = final_scores(x, options)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = final_scores(x, options)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y, expected)


def test_debug_synchronization(monkeypatch):
    monkeypatch.setenv("DBS_CUDA_SYNC_CHECK", "1")
    x = random_log_probs(2, 3, 2, 64, seed=6)
    assert torch.equal(
        final_scores(x.cuda(), BeamOptions(beam_size=2)).cpu(), final_scores(x, BeamOptions(beam_size=2))
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_second_device():
    x = random_log_probs(2, 4, 3, 100, seed=8)
    y = final_scores(x.to("cuda:1"), BeamOptions(beam_size=3))
    assert y.device == torch.device("cuda:1")
    assert torch.equal(y.cpu(), final_scores(x, BeamOptions(beam_size=3)))


def test_errors():
    with pytest.raises(ValueError, match="maximum"):
        final_scores(torch.zeros(1, 2, 1025, 3, device="cuda"), BeamOptions(beam_size=1025))
    x = random_log_probs(3, 2, 16).cuda()
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        final_scores(x, BeamOptions(beam_size=2))


@pytest.mark.parametrize("K", [3, 16, 17, 40])
def test_constraints_match_cpu(K):
    x = random_log_probs(2, 9, K, 12, seed=K, ties=True)
    options = BeamOptions(
        beam_size=K,
        eos_token=11,
        length_penalty_alpha=0.5,
        banned_tokens=[2, 7],
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
    )
    assert_same_decode(decode(x.cuda(), options), decode(x, options))
    cpu = x.clone().requires_grad_(True)
    final_scores(cpu, options).sum().backward()
    gpu = x.cuda().requires_grad_(True)
    final_scores(gpu, options).sum().backward()
    assert torch.equal(gpu.grad.cpu(), cpu.grad)


def test_validation_on_device():
    x = random_log_probs(2, 3, 2, 16).cuda()
    x[0, 0, 1, 4] = float("nan")  # beam 1 is not live at step 0: never read
    final_scores(x, BeamOptions(beam_size=2))
    x[1, 1, 0, 3] = float("inf")
    with pytest.raises(ValueError, match="example 1"):
        final_scores(x, BeamOptions(beam_size=2))


def test_torch_func_grad_matches_cpu():
    options = BeamOptions(beam_size=3, eos_token=2, banned_tokens=[5])
    x = random_log_probs(2, 5, 3, 64, seed=10)
    cpu = torch.func.grad(lambda y: final_scores(y, options).sum())(x)
    gpu = torch.func.grad(lambda y: final_scores(y, options).sum())(x.cuda())
    assert torch.equal(gpu.cpu(), cpu)


def test_torch_compile():
    options = BeamOptions(beam_size=4, eos_token=3)
    x = random_log_probs(2, 6, 4, 300, seed=9).cuda()
    compiled = torch.compile(lambda y: final_scores(y, options), fullgraph=True)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    final_scores(a, options).sum().backward()
    compiled(b).sum().backward()
    assert torch.equal(a.grad, b.grad)


def test_beam_search_matches_cpu():
    torch.manual_seed(0)
    embed, head = torch.nn.Embedding(20, 8), torch.nn.Linear(8, 19)

    def step_on(device):
        def step(beams):
            # Rows are computed on the CPU from the beam state, then moved, so
            # both searches see identical rows.
            prefix = beams.sequences.cpu().clamp(min=0)
            h = embed(torch.full(prefix.shape[:2], 19)) + embed(prefix).sum(2)
            return head(torch.tanh(h)).log_softmax(-1).to(device)

        return step

    options = BeamOptions(
        beam_size=5, eos_token=3, length_penalty_alpha=0.6, no_repeat_ngram_size=2, repetition_penalty=1.3
    )
    cpu = beamgrad.beam_search(step_on("cpu"), options, max_steps=7, batch_size=3)
    cpu.scores.sum().backward()
    cpu_grad = head.weight.grad.clone()
    head.weight.grad = None
    gpu = beamgrad.beam_search(step_on("cuda"), options, max_steps=7, batch_size=3)
    assert gpu.scores.is_cuda and gpu.sequences.is_cuda
    gpu.scores.sum().backward()
    assert torch.equal(gpu.sequences.cpu(), cpu.sequences)
    assert torch.equal(gpu.scores.detach().cpu(), cpu.scores.detach())
    assert_same_decode(gpu.trace, cpu.trace)
    assert torch.equal(head.weight.grad, cpu_grad)


def test_step_counts_beyond_int32_are_rejected():
    x = random_log_probs(2, 4, 3, 10).cuda()
    with pytest.raises(ValueError, match="steps"):
        decode(x, BeamOptions(beam_size=3), steps=torch.tensor([4, 2**32 + 1], device="cuda"))


def test_decode_step_rejects_live_beams_of_different_lengths():
    # The CUDA scan ranks by raw score, which is the CPU's ranking only when
    # every live, unfinished beam of an example has the same length (with a
    # length penalty). States the search produces always do; with validation
    # a state that does not is rejected instead of silently mis-ranked.
    op = torch.ops.beamgrad.decode_step

    def run(device, lengths, alpha, finished=(0, 0), validate=True, eos=-1):
        lp = torch.full((1, 2, 3), -9.0, device=device)
        lp[0, 0, 0], lp[0, 1, 0] = -1.0, -0.5
        raw = torch.tensor([[-2.0, -6.0]], device=device)
        state = (
            raw,
            torch.tensor([lengths], dtype=torch.int32, device=device),
            torch.tensor([finished], dtype=torch.uint8, device=device),
            torch.zeros(1, 2, 0, dtype=torch.int32, device=device),
        )
        return op(lp, *state, eos, 0, alpha, None, 0, 1.0, validate)

    with pytest.raises(ValueError, match="example 0: .* same length .* from 1 to 5"):
        run("cuda", [1, 5], 3.0)
    # Without EOS handling the finished flag is ignored, so that beam is live too.
    with pytest.raises(ValueError, match="same length"):
        run("cuda", [1, 5], 3.0, finished=(0, 1), eos=-1)
    # Equal lengths, no length penalty, or a finished beam (carried forward
    # with EOS) of another length: CUDA selects exactly what the CPU selects.
    for lengths, alpha, finished, eos in (
        ([3, 3], 3.0, (0, 0), -1),
        ([1, 5], 0.0, (0, 0), -1),
        ([1, 5], 3.0, (0, 1), 2),
    ):
        cpu = run("cpu", lengths, alpha, finished, eos=eos)
        gpu = run("cuda", lengths, alpha, finished, eos=eos)
        for a, b in zip(cpu, gpu, strict=True):
            assert torch.equal(a, b.cpu())
    run("cuda", [1, 5], 3.0, validate=False)  # unchecked without validation, as documented
