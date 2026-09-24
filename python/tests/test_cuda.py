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
    for name in ("tokens", "parents", "lengths", "from_logprob", "final_lengths", "steps"):
        assert torch.equal(getattr(cuda_out, name).cpu(), getattr(cpu_out, name)), name
    for name in ("final_scores", "final_raw_scores", "scores", "raw_scores"):
        torch.testing.assert_close(
            getattr(cuda_out, name).cpu(), getattr(cpu_out, name), rtol=1e-6, atol=1e-6, msg=name
        )


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
    torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=1e-6, atol=1e-6)


def test_variable_steps_match_cpu():
    x = random_log_probs(4, 7, 3, 50, seed=3)
    options = BeamOptions(beam_size=3, eos_token=5, length_penalty_alpha=0.3)
    steps = [7, 1, 4, 6]
    assert_same_decode(decode(x.cuda(), options, steps=steps), decode(x, options, steps=steps))
    cpu = x.clone().requires_grad_(True)
    final_scores(cpu, options, steps=steps).sum().backward()
    gpu = x.cuda().requires_grad_(True)
    final_scores(gpu, options, steps=torch.tensor(steps, device="cuda")).sum().backward()
    torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=1e-6, atol=1e-6)


def test_unbatched_and_half_precision():
    x = random_log_probs(5, 3, 257, seed=4)
    options = BeamOptions(beam_size=3)
    torch.testing.assert_close(final_scores(x.cuda(), options).cpu(), final_scores(x, options), rtol=1e-6, atol=1e-6)
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
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


def test_debug_synchronization(monkeypatch):
    monkeypatch.setenv("DBS_CUDA_SYNC_CHECK", "1")
    x = random_log_probs(2, 3, 2, 64, seed=6)
    torch.testing.assert_close(
        final_scores(x.cuda(), BeamOptions(beam_size=2)).cpu(), final_scores(x, BeamOptions(beam_size=2))
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_second_device():
    x = random_log_probs(2, 4, 3, 100, seed=8)
    y = final_scores(x.to("cuda:1"), BeamOptions(beam_size=3))
    assert y.device == torch.device("cuda:1")
    torch.testing.assert_close(y.cpu(), final_scores(x, BeamOptions(beam_size=3)), rtol=1e-6, atol=1e-6)


def test_errors():
    with pytest.raises(ValueError, match="maximum"):
        final_scores(torch.zeros(1, 2, 1025, 3, device="cuda"), BeamOptions(beam_size=1025))
    x = random_log_probs(3, 2, 16).cuda()
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        final_scores(x, BeamOptions(beam_size=2))
