# SPDX-License-Identifier: MIT
"""JAX integration: same values and gradients as the PyTorch API."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import beamgrad  # noqa: E402
import beamgrad.jax as bjax  # noqa: E402
from beamgrad import BeamOptions  # noqa: E402


def random_log_probs(*shape, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=shape).astype(np.float32) * 2.0
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


@pytest.mark.parametrize("shape", [(5, 3, 11), (2, 5, 3, 11)])
def test_matches_torch(shape):
    x = random_log_probs(*shape)
    options = BeamOptions(beam_size=3, eos_token=2, length_penalty_alpha=0.4)
    weights = np.linspace(-1.0, 1.0, 3, dtype=np.float32)

    def loss(y):
        return (bjax.final_scores(y, options) * weights).sum()

    value, grad = jax.value_and_grad(loss)(jnp.asarray(x))
    xt = torch.from_numpy(x).requires_grad_(True)
    expected = (beamgrad.final_scores(xt, options) * torch.from_numpy(weights)).sum()
    expected.backward()
    np.testing.assert_allclose(float(value), expected.item(), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(grad), xt.grad.numpy(), rtol=1e-6, atol=1e-7)


def test_jit():
    x = jnp.asarray(random_log_probs(2, 4, 2, 9, seed=1))
    options = BeamOptions(beam_size=2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda y: bjax.final_scores(y, options))(x)),
        np.asarray(bjax.final_scores(x, options)),
    )


def test_shape_errors():
    with pytest.raises(ValueError):
        bjax.final_scores(jnp.zeros((3, 4)), BeamOptions(beam_size=2))
    with pytest.raises(ValueError):
        bjax.final_scores(jnp.zeros((3, 4, 8)), BeamOptions(beam_size=2))
