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
    # Same library, same arithmetic: identical results.
    np.testing.assert_array_equal(np.float32(value), np.float32(expected.item()))
    np.testing.assert_array_equal(np.asarray(grad), xt.grad.numpy())


def test_jit():
    x = jnp.asarray(random_log_probs(2, 4, 2, 9, seed=1))
    options = BeamOptions(beam_size=2)
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda y: bjax.final_scores(y, options))(x)),
        np.asarray(bjax.final_scores(x, options)),
    )


def test_matches_torch_with_steps_and_constraints():
    x = random_log_probs(3, 7, 2, 6, seed=3)
    options = BeamOptions(beam_size=2, eos_token=4, banned_tokens=[1], no_repeat_ngram_size=2, repetition_penalty=1.4)
    steps = np.array([7, 3, 5], dtype=np.int32)
    weights = np.array([1.0, -0.5], dtype=np.float32)

    def loss(y):
        return (bjax.final_scores(y, options, steps=steps) * weights).sum()

    value, grad = jax.value_and_grad(loss)(jnp.asarray(x))
    xt = torch.from_numpy(x).requires_grad_(True)
    expected = (beamgrad.final_scores(xt, options, steps=torch.from_numpy(steps)) * torch.from_numpy(weights)).sum()
    expected.backward()
    np.testing.assert_array_equal(np.float32(value), np.float32(expected.item()))
    np.testing.assert_array_equal(np.asarray(grad), xt.grad.numpy())


def test_vmap_matches_a_batched_call():
    x = jnp.asarray(random_log_probs(4, 3, 5, 2, 9, seed=5))  # [N, B, T, K, V]
    options = BeamOptions(beam_size=2, eos_token=1)
    batched = bjax.final_scores(x.reshape(12, 5, 2, 9), options).reshape(4, 3, 2)
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambda y: bjax.final_scores(y, options))(x)), np.asarray(batched))
    grad = jax.grad(lambda y: jax.vmap(lambda z: bjax.final_scores(z, options).sum())(y).sum())(x)
    expected = jax.grad(lambda y: bjax.final_scores(y, options).sum())(x.reshape(12, 5, 2, 9))
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(expected).reshape(x.shape))


def test_nan_and_positive_infinity_are_rejected_unless_disabled():
    x = random_log_probs(3, 2, 8)
    x[1, 1, 3] = np.nan
    with pytest.raises(Exception, match="NaN or \\+inf"):
        np.asarray(bjax.final_scores(jnp.asarray(x), BeamOptions(beam_size=2)))
    out = bjax.final_scores(jnp.asarray(x), BeamOptions(beam_size=2, validate_inputs=False))
    assert out.shape == (2,)


def test_shape_errors():
    with pytest.raises(ValueError):
        bjax.final_scores(jnp.zeros((3, 4)), BeamOptions(beam_size=2))
    with pytest.raises(ValueError):
        bjax.final_scores(jnp.zeros((3, 4, 8)), BeamOptions(beam_size=2))


def test_step_counts_beyond_int32_are_rejected():
    x = jnp.asarray(random_log_probs(2, 4, 3, 10, seed=3))
    options = BeamOptions(beam_size=3)
    with pytest.raises(ValueError, match="steps"):
        bjax.final_scores(x, options, steps=np.array([4, 2**32 + 1], dtype=np.int64))
    with pytest.raises(TypeError, match="integers"):
        bjax.final_scores(x, options, steps=np.array([4.0, 2.0]))
    # Traced values cannot be checked eagerly; out-of-range ones fail in the library.
    with pytest.raises(Exception, match="steps"):
        jax.jit(lambda y, s: bjax.final_scores(y, options, steps=s))(x, jnp.asarray([4, 0])).block_until_ready()
