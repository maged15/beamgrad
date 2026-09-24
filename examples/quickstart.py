# SPDX-License-Identifier: MIT
"""beamgrad in one screen: decode, inspect the beams, and backpropagate."""

import torch

import beamgrad

device = "cuda" if beamgrad.cuda_available() else "cpu"
torch.manual_seed(0)

# A batch of 2 examples, 6 decoding steps, 4 beams, vocabulary of 50 tokens.
# log_probs[b, t, k] is the next-token distribution for beam k at step t.
B, T, K, V = 2, 6, 4, 50
log_probs = torch.randn(B, T, K, V, device=device).log_softmax(-1).requires_grad_()

options = beamgrad.BeamOptions(beam_size=K, eos_token=0, length_penalty_alpha=0.6)

# Differentiable final scores: [B, K], best beam first.
scores = beamgrad.final_scores(log_probs, options)
print(f"final scores on {device}:\n{scores}")

# Surrogate gradients flow back to the log-probs along each beam's path.
scores[:, 0].sum().backward()
print("non-zero gradient entries:", int((log_probs.grad != 0).sum()))

# The full trace, and each final beam's token sequence.
trace = beamgrad.decode(log_probs, options)
print("best sequences:", beamgrad.backtrack(trace)[:, 0].tolist())
