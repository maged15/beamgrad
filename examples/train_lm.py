# SPDX-License-Identifier: MIT
"""Train an autoregressive model through beam search (beamgrad.beam_search).

A small GRU language model, conditioned on an example id, must make a
reference sequence win beam search by a margin. beam_search runs the model
inside the search, one step at a time on the beams' actual prefixes (its
hidden states are reordered by each beam's parent slot, as a key/value cache
would be), and returns scores that are differentiable with respect to the
model. The loss compares the best beam that is *not* the reference with the
reference's own teacher-forced score, so training targets the decoder that is
used at test time ("beam search optimization").

    python examples/train_lm.py
"""

import torch
from torch import nn

import beamgrad

EOS = 0


class GRULM(nn.Module):
    def __init__(self, sources: int, vocab: int, hidden: int = 48):
        super().__init__()
        self.source = nn.Embedding(sources, hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def sequence_log_prob(self, src: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        """Teacher-forced log p(gold | src) for [B, T] references padded with -1."""
        h = self.source(src)
        total = h.new_zeros(src.shape[0])
        for t in range(gold.shape[1]):
            token = gold[:, t]
            valid = token >= 0
            log_probs = self.out(h).log_softmax(-1)
            total = total + torch.where(valid, log_probs.gather(1, token.clamp(min=0)[:, None])[:, 0], 0.0)
            h = torch.where(valid[:, None], self.cell(self.embed(token.clamp(min=0)), h), h)
        return total


def main() -> None:
    torch.manual_seed(0)
    B, K, V, T = 4, 4, 12, 7  # examples, beams, vocabulary, maximum steps
    model = GRULM(B, V)
    src = torch.arange(B)
    # Reference sequences of different lengths, ending with EOS, padded with -1.
    gold = torch.full((B, T), -1)
    for b, length in enumerate([3, 5, 4, 6]):
        gold[b, : length - 1] = torch.randint(1, V, (length - 1,))
        gold[b, length - 1] = EOS
    options = beamgrad.BeamOptions(beam_size=K, eos_token=EOS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    def step(beams: beamgrad.BeamState) -> torch.Tensor:
        nonlocal hidden
        if beams.step == 0:
            hidden = model.source(src).repeat_interleave(K, 0)  # [B*K, H]
        else:
            # Slot k continues the hypothesis in slot parents[b, k]: reorder, then advance.
            order = (torch.arange(B)[:, None] * K + beams.parents.clamp(min=0)).flatten()
            hidden = model.cell(model.embed(beams.tokens.clamp(min=0).flatten()), hidden[order])
        return model.out(hidden).log_softmax(-1).view(B, K, V)

    hidden = None
    for update in range(301):
        result = beamgrad.beam_search(step, options, max_steps=T, batch_size=B)
        # No length penalty, so the summed log-probability is on the beams' scale
        # (otherwise: beamgrad.sequence_scores).
        gold_score = model.sequence_log_prob(src, gold)
        loss = beamgrad.losses.structured_margin(result, gold, gold_score, margin=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 50 == 0:
            won = int(beamgrad.losses.matches(result.sequences, gold)[:, 0].sum())
            print(f"update {update:3d}  loss {loss.item():.4f}  references won {won}/{B}")
        if loss.item() == 0.0:
            print(f"update {update:3d}  loss 0: every reference beats its best rival by the margin")
            break

    with torch.no_grad():
        best = beamgrad.beam_search(step, options, max_steps=T, batch_size=B).sequences[:, 0]
    for b in range(B):
        print(f"example {b}: reference {gold[b].tolist()}  beam search {best[b].tolist()}")
    assert torch.equal(best, gold), "training did not converge"


if __name__ == "__main__":
    main()
