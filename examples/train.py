# SPDX-License-Identifier: MIT
"""Train through beam search with a structured margin loss.

The model must make a gold sequence win the beam search by a margin. Each
update runs beam search and takes the best beam that is *not* the gold
sequence; if it scores within `margin` of the gold sequence, the hinge loss
pushes its score down (through beamgrad's surrogate gradient) and the gold
score up. This is the classic "beam search optimization" recipe: the loss is
defined on beam search's own output, so training targets the decoder that is
used at test time.

    python examples/train.py
"""

import torch
from torch import nn

import beamgrad


class TinyLM(nn.Module):
    """Produces [T, V] next-token log-probabilities (shared by every beam)."""

    def __init__(self, steps: int, vocab: int, hidden: int = 32):
        super().__init__()
        self.context = nn.Parameter(torch.randn(hidden))
        self.position = nn.Embedding(steps, hidden)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hidden, vocab))

    def forward(self) -> torch.Tensor:
        positions = torch.arange(self.position.num_embeddings, device=self.context.device)
        return self.out(self.context + self.position(positions)).log_softmax(-1)


def main() -> None:
    device = "cuda" if beamgrad.cuda_available() else "cpu"
    torch.manual_seed(0)
    steps, beams, vocab, margin = 6, 4, 20, 1.0
    gold = torch.randint(1, vocab, (steps,), device=device)
    model = TinyLM(steps, vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    options = beamgrad.BeamOptions(beam_size=beams)

    for update in range(201):
        per_step = model()  # [T, V]
        log_probs = per_step.unsqueeze(1).expand(steps, beams, vocab)  # [T, K, V]
        scores = beamgrad.final_scores(log_probs, options)  # [K], differentiable
        paths = beamgrad.backtrack(beamgrad.decode(log_probs, options))  # [K, T]
        rival = int((paths != gold).any(dim=1).nonzero()[0])  # best beam that is not gold
        best = scores[rival]
        gold_score = per_step.gather(1, gold[:, None]).sum()
        loss = torch.relu(best - gold_score + margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 40 == 0:
            print(
                f"update {update:3d}  loss {loss.item():.4f}  rival {best.item():8.3f}  gold {gold_score.item():8.3f}"
            )

    with torch.no_grad():
        per_step = model()
        trace = beamgrad.decode(per_step.unsqueeze(1).expand(steps, beams, vocab), options)
        decoded = beamgrad.backtrack(trace)[0]
    print("gold   :", gold.tolist())
    print("decoded:", decoded.tolist())
    assert torch.equal(decoded, gold) and loss.item() == 0.0, "training did not converge"


if __name__ == "__main__":
    main()
