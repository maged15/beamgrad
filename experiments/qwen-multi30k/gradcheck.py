# SPDX-License-Identifier: MIT
"""beamgrad minimum_risk vs the DIY loss on the same beams, in float32 (no autocast).

python gradcheck.py   # needs runs/sft.pt from train_mt.py
"""

import random

import torch
import train_mt as T
from diy_mrt import K, costs_of, mrt_loss
from transformers import AutoModelForCausalLM

import beamgrad
from beamgrad.hf import CausalLMRescorer, CausalLMStep

model = AutoModelForCausalLM.from_pretrained(T.NAME, dtype=torch.float32).cuda().eval()
model.load_state_dict(torch.load("runs/sft.pt", map_location="cpu")["model"])
rows = T.load("train")
rng = random.Random(0)
for trial in range(3):
    chunk = rng.sample(rows, 4)
    ids, mask = T.prompts(chunk)
    gold, _ = T.golds(chunk)
    model.zero_grad()
    r = beamgrad.beam_search(
        CausalLMStep(model, ids, mask, K),
        T.OPTS,
        max_steps=int(gold.shape[1] * 1.5) + 4,
        batch_size=4,
        rescore_fn=CausalLMRescorer(model, ids, mask),
    )
    costs = costs_of(r.sequences, chunk)
    l1 = beamgrad.losses.minimum_risk(r, costs)
    l1.backward()
    g1 = torch.cat([p.grad.flatten().double().cpu() for p in model.parameters()])
    model.zero_grad()
    l2 = mrt_loss(model, ids, mask, r.sequences, r.lengths, costs)
    l2.backward()
    g2 = torch.cat([p.grad.flatten().double().cpu() for p in model.parameters()])
    cos = torch.nn.functional.cosine_similarity(g1, g2, 0).item()
    rel = ((g1 - g2).norm() / g1.norm()).item()
    print(
        f"batch {trial}: loss beamgrad {l1.item():.6f}  DIY {l2.item():.6f}  grad cosine {cos:.8f}  rel diff {rel:.2e}"
    )
