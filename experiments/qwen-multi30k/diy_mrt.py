# SPDX-License-Identifier: MIT
"""Minimum-risk training WITHOUT beamgrad: HF generate() + teacher-forced re-scoring.

The loss is the same as beamgrad.losses.minimum_risk on beamgrad's scores:
softmax over the K beams of (sum of token log-probs) / ((5 + L) / 6) ** 0.6,
expected (1 - sentence BLEU), + 0.3 x token NLL. Only the beams (from
generate()) and the plumbing differ. Starts from the same SFT checkpoint and
the same batches as train_mt.py's seeds, and is evaluated by the same decoder.

    python diy_mrt.py --seeds 3 4 5 6 7 --out runs/results_diy.json
"""

from __future__ import annotations

import argparse
import json
import random
import time

import sacrebleu
import torch
import train_mt as T  # tokenizer, data, prompts, evaluate() -- evaluation only uses beamgrad
from transformers import AutoModelForCausalLM

tok, EOS = T.tok, T.EOS
BAN = tok.convert_tokens_to_ids("<|endoftext|>")
K, ALPHA = 4, 0.6


def teacher_forced(model, ids, mask, seqs, lengths):
    """log p of each token of seqs [N, T] given its prompt, [N, T], 0 at or past lengths."""
    N, L = seqs.shape
    gen_mask = (torch.arange(L, device=seqs.device)[None] < lengths[:, None]).to(mask.dtype)
    full = torch.cat([mask, gen_mask], 1)
    pos = (full.cumsum(-1) - 1).masked_fill(full == 0, 1)
    logits = model(
        input_ids=torch.cat([ids, seqs.clamp(min=0)], 1),
        attention_mask=full,
        position_ids=pos,
        use_cache=False,
        logits_to_keep=L + 1,
    ).logits[:, :L]
    lp = logits.float().log_softmax(-1).gather(-1, seqs.clamp(min=0)[..., None])[..., 0]
    return lp * gen_mask


def generate_beams(model, ids, mask, max_new):
    """HF beam search: K beams per prompt, [B, K, T] (-1 after EOS) and lengths [B, K] (EOS counted)."""
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        num_beams=K,
        num_return_sequences=K,
        max_new_tokens=max_new,
        min_new_tokens=2,
        length_penalty=ALPHA,
        eos_token_id=EOS,
        pad_token_id=BAN,
        bad_words_ids=[[BAN]],
        do_sample=False,
        repetition_penalty=1.0,
        temperature=None,
        top_p=None,
        top_k=None,
    )[:, ids.shape[1] :]
    is_eos = out == EOS
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1) + 1, torch.full_like(out[:, 0], out.shape[1]))
    seqs = out.masked_fill(torch.arange(out.shape[1], device=out.device)[None] >= first[:, None], -1)
    return seqs.view(ids.shape[0], K, -1), first.view(ids.shape[0], K)


def mrt_loss(model, ids, mask, seqs, lengths, costs):
    B = ids.shape[0]
    lp = teacher_forced(
        model, ids.repeat_interleave(K, 0), mask.repeat_interleave(K, 0), seqs.reshape(B * K, -1), lengths.reshape(-1)
    )
    scores = (lp.sum(-1) / ((5 + lengths.reshape(-1).float()) / 6) ** ALPHA).view(B, K)
    return (scores.softmax(-1) * costs).sum(-1).mean()


def nll(model, ids, mask, gold, glen):
    lp = teacher_forced(model, ids, mask, gold, glen)
    return -lp.sum() / glen.sum()


def costs_of(seqs, rows):
    return torch.tensor(
        [
            [1 - sacrebleu.sentence_bleu(T.text(s), [r["de"]]).score / 100 for s in beams]
            for beams, r in zip(seqs, rows, strict=True)
        ],
        device="cuda",
    )


def diagnose(model, rows, batches=10, batch=8):
    """Same batches, same model: beams, gradients and speed of beamgrad vs generate()."""
    import beamgrad
    from beamgrad.hf import CausalLMRescorer, CausalLMStep

    rng = random.Random(0)
    same_set = same_best = n = 0
    t_bg = t_hf = 0.0
    for i in range(batches):
        chunk = rng.sample(rows, batch)
        ids, mask = T.prompts(chunk)
        gold, _ = T.golds(chunk)
        max_new = int(gold.shape[1] * 1.5) + 4
        model.eval()
        with torch.no_grad(), torch.autocast(**T.AMP):
            torch.cuda.synchronize()
            t = time.time()
            res = beamgrad.beam_search(CausalLMStep(model, ids, mask, K), T.OPTS, max_steps=max_new, batch_size=batch)
            torch.cuda.synchronize()
            t_bg += time.time() - t if i else 0.0
            t = time.time()
            hf_seqs, _ = generate_beams(model, ids, mask, max_new)
            torch.cuda.synchronize()
            t_hf += time.time() - t if i else 0.0
        for b in range(batch):
            a = {T.text(s) for s in res.sequences[b]}
            h = {T.text(s) for s in hf_seqs[b]}
            same_set += a == h
            same_best += T.text(res.sequences[b, 0]) == T.text(hf_seqs[b, 0])
            n += 1
        if i == 0:  # gradient: beamgrad's minimum_risk vs the DIY loss, on beamgrad's own beams
            # The search needs the KV cache (off in train mode with checkpointing); no dropout, so the same function.
            model.eval()
            costs = costs_of(res.sequences, chunk)
            with torch.autocast(**T.AMP):
                model.zero_grad()
                r = beamgrad.beam_search(
                    CausalLMStep(model, ids, mask, K),
                    T.OPTS,
                    max_steps=max_new,
                    batch_size=batch,
                    rescore_fn=CausalLMRescorer(model, ids, mask),
                )
                beamgrad.losses.minimum_risk(r, costs).backward()
                g1 = torch.cat([p.grad.flatten().double().cpu() for p in model.parameters()])
                model.zero_grad()
                mrt_loss(model, ids, mask, r.sequences, r.lengths, costs).backward()
                g2 = torch.cat([p.grad.flatten().double().cpu() for p in model.parameters()])
            cos = torch.nn.functional.cosine_similarity(g1, g2, 0).item()
            rel = ((g1 - g2).norm() / g1.norm()).item()
            model.zero_grad(set_to_none=True)
            del g1, g2
    print(f"diagnostic on {n} training prompts:")
    print(
        f"  generate() returned the same 4 beams as beamgrad: {same_set / n:.1%};  same best beam: {same_best / n:.1%}"
    )
    print(f"  gradient, beamgrad minimum_risk vs DIY loss on the same beams: cosine {cos:.6f}, rel diff {rel:.2e}")
    print(
        f"  search time per batch of {batch} (warm, bf16): beamgrad {t_bg / (batches - 1):.3f}s  "
        f"generate() {t_hf / (batches - 1):.3f}s"
    )


def train(model, rows, steps, batch, lr, seed, log_every=50):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30))
    rng = random.Random(seed)
    t0, bleus = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        chunk = rng.sample(rows, batch)
        ids, mask = T.prompts(chunk)
        gold, glen = T.golds(chunk)
        model.eval()
        with torch.no_grad(), torch.autocast(**T.AMP):
            seqs, lengths = generate_beams(model, ids, mask, int(gold.shape[1] * 1.5) + 4)
        costs = costs_of(seqs, chunk)
        bleus.append(1 - costs[:, 0].mean().item())
        model.train()
        with torch.autocast(**T.AMP):
            token_nll = nll(model, ids, mask, gold, glen)
            risk = mrt_loss(model, ids, mask, seqs, lengths, costs)
            loss = risk + 0.3 * token_nll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(
                f"  [diy] step {step:4d}  nll {token_nll.item():.3f}  risk {risk.item():.3f}  "
                f"beam0 sBLEU {sum(bleus[-log_every:]) / len(bleus[-log_every:]):.3f}  "
                f"{(time.time() - t0) / (step + 1):.2f}s/step  "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--ft-steps", type=int, default=500)
    ap.add_argument("--ft-batch", type=int, default=8)
    ap.add_argument("--sft-ckpt", default="runs/sft.pt")
    ap.add_argument("--out", default="runs/results_diy.json")
    args = ap.parse_args()
    rows, test_rows = T.load("train"), T.load("test")
    model = AutoModelForCausalLM.from_pretrained(T.NAME, dtype=torch.float32).cuda()
    saved = torch.load(args.sft_ckpt, map_location="cpu")
    base = saved["model"]
    model.load_state_dict(base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # train mode only
    print(f"SFT checkpoint {args.sft_ckpt}: test BLEU {saved['sft']:.2f}")
    diagnose(model, rows)
    out = {"args": vars(args), "sft": saved["sft"], "runs": []}
    for seed in args.seeds:
        model.load_state_dict(base)
        print(f"\nDIY MRT seed {seed}")
        train(model, rows, args.ft_steps, args.ft_batch, 5e-6, seed=100 + seed)  # same batches as train_mt.py
        bleu, hyps = T.evaluate(model, test_rows)
        print(
            f"  -> test BLEU {bleu:.2f}  (mean output length {sum(len(h.split()) for h in hyps) / len(hyps):.1f} words)"
        )
        out["runs"].append({"seed": seed, "diy": bleu, "diy_hyps": hyps})
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
