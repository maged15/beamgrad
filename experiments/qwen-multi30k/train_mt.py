# SPDX-License-Identifier: MIT
"""Fine-tune an LLM (Qwen2.5-0.5B-Instruct) through beam search with beamgrad.

Task: Multi30k English -> German translation, evaluated with corpus BLEU
(sacrebleu) of beam-4 output on test2016.

  0. zero-shot            the instruct model as downloaded
  1. SFT                  supervised fine-tuning (teacher-forced NLL)
  2. from the SFT model, on the same batches:
       continued SFT      control
       MRT                beamgrad.losses.minimum_risk over the beams that
                          beamgrad.beam_search returns, cost = 1 - sentence BLEU,
                          + 0.3 x NLL; gradient via rescore_fn (exact here:
                          Qwen has no dropout)

Run from this directory (see README.md):

    python train_mt.py --sft-steps 1800 --ft-steps 500 --seeds 0 1 2 --sft-ckpt runs/sft.pt --out runs/results.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import beamgrad
from beamgrad.hf import CausalLMRescorer, CausalLMStep

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = (
    "You are a translator. Translate the user's English sentence into German. Reply with the German translation only."
)
DATA = Path(__file__).resolve().parent / "data"
DATA_URL = "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/{split}.jsonl"
AMP = dict(device_type="cuda", dtype=torch.bfloat16)

tok = AutoTokenizer.from_pretrained(NAME, padding_side="left")
EOS = tok.convert_tokens_to_ids("<|im_end|>")
OPTS = beamgrad.BeamOptions(
    beam_size=4,
    eos_token=EOS,
    min_length=2,
    length_penalty_alpha=0.6,
    banned_tokens=(tok.convert_tokens_to_ids("<|endoftext|>"),),
)


def load(split):
    path = DATA / f"{split}.jsonl"
    if not path.exists():  # Multi30k (Elliott et al., 2016), as experiments/multi30k downloads it
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(split=split), path)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def prompts(rows):
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["en"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    return enc.input_ids, enc.attention_mask


def golds(rows):
    ids = [tok(r["de"]).input_ids + [EOS] for r in rows]
    T = max(map(len, ids))
    gold = torch.full((len(ids), T), -1, dtype=torch.long)
    for i, s in enumerate(ids):
        gold[i, : len(s)] = torch.tensor(s)
    return gold.cuda(), torch.tensor([len(s) for s in ids]).cuda()


def text(seq):
    s = seq[(seq >= 0) & (seq != EOS)]
    return tok.decode(s, skip_special_tokens=True).strip()


def nll(model, ids, mask, gold, glen):
    lp = CausalLMRescorer(model, ids, mask, gradient_checkpointing=True)(gold[:, None], glen[:, None])[:, 0]  # [B, T]
    valid = torch.arange(gold.shape[1], device=gold.device)[None] < glen[:, None]
    return -(lp * valid).sum() / valid.sum()


@torch.no_grad()
def evaluate(model, rows, batch=50, show=0):
    model.eval()
    hyps = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        ids, mask = prompts(chunk)
        with torch.autocast(**AMP):
            res = beamgrad.beam_search(
                CausalLMStep(model, ids, mask, OPTS.beam_size), OPTS, max_steps=96, batch_size=len(chunk)
            )
        hyps += [text(s) for s in res.sequences[:, 0]]
    model.train()
    for r, h in list(zip(rows, hyps, strict=True))[:show]:
        print(f"    EN  {r['en']}\n    REF {r['de']}\n    OUT {h}")
    return sacrebleu.corpus_bleu(hyps, [[r["de"] for r in rows]]).score, hyps


def train(model, rows, steps, batch, lr, mode, seed, log_every=50):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    warm = 30
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    rng = random.Random(seed)
    model.train()
    t0, stats = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        chunk = rng.sample(rows, batch)
        ids, mask = prompts(chunk)
        gold, glen = golds(chunk)
        with torch.autocast(**AMP):
            token_nll = nll(model, ids, mask, gold, glen)
            if mode == "sft":
                loss, extra = token_nll, ""
            else:
                model.eval()  # the search needs the KV cache; the rescorer switches to train mode for its pass
                res = beamgrad.beam_search(
                    CausalLMStep(model, ids, mask, OPTS.beam_size),
                    OPTS,
                    max_steps=int(gold.shape[1] * 1.5) + 4,
                    batch_size=batch,
                    rescore_fn=CausalLMRescorer(model, ids, mask, gradient_checkpointing=True),
                )
                model.train()
                bleu = torch.tensor(
                    [
                        [sacrebleu.sentence_bleu(text(s), [r["de"]]).score / 100 for s in beams]
                        for beams, r in zip(res.sequences, chunk, strict=True)
                    ],
                    device="cuda",
                )
                risk = beamgrad.losses.minimum_risk(res, 1 - bleu)
                loss = risk + 0.3 * token_nll
                stats.append(bleu[:, 0].mean().item())
                extra = f"  risk {risk.item():.3f}  beam0 sBLEU {sum(stats[-log_every:]) / len(stats[-log_every:]):.3f}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(
                f"  [{mode}] step {step:4d}  nll {token_nll.item():.3f}{extra}  "
                f"{(time.time() - t0) / (step + 1):.2f}s/step  "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--ft-steps", type=int, default=300)
    ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--ft-batch", type=int, default=8)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/results.json")
    ap.add_argument("--sft-ckpt", default=None, help="load the SFT model from here if it exists, else save it here")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"beamgrad {beamgrad.__version__} (CUDA ops: {beamgrad.cuda_available()})  torch {torch.__version__}  "
        f"{torch.cuda.get_device_name()}"
    )

    train_rows, test_rows = load("train"), load("test")[: args.test]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).cuda()
    print(f"{NAME}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, all trained")
    out = {"args": vars(args)}

    ckpt = Path(args.sft_ckpt) if args.sft_ckpt else None
    if ckpt is not None and ckpt.exists():
        saved = torch.load(ckpt)
        model.load_state_dict(saved["model"])
        out["zero_shot"], out["sft"] = saved["zero_shot"], saved["sft"]
        print(f"loaded SFT checkpoint {ckpt} (zero-shot {out['zero_shot']:.2f}, SFT {out['sft']:.2f})\n")
    else:
        t = time.time()
        out["zero_shot"], _ = evaluate(model, test_rows, show=3)
        print(f"zero-shot test BLEU {out['zero_shot']:.2f}  ({time.time() - t:.0f}s)\n")

        print("SFT")
        train(model, train_rows, args.sft_steps, args.sft_batch, 1e-5, "sft", seed=1)
        out["sft"], _ = evaluate(model, test_rows, show=3)
        print(f"SFT test BLEU {out['sft']:.2f}\n")
        if ckpt is not None:
            torch.save({"model": model.state_dict(), "zero_shot": out["zero_shot"], "sft": out["sft"]}, ckpt)
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out["runs"] = []
    for seed in args.seeds:
        row = {"seed": seed}
        for mode in ("sft", "mrt"):
            model.load_state_dict(base)
            print(f"fine-tune seed {seed}: {'continued SFT' if mode == 'sft' else 'MRT through beam search'}")
            train(model, train_rows, args.ft_steps, args.ft_batch, 5e-6, mode, seed=100 + seed)
            row[mode], hyps = evaluate(model, test_rows, show=3 if mode == "mrt" else 0)
            row[mode + "_mean_len"] = sum(len(h.split()) for h in hyps) / len(hyps)
            row[mode + "_hyps"] = hyps
            print(f"  -> test BLEU {row[mode]:.2f}  (mean output length {row[mode + '_mean_len']:.1f} words)\n")
        out["runs"].append(row)
        json.dump(out, open(args.out, "w"), indent=1)

    ref_len = sum(len(r["de"].split()) for r in test_rows) / len(test_rows)
    print(
        f"summary: zero-shot {out['zero_shot']:.2f}  SFT {out['sft']:.2f}  (reference mean length {ref_len:.1f} words)"
    )
    for r in out["runs"]:
        print(
            f"  seed {r['seed']}: continued SFT {r['sft']:.2f}  MRT {r['mrt']:.2f}  (delta {r['mrt'] - r['sft']:+.2f})"
        )


if __name__ == "__main__":
    main()
