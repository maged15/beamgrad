# SPDX-License-Identifier: MIT
"""Per seed, from the same SFT checkpoint and batches: beamgrad MRT and DIY MRT,
each evaluated with two decoders (beamgrad's search and HF generate()).

    python run_pair.py --seeds 8 9 10 11 12 13 --out runs/results_pair.json
"""

import argparse
import json

import diy_mrt as D
import sacrebleu
import torch
import train_mt as T
from transformers import AutoModelForCausalLM


@torch.no_grad()
def evaluate_hf(model, rows, batch=50):
    """Test BLEU of HF generate(num_beams=4, length_penalty=0.6), best beam."""
    model.eval()
    hyps = []
    for i in range(0, len(rows), batch):
        ids, mask = T.prompts(rows[i : i + batch])
        with torch.autocast(**T.AMP):
            seqs, _ = D.generate_beams(model, ids, mask, 96)
        hyps += [T.text(s) for s in seqs[:, 0]]
    model.train()
    return sacrebleu.corpus_bleu(hyps, [[r["de"] for r in rows]]).score, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--ft-steps", type=int, default=500)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--out", default="runs/results_pair.json")
    args = ap.parse_args()
    rows, test_rows = T.load("train"), T.load("test")[: args.test]
    model = AutoModelForCausalLM.from_pretrained(T.NAME, dtype=torch.float32).cuda()
    saved = torch.load("runs/sft.pt", map_location="cpu")
    base = saved["model"]
    print(f"SFT checkpoint runs/sft.pt: test BLEU {saved['sft']:.2f}")
    out = {"args": vars(args), "sft": saved["sft"], "runs": []}
    for seed in args.seeds:
        row = {"seed": seed}
        for method in ("beamgrad", "diy"):
            model.load_state_dict(base)
            print(f"\nseed {seed}: {method} MRT", flush=True)
            if method == "beamgrad":  # exactly as train_mt.py ran it
                model.gradient_checkpointing_disable()
                T.train(model, rows, args.ft_steps, 8, 5e-6, "mrt", seed=100 + seed)
            else:  # exactly as diy_mrt.py ran it
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                D.train(model, rows, args.ft_steps, 8, 5e-6, seed=100 + seed)
                model.gradient_checkpointing_disable()
            row[f"{method}_bgdec"], row[f"{method}_bgdec_hyps"] = T.evaluate(model, test_rows)
            row[f"{method}_hfdec"], row[f"{method}_hfdec_hyps"] = evaluate_hf(model, test_rows)
            print(
                f"  -> test BLEU  beamgrad decoder {row[f'{method}_bgdec']:.2f}   "
                f"generate() decoder {row[f'{method}_hfdec']:.2f}",
                flush=True,
            )
        out["runs"].append(row)
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
