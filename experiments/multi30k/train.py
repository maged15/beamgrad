# SPDX-License-Identifier: MIT
"""Multi30k En->De: MLE pretraining, then fine-tuning with one training objective.

    python experiments/multi30k/train.py pretrain --seed 0
    python experiments/multi30k/train.py finetune --arm mrt --seed 0

Every fine-tuning arm starts from the same pretrained checkpoint (per seed)
and differs only in its loss; see README.md for the protocol. Each run writes
metrics.json into its directory under --out.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from seq2seq import (
    BOS,
    EOS,
    PAD,
    UNK,
    BeamStep,
    Rescorer,
    Transformer,
    batches,
    corpus_bleu,
    encode_pairs,
    load_split,
    max_steps,
    sentence_bleu,
    strip,
    train_tokenizer,
    translate,
)

import beamgrad
from beamgrad import estimators, losses

ARMS = ("mle", "mrt", "margin", "retain")


def setup(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    data = {split: load_split(args.data, split) for split in ("train", "valid", "test")}
    tokenizer = train_tokenizer(
        [r["en"] for r in data["train"]] + [r["de"] for r in data["train"]], args.vocab, args.out / "tokenizer.json"
    )
    pairs = {split: encode_pairs(tokenizer, rows) for split, rows in data.items()}
    references = {split: [r["de"] for r in rows] for split, rows in data.items()}
    return device, tokenizer, pairs, references


def label_smoothed_nll(gold_lp, all_lp, gold, smoothing):
    valid = gold >= 0
    smooth = -all_lp.mean(-1)
    loss = (1 - smoothing) * -gold_lp + smoothing * smooth
    return loss.masked_fill(~valid, 0.0).sum() / valid.sum()


def evaluate(model, tokenizer, pairs, references, device, args) -> float:
    hyps = translate(model, pairs, device, beam_size=args.eval_beam, alpha=args.eval_alpha)
    return corpus_bleu(tokenizer, hyps, references)


# ---------------------------------------------------------------------------
# Pretraining (MLE)
# ---------------------------------------------------------------------------


def pretrain(args):
    device, tokenizer, pairs, references = setup(args)
    torch.manual_seed(args.seed)
    run = args.out / f"pretrain-s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    model = Transformer(tokenizer.get_vocab_size(), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / args.warmup, (args.warmup / (s + 1)) ** 0.5)
    )
    generator = torch.Generator().manual_seed(args.seed)
    best, history, start = -1.0, [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in batches(pairs["train"], args.batch_size, device, generator):
            gold_lp, all_lp = model.token_log_probs(batch.src, batch.src_mask, batch.tgt_in, batch.gold)
            loss = label_smoothed_nll(gold_lp, all_lp, batch.gold, args.smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total, count = total + loss.item(), count + 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            bleu = evaluate(model, tokenizer, pairs["valid"], references["valid"], device, args)
            history.append({"epoch": epoch, "train_loss": total / count, "valid_bleu": bleu})
            print(f"seed {args.seed} epoch {epoch:3d}  loss {total / count:.3f}  valid BLEU {bleu:.2f}", flush=True)
            if bleu > best:
                best = bleu
                torch.save(model.state_dict(), run / "model.pt")
    model.load_state_dict(torch.load(run / "model.pt"))
    test = evaluate(model, tokenizer, pairs["test"], references["test"], device, args)
    metrics = {
        "seed": args.seed,
        "valid_bleu": best,
        "test_bleu": test,
        "minutes": (time.perf_counter() - start) / 60,
        "history": history,
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"seed {args.seed}: best valid BLEU {best:.2f}, test BLEU {test:.2f}", flush=True)


# ---------------------------------------------------------------------------
# Fine-tuning arms
# ---------------------------------------------------------------------------


def retention_loss(result, relaxed, gold, gold_lengths):
    """-log relaxed weight of the candidate that extends the reference prefix, while it is in the beam.

    Before step t the reference prefix gold[:t] is either one of the beams (slot
    s) or has been pruned. While it is a beam, its extension by gold[t] is a
    candidate of step t; if that candidate is in the relaxed pool, its relaxed
    membership weight should be 1. Once the reference is pruned the example
    stops contributing (an "early update", as in LaSO / beam-search
    optimisation).
    """
    trace = result.trace
    B, T, _ = trace.tokens.shape
    slot = torch.zeros(B, dtype=torch.long, device=gold.device)  # the empty prefix is slot 0
    total = gold.new_zeros((), dtype=torch.float32)
    count = gold.new_zeros((), dtype=torch.float32)
    for t in range(min(T, gold.shape[1])):
        target = gold[:, t]
        alive = (slot >= 0) & (target >= 0)
        hit = (
            alive[:, None]
            & relaxed.from_logprob[:, t]
            & (relaxed.parents[:, t] == slot[:, None])
            & (relaxed.tokens[:, t] == target[:, None])
        )
        found = hit.any(1)
        weight = (relaxed.weights[:, t] * hit).sum(1)
        total = total + torch.where(found, -weight.clamp_min(1e-12).log(), 0.0).sum()
        count = count + found.sum()
        kept = (
            alive[:, None]
            & trace.from_logprob[:, t]
            & (trace.parents[:, t] == slot[:, None])
            & (trace.tokens[:, t] == target[:, None])
        )
        slot = torch.where(kept.any(1), kept.float().argmax(1), -1)
    return total / count.clamp(min=1), count


def finetune_loss(model, batch, args, options, stats):
    gold_lp, all_lp = model.token_log_probs(batch.src, batch.src_mask, batch.tgt_in, batch.gold)
    nll = label_smoothed_nll(gold_lp, all_lp, batch.gold, args.smoothing)
    if args.arm == "mle":
        return nll
    step = BeamStep(model, batch.src, batch.src_mask, options.beam_size)
    rescore = Rescorer(model, batch.src, batch.src_mask) if args.gradient == "rescore" else None
    result = beamgrad.beam_search(step, options, max_steps(batch), batch_size=batch.size, rescore_fn=rescore)
    if args.arm == "mrt":
        sequences = result.sequences.tolist()
        costs = torch.tensor(
            [
                [1.0 - sentence_bleu(strip(s), ref) for s in beams]
                for beams, ref in zip(sequences, batch.references, strict=True)
            ],
            device=result.scores.device,
        )
        stats["beam_bleu"] = stats.get("beam_bleu", 0.0) + float(1 - costs[:, 0].mean())
        sequence_loss = losses.minimum_risk(result, costs, temperature=args.temperature)
    elif args.arm == "margin":
        reference_scores = beamgrad.sequence_scores(gold_lp, batch.gold_lengths, options)
        sequence_loss = losses.structured_margin(result, batch.gold, reference_scores, margin=args.margin)
        stats["gold_in_beam"] = stats.get("gold_in_beam", 0.0) + float(
            losses.matches(result.sequences, batch.gold).any(1).float().mean()
        )
    else:  # retain
        relaxed = estimators.relaxed_topk(
            result, options, pool_multiplier=args.pool_multiplier, temperature=args.temperature
        )
        sequence_loss, count = retention_loss(result, relaxed, batch.gold, batch.gold_lengths)
        stats["retained_steps"] = stats.get("retained_steps", 0.0) + float(count) / float(batch.gold_lengths.sum())
    return sequence_loss + args.nll_weight * nll


def finetune(args):
    device, tokenizer, pairs, references = setup(args)
    name = args.name or args.arm
    run = args.out / f"{name}-s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = Transformer(tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(args.out / f"pretrain-s{args.seed}" / "model.pt"))
    model.set_dropout(args.dropout)
    options = beamgrad.BeamOptions(
        beam_size=args.beam, eos_token=EOS, length_penalty_alpha=args.alpha, banned_tokens=(PAD, BOS, UNK)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    generator = torch.Generator().manual_seed(1000 + args.seed)
    if args.time_only:
        history, best, best_state = [], float("nan"), None
    else:
        history = [
            {"step": 0, "valid_bleu": evaluate(model, tokenizer, pairs["valid"], references["valid"], device, args)}
        ]
        best, best_state = history[0]["valid_bleu"], copy.deepcopy(model.state_dict())
        print(f"{name} seed {args.seed} step     0  valid BLEU {best:.2f}", flush=True)
    step, step_seconds, stats, losses_sum = 0, [], {}, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        for batch in batches(pairs["train"], args.batch_size, device, generator):
            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            loss = finetune_loss(model, batch, args, options, stats)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - start)
            losses_sum += loss.item()
            step += 1
            if args.time_only:
                if step == args.steps:
                    break
                continue
            if step % args.eval_every == 0 or step == args.steps:
                bleu = evaluate(model, tokenizer, pairs["valid"], references["valid"], device, args)
                entry = {"step": step, "valid_bleu": bleu, "train_loss": losses_sum / args.eval_every}
                entry.update({k: v / args.eval_every for k, v in stats.items()})
                history.append(entry)
                stats, losses_sum = {}, 0.0
                extra = "  ".join(f"{k} {v:.3f}" for k, v in entry.items() if k not in ("step", "valid_bleu"))
                print(f"{name} seed {args.seed} step {step:5d}  valid BLEU {bleu:.2f}  {extra}", flush=True)
                if bleu > best:
                    best, best_state = bleu, copy.deepcopy(model.state_dict())
            if step == args.steps:
                break
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    # The first steps include warm-up (allocator, kernel selection).
    timed = step_seconds[10:] if len(step_seconds) > 20 else step_seconds
    test = float("nan")
    if not args.time_only:
        model.load_state_dict(best_state)
        test = evaluate(model, tokenizer, pairs["test"], references["test"], device, args)
    metrics = {
        "arm": args.arm,
        "name": name,
        "seed": args.seed,
        "valid_bleu": best,
        "final_valid_bleu": history[-1]["valid_bleu"] if history else float("nan"),
        "test_bleu": test,
        "seconds_per_step": sum(timed) / len(timed),
        "peak_gib": peak,
        "history": history,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"{name} seed {args.seed}: best valid BLEU {best:.2f}, test BLEU {test:.2f}, "
        f"{metrics['seconds_per_step']:.3f} s/step, peak {peak or 0:.2f} GiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    here = Path(__file__).resolve().parent
    common.add_argument("--data", type=Path, default=here / "data")
    common.add_argument("--out", type=Path, default=here / "runs")
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--vocab", type=int, default=8000)
    common.add_argument("--smoothing", type=float, default=0.1)
    common.add_argument("--eval-beam", type=int, default=4)
    common.add_argument("--eval-alpha", type=float, default=0.6)

    p = sub.add_parser("pretrain", parents=[common], help="train the MLE model")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=5, help="epochs")
    p.set_defaults(func=pretrain)

    f = sub.add_parser("finetune", parents=[common], help="fine-tune a pretrained model with one objective")
    f.add_argument("--arm", choices=ARMS, required=True)
    f.add_argument("--name", help="run name (default: the arm)")
    f.add_argument("--steps", type=int, default=1000)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--dropout", type=float, default=0.1)
    f.add_argument("--beam", type=int, default=4, help="training beam size")
    f.add_argument("--alpha", type=float, default=0.6, help="training length-penalty alpha")
    f.add_argument("--nll-weight", type=float, default=0.3, help="weight of the token NLL added to sequence losses")
    f.add_argument("--temperature", type=float, default=1.0, help="MRT softmax / relaxed top-k temperature")
    f.add_argument("--margin", type=float, default=1.0)
    f.add_argument("--pool-multiplier", type=int, default=4)
    f.add_argument("--gradient", choices=["steps", "rescore"], default="steps")
    f.add_argument("--eval-every", type=int, default=200, help="steps")
    f.add_argument("--time-only", action="store_true", help="no evaluation: measure time per step and memory")
    f.set_defaults(func=finetune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
