# SPDX-License-Identifier: MIT
"""beamgrad.beam_search on a Hugging Face causal LM, against `model.generate`.

Runs the same open-weights model two ways and compares them:

1. Parity. Both searches rank by total log-probability and EOS is suppressed
   (`min_new_tokens` in transformers, a banned token in beamgrad), so both are
   plain beam search over the same rows and must return the same beams.
2. Natural decoding with EOS. transformers keeps `num_beams` unfinished beams
   and moves finished hypotheses to a separate pool; beamgrad keeps finished
   hypotheses in their beam slots (see docs/algorithm.md). Both rank by total
   log-probability here; the script reports how often the best hypotheses
   agree and which search found the more probable one when they differ.
3. Training through the search (`--train`). Fine-tunes the model's last layers
   with a structured margin so that a chosen continuation wins beam search,
   the use case beamgrad exists for; `generate` has no gradient to offer.

Timings are medians of `--repeats` runs after a warm-up, on the same device.

    pip install transformers accelerate
    python benchmarks/hf_beam_search.py --model Qwen/Qwen2.5-0.5B --beams 4 --max-new-tokens 32 --train
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import beamgrad

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1905, Albert Einstein published",
    "The three primary colors are",
    "Photosynthesis is the process by which",
    "The quick brown fox",
    "To make a cup of tea, first",
    "The largest planet in the solar system is",
]


class CausalLMStep:
    """A beamgrad step function for a Hugging Face causal LM.

    Step 0 runs the (left-padded) prompts, replicated for each beam slot; later
    steps feed each beam's last token. The key/value cache is reordered by
    `beams.parents` before every step, so each slot continues the right
    hypothesis. Rows are float32 log-softmaxed logits, as `generate` uses.
    """

    def __init__(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, beams: int):
        self.model = model
        self.B, self.K = input_ids.shape[0], beams
        self.input_ids = input_ids.repeat_interleave(beams, 0)
        self.mask = attention_mask.repeat_interleave(beams, 0)
        self.cache = None

    def __call__(self, beams: beamgrad.BeamState) -> torch.Tensor:
        # Model inputs as generate() prepares them (positions, cache positions), so
        # that the logits are the same in low precision too.
        if beams.step == 0:
            positions = (self.mask.cumsum(-1) - 1).masked_fill(self.mask == 0, 1)
            out = self.model(
                input_ids=self.input_ids,
                attention_mask=self.mask,
                position_ids=positions,
                cache_position=torch.arange(self.mask.shape[1], device=self.mask.device),
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            slots = torch.arange(self.B, device=beams.parents.device)[:, None] * self.K
            self.cache.reorder_cache((slots + beams.parents.clamp(min=0)).flatten())
            self.mask = torch.cat([self.mask, self.mask.new_ones(self.mask.shape[0], 1)], dim=1)
            out = self.model(
                input_ids=beams.tokens.clamp(min=0).view(-1, 1),
                attention_mask=self.mask,
                position_ids=self.mask.sum(-1, keepdim=True) - 1,
                cache_position=torch.tensor([self.mask.shape[1] - 1], device=self.mask.device),
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
            )
        self.cache = out.past_key_values
        return out.logits[:, -1].float().log_softmax(-1).view(self.B, self.K, -1)


def timed(fn, repeats: int):
    fn()  # warm-up
    torch.cuda.synchronize()
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times) * 1e3


def hf_beam_search(model, batch, beams: int, new_tokens: int, suppress_eos: bool):
    out = model.generate(
        **batch,
        num_beams=beams,
        num_return_sequences=beams,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens if suppress_eos else 0,
        do_sample=False,
        length_penalty=0.0,  # rank by total log-probability, like beamgrad with length_penalty_alpha=0
        early_stopping=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    B = batch["input_ids"].shape[0]
    generated = out.sequences[:, batch["input_ids"].shape[1] :]
    return generated.view(B, beams, -1), out.sequences_scores.view(B, beams)


def beamgrad_search(model, batch, beams: int, new_tokens: int, eos: int, suppress_eos: bool):
    options = beamgrad.BeamOptions(
        beam_size=beams,
        eos_token=-1 if suppress_eos else eos,
        banned_tokens=(eos,) if suppress_eos else None,
        validate_inputs=False,
    )
    step = CausalLMStep(model, batch["input_ids"], batch["attention_mask"], beams)
    return beamgrad.beam_search(step, options, max_steps=new_tokens, batch_size=batch["input_ids"].shape[0])


def tokens_until_eos(tokens, eos: int, pad: int) -> list[int]:
    out = []
    for token in tokens.tolist():
        if token < 0 or token == pad and token != eos:
            break
        out.append(token)
        if token == eos:
            break
    return out


def exact_ties(result, beams: int, banned: int) -> torch.Tensor:
    """[B] bool: some step had two of its K + 1 best candidates with exactly equal scores.

    Low-precision logits make such ties common. beamgrad breaks them by a fixed
    rule (lower parent slot, then lower token id); torch.topk, which generate
    uses, returns tied values in an unspecified order, so the two searches may
    legitimately keep different beams from there on.
    """
    B = result.scores.shape[0]
    tied = torch.zeros(B, dtype=torch.bool, device=result.scores.device)
    for t, log_probs in enumerate(result.step_log_probs):
        if t == 0:
            parent = torch.full((B, beams), float("-inf"), device=tied.device)
            parent[:, 0] = 0.0
        else:
            parent = result.trace.raw_scores[:, t - 1]
        candidates = parent[..., None] + log_probs.detach().float().index_fill(
            -1, torch.tensor([banned], device=tied.device), float("-inf")
        )
        best = candidates.view(B, -1).topk(beams + 1).values
        tied |= ((best[:, :-1] == best[:, 1:]) & torch.isfinite(best[:, 1:])).any(-1)
    return tied


def compare_parity(model, tokenizer, batch, args) -> None:
    eos = tokenizer.eos_token_id
    (hf_seqs, hf_scores), hf_ms = timed(
        lambda: hf_beam_search(model, batch, args.beams, args.max_new_tokens, True), args.repeats
    )
    with torch.no_grad():
        result, bg_ms = timed(
            lambda: beamgrad_search(model, batch, args.beams, args.max_new_tokens, eos, True), args.repeats
        )
    same_beams = (hf_seqs == result.sequences).all(-1)  # [B, K]
    same_scores = hf_scores == result.scores  # bitwise
    tied = exact_ties(result, args.beams, eos)
    clean = ~tied
    B, K = same_beams.shape
    print("1. Parity: EOS suppressed, both rank by total log-probability")
    print(f"   prompts whose search met an exact score tie: {int(tied.sum())}/{B}")
    print(
        f"   prompts without ties: identical beams {int(same_beams[clean].sum())}/{int(clean.sum()) * K}, "
        f"bitwise-identical scores {int(same_scores[clean].sum())}/{int(clean.sum()) * K}"
    )
    tied_same = int(same_beams[tied].sum())
    print(f"   prompts with ties:    identical beams {tied_same}/{int(tied.sum()) * K} (tie order may differ)")
    print(f"   time: transformers generate {hf_ms:8.1f} ms   beamgrad.beam_search {bg_ms:8.1f} ms")
    for b in range(min(B, 3)):
        print(f"   [{PROMPTS[b]!r}] -> {tokenizer.decode(result.sequences[b, 0])!r}")


def compare_natural(model, tokenizer, batch, args) -> None:
    eos = tokenizer.eos_token_id
    (hf_seqs, hf_scores), hf_ms = timed(
        lambda: hf_beam_search(model, batch, args.beams, args.max_new_tokens, False), args.repeats
    )
    with torch.no_grad():
        result, bg_ms = timed(
            lambda: beamgrad_search(model, batch, args.beams, args.max_new_tokens, eos, False), args.repeats
        )
    B = hf_seqs.shape[0]
    agree, hf_better, bg_better = 0, 0, 0
    for b in range(B):
        hf_best = tokens_until_eos(hf_seqs[b, 0], eos, tokenizer.pad_token_id)
        bg_best = tokens_until_eos(result.sequences[b, 0], eos, tokenizer.pad_token_id)
        if hf_best == bg_best:
            agree += 1
        elif hf_scores[b, 0] > result.scores[b, 0]:
            hf_better += 1
        else:
            bg_better += 1
    print("2. Natural decoding with EOS (both rank by total log-probability)")
    print(f"   same best hypothesis: {agree}/{B}")
    print(f"   more probable when they differ: transformers {hf_better}, beamgrad {bg_better}")
    print(f"   time: transformers generate {hf_ms:8.1f} ms   beamgrad.beam_search {bg_ms:8.1f} ms")


def train(model, tokenizer, args) -> None:
    """Fine-tune the last layers so that a chosen continuation wins beam search by a margin."""
    prompt, target = "My favorite color is", " teal, like the sea at dawn."
    device = next(model.parameters()).device
    batch = tokenizer([prompt], return_tensors="pt").to(device)
    gold = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)  # [1, T]
    T = gold.shape[1]
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [p for layer in model.model.layers[-2:] for p in layer.parameters()] + list(
        model.model.norm.parameters()
    )
    for p in trainable:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    eos = tokenizer.eos_token_id
    options = beamgrad.BeamOptions(beam_size=args.beams, eos_token=-1, banned_tokens=(eos,))

    def gold_log_prob():
        ids = torch.cat([batch.input_ids, gold], dim=1)
        logits = model(input_ids=ids).logits[:, batch.input_ids.shape[1] - 1 : -1].float()
        return logits.log_softmax(-1).gather(-1, gold[..., None]).sum()

    print(f"3. Training through beam search: make {target!r} win after {prompt!r}")
    start = time.perf_counter()
    for update in range(args.train_steps):
        step = CausalLMStep(model, batch.input_ids, batch.attention_mask, args.beams)
        result = beamgrad.beam_search(step, options, max_steps=T)
        is_gold = (result.sequences == gold[:, None]).all(-1)
        rival = result.scores.masked_fill(is_gold, float("-inf")).amax(1)
        gold_score = gold_log_prob()
        loss = torch.relu(rival - gold_score + 1.0).sum()
        best = tokenizer.decode(result.sequences[0, 0])
        print(f"   update {update:2d}  loss {loss.item():7.3f}  gold {gold_score.item():8.3f}  beam search: {best!r}")
        if loss.item() == 0.0:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"   {update + 1} updates in {elapsed:.1f} s; peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch", type=int, default=len(PROMPTS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert beamgrad.cuda_available(), "this benchmark runs on a CUDA device"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).cuda().eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    batch = tokenizer(PROMPTS[: args.batch], return_tensors="pt", padding=True).to("cuda")
    print(
        f"{args.model} ({args.dtype}) on {torch.cuda.get_device_name()}: batch {args.batch}, "
        f"{args.beams} beams, {args.max_new_tokens} new tokens, vocabulary {model.config.vocab_size}"
    )
    compare_parity(model, tokenizer, batch, args)
    compare_natural(model, tokenizer, batch, args)
    if args.train:
        train(model, tokenizer, args)


if __name__ == "__main__":
    main()
