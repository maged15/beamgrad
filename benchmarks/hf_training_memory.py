# SPDX-License-Identifier: MIT
"""Peak GPU memory and time of a training step through beam search, by gradient mode.

One step = beam search on a Hugging Face causal LM plus the backward pass of a
loss on the final scores, with every model parameter trainable. Three ways to
take the gradient are compared (see beamgrad.beam_search):

* steps:    the incremental decode keeps its autograd graph (default).
* rescore:  the search runs without a graph; the gradient comes from one
            teacher-forced pass over the final beams (beamgrad.hf.CausalLMRescorer).
* rescore+checkpointing: the same, with the model's gradient checkpointing on,
            which incremental decoding with a key/value cache cannot use.

Memory is the peak allocated during the step minus what was allocated before
it (the weights), so it includes the parameter gradients but no optimizer
state. `--check` first verifies, in float32, that the three modes produce the
same parameter gradients (up to float32 rounding).

    python benchmarks/hf_training_memory.py --model Qwen/Qwen2.5-0.5B --check
"""

from __future__ import annotations

import argparse
import time

import torch

import beamgrad
from beamgrad.hf import CausalLMRescorer, CausalLMStep

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
MODES = ("steps", "rescore", "rescore+checkpointing")


def training_step(model, batch, options, steps: int, mode: str) -> torch.Tensor:
    B = batch["input_ids"].shape[0]
    step = CausalLMStep(model, batch["input_ids"], batch["attention_mask"], options.beam_size)
    rescore = None
    if mode != "steps":
        rescore = CausalLMRescorer(
            model, batch["input_ids"], batch["attention_mask"], gradient_checkpointing=mode.endswith("checkpointing")
        )
    result = beamgrad.beam_search(step, options, max_steps=steps, batch_size=B, rescore_fn=rescore)
    # A loss on every beam, so every beam's path is differentiated.
    weights = torch.linspace(1.0, -1.0, options.beam_size, device=result.scores.device)
    (result.scores * weights).sum().backward()
    return result.sequences


def measure(model, batch, options, steps: int, mode: str) -> tuple[float, float] | None:
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        training_step(model, batch, options, steps, mode)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return None
    elapsed = time.perf_counter() - start
    peak = (torch.cuda.max_memory_allocated() - before) / 2**30
    model.zero_grad(set_to_none=True)
    return peak, elapsed


def check_gradients(model_name: str, tokenizer) -> None:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda().eval()
    batch = tokenizer(PROMPTS[:2], return_tensors="pt", padding=True).to("cuda")
    options = beamgrad.BeamOptions(beam_size=3, eos_token=-1, banned_tokens=(tokenizer.eos_token_id,))
    grads, sequences = {}, {}
    for mode in MODES:
        model.zero_grad(set_to_none=True)
        sequences[mode] = training_step(model, batch, options, 8, mode)
        grads[mode] = torch.cat([p.grad.flatten().cpu() for p in model.parameters() if p.grad is not None])
    print("Gradient agreement with `steps` (float32, B=2, K=3, T=8):")
    for mode in MODES[1:]:
        a, b = grads[mode].double(), grads["steps"].double()  # float32 sums over 10^8 entries drift
        cos = (a @ b / (a.norm() * b.norm())).item()
        rel = ((a - b).norm() / b.norm()).item()
        same = torch.equal(sequences[mode], sequences["steps"])
        print(f"  {mode:22s} same beams: {same}   cosine {cos:.6f}   relative difference {rel:.2e}")
    model.zero_grad(set_to_none=True)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--shapes", default="4x4x32,8x8x32,8x8x64,8x8x128", help="comma-separated BxKxT (batch x beams x steps)"
    )
    parser.add_argument("--check", action="store_true", help="first check gradient agreement in float32")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.check:
        check_gradients(args.model, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).cuda().eval()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(
        f"{args.model} ({args.dtype}, all parameters trainable) on {torch.cuda.get_device_name()} "
        f"({total:.1f} GiB), torch {torch.__version__}, beamgrad {beamgrad.__version__}"
    )
    print(f"{'B x K x T':>12} | " + " | ".join(f"{m:>28}" for m in MODES))
    for shape in args.shapes.split(","):
        B, K, T = (int(v) for v in shape.split("x"))
        prompts = (PROMPTS * ((B + len(PROMPTS) - 1) // len(PROMPTS)))[:B]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        options = beamgrad.BeamOptions(beam_size=K, eos_token=-1, banned_tokens=(tokenizer.eos_token_id,))
        cells = []
        for mode in MODES:
            m = measure(model, batch, options, T, mode)
            cells.append("out of memory" if m is None else f"{m[0]:6.2f} GiB {m[1]:6.2f} s")
        print(f"{shape:>12} | " + " | ".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
