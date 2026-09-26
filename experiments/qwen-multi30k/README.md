# Qwen2.5-0.5B on Multi30k: minimum-risk training through beam search

[`experiments/multi30k`](../multi30k) found that minimum-risk training (MRT)
on beamgrad's beams helps a small translation model trained from scratch.
This experiment asks the same question of a pretrained LLM, fully
fine-tuned. It also compares beamgrad with the do-it-yourself route:
`generate()`, then teacher-forced re-scoring, then the same loss by hand.

## Protocol

**Model.** `Qwen/Qwen2.5-0.5B-Instruct`, all 494M parameters trained, with
float32 weights and bfloat16 autocast. The prompt is the chat template, with
the system message "You are a translator. Translate the user's English
sentence into German. Reply with the German translation only." and the
English sentence as the user turn.

**Data.** Multi30k En→De (Elliott et al., 2016), the same files as
`experiments/multi30k`: 29,000 training pairs, and test2016 (1,000 pairs)
for evaluation.

**Search.** Beam 4, length penalty α = 0.6, `min_length=2`. The EOS token is
`<|im_end|>`, and `<|endoftext|>` is banned: beamgrad 2.2.0, which ran this,
took one EOS token (2.2.1 takes both). Evaluation decodes up to 96 tokens and
scores the best beam with sacreBLEU (default `13a` tokenization).

**Supervised fine-tuning (SFT).** 1,800 steps of batch 16 (about one epoch),
teacher-forced NLL, AdamW at 1e-5, 30 warm-up steps, gradient clipping 1.0.

**Arms.** Each starts from the SFT checkpoint and runs 500 steps of batch 8
at 5e-6, on the same batches for a given seed:

| arm | loss | gradient |
|---|---|---|
| continued SFT (the control) | token NLL | teacher forcing |
| beamgrad MRT | `losses.minimum_risk` over the 4 beams of `beamgrad.beam_search` (cost 1 − sentence BLEU) + 0.3 × token NLL | `rescore_fn=CausalLMRescorer(..., gradient_checkpointing=True)`; re-scoring is exact here, since Qwen has no dropout |
| DIY MRT ([`diy_mrt.py`](diy_mrt.py)) | the same loss, written by hand, over the 4 beams of `generate(num_beams=4)` | a teacher-forced pass over prompt + beams, with the model's gradient checkpointing |

**Seeds.** Seeds 0–2 started from one SFT checkpoint (test BLEU 38.56),
seeds 3–13 from a second one (38.94). The second was rebuilt from the same
steps, but GPU nondeterminism made it differ.

**Environment.** RTX 4080 SUPER (16 GB), PyTorch 2.14.0 (CUDA 13.0),
transformers 5.17.0, beamgrad 2.2.0 from the wheel index.

## Results

The zero-shot model scores 16.41 test BLEU; after SFT it scores 38.56 and
38.94 (the two checkpoints).

**MRT against continued SFT, 8 seeds** ([`results/significance.txt`](results/significance.txt)):

| seed | continued SFT | beamgrad MRT | Δ | 95% CI (bootstrap over sentences) |
|---|---|---|---|---|
| 0 | 40.43 | 41.09 | +0.66 | [−0.24, +1.51] |
| 1 | 40.00 | 39.68 | −0.32 | [−1.19, +0.66] |
| 2 | 40.18 | 40.47 | +0.29 | [−0.51, +1.29] |
| 3 | 39.66 | 41.34 | +1.68 | [+0.83, +2.57] |
| 4 | 40.96 | 41.43 | +0.47 | [−0.47, +1.33] |
| 5 | 39.53 | 41.10 | +1.58 | [+0.57, +2.61] |
| 6 | 38.54 | 39.74 | +1.21 | [+0.18, +2.25] |
| 7 | 39.35 | 39.58 | +0.23 | [−0.84, +1.26] |

MRT beat continued SFT on 7 of 8 seeds, by **+0.72 ± 0.70 BLEU**. The paired
t-test gives p = 0.023, and the exact sign-flip test p = 0.031. Pooled over
the seeds, the bootstrap 95% interval is [+0.36, +1.09].

**beamgrad MRT against DIY MRT** ([`results/compare_diy.txt`](results/compare_diy.txt),
[`results/final_compare.txt`](results/final_compare.txt)):

| evaluated with | seeds | DIY | beamgrad | Δ | beamgrad ahead | t-test p | sign-flip p |
|---|---|---|---|---|---|---|---|
| beamgrad's search | 3–13 | 40.09 | 40.42 | +0.34 ± 0.55 | 7/11 | 0.069 | 0.076 |
| `generate()` | 8–13 | 40.12 | 40.18 | +0.06 ± 0.42 | 2/6 | 0.72 | 0.88 |

The first five seeds alone gave +0.63 (4 of 5); the next six gave +0.09.
There is no reliable BLEU difference between the two ways of running MRT.

**Cost of a training step** (batch 8, 4 beams; from the training logs):

| | time per step | peak memory |
|---|---|---|
| continued SFT | 0.14–0.18 s | 9.2 GiB |
| beamgrad MRT | 0.74–0.87 s | **9.2 GiB** on all 14 runs |
| DIY MRT | 0.72–0.78 s | 11.4–12.4 GiB over its 11 runs |
| beamgrad MRT without `gradient_checkpointing` | | out of memory (15.6 GiB card) |

**Gradients** ([`results/gradcheck.txt`](results/gradcheck.txt)). On the same
beams in float32, beamgrad's `minimum_risk` and the hand-written loss give
the same loss to six digits. Their parameter gradients have cosine
1.00000000 and relative difference at most 1.9e-5 (3 batches). On 80
training prompts in bfloat16, `generate()` returned the same 4 beams as
beamgrad 80% of the time, and the same best beam 97.5% of the time. The two
searches treat finished hypotheses differently: beamgrad keeps them in their
beam slots, and `generate()` moves them to a separate pool. That is the
likely cause; bfloat16 ties, which the two break differently, can also
contribute.

## What this shows

- **MRT through beam search improves an LLM's translations.** It gained
  +0.72 BLEU over continued SFT, on 7 of 8 seeds (p = 0.023). This agrees
  with the small model of `experiments/multi30k` (+0.39, 7 of 8, p = 0.009).
- **For quality, beamgrad and the do-it-yourself pipeline are the same.**
  They compute the same gradient on the same beams, and their BLEU does not
  differ reliably over 11 seeds. Evaluating with beamgrad's own search may
  favour the model trained on it: with `generate()` as the decoder the
  difference is +0.06.
- **beamgrad's advantage is memory, and one call instead of a pipeline.**
  It peaked at 9.2 GiB, no more than plain SFT; the DIY route peaked at
  11.4–12.4 GiB, at about the same speed. The DIY loss takes a float32
  `log_softmax` over the whole vocabulary at every position of the B·K
  sequences, where `CausalLMRescorer` projects to the vocabulary in
  checkpointed chunks; that is the most likely source of the difference.
- **Full fine-tuning of a 0.5B model on 16 GB needs
  `CausalLMRescorer(gradient_checkpointing=True)`.** Without it, MRT ran out
  of memory within its first steps.

The limits are the same as for the small model's experiment: one model, one
task, 500 steps, and two SFT checkpoints across the seeds.

## Reproduce

From this directory, with `transformers`, `sacrebleu` and `scipy`
installed. The data downloads on first use. Checkpoints and per-sentence
outputs go to `runs/`, which git ignores; the SFT checkpoint takes 2 GB.

```bash
python train_mt.py --seeds 0 1 2 --sft-steps 1800 --ft-steps 500 --out runs/results.json
python train_mt.py --seeds 3 4 5 6 7 --sft-steps 1800 --ft-steps 500 --sft-ckpt runs/sft.pt --out runs/results_seeds3-7.json
python diy_mrt.py --seeds 3 4 5 6 7            # DIY MRT from runs/sft.pt
python run_pair.py --seeds 8 9 10 11 12 13     # both methods, both decoders
python significance.py runs/results.json runs/results_seeds3-7.json
python compare_diy.py
python final_compare.py
python gradcheck.py
```

With beamgrad 2.2.1 or later, `CausalLMStep` runs each prompt once per
example by default. Its scores can then differ in the last bits from the
2.2.0 runs recorded here; the beams do not. `share_prompt=False` restores
the 2.2.0 behaviour. [`results/results.json`](results/results.json) has
every per-seed number above.
