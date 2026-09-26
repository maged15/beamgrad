# Benchmarks

The benchmarks measure different things. Kernel benchmarks time the search
engine on precomputed rows. End-to-end benchmarks include the model, which
usually dominates. Pick the level that matches the question.

| question | script | level |
|---|---|---|
| How fast is the search itself, forward and backward? | `build/dbs_bench` (C ABI), `benchmarks/benchmark.py` (PyTorch, CPU and CUDA) | kernel |
| Does `beam_search` driving a real model match `generate()`, and at what speed? | `benchmarks/hf_beam_search.py` | end to end, inference |
| What does a training step through the search cost in memory and time, per gradient mode? | `benchmarks/hf_training_memory.py` | end to end, training |
| Does training through the search improve a model? | `experiments/multi30k/run.py` | training quality |

## Kernel

`dbs_bench` times `dbs_decode` and the dense and sparse backward passes over
a grid of `T × K × V × B` shapes, or over one shape given on the command
line. `dbs_bench model-steps` times `dbs_decode_model_steps_ex` over long
searches, with a callback that only copies rows, so the search's own
per-step work is measured. `dbs_bench batch-threads` compares
`dbs_decode_batch_into` with automatic threading and with one thread. That
comparison sets the size below which automatic threading uses one thread.
`benchmark.py` times `beamgrad.final_scores` forward, forward +
backward, and a reference beam search written with `torch.topk`, on every
available device. It also asserts that the scores agree. `--csv` records the
results together with the environment.

These numbers bound the search's overhead: the time spent selecting beams
and building gradients, given rows that already exist. On CPU the backward is
dominated by writing the dense `[B, T, K, V]` gradient that
`final_scores`' autograd contract requires. `beam_search` avoids that, since
it hands each step a `[B, K]` path gradient. So `benchmark.py`'s backward
times are an upper bound for training with `beam_search`.

## End to end: inference

`hf_beam_search.py` runs an open-weights causal LM (default
`Qwen/Qwen2.5-0.5B`) through `beamgrad.beam_search` with
`beamgrad.hf.CausalLMStep`, and through `generate(num_beams=K)`. It reports:

- whether the two return the same beams when EOS is suppressed. They do. In
  float32 the scores are bit-identical when beamgrad runs the prompt once per
  beam as `generate()` does (`share_prompt=False`), and within 1e-4 when it
  runs each prompt once (the default). In bfloat16, logits often tie exactly
  and the two break ties differently; the script counts those prompts.
- how the two compare with natural EOS handling. beamgrad keeps finished
  hypotheses competing in their slots; `transformers` moves them to a
  separate pool.
- median wall time for both. The model's forward passes dominate, so the two
  are close on short prompts.
- with `--long-prompt N`, time and peak memory on prompts of `N` tokens. On
  an RTX 4080 SUPER with Qwen2.5-0.5B-Instruct in float32 (batch 8, 4 beams,
  8 new tokens, 2,048-token prompts): `generate()` 2,759 ms and 7.82 GiB,
  beamgrad with `share_prompt=False` 2,754 ms and 7.82 GiB, beamgrad by
  default 916 ms and 3.41 GiB.

Every EOS token of the model's generation config counts on both sides, and
the script turns off the repetition penalty that instruct models'
generation configs set, which `generate()` would otherwise apply.

## End to end: training

`hf_training_memory.py` measures one training step (the search plus the
backward of a loss on every final beam, with all parameters trainable) in
three gradient modes: through the steps, re-scoring, and re-scoring with
activation checkpointing. It reports peak memory above the weights, and
wall time. `--check` first verifies in float32 that the three modes give
the same parameter gradients (cosine similarity and relative difference).
Results for Qwen2.5-0.5B are in
[training.md](training.md#memory-three-ways-to-take-the-gradient).

## Training quality

`experiments/multi30k/run.py` pretrains a small transformer with MLE, then
fine-tunes it with each objective over several seeds. It reports test BLEU,
the paired difference from continued MLE, time per step and peak memory. The
protocol and results are in
[experiments/multi30k/README.md](../experiments/multi30k/README.md).

## Reading the numbers

- Run on your own hardware before relying on any figure; every script prints
  or records its environment.
- GPU timings synchronize before reading the clock. The first call is a
  warm-up and is not timed.
- Wall times of Python-driven loops such as `beam_search` are sensitive to
  CPU contention. Running a CPU-heavy job at the same time, even with PyTorch
  using every core, inflates them.
