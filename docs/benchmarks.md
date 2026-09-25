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
line. `benchmark.py` times `beamgrad.final_scores` forward, forward +
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

- whether the two return the same beams when EOS is suppressed. They do,
  with bit-identical scores, in float32. In bfloat16, logits often tie
  exactly and the two break ties differently; the script counts those
  prompts.
- how the two compare with natural EOS handling. beamgrad keeps finished
  hypotheses competing in their slots; `transformers` moves them to a
  separate pool.
- median wall time for both. The model's forward passes dominate, so the two
  are close.

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
