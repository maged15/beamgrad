# The CUDA engine

`cuda/dbs_cuda.cu` implements the same beam search as the CPU decoder,
entirely on the GPU, and the same final-score backward pass. The PyTorch
operators in `python/csrc/cuda_ops.cpp` call it on PyTorch's current stream
with scratch memory from PyTorch's caching allocator.

## Forward

A beam search step must pick the `K` best of `K * V` candidates (up to tens
of millions) with the CPU's exact tie-breaking. Each step runs:

0. **`constraint_kernel`** (only with `no_repeat_ngram_size` or
   `repetition_penalty`): for every live beam, rebuilds bitmaps of the tokens
   its prefix blocks and penalises, with the CPU decoder's rules. Each beam's
   token prefix is kept on the device and extended by the select kernel.
1. **A scan kernel** turns the candidates `(parent, token)` of every example
   into 64-bit keys

   ```
   key = ordered_bits(raw score) << 32 | (0xffffffff - (parent * V + token))
   ```

   and keeps the `K` largest per block. Rows are read with coalesced loads,
   and every element of the rows of live, unfinished beams is checked for
   NaN and `+inf` on the way (per-example flags in
   `DBSCudaDecodeOutputs::invalid_input`).
   - **`scan_small_kernel`** (`K <= 16`): each of the 256 threads scans 4 to
     32 candidates and keeps its own top `K` in registers (most candidates are
     rejected by one comparison). The number per thread is chosen from the
     problem size so that the whole batch spans about 512 blocks: a single
     example still fills a large GPU, and big batches keep 32 per thread.
   - **`scan_tiles_kernel`** (`K > 16`), grid `(B, ceil(K*V / 4096))`: each
     thread holds 16 candidates of the block's tile in registers.

   Both then find the block's `K` largest keys with `block_top_k`, a radix
   select over 4-bit digits: a 16-bin shared histogram of the keys still in
   the running locates the digit of the `K`-th largest key, most significant
   digit first, until the keys above it and its digit bucket are exactly
   `K`. Nothing is sorted, and the keys never leave registers, so the scans
   use a few hundred bytes of shared memory. The winners are written in no
   particular order: keys are unique, so the `K` largest are one set however
   the candidates are split into blocks, and step 3 sorts them.
2. **`reduce_tiles_kernel`** repeats the top-`K` selection over the block
   winners until at most 4096 keys per example remain. For typical shapes
   this needs zero or one launch.
3. **`select_step_kernel`**, one block per example, selects the `K` best
   survivors (`block_top_k`), sorts only those, decodes their keys, merges
   them with the EOS carry-forward candidates of finished beams, and writes
   the step's tokens, parents, lengths, scores and the new beam state.

**Why the key reproduces the CPU order.** The CPU ranks candidates by
`(score, raw, parent, token, length, origin)`. All candidates scanned in one
step extend live, unfinished beams, and every such beam has the same length,
so they share one positive length-penalty factor. The score is then a
monotone function of the raw score, and ordering by `(raw descending,
index ascending)` is exactly the CPU's order. Carry-forward candidates have
other lengths, so they are merged in step 3 with the full comparator.

This makes equal lengths a precondition of the step API. `dbs_cuda_decode_step`
takes the beam state from the caller (`DBSCudaBeamState`), and within each
example every live, unfinished beam in it must have the same length. The
initial state and every state a previous step produced satisfy this, so
`dbs_cuda_decode`, `beamgrad.beam_search` and any loop that feeds each step's
state into the next are unaffected. A hand-built state that violates it, with
`length_penalty_alpha != 0`, can select different beams than the CPU decoder,
and not best first. For example, beams of lengths 1 and 5 with α = 3 come back
in the opposite order to the CPU's. The step does not check this: it would
need a device-side check and a stream synchronization on every step, and the
existing synchronization (validating per-example arrays) only happens when
such arrays are passed.

**Why the numbers match bit for bit.** Every floating-point operation that
feeds a result uses an explicitly rounded intrinsic (`__fadd_rn`,
`__fmul_rn`, ...), so nothing is fused into a multiply-add, and the length
penalty comes from `src/penalty.hpp`, which the CPU decoder shares: it is
evaluated in double precision from basic IEEE operations only (no `pow`),
which round identically on the host and the device. `log(repetition_penalty)`
is computed once on the host, as the CPU does.

Resource use (CUDA 13.2, sm_80 and sm_90, from `ptxas -v`): the register
scan uses 32–64 registers and under 300 bytes of static shared memory, the
tile scan and reduce kernels 48–56 registers and 96 bytes, the select kernel
56–62 registers and 38 KB, the backward 36–39 registers and 16 KB; no kernel
spills.

Decode time through the C APIs on an RTX 4080 SUPER, against the CPU decoder
(`dbs_decode_batch_into`) on all 8 hardware threads of the same machine
(steady state, `eos_token=2`, `length_penalty_alpha=0.6`):

| B × T × K × V | CUDA | CPU |
|---|--:|--:|
| 1 × 16 × 4 × 32k | 0.27 ms | 0.22 ms |
| 8 × 16 × 4 × 32k | 0.43 ms | 0.46 ms |
| 8 × 32 × 8 × 32k | 1.5 ms | 3.2 ms |
| 4 × 16 × 8 × 128k | 1.0 ms | 2.8 ms |
| 16 × 64 × 4 × 50k | 3.2 ms | 13.0 ms |
| 8 × 16 × 64 × 32k | 4.3 ms | 16.2 ms |
| 4 × 8 × 256 × 32k | 4.2 ms | 14.8 ms |

Small problems are bound by kernel launches (two or three per step).

## Backward

`backward_kernel` runs one block per example and walks its steps from the
last to the first. Within a step, each beam is handled by its own thread: it
computes the beam's raw-score gradient and adds it to the beam's log-prob
entry (distinct beams of a step have distinct entries, so no two threads
write the same element). The parents' gradients are then formed by sorting
the beams by `(parent, beam)` and letting each parent sum its children in beam
order, which is the CPU backward's order. The result is deterministic, uses no
atomics, and equals the CPU's bit for bit. Zero-filling the dense
`[B, T, K, V]` gradient dominates the cost.

## Limits

- `beam_size <= DBS_CUDA_MAX_BEAM` (1024).
- `K * V <= 2^31 - 1`, and `ceil(K * V / 4096)` must fit in `gridDim.y`
  (65535): about 268 million candidates per step.
- Inputs are float32; the PyTorch layer converts other dtypes.
- Per-example `steps`/beam arrays are validated on the device, which reads one
  status flag back and therefore synchronizes the stream once per call. The
  PyTorch operator also reads the NaN/`+inf` flags back when
  `validate_inputs` is set. With neither, a decode never waits for the device.

## Streams, errors and debugging

Calls enqueue work on the given stream and return immediately; launch errors
are reported through the status code. For debugging, set
`DBS_CUDA_SYNC_CHECK=1` (or call `dbs_cuda_set_synchronization(1)`) to
synchronize after each call, so device faults surface at the call site.

Without a GPU, `dbs_cuda_available()` returns 0. CPU-only builds link a stub
`libdbs_cuda` that exports the same functions and returns
`DBS_CUDA_STATUS_UNAVAILABLE`, so downstream code links everywhere.

## Testing without a GPU

`cuda/emulation/cuda_emulation.hpp` is a small, deterministic emulation of
the CUDA features the kernels use: blocks run as cooperative fibers, and
`__syncthreads()` switches between them. It reports divergent barriers and
invalid launch configurations. Between two barriers the threads of a block
run one after another in an order the test chooses (forward, reverse, or
shuffled at every barrier), so a data race between barriers changes the
result under some order.

`tests/cuda_emulation_tests.cpp` compiles the unmodified `dbs_cuda.cu`
against it and checks over 200 seeded cases, each under all three orders,
against libdbs, requiring every forward output, NaN/`+inf` flag and gradient
to match bit for bit. The cases cover ties, `-inf`, NaN and `+inf` entries,
EOS, `min_length`, length penalty, banned tokens, n-gram blocking, repetition
penalty, variable batches, both scan kernels and the boundary between them,
multi-level reduction, and `K = 1024`. CI runs this suite, with and without
sanitizers, on every change.

The emulation checks the kernels' logic, not their performance or warp-level
timing. `python/tests/test_cuda.py` (exact CPU/CUDA parity, constraints,
validation, `torch.compile`) and `.github/workflows/gpu.yml` run on real
hardware.

## Performance notes

- Each decode step is two or three kernel launches (plus one with n-gram or
  repetition constraints), so tiny problems are dominated by launch latency;
  batching examples amortises it. The operators can be captured in a CUDA
  graph (for example by `torch.compile(mode="reduce-overhead")`) when neither
  per-example steps nor `validate_inputs` needs a read-back.
- `benchmarks/benchmark.py` compares against a `torch.topk` beam search on
  the same device.
