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
   - **`scan_small_kernel`** (`K <= 16`), grid `(B, ceil(K*V / 8192))`: each
     of the 256 threads scans 32 candidates and keeps its own top `K` in
     registers (most candidates are rejected by one comparison); the block
     then sorts only the `256 * K` thread winners.
   - **`scan_tiles_kernel`** (`K > 16`), grid `(B, ceil(K*V / 4096))`: each
     block sorts its tile of 4096 keys with a shared-memory bitonic network.

   Blocks with no valid candidate (dead or finished parents) skip the sort.
2. **`reduce_tiles_kernel`** repeats the top-`K` reduction over the block
   winners until at most 4096 keys per example remain. For typical shapes
   this needs no launch with the register scan, and zero or one otherwise.
3. **`select_step_kernel`**, one block per example, sorts the survivors,
   decodes their keys, merges them with the EOS carry-forward candidates of
   finished beams, and writes the step's tokens, parents, lengths, scores and
   the new beam state.

**Why the key reproduces the CPU order.** The CPU ranks candidates by
`(score, raw, parent, token, length, origin)`. All candidates scanned in one
step extend live, unfinished beams, and every such beam has the same length,
so they share one positive length-penalty factor. The score is then a
monotone function of the raw score, and ordering by `(raw descending,
index ascending)` is exactly the CPU's order. Carry-forward candidates have
other lengths, so they are merged in step 3 with the full comparator.

**Why the numbers match bit for bit.** Every floating-point operation that
feeds a result uses an explicitly rounded intrinsic (`__fadd_rn`,
`__fmul_rn`, ...), so nothing is fused into a multiply-add, and the length
penalty comes from `src/penalty.hpp`, which the CPU decoder shares: it is
evaluated in double precision from basic IEEE operations only (no `pow`),
which round identically on the host and the device. `log(repetition_penalty)`
is computed once on the host, as the CPU does.

Resource use (CUDA 13, sm_80 and sm_90, from `ptxas -v`): the register scan
uses 31–64 registers and up to 33 KB of static shared memory, the tile scan
and reduce kernels 32–44 registers and 33 KB, the select kernel 48 registers
and 42 KB, the backward 36 registers and 16 KB; no kernel spills.

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
