# The CUDA engine

`cuda/dbs_cuda.cu` implements the same beam search as the CPU decoder,
entirely on the GPU, and the same final-score backward pass. The PyTorch
operators in `python/csrc/cuda_ops.cpp` call it on PyTorch's current stream
with scratch memory from PyTorch's caching allocator, so there are no host
round trips, device-to-host copies or implicit synchronizations.

## Forward

A beam search step must pick the `K` best of `K * V` candidates (up to tens
of millions) with the CPU's exact tie-breaking. Each step runs three kernels:

1. **`scan_tiles_kernel`**, grid `(B, ceil(K*V / 4096))`, 256 threads. Each
   block owns 4096 consecutive candidates `(parent, token)` of one example.
   It reads its slice of `log_probs` with coalesced loads, builds a 64-bit
   key per valid candidate:

   ```
   key = ordered_bits(raw score) << 32 | (0xffffffff - (parent * V + token))
   ```

   sorts the tile with a shared-memory bitonic network, and writes its top
   `K` keys. Blocks whose tile has no valid candidate (dead or finished
   parents) skip the sort.
2. **`reduce_tiles_kernel`** repeats the same top-`K` reduction over the tile
   winners until at most 4096 keys per example remain. For typical shapes
   this is zero or one extra launch.
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
other lengths, so they are merged in step 3 with the full comparator. The
emulated parity suite checks this bit for bit (see below).

Resource use (sm_80, CUDA 13): the tile kernels use 32.8 KB of static shared
memory and 32–34 registers per thread; the select kernel uses 41.0 KB and 42
registers; no kernel spills. Everything is deterministic: no atomics feed any output.

## Backward

`backward_kernel` assigns one thread per example and walks the example's
selected paths from the last step to the first, exactly like the CPU sparse
backward. Accumulation order is fixed, so gradients are deterministic and match
the CPU's bit for bit. The work is `O(B * T * K)` and negligible next to the
forward pass; zero-filling the dense `[B, T, K, V]` gradient dominates.

## Limits

- `beam_size <= DBS_CUDA_MAX_BEAM` (1024).
- `K * V <= 2^31 - 1`, and the tile count `ceil(K * V / 4096)` must fit in
  `gridDim.y` (65535): about 268 million candidates per step.
- Inputs are float32; the PyTorch layer converts other dtypes.
- Per-example `steps`/beam arrays are validated on the device, which reads one
  status flag back and therefore synchronizes the stream once per call.

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
invalid launch configurations. `tests/cuda_emulation_tests.cpp` compiles the
unmodified `dbs_cuda.cu` against it and checks 164 seeded, randomized cases
against libdbs, requiring every forward output and every gradient to match
bit for bit. The cases cover ties, `-inf` rows, EOS, `min_length`, length
penalty, variable batches, multi-level tile reduction, and `K = 1024`. CI runs
this suite, with and without sanitizers, on every change. It checks the
kernels' logic, not their performance or warp-level timing, which is why
`python/tests/test_cuda.py` and `.github/workflows/gpu.yml` exist for real
hardware.

## Performance notes

- The tile sort is a full bitonic sort of 4096 keys (78 compare-exchange
  stages). A cheaper threshold filter before the sort is the natural next
  optimisation for very large vocabularies.
- Each decode step is three or four kernel launches, so tiny problems are
  dominated by launch latency; batching examples amortises it. Capturing the
  step loop in a CUDA graph would remove most of that overhead.
- `benchmarks/benchmark.py` compares against a `torch.topk` beam search on
  the same device.
