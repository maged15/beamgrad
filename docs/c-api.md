# C API

beamgrad's native core is **libdbs** ([`include/dbs.h`](../include/dbs.h)), a
C ABI usable from C, C++, Rust, Go, or any FFI. The CUDA backend is
**libdbs_cuda** ([`include/dbs_cuda.h`](../include/dbs_cuda.h)). Both headers
are plain C99.

## Using the library

With CMake:

```cmake
find_package(beamgrad 2 REQUIRED)
target_link_libraries(my_app PRIVATE beamgrad::dbs)        # CPU
target_link_libraries(my_app PRIVATE beamgrad::dbs_cuda)   # CUDA (a stub in CPU-only builds)
```

With pkg-config: `pkg-config --cflags --libs dbs`. See
[development.md](development.md#building) for build options.

A complete, compiling example is [`examples/c_api.c`](../examples/c_api.c).

## Conventions

- **Return codes.** Functions returning a status return `DBS_OK` (`0`) on
  success, `DBS_ERROR_INVALID_ARGUMENT` (`-1`) for invalid arguments or inputs
  (null pointers, bad shapes or options, sizes that overflow, NaN/`+inf`
  log-probs when `validate_inputs` is set), `DBS_ERROR_RUNTIME` (`-2`) for a
  failure during the computation (out of memory, a callback that reported an
  error) and `DBS_ERROR_UNKNOWN` (`-3`).
- **Errors.** Every failing call records a message, including early failures
  such as a null handle. `dbs_last_global_error()` returns the calling
  thread's latest message and `dbs_last_error(handle)` a decoder's latest one;
  a successful call clears both. When threads share a decoder, a call on one
  thread can replace the decoder's message before another thread reads it, so
  prefer the thread-local `dbs_last_global_error()` there. Returned pointers
  are valid until the next call on the same thread.
- **Ownership.** Every handle returned through an out-parameter belongs to the
  caller and is released with its matching function (`dbs_destroy`,
  `dbs_free_result`, `dbs_free_backward`, `dbs_free_batch_result`,
  `dbs_workspace_destroy`). Pointers returned by accessors are borrowed from
  their handle.
- **Layout.** Tensors are dense, row-major float32 unless stated otherwise.
  `log_probs` is `[T, K, V]`, element `(t, k, v)` at `(t*K + k)*V + v`.
  Batches are `[B, T, K, V]`. Sparse gradient indices use the flattened
  `[T, K, V]` layout.
- **Threads.** A decoder's options are immutable after creation; one decoder
  may be used from several threads at once. Stats are per decoder (last writer
  wins); error messages are per thread and per decoder (see above).

## Options

`DBSOptionsC` configures a decoder. A zero-initialised field selects its
default, so `DBSOptionsC opt = {0}` is valid; explicit negative or non-finite
values are rejected by `dbs_create_ex`. **Note:** `eos_token = 0` means
token 0, so set it to `-1` to disable EOS handling.

| field | default | meaning |
|---|---|---|
| `beam_size` | 8 | beams `K` |
| `eos_token` | (none) | EOS token id, or `-1` |
| `min_length` | 0 | EOS is masked until a hypothesis has this many tokens |
| `length_penalty_alpha` | 0 | GNMT length penalty exponent |
| `selected_temperature` | 1.0 | softmax temperature of the selected-beam weights |
| `soft_topk_temperature` | 0.25 | sigmoid temperature of the relaxed pool |
| `relaxed_pool_multiplier` | 0 | relaxed pool size `P = K * multiplier`; 0 disables the pool |
| `soft_topk_tolerance` | 1e-4 | the relaxed pool's bisection stops when the weights sum to `K` within this, or `theta` is bracketed to this × `soft_topk_temperature` (at any score magnitude) |
| `soft_topk_max_iters` | 48 | bisection iteration cap; the defaults need about 21 iterations for a pool a few units wide |
| `vocab_block` | (ignored) | kept for source compatibility |
| `validate_inputs` | (none) | non-zero: fail with `-1` when a row the search reads contains NaN or `+inf` |

`validate_inputs` checks every element of every row the search reads (the rows
of live, unfinished beams, including tokens that are banned or masked) as part
of the scan, so it costs no extra pass. Rows that are never read, such as
beams 1..K-1 at step 0 or the rows of finished beams, are not checked. Without
it, NaN and `+inf` entries are never selected.

The relaxed pool (the `pool_*` and `relaxed_weights` outputs and
`grad_relaxed_weights`) is only kept when `relaxed_pool_multiplier` is at
least 1; it never changes which beams are selected, and keeping `P` candidates
per step costs time, so it is off by default.
| `max_dense_gradient_elements` | 1e8 | cap on `T*K*V` for `dbs_backward_dense` |

## Decoding

| function | input |
|---|---|
| `dbs_decode` | float32 `[T, K, V]` |
| `dbs_decode_typed` | float32, IEEE fp16 or bf16 `[T, K, V]` (`DBSDataTypeC`) |
| `dbs_decode_constrained` | banned tokens `[V]`, forced tokens `[T]` (`-1` = free), per-call min length |
| `dbs_decode_constrained_ex` | `DBSAdvancedConstraintsC`: the above plus repetition penalty, no-repeat n-gram size, and a token-filter callback |
| `dbs_decode_batch` / `dbs_decode_batch_typed` | `[B, T, K, V]`, decoded on `num_threads` threads (0 = all cores) |
| `dbs_decode_batch_variable` | `[B, maxT, maxK, V]` with per-example steps, beam sizes, EOS tokens, min lengths, banned masks and forced schedules |
| `dbs_decode_model_steps_ex` | a callback produces each step's `[K, V]` rows from the current beams (see below); optional constraints |
| `dbs_decode_model_steps` / `..._with_workspace` | the original callback, which sees only each beam's previous token and score |
| `dbs_decode_batch_into` | `[B, T, K, V]` with optional per-example steps and constraints, straight into caller-owned arrays (`DBSDecodeOutputsC`), no result handles |

Constraints (`DBSAdvancedConstraintsC`) are applied per beam: `banned_tokens`
removes tokens everywhere, `no_repeat_ngram_size = n` blocks every token that
would complete an n-gram already present in the beam's own prefix, and
`repetition_penalty > 1` subtracts `log(repetition_penalty)` from tokens the
beam has already emitted. The blocked and penalised sets are computed once per
beam and step, so constrained decoding runs at nearly the speed of
unconstrained decoding (a `token_filter` callback is still called per token).

### Model-step decoding

When each step's rows depend on the beams chosen so far (an autoregressive
model), `dbs_decode_model_steps_ex` runs the search one step at a time and
calls

```c
int step_fn(void* user_data, const DBSModelStepInfoC* info, float* out_log_probs /* [K, V] */);
```

once per step. `info` describes the beams entering the step: for each slot
`k`, the slot it came from at the previous step (`parents[k]`), the token it
emitted (`tokens[k]`), its length, scores and finished flag, and its whole
token prefix (`prefixes[k * step + s]`). Beams are re-ranked every step, so
slot `k` usually continues a different hypothesis than slot `k` did one step
earlier: a model that keeps per-beam state (a KV cache) reorders it by
`parents`, exactly like `index_select` on the beam dimension. Entries the
callback leaves untouched are `-inf`; returning non-zero aborts the decode with
`DBS_ERROR_RUNTIME`. The search runs incrementally, `T` callbacks for `T`
steps.

A result (`DBSResultHandle`) exposes, per step `[T, K]`: `tokens`, `parents`,
`lengths`, `scores`, `raw_scores`, `weights` (selected-beam softmax); per final
beam `[K]`: `final_scores`, `final_raw_scores`; and the relaxed pool
`[T, P]`: `pool_tokens`, `pool_parents`, `pool_lengths`, `pool_scores`,
`pool_raw_scores`, `relaxed_weights`. Sizes come from `dbs_result_steps`,
`dbs_result_beam_size`, `dbs_result_vocab_size` and `dbs_result_pool_size`.

## Backward

```c
int dbs_backward(DBSDecoderHandle*, const DBSResultHandle*,
                 const float* grad_selected_weights,  /* [T*K] or NULL */
                 const float* grad_relaxed_weights,   /* [T*P] or NULL */
                 const float* grad_final_scores,      /* [K]   or NULL */
                 DBSBackwardHandle** out);
```

`dbs_backward` (equivalently `dbs_backward_sparse` / `dbs_backward_default`)
returns a sparse gradient: `dbs_backward_sparse_logprob_indices` /
`_values` / `_count`, plus `dbs_backward_grad_initial_scores` `[K]`.
`dbs_backward_dense` materialises the full `[T, K, V]` gradient and fails if
`T*K*V` exceeds `max_dense_gradient_elements`. The gradients are defined in
[algorithm.md](algorithm.md#backward-surrogate-gradients).

The backward reads the beam size, vocabulary, temperatures and length penalty
from the result, so a result decoded with other options (for example an
example of `dbs_decode_batch_variable` with its own beam size) can go through
any decoder handle.

`dbs_backward_batch_into` is the array counterpart of `dbs_decode_batch_into`:
it takes that call's `parents`, `tokens`, `lengths` and `from_logprob` outputs
and `grad_final_scores [B, K]`, checks that the trace is in range, and
accumulates the final-score gradient into `grad_log_probs [B, T, K, V]`.

## Introspection

- `dbs_abi_version()`, `dbs_version_string()`
- `dbs_selected_kernel_name()`: `"avx512"`, `"avx2"`, `"sse4.2"`, `"neon"` or
  `"scalar"`, chosen at runtime from the CPU; `dbs_has_avx512()` and friends
- `dbs_get_stats` / `dbs_get_stats_json` / `dbs_reset_stats`: timings, sizes,
  kernel, and error category of the last call on a decoder
- `dbs_result_summary_json`, `dbs_result_eos_count`,
  `dbs_result_validate_deterministic_order` (checks each step's beams against
  the decoder's full candidate order)
- `dbs_allocator_call_count` (allocations since the last
  `dbs_allocator_counters_reset`) and `dbs_allocator_byte_count` (bytes
  currently allocated; a gauge the reset leaves alone)

`dbs_is_deterministic()` always returns 1: decoding uses no randomness and a
total order over candidates. `dbs_set_deterministic_seed` /
`dbs_get_deterministic_seed` only store a value with the decoder (nothing
reads it); they remain for ABI compatibility.

## CUDA

```c
DBSCudaDecodeArgs args = {0};
args.batch_size = B; args.steps = T; args.beam_size = K; args.vocab_size = V;
args.eos_token = -1;

DBSCudaDecodeOutputs out = {0};
out.final_scores = d_scores;            /* [B, K], required */
out.tokens = d_tokens;                  /* [B, T, K], optional */
out.parents = d_parents;                /* ...and lengths, scores, raw_scores, from_logprob */

int status = dbs_cuda_decode(d_log_probs, &args, &out, NULL, 0, stream);
```

- `dbs_cuda_decode` runs the full search on the device; optional per-example
  `steps_per_example`, `beam_sizes_per_example`, `eos_tokens_per_example` and
  `min_lengths_per_example` arrays give variable-length batches, and
  `banned_tokens`, `no_repeat_ngram_size` and `repetition_penalty` apply the
  CPU decoder's constraints. `outputs->invalid_input` (optional, `[B]`)
  reports the examples whose rows contained NaN or `+inf`.
- `dbs_cuda_decode_step` runs one step from a caller-held state
  (`DBSCudaBeamState`: `[B, K]` raw scores, lengths and finished flags,
  updated in place, plus each beam's token prefix), for loops that ask a model
  for each step's `[B, K, V]` rows. `args->steps` must be 1; the step's
  `[B, K]` outputs go to the usual `DBSCudaDecodeOutputs` fields. Stepping
  through the rows of a tensor selects exactly what `dbs_cuda_decode` selects.
  Its workspace size is `dbs_cuda_decode_step_workspace_size(&args,
  prefix_stride)`. `beamgrad.beam_search` is built on it.
- `dbs_cuda_backward` takes the `parents`, `tokens`, `lengths` and
  `from_logprob` outputs plus `grad_final_scores [B, K]` and accumulates the
  final-score gradient into `grad_log_probs [B, T, K, V]`.
- Scratch memory: pass `NULL` to have it allocated with `cudaMallocAsync` on
  the stream, or provide `dbs_cuda_decode_workspace_size(&args)` bytes
  yourself (`dbs_cuda_backward_workspace_size` is 0: the backward needs none).
- Status codes: `DBS_CUDA_STATUS_OK`, `_UNAVAILABLE` (CPU-only build or no
  device), `_INVALID_ARGUMENT`, `_LAUNCH_FAILED`, `_OUT_OF_MEMORY`;
  `dbs_cuda_status_string` describes them.
- Beam sizes up to `DBS_CUDA_MAX_BEAM` (1024).

See [cuda.md](cuda.md) for the engine's design, streams and debugging.

## Stability and versioning

- `DBS_ABI_VERSION` (currently **10**) is the shared library's `SOVERSION`. It
  changes only on binary-incompatible changes: a removed or changed exported
  function, or a changed struct layout.
- The exact set of exported `dbs_*` symbols is recorded in
  [`abi/libdbs.symbols`](../abi/libdbs.symbols) and checked by ctest on
  Linux, together with a baseline ([`abi/baseline-v0.5.symbols`](../abi/baseline-v0.5.symbols))
  whose symbols may never be removed.
- `DBS_VERSION_MAJOR/MINOR/PATCH` is the release version, shared with the
  Python package.
- `dbs_cuda.h` is versioned with the release, not the ABI number; see the
  [changelog](../CHANGELOG.md) for its changes.
