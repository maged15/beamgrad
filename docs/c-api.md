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

- **Return codes.** Functions returning `int` return `0` on success, `-1` for
  invalid arguments, `-2` for an error raised during the computation, `-3` for
  an unexpected internal error.
- **Errors.** `dbs_last_error(handle)` returns the last error message for a
  decoder; `dbs_last_global_error()` returns the calling thread's last error
  (use it when `dbs_create_ex` fails). The returned pointer is valid until the
  next call on the same thread.
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
  may be used from several threads at once. Stats and error messages are
  per decoder.

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
| `relaxed_pool_multiplier` | 8 | relaxed pool size `P = K * multiplier` |
| `soft_topk_tolerance` | 1e-4 | bisection tolerance for the relaxed pool |
| `soft_topk_max_iters` | 48 | bisection iteration cap |
| `vocab_block` | 4096 | vocabulary scan block (tuning only; results do not depend on it) |
| `validate_inputs` | (none) | non-zero: reject NaN/`+inf` log-probs (checks up to 1000 sampled entries) |
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
| `dbs_decode_model_steps` / `..._with_workspace` | a callback produces each step's `[K, V]` rows from the current beams |

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

## Introspection

- `dbs_abi_version()`, `dbs_version_string()`
- `dbs_selected_kernel_name()`: `"avx512"`, `"avx2"`, `"sse4.2"`, `"neon"` or
  `"scalar"`, chosen at runtime from the CPU; `dbs_has_avx512()` and friends
- `dbs_get_stats` / `dbs_get_stats_json` / `dbs_reset_stats`: timings, sizes,
  kernel, and error category of the last call on a decoder
- `dbs_result_summary_json`, `dbs_result_eos_count`,
  `dbs_result_validate_deterministic_order`
- `dbs_allocator_*`: process-wide allocation counters

`dbs_set_deterministic_seed` / `dbs_get_deterministic_seed` store a seed with
the decoder for reproducibility metadata; decoding itself uses no randomness.
`dbs_validate_production_gate_manifest` is a deprecated no-op kept for ABI
compatibility; it always returns non-zero.

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
  `min_lengths_per_example` arrays give variable-length batches.
- `dbs_cuda_backward` takes the `parents`, `tokens`, `lengths` and
  `from_logprob` outputs plus `grad_final_scores [B, K]` and accumulates the
  final-score gradient into `grad_log_probs [B, T, K, V]`.
- Scratch memory: pass `NULL` to have it allocated with `cudaMallocAsync` on
  the stream, or provide `dbs_cuda_decode_workspace_size(&args)` /
  `dbs_cuda_backward_workspace_size(&args)` bytes yourself.
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
