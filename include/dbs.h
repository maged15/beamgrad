// SPDX-License-Identifier: MIT
//
// libdbs: the C ABI of beamgrad (differentiable beam search).
//
// Conventions
//   * Functions returning an int status return DBS_OK (0) on success and a
//     negative code on failure: DBS_ERROR_INVALID_ARGUMENT (-1) for invalid
//     arguments or inputs (null pointers, bad shapes or options, NaN/+inf
//     log-probs when validate_inputs is set, sizes that overflow),
//     DBS_ERROR_RUNTIME (-2) for failures during the computation (out of memory,
//     a callback that reported an error), and DBS_ERROR_UNKNOWN (-3).
//     Every failing call records a message: dbs_last_global_error() returns the
//     calling thread's latest one, and dbs_last_error(handle) the latest one on a
//     decoder handle. When threads share a handle, a call on one thread can
//     replace the handle's message before another reads it; the thread-local
//     dbs_last_global_error() is the reliable source there.
//   * Every handle returned through an out-parameter is owned by the caller and
//     must be released with its matching dbs_free_* / dbs_destroy function.
//     Pointers returned by dbs_result_* / dbs_backward_* accessors are borrowed
//     and stay valid until the owning handle is freed.
//   * Tensors are dense, row-major float32 unless stated otherwise:
//     log_probs is [T, K, V] indexed as (t*K + k)*V + v; batched inputs are
//     [B, T, K, V]. Sparse gradient indices use the same flattened [T, K, V] layout.
//   * A decoder handle may be shared between threads for decoding and backward;
//     its options are immutable after creation.
//
// DBS_ABI_VERSION changes only on binary-incompatible changes and is the shared
// library SOVERSION. DBS_VERSION_* is the release version.
#pragma once

#include <stdint.h>

#if defined(_WIN32) && defined(DBS_STATIC)
#define DBS_EXPORT
#elif defined(_WIN32) && defined(DBS_BUILD_SHARED) && defined(DBS_COMPILING_LIBRARY)
#define DBS_EXPORT __declspec(dllexport)
#elif defined(_WIN32) && defined(DBS_BUILD_SHARED)
#define DBS_EXPORT __declspec(dllimport)
#elif defined(_WIN32)
#define DBS_EXPORT
#elif defined(__GNUC__) || defined(__clang__)
#define DBS_EXPORT __attribute__((visibility("default")))
#else
#define DBS_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DBS_ABI_VERSION 10
#define DBS_VERSION_MAJOR 2
#define DBS_VERSION_MINOR 0
#define DBS_VERSION_PATCH 0

#define DBS_OK 0
#define DBS_ERROR_INVALID_ARGUMENT (-1)
#define DBS_ERROR_RUNTIME (-2)
#define DBS_ERROR_UNKNOWN (-3)

/* Decoder options. A zero-initialised field selects its default (shown in
 * brackets); negative or non-finite values are rejected by dbs_create_ex().
 * Note that eos_token and validate_inputs have no "zero = default" rule. */
typedef struct DBSOptionsC {
    int beam_size;                      /* beams K [8] */
    int eos_token;                      /* -1 disables EOS handling; 0 is token 0 */
    float selected_temperature;         /* softmax temperature for selected-beam weights [1.0] */
    float soft_topk_temperature;        /* sigmoid temperature of the relaxed pool [0.25] */
    int relaxed_pool_multiplier;        /* relaxed pool size P = K * multiplier; 0 disables the pool [0] */
    int vocab_block;                    /* ignored; kept for source compatibility */
    float length_penalty_alpha;         /* GNMT length penalty ((5 + len) / 6)^alpha; 0 disables [0] */
    float soft_topk_tolerance;          /* bisection tolerance for the relaxed pool [1e-4] */
    int soft_topk_max_iters;            /* bisection iteration cap [48] */
    int min_length;                     /* EOS is masked until a hypothesis reaches this length [0] */
    int validate_inputs;                /* non-zero: fail when a row the search reads contains NaN or +inf */
    int64_t max_dense_gradient_elements;/* cap on T*K*V for dbs_backward_dense() [1e8] */
    int reserved0;
    int reserved1;
} DBSOptionsC;

typedef struct DBSDecoderHandle DBSDecoderHandle;
typedef struct DBSResultHandle DBSResultHandle;
typedef struct DBSBackwardHandle DBSBackwardHandle;
typedef struct DBSBatchResultHandle DBSBatchResultHandle;
typedef struct DBSWorkspaceHandle DBSWorkspaceHandle;


/* Model-step callback of dbs_decode_model_steps (superseded by DBSModelStepExFn,
 * which also reports each beam's parent). Fills out_log_probs [K, V] for `step`:
 * row k scores the next token of beam k. prev_tokens [K] holds the token each
 * beam emitted at step - 1 and prev_scores [K] its length-penalised score (at
 * step 0: -1 and 0). Returns 0 on success. */
typedef int (*DBSModelStepFn)(
    void* user_data,
    int batch_index,
    int step,
    const int32_t* prev_tokens,
    const float* prev_scores,
    int beam_size,
    int vocab_size,
    float* out_log_probs
);

/* The beams entering a model step. Slot k is the k-th best hypothesis after
 * step - 1; beams are re-ranked every step, so slot k usually extends a
 * different hypothesis than slot k did one step earlier: parents[k] names the
 * slot it came from. At step 0 only beam 0 is live. Dead slots have parent and
 * token -1 and score -inf. All pointers are valid during the callback only. */
typedef struct DBSModelStepInfoC {
    int batch_index;
    int step;                     /* t: the rows requested extend the beams after step t - 1 */
    int beam_size;                /* K */
    int vocab_size;               /* V */
    const int32_t* parents;       /* [K] slot at step t - 1 that beam k extends (-1 at step 0) */
    const int32_t* tokens;        /* [K] token beam k emitted at step t - 1 (-1 at step 0) */
    const int32_t* lengths;       /* [K] hypothesis lengths */
    const float* scores;          /* [K] length-penalised scores */
    const float* raw_scores;      /* [K] cumulative log-probabilities */
    const uint8_t* finished;      /* [K] non-zero once the beam emitted EOS; its row is not read */
    const int32_t* prefixes;      /* [K * t] row k: the tokens of beam k's path at steps 0..t-1 */
    const void* reserved[4];
} DBSModelStepInfoC;

/* Fills out_log_probs [K, V] (pre-filled with -inf) with the next-token
 * log-probabilities of every beam described by `info`. Returns 0 on success;
 * any other value aborts the decode with DBS_ERROR_RUNTIME. */
typedef int (*DBSModelStepExFn)(void* user_data, const DBSModelStepInfoC* info, float* out_log_probs);

typedef int (*DBSTokenFilterFn)(
    void* user_data,
    int batch_index,
    int step,
    int parent_beam,
    const int32_t* prefix_tokens,
    int prefix_len,
    int token
);

typedef struct DBSAdvancedConstraintsC {
    const uint8_t* banned_tokens;      /* [V], optional; non-zero bans the token */
    const int32_t* forced_tokens;      /* [T], optional, -1 = not forced */
    int min_length;                    /* negative = decoder default */
    float repetition_penalty;          /* <=1 disables; tokens already in the beam's prefix subtract log(penalty) */
    int no_repeat_ngram_size;          /* <=0 disables; n blocks every n-gram already in the beam's prefix */
    DBSTokenFilterFn token_filter;     /* optional; return non-zero to allow token */
    void* token_filter_user_data;
    int batch_index;                   /* passed to token_filter; batch functions pass the example index */
} DBSAdvancedConstraintsC;

/* Caller-owned output arrays of dbs_decode_batch_into(). final_scores is
 * required; any other pointer may be NULL. Steps past an example's own step
 * count hold token/parent -1, length 0, -inf scores and from_logprob 0. */
typedef struct DBSDecodeOutputsC {
    float* final_scores;          /* [B, K] length-penalised final scores, best first */
    float* final_raw_scores;      /* [B, K] cumulative log-probabilities */
    int32_t* final_lengths;       /* [B, K] */
    int32_t* tokens;              /* [B, T, K] token chosen at each step */
    int32_t* parents;             /* [B, T, K] parent beam at the previous step */
    int32_t* lengths;             /* [B, T, K] hypothesis length after each step */
    float* scores;                /* [B, T, K] length-penalised ranking scores */
    float* raw_scores;            /* [B, T, K] cumulative log-probabilities */
    uint8_t* from_logprob;        /* [B, T, K] 0 for EOS carry-forward and padding slots */
} DBSDecodeOutputsC;

typedef struct DBSStatsC {
    int abi_version;
    int last_kernel;                   /* 0 scalar, 1 SSE4.2, 2 AVX2, 3 AVX-512, 4 NEON */
    int used_sparse_backward;
    int used_dense_backward;
    int used_batch_threads;
    int used_model_step_callback;
    int last_error_category;           /* 0 none, 1 invalid argument, 2 runtime, 3 allocation, 4 overflow */
    int64_t last_decode_ns;
    int64_t last_backward_ns;
    int64_t last_allocation_bytes;
    int64_t last_selected_count;
    int64_t last_pool_count;
    int64_t last_logprob_count;
    int64_t last_sparse_grad_count;
    int64_t total_allocator_calls;
    int64_t total_allocator_bytes;
    int64_t reserved_stats0;
    int64_t reserved_stats1;
} DBSStatsC;

typedef enum DBSDataTypeC {
    DBS_DTYPE_F32 = 0,
    DBS_DTYPE_F16 = 1,
    DBS_DTYPE_BF16 = 2
} DBSDataTypeC;

DBS_EXPORT int dbs_abi_version(void);
DBS_EXPORT const char* dbs_version_string(void);
DBS_EXPORT const char* dbs_last_global_error(void);

DBS_EXPORT int dbs_workspace_create(DBSWorkspaceHandle** out_workspace);
DBS_EXPORT void dbs_workspace_destroy(DBSWorkspaceHandle* workspace);
DBS_EXPORT int dbs_workspace_reserve(DBSWorkspaceHandle* workspace, int64_t float_count, int64_t int_count);
DBS_EXPORT int64_t dbs_workspace_allocated_bytes(DBSWorkspaceHandle* workspace);

DBS_EXPORT int dbs_create_ex(DBSOptionsC options, DBSDecoderHandle** out_handle);
DBS_EXPORT DBSDecoderHandle* dbs_create(DBSOptionsC options);
DBS_EXPORT void dbs_destroy(DBSDecoderHandle* handle);
DBS_EXPORT const char* dbs_last_error(DBSDecoderHandle* handle);

DBS_EXPORT int dbs_decode(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_batch_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
);

DBS_EXPORT int dbs_decode_constrained(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const uint8_t* banned_tokens,
    const int32_t* forced_tokens,
    int min_length,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_constrained_ex(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_model_steps(
    DBSDecoderHandle* handle,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_model_steps_with_workspace(
    DBSDecoderHandle* handle,
    DBSWorkspaceHandle* workspace,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

/* Like dbs_decode_model_steps, with the richer DBSModelStepInfoC (parent
 * slots, lengths, finished flags and token prefixes) and optional constraints.
 * The search runs incrementally: step_fn is called once per step. */
DBS_EXPORT int dbs_decode_model_steps_ex(
    DBSDecoderHandle* handle,
    DBSModelStepExFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints,
    DBSResultHandle** out_result
);

/* Decodes a batch [B, T, K, V] straight into caller-owned arrays (no result
 * handles). steps_per_example [B] (values in [1, T]) may be NULL for T steps
 * each; constraints (may be NULL) apply to every example. num_threads <= 0 uses
 * one thread per hardware thread; the calling thread takes part. */
DBS_EXPORT int dbs_decode_batch_into(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int steps,
    int vocab_size,
    const int32_t* steps_per_example,
    const DBSAdvancedConstraintsC* constraints,
    int num_threads,
    const DBSDecodeOutputsC* outputs
);

/* Final-score surrogate gradient for a batch decoded by dbs_decode_batch_into
 * with the same handle, batch_size, steps and steps_per_example. parents,
 * tokens, lengths and from_logprob are its [B, T, K] outputs, grad_final_scores
 * is [B, K]. The gradient is accumulated (+=) into grad_log_probs
 * [B, T, K, V], which the caller normally zero-fills first. */
DBS_EXPORT int dbs_backward_batch_into(
    DBSDecoderHandle* handle,
    int batch_size,
    int steps,
    int vocab_size,
    const int32_t* steps_per_example,
    const int32_t* parents,
    const int32_t* tokens,
    const int32_t* lengths,
    const uint8_t* from_logprob,
    const float* grad_final_scores,
    int num_threads,
    float* grad_log_probs
);

DBS_EXPORT int dbs_decode_batch(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
);

DBS_EXPORT int dbs_decode_batch_variable(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int max_steps,
    int max_beam_size,
    int vocab_size,
    const int32_t* steps_per_example,
    const int32_t* beam_sizes_per_example,
    const int32_t* eos_tokens_per_example,
    const int32_t* min_lengths_per_example,
    const uint8_t* banned_tokens_per_example,
    const int32_t* forced_tokens_per_example,
    int num_threads,
    DBSBatchResultHandle** out_result
);

DBS_EXPORT int dbs_backward(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT int dbs_backward_dense(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT int dbs_backward_sparse(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT int dbs_backward_default(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT void dbs_free_result(DBSResultHandle* result);
DBS_EXPORT void dbs_free_batch_result(DBSBatchResultHandle* result);
DBS_EXPORT void dbs_free_backward(DBSBackwardHandle* result);

DBS_EXPORT int dbs_batch_result_size(const DBSBatchResultHandle* result);
DBS_EXPORT const DBSResultHandle* dbs_batch_result_at(const DBSBatchResultHandle* result, int batch_index);

DBS_EXPORT int dbs_result_steps(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_beam_size(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_vocab_size(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_pool_size(const DBSResultHandle* result);

DBS_EXPORT const int32_t* dbs_result_tokens(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_parents(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_lengths(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_raw_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_weights(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_final_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_final_raw_scores(const DBSResultHandle* result);

DBS_EXPORT const float* dbs_result_relaxed_weights(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_pool_tokens(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_pool_parents(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_pool_lengths(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_pool_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_pool_raw_scores(const DBSResultHandle* result);

DBS_EXPORT const float* dbs_backward_grad_log_probs(const DBSBackwardHandle* result);
DBS_EXPORT const float* dbs_backward_grad_initial_scores(const DBSBackwardHandle* result);
DBS_EXPORT const int64_t* dbs_backward_sparse_logprob_indices(const DBSBackwardHandle* result);
DBS_EXPORT const float* dbs_backward_sparse_logprob_values(const DBSBackwardHandle* result);
DBS_EXPORT int64_t dbs_backward_sparse_logprob_count(const DBSBackwardHandle* result);
DBS_EXPORT int dbs_backward_is_sparse(const DBSBackwardHandle* result);

DBS_EXPORT int64_t dbs_result_selected_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_result_pool_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_result_logprob_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_backward_grad_log_probs_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_backward_grad_initial_scores_count(const DBSResultHandle* result);

DBS_EXPORT int64_t dbs_result_eos_count(const DBSResultHandle* result, int eos_token);
/* 0 if every step's beams follow the decoder's candidate order (score, raw
 * score, parent, token, length, origin); -3 if scores are out of order, -4 if
 * a tie is broken out of order, -1 for a NULL result. */
DBS_EXPORT int dbs_result_validate_deterministic_order(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_summary_json(const DBSResultHandle* result, int eos_token, char* out_json, int64_t out_json_capacity);

DBS_EXPORT int dbs_has_avx512(void);
DBS_EXPORT int dbs_has_avx2(void);
DBS_EXPORT int dbs_has_sse42(void);
DBS_EXPORT int dbs_has_neon(void);
DBS_EXPORT const char* dbs_selected_kernel_name(void);
DBS_EXPORT int dbs_get_stats(DBSDecoderHandle* handle, DBSStatsC* out_stats);
DBS_EXPORT int dbs_get_stats_json(DBSDecoderHandle* handle, char* out_json, int64_t out_json_capacity);
/* Always 1: decoding uses no randomness and a total order over candidates, so
 * equal inputs give equal outputs. */
DBS_EXPORT int dbs_is_deterministic(void);
/* Store and read back a per-handle seed. Nothing in libdbs is random, so the
 * seed has no effect; the functions remain for ABI compatibility. */
DBS_EXPORT int dbs_set_deterministic_seed(DBSDecoderHandle* handle, uint64_t seed);
DBS_EXPORT uint64_t dbs_get_deterministic_seed(DBSDecoderHandle* handle);
/* Process-wide statistics of libdbs' aligned allocator: the number of
 * allocations since the last reset, and the bytes currently allocated (a gauge
 * that dbs_allocator_counters_reset leaves unchanged). */
DBS_EXPORT void dbs_allocator_counters_reset(void);
DBS_EXPORT int64_t dbs_allocator_call_count(void);
DBS_EXPORT int64_t dbs_allocator_byte_count(void);
DBS_EXPORT void dbs_reset_stats(DBSDecoderHandle* handle);

#ifdef __cplusplus
}
#endif
