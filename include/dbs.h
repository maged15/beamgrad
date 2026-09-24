// SPDX-License-Identifier: MIT
//
// libdbs: the C ABI of beamgrad (differentiable beam search).
//
// Conventions
//   * Every function returning int returns 0 on success and a negative value on
//     failure: -1 invalid argument, -2 error raised during the computation,
//     -3 unknown internal error. Details are available from dbs_last_error()
//     (per decoder handle) or dbs_last_global_error() (per thread).
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

/* Decoder options. A zero-initialised field selects its default (shown in
 * brackets); negative or non-finite values are rejected by dbs_create_ex().
 * Note that eos_token and validate_inputs have no "zero = default" rule. */
typedef struct DBSOptionsC {
    int beam_size;                      /* beams K [8] */
    int eos_token;                      /* -1 disables EOS handling; 0 is token 0 */
    float selected_temperature;         /* softmax temperature for selected-beam weights [1.0] */
    float soft_topk_temperature;        /* sigmoid temperature of the relaxed pool [0.25] */
    int relaxed_pool_multiplier;        /* relaxed pool size P = K * multiplier [8] */
    int vocab_block;                    /* vocabulary scan block (a tuning knob; results do not depend on it) [4096] */
    float length_penalty_alpha;         /* GNMT length penalty ((5 + len) / 6)^alpha; 0 disables [0] */
    float soft_topk_tolerance;          /* bisection tolerance for the relaxed pool [1e-4] */
    int soft_topk_max_iters;            /* bisection iteration cap [48] */
    int min_length;                     /* EOS is masked until a hypothesis reaches this length [0] */
    int validate_inputs;                /* non-zero: reject NaN/+Inf log-probs (samples up to 1000 entries) */
    int64_t max_dense_gradient_elements;/* cap on T*K*V for dbs_backward_dense() [1e8] */
    int reserved0;
    int reserved1;
} DBSOptionsC;

typedef struct DBSDecoderHandle DBSDecoderHandle;
typedef struct DBSResultHandle DBSResultHandle;
typedef struct DBSBackwardHandle DBSBackwardHandle;
typedef struct DBSBatchResultHandle DBSBatchResultHandle;
typedef struct DBSWorkspaceHandle DBSWorkspaceHandle;


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
    const uint8_t* banned_tokens;      /* [V], optional */
    const int32_t* forced_tokens;      /* [T], optional, -1 = not forced */
    int min_length;                    /* negative = decoder default */
    float repetition_penalty;          /* <=1 disables; repeated tokens subtract log(penalty) */
    int no_repeat_ngram_size;          /* <=0 disables */
    DBSTokenFilterFn token_filter;     /* optional; return non-zero to allow token */
    void* token_filter_user_data;
    int batch_index;
} DBSAdvancedConstraintsC;

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
DBS_EXPORT int dbs_result_validate_deterministic_order(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_summary_json(const DBSResultHandle* result, int eos_token, char* out_json, int64_t out_json_capacity);
DBS_EXPORT int dbs_validate_production_gate_manifest(const char* manifest_json, char* out_error, int64_t out_error_capacity); /* deprecated compatibility stub; always returns non-zero */

DBS_EXPORT int dbs_has_avx512(void);
DBS_EXPORT int dbs_has_avx2(void);
DBS_EXPORT int dbs_has_sse42(void);
DBS_EXPORT int dbs_has_neon(void);
DBS_EXPORT const char* dbs_selected_kernel_name(void);
DBS_EXPORT int dbs_get_stats(DBSDecoderHandle* handle, DBSStatsC* out_stats);
DBS_EXPORT int dbs_get_stats_json(DBSDecoderHandle* handle, char* out_json, int64_t out_json_capacity);
DBS_EXPORT int dbs_is_deterministic(void); /* reports deterministic hard-decode contract; not a runtime proof */
DBS_EXPORT int dbs_set_deterministic_seed(DBSDecoderHandle* handle, uint64_t seed);
DBS_EXPORT uint64_t dbs_get_deterministic_seed(DBSDecoderHandle* handle);
DBS_EXPORT void dbs_allocator_counters_reset(void);
DBS_EXPORT int64_t dbs_allocator_call_count(void);
DBS_EXPORT int64_t dbs_allocator_byte_count(void);
DBS_EXPORT void dbs_reset_stats(DBSDecoderHandle* handle);

#ifdef __cplusplus
}
#endif
