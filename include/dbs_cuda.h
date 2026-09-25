// SPDX-License-Identifier: MIT
//
// libdbs_cuda: native CUDA beam search with the same semantics as the CPU
// decoder in libdbs (include/dbs.h).
//
// dbs_cuda_decode() runs hard beam search on the GPU: deterministic candidate
// ordering (score, raw score, parent, token), GNMT length penalty, EOS
// carry-forward, EOS min-length masking, banned tokens, n-gram blocking, a
// repetition penalty, and per-example variable steps, beam sizes, EOS tokens
// and minimum lengths. Given identical inputs it selects the same beams as the
// CPU decoder and produces the same scores, bit for bit.
//
// dbs_cuda_backward() computes the surrogate gradient of the final beam scores
// with respect to log_probs: each final beam's upstream gradient (scaled by its
// length penalty) flows back along its selected path. This matches
// dbs_backward() on the CPU with only grad_final_scores supplied, and is
// deterministic (no atomics).
//
// All tensor pointers are device pointers to dense row-major data. log_probs
// and grad_log_probs are float32 [B, T, K, V], where T and K are the (maximum)
// steps and beam size; per-step outputs are [B, T, K] and per-beam outputs are
// [B, K]. Slots outside an example's active steps/beams are filled with token
// and parent -1, length 0, and -inf scores.
//
// Calls are asynchronous with respect to the host: work is queued on `stream`
// (a cudaStream_t, or NULL for the legacy default stream) and launch errors are
// reported through the return value. Set DBS_CUDA_SYNC_CHECK=1 (or call
// dbs_cuda_set_synchronization(1)) to synchronize after every call when
// debugging, so device-side faults surface at the call site.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(DBS_STATIC)
#define DBS_CUDA_EXPORT
#elif defined(_WIN32) && defined(DBS_BUILD_SHARED) && defined(DBS_COMPILING_LIBRARY)
#define DBS_CUDA_EXPORT __declspec(dllexport)
#elif defined(_WIN32) && defined(DBS_BUILD_SHARED)
#define DBS_CUDA_EXPORT __declspec(dllimport)
#elif defined(_WIN32)
#define DBS_CUDA_EXPORT
#elif defined(__GNUC__) || defined(__clang__)
#define DBS_CUDA_EXPORT __attribute__((visibility("default")))
#else
#define DBS_CUDA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DBS_CUDA_STATUS_OK 0
#define DBS_CUDA_STATUS_UNAVAILABLE 1      /* library built without CUDA, or no device */
#define DBS_CUDA_STATUS_INVALID_ARGUMENT 2
#define DBS_CUDA_STATUS_LAUNCH_FAILED 3
#define DBS_CUDA_STATUS_OUT_OF_MEMORY 4

/* Largest supported beam size. */
#define DBS_CUDA_MAX_BEAM 1024

typedef struct DBSCudaDecodeArgs {
    int batch_size;               /* B */
    int steps;                    /* T; the maximum when steps_per_example is set */
    int beam_size;                /* K; also the beam stride of every [.., K, ..] tensor */
    int vocab_size;               /* V */
    int eos_token;                /* -1 disables EOS handling */
    int min_length;               /* EOS is masked until a hypothesis reaches this length */
    float length_penalty_alpha;   /* GNMT penalty ((5 + len) / 6)^alpha; 0 disables */
    int no_repeat_ngram_size;     /* n > 0 blocks tokens that would repeat an n-gram of the beam's prefix */
    float repetition_penalty;     /* > 1: tokens already in the beam's prefix lose log(penalty); <= 1 disables */
    int reserved0;                /* must be 0 */
    /* Optional per-example overrides (device pointers to int32 [B], or NULL).
     * steps must be in [1, T], beams in [1, K], EOS in [-1, V), min lengths >= 0. */
    const int32_t* steps_per_example;
    const int32_t* beam_sizes_per_example;
    const int32_t* eos_tokens_per_example;
    const int32_t* min_lengths_per_example;
    const uint8_t* banned_tokens; /* optional device [V]: non-zero bans the token in every example */
} DBSCudaDecodeArgs;

typedef struct DBSCudaDecodeOutputs {
    float* final_scores;          /* required: [B, K] length-penalised final scores */
    float* final_raw_scores;      /* optional: [B, K] cumulative log-probabilities */
    int32_t* final_lengths;       /* optional: [B, K] */
    int32_t* tokens;              /* optional: [B, T, K] token chosen at each step */
    int32_t* parents;             /* optional: [B, T, K] parent beam at the previous step */
    int32_t* lengths;             /* optional: [B, T, K] hypothesis length after each step */
    float* scores;                /* optional: [B, T, K] length-penalised ranking scores */
    float* raw_scores;            /* optional: [B, T, K] cumulative log-probabilities */
    uint8_t* from_logprob;        /* optional: [B, T, K] 0 for EOS carry-forward slots */
    uint8_t* invalid_input;       /* optional: [B] 1 if a row the search read for example b holds NaN or +inf */
} DBSCudaDecodeOutputs;

DBS_CUDA_EXPORT int dbs_cuda_available(void);
DBS_CUDA_EXPORT const char* dbs_cuda_status_string(int status);

/* 1: synchronize the stream after every call (debugging). 0 (default): async. */
DBS_CUDA_EXPORT int dbs_cuda_set_synchronization(int synchronize);
DBS_CUDA_EXPORT int dbs_cuda_get_synchronization(void);

/* Scratch memory needed by dbs_cuda_decode / dbs_cuda_backward, in bytes, or a
 * negative value if the arguments are invalid. */
DBS_CUDA_EXPORT int64_t dbs_cuda_decode_workspace_size(const DBSCudaDecodeArgs* args);
DBS_CUDA_EXPORT int64_t dbs_cuda_backward_workspace_size(const DBSCudaDecodeArgs* args);

/* Hard beam search. `workspace` may be NULL, in which case scratch memory is
 * allocated and released on `stream` with cudaMallocAsync/cudaFreeAsync.
 * log_probs entries that are NaN or +/-Inf are never selected. Every element of
 * every row the search reads (the rows of live, unfinished beams) is checked
 * for NaN and +inf while it is scanned, and outputs->invalid_input reports the
 * examples that had any; reading the flags back is up to the caller. When
 * per-example arrays are supplied they are validated on the device, which
 * synchronizes the stream once. */
DBS_CUDA_EXPORT int dbs_cuda_decode(
    const float* log_probs,
    const DBSCudaDecodeArgs* args,
    const DBSCudaDecodeOutputs* outputs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream);

/* The beams of a search between steps, for callers that run the search loop
 * themselves (one dbs_cuda_decode_step() per step, e.g. to ask a model for each
 * step's rows given the beams chosen so far). Device arrays. Before the first
 * step: raw_scores is 0 for beam 0 and -inf for the others, lengths and
 * finished are 0. */
typedef struct DBSCudaBeamState {
    float* raw_scores;          /* [B, K] cumulative log-probabilities (updated in place) */
    int32_t* lengths;           /* [B, K] hypothesis lengths (updated in place) */
    uint8_t* finished;          /* [B, K] non-zero once a beam emitted EOS (updated in place) */
    const int32_t* prefixes;    /* [B, K, prefix_stride]: row (b, k) starts with the lengths[b, k]
                                   tokens of beam k's hypothesis. Read only with n-gram blocking or
                                   a repetition penalty; tokens outside [0, V) are ignored. */
    int prefix_stride;          /* tokens per prefix row, >= every live beam's length */
    int reserved0;              /* must be 0 */
} DBSCudaBeamState;

/* Scratch memory needed by dbs_cuda_decode_step, in bytes, or a negative value
 * if the arguments are invalid. */
DBS_CUDA_EXPORT int64_t dbs_cuda_decode_step_workspace_size(const DBSCudaDecodeArgs* args, int prefix_stride);

/* One step of the search. log_probs is the step's [B, K, V] rows (row k
 * extends beam k); args->steps must be 1 and args->steps_per_example NULL.
 * Advances `state`, and writes the step's [B, K] tokens, parents, lengths,
 * scores, raw_scores and from_logprob, the invalid_input flags, and (when
 * final_scores is not NULL) the final_* arrays of the beams after this step.
 * Stepping from the initial state through the rows of a [B, T, K, V] tensor
 * selects exactly what dbs_cuda_decode() selects. */
DBS_CUDA_EXPORT int dbs_cuda_decode_step(
    const float* log_probs,
    const DBSCudaDecodeArgs* args,
    const DBSCudaBeamState* state,
    const DBSCudaDecodeOutputs* outputs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream);

/* Surrogate gradient of the final scores. parents, tokens, lengths and
 * from_logprob come from dbs_cuda_decode() with the same args.
 * grad_final_scores is [B, K]; the gradient is accumulated (+=) into
 * grad_log_probs [B, T, K, V], which the caller normally zero-fills first.
 * One thread block per example walks the steps backwards; the beams of a step
 * are processed in parallel, and each parent sums its children in slot order,
 * so the result is deterministic and equal to the CPU backward. */
DBS_CUDA_EXPORT int dbs_cuda_backward(
    const DBSCudaDecodeArgs* args,
    const int32_t* parents,
    const int32_t* tokens,
    const int32_t* lengths,
    const uint8_t* from_logprob,
    const float* grad_final_scores,
    float* grad_log_probs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream);

#ifdef __cplusplus
}
#endif
