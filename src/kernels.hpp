// SPDX-License-Identifier: MIT
//
// CPU numerical kernels and their runtime ISA dispatch.
//
// Every kernel has a scalar reference implementation. SIMD variants (AVX-512,
// AVX2, SSE4.2, NEON) are compiled with per-function target attributes and are
// selected at runtime from the host's CPU features, so one binary runs on any
// x86-64 machine. The dispatchers in this header always pick the best path the
// host supports, unless a kernel override is active (used by parity tests).
#pragma once

#include "common.hpp"

namespace dbs {

// Values match DBSStatsC::last_kernel in include/dbs.h.
enum class KernelPath : int {
    Scalar = 0,
    SSE42 = 1,
    AVX2 = 2,
    AVX512 = 3,
    NEON = 4,
};

bool runtime_has_avx512() noexcept;
bool runtime_has_avx2() noexcept;
bool runtime_has_sse42() noexcept;
bool runtime_has_neon() noexcept;

// Thread-local kernel override. When enabled, dispatchers only use `path`
// (falling back to scalar if the host lacks it).
struct KernelOverride {
    bool enabled = false;
    KernelPath path = KernelPath::Scalar;
};

KernelOverride current_kernel_override() noexcept;
void set_kernel_override(KernelOverride value) noexcept;
bool kernel_path_enabled(KernelPath path) noexcept;
KernelPath selected_kernel_path() noexcept;
const char* kernel_path_name(KernelPath path) noexcept;

// Scalar reference kernels.
float dot_scalar(const float* a, const float* b, int n);
void softmax_selected_scalar(const float* scores, float* out, int n, float temperature);
float sum_sigmoid_shifted_scalar(const float* scores, int n, float theta, float temperature);
void soft_topk_write_scalar(const float* scores, float* out, int n, float theta, float temperature);
void scan_parent_row_scalar(
    const float* row,
    float parent_raw,
    int parent_length,
    int parent,
    int vocab_size,
    Candidate* top,
    int top_count,
    int vocab_block,
    float length_penalty_alpha,
    const uint8_t* banned_tokens,
    int forced_token,
    int eos_token,
    int min_length);

#if DBS_CAN_COMPILE_AVX512
namespace avx512 {
float dot(const float* a, const float* b, int n);
void softmax_selected(const float* scores, float* out, int n, float temperature);
float sum_sigmoid_shifted(const float* scores, int n, float theta, float temperature);
void soft_topk_write(const float* scores, float* out, int n, float theta, float temperature);
void scan_parent_row(
    const float* row, float parent_raw, int parent_length, int parent, int vocab_size,
    Candidate* top, int top_count, int vocab_block, float length_penalty_alpha,
    const uint8_t* banned_tokens, int forced_token, int eos_token, int min_length);
// 16-lane vector math entry points, exposed for the internal parity tests.
void exp16(const float* in, float* out);
void sigmoid16(const float* in, float* out);
} // namespace avx512
#endif

#if DBS_CAN_COMPILE_AVX2
namespace avx2 {
float dot(const float* a, const float* b, int n);
void softmax_selected(const float* scores, float* out, int n, float temperature);
void scan_parent_row(
    const float* row, float parent_raw, int parent_length, int parent, int vocab_size,
    Candidate* top, int top_count, int vocab_block, float length_penalty_alpha,
    const uint8_t* banned_tokens, int forced_token, int eos_token, int min_length);
} // namespace avx2
#endif

#if DBS_CAN_COMPILE_SSE42
namespace sse42 {
float dot(const float* a, const float* b, int n);
void softmax_selected(const float* scores, float* out, int n, float temperature);
void scan_parent_row(
    const float* row, float parent_raw, int parent_length, int parent, int vocab_size,
    Candidate* top, int top_count, int vocab_block, float length_penalty_alpha,
    const uint8_t* banned_tokens, int forced_token, int eos_token, int min_length);
} // namespace sse42
#endif

// Dispatchers: pick the best kernel for the host (honouring any override).
float dot(const float* a, const float* b, int n);
void softmax_selected(const float* scores, float* out, int n, float temperature);
float sum_sigmoid_shifted(const float* scores, int n, float theta, float temperature);
void soft_topk_write(const float* scores, float* out, int n, float theta, float temperature);
void scan_parent_row(
    const float* row, float parent_raw, int parent_length, int parent, int vocab_size,
    Candidate* top, int top_count, int vocab_block, float length_penalty_alpha,
    const uint8_t* banned_tokens, int forced_token, int eos_token, int min_length);

// Sigmoid k-hot relaxation: finds theta by bisection so that
// sum_i sigmoid((scores[i] - theta) / temperature) ~= target_k, then writes the
// per-candidate inclusion weights. Scores <= -1e30 are treated as padding.
void soft_topk_inclusion(
    const float* scores, float* out, int n, int target_k,
    float temperature, float tolerance, int max_iters);

} // namespace dbs
