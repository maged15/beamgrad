// SPDX-License-Identifier: MIT
//
// CPU numerical kernels and their runtime ISA dispatch.
//
// The hot loop of the decoder is the row scan: turning one parent beam's
// [V] log-probabilities into candidates and keeping the best of them. It has a
// scalar reference implementation and SIMD variants (AVX-512, AVX2, SSE4.2,
// NEON). The x86 variants are compiled with per-function target attributes (or
// as plain intrinsics on MSVC) and chosen at runtime from the host's CPU
// features, so one binary runs on any x86-64 machine. Every variant produces
// exactly the same candidates as the scalar reference.
//
// The small per-step reductions over K selected beams (softmax, dot product,
// relaxed top-k) are scalar on every platform: they are cheap, and a fixed
// evaluation order keeps their results identical on all kernel paths.
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

// Thread-local kernel override. When enabled, the dispatcher only uses `path`
// (falling back to scalar if the host lacks it). Used by the parity tests.
struct KernelOverride {
    bool enabled = false;
    KernelPath path = KernelPath::Scalar;
};

KernelOverride current_kernel_override() noexcept;
void set_kernel_override(KernelOverride value) noexcept;
bool kernel_path_enabled(KernelPath path) noexcept;
KernelPath selected_kernel_path() noexcept;
const char* kernel_path_name(KernelPath path) noexcept;

// One parent beam's expansion: every token v of `row` is a candidate
// (parent, v) with raw score parent_raw + row[v] and ranking score
// raw * inv_penalty, unless it is excluded.
struct RowScan {
    const float* row;          // [V] log-probabilities conditioned on the parent's prefix
    const uint8_t* banned;     // [V], non-zero excludes the token; may be null
    float parent_raw;          // the parent's cumulative log-probability (finite)
    float inv_penalty;         // 1 / gnmt_length_penalty(new_length, alpha)
    int parent;
    int new_length;            // parent length + 1
    int vocab_size;
    int forced_token;          // >= 0: only this token is a candidate
    int masked_token;          // >= 0: this token is excluded (EOS below min_length)
};

// Inserts every finite, allowed candidate of the row into `top` (a descending
// buffer of top_count candidates, see insert_topk). Returns true if the row
// contains NaN or +inf anywhere, including at excluded tokens; such entries
// are never candidates.
bool scan_row(const RowScan& scan, Candidate* top, int top_count);
bool scan_row_scalar(const RowScan& scan, Candidate* top, int top_count);

#if DBS_CAN_COMPILE_AVX512
namespace avx512 {
bool scan_row(const RowScan& scan, Candidate* top, int top_count);
}
#endif
#if DBS_CAN_COMPILE_AVX2
namespace avx2 {
bool scan_row(const RowScan& scan, Candidate* top, int top_count);
}
#endif
#if DBS_CAN_COMPILE_SSE42
namespace sse42 {
bool scan_row(const RowScan& scan, Candidate* top, int top_count);
}
#endif
#if DBS_ARM_NEON
namespace neon {
bool scan_row(const RowScan& scan, Candidate* top, int top_count);
}
#endif

// Scalar reference for one token (the masked token is handled by the caller,
// see scan_around_masked); the SIMD kernels use it for their tails. Returns
// true if row[v] is NaN or +inf.
inline bool scan_token(const RowScan& s, int v, Candidate* top, int top_count) noexcept {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    const float lp = s.row[v];
    if (!(lp < kInf)) return true;
    if (lp == -kInf) return false;
    if (s.banned && s.banned[v]) return false;
    const float raw = s.parent_raw + lp;
    insert_topk(top, top_count, Candidate{raw * s.inv_penalty, raw, s.parent, v, s.new_length, 1});
    return false;
}

// Runs scan_range(begin, end) over the row without the masked token, which is
// only checked for NaN/+inf, and returns whether any entry was NaN or +inf.
template <class ScanRange>
inline bool scan_around_masked(const RowScan& s, ScanRange&& scan_range) {
    const int m = s.masked_token;
    if (m < 0) return scan_range(0, s.vocab_size);
    bool invalid = !(s.row[m] < std::numeric_limits<float>::infinity());
    invalid |= scan_range(0, m);
    invalid |= scan_range(m + 1, s.vocab_size);
    return invalid;
}

inline int count_trailing_zeros(uint32_t x) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index = 0;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
#else
    return __builtin_ctz(x);
#endif
}

// Deterministic scalar reductions over the K selected beams.
float dot(const float* a, const float* b, int n);
void softmax_selected(const float* scores, float* out, int n, float temperature);

// Sigmoid k-hot relaxation: finds theta by bisection so that
// sum_i sigmoid((scores[i] - theta) / temperature) ~= target_k, then writes the
// per-candidate inclusion weights. Scores <= -1e30 are treated as padding.
// Bisection stops when the sum is within `tolerance` of target_k or theta is
// bracketed to tolerance * temperature, whatever the scores' magnitude (theta
// is bisected as an offset from the best score), or after max_iters.
void soft_topk_inclusion(
    const float* scores, float* out, int n, int target_k,
    float temperature, float tolerance, int max_iters);

} // namespace dbs
