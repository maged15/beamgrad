// SPDX-License-Identifier: MIT
//
// Internal types shared by the CPU decoder, the SIMD kernels, and the C ABI.
// Nothing in this header is part of the public interface (see include/dbs.h).
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define DBS_X86 1
#else
#define DBS_X86 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define DBS_ARM_NEON 1
#else
#define DBS_ARM_NEON 0
#endif

#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
#define DBS_CAN_COMPILE_AVX512 1
#define DBS_AVX512_TARGET __attribute__((target("avx512f,fma")))
#else
#define DBS_CAN_COMPILE_AVX512 0
#define DBS_AVX512_TARGET
#endif

#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
#define DBS_CAN_COMPILE_AVX2 1
#define DBS_AVX2_TARGET __attribute__((target("avx2,fma")))
#define DBS_CAN_COMPILE_SSE42 1
#define DBS_SSE42_TARGET __attribute__((target("sse4.2")))
#else
#define DBS_CAN_COMPILE_AVX2 0
#define DBS_AVX2_TARGET
#define DBS_CAN_COMPILE_SSE42 0
#define DBS_SSE42_TARGET
#endif

namespace dbs {

// Process-wide counters for the aligned allocator (reported through dbs_get_stats).
inline std::atomic<int64_t> g_allocator_calls{0};
inline std::atomic<int64_t> g_allocator_bytes{0};

using TokenFilterFn = int (*)(
    void* user_data,
    int batch_index,
    int step,
    int parent_beam,
    const int32_t* prefix_tokens,
    int prefix_len,
    int token
);

template <class T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;

        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        const std::size_t bytes = n * sizeof(T);
        void* p = nullptr;

#if defined(_MSC_VER)
        p = _aligned_malloc(bytes, Alignment);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif

        g_allocator_calls.fetch_add(1, std::memory_order_relaxed);
        g_allocator_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        g_allocator_bytes.fetch_sub(static_cast<int64_t>(n * sizeof(T)), std::memory_order_relaxed);
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <class T, class U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return true;
}

template <class T, class U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return false;
}

using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;
using AlignedIntVector = std::vector<int32_t, AlignedAllocator<int32_t, 64>>;
using AlignedInt64Vector = std::vector<int64_t, AlignedAllocator<int64_t, 64>>;

struct BeamOptions {
    int beam_size = 8;
    int eos_token = -1;

    float selected_temperature = 1.0f;
    float soft_topk_temperature = 0.25f;

    int relaxed_pool_multiplier = 8;
    int vocab_block = 4096;

    float length_penalty_alpha = 0.0f;
    float soft_topk_tolerance = 1.0e-4f;
    int soft_topk_max_iters = 48;

    int min_length = 0;
    int validate_inputs = 1;
    int64_t max_dense_gradient_elements = 100000000LL;
};

struct DecodeConstraints {
    // banned_tokens is optional and has shape [V]. Non-zero means token is not allowed.
    const uint8_t* banned_tokens = nullptr;

    // forced_tokens is optional and has shape [T]. A value >= 0 forces that token at the step.
    const int32_t* forced_tokens = nullptr;

    // Optional per-call minimum output length. Negative means use BeamOptions::min_length.
    int min_length = -1;

    // Optional decode constraints. Disabled when repetition_penalty <= 1 and no_repeat_ngram_size <= 0.
    float repetition_penalty = 1.0f;
    int no_repeat_ngram_size = 0;
    TokenFilterFn token_filter = nullptr;
    void* token_filter_user_data = nullptr;
    int batch_index = 0;
};

struct DecodeResult {
    int steps = 0;
    int beam_size = 0;
    int vocab_size = 0;
    int eos_token = -1;
    int relaxed_pool_size = 0;

    float selected_temperature = 1.0f;
    float soft_topk_temperature = 0.25f;
    float length_penalty_alpha = 0.0f;

    AlignedIntVector parents;       // [T * K]
    AlignedIntVector tokens;        // [T * K]
    AlignedIntVector lengths;       // [T * K]

    AlignedFloatVector raw_scores;  // [T * K]
    AlignedFloatVector scores;      // length-penalized ranking scores [T * K]
    AlignedFloatVector weights;     // selected-beam softmax weights [T * K]

    AlignedFloatVector final_raw_scores; // [K]
    AlignedFloatVector final_scores;     // [K]

    std::vector<uint8_t> from_logprob;   // [T * K]

    AlignedIntVector pool_parents;       // [T * P]
    AlignedIntVector pool_tokens;        // [T * P]
    AlignedIntVector pool_lengths;       // [T * P]

    AlignedFloatVector pool_raw_scores;  // [T * P]
    AlignedFloatVector pool_scores;      // [T * P]
    AlignedFloatVector relaxed_weights;  // [T * P]

    std::vector<uint8_t> pool_from_logprob; // [T * P]

    std::vector<std::vector<int32_t>> sequences;
};

struct BackwardResult {
    AlignedFloatVector grad_log_probs;      // [T * K * V], populated only by explicit dense backward.
    AlignedFloatVector grad_initial_scores; // [K]

    AlignedInt64Vector sparse_logprob_indices; // flattened [T*K*V] indices, populated by sparse/default backward.
    AlignedFloatVector sparse_logprob_values;  // same count as sparse_logprob_indices.
    bool sparse = false;
};

struct SparseGradEntry {
    int64_t index;
    float value;
};

struct Candidate {
    float score;
    float raw_score;
    int32_t parent;
    int32_t token;
    int32_t length;
    uint8_t from_logprob;
};

inline float safe_exp_scalar(float x) {
    x = std::min(88.3762626647949f, std::max(-88.3762626647949f, x));
    return std::exp(x);
}

inline float sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        const float e = safe_exp_scalar(-x);
        return 1.0f / (1.0f + e);
    }

    const float e = safe_exp_scalar(x);
    return e / (1.0f + e);
}

// GNMT length penalty ((5 + len) / 6)^alpha. Evaluated in double precision and
// rounded once, so every platform and the CUDA backend get the same float.
inline float gnmt_length_penalty(int length, float alpha) {
    if (alpha == 0.0f) return 1.0f;

    const int l = std::max(1, length);
    return static_cast<float>(std::pow((5.0 + static_cast<double>(l)) / 6.0, static_cast<double>(alpha)));
}

inline size_t checked_mul_size(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(what);
    }
    return a * b;
}

inline bool candidate_better(const Candidate& a, const Candidate& b) noexcept {
    if (a.score != b.score) return a.score > b.score;
    if (a.raw_score != b.raw_score) return a.raw_score > b.raw_score;
    if (a.parent != b.parent) return a.parent < b.parent;
    if (a.token != b.token) return a.token < b.token;
    if (a.length != b.length) return a.length < b.length;
    return a.from_logprob > b.from_logprob;
}

inline void insert_topk(Candidate* top, int k, const Candidate& c) noexcept {
    if (!candidate_better(c, top[k - 1])) return;

    // Maintains a descending top-P buffer with deterministic tie-breaking.
    int lo = 0;
    int hi = k - 1;

    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);

        if (candidate_better(c, top[mid])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    const int pos = lo;

    for (int i = k - 1; i > pos; --i) {
        top[i] = top[i - 1];
    }

    top[pos] = c;
}

} // namespace dbs
