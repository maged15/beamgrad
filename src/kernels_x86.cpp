// SPDX-License-Identifier: MIT
//
// x86 SIMD row scans. Each function carries its own target attribute (GCC,
// Clang) or relies on MSVC accepting intrinsics anywhere, so this file is
// compiled for the baseline ISA and a fast path is only entered after runtime
// feature detection (see cpu_features.cpp).
//
// Every scan matches scan_row_scalar exactly: a lane becomes a candidate if its
// log-probability is finite, the token is not banned, and its ranking score
// reaches the current worst entry of `top` (ties must still reach insert_topk,
// whose full comparator can rank them higher). The final top-k set does not
// depend on the order candidates are inserted in. The masked token (EOS below
// the minimum length) is excluded by scanning the ranges around it.
#include "kernels.hpp"

#include <cstring>

namespace dbs {

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
} // namespace

#if DBS_CAN_COMPILE_AVX512
namespace avx512 {

namespace {

template <bool kHasBanned>
DBS_AVX512_TARGET bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const __m512 parent_vec = _mm512_set1_ps(s.parent_raw);
    const __m512 scale_vec = _mm512_set1_ps(s.inv_penalty);
    const __m512 pos_inf = _mm512_set1_ps(kInf);
    const __m512 neg_inf = _mm512_set1_ps(-kInf);
    alignas(64) float rank_tmp[16];
    alignas(64) float raw_tmp[16];
    // About four cache lines ahead: enough for streaming rows without pulling
    // too much cold vocabulary into L1.
    constexpr int kPrefetchFloats = 256;
    bool invalid = false;

    for (int v = begin; v < end; v += 16) {
        const int rem = end - v;
        const __mmask16 lanes = rem >= 16 ? static_cast<__mmask16>(0xffff) : static_cast<__mmask16>((1u << rem) - 1u);
        __m512 lp;
        if (rem >= 16) {
            if (v + kPrefetchFloats < end) _mm_prefetch(reinterpret_cast<const char*>(s.row + v + kPrefetchFloats), _MM_HINT_T0);
            lp = _mm512_loadu_ps(s.row + v);
        } else {
            lp = _mm512_maskz_loadu_ps(lanes, s.row + v);
        }
        const __mmask16 below_inf = _mm512_mask_cmp_ps_mask(lanes, lp, pos_inf, _CMP_LT_OQ);  // false for NaN and +inf
        if (below_inf != lanes) invalid = true;
        __mmask16 ok = _mm512_mask_cmp_ps_mask(below_inf, lp, neg_inf, _CMP_NEQ_OQ);
        if (kHasBanned) {
            __m128i bytes;
            if (rem >= 16) {
                bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s.banned + v));
            } else {
                alignas(16) uint8_t tail[16] = {};
                std::memcpy(tail, s.banned + v, static_cast<size_t>(rem));
                bytes = _mm_load_si128(reinterpret_cast<const __m128i*>(tail));
            }
            const __m512i wide = _mm512_cvtepu8_epi32(bytes);
            ok = _mm512_mask_testn_epi32_mask(ok, wide, wide);
        }
        const __m512 raw_vec = _mm512_add_ps(lp, parent_vec);
        const __m512 rank_vec = _mm512_mul_ps(raw_vec, scale_vec);
        const __mmask16 hit = _mm512_mask_cmp_ps_mask(ok, rank_vec, _mm512_set1_ps(top[top_count - 1].score), _CMP_GE_OQ);
        if (!hit) continue;

        _mm512_store_ps(rank_tmp, rank_vec);
        _mm512_store_ps(raw_tmp, raw_vec);
        for (uint32_t m = hit; m; m &= m - 1) {
            const int lane = count_trailing_zeros(m);
            insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], s.parent, v + lane, s.new_length, 1});
        }
    }
    return invalid;
}

} // namespace

DBS_AVX512_TARGET bool scan_row(const RowScan& s, Candidate* top, int top_count) {
    if (s.forced_token >= 0) return scan_row_scalar(s, top, top_count);
    return scan_around_masked(s, [&](int begin, int end) {
        return s.banned ? scan_range<true>(s, begin, end, top, top_count) : scan_range<false>(s, begin, end, top, top_count);
    });
}

} // namespace avx512
#endif

#if DBS_CAN_COMPILE_AVX2
namespace avx2 {

namespace {

template <bool kHasBanned>
DBS_AVX2_TARGET bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const __m256 parent_vec = _mm256_set1_ps(s.parent_raw);
    const __m256 scale_vec = _mm256_set1_ps(s.inv_penalty);
    const __m256 pos_inf = _mm256_set1_ps(kInf);
    const __m256 neg_inf = _mm256_set1_ps(-kInf);
    const __m256i zero = _mm256_setzero_si256();
    alignas(32) float rank_tmp[8];
    alignas(32) float raw_tmp[8];
    __m256 below_all = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

    int v = begin;
    for (; v + 8 <= end; v += 8) {
        const __m256 lp = _mm256_loadu_ps(s.row + v);
        const __m256 below_inf = _mm256_cmp_ps(lp, pos_inf, _CMP_LT_OQ);  // false for NaN and +inf
        below_all = _mm256_and_ps(below_all, below_inf);
        __m256 ok = _mm256_and_ps(below_inf, _mm256_cmp_ps(lp, neg_inf, _CMP_NEQ_OQ));
        if (kHasBanned) {
            const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s.banned + v));
            ok = _mm256_and_ps(ok, _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_cvtepu8_epi32(bytes), zero)));
        }
        const __m256 raw_vec = _mm256_add_ps(lp, parent_vec);
        const __m256 rank_vec = _mm256_mul_ps(raw_vec, scale_vec);
        const __m256 reaches = _mm256_cmp_ps(rank_vec, _mm256_set1_ps(top[top_count - 1].score), _CMP_GE_OQ);
        const int hit = _mm256_movemask_ps(_mm256_and_ps(ok, reaches));
        if (!hit) continue;

        _mm256_store_ps(rank_tmp, rank_vec);
        _mm256_store_ps(raw_tmp, raw_vec);
        for (uint32_t m = static_cast<uint32_t>(hit); m; m &= m - 1) {
            const int lane = count_trailing_zeros(m);
            insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], s.parent, v + lane, s.new_length, 1});
        }
    }
    bool invalid = _mm256_movemask_ps(below_all) != 0xff;
    for (; v < end; ++v) invalid |= scan_token(s, v, top, top_count);
    return invalid;
}

} // namespace

DBS_AVX2_TARGET bool scan_row(const RowScan& s, Candidate* top, int top_count) {
    if (s.forced_token >= 0) return scan_row_scalar(s, top, top_count);
    return scan_around_masked(s, [&](int begin, int end) {
        return s.banned ? scan_range<true>(s, begin, end, top, top_count) : scan_range<false>(s, begin, end, top, top_count);
    });
}

} // namespace avx2
#endif

#if DBS_CAN_COMPILE_SSE42
namespace sse42 {

namespace {

template <bool kHasBanned>
DBS_SSE42_TARGET bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const __m128 parent_vec = _mm_set1_ps(s.parent_raw);
    const __m128 scale_vec = _mm_set1_ps(s.inv_penalty);
    const __m128 pos_inf = _mm_set1_ps(kInf);
    const __m128 neg_inf = _mm_set1_ps(-kInf);
    const __m128i zero = _mm_setzero_si128();
    alignas(16) float rank_tmp[4];
    alignas(16) float raw_tmp[4];
    __m128 below_all = _mm_castsi128_ps(_mm_set1_epi32(-1));

    int v = begin;
    for (; v + 4 <= end; v += 4) {
        const __m128 lp = _mm_loadu_ps(s.row + v);
        const __m128 below_inf = _mm_cmplt_ps(lp, pos_inf);  // false for NaN and +inf
        below_all = _mm_and_ps(below_all, below_inf);
        __m128 ok = _mm_and_ps(below_inf, _mm_cmpneq_ps(lp, neg_inf));
        if (kHasBanned) {
            int32_t word = 0;
            std::memcpy(&word, s.banned + v, 4);
            ok = _mm_and_ps(ok, _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(word)), zero)));
        }
        const __m128 raw_vec = _mm_add_ps(lp, parent_vec);
        const __m128 rank_vec = _mm_mul_ps(raw_vec, scale_vec);
        const int hit = _mm_movemask_ps(_mm_and_ps(ok, _mm_cmpge_ps(rank_vec, _mm_set1_ps(top[top_count - 1].score))));
        if (!hit) continue;

        _mm_store_ps(rank_tmp, rank_vec);
        _mm_store_ps(raw_tmp, raw_vec);
        for (uint32_t m = static_cast<uint32_t>(hit); m; m &= m - 1) {
            const int lane = count_trailing_zeros(m);
            insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], s.parent, v + lane, s.new_length, 1});
        }
    }
    bool invalid = _mm_movemask_ps(below_all) != 0xf;
    for (; v < end; ++v) invalid |= scan_token(s, v, top, top_count);
    return invalid;
}

} // namespace

DBS_SSE42_TARGET bool scan_row(const RowScan& s, Candidate* top, int top_count) {
    if (s.forced_token >= 0) return scan_row_scalar(s, top, top_count);
    return scan_around_masked(s, [&](int begin, int end) {
        return s.banned ? scan_range<true>(s, begin, end, top, top_count) : scan_range<false>(s, begin, end, top, top_count);
    });
}

} // namespace sse42
#endif

} // namespace dbs
