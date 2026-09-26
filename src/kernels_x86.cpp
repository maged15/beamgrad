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

template <bool kHasBanned, bool kHasOffset>
DBS_AVX512_TARGET bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const __m512 offset_vec = _mm512_set1_ps(s.offset);
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
        if (kHasOffset) lp = _mm512_sub_ps(lp, offset_vec);
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
        if (s.has_offset) {
            return s.banned ? scan_range<true, true>(s, begin, end, top, top_count)
                            : scan_range<false, true>(s, begin, end, top, top_count);
        }
        return s.banned ? scan_range<true, false>(s, begin, end, top, top_count)
                        : scan_range<false, false>(s, begin, end, top, top_count);
    });
}

namespace {

// exp_nonpositive on 16 lanes, in the scalar operation order; lanes outside
// `valid` give 0.
DBS_AVX512_TARGET inline __m512 exp16(__m512 x, __mmask16 valid) {
    valid &= _mm512_cmp_ps_mask(x, _mm512_set1_ps(det::kExpMin), _CMP_GE_OQ);
    const __m512 k = _mm512_roundscale_ps(_mm512_add_ps(_mm512_mul_ps(x, _mm512_set1_ps(det::kLog2e)), _mm512_set1_ps(0.5f)),
                                          _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    const __m512 r = _mm512_sub_ps(_mm512_sub_ps(x, _mm512_mul_ps(k, _mm512_set1_ps(det::kLn2Hi))),
                                   _mm512_mul_ps(k, _mm512_set1_ps(det::kLn2Lo)));
    const __m512 r2 = _mm512_mul_ps(r, r);
    __m512 p = _mm512_set1_ps(det::kExpP0);
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(det::kExpP1));
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(det::kExpP2));
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(det::kExpP3));
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(det::kExpP4));
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(det::kExpP5));
    p = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(p, r2), r), _mm512_set1_ps(1.0f));
    const __m512i e = _mm512_add_epi32(_mm512_cvttps_epi32(k), _mm512_set1_epi32(127));
    return _mm512_maskz_mul_ps(valid, p, _mm512_castsi512_ps(_mm512_slli_epi32(e, 23)));
}

DBS_AVX512_TARGET inline __mmask16 finite16(__m512 x) {
    return _mm512_cmp_ps_mask(x, _mm512_set1_ps(kInf), _CMP_LT_OQ) & _mm512_cmp_ps_mask(x, _mm512_set1_ps(-kInf), _CMP_GT_OQ);
}

} // namespace

DBS_AVX512_TARGET LogitStats logit_stats(const float* row, int V) {
    LogitStats st{-kInf, -kInf, false};
    __m512 vmax = _mm512_set1_ps(-kInf);
    for (int v = 0; v < V; v += 16) {
        const __mmask16 lanes = V - v >= 16 ? static_cast<__mmask16>(0xffff) : static_cast<__mmask16>((1u << (V - v)) - 1u);
        const __m512 x = _mm512_maskz_loadu_ps(lanes, row + v);
        if (_mm512_mask_cmp_ps_mask(lanes, x, _mm512_set1_ps(kInf), _CMP_LT_OQ) != lanes) st.invalid = true;
        vmax = _mm512_mask_max_ps(vmax, lanes & finite16(x), vmax, x);
    }
    alignas(64) float maxima[16];
    _mm512_store_ps(maxima, vmax);
    for (float x : maxima) st.max = std::max(st.max, x);
    if (st.max == -kInf) return st;
    const __m512 m = _mm512_set1_ps(st.max);
    __m512 lanes[kLogitLanes / 16];
    for (auto& lane : lanes) lane = _mm512_setzero_ps();
    for (int base = 0; base < V; base += kLogitLanes) {
        const int count = std::min(kLogitLanes, V - base);
        for (int j = 0; j < kLogitLanes / 16 && 16 * j < count; ++j) {
            const int n = count - 16 * j;
            const __mmask16 in = n >= 16 ? static_cast<__mmask16>(0xffff) : static_cast<__mmask16>((1u << n) - 1u);
            const __m512 x = _mm512_maskz_loadu_ps(in, row + base + 16 * j);
            lanes[j] = _mm512_add_ps(lanes[j], exp16(_mm512_sub_ps(x, m), in & finite16(x)));
        }
    }
    alignas(64) float flat[kLogitLanes];
    for (int j = 0; j < kLogitLanes / 16; ++j) _mm512_store_ps(flat + 16 * j, lanes[j]);
    st.lse = logsumexp_from(st.max, combine_lanes(flat));
    return st;
}

DBS_AVX512_TARGET void softmax_gradient_row(const float* row, int V, float lse, float scale, float* out) {
    const __m512 l = _mm512_set1_ps(lse);
    const __m512 sc = _mm512_set1_ps(scale);
    for (int v = 0; v < V; v += 16) {
        const __mmask16 in = V - v >= 16 ? static_cast<__mmask16>(0xffff) : static_cast<__mmask16>((1u << (V - v)) - 1u);
        const __m512 x = _mm512_maskz_loadu_ps(in, row + v);
        const __m512 p = exp16(_mm512_sub_ps(x, l), in & finite16(x));
        _mm512_mask_storeu_ps(out + v, in, _mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(p, sc)));
    }
}

} // namespace avx512
#endif

#if DBS_CAN_COMPILE_AVX2
namespace avx2 {

namespace {

template <bool kHasBanned, bool kHasOffset>
DBS_AVX2_TARGET bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const __m256 offset_vec = _mm256_set1_ps(s.offset);
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
        __m256 lp = _mm256_loadu_ps(s.row + v);
        if (kHasOffset) lp = _mm256_sub_ps(lp, offset_vec);
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
        if (s.has_offset) {
            return s.banned ? scan_range<true, true>(s, begin, end, top, top_count)
                            : scan_range<false, true>(s, begin, end, top, top_count);
        }
        return s.banned ? scan_range<true, false>(s, begin, end, top, top_count)
                        : scan_range<false, false>(s, begin, end, top, top_count);
    });
}

namespace {

DBS_AVX2_TARGET inline __m256 exp8(__m256 x, __m256 valid) {
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(x, _mm256_set1_ps(det::kExpMin), _CMP_GE_OQ));
    const __m256 k = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(det::kLog2e)), _mm256_set1_ps(0.5f)));
    const __m256 r = _mm256_sub_ps(_mm256_sub_ps(x, _mm256_mul_ps(k, _mm256_set1_ps(det::kLn2Hi))),
                                   _mm256_mul_ps(k, _mm256_set1_ps(det::kLn2Lo)));
    const __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p = _mm256_set1_ps(det::kExpP0);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(det::kExpP1));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(det::kExpP2));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(det::kExpP3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(det::kExpP4));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(det::kExpP5));
    p = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(p, r2), r), _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_add_epi32(_mm256_cvttps_epi32(k), _mm256_set1_epi32(127));
    return _mm256_and_ps(valid, _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(e, 23))));
}

DBS_AVX2_TARGET inline __m256 finite8(__m256 x) {
    return _mm256_and_ps(_mm256_cmp_ps(x, _mm256_set1_ps(kInf), _CMP_LT_OQ), _mm256_cmp_ps(x, _mm256_set1_ps(-kInf), _CMP_GT_OQ));
}

// Loads up to 8 floats; missing lanes read -inf (not finite, so they add nothing).
DBS_AVX2_TARGET inline __m256 load8(const float* p, int n) {
    if (n >= 8) return _mm256_loadu_ps(p);
    alignas(32) float tmp[8] = {-kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf};
    std::memcpy(tmp, p, static_cast<size_t>(n) * sizeof(float));
    return _mm256_load_ps(tmp);
}

} // namespace

DBS_AVX2_TARGET LogitStats logit_stats(const float* row, int V) {
    LogitStats st{-kInf, -kInf, false};
    __m256 vmax = _mm256_set1_ps(-kInf);
    int v = 0;
    for (; v + 8 <= V; v += 8) {
        const __m256 x = _mm256_loadu_ps(row + v);
        if (_mm256_movemask_ps(_mm256_cmp_ps(x, _mm256_set1_ps(kInf), _CMP_LT_OQ)) != 0xff) st.invalid = true;
        vmax = _mm256_max_ps(vmax, _mm256_blendv_ps(_mm256_set1_ps(-kInf), x, finite8(x)));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);
    for (float x : tmp) st.max = std::max(st.max, x);
    for (; v < V; ++v) {
        const float x = row[v];
        if (!(x < kInf)) st.invalid = true;
        else if (x > st.max) st.max = x;
    }
    if (st.max == -kInf) return st;
    const __m256 m = _mm256_set1_ps(st.max);
    __m256 lanes[kLogitLanes / 8];
    for (auto& lane : lanes) lane = _mm256_setzero_ps();
    for (int base = 0; base < V; base += kLogitLanes) {
        const int count = std::min(kLogitLanes, V - base);
        for (int j = 0; j < kLogitLanes / 8 && 8 * j < count; ++j) {
            const __m256 x = load8(row + base + 8 * j, count - 8 * j);
            lanes[j] = _mm256_add_ps(lanes[j], exp8(_mm256_sub_ps(x, m), finite8(x)));
        }
    }
    alignas(32) float flat[kLogitLanes];
    for (int j = 0; j < kLogitLanes / 8; ++j) _mm256_store_ps(flat + 8 * j, lanes[j]);
    st.lse = logsumexp_from(st.max, combine_lanes(flat));
    return st;
}

DBS_AVX2_TARGET void softmax_gradient_row(const float* row, int V, float lse, float scale, float* out) {
    const __m256 l = _mm256_set1_ps(lse);
    const __m256 sc = _mm256_set1_ps(scale);
    int v = 0;
    for (; v + 8 <= V; v += 8) {
        const __m256 x = _mm256_loadu_ps(row + v);
        const __m256 p = exp8(_mm256_sub_ps(x, l), finite8(x));
        _mm256_storeu_ps(out + v, _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(p, sc)));
    }
    for (; v < V; ++v) out[v] = det::sub(0.0f, det::mul(softmax_probability(row[v], lse), scale));
}

} // namespace avx2
#endif

#if DBS_CAN_COMPILE_SSE42
namespace sse42 {

namespace {

template <bool kHasBanned, bool kHasOffset>
DBS_SSE42_TARGET bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const __m128 offset_vec = _mm_set1_ps(s.offset);
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
        __m128 lp = _mm_loadu_ps(s.row + v);
        if (kHasOffset) lp = _mm_sub_ps(lp, offset_vec);
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
        if (s.has_offset) {
            return s.banned ? scan_range<true, true>(s, begin, end, top, top_count)
                            : scan_range<false, true>(s, begin, end, top, top_count);
        }
        return s.banned ? scan_range<true, false>(s, begin, end, top, top_count)
                        : scan_range<false, false>(s, begin, end, top, top_count);
    });
}

namespace {

DBS_SSE42_TARGET inline __m128 exp4(__m128 x, __m128 valid) {
    valid = _mm_and_ps(valid, _mm_cmpge_ps(x, _mm_set1_ps(det::kExpMin)));
    const __m128 k = _mm_floor_ps(_mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(det::kLog2e)), _mm_set1_ps(0.5f)));
    const __m128 r = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(k, _mm_set1_ps(det::kLn2Hi))), _mm_mul_ps(k, _mm_set1_ps(det::kLn2Lo)));
    const __m128 r2 = _mm_mul_ps(r, r);
    __m128 p = _mm_set1_ps(det::kExpP0);
    p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(det::kExpP1));
    p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(det::kExpP2));
    p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(det::kExpP3));
    p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(det::kExpP4));
    p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(det::kExpP5));
    p = _mm_add_ps(_mm_add_ps(_mm_mul_ps(p, r2), r), _mm_set1_ps(1.0f));
    const __m128i e = _mm_add_epi32(_mm_cvttps_epi32(k), _mm_set1_epi32(127));
    return _mm_and_ps(valid, _mm_mul_ps(p, _mm_castsi128_ps(_mm_slli_epi32(e, 23))));
}

DBS_SSE42_TARGET inline __m128 finite4(__m128 x) {
    return _mm_and_ps(_mm_cmplt_ps(x, _mm_set1_ps(kInf)), _mm_cmpgt_ps(x, _mm_set1_ps(-kInf)));
}

DBS_SSE42_TARGET inline __m128 load4(const float* p, int n) {
    if (n >= 4) return _mm_loadu_ps(p);
    alignas(16) float tmp[4] = {-kInf, -kInf, -kInf, -kInf};
    std::memcpy(tmp, p, static_cast<size_t>(n) * sizeof(float));
    return _mm_load_ps(tmp);
}

} // namespace

DBS_SSE42_TARGET LogitStats logit_stats(const float* row, int V) {
    LogitStats st{-kInf, -kInf, false};
    __m128 vmax = _mm_set1_ps(-kInf);
    int v = 0;
    for (; v + 4 <= V; v += 4) {
        const __m128 x = _mm_loadu_ps(row + v);
        if (_mm_movemask_ps(_mm_cmplt_ps(x, _mm_set1_ps(kInf))) != 0xf) st.invalid = true;
        vmax = _mm_max_ps(vmax, _mm_blendv_ps(_mm_set1_ps(-kInf), x, finite4(x)));
    }
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, vmax);
    for (float x : tmp) st.max = std::max(st.max, x);
    for (; v < V; ++v) {
        const float x = row[v];
        if (!(x < kInf)) st.invalid = true;
        else if (x > st.max) st.max = x;
    }
    if (st.max == -kInf) return st;
    const __m128 m = _mm_set1_ps(st.max);
    __m128 lanes[kLogitLanes / 4];
    for (auto& lane : lanes) lane = _mm_setzero_ps();
    for (int base = 0; base < V; base += kLogitLanes) {
        const int count = std::min(kLogitLanes, V - base);
        for (int j = 0; j < kLogitLanes / 4 && 4 * j < count; ++j) {
            const __m128 x = load4(row + base + 4 * j, count - 4 * j);
            lanes[j] = _mm_add_ps(lanes[j], exp4(_mm_sub_ps(x, m), finite4(x)));
        }
    }
    alignas(16) float flat[kLogitLanes];
    for (int j = 0; j < kLogitLanes / 4; ++j) _mm_store_ps(flat + 4 * j, lanes[j]);
    st.lse = logsumexp_from(st.max, combine_lanes(flat));
    return st;
}

DBS_SSE42_TARGET void softmax_gradient_row(const float* row, int V, float lse, float scale, float* out) {
    const __m128 l = _mm_set1_ps(lse);
    const __m128 sc = _mm_set1_ps(scale);
    int v = 0;
    for (; v + 4 <= V; v += 4) {
        const __m128 x = _mm_loadu_ps(row + v);
        const __m128 p = exp4(_mm_sub_ps(x, l), finite4(x));
        _mm_storeu_ps(out + v, _mm_sub_ps(_mm_setzero_ps(), _mm_mul_ps(p, sc)));
    }
    for (; v < V; ++v) out[v] = det::sub(0.0f, det::mul(softmax_probability(row[v], lse), scale));
}

} // namespace sse42
#endif

} // namespace dbs
