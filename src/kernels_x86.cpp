// SPDX-License-Identifier: MIT
//
// x86 SIMD kernels. Each function carries its own target attribute, so this file
// is compiled for the baseline ISA and the fast paths are only entered after
// runtime feature detection (see cpu_features.cpp).
#include "kernels.hpp"

namespace dbs {

#if DBS_CAN_COMPILE_AVX512
namespace avx512 {

DBS_AVX512_TARGET static inline float reduce_add512(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

DBS_AVX512_TARGET static inline float reduce_max512(__m512 x) {
    return _mm512_reduce_max_ps(x);
}

DBS_AVX512_TARGET static inline __m512 exp512_ps(__m512 x) {
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2f = _mm512_set1_ps(0.6931471805599453f);

    x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    __m512 fx = _mm512_fmadd_ps(x, log2ef, _mm512_set1_ps(0.5f));
    fx = _mm512_floor_ps(fx);

    x = _mm512_fnmadd_ps(fx, ln2f, x);

    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.0f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.0f));

    __m512i emm0 = _mm512_cvttps_epi32(fx);
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));
    emm0 = _mm512_slli_epi32(emm0, 23);

    return _mm512_mul_ps(y, _mm512_castsi512_ps(emm0));
}

DBS_AVX512_TARGET static inline __m512 sigmoid512_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __mmask16 nonnegative = _mm512_cmp_ps_mask(x, zero, _CMP_GE_OS);
    const __m512 exp_neg_x = exp512_ps(_mm512_sub_ps(zero, x));
    const __m512 pos = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));
    const __m512 min_normal_arg = _mm512_set1_ps(-87.3365447505531f);
    const __m512 min_normal = _mm512_set1_ps(std::numeric_limits<float>::min());
    const __m512 exp_x = _mm512_max_ps(exp512_ps(_mm512_max_ps(x, min_normal_arg)), min_normal);
    const __m512 neg = _mm512_div_ps(exp_x, _mm512_add_ps(one, exp_x));
    return _mm512_mask_blend_ps(nonnegative, neg, pos);
}

DBS_AVX512_TARGET float dot(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 av = _mm512_loadu_ps(a + i);
        const __m512 bv = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(av, bv, acc);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 av = _mm512_maskz_loadu_ps(mask, a + i);
        const __m512 bv = _mm512_maskz_loadu_ps(mask, b + i);
        acc = _mm512_fmadd_ps(av, bv, acc);
    }

    return reduce_add512(acc);
}

DBS_AVX512_TARGET void softmax_selected(
    const float* scores,
    float* out,
    int n,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    const __m512 inv_temp = _mm512_set1_ps(1.0f / temperature);
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 guard = _mm512_set1_ps(NEG_GUARD);

    __m512 vmax = neg_inf;

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 raw = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_mul_ps(raw, inv_temp);
        scaled = _mm512_mask_mov_ps(neg_inf, valid, scaled);
        vmax = _mm512_max_ps(vmax, scaled);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 raw = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_mul_ps(raw, inv_temp);
        scaled = _mm512_mask_mov_ps(neg_inf, valid, scaled);
        vmax = _mm512_max_ps(vmax, scaled);
    }

    const float maxv = reduce_max512(vmax);
    if (!std::isfinite(maxv)) return;

    const __m512 max_vec = _mm512_set1_ps(maxv);
    __m512 sum_vec = _mm512_setzero_ps();

    i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 raw = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_sub_ps(_mm512_mul_ps(raw, inv_temp), max_vec);
        __m512 e = exp512_ps(scaled);
        e = _mm512_maskz_mov_ps(valid, e);
        sum_vec = _mm512_add_ps(sum_vec, e);
        _mm512_storeu_ps(out + i, e);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 raw = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_sub_ps(_mm512_mul_ps(raw, inv_temp), max_vec);
        __m512 e = exp512_ps(scaled);
        e = _mm512_maskz_mov_ps(valid, e);
        sum_vec = _mm512_add_ps(sum_vec, e);
        _mm512_mask_storeu_ps(out + i, lane_mask, e);
    }

    const float sum = reduce_add512(sum_vec);
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        std::fill(out, out + n, 0.0f);
        return;
    }

    const __m512 inv_sum = _mm512_set1_ps(1.0f / sum);
    int j = 0;
    for (; j + 15 < n; j += 16) {
        const __m512 y = _mm512_mul_ps(_mm512_loadu_ps(out + j), inv_sum);
        _mm512_storeu_ps(out + j, y);
    }

    if (j < n) {
        const int rem = n - j;
        const __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 y = _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, out + j), inv_sum);
        _mm512_mask_storeu_ps(out + j, mask, y);
    }
}

DBS_AVX512_TARGET float sum_sigmoid_shifted(
    const float* scores,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    const __m512 theta_vec = _mm512_set1_ps(theta);
    const __m512 inv_temp = _mm512_set1_ps(1.0f / temperature);
    const __m512 guard = _mm512_set1_ps(NEG_GUARD);
    __m512 sum_vec = _mm512_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 s = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        __m512 y = sigmoid512_ps(z);
        y = _mm512_maskz_mov_ps(valid, y);
        sum_vec = _mm512_add_ps(sum_vec, y);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 s = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        __m512 y = sigmoid512_ps(z);
        y = _mm512_maskz_mov_ps(valid, y);
        sum_vec = _mm512_add_ps(sum_vec, y);
    }

    return reduce_add512(sum_vec);
}

DBS_AVX512_TARGET void soft_topk_write(
    const float* scores,
    float* out,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    const __m512 theta_vec = _mm512_set1_ps(theta);
    const __m512 inv_temp = _mm512_set1_ps(1.0f / temperature);
    const __m512 guard = _mm512_set1_ps(NEG_GUARD);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 s = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        const __m512 y = _mm512_maskz_mov_ps(valid, sigmoid512_ps(z));
        _mm512_storeu_ps(out + i, y);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 s = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        const __m512 y = _mm512_maskz_mov_ps(valid, sigmoid512_ps(z));
        _mm512_mask_storeu_ps(out + i, lane_mask, y);
    }
}

DBS_AVX512_TARGET void scan_parent_row(
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
    int min_length
) {
    if (banned_tokens || forced_token >= 0 || (eos_token >= 0 && parent_length + 1 < min_length)) {
        scan_parent_row_scalar(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block, length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    const __m512 parent_vec = _mm512_set1_ps(parent_raw);
    const __m512 scale_vec = _mm512_set1_ps(inv_penalty);
    const __m512 neg_inf_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 pos_inf_vec = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    alignas(64) float rank_tmp[16];
    alignas(64) float raw_tmp[16];

    // Roughly four 64B cache lines ahead. This is far enough for streaming rows
    // without pulling too much cold vocabulary data into L1.
    constexpr int PREFETCH_FLOATS = 256;

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        int v = base;

        for (; v + 15 < end; v += 16) {
            const int pf = v + PREFETCH_FLOATS;
            if (pf < vocab_size) {
                _mm_prefetch(reinterpret_cast<const char*>(row + pf), _MM_HINT_T0);
            }

            const __m512 lp = _mm512_loadu_ps(row + v);
            const __m512 raw_vec = _mm512_add_ps(lp, parent_vec);
            const __m512 rank_vec = _mm512_mul_ps(raw_vec, scale_vec);

            // Lanes tied with the current worst entry must still reach insert_topk:
            // the full comparator can rank them higher (raw score, parent, token).
            const float threshold = top[top_count - 1].score;
            const __mmask16 finite_mask =
                _mm512_cmp_ps_mask(lp, neg_inf_vec, _CMP_GT_OS) &
                _mm512_cmp_ps_mask(lp, pos_inf_vec, _CMP_LT_OS);
            const __mmask16 mask = finite_mask & _mm512_cmp_ps_mask(rank_vec, _mm512_set1_ps(threshold), _CMP_GE_OS);

            if (mask) {
                _mm512_store_ps(rank_tmp, rank_vec);
                _mm512_store_ps(raw_tmp, raw_vec);

                uint32_t m = static_cast<uint32_t>(mask);
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }

        if (v < end) {
            const int rem = end - v;
            const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);

            const __m512 lp = _mm512_maskz_loadu_ps(lane_mask, row + v);
            const __m512 raw_vec = _mm512_add_ps(lp, parent_vec);
            const __m512 rank_vec = _mm512_mul_ps(raw_vec, scale_vec);

            const float threshold = top[top_count - 1].score;
            const __mmask16 finite_mask =
                _mm512_cmp_ps_mask(lp, neg_inf_vec, _CMP_GT_OS) &
                _mm512_cmp_ps_mask(lp, pos_inf_vec, _CMP_LT_OS);
            __mmask16 mask = lane_mask & finite_mask & _mm512_cmp_ps_mask(rank_vec, _mm512_set1_ps(threshold), _CMP_GE_OS);

            if (mask) {
                _mm512_store_ps(rank_tmp, rank_vec);
                _mm512_store_ps(raw_tmp, raw_vec);

                uint32_t m = static_cast<uint32_t>(mask);
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }
    }
}

DBS_AVX512_TARGET void exp16(const float* in, float* out) {
    _mm512_storeu_ps(out, exp512_ps(_mm512_loadu_ps(in)));
}

DBS_AVX512_TARGET void sigmoid16(const float* in, float* out) {
    _mm512_storeu_ps(out, sigmoid512_ps(_mm512_loadu_ps(in)));
}

} // namespace avx512
#endif


#if DBS_CAN_COMPILE_AVX2
namespace avx2 {
DBS_AVX2_TARGET static inline float hsum256(__m256 v) {
    // Reduce 8 lanes to 1 using two hadd + cross-lane extract.
    __m256 t = _mm256_hadd_ps(v, v);       // [a0+a1, a2+a3, a0+a1, a2+a3 | a4+a5, a6+a7, a4+a5, a6+a7]
    t = _mm256_hadd_ps(t, t);              // [a0..3, a0..3, a0..3, a0..3 | a4..7, a4..7, a4..7, a4..7]
    __m128 lo = _mm256_castps256_ps128(t);
    __m128 hi = _mm256_extractf128_ps(t, 1);
    return _mm_cvtss_f32(_mm_add_ps(lo, hi));
}

DBS_AVX2_TARGET float dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float s = hsum256(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

DBS_AVX2_TARGET void softmax_selected(const float* scores, float* out, int n, float temperature) {
    constexpr float NEG_GUARD = -1.0e30f;
    std::fill(out, out + n, 0.0f);
    const __m256 inv_temp = _mm256_set1_ps(1.0f / temperature);
    const __m256 guard = _mm256_set1_ps(NEG_GUARD);
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256 vmax = neg_inf;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 raw = _mm256_loadu_ps(scores + i);
        __m256 valid = _mm256_cmp_ps(raw, guard, _CMP_GT_OS);
        __m256 scaled = _mm256_mul_ps(raw, inv_temp);
        scaled = _mm256_blendv_ps(neg_inf, scaled, valid);
        vmax = _mm256_max_ps(vmax, scaled);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);
    float maxv = -std::numeric_limits<float>::infinity();
    for (float x : tmp) maxv = std::max(maxv, x);
    for (; i < n; ++i) if (scores[i] > NEG_GUARD) maxv = std::max(maxv, scores[i] / temperature);
    if (!std::isfinite(maxv)) return;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (scores[j] > NEG_GUARD) {
            out[j] = safe_exp_scalar(scores[j] / temperature - maxv);
            sum += out[j];
        }
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) { std::fill(out, out + n, 0.0f); return; }
    const float inv_sum = 1.0f / sum;
    i = 0;
    const __m256 inv = _mm256_set1_ps(inv_sum);
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), inv));
    }
    for (; i < n; ++i) out[i] *= inv_sum;
}

DBS_AVX2_TARGET void scan_parent_row(
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
    int min_length
) {
    if (banned_tokens || forced_token >= 0 || (eos_token >= 0 && parent_length + 1 < min_length)) {
        scan_parent_row_scalar(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block, length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    const __m256 parent_vec = _mm256_set1_ps(parent_raw);
    const __m256 scale_vec = _mm256_set1_ps(inv_penalty);
    const __m256 neg_inf_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256 pos_inf_vec = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        int v = base;

        for (; v + 7 < end; v += 8) {
            const __m256 lp = _mm256_loadu_ps(row + v);
            const __m256 raw_vec = _mm256_add_ps(lp, parent_vec);
            const __m256 rank_vec = _mm256_mul_ps(raw_vec, scale_vec);

            const float threshold = top[top_count - 1].score;
            const __m256 finite = _mm256_and_ps(
                _mm256_cmp_ps(lp, neg_inf_vec, _CMP_GT_OS),
                _mm256_cmp_ps(lp, pos_inf_vec, _CMP_LT_OS));
            const __m256 better = _mm256_cmp_ps(rank_vec, _mm256_set1_ps(threshold), _CMP_GE_OS);
            const int mask = _mm256_movemask_ps(_mm256_and_ps(finite, better));

            if (mask) {
                alignas(32) float rank_tmp[8];
                alignas(32) float raw_tmp[8];
                _mm256_store_ps(rank_tmp, rank_vec);
                _mm256_store_ps(raw_tmp, raw_vec);
                int m = mask;
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }

        for (; v < end; ++v) {
            const float lp = row[v];
            if (!std::isfinite(lp)) continue;
            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}
} // namespace avx2
#endif

#if DBS_CAN_COMPILE_SSE42
namespace sse42 {
DBS_SSE42_TARGET static inline float hsum128(__m128 v) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

DBS_SSE42_TARGET float dot(const float* a, const float* b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < n; i += 4) {
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float s = hsum128(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

DBS_SSE42_TARGET void softmax_selected(const float* scores, float* out, int n, float temperature) {
    constexpr float NEG_GUARD = -1.0e30f;
    std::fill(out, out + n, 0.0f);
    const __m128 inv_temp = _mm_set1_ps(1.0f / temperature);
    const __m128 guard = _mm_set1_ps(NEG_GUARD);
    const __m128 neg_inf = _mm_set1_ps(-std::numeric_limits<float>::infinity());
    __m128 vmax = neg_inf;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 raw = _mm_loadu_ps(scores + i);
        __m128 valid = _mm_cmpgt_ps(raw, guard);
        __m128 scaled = _mm_mul_ps(raw, inv_temp);
        scaled = _mm_or_ps(_mm_and_ps(valid, scaled), _mm_andnot_ps(valid, neg_inf));
        vmax = _mm_max_ps(vmax, scaled);
    }
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, vmax);
    float maxv = -std::numeric_limits<float>::infinity();
    for (float x : tmp) maxv = std::max(maxv, x);
    for (; i < n; ++i) if (scores[i] > NEG_GUARD) maxv = std::max(maxv, scores[i] / temperature);
    if (!std::isfinite(maxv)) return;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (scores[j] > NEG_GUARD) { out[j] = safe_exp_scalar(scores[j] / temperature - maxv); sum += out[j]; }
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) { std::fill(out, out + n, 0.0f); return; }
    const float inv_sum = 1.0f / sum;
    i = 0;
    const __m128 inv = _mm_set1_ps(inv_sum);
    for (; i + 3 < n; i += 4) _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), inv));
    for (; i < n; ++i) out[i] *= inv_sum;
}

DBS_SSE42_TARGET void scan_parent_row(
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
    int min_length
) {
    if (banned_tokens || forced_token >= 0 || (eos_token >= 0 && parent_length + 1 < min_length)) {
        scan_parent_row_scalar(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block, length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }

    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);
    const __m128 parent_vec = _mm_set1_ps(parent_raw);
    const __m128 scale_vec = _mm_set1_ps(inv_penalty);
    const __m128 neg_inf_vec = _mm_set1_ps(-std::numeric_limits<float>::infinity());
    const __m128 pos_inf_vec = _mm_set1_ps(std::numeric_limits<float>::infinity());

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        int v = base;

        for (; v + 3 < end; v += 4) {
            const __m128 lp = _mm_loadu_ps(row + v);
            const __m128 raw_vec = _mm_add_ps(lp, parent_vec);
            const __m128 rank_vec = _mm_mul_ps(raw_vec, scale_vec);

            const float threshold = top[top_count - 1].score;
            const __m128 finite = _mm_and_ps(
                _mm_cmpgt_ps(lp, neg_inf_vec),
                _mm_cmplt_ps(lp, pos_inf_vec));
            const __m128 better = _mm_cmpge_ps(rank_vec, _mm_set1_ps(threshold));
            const int mask = _mm_movemask_ps(_mm_and_ps(finite, better));

            if (mask) {
                alignas(16) float rank_tmp[4];
                alignas(16) float raw_tmp[4];
                _mm_store_ps(rank_tmp, rank_vec);
                _mm_store_ps(raw_tmp, raw_vec);
                int m = mask;
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }

        for (; v < end; ++v) {
            const float lp = row[v];
            if (!std::isfinite(lp)) continue;
            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}
} // namespace sse42
#endif

} // namespace dbs
