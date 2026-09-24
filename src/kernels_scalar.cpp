// SPDX-License-Identifier: MIT
//
// Scalar reference kernels, the NEON row scan, and the runtime dispatcher.
#include "kernels.hpp"

#include <cstring>

namespace dbs {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Only one candidate, but every element of the row is still checked.
bool scan_row_forced(const RowScan& s, Candidate* top, int top_count) {
    bool invalid = false;
    for (int v = 0; v < s.vocab_size; ++v) invalid |= !(s.row[v] < kInf);
    const int v = s.forced_token;
    const float lp = s.row[v];
    if (lp < kInf && lp != -kInf && !(s.banned && s.banned[v]) && v != s.masked_token) {
        const float raw = s.parent_raw + lp;
        insert_topk(top, top_count, Candidate{raw * s.inv_penalty, raw, s.parent, v, s.new_length, 1});
    }
    return invalid;
}

float safe_exp(float x) {
    x = std::min(88.3762626647949f, std::max(-88.3762626647949f, x));
    return std::exp(x);
}

float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + safe_exp(-x));
    const float e = safe_exp(x);
    return e / (1.0f + e);
}

float sum_sigmoid_shifted(const float* scores, int n, float theta, float temperature) {
    constexpr float kNegGuard = -1.0e30f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > kNegGuard) sum += sigmoid((scores[i] - theta) / temperature);
    }
    return sum;
}

} // namespace

bool scan_row_scalar(const RowScan& s, Candidate* top, int top_count) {
    if (s.forced_token >= 0) return scan_row_forced(s, top, top_count);
    return scan_around_masked(s, [&](int begin, int end) {
        bool invalid = false;
        for (int v = begin; v < end; ++v) invalid |= scan_token(s, v, top, top_count);
        return invalid;
    });
}

#if DBS_ARM_NEON
namespace neon {

namespace {

bool scan_range(const RowScan& s, int begin, int end, Candidate* top, int top_count) {
    const float32x4_t parent_vec = vdupq_n_f32(s.parent_raw);
    const float32x4_t scale_vec = vdupq_n_f32(s.inv_penalty);
    const float32x4_t pos_inf = vdupq_n_f32(kInf);
    const float32x4_t neg_inf = vdupq_n_f32(-kInf);
    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t below_all = vdupq_n_u32(0xffffffffu);

    int v = begin;
    for (; v + 4 <= end; v += 4) {
        const float32x4_t lp = vld1q_f32(s.row + v);
        const uint32x4_t below_inf = vcltq_f32(lp, pos_inf);  // false for NaN and +inf
        below_all = vandq_u32(below_all, below_inf);
        uint32x4_t ok = vandq_u32(below_inf, vmvnq_u32(vceqq_f32(lp, neg_inf)));
        if (s.banned) {
            uint32_t bytes = 0;
            std::memcpy(&bytes, s.banned + v, 4);
            const uint16x8_t wide = vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(bytes)));
            ok = vandq_u32(ok, vceqq_u32(vmovl_u16(vget_low_u16(wide)), zero));
        }
        const float32x4_t raw = vaddq_f32(lp, parent_vec);
        const float32x4_t rank = vmulq_f32(raw, scale_vec);
        const uint32x4_t hit = vandq_u32(ok, vcgeq_f32(rank, vdupq_n_f32(top[top_count - 1].score)));
        const uint32x2_t any = vorr_u32(vget_low_u32(hit), vget_high_u32(hit));
        if ((vget_lane_u32(any, 0) | vget_lane_u32(any, 1)) == 0) continue;

        float rank_tmp[4];
        float raw_tmp[4];
        uint32_t hit_tmp[4];
        vst1q_f32(rank_tmp, rank);
        vst1q_f32(raw_tmp, raw);
        vst1q_u32(hit_tmp, hit);
        for (int lane = 0; lane < 4; ++lane) {
            if (hit_tmp[lane] == 0) continue;
            insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], s.parent, v + lane, s.new_length, 1});
        }
    }
    const uint32x2_t all = vand_u32(vget_low_u32(below_all), vget_high_u32(below_all));
    bool invalid = (vget_lane_u32(all, 0) & vget_lane_u32(all, 1)) == 0;
    for (; v < end; ++v) invalid |= scan_token(s, v, top, top_count);
    return invalid;
}

} // namespace

bool scan_row(const RowScan& s, Candidate* top, int top_count) {
    if (s.forced_token >= 0) return scan_row_forced(s, top, top_count);
    return scan_around_masked(s, [&](int begin, int end) { return scan_range(s, begin, end, top, top_count); });
}

} // namespace neon
#endif

bool scan_row(const RowScan& s, Candidate* top, int top_count) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::scan_row(s, top, top_count);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) return avx2::scan_row(s, top, top_count);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) return sse42::scan_row(s, top, top_count);
#endif
#if DBS_ARM_NEON
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) return neon::scan_row(s, top, top_count);
#endif
    return scan_row_scalar(s, top, top_count);
}

float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void softmax_selected(const float* scores, float* out, int n, float temperature) {
    constexpr float kNegGuard = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    float maxv = -kInf;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > kNegGuard) maxv = std::max(maxv, scores[i] / temperature);
    }
    if (!std::isfinite(maxv)) return;

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > kNegGuard) {
            out[i] = safe_exp(scores[i] / temperature - maxv);
            sum += out[i];
        }
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        std::fill(out, out + n, 0.0f);
        return;
    }

    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) out[i] *= inv_sum;
}

void soft_topk_inclusion(
    const float* scores,
    float* out,
    int n,
    int target_k,
    float temperature,
    float tolerance,
    int max_iters
) {
    constexpr float kNegGuard = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    int active = 0;
    float min_s = kInf;
    float max_s = -kInf;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > kNegGuard) {
            ++active;
            min_s = std::min(min_s, scores[i]);
            max_s = std::max(max_s, scores[i]);
        }
    }
    if (active == 0 || target_k <= 0) return;

    if (target_k >= active) {
        for (int i = 0; i < n; ++i) {
            if (scores[i] > kNegGuard) out[i] = 1.0f;
        }
        return;
    }

    float lo = min_s - 80.0f * temperature;
    float hi = max_s + 80.0f * temperature;
    for (int it = 0; it < max_iters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float err = sum_sigmoid_shifted(scores, n, mid, temperature) - static_cast<float>(target_k);
        if (std::fabs(err) <= tolerance || std::fabs(hi - lo) <= tolerance * std::max(1.0f, std::fabs(mid))) {
            lo = hi = mid;
            break;
        }
        if (err > 0.0f) lo = mid;
        else hi = mid;
    }

    const float theta = 0.5f * (lo + hi);
    for (int i = 0; i < n; ++i) {
        out[i] = scores[i] > kNegGuard ? sigmoid((scores[i] - theta) / temperature) : 0.0f;
    }
}

} // namespace dbs
