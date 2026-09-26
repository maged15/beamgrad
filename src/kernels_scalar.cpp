// SPDX-License-Identifier: MIT
//
// Scalar reference kernels, the NEON row scan, and the runtime dispatcher.
#include "kernels.hpp"

#include <cstring>
#include <limits>

namespace dbs {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Only one candidate, but every element of the row is still checked.
bool scan_row_forced(const RowScan& s, Candidate* top, int top_count) {
    bool invalid = false;
    for (int v = 0; v < s.vocab_size; ++v) invalid |= !(row_log_prob(s, v) < kInf);
    const int v = s.forced_token;
    const float lp = row_log_prob(s, v);
    if (lp < kInf && lp != -kInf && !(s.banned && s.banned[v]) && !is_masked(s, v)) {
        const float raw = s.parent_raw + lp;
        insert_topk(top, top_count, Candidate{raw * s.inv_penalty, raw, s.parent, v, s.new_length, 1});
    }
    return invalid;
}

// The deterministic exp of logits.hpp, so the C-level surrogates below give
// the same bits on every platform (the arguments are never positive).
float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + det::exp_nonpositive(-x));
    const float e = det::exp_nonpositive(x);
    return e / (1.0f + e);
}

// sum_i sigmoid(((scores[i] - ref) - offset) / temperature), i.e. at theta = ref + offset.
float sum_sigmoid_offset(const float* scores, int n, float ref, float offset, float temperature) {
    constexpr float kNegGuard = -1.0e30f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > kNegGuard) sum += sigmoid(((scores[i] - ref) - offset) / temperature);
    }
    return sum;
}

} // namespace

LogitStats logit_stats_scalar(const float* row, int V) {
    LogitStats st{-kInf, -kInf, false};
    for (int v = 0; v < V; ++v) {
        const float x = row[v];
        if (!(x < kInf)) st.invalid = true;
        else if (x > st.max) st.max = x;
    }
    if (st.max == -kInf) return st;
    float lanes[kLogitLanes] = {};
    for (int v = 0; v < V; ++v) {
        const float x = row[v];
        if (x < kInf && x != -kInf) {
            float& lane = lanes[v & (kLogitLanes - 1)];
            lane = det::add(lane, det::exp_nonpositive(det::sub(x, st.max)));
        }
    }
    st.lse = logsumexp_from(st.max, combine_lanes(lanes));
    return st;
}

void softmax_gradient_row_scalar(const float* row, int V, float lse, float scale, float* out) {
    for (int v = 0; v < V; ++v) out[v] = det::sub(0.0f, det::mul(softmax_probability(row[v], lse), scale));
}

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
    const float32x4_t offset_vec = vdupq_n_f32(s.offset);
    const float32x4_t parent_vec = vdupq_n_f32(s.parent_raw);
    const float32x4_t scale_vec = vdupq_n_f32(s.inv_penalty);
    const float32x4_t pos_inf = vdupq_n_f32(kInf);
    const float32x4_t neg_inf = vdupq_n_f32(-kInf);
    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t below_all = vdupq_n_u32(0xffffffffu);

    int v = begin;
    for (; v + 4 <= end; v += 4) {
        float32x4_t lp = vld1q_f32(s.row + v);
        if (s.has_offset) lp = vsubq_f32(lp, offset_vec);
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

#if defined(__aarch64__)
namespace {

// exp_nonpositive on four lanes, in the scalar operation order; lanes where
// `valid` is false give 0.
inline float32x4_t exp4(float32x4_t x, uint32x4_t valid) {
    valid = vandq_u32(valid, vcgeq_f32(x, vdupq_n_f32(det::kExpMin)));
    const float32x4_t k = vrndmq_f32(vaddq_f32(vmulq_f32(x, vdupq_n_f32(det::kLog2e)), vdupq_n_f32(0.5f)));
    const float32x4_t r = vsubq_f32(vsubq_f32(x, vmulq_f32(k, vdupq_n_f32(det::kLn2Hi))), vmulq_f32(k, vdupq_n_f32(det::kLn2Lo)));
    const float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p = vdupq_n_f32(det::kExpP0);
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(det::kExpP1));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(det::kExpP2));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(det::kExpP3));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(det::kExpP4));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(det::kExpP5));
    p = vaddq_f32(vaddq_f32(vmulq_f32(p, r2), r), vdupq_n_f32(1.0f));
    const int32x4_t e = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127));
    const float32x4_t scale = vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vmulq_f32(p, scale)), valid));
}

inline uint32x4_t finite4(float32x4_t x) {
    return vandq_u32(vcltq_f32(x, vdupq_n_f32(kInf)), vcgtq_f32(x, vdupq_n_f32(-kInf)));
}

} // namespace

LogitStats logit_stats(const float* row, int V) {
    LogitStats st{-kInf, -kInf, false};
    float32x4_t vmax = vdupq_n_f32(-kInf);
    uint32x4_t below_all = vdupq_n_u32(0xffffffffu);
    int v = 0;
    for (; v + 4 <= V; v += 4) {
        const float32x4_t x = vld1q_f32(row + v);
        below_all = vandq_u32(below_all, vcltq_f32(x, vdupq_n_f32(kInf)));
        vmax = vmaxq_f32(vmax, vbslq_f32(finite4(x), x, vdupq_n_f32(-kInf)));
    }
    st.max = vmaxvq_f32(vmax);
    st.invalid = vminvq_u32(below_all) == 0;
    for (; v < V; ++v) {
        const float x = row[v];
        if (!(x < kInf)) st.invalid = true;
        else if (x > st.max) st.max = x;
    }
    if (st.max == -kInf) return st;
    float32x4_t lanes[kLogitLanes / 4];
    for (auto& lane : lanes) lane = vdupq_n_f32(0.0f);
    const float32x4_t m = vdupq_n_f32(st.max);
    for (int base = 0; base < V; base += kLogitLanes) {
        const int count = std::min(kLogitLanes, V - base);
        for (int j = 0; j < kLogitLanes / 4 && 4 * j < count; ++j) {
            float32x4_t x;
            if (4 * j + 4 <= count) {
                x = vld1q_f32(row + base + 4 * j);
            } else {
                float tmp[4] = {-kInf, -kInf, -kInf, -kInf};
                std::memcpy(tmp, row + base + 4 * j, static_cast<size_t>(count - 4 * j) * sizeof(float));
                x = vld1q_f32(tmp);
            }
            lanes[j] = vaddq_f32(lanes[j], exp4(vsubq_f32(x, m), finite4(x)));
        }
    }
    float flat[kLogitLanes];
    for (int j = 0; j < kLogitLanes / 4; ++j) vst1q_f32(flat + 4 * j, lanes[j]);
    st.lse = logsumexp_from(st.max, combine_lanes(flat));
    return st;
}

void softmax_gradient_row(const float* row, int V, float lse, float scale, float* out) {
    const float32x4_t l = vdupq_n_f32(lse);
    const float32x4_t sc = vdupq_n_f32(scale);
    int v = 0;
    for (; v + 4 <= V; v += 4) {
        const float32x4_t x = vld1q_f32(row + v);
        const float32x4_t p = exp4(vsubq_f32(x, l), finite4(x));
        vst1q_f32(out + v, vsubq_f32(vdupq_n_f32(0.0f), vmulq_f32(p, sc)));
    }
    for (; v < V; ++v) out[v] = det::sub(0.0f, det::mul(softmax_probability(row[v], lse), scale));
}
#endif

} // namespace neon
#endif

LogitStats logit_stats(const float* row, int V) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::logit_stats(row, V);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) return avx2::logit_stats(row, V);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) return sse42::logit_stats(row, V);
#endif
#if DBS_ARM_NEON && defined(__aarch64__)
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) return neon::logit_stats(row, V);
#endif
    return logit_stats_scalar(row, V);
}

void softmax_gradient_row(const float* row, int V, float lse, float scale, float* out) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::softmax_gradient_row(row, V, lse, scale, out);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) return avx2::softmax_gradient_row(row, V, lse, scale, out);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) return sse42::softmax_gradient_row(row, V, lse, scale, out);
#endif
#if DBS_ARM_NEON && defined(__aarch64__)
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) return neon::softmax_gradient_row(row, V, lse, scale, out);
#endif
    softmax_gradient_row_scalar(row, V, lse, scale, out);
}

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
            out[i] = det::exp_nonpositive(scores[i] / temperature - maxv);
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

    // Bisect theta's offset from the best score, not theta itself: a long
    // search's scores are in the thousands, where float32 resolves theta only
    // to about 1e-3, too coarse for weights that move by up to 1 / (4 *
    // temperature) per unit of theta. The offset keeps full precision.
    const float ref = max_s;
    float lo = (min_s - ref) - 80.0f * temperature;
    float hi = 80.0f * temperature;
    // Stop once the weights sum to target_k within tolerance, or theta is
    // bracketed to tolerance * temperature (each weight is then within
    // tolerance / 8 of its value at the root), or to a few float ULPs of the
    // offset, where halving no longer moves it. From this bracket that takes
    // at most log2((max_s - min_s) / (tolerance * temperature) + 160 /
    // tolerance) iterations: 21 with the defaults and a pool a few units wide.
    const float bracket = tolerance * temperature;
    for (int it = 0; it < max_iters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float err = sum_sigmoid_offset(scores, n, ref, mid, temperature) - static_cast<float>(target_k);
        const float ulps = 4.0f * std::numeric_limits<float>::epsilon() * std::fabs(mid);
        if (std::fabs(err) <= tolerance || hi - lo <= std::max(bracket, ulps)) {
            lo = hi = mid;
            break;
        }
        if (err > 0.0f) lo = mid;
        else hi = mid;
    }

    const float offset = 0.5f * (lo + hi);
    for (int i = 0; i < n; ++i) {
        out[i] = scores[i] > kNegGuard ? sigmoid(((scores[i] - ref) - offset) / temperature) : 0.0f;
    }
}

} // namespace dbs
