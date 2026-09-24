// SPDX-License-Identifier: MIT
//
// Scalar reference kernels, the NEON helpers, and the runtime dispatchers.
#include "kernels.hpp"

namespace dbs {

float dot_scalar(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void softmax_selected_scalar(
    const float* scores,
    float* out,
    int n,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    float maxv = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) maxv = std::max(maxv, scores[i] / temperature);
    }

    if (!std::isfinite(maxv)) return;

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) {
            out[i] = safe_exp_scalar(scores[i] / temperature - maxv);
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

float sum_sigmoid_shifted_scalar(
    const float* scores,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) {
            sum += sigmoid_scalar((scores[i] - theta) / temperature);
        }
    }
    return sum;
}

void soft_topk_write_scalar(
    const float* scores,
    float* out,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) out[i] = sigmoid_scalar((scores[i] - theta) / temperature);
        else out[i] = 0.0f;
    }
}

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
    int min_length
) {
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        for (int v = base; v < end; ++v) {
            if (forced_token >= 0 && v != forced_token) continue;
            if (banned_tokens && banned_tokens[v]) continue;
            if (eos_token >= 0 && v == eos_token && new_len < min_length) continue;

            const float lp = row[v];
            if (!std::isfinite(lp)) continue;

            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}

#if DBS_ARM_NEON
namespace neon {
namespace {

float dot(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float tmp[4];
    vst1q_f32(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

void softmax_selected(const float* scores, float* out, int n, float temperature) {
    // Scalar exp is deliberate here: it keeps results bit-identical across platforms.
    softmax_selected_scalar(scores, out, n, temperature);
}

} // namespace
} // namespace neon
#endif

float dot(const float* a, const float* b, int n) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::dot(a, b, n);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) return avx2::dot(a, b, n);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) return sse42::dot(a, b, n);
#endif
#if DBS_ARM_NEON
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) return neon::dot(a, b, n);
#endif
    return dot_scalar(a, b, n);
}

void softmax_selected(
    const float* scores,
    float* out,
    int n,
    float temperature
) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) {
        avx512::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) {
        avx2::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) {
        sse42::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
#if DBS_ARM_NEON
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) {
        neon::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
    softmax_selected_scalar(scores, out, n, temperature);
}

float sum_sigmoid_shifted(
    const float* scores,
    int n,
    float theta,
    float temperature
) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::sum_sigmoid_shifted(scores, n, theta, temperature);
#endif
    return sum_sigmoid_shifted_scalar(scores, n, theta, temperature);
}

void soft_topk_write(
    const float* scores,
    float* out,
    int n,
    float theta,
    float temperature
) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) {
        avx512::soft_topk_write(scores, out, n, theta, temperature);
        return;
    }
#endif
    soft_topk_write_scalar(scores, out, n, theta, temperature);
}

void scan_parent_row(
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
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) {
        avx512::scan_parent_row(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block,
                                length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) {
        avx2::scan_parent_row(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block,
                              length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) {
        sse42::scan_parent_row(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block,
                               length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
#endif
    scan_parent_row_scalar(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block,
                           length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
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
    constexpr float NEG_GUARD = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    int active = 0;
    float min_s = std::numeric_limits<float>::infinity();
    float max_s = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) {
            ++active;
            min_s = std::min(min_s, scores[i]);
            max_s = std::max(max_s, scores[i]);
        }
    }

    if (active == 0 || target_k <= 0) return;

    if (target_k >= active) {
        for (int i = 0; i < n; ++i) {
            if (scores[i] > NEG_GUARD) out[i] = 1.0f;
        }
        return;
    }

    float lo = min_s - 80.0f * temperature;
    float hi = max_s + 80.0f * temperature;

    for (int it = 0; it < max_iters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float s = sum_sigmoid_shifted(scores, n, mid, temperature);
        const float err = s - static_cast<float>(target_k);

        if (std::fabs(err) <= tolerance || std::fabs(hi - lo) <= tolerance * std::max(1.0f, std::fabs(mid))) {
            lo = hi = mid;
            break;
        }

        if (err > 0.0f) lo = mid;
        else hi = mid;
    }

    const float theta = 0.5f * (lo + hi);
    soft_topk_write(scores, out, n, theta, temperature);
}

} // namespace dbs
