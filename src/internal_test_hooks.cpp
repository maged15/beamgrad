// SPDX-License-Identifier: MIT
//
// Scalar-vs-SIMD parity checks exposed to tests/simd_internal_tests.cpp.
// These run every compiled SIMD path the host supports against the scalar
// reference on adversarial decode/backward cases (ties, EOS, -inf, constraints).
#include "internal_test_hooks.hpp"

#include "decoder.hpp"

#include <cstdio>
#include <exception>

namespace dbs {

namespace internal_test {

namespace {

static void reset_report(ParityReport* report) {
    if (!report) return;
    report->cases_run = 0;
    report->simd_paths_run = 0;
    report->failures = 0;
    report->message[0] = '\0';
}

static void report_failure(ParityReport* report, const char* detail) {
    if (!report) return;
    ++report->failures;
    if (report->message[0] == '\0') {
        std::snprintf(report->message, sizeof(report->message), "%s", detail);
    }
}

static bool close_enough(float a, float b, float atol, float rtol) {
    if (a == b) return true;
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float scale = std::max(1.0f, std::fabs(b));
    return std::fabs(a - b) <= atol + rtol * scale;
}

template <class ActualVec, class ExpectedVec>
static bool compare_float_vectors(
    const ActualVec& actual,
    const ExpectedVec& expected,
    float atol,
    float rtol,
    const char* label,
    ParityReport* report
) {
    if (actual.size() != expected.size()) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s size mismatch: %zu != %zu", label, actual.size(), expected.size());
        report_failure(report, msg);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i], atol, rtol)) {
            char msg[256];
            std::snprintf(
                msg,
                sizeof(msg),
                "%s[%zu] mismatch: actual=%g expected=%g",
                label,
                i,
                static_cast<double>(actual[i]),
                static_cast<double>(expected[i]));
            report_failure(report, msg);
            return false;
        }
    }
    return true;
}

template <class ActualVec, class ExpectedVec>
static bool compare_int_vectors(
    const ActualVec& actual,
    const ExpectedVec& expected,
    const char* label,
    ParityReport* report
) {
    if (actual.size() != expected.size()) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s size mismatch: %zu != %zu", label, actual.size(), expected.size());
        report_failure(report, msg);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            char msg[256];
            std::snprintf(
                msg,
                sizeof(msg),
                "%s[%zu] mismatch: actual=%lld expected=%lld",
                label,
                i,
                static_cast<long long>(actual[i]),
                static_cast<long long>(expected[i]));
            report_failure(report, msg);
            return false;
        }
    }
    return true;
}

struct KernelOverrideScope {
    KernelOverride previous;

    explicit KernelOverrideScope(KernelPath path)
        : previous(current_kernel_override()) {
        set_kernel_override(KernelOverride{true, path});
    }

    ~KernelOverrideScope() {
        set_kernel_override(previous);
    }

    KernelOverrideScope(const KernelOverrideScope&) = delete;
    KernelOverrideScope& operator=(const KernelOverrideScope&) = delete;
};

static size_t lp_index(int t, int k, int v, int K, int V) {
    return (static_cast<size_t>(t) * static_cast<size_t>(K) + static_cast<size_t>(k)) * static_cast<size_t>(V) + static_cast<size_t>(v);
}

struct InternalParityCase {
    const char* name;
    BeamOptions options;
    int steps;
    int vocab_size;
    std::vector<float> log_probs;
    std::vector<uint8_t> banned_tokens;
    std::vector<int32_t> forced_tokens;
    int constraint_min_length;
    bool expect_invalid;
};

static BeamOptions base_options(int beam_size) {
    BeamOptions opt;
    opt.beam_size = beam_size;
    opt.selected_temperature = 0.85f;
    opt.soft_topk_temperature = 0.4f;
    opt.relaxed_pool_multiplier = 3;
    opt.vocab_block = 3;
    opt.soft_topk_tolerance = 1.0e-5f;
    opt.soft_topk_max_iters = 72;
    opt.validate_inputs = 1;
    return opt;
}

static std::vector<float> filled_log_probs(int steps, int beam_size, int vocab_size, float value) {
    return std::vector<float>(
        static_cast<size_t>(steps) * static_cast<size_t>(beam_size) * static_cast<size_t>(vocab_size),
        value);
}

static std::vector<InternalParityCase> build_internal_parity_cases() {
    std::vector<InternalParityCase> cases;

    {
        const int T = 3, K = 3, V = 6;
        BeamOptions opt = base_options(K);
        std::vector<float> x = filled_log_probs(T, K, V, -8.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 0, K, V)] = -0.25f;
                x[lp_index(t, k, 1, K, V)] = -0.25f;
                x[lp_index(t, k, 2, K, V)] = -0.5f;
            }
        }
        x[lp_index(1, 1, 3, K, V)] = -0.25f;
        x[lp_index(2, 2, 4, K, V)] = -0.25f;
        cases.push_back(InternalParityCase{"ties", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 4, K = 3, V = 5;
        BeamOptions opt = base_options(K);
        opt.eos_token = 2;
        std::vector<float> x = filled_log_probs(T, K, V, -7.0f);
        x[lp_index(0, 0, 2, K, V)] = 0.0f;
        x[lp_index(0, 0, 3, K, V)] = -0.1f;
        x[lp_index(0, 0, 1, K, V)] = -0.2f;
        for (int t = 1; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 4, K, V)] = -0.05f;
                x[lp_index(t, k, 0, K, V)] = -0.2f;
            }
        }
        cases.push_back(InternalParityCase{"eos_carry_forward", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 3, K = 4, V = 7;
        BeamOptions opt = base_options(K);
        std::vector<float> x = filled_log_probs(T, K, V, -std::numeric_limits<float>::infinity());
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, (k + t) % V, K, V)] = -0.1f * static_cast<float>(k + 1);
                x[lp_index(t, k, (k + t + 2) % V, K, V)] = -0.4f;
            }
        }
        cases.push_back(InternalParityCase{"negative_infinity", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 2, K = 3, V = 8;
        BeamOptions opt = base_options(K);
        opt.validate_inputs = 0;
        std::vector<float> x = filled_log_probs(T, K, V, -4.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 1, K, V)] = -0.05f;
                x[lp_index(t, k, 3, K, V)] = -0.10f;
                x[lp_index(t, k, 6, K, V)] = -0.25f;
            }
        }
        x[lp_index(0, 0, 2, K, V)] = std::numeric_limits<float>::infinity();
        x[lp_index(1, 1, 5, K, V)] = std::numeric_limits<float>::infinity();
        cases.push_back(InternalParityCase{"positive_infinity_validation_disabled", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 3, K = 3, V = 7;
        BeamOptions opt = base_options(K);
        opt.eos_token = 5;
        opt.min_length = 2;
        opt.length_penalty_alpha = 0.35f;
        std::vector<float> x = filled_log_probs(T, K, V, -6.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 0, K, V)] = -0.05f;
                x[lp_index(t, k, 3, K, V)] = -0.04f;
                x[lp_index(t, k, 5, K, V)] = -0.01f;
            }
        }
        std::vector<uint8_t> banned(static_cast<size_t>(V), 0);
        banned[0] = 1;
        banned[4] = 1;
        std::vector<int32_t> forced = {-1, 3, -1};
        cases.push_back(InternalParityCase{"constraints_min_length_forced_banned", opt, T, V, x, banned, forced, 2, false});
    }

    {
        const int T = 5, K = 4, V = 8;
        BeamOptions opt = base_options(K);
        opt.eos_token = 6;
        opt.length_penalty_alpha = 0.75f;
        std::vector<float> x = filled_log_probs(T, K, V, -5.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, (t + k) % V, K, V)] = -0.08f;
                x[lp_index(t, k, 6, K, V)] = (t < 2) ? -0.6f : -0.04f;
                x[lp_index(t, k, 7, K, V)] = -0.12f;
            }
        }
        cases.push_back(InternalParityCase{"length_penalty", opt, T, V, x, {}, {}, -1, false});
    }

    {
        // Many exact ties at the pool boundary (pool == beam), across several
        // SIMD vectors per row, with a length penalty applied to the ranks.
        const int T = 4, K = 3, V = 53;
        BeamOptions opt = base_options(K);
        opt.relaxed_pool_multiplier = 1;
        opt.length_penalty_alpha = 0.9f;
        opt.eos_token = 7;
        std::vector<float> x = filled_log_probs(T, K, V, 0.0f);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = -0.25f * static_cast<float>((i * 7919u) % 4u);
        }
        cases.push_back(InternalParityCase{"threshold_ties_length_penalty", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 2, K = 2, V = 4;
        BeamOptions opt = base_options(K);
        std::vector<float> x = filled_log_probs(T, K, V, -1.0f);
        x[lp_index(1, 1, 2, K, V)] = std::numeric_limits<float>::quiet_NaN();
        cases.push_back(InternalParityCase{"nan_rejection", opt, T, V, x, {}, {}, -1, true});
    }

    return cases;
}

struct DecodeBackwardBundle {
    DecodeResult forward;
    BackwardResult dense_backward;
    BackwardResult sparse_backward;
};

static DecodeBackwardBundle run_decode_backward_case(const InternalParityCase& tc, KernelPath path) {
    KernelOverrideScope override(path);
    BeamSearchDecoder decoder(tc.options);

    DecodeConstraints constraints;
    DecodeConstraints* constraint_ptr = nullptr;
    if (!tc.banned_tokens.empty() || !tc.forced_tokens.empty() || tc.constraint_min_length >= 0) {
        constraints.banned_tokens = tc.banned_tokens.empty() ? nullptr : tc.banned_tokens.data();
        constraints.forced_tokens = tc.forced_tokens.empty() ? nullptr : tc.forced_tokens.data();
        constraints.min_length = tc.constraint_min_length;
        constraint_ptr = &constraints;
    }

    DecodeBackwardBundle bundle;
    bundle.forward = decoder.decode_constrained(tc.log_probs.data(), tc.steps, tc.vocab_size, constraint_ptr);

    const size_t selected_count = static_cast<size_t>(bundle.forward.steps) * static_cast<size_t>(bundle.forward.beam_size);
    const size_t pool_count = static_cast<size_t>(bundle.forward.steps) * static_cast<size_t>(bundle.forward.relaxed_pool_size);

    std::vector<float> grad_selected(selected_count, 0.0f);
    std::vector<float> grad_relaxed(pool_count, 0.0f);
    std::vector<float> grad_final(static_cast<size_t>(bundle.forward.beam_size), 0.0f);
    for (size_t i = 0; i < grad_selected.size(); ++i) {
        grad_selected[i] = static_cast<float>((static_cast<int>(i % 7) - 3)) * 0.125f;
    }
    for (size_t i = 0; i < grad_relaxed.size(); ++i) {
        grad_relaxed[i] = static_cast<float>((static_cast<int>(i % 11) - 5)) * 0.03125f;
    }
    for (size_t i = 0; i < grad_final.size(); ++i) {
        grad_final[i] = static_cast<float>(static_cast<int>(i) + 1) * 0.2f;
    }

    bundle.dense_backward = decoder.backward(bundle.forward, grad_selected.data(), grad_relaxed.data(), grad_final.data());
    bundle.sparse_backward = decoder.backward_sparse(bundle.forward, grad_selected.data(), grad_relaxed.data(), grad_final.data());
    return bundle;
}

static bool compare_decode_backward_bundles(
    const DecodeBackwardBundle& actual,
    const DecodeBackwardBundle& expected,
    const char* case_name,
    const char* path_name,
    ParityReport* report
) {
    char label[128];

    std::snprintf(label, sizeof(label), "%s/%s parents", path_name, case_name);
    if (!compare_int_vectors(actual.forward.parents, expected.forward.parents, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s tokens", path_name, case_name);
    if (!compare_int_vectors(actual.forward.tokens, expected.forward.tokens, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s lengths", path_name, case_name);
    if (!compare_int_vectors(actual.forward.lengths, expected.forward.lengths, label, report)) return false;

    std::snprintf(label, sizeof(label), "%s/%s final_scores", path_name, case_name);
    if (!compare_float_vectors(actual.forward.final_scores, expected.forward.final_scores, 2.0e-4f, 2.0e-4f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s weights", path_name, case_name);
    if (!compare_float_vectors(actual.forward.weights, expected.forward.weights, 3.0e-3f, 3.0e-3f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s relaxed_weights", path_name, case_name);
    if (!compare_float_vectors(actual.forward.relaxed_weights, expected.forward.relaxed_weights, 5.0e-3f, 5.0e-3f, label, report)) return false;

    std::snprintf(label, sizeof(label), "%s/%s dense_grad_log_probs", path_name, case_name);
    if (!compare_float_vectors(actual.dense_backward.grad_log_probs, expected.dense_backward.grad_log_probs, 8.0e-3f, 8.0e-3f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s dense_grad_initial", path_name, case_name);
    if (!compare_float_vectors(actual.dense_backward.grad_initial_scores, expected.dense_backward.grad_initial_scores, 8.0e-3f, 8.0e-3f, label, report)) return false;

    std::snprintf(label, sizeof(label), "%s/%s sparse_indices", path_name, case_name);
    if (!compare_int_vectors(actual.sparse_backward.sparse_logprob_indices, expected.sparse_backward.sparse_logprob_indices, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s sparse_values", path_name, case_name);
    if (!compare_float_vectors(actual.sparse_backward.sparse_logprob_values, expected.sparse_backward.sparse_logprob_values, 8.0e-3f, 8.0e-3f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s sparse_grad_initial", path_name, case_name);
    return compare_float_vectors(actual.sparse_backward.grad_initial_scores, expected.sparse_backward.grad_initial_scores, 8.0e-3f, 8.0e-3f, label, report);
}

static std::vector<KernelPath> available_simd_paths() {
    std::vector<KernelPath> paths;
#if DBS_CAN_COMPILE_AVX512
    if (runtime_has_avx512()) paths.push_back(KernelPath::AVX512);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (runtime_has_avx2()) paths.push_back(KernelPath::AVX2);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (runtime_has_sse42()) paths.push_back(KernelPath::SSE42);
#endif
#if DBS_ARM_NEON
    if (runtime_has_neon()) paths.push_back(KernelPath::NEON);
#endif
    return paths;
}

#if DBS_CAN_COMPILE_AVX512
static int run_avx512_vector_math_parity_impl(ParityReport* report) {
    {
        alignas(64) float input[16] = {
            -12.0f, -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.125f, 0.0f,
             0.125f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
        };
        alignas(64) float output[16] = {};
        avx512::exp16(input, output);
        for (int i = 0; i < 16; ++i) {
            if (!close_enough(output[i], safe_exp_scalar(input[i]), 2.5e-4f, 2.5e-4f)) {
                report_failure(report, "avx512 exp512_ps diverged from scalar exp");
                return 1;
            }
        }
        if (report) ++report->cases_run;
    }

    {
        alignas(64) float input[16] = {
            -88.0f, -80.0f, -60.0f, -45.0f, -20.0f, -12.0f, -8.0f, -4.0f,
             -1.0f, 0.0f, 1.0f, 4.0f, 8.0f, 12.0f, 20.0f, 45.0f,
        };
        alignas(64) float output[16] = {};
        avx512::sigmoid16(input, output);
        for (int i = 0; i < 16; ++i) {
            const float expected = sigmoid_scalar(input[i]);
            const bool tiny = expected > 0.0f && expected < 1.0e-6f;
            const bool denormal_floor = input[i] < -87.0f;
            const bool ok = denormal_floor
                ? (output[i] >= std::numeric_limits<float>::min())
                : tiny
                ? (output[i] > 0.0f && std::fabs(output[i] - expected) <= std::fabs(expected) * 2.5e-1f)
                : close_enough(output[i], expected, 2.5e-4f, 2.5e-4f);
            if (!ok) {
                report_failure(report, "avx512 sigmoid512_ps diverged from scalar sigmoid");
                return 1;
            }
        }
        if (report) ++report->cases_run;
    }

    {
        std::vector<float> scores = {
            -4.0f, -1.5f, -0.25f, 0.0f, 0.125f, 0.5f, 1.0f, 2.0f,
            -1.0e31f, -3.0f, 3.5f, -2.25f, 0.75f, -0.75f, 1.5f, -5.0f,
            2.75f, -1.25f, 0.25f,
        };
        std::vector<float> scalar(scores.size(), 0.0f);
        std::vector<float> simd(scores.size(), 0.0f);
        softmax_selected_scalar(scores.data(), scalar.data(), static_cast<int>(scores.size()), 0.7f);
        avx512::softmax_selected(scores.data(), simd.data(), static_cast<int>(scores.size()), 0.7f);
        if (!compare_float_vectors(simd, scalar, 8.0e-4f, 8.0e-4f, "avx512 selected softmax", report)) return 1;
        if (report) ++report->cases_run;
    }

    {
        std::vector<float> scores = {
            3.0f, 2.75f, 2.0f, 1.25f, 0.5f, 0.25f, -0.25f, -0.75f,
            -1.0f, -1.5f, -2.0f, -3.5f, -1.0e31f, 1.0f, 1.5f, -4.0f,
            0.0f, 2.25f, -2.5f, 0.875f,
        };
        std::vector<float> scalar(scores.size(), 0.0f);
        std::vector<float> simd(scores.size(), 0.0f);
        soft_topk_inclusion(scores.data(), scalar.data(), static_cast<int>(scores.size()), 5, 0.35f, 1.0e-5f, 80);

        KernelOverrideScope override(KernelPath::AVX512);
        soft_topk_inclusion(scores.data(), simd.data(), static_cast<int>(scores.size()), 5, 0.35f, 1.0e-5f, 80);
        if (!compare_float_vectors(simd, scalar, 1.5e-3f, 1.5e-3f, "avx512 relaxed topk", report)) return 1;
        if (report) ++report->cases_run;
    }

    if (report) report->simd_paths_run = 1;
    return report && report->failures ? 1 : 0;
}
#endif

} // namespace

SimdCapabilities simd_capabilities() {
    SimdCapabilities caps{};
    caps.can_compile_avx512 = DBS_CAN_COMPILE_AVX512;
    caps.can_compile_avx2 = DBS_CAN_COMPILE_AVX2;
    caps.can_compile_sse42 = DBS_CAN_COMPILE_SSE42;
    caps.can_compile_neon = DBS_ARM_NEON;
    caps.runtime_avx512 = runtime_has_avx512() ? 1 : 0;
    caps.runtime_avx2 = runtime_has_avx2() ? 1 : 0;
    caps.runtime_sse42 = runtime_has_sse42() ? 1 : 0;
    caps.runtime_neon = runtime_has_neon() ? 1 : 0;
    return caps;
}

int run_avx512_vector_math_parity(ParityReport* report) {
    reset_report(report);
#if DBS_CAN_COMPILE_AVX512
    if (!runtime_has_avx512()) {
        if (report) std::snprintf(report->message, sizeof(report->message), "AVX-512 runtime support unavailable");
        return 0;
    }
    return run_avx512_vector_math_parity_impl(report);
#else
    if (report) std::snprintf(report->message, sizeof(report->message), "AVX-512 compile support unavailable");
    return 0;
#endif
}

int run_scalar_vs_simd_decode_backward_parity(ParityReport* report) {
    reset_report(report);
    const std::vector<KernelPath> simd_paths = available_simd_paths();
    if (report) report->simd_paths_run = static_cast<int>(simd_paths.size());
    if (simd_paths.empty()) {
        if (report) std::snprintf(report->message, sizeof(report->message), "SIMD runtime support unavailable");
        return 0;
    }

    const std::vector<InternalParityCase> cases = build_internal_parity_cases();
    for (const InternalParityCase& tc : cases) {
        if (tc.expect_invalid) {
            bool scalar_rejected = false;
            try {
                (void)run_decode_backward_case(tc, KernelPath::Scalar);
            } catch (const std::invalid_argument&) {
                scalar_rejected = true;
            }
            if (!scalar_rejected) {
                report_failure(report, "scalar path accepted invalid NaN input");
                return 1;
            }

            for (KernelPath path : simd_paths) {
                bool simd_rejected = false;
                try {
                    (void)run_decode_backward_case(tc, path);
                } catch (const std::invalid_argument&) {
                    simd_rejected = true;
                }
                if (!simd_rejected) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg), "%s path accepted invalid NaN input", kernel_path_name(path));
                    report_failure(report, msg);
                    return 1;
                }
                if (report) ++report->cases_run;
            }
            continue;
        }

        DecodeBackwardBundle scalar;
        try {
            scalar = run_decode_backward_case(tc, KernelPath::Scalar);
        } catch (const std::exception& exc) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "scalar path failed case %s: %s", tc.name, exc.what());
            report_failure(report, msg);
            return 1;
        }

        for (KernelPath path : simd_paths) {
            DecodeBackwardBundle simd;
            try {
                simd = run_decode_backward_case(tc, path);
            } catch (const std::exception& exc) {
                char msg[256];
                std::snprintf(msg, sizeof(msg), "%s path failed case %s: %s", kernel_path_name(path), tc.name, exc.what());
                report_failure(report, msg);
                return 1;
            }

            if (!compare_decode_backward_bundles(simd, scalar, tc.name, kernel_path_name(path), report)) {
                return 1;
            }
            if (report) ++report->cases_run;
        }
    }

    return report && report->failures ? 1 : 0;
}

} // namespace internal_test

} // namespace dbs
