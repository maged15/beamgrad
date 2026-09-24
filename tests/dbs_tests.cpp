// SPDX-License-Identifier: MIT
#include "dbs.h"

#include "check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <cstring>
#include <vector>

static DBSOptionsC test_options() {
    DBSOptionsC opt{};
    opt.beam_size = 2;
    opt.eos_token = -1;
    opt.selected_temperature = 1.0f;
    opt.soft_topk_temperature = 0.25f;
    opt.relaxed_pool_multiplier = 4;
    opt.vocab_block = 64;
    opt.length_penalty_alpha = 0.0f;
    opt.soft_topk_tolerance = 1.0e-4f;
    opt.soft_topk_max_iters = 48;
    opt.min_length = 0;
    opt.validate_inputs = 1;
    opt.max_dense_gradient_elements = 1000000;
    return opt;
}

static DBSDecoderHandle* make_decoder() {
    DBSDecoderHandle* h = nullptr;
    const int rc = dbs_create_ex(test_options(), &h);
    if (rc != 0 || !h) {
        std::cerr << "create failed: " << dbs_last_global_error() << "\n";
        std::abort();
    }
    return h;
}

static float final_score0(DBSDecoderHandle* h, const std::vector<float>& x, int T, int V) {
    DBSResultHandle* r = nullptr;
    const int rc = dbs_decode(h, x.data(), T, V, &r);
    CHECK(rc == 0 && r);
    const float y = dbs_result_final_scores(r)[0];
    dbs_free_result(r);
    return y;
}

static void test_deterministic_ties() {
    auto* h = make_decoder();
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 5;
    std::vector<float> x(T * K * V, -1.0f);

    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    const int32_t* tok = dbs_result_tokens(r);
    CHECK(tok[0] == 0);
    CHECK(tok[1] == 1);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_constraints() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 5;
    std::vector<float> x(T * K * V, -10.0f);
    for (int i = 0; i < T * K * V; ++i) x[i] = -0.01f * static_cast<float>(i % V);
    int32_t forced[T] = {3, -1};

    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_constrained(h, x.data(), T, V, nullptr, forced, 0, &r) == 0);
    const int32_t* tok = dbs_result_tokens(r);
    CHECK(tok[0] == 3);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_invalid_nan_rejected() {
    auto* h = make_decoder();
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -1.0f);
    x[3] = std::numeric_limits<float>::quiet_NaN();
    DBSResultHandle* r = nullptr;
    const int rc = dbs_decode(h, x.data(), T, V, &r);
    CHECK(rc != 0);
    CHECK(r == nullptr);
    dbs_destroy(h);
}

static void test_sparse_gradient_matches_finite_difference() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x = {
        -0.10f, -0.40f, -1.00f, -2.00f,
        -3.00f, -3.20f, -3.40f, -3.60f,
        -0.20f, -0.50f, -1.10f, -2.10f,
        -0.30f, -1.00f, -1.20f, -2.20f,
    };

    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);

    float grad_final[K] = {1.0f, 0.0f};
    DBSBackwardHandle* b = nullptr;
    CHECK(dbs_backward_sparse(h, r, nullptr, nullptr, grad_final, &b) == 0);

    std::vector<float> analytic(T * K * V, 0.0f);
    const int64_t n = dbs_backward_sparse_logprob_count(b);
    const int64_t* idx = dbs_backward_sparse_logprob_indices(b);
    const float* val = dbs_backward_sparse_logprob_values(b);
    for (int64_t i = 0; i < n; ++i) analytic[static_cast<size_t>(idx[i])] += val[i];

    const float eps = 1.0e-3f;
    for (size_t i = 0; i < x.size(); ++i) {
        if (analytic[i] == 0.0f) continue;
        std::vector<float> xp = x;
        std::vector<float> xm = x;
        xp[i] += eps;
        xm[i] -= eps;
        const float fd = (final_score0(h, xp, T, V) - final_score0(h, xm, T, V)) / (2.0f * eps);
        CHECK(std::fabs(fd - analytic[i]) < 5.0e-2f);
    }

    dbs_free_backward(b);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_batch_decode() {
    auto* h = make_decoder();
    constexpr int B = 3;
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(B * T * K * V, -1.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.001f * static_cast<float>(i % 11);

    DBSBatchResultHandle* br = nullptr;
    CHECK(dbs_decode_batch(h, x.data(), B, T, V, 2, &br) == 0);
    CHECK(dbs_batch_result_size(br) == B);
    for (int b = 0; b < B; ++b) {
        const DBSResultHandle* r = dbs_batch_result_at(br, b);
        CHECK(r);
        CHECK(dbs_result_steps(r) == T);
        CHECK(dbs_result_beam_size(r) == K);
    }
    dbs_free_batch_result(br);
    dbs_destroy(h);
}


static int model_step_callback(
    void*,
    int,
    int step,
    const int32_t* prev_tokens,
    const float*,
    int beam_size,
    int vocab_size,
    float* out_log_probs
) {
    for (int k = 0; k < beam_size; ++k) {
        for (int v = 0; v < vocab_size; ++v) {
            out_log_probs[static_cast<size_t>(k) * vocab_size + v] = -10.0f;
        }
        const int preferred = step == 0 ? k : ((prev_tokens[k] + 1) % vocab_size);
        out_log_probs[static_cast<size_t>(k) * vocab_size + preferred] = 0.0f;
    }
    return 0;
}

static void test_default_backward_is_sparse() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -1.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.01f * static_cast<float>(i % 7);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    float grad_final[K] = {1.0f, 0.0f};
    DBSBackwardHandle* b = nullptr;
    CHECK(dbs_backward(h, r, nullptr, nullptr, grad_final, &b) == 0);
    CHECK(dbs_backward_is_sparse(b) == 1);
    CHECK(dbs_backward_sparse_logprob_count(b) > 0);
    dbs_free_backward(b);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_model_step_decode() {
    auto* h = make_decoder();
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_model_steps(h, model_step_callback, nullptr, 0, 3, 6, &r) == 0);
    CHECK(dbs_result_steps(r) == 3);
    const int32_t* tok = dbs_result_tokens(r);
    CHECK(tok[0] == 0);
    DBSStatsC stats{};
    CHECK(dbs_get_stats(h, &stats) == 0);
    CHECK(stats.used_model_step_callback == 1);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_advanced_constraints_no_repeat() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -5.0f);
    // Token 0 is best at both steps; no_repeat_ngram_size=1 should force a different token on step 2.
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            x[(static_cast<size_t>(t) * K + k) * V + 0] = 0.0f;
            x[(static_cast<size_t>(t) * K + k) * V + 1] = -0.1f;
        }
    }
    DBSAdvancedConstraintsC c{};
    c.min_length = -1;
    c.repetition_penalty = 1.0f;
    c.no_repeat_ngram_size = 1;
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_constrained_ex(h, x.data(), T, V, &c, &r) == 0);
    const int32_t* tok = dbs_result_tokens(r);
    CHECK(tok[K] != 0);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_variable_batch_decode() {
    auto* h = make_decoder();
    constexpr int B = 2;
    constexpr int maxT = 3;
    constexpr int maxK = 2;
    constexpr int V = 5;
    std::vector<float> x(B * maxT * maxK * V, -1.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.001f * static_cast<float>(i % 13);
    int32_t steps[B] = {3, 2};
    int32_t beams[B] = {2, 1};
    int32_t eos[B] = {-1, -1};
    int32_t minlen[B] = {0, 0};
    DBSBatchResultHandle* br = nullptr;
    CHECK(dbs_decode_batch_variable(h, x.data(), B, maxT, maxK, V, steps, beams, eos, minlen, nullptr, nullptr, 2, &br) == 0);
    CHECK(dbs_batch_result_size(br) == B);
    CHECK(dbs_result_steps(dbs_batch_result_at(br, 0)) == 3);
    CHECK(dbs_result_steps(dbs_batch_result_at(br, 1)) == 2);
    CHECK(dbs_result_beam_size(dbs_batch_result_at(br, 1)) == 1);
    dbs_free_batch_result(br);
    dbs_destroy(h);
}

static void test_observability_and_dispatch() {
    auto* h = make_decoder();
    DBSStatsC stats{};
    CHECK(dbs_get_stats(h, &stats) == 0);
    CHECK(stats.abi_version == DBS_ABI_VERSION);
    CHECK(dbs_selected_kernel_name() != nullptr);
    dbs_reset_stats(h);
    CHECK(dbs_get_stats(h, &stats) == 0);
    CHECK(stats.abi_version == DBS_ABI_VERSION);
    dbs_destroy(h);
}


static uint16_t f32_to_bf16(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static void test_typed_bf16_decode() {
    auto* h = make_decoder();
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> xf = {-0.1f, -0.2f, -0.3f, -0.4f, -2.0f, -2.1f, -2.2f, -2.3f};
    std::vector<uint16_t> xb(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) xb[i] = f32_to_bf16(xf[i]);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_typed(h, xb.data(), DBS_DTYPE_BF16, T, V, &r) == 0);
    CHECK(dbs_result_tokens(r)[0] == 0);
    CHECK(dbs_result_beam_size(r) == K);
    dbs_free_result(r);
    dbs_destroy(h);
}

static int even_token_filter(void*, int, int, int, const int32_t*, int, int token) {
    return (token % 2) == 0;
}

static void test_token_filter_constraint() {
    auto* h = make_decoder();
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 5;
    std::vector<float> x(T * K * V, -10.0f);
    x[1] = 0.0f;  // best token would be odd and must be rejected
    x[2] = -0.1f;
    DBSAdvancedConstraintsC c{};
    c.min_length = -1;
    c.repetition_penalty = 1.0f;
    c.token_filter = even_token_filter;
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_constrained_ex(h, x.data(), T, V, &c, &r) == 0);
    CHECK((dbs_result_tokens(r)[0] % 2) == 0);
    dbs_free_result(r);
    dbs_destroy(h);
}


static void test_banned_token_mask_rejects_best_token() {
    auto* h = make_decoder();
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 5;
    std::vector<float> x(T * K * V, -10.0f);
    x[0] = 0.0f;   // best token but banned
    x[1] = -0.1f;  // next-best allowed token
    std::vector<uint8_t> banned(V, 0);
    banned[0] = 1;
    DBSAdvancedConstraintsC c{};
    c.banned_tokens = banned.data();
    c.min_length = -1;
    c.repetition_penalty = 1.0f;
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_constrained_ex(h, x.data(), T, V, &c, &r) == 0);
    CHECK(dbs_result_tokens(r)[0] != 0);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_forced_token_sequence_overrides_scores() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 6;
    std::vector<float> x(T * K * V, -10.0f);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            x[(static_cast<size_t>(t) * K + k) * V + 0] = 0.0f;
        }
    }
    int32_t forced[T] = {4, 5};
    DBSAdvancedConstraintsC c{};
    c.forced_tokens = forced;
    c.min_length = -1;
    c.repetition_penalty = 1.0f;
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_constrained_ex(h, x.data(), T, V, &c, &r) == 0);
    const int32_t* tok = dbs_result_tokens(r);
    CHECK(tok[0] == 4);
    CHECK(tok[K] == 5);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_workspace_reuse_no_growth() {
    auto* h = make_decoder();
    DBSWorkspaceHandle* ws = nullptr;
    CHECK(dbs_workspace_create(&ws) == 0);
    CHECK(dbs_workspace_reserve(ws, 4096, 128) == 0);
    const int64_t before = dbs_workspace_allocated_bytes(ws);
    for (int i = 0; i < 2; ++i) {
        DBSResultHandle* r = nullptr;
        CHECK(dbs_decode_model_steps_with_workspace(h, ws, model_step_callback, nullptr, 0, 3, 6, &r) == 0);
        dbs_free_result(r);
    }
    const int64_t after = dbs_workspace_allocated_bytes(ws);
    CHECK(after == before);
    dbs_workspace_destroy(ws);
    dbs_destroy(h);
}

static void test_extreme_logits_stable() {
    auto* h = make_decoder();
    constexpr int T = 3;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -1.0e20f);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            x[(static_cast<size_t>(t) * K + k) * V + (t % V)] = 0.0f;
        }
    }
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    const float* fs = dbs_result_final_scores(r);
    CHECK(std::isfinite(fs[0]));
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_stats_json_and_determinism_contract() {
    auto* h = make_decoder();
    CHECK(dbs_is_deterministic() == 1);
    char buf[1024];
    CHECK(dbs_get_stats_json(h, buf, sizeof(buf)) == 0);
    CHECK(std::strstr(buf, "\"abi_version\"") != nullptr);
    dbs_destroy(h);
}

static void test_golden_output() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 3;
    std::vector<float> x = {
        0.0f, -1.0f, -2.0f,
        -3.0f, -4.0f, -5.0f,
        -0.5f, -0.1f, -2.0f,
        -0.2f, -0.3f, -2.0f,
    };
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    const int32_t* tok = dbs_result_tokens(r);
    CHECK(tok[0] == 0);
    CHECK(tok[K] == 1);
    dbs_free_result(r);
    dbs_destroy(h);
}


static void test_allocator_counters_and_seed() {
    dbs_allocator_counters_reset();
    CHECK(dbs_allocator_call_count() == 0);
    auto* h = make_decoder();
    CHECK(dbs_set_deterministic_seed(h, 123456789ULL) == 0);
    CHECK(dbs_get_deterministic_seed(h) == 123456789ULL);
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -0.1f);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    DBSStatsC stats{};
    CHECK(dbs_get_stats(h, &stats) == 0);
    CHECK(stats.total_allocator_calls == dbs_allocator_call_count());
    CHECK(stats.total_allocator_bytes == dbs_allocator_byte_count());
    CHECK(stats.total_allocator_calls > 0);
    char buf[2048];
    CHECK(dbs_get_stats_json(h, buf, sizeof(buf)) == 0);
    CHECK(std::strstr(buf, "\"allocator_calls\"") != nullptr);
    dbs_free_result(r);
    dbs_destroy(h);
}

static uint16_t f32_to_f16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static void test_typed_fp16_decode() {
    auto* h = make_decoder();
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> xf = {-0.1f, -0.2f, -0.3f, -0.4f, -2.0f, -2.1f, -2.2f, -2.3f};
    std::vector<uint16_t> xh(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) xh[i] = f32_to_f16(xf[i]);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_typed(h, xh.data(), DBS_DTYPE_F16, T, V, &r) == 0);
    CHECK(dbs_result_tokens(r)[0] == 0);
    CHECK(dbs_result_beam_size(r) == K);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_dense_backward_memory_cap_fails_closed() {
    DBSOptionsC opt = test_options();
    opt.max_dense_gradient_elements = 4;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -0.1f);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    DBSBackwardHandle* b = nullptr;
    float grad_final[K] = {1.0f, 0.0f};
    CHECK(dbs_backward_dense(h, r, nullptr, nullptr, grad_final, &b) != 0);
    CHECK(b == nullptr);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_result_summary_and_ordering_api() {
    auto* h = make_decoder();
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 4;
    std::vector<float> x(T * K * V, -1.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.01f * static_cast<float>(i % 5);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    CHECK(dbs_result_validate_deterministic_order(r) == 0);
    char buf[1024];
    CHECK(dbs_result_summary_json(r, -1, buf, sizeof(buf)) == 0);
    CHECK(std::strstr(buf, "\"deterministic_order\":1") != nullptr);
    CHECK(std::strstr(buf, "\"selected_count\":4") != nullptr);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_zero_initialized_options_select_defaults() {
    DBSOptionsC opt{};
    opt.eos_token = -1;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    constexpr int T = 2;
    constexpr int K = 8;  // default beam size
    constexpr int V = 16;
    std::vector<float> x(T * K * V, -1.0f);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    CHECK(dbs_result_beam_size(r) == K);
    CHECK(dbs_result_pool_size(r) == K * 8);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_invalid_options_are_rejected() {
    const auto rejects = [](DBSOptionsC opt, const char* fragment) {
        DBSDecoderHandle* h = nullptr;
        CHECK(dbs_create_ex(opt, &h) != 0);
        CHECK(h == nullptr);
        CHECK(std::strstr(dbs_last_global_error(), fragment) != nullptr);
    };
    DBSOptionsC opt = test_options();
    opt.beam_size = -2;
    rejects(opt, "beam_size");
    opt = test_options();
    opt.eos_token = -7;
    rejects(opt, "eos_token");
    opt = test_options();
    opt.selected_temperature = -1.0f;
    rejects(opt, "selected_temperature");
    opt = test_options();
    opt.soft_topk_temperature = std::numeric_limits<float>::quiet_NaN();
    rejects(opt, "soft_topk_temperature");
    opt = test_options();
    opt.length_penalty_alpha = -0.5f;
    rejects(opt, "length_penalty_alpha");
    opt = test_options();
    opt.min_length = -1;
    rejects(opt, "min_length");
    opt = test_options();
    opt.max_dense_gradient_elements = -1;
    rejects(opt, "max_dense_gradient_elements");
}

static void test_last_error_is_reported_per_handle() {
    auto* h = make_decoder();
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, nullptr, 1, 4, &r) != 0);
    CHECK(r == nullptr);
    CHECK(std::strlen(dbs_last_error(h)) > 0);
    std::vector<float> x(2 * 4, -1.0f);
    CHECK(dbs_decode(h, x.data(), 1, 4, &r) == 0);
    CHECK(std::strlen(dbs_last_error(h)) == 0);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_deprecated_production_manifest_api_fails_closed() {
    char err[256];
    const char* manifest = "{\"cuda_parity\":true}";
    CHECK(dbs_validate_production_gate_manifest(manifest, err, sizeof(err)) != 0);
    CHECK(std::strstr(err, "deprecated") != nullptr);
}

int main() {
    CHECK(dbs_abi_version() == DBS_ABI_VERSION);
    char expected_version[32];
    std::snprintf(expected_version, sizeof(expected_version), "%d.%d.%d", DBS_VERSION_MAJOR, DBS_VERSION_MINOR, DBS_VERSION_PATCH);
    CHECK(std::strcmp(dbs_version_string(), expected_version) == 0);
    test_deterministic_ties();
    test_constraints();
    test_invalid_nan_rejected();
    test_sparse_gradient_matches_finite_difference();
    test_batch_decode();
    test_default_backward_is_sparse();
    test_model_step_decode();
    test_advanced_constraints_no_repeat();
    test_variable_batch_decode();
    test_observability_and_dispatch();
    test_typed_bf16_decode();
    test_typed_fp16_decode();
    test_token_filter_constraint();
    test_banned_token_mask_rejects_best_token();
    test_forced_token_sequence_overrides_scores();
    test_workspace_reuse_no_growth();
    test_extreme_logits_stable();
    test_stats_json_and_determinism_contract();
    test_golden_output();
    test_allocator_counters_and_seed();
    test_dense_backward_memory_cap_fails_closed();
    test_result_summary_and_ordering_api();
    test_zero_initialized_options_select_defaults();
    test_invalid_options_are_rejected();
    test_last_error_is_reported_per_handle();
    test_deprecated_production_manifest_api_fails_closed();
    std::cout << "dbs_tests passed\n";
    return 0;
}
