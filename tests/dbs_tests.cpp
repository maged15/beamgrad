// SPDX-License-Identifier: MIT
#include "dbs.h"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <cstring>
#include <thread>
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

// One step from a single start beam: every pool candidate is one log-prob
// entry x[0, 0, token], so the gradient of sum(g * relaxed_weights) is the
// implicit-function result a_p / tau * (g_p - sum(g a) / sum(a)), with
// a_p = r_p (1 - r_p), at exactly that entry and nowhere else.
static void test_relaxed_pool_gradient_closed_form() {
    DBSOptionsC opt = test_options();
    opt.beam_size = 2;
    opt.relaxed_pool_multiplier = 3;  // P = 6 of the V = 8 candidates
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    constexpr int T = 1;
    constexpr int K = 2;
    constexpr int V = 8;
    constexpr int P = 6;
    const std::vector<float> x = {-0.3f, -1.1f, -0.7f, -2.0f, -0.9f, -1.6f, -3.0f, -1.3f,  // beam 0
                                  -9.f,  -9.f,  -9.f,  -9.f,  -9.f,  -9.f,  -9.f,  -9.f};  // unused at t = 0
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    CHECK(dbs_result_pool_size(r) == P);
    const float* w = dbs_result_relaxed_weights(r);
    const int32_t* tokens = dbs_result_pool_tokens(r);
    const int32_t* parents = dbs_result_pool_parents(r);
    const float g[P] = {0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.5f};
    DBSBackwardHandle* b = nullptr;
    CHECK(dbs_backward_dense(h, r, nullptr, g, nullptr, &b) == 0);
    const float* grad = dbs_backward_grad_log_probs(b);
    double sum_a = 0.0;
    double sum_ga = 0.0;
    for (int p = 0; p < P; ++p) {
        const double a = static_cast<double>(w[p]) * (1.0 - w[p]);
        sum_a += a;
        sum_ga += g[p] * a;
    }
    std::vector<double> expected(static_cast<size_t>(T * K * V), 0.0);
    for (int p = 0; p < P; ++p) {
        CHECK(parents[p] == 0);
        const double a = static_cast<double>(w[p]) * (1.0 - w[p]);
        expected[static_cast<size_t>(tokens[p])] = a / opt.soft_topk_temperature * (g[p] - sum_ga / sum_a);
    }
    for (int i = 0; i < T * K * V; ++i) {
        CHECK(std::fabs(grad[i] - expected[static_cast<size_t>(i)]) <= 1.0e-5 + 1.0e-4 * std::fabs(expected[i]));
    }
    dbs_free_backward(b);
    dbs_free_result(r);
    dbs_destroy(h);
}

// Automatic threading (num_threads <= 0) runs a small batch on the calling
// thread, where starting threads would cost more than the work; an explicit
// thread count is honoured, capped at the batch size.
static void test_small_batches_run_on_one_thread() {
    DBSOptionsC opt = test_options();
    opt.relaxed_pool_multiplier = 0;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    const int K = opt.beam_size;
    auto threads_used = [&](int B, int T, int V, int num_threads) {
        std::vector<float> x(static_cast<size_t>(B) * T * K * V, -1.0f);
        for (size_t i = 0; i < x.size(); ++i) x[i] = -static_cast<float>((i * 7919) % 1000) / 100.0f;
        std::vector<float> final_scores(static_cast<size_t>(B) * K);
        DBSDecodeOutputsC out{};
        out.final_scores = final_scores.data();
        CHECK(dbs_decode_batch_into(h, x.data(), B, T, V, nullptr, nullptr, num_threads, &out) == 0);
        DBSStatsC stats{};
        CHECK(dbs_get_stats(h, &stats) == 0);
        return stats.used_batch_threads;
    };
    CHECK(threads_used(4, 4, 16, 0) == 1);  // 512 candidates: one thread
    CHECK(threads_used(4, 4, 16, 3) == 3);  // explicit: honoured
    CHECK(threads_used(2, 4, 16, 8) == 2);  // capped at the batch size
    // Enough work for threads: one per hardware thread, capped at the batch size.
    const int hardware = static_cast<int>(std::thread::hardware_concurrency());
    const int expected = std::max(1, std::min(4, hardware));
    CHECK(threads_used(4, 64, 1024, 0) == expected);  // 4 * 64 * 2 * 1024 = 524288 candidates
    dbs_destroy(h);
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
    CHECK(dbs_allocator_byte_count() >= 0);
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
    CHECK(dbs_result_pool_size(r) == 0);  // the relaxed pool is opt-in
    CHECK(dbs_result_relaxed_weights(r) == nullptr);
    float grad_relaxed[1] = {1.0f};
    DBSBackwardHandle* b = nullptr;
    CHECK(dbs_backward(h, r, nullptr, grad_relaxed, nullptr, &b) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(b == nullptr);
    CHECK(std::strstr(dbs_last_error(h), "relaxed_pool_multiplier") != nullptr);
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_invalid_options_are_rejected() {
    const auto rejects = [](DBSOptionsC opt, const char* fragment) {
        DBSDecoderHandle* h = nullptr;
        CHECK(dbs_create_ex(opt, &h) == DBS_ERROR_INVALID_ARGUMENT);
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

static void test_error_codes_match_the_header() {
    auto* h = make_decoder();
    DBSResultHandle* r = nullptr;
    std::vector<float> x(2 * 2 * 4, -1.0f);
    CHECK(dbs_decode(h, nullptr, 2, 4, &r) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(dbs_decode(h, x.data(), 0, 4, &r) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(dbs_decode(h, x.data(), 2, -3, &r) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(dbs_decode(h, x.data(), 2, 4, nullptr) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(dbs_decode(nullptr, x.data(), 2, 4, &r) == DBS_ERROR_INVALID_ARGUMENT);
    x[1] = std::numeric_limits<float>::quiet_NaN();
    CHECK(dbs_decode(h, x.data(), 2, 4, &r) == DBS_ERROR_INVALID_ARGUMENT);
    DBSStatsC stats{};
    CHECK(dbs_get_stats(h, &stats) == DBS_OK);
    CHECK(stats.last_error_category == 1);
    // A callback failure is a runtime error.
    const DBSModelStepExFn failing = [](void*, const DBSModelStepInfoC*, float*) { return 7; };
    CHECK(dbs_decode_model_steps_ex(h, failing, nullptr, 0, 2, 4, nullptr, &r) == DBS_ERROR_RUNTIME);
    CHECK(r == nullptr);
    CHECK(std::strstr(dbs_last_error(h), "callback") != nullptr);
    dbs_destroy(h);
}

static void test_every_failure_reports_its_own_error() {
    auto* h = make_decoder();
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, nullptr, 1, 4, &r) != 0);
    CHECK(std::strstr(dbs_last_global_error(), "log_probs") != nullptr);
    // Early failures (null handle, null output) replace the previous message.
    CHECK(dbs_decode(nullptr, nullptr, 1, 4, &r) != 0);
    CHECK(std::strstr(dbs_last_global_error(), "handle") != nullptr);
    CHECK(dbs_backward(h, nullptr, nullptr, nullptr, nullptr, nullptr) != 0);
    CHECK(std::strstr(dbs_last_global_error(), "out_backward") != nullptr);
    CHECK(dbs_get_stats(nullptr, nullptr) != 0);
    CHECK(std::strstr(dbs_last_global_error(), "out_stats") != nullptr);
    CHECK(dbs_workspace_reserve(nullptr, 1, 1) != 0);
    CHECK(std::strstr(dbs_last_global_error(), "workspace") != nullptr);
    // Success clears the thread's message.
    std::vector<float> x(2 * 4, -1.0f);
    CHECK(dbs_decode(h, x.data(), 1, 4, &r) == 0);
    CHECK(dbs_last_global_error()[0] == '\0');
    dbs_free_result(r);
    dbs_destroy(h);
}

static void test_deterministic_order_accepts_decoder_output() {
    // Length penalty plus EOS carry-forward produce score ties broken by the raw score.
    uint32_t seed = 12345u;
    auto next = [&seed]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (int trial = 0; trial < 300; ++trial) {
        DBSOptionsC opt = test_options();
        opt.beam_size = 1 + static_cast<int>(next() % 6);
        opt.eos_token = static_cast<int>(next() % 3);
        opt.length_penalty_alpha = (next() % 2) ? 0.6f : 1.0f;
        opt.min_length = static_cast<int>(next() % 3);
        DBSDecoderHandle* h = nullptr;
        CHECK(dbs_create_ex(opt, &h) == 0);
        const int T = 1 + static_cast<int>(next() % 6), V = 3 + static_cast<int>(next() % 6);
        std::vector<float> x(static_cast<size_t>(T) * opt.beam_size * V);
        for (float& v : x) v = -0.5f * static_cast<float>(next() % 5);
        DBSResultHandle* r = nullptr;
        CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
        CHECK(dbs_result_validate_deterministic_order(r) == 0);
        dbs_free_result(r);
        dbs_destroy(h);
    }
}

static void test_allocator_gauge_survives_reset() {
    auto* h = make_decoder();
    std::vector<float> x(3 * 2 * 8, -1.0f);
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, x.data(), 3, 8, &r) == 0);
    const int64_t live = dbs_allocator_byte_count();
    CHECK(live > 0);
    dbs_allocator_counters_reset();
    CHECK(dbs_allocator_call_count() == 0);
    CHECK(dbs_allocator_byte_count() == live);
    dbs_free_result(r);
    CHECK(dbs_allocator_byte_count() >= 0);
    CHECK(dbs_allocator_byte_count() < live);
    dbs_destroy(h);
}

static void test_variable_beam_batch_backward() {
    // Examples decoded with their own beam sizes go through backward on the
    // batch's handle, and match a handle created for that beam size.
    DBSOptionsC opt = test_options();
    opt.beam_size = 4;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    constexpr int B = 2, maxT = 3, maxK = 4, V = 6;
    std::vector<float> x(static_cast<size_t>(B) * maxT * maxK * V);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.1f * static_cast<float>((i * 7) % 13);
    int32_t steps[B] = {3, 2};
    int32_t beams[B] = {2, 4};
    DBSBatchResultHandle* br = nullptr;
    CHECK(dbs_decode_batch_variable(h, x.data(), B, maxT, maxK, V, steps, beams, nullptr, nullptr, nullptr, nullptr, 1, &br) == 0);
    const DBSResultHandle* r0 = dbs_batch_result_at(br, 0);
    CHECK(dbs_result_beam_size(r0) == 2);

    float grad_final[2] = {1.0f, -0.5f};
    DBSBackwardHandle* b = nullptr;
    CHECK(dbs_backward(h, r0, nullptr, nullptr, grad_final, &b) == 0);
    DBSBackwardHandle* dense = nullptr;
    CHECK(dbs_backward_dense(h, r0, nullptr, nullptr, grad_final, &dense) == 0);

    // The same example decoded on its own with a beam-2 handle.
    DBSOptionsC opt2 = test_options();
    opt2.beam_size = 2;
    DBSDecoderHandle* h2 = nullptr;
    CHECK(dbs_create_ex(opt2, &h2) == 0);
    std::vector<float> x0;
    for (int t = 0; t < 3; ++t) {
        for (int k = 0; k < 2; ++k) {
            const float* row = x.data() + (static_cast<size_t>(t) * maxK + static_cast<size_t>(k)) * V;
            x0.insert(x0.end(), row, row + V);
        }
    }
    DBSResultHandle* r2 = nullptr;
    CHECK(dbs_decode(h2, x0.data(), 3, V, &r2) == 0);
    for (int i = 0; i < 3 * 2; ++i) CHECK(dbs_result_tokens(r2)[i] == dbs_result_tokens(r0)[i]);
    DBSBackwardHandle* b2 = nullptr;
    CHECK(dbs_backward(h2, r2, nullptr, nullptr, grad_final, &b2) == 0);
    CHECK(dbs_backward_sparse_logprob_count(b) == dbs_backward_sparse_logprob_count(b2));
    for (int64_t i = 0; i < dbs_backward_sparse_logprob_count(b); ++i) {
        CHECK(dbs_backward_sparse_logprob_indices(b)[i] == dbs_backward_sparse_logprob_indices(b2)[i]);
        CHECK(dbs_backward_sparse_logprob_values(b)[i] == dbs_backward_sparse_logprob_values(b2)[i]);
        CHECK(dbs_backward_grad_log_probs(dense)[dbs_backward_sparse_logprob_indices(b)[i]] == dbs_backward_sparse_logprob_values(b)[i]);
    }
    dbs_free_backward(b);
    dbs_free_backward(b2);
    dbs_free_backward(dense);
    dbs_free_result(r2);
    dbs_free_batch_result(br);
    dbs_destroy(h2);
    dbs_destroy(h);
}

struct RecordingModel {
    int K = 0;
    int V = 0;
    std::vector<float> rows;  // every step's rows, as produced
    bool consistent = true;
};

// Next-token scores that depend on the whole prefix of each beam.
static int prefix_model(void* user_data, const DBSModelStepInfoC* info, float* out) {
    auto* m = static_cast<RecordingModel*>(user_data);
    for (int k = 0; k < info->beam_size; ++k) {
        uint32_t h = 2166136261u;
        for (int s = 0; s < info->step; ++s) {
            h = (h ^ static_cast<uint32_t>(info->prefixes[static_cast<size_t>(k) * info->step + s] + 3)) * 16777619u;
        }
        if (info->step > 0 && info->parents[k] >= 0) {
            m->consistent = m->consistent && info->prefixes[static_cast<size_t>(k) * info->step + info->step - 1] == info->tokens[k];
        }
        for (int v = 0; v < info->vocab_size; ++v) {
            h = (h ^ static_cast<uint32_t>(v)) * 16777619u;
            out[static_cast<size_t>(k) * info->vocab_size + v] = -static_cast<float>(h % 89u) / 8.0f;
        }
    }
    m->rows.insert(m->rows.end(), out, out + static_cast<size_t>(info->beam_size) * info->vocab_size);
    return 0;
}

static void test_model_steps_ex_tracks_beam_prefixes() {
    DBSOptionsC opt = test_options();
    opt.beam_size = 3;
    opt.eos_token = 2;
    opt.length_penalty_alpha = 0.6f;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    constexpr int T = 6, V = 7;
    RecordingModel model;
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode_model_steps_ex(h, prefix_model, &model, 0, T, V, nullptr, &r) == 0);
    CHECK(model.consistent);
    CHECK(model.rows.size() == static_cast<size_t>(T) * 3 * V);
    // Decoding the rows the model produced gives the same beams.
    DBSResultHandle* r2 = nullptr;
    CHECK(dbs_decode(h, model.rows.data(), T, V, &r2) == 0);
    for (int i = 0; i < T * 3; ++i) {
        CHECK(dbs_result_tokens(r)[i] == dbs_result_tokens(r2)[i]);
        CHECK(dbs_result_parents(r)[i] == dbs_result_parents(r2)[i]);
    }
    for (int k = 0; k < 3; ++k) CHECK(dbs_result_final_scores(r)[k] == dbs_result_final_scores(r2)[k]);
    // Constraints apply to model-driven decoding too.
    DBSAdvancedConstraintsC c{};
    c.min_length = -1;
    c.no_repeat_ngram_size = 1;
    RecordingModel constrained;
    DBSResultHandle* r3 = nullptr;
    CHECK(dbs_decode_model_steps_ex(h, prefix_model, &constrained, 0, T, V, &c, &r3) == 0);
    const int32_t* tok = dbs_result_tokens(r3);
    const int32_t* par = dbs_result_parents(r3);
    for (int k = 0; k < 3; ++k) {
        // Walk back the final beam: no token may repeat (EOS carry-forward aside).
        std::vector<int> seen(V, 0);
        int beam = k;
        for (int t = T - 1; t >= 0 && beam >= 0; --t) {
            const int token = tok[t * 3 + beam];
            const int parent = par[t * 3 + beam];
            const bool carried = t > 0 && parent >= 0 && tok[(t - 1) * 3 + parent] == 2 && token == 2;
            if (!carried && token >= 0) CHECK(++seen[token] == 1);
            beam = parent;
        }
    }
    dbs_free_result(r);
    dbs_free_result(r2);
    dbs_free_result(r3);
    dbs_destroy(h);
}

static void test_batch_into_matches_result_handles() {
    DBSOptionsC opt = test_options();
    opt.beam_size = 3;
    opt.eos_token = 1;
    opt.min_length = 2;
    opt.length_penalty_alpha = 0.8f;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    constexpr int B = 4, T = 5, K = 3, V = 9;
    std::vector<float> x(static_cast<size_t>(B) * T * K * V);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.05f * static_cast<float>((i * 31) % 17);
    int32_t steps[B] = {5, 1, 3, 5};

    std::vector<float> final_scores(B * K), final_raw(B * K), scores(B * T * K), raw(B * T * K);
    std::vector<int32_t> final_lengths(B * K), tokens(B * T * K), parents(B * T * K), lengths(B * T * K);
    std::vector<uint8_t> from_logprob(B * T * K);
    DBSDecodeOutputsC out{};
    out.final_scores = final_scores.data();
    out.final_raw_scores = final_raw.data();
    out.final_lengths = final_lengths.data();
    out.tokens = tokens.data();
    out.parents = parents.data();
    out.lengths = lengths.data();
    out.scores = scores.data();
    out.raw_scores = raw.data();
    out.from_logprob = from_logprob.data();
    CHECK(dbs_decode_batch_into(h, x.data(), B, T, V, steps, nullptr, 3, &out) == 0);

    std::vector<float> grad_final(B * K);
    for (size_t i = 0; i < grad_final.size(); ++i) grad_final[i] = 0.25f * static_cast<float>(i % 5) - 0.5f;
    std::vector<float> grad(x.size(), 0.0f);
    CHECK(dbs_backward_batch_into(h, B, T, V, steps, parents.data(), tokens.data(), lengths.data(), from_logprob.data(),
                                  grad_final.data(), 2, grad.data()) == 0);

    for (int b = 0; b < B; ++b) {
        DBSResultHandle* r = nullptr;
        CHECK(dbs_decode(h, x.data() + static_cast<size_t>(b) * T * K * V, steps[b], V, &r) == 0);
        for (int i = 0; i < T * K; ++i) {
            const size_t o = static_cast<size_t>(b) * T * K + static_cast<size_t>(i);
            if (i < steps[b] * K) {
                CHECK(tokens[o] == dbs_result_tokens(r)[i] && parents[o] == dbs_result_parents(r)[i]);
                CHECK(lengths[o] == dbs_result_lengths(r)[i] && scores[o] == dbs_result_scores(r)[i]);
                CHECK(raw[o] == dbs_result_raw_scores(r)[i]);
            } else {
                CHECK(tokens[o] == -1 && parents[o] == -1 && lengths[o] == 0 && from_logprob[o] == 0);
                CHECK(std::isinf(scores[o]) && scores[o] < 0);
            }
        }
        for (int k = 0; k < K; ++k) {
            CHECK(final_scores[b * K + k] == dbs_result_final_scores(r)[k]);
            CHECK(final_raw[b * K + k] == dbs_result_final_raw_scores(r)[k]);
        }
        DBSBackwardHandle* bw = nullptr;
        CHECK(dbs_backward_dense(h, r, nullptr, nullptr, grad_final.data() + b * K, &bw) == 0);
        const float* dense = dbs_backward_grad_log_probs(bw);
        for (int i = 0; i < steps[b] * K * V; ++i) CHECK(grad[static_cast<size_t>(b) * T * K * V + i] == dense[i]);
        for (int i = steps[b] * K * V; i < T * K * V; ++i) CHECK(grad[static_cast<size_t>(b) * T * K * V + i] == 0.0f);
        dbs_free_backward(bw);
        dbs_free_result(r);
    }

    // Malformed traces and steps are rejected before any memory is touched.
    parents[0] = 7;
    CHECK(dbs_backward_batch_into(h, B, T, V, steps, parents.data(), tokens.data(), lengths.data(), from_logprob.data(),
                                  grad_final.data(), 1, grad.data()) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(std::strstr(dbs_last_error(h), "example 0") != nullptr);
    steps[2] = T + 1;
    CHECK(dbs_decode_batch_into(h, x.data(), B, T, V, steps, nullptr, 1, &out) == DBS_ERROR_INVALID_ARGUMENT);
    dbs_destroy(h);
}

static void test_validation_checks_every_element() {
    // The old implementation sampled 1000 entries; now every element of every
    // row the search reads is checked.
    auto* h = make_decoder();
    constexpr int T = 2, K = 2, V = 5003;
    std::vector<float> x(static_cast<size_t>(T) * K * V, -2.0f);
    DBSResultHandle* r = nullptr;
    for (size_t pos : {size_t{1}, size_t{4999}, static_cast<size_t>(K) * V + 17, static_cast<size_t>(K) * V + V + 4242}) {
        std::vector<float> y = x;
        y[pos] = std::numeric_limits<float>::infinity();
        CHECK(dbs_decode(h, y.data(), T, V, &r) == DBS_ERROR_INVALID_ARGUMENT);
        CHECK(std::strstr(dbs_last_error(h), "NaN or +inf") != nullptr);
    }
    // Row 1 at step 0 is never read (only beam 0 is live), so it is not checked.
    x[static_cast<size_t>(V) + 3] = std::numeric_limits<float>::quiet_NaN();
    CHECK(dbs_decode(h, x.data(), T, V, &r) == 0);
    dbs_free_result(r);
    dbs_destroy(h);
}

// ---------------------------------------------------------------------------
// dbs_decode_batch_into_ex and dbs_backward_batch_into_ex
// ---------------------------------------------------------------------------

namespace {

struct BatchTrace {
    int B, T, K, P;
    std::vector<float> final_scores, final_raw, scores, raw, pool_scores, pool_raw, row_lse;
    std::vector<int32_t> final_lengths, tokens, parents, lengths, pool_parents, pool_tokens, pool_lengths;
    std::vector<uint8_t> from_logprob, pool_from_logprob;
    DBSDecodeOutputsExC out{};

    BatchTrace(int B_, int T_, int K_, int P_) : B(B_), T(T_), K(K_), P(P_) {
        const size_t bk = static_cast<size_t>(B) * K, btk = bk * T, btp = static_cast<size_t>(B) * T * P;
        final_scores.assign(bk, 7.0f); final_raw.assign(bk, 7.0f); final_lengths.assign(bk, 7);
        scores.assign(btk, 7.0f); raw.assign(btk, 7.0f); tokens.assign(btk, 7); parents.assign(btk, 7);
        lengths.assign(btk, 7); from_logprob.assign(btk, 7); row_lse.assign(btk, 7.0f);
        out.base = DBSDecodeOutputsC{final_scores.data(), final_raw.data(), final_lengths.data(), tokens.data(),
                                     parents.data(), lengths.data(), scores.data(), raw.data(), from_logprob.data()};
        out.row_lse = row_lse.data();
        if (P > 0) {
            pool_scores.assign(btp, 7.0f); pool_raw.assign(btp, 7.0f); pool_parents.assign(btp, 7);
            pool_tokens.assign(btp, 7); pool_lengths.assign(btp, 7); pool_from_logprob.assign(btp, 7);
            out.pool_parents = pool_parents.data(); out.pool_tokens = pool_tokens.data();
            out.pool_lengths = pool_lengths.data(); out.pool_scores = pool_scores.data();
            out.pool_raw_scores = pool_raw.data(); out.pool_from_logprob = pool_from_logprob.data();
        }
    }
};

template <class A>
bool same_bits(const A& a, const A& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0);
}

bool same_selection(const BatchTrace& a, const BatchTrace& b) {
    return same_bits(a.tokens, b.tokens) && same_bits(a.parents, b.parents) && same_bits(a.lengths, b.lengths) &&
           same_bits(a.from_logprob, b.from_logprob) && same_bits(a.pool_tokens, b.pool_tokens) &&
           same_bits(a.pool_parents, b.pool_parents);
}

bool same_trace(const BatchTrace& a, const BatchTrace& b) {
    return same_selection(a, b) && same_bits(a.scores, b.scores) && same_bits(a.raw, b.raw) &&
           same_bits(a.final_scores, b.final_scores) && same_bits(a.final_raw, b.final_raw) &&
           same_bits(a.final_lengths, b.final_lengths) && same_bits(a.pool_scores, b.pool_scores) &&
           same_bits(a.pool_raw, b.pool_raw) && same_bits(a.pool_lengths, b.pool_lengths) &&
           same_bits(a.pool_from_logprob, b.pool_from_logprob);
}

uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

} // namespace

static void test_decode_batch_into_ex() {
    DBSOptionsC opt = test_options();
    opt.beam_size = 3;
    opt.eos_token = 2;
    opt.length_penalty_alpha = 0.6f;
    opt.relaxed_pool_multiplier = 2;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    const int B = 3, T = 5, K = 3, V = 11, P = K * 2;
    const int32_t steps[B] = {5, 3, 4};
    std::vector<float> x(static_cast<size_t>(B) * T * K * V);
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i * 7919 + 13) % 997) / 83.0f - 6.0f;
    // Round to bf16 values, so the float and bf16 inputs are the same numbers.
    std::vector<uint16_t> xb(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xb[i] = to_bf16(x[i]);
        x[i] = from_bf16(xb[i]);
    }

    // Float log-probs through _ex equal dbs_decode_batch_into, bit for bit.
    BatchTrace a(B, T, K, 0), b(B, T, K, 0);
    CHECK(dbs_decode_batch_into(h, x.data(), B, T, V, steps, nullptr, 1, &a.out.base) == 0);
    CHECK(dbs_decode_batch_into_ex(h, x.data(), DBS_DTYPE_F32, 0, B, T, V, steps, nullptr, 2, &b.out) == 0);
    CHECK(same_trace(a, b));
    for (float v : b.row_lse) CHECK(v == 0.0f);  // not from logits: every row_lse entry is 0

    // From logits: the same search as over x - row_lse, row by row; bf16 input too.
    BatchTrace lf(B, T, K, P), lb(B, T, K, P);
    CHECK(dbs_decode_batch_into_ex(h, x.data(), DBS_DTYPE_F32, 1, B, T, V, steps, nullptr, 1, &lf.out) == 0);
    CHECK(dbs_decode_batch_into_ex(h, xb.data(), DBS_DTYPE_BF16, 1, B, T, V, steps, nullptr, 2, &lb.out) == 0);
    CHECK(same_trace(lf, lb) && same_bits(lf.row_lse, lb.row_lse));
    std::vector<float> normalised(x.size());
    for (size_t r = 0; r < lf.row_lse.size(); ++r) {
        for (int v = 0; v < V; ++v) normalised[r * V + static_cast<size_t>(v)] = x[r * V + static_cast<size_t>(v)] - lf.row_lse[r];
    }
    BatchTrace n(B, T, K, P);
    CHECK(dbs_decode_batch_into_ex(h, normalised.data(), DBS_DTYPE_F32, 0, B, T, V, steps, nullptr, 1, &n.out) == 0);
    CHECK(same_trace(lf, n));
    CHECK(lf.row_lse[0] != 0.0f && std::isfinite(lf.row_lse[0]));  // step 0, beam 0 is always read

    // Pool outputs equal the result-handle API's (example 0 decodes all T steps).
    DBSResultHandle* r = nullptr;
    CHECK(dbs_decode(h, normalised.data(), T, V, &r) == 0);
    CHECK(dbs_result_pool_size(r) == P);
    const size_t tp = static_cast<size_t>(T) * P;
    CHECK(std::memcmp(dbs_result_pool_tokens(r), n.pool_tokens.data(), tp * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(dbs_result_pool_parents(r), n.pool_parents.data(), tp * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(dbs_result_pool_scores(r), n.pool_scores.data(), tp * sizeof(float)) == 0);
    dbs_free_result(r);
    // Padding past an example's steps.
    for (int t = steps[1]; t < T; ++t) {
        for (int p2 = 0; p2 < P; ++p2) CHECK(n.pool_parents[(static_cast<size_t>(T) + t) * P + p2] == -1);
    }

    // Errors.
    BatchTrace bad(B, T, K, P);
    bad.out.reserved[0] = &bad;
    CHECK(dbs_decode_batch_into_ex(h, x.data(), DBS_DTYPE_F32, 1, B, T, V, steps, nullptr, 1, &bad.out) == DBS_ERROR_INVALID_ARGUMENT);
    CHECK(dbs_decode_batch_into_ex(h, x.data(), 7, 1, B, T, V, steps, nullptr, 1, &lf.out) == DBS_ERROR_INVALID_ARGUMENT);
    std::vector<float> nan_row = x;
    nan_row[0] = std::numeric_limits<float>::quiet_NaN();
    const int rc = dbs_decode_batch_into_ex(h, nan_row.data(), DBS_DTYPE_F32, 1, B, T, V, steps, nullptr, 1, &lf.out);
    CHECK(rc == DBS_ERROR_INVALID_ARGUMENT && std::strstr(dbs_last_error(h), "logits contain NaN") != nullptr);
    dbs_destroy(h);

    DBSOptionsC no_pool = opt;
    no_pool.relaxed_pool_multiplier = 0;
    CHECK(dbs_create_ex(no_pool, &h) == 0);
    BatchTrace with_pool(B, T, K, P);
    CHECK(dbs_decode_batch_into_ex(h, x.data(), DBS_DTYPE_F32, 0, B, T, V, steps, nullptr, 1, &with_pool.out) ==
          DBS_ERROR_INVALID_ARGUMENT);
    dbs_destroy(h);
}

static void test_backward_batch_into_ex() {
    DBSOptionsC opt = test_options();
    opt.beam_size = 2;
    opt.eos_token = -1;
    opt.length_penalty_alpha = 0.6f;
    opt.relaxed_pool_multiplier = 2;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    const int B = 2, T = 3, K = 2, V = 5, P = K * 2;
    std::vector<float> x(static_cast<size_t>(B) * T * K * V);
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i * 104729 + 7) % 613) / 61.0f - 5.0f;
    BatchTrace tr(B, T, K, P);
    CHECK(dbs_decode_batch_into_ex(h, x.data(), DBS_DTYPE_F32, 1, B, T, V, nullptr, nullptr, 1, &tr.out) == 0);

    const size_t bk = static_cast<size_t>(B) * K, btk = bk * T, btp = static_cast<size_t>(B) * T * P;
    auto grads = [](size_t n, float scale) {
        std::vector<float> g(n);
        for (size_t i = 0; i < n; ++i) g[i] = scale * static_cast<float>(static_cast<int>(i % 7) - 3);
        return g;
    };
    const std::vector<float> gf = grads(bk, 0.5f), gfr = grads(bk, -0.25f), gs = grads(btk, 0.3f), gr = grads(btk, 0.2f),
                             gps = grads(btp, 0.1f), gpr = grads(btp, -0.15f);
    DBSBackwardInputsC in{};
    in.batch_size = B; in.steps = T; in.vocab_size = V; in.pool_size = P;
    in.parents = tr.parents.data(); in.tokens = tr.tokens.data(); in.lengths = tr.lengths.data();
    in.from_logprob = tr.from_logprob.data();
    in.pool_parents = tr.pool_parents.data(); in.pool_tokens = tr.pool_tokens.data();
    in.pool_lengths = tr.pool_lengths.data(); in.pool_from_logprob = tr.pool_from_logprob.data();

    // Final-score gradients of log-probs: exactly dbs_backward_batch_into.
    in.grad_final_scores = gf.data();
    std::vector<float> g1(x.size(), 0.0f), g2(x.size(), 0.0f);
    CHECK(dbs_backward_batch_into_ex(h, &in, 1, g1.data()) == 0);
    CHECK(dbs_backward_batch_into(h, B, T, V, nullptr, tr.parents.data(), tr.tokens.data(), tr.lengths.data(),
                                  tr.from_logprob.data(), gf.data(), 2, g2.data()) == 0);
    CHECK(same_bits(g1, g2));

    // Every output's gradient, through the logits: central differences.
    in.grad_final_raw_scores = gfr.data(); in.grad_scores = gs.data(); in.grad_raw_scores = gr.data();
    in.grad_pool_scores = gps.data(); in.grad_pool_raw_scores = gpr.data();
    in.logits = x.data(); in.logits_type = DBS_DTYPE_F32; in.row_lse = tr.row_lse.data();
    std::vector<float> grad(x.size(), 0.0f);
    CHECK(dbs_backward_batch_into_ex(h, &in, 2, grad.data()) == 0);
    auto loss = [&](const std::vector<float>& logits, BatchTrace& t) {
        CHECK(dbs_decode_batch_into_ex(h, logits.data(), DBS_DTYPE_F32, 1, B, T, V, nullptr, nullptr, 1, &t.out) == 0);
        double l = 0.0;
        for (size_t i = 0; i < bk; ++i) l += static_cast<double>(gf[i]) * t.final_scores[i] + static_cast<double>(gfr[i]) * t.final_raw[i];
        for (size_t i = 0; i < btk; ++i) {
            if (t.parents[i] >= 0) l += static_cast<double>(gs[i]) * t.scores[i] + static_cast<double>(gr[i]) * t.raw[i];
        }
        for (size_t i = 0; i < btp; ++i) {
            if (t.pool_parents[i] >= 0) l += static_cast<double>(gps[i]) * t.pool_scores[i] + static_cast<double>(gpr[i]) * t.pool_raw[i];
        }
        return l;
    };
    int checked = 0;
    const float eps = 1e-2f;
    for (size_t i = 0; i < x.size(); i += 3) {
        std::vector<float> xp = x, xm = x;
        xp[i] += eps;
        xm[i] -= eps;
        BatchTrace tp(B, T, K, P), tm(B, T, K, P);
        const double fd = (loss(xp, tp) - loss(xm, tm)) / (2.0 * eps);
        if (!same_selection(tp, tr) || !same_selection(tm, tr)) continue;  // the selection moved: not differentiable here
        CHECK(std::fabs(fd - grad[i]) <= 2e-3 * std::max(1.0, std::fabs(fd)));
        ++checked;
    }
    CHECK(checked > 10);

    // Errors.
    DBSBackwardInputsC bad = in;
    bad.row_lse = nullptr;
    CHECK(dbs_backward_batch_into_ex(h, &bad, 1, grad.data()) == DBS_ERROR_INVALID_ARGUMENT);
    bad = in;
    bad.pool_parents = nullptr;
    CHECK(dbs_backward_batch_into_ex(h, &bad, 1, grad.data()) == DBS_ERROR_INVALID_ARGUMENT);
    bad = in;
    bad.reserved[1] = &bad;
    CHECK(dbs_backward_batch_into_ex(h, &bad, 1, grad.data()) == DBS_ERROR_INVALID_ARGUMENT);
    bad = in;
    bad.logits_type = 9;
    CHECK(dbs_backward_batch_into_ex(h, &bad, 1, grad.data()) == DBS_ERROR_INVALID_ARGUMENT);
    std::vector<int32_t> corrupt = tr.pool_tokens;
    corrupt[1] = V;
    bad = in;
    bad.pool_tokens = corrupt.data();
    CHECK(dbs_backward_batch_into_ex(h, &bad, 1, grad.data()) == DBS_ERROR_INVALID_ARGUMENT);
    dbs_destroy(h);
}

// dbs_set_extra_eos_tokens: a second EOS token finishes hypotheses in every
// decode on the handle, is carried forward as itself, and needs eos_token >= 0.
static void test_extra_eos_tokens() {
    constexpr int B = 2, T = 3, K = 2, V = 5;
    DBSOptionsC opt{};
    opt.beam_size = K;
    opt.eos_token = 3;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == DBS_OK);
    std::vector<float> x(static_cast<size_t>(B) * T * K * V, -9.0f);
    for (int b = 0; b < B; ++b) {
        x[(static_cast<size_t>(b) * T * K) * V + 4] = -0.1f;  // step 0, beam 0: token 4 is best
        x[(static_cast<size_t>(b) * T * K) * V + 1] = -0.5f;
        for (int t = 1; t < T; ++t) {
            for (int k = 0; k < K; ++k) x[((static_cast<size_t>(b) * T + t) * K + k) * V + 2] = -0.2f;
        }
    }
    const auto decode = [&](std::vector<int32_t>& tokens, std::vector<uint8_t>& flp, std::vector<int32_t>& lengths) {
        std::vector<float> final_scores(B * K);
        tokens.assign(B * T * K, 0);
        flp.assign(B * T * K, 0);
        lengths.assign(B * T * K, 0);
        DBSDecodeOutputsC out{};
        out.final_scores = final_scores.data();
        out.tokens = tokens.data();
        out.from_logprob = flp.data();
        out.lengths = lengths.data();
        CHECK(dbs_decode_batch_into(h, x.data(), B, T, V, nullptr, nullptr, 1, &out) == DBS_OK);
    };
    std::vector<int32_t> tokens, lengths;
    std::vector<uint8_t> flp;
    decode(tokens, flp, lengths);
    CHECK(tokens[0] == 4 && lengths[K + 0] == 2);  // token 4 is not an EOS yet: the beam goes on

    const int32_t extra[] = {4};
    CHECK(dbs_set_extra_eos_tokens(h, extra, 1) == DBS_OK);
    decode(tokens, flp, lengths);
    for (int b = 0; b < B; ++b) {
        const size_t s0 = static_cast<size_t>(b) * T * K;
        CHECK(tokens[s0] == 4 && flp[s0] == 1);
        for (int t = 1; t < T; ++t) {
            bool carried = false;
            for (int k = 0; k < K; ++k) {
                const size_t i = s0 + static_cast<size_t>(t) * K + k;
                if (!flp[i] && lengths[i] == 1) {
                    carried = true;
                    CHECK(tokens[i] == 4);  // carried forward with the EOS it ended with
                }
            }
            CHECK(carried);
        }
    }
    CHECK(dbs_set_extra_eos_tokens(h, nullptr, 0) == DBS_OK);  // removed again
    decode(tokens, flp, lengths);
    CHECK(lengths[K + 0] == 2);

    CHECK(dbs_set_extra_eos_tokens(h, nullptr, 1) == DBS_ERROR_INVALID_ARGUMENT);
    const int32_t negative[] = {-4};
    CHECK(dbs_set_extra_eos_tokens(h, negative, 1) == DBS_ERROR_INVALID_ARGUMENT);
    const int32_t outside[] = {V};
    CHECK(dbs_set_extra_eos_tokens(h, outside, 1) == DBS_OK);  // the vocabulary is only known to the decode
    std::vector<float> final_scores(B * K);
    DBSDecodeOutputsC out{};
    out.final_scores = final_scores.data();
    CHECK(dbs_decode_batch_into(h, x.data(), B, T, V, nullptr, nullptr, 1, &out) == DBS_ERROR_INVALID_ARGUMENT);
    dbs_destroy(h);

    opt.eos_token = -1;  // extra EOS tokens need EOS handling
    CHECK(dbs_create_ex(opt, &h) == DBS_OK);
    CHECK(dbs_set_extra_eos_tokens(h, extra, 1) == DBS_ERROR_INVALID_ARGUMENT);
    dbs_destroy(h);
}

int main() {
    CHECK(dbs_abi_version() == DBS_ABI_VERSION);
    char expected_version[32];
    std::snprintf(expected_version, sizeof(expected_version), "%d.%d.%d", DBS_VERSION_MAJOR, DBS_VERSION_MINOR, DBS_VERSION_PATCH);
    CHECK(std::strcmp(dbs_version_string(), expected_version) == 0);
    test_deterministic_ties();
    test_extra_eos_tokens();
    test_constraints();
    test_invalid_nan_rejected();
    test_sparse_gradient_matches_finite_difference();
    test_relaxed_pool_gradient_closed_form();
    test_batch_decode();
    test_small_batches_run_on_one_thread();
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
    test_error_codes_match_the_header();
    test_every_failure_reports_its_own_error();
    test_deterministic_order_accepts_decoder_output();
    test_allocator_gauge_survives_reset();
    test_variable_beam_batch_backward();
    test_model_steps_ex_tracks_beam_prefixes();
    test_batch_into_matches_result_handles();
    test_decode_batch_into_ex();
    test_backward_batch_into_ex();
    test_validation_checks_every_element();
    std::cout << "dbs_tests passed\n";
    return 0;
}
