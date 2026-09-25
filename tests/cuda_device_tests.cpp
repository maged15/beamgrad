// SPDX-License-Identifier: MIT
//
// The native CUDA backend on a real GPU, checked against the CPU decoder in
// libdbs through the public C APIs (the emulation tests run the same kernels
// on the CPU; this test covers the compiled device code):
//   * every per-step output, every final score and the final-score gradient
//     must be bitwise identical, on randomized cases covering ties, -inf, EOS
//     carry-forward, min_length, length penalty, banned tokens, n-gram
//     blocking, repetition penalty, per-example steps, both scan kernels,
//     multi-level reductions and the maximum beam size;
//   * dbs_cuda_decode_step, stepped through the same rows, must select exactly
//     what dbs_cuda_decode selects;
//   * the NaN/+inf flags must point at the examples that have them.
// Exits with 77 (skipped) when no CUDA device is available.
#include "dbs.h"
#include "dbs_cuda.h"

#include "check.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

#define CUDA_CHECK(expr) CHECK((expr) == cudaSuccess)

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) { CUDA_CHECK(cudaMalloc(&ptr_, sizeof(T) * (count ? count : 1))); }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), sizeof(T) * count_, cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    std::vector<T> to_host() const {
        std::vector<T> host(count_);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, sizeof(T) * count_, cudaMemcpyDeviceToHost));
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_;
};

struct Case {
    int B, T, K, V;
    int eos;
    int min_length;
    float alpha;
    int ngram;
    float penalty;
    bool ties;
    bool banned;
    bool variable_steps;
    bool user_workspace;
};

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// Decodes one case on both backends; returns the number of mismatching outputs.
int run_case(const Case& c, unsigned seed) {
    std::mt19937 rng(seed);
    const size_t n = static_cast<size_t>(c.B) * c.T * c.K * c.V;
    const size_t tk = static_cast<size_t>(c.B) * c.T * c.K;
    const size_t bk = static_cast<size_t>(c.B) * c.K;
    std::vector<float> x(n);
    std::normal_distribution<float> normal(0.0f, 2.0f);
    std::uniform_int_distribution<int> level(0, 3);
    for (float& v : x) v = c.ties ? -0.5f * static_cast<float>(level(rng)) : normal(rng) - 10.0f;
    for (size_t i = 0; i < n; i += 1 + rng() % 97) x[i] = -std::numeric_limits<float>::infinity();
    std::vector<uint8_t> banned(static_cast<size_t>(c.V), 0);
    if (c.banned) {
        for (uint8_t& b : banned) b = rng() % 11 == 0;
    }
    std::vector<int32_t> steps(static_cast<size_t>(c.B), c.T);
    if (c.variable_steps) {
        for (int32_t& s : steps) s = 1 + static_cast<int32_t>(rng() % c.T);
    }

    // CPU reference.
    DBSOptionsC opt{};
    opt.beam_size = c.K;
    opt.eos_token = c.eos;
    opt.min_length = c.min_length;
    opt.length_penalty_alpha = c.alpha;
    opt.validate_inputs = 1;
    DBSDecoderHandle* handle = nullptr;
    CHECK(dbs_create_ex(opt, &handle) == DBS_OK);
    DBSAdvancedConstraintsC constraints{};
    constraints.min_length = -1;
    constraints.no_repeat_ngram_size = c.ngram;
    constraints.repetition_penalty = c.penalty;
    constraints.banned_tokens = c.banned ? banned.data() : nullptr;
    std::vector<float> final_scores(bk), final_raw(bk), scores(tk), raw_scores(tk);
    std::vector<int32_t> final_lengths(bk), tokens(tk), parents(tk), lengths(tk);
    std::vector<uint8_t> from_logprob(tk);
    const DBSDecodeOutputsC cpu_out{final_scores.data(), final_raw.data(), final_lengths.data(), tokens.data(),
                                    parents.data(),      lengths.data(),   scores.data(),        raw_scores.data(),
                                    from_logprob.data()};
    CHECK(dbs_decode_batch_into(handle, x.data(), c.B, c.T, c.V, c.variable_steps ? steps.data() : nullptr, &constraints,
                                0, &cpu_out) == DBS_OK);

    // GPU.
    const DeviceBuffer<float> d_x(x);
    const DeviceBuffer<uint8_t> d_banned(banned);
    const DeviceBuffer<int32_t> d_steps(steps);
    const DeviceBuffer<float> d_final(bk), d_final_raw(bk), d_scores(tk), d_raw(tk);
    const DeviceBuffer<int32_t> d_final_len(bk), d_tokens(tk), d_parents(tk), d_lengths(tk);
    const DeviceBuffer<uint8_t> d_from_logprob(tk), d_invalid(static_cast<size_t>(c.B));
    DBSCudaDecodeArgs args{};
    args.batch_size = c.B;
    args.steps = c.T;
    args.beam_size = c.K;
    args.vocab_size = c.V;
    args.eos_token = c.eos;
    args.min_length = c.min_length;
    args.length_penalty_alpha = c.alpha;
    args.no_repeat_ngram_size = c.ngram;
    args.repetition_penalty = c.penalty;
    args.steps_per_example = c.variable_steps ? d_steps.get() : nullptr;
    args.banned_tokens = c.banned ? d_banned.get() : nullptr;
    const DBSCudaDecodeOutputs gpu_out{d_final.get(),   d_final_raw.get(), d_final_len.get(),
                                       d_tokens.get(),  d_parents.get(),   d_lengths.get(),
                                       d_scores.get(),  d_raw.get(),       d_from_logprob.get(),
                                       d_invalid.get()};
    const int64_t workspace_bytes = dbs_cuda_decode_workspace_size(&args);
    CHECK(workspace_bytes >= 0);
    const DeviceBuffer<uint8_t> workspace(c.user_workspace ? static_cast<size_t>(workspace_bytes) : 0);
    CHECK(dbs_cuda_decode(d_x.get(), &args, &gpu_out, c.user_workspace ? workspace.get() : nullptr,
                          c.user_workspace ? workspace_bytes : 0, nullptr) == DBS_CUDA_STATUS_OK);
    CUDA_CHECK(cudaDeviceSynchronize());

    int mismatches = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "  mismatch: %s\n", what);
            ++mismatches;
        }
    };
    expect(same_bits(tokens, d_tokens.to_host()), "tokens");
    expect(same_bits(parents, d_parents.to_host()), "parents");
    expect(same_bits(lengths, d_lengths.to_host()), "lengths");
    expect(same_bits(from_logprob, d_from_logprob.to_host()), "from_logprob");
    expect(same_bits(scores, d_scores.to_host()), "scores");
    expect(same_bits(raw_scores, d_raw.to_host()), "raw_scores");
    expect(same_bits(final_scores, d_final.to_host()), "final_scores");
    expect(same_bits(final_raw, d_final_raw.to_host()), "final_raw_scores");
    expect(same_bits(final_lengths, d_final_len.to_host()), "final_lengths");
    expect(same_bits(std::vector<uint8_t>(static_cast<size_t>(c.B), 0), d_invalid.to_host()), "invalid_input flags");

    // Final-score gradient.
    std::vector<float> grad_final(bk);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    for (float& g : grad_final) g = unit(rng);
    std::vector<float> grad(n, 0.0f);
    CHECK(dbs_backward_batch_into(handle, c.B, c.T, c.V, c.variable_steps ? steps.data() : nullptr, parents.data(),
                                  tokens.data(), lengths.data(), from_logprob.data(), grad_final.data(), 0,
                                  grad.data()) == DBS_OK);
    const DeviceBuffer<float> d_grad_final(grad_final);
    const DeviceBuffer<float> d_grad(std::vector<float>(n, 0.0f));
    CHECK(dbs_cuda_backward(&args, d_parents.get(), d_tokens.get(), d_lengths.get(), d_from_logprob.get(),
                            d_grad_final.get(), d_grad.get(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect(same_bits(grad, d_grad.to_host()), "gradient");

    dbs_destroy(handle);
    return mismatches;
}

int test_randomized_parity() {
    const int beams[] = {1, 2, 3, 4, 5, 8, 13, 16, 17, 33, 64, 128, 1024};
    const int vocabs[] = {2, 7, 31, 256, 1000, 5003, 32000, 50257};
    int failures = 0;
    for (unsigned seed = 0; seed < 240; ++seed) {
        std::mt19937 rng(seed * 7919u + 1u);
        Case c{};
        c.K = beams[rng() % (sizeof(beams) / sizeof(beams[0]))];
        c.V = vocabs[rng() % (sizeof(vocabs) / sizeof(vocabs[0]))];
        if (static_cast<int64_t>(c.K) * c.V > 2000000) c.V = 2000000 / c.K;
        c.B = 1 + static_cast<int>(rng() % 4);
        c.T = 1 + static_cast<int>(rng() % 8);
        c.eos = rng() % 3 == 0 ? -1 : static_cast<int>(rng() % c.V);
        c.min_length = static_cast<int>(rng() % 4);
        c.alpha = rng() % 2 ? 0.0f : 0.3f * static_cast<float>(rng() % 5);
        c.ngram = rng() % 3 == 0 ? 1 + static_cast<int>(rng() % 3) : 0;
        c.penalty = rng() % 3 == 0 ? 1.0f + 0.5f * static_cast<float>(1 + rng() % 3) : 1.0f;
        c.ties = rng() % 2 == 0;
        c.banned = rng() % 3 == 0;
        c.variable_steps = rng() % 3 == 0;
        c.user_workspace = rng() % 2 == 0;
        if (run_case(c, seed) != 0) {
            std::fprintf(stderr, "case %u failed: B=%d T=%d K=%d V=%d eos=%d min_length=%d alpha=%g ngram=%d penalty=%g\n",
                         seed, c.B, c.T, c.K, c.V, c.eos, c.min_length, c.alpha, c.ngram, c.penalty);
            ++failures;
        }
    }
    return failures;
}

// dbs_cuda_decode_step, driven through the rows of a tensor with the prefixes
// tracked by the caller, must select exactly what dbs_cuda_decode selects.
int test_step_api_parity() {
    int failures = 0;
    for (unsigned seed = 0; seed < 40; ++seed) {
        std::mt19937 rng(seed + 1000u);
        const int beams[] = {1, 3, 4, 16, 17, 64};
        const int B = 1 + static_cast<int>(rng() % 3), T = 1 + static_cast<int>(rng() % 7);
        const int K = beams[rng() % 6], V = 5 + static_cast<int>(rng() % 3000);
        const size_t BK = static_cast<size_t>(B) * K, n = BK * T * V;
        std::vector<float> x(n);
        std::uniform_int_distribution<int> level(0, 5);
        for (float& v : x) v = -0.5f * static_cast<float>(level(rng));
        std::vector<uint8_t> banned(static_cast<size_t>(V), 0);
        for (uint8_t& b : banned) b = rng() % 9 == 0;

        DBSCudaDecodeArgs args{};
        args.batch_size = B;
        args.steps = T;
        args.beam_size = K;
        args.vocab_size = V;
        args.eos_token = static_cast<int>(rng() % V);
        args.min_length = static_cast<int>(rng() % 3);
        args.length_penalty_alpha = 0.6f;
        args.no_repeat_ngram_size = static_cast<int>(rng() % 3);
        args.repetition_penalty = rng() % 2 ? 1.5f : 1.0f;
        const DeviceBuffer<uint8_t> d_banned(banned);
        args.banned_tokens = rng() % 2 ? d_banned.get() : nullptr;

        // Full decode.
        const DeviceBuffer<float> d_x(x);
        const size_t tk = BK * T;
        const DeviceBuffer<float> f_final(BK), f_scores(tk), f_raw(tk);
        const DeviceBuffer<int32_t> f_tokens(tk), f_parents(tk), f_lengths(tk);
        DBSCudaDecodeOutputs full{};
        full.final_scores = f_final.get();
        full.tokens = f_tokens.get();
        full.parents = f_parents.get();
        full.lengths = f_lengths.get();
        full.scores = f_scores.get();
        full.raw_scores = f_raw.get();
        CHECK(dbs_cuda_decode(d_x.get(), &args, &full, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
        const std::vector<int32_t> tokens = f_tokens.to_host(), parents = f_parents.to_host(), lengths = f_lengths.to_host();
        const std::vector<float> scores = f_scores.to_host(), raw_scores = f_raw.to_host(), final_scores = f_final.to_host();

        // Stepped.
        DBSCudaDecodeArgs step_args = args;
        step_args.steps = 1;
        std::vector<float> raw0(BK, -std::numeric_limits<float>::infinity());
        for (int b = 0; b < B; ++b) raw0[static_cast<size_t>(b) * K] = 0.0f;
        const DeviceBuffer<float> s_raw(raw0);
        const DeviceBuffer<int32_t> s_len(std::vector<int32_t>(BK, 0));
        const DeviceBuffer<uint8_t> s_finished(std::vector<uint8_t>(BK, 0));
        const DeviceBuffer<float> o_scores(BK), o_raw(BK), o_final(BK);
        const DeviceBuffer<int32_t> o_tokens(BK), o_parents(BK), o_lengths(BK);
        const DeviceBuffer<uint8_t> o_flp(BK);
        std::vector<int32_t> paths(BK * T, -1), next(BK * T, -1);
        bool ok = true;
        for (int t = 0; t < T && ok; ++t) {
            std::vector<float> rows(BK * V);
            for (int b = 0; b < B; ++b) {
                std::memcpy(&rows[static_cast<size_t>(b) * K * V], &x[(static_cast<size_t>(b) * T + t) * K * V],
                            sizeof(float) * K * V);
            }
            const DeviceBuffer<float> d_rows(rows);
            const DeviceBuffer<int32_t> d_paths(paths);
            DBSCudaDecodeOutputs out{};
            out.tokens = o_tokens.get();
            out.parents = o_parents.get();
            out.lengths = o_lengths.get();
            out.scores = o_scores.get();
            out.raw_scores = o_raw.get();
            out.from_logprob = o_flp.get();
            out.final_scores = t == T - 1 ? o_final.get() : nullptr;
            const DBSCudaBeamState state{s_raw.get(), s_len.get(), s_finished.get(), d_paths.get(), T, 0};
            const int64_t ws_bytes = dbs_cuda_decode_step_workspace_size(&step_args, T);
            CHECK(ws_bytes >= 0);
            const DeviceBuffer<uint8_t> ws(static_cast<size_t>(ws_bytes));
            CHECK(dbs_cuda_decode_step(d_rows.get(), &step_args, &state, &out, ws.get(), ws_bytes, nullptr) ==
                  DBS_CUDA_STATUS_OK);
            const std::vector<int32_t> tok = o_tokens.to_host(), par = o_parents.to_host(), len = o_lengths.to_host();
            const std::vector<float> sc = o_scores.to_host(), rs = o_raw.to_host();
            const std::vector<uint8_t> flp = o_flp.to_host();
            for (int b = 0; b < B; ++b) {
                for (int k = 0; k < K; ++k) {
                    const size_t i = static_cast<size_t>(b) * K + k;
                    const size_t f = (static_cast<size_t>(b) * T + t) * K + k;
                    ok = ok && tok[i] == tokens[f] && par[i] == parents[f] && len[i] == lengths[f] &&
                         std::memcmp(&sc[i], &scores[f], 4) == 0 && std::memcmp(&rs[i], &raw_scores[f], 4) == 0;
                    int32_t* dst = &next[i * T];
                    if (par[i] < 0) {
                        std::fill(dst, dst + T, -1);
                    } else {
                        const int32_t* src = &paths[(static_cast<size_t>(b) * K + par[i]) * T];
                        std::copy(src, src + T, dst);
                        if (flp[i]) dst[t] = tok[i];
                    }
                }
            }
            paths.swap(next);
        }
        ok = ok && same_bits(o_final.to_host(), final_scores);
        if (!ok) {
            std::fprintf(stderr, "step API case %u differs from the full decode: B=%d T=%d K=%d V=%d\n", seed, B, T, K, V);
            ++failures;
        }
    }
    return failures;
}

int test_invalid_input_flags() {
    const int B = 3, T = 4, K = 4, V = 1000;
    std::vector<float> x(static_cast<size_t>(B) * T * K * V, -1.0f);
    const auto at = [&](int b, int t, int k, int v) { return ((static_cast<size_t>(b) * T + t) * K + k) * V + v; };
    x[at(1, 2, 0, 17)] = std::numeric_limits<float>::quiet_NaN();  // a live row of example 1
    x[at(2, 0, 3, 5)] = std::numeric_limits<float>::infinity();     // beam 3 is not live at step 0
    const DeviceBuffer<float> d_x(x);
    const DeviceBuffer<float> d_final(static_cast<size_t>(B) * K);
    const DeviceBuffer<uint8_t> d_invalid(static_cast<size_t>(B));
    DBSCudaDecodeArgs args{};
    args.batch_size = B;
    args.steps = T;
    args.beam_size = K;
    args.vocab_size = V;
    args.eos_token = -1;
    DBSCudaDecodeOutputs out{};
    out.final_scores = d_final.get();
    out.invalid_input = d_invalid.get();
    CHECK(dbs_cuda_decode(d_x.get(), &args, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
    const std::vector<uint8_t> flags = d_invalid.to_host();
    const bool ok = flags[0] == 0 && flags[1] == 1 && flags[2] == 0;
    if (!ok) std::fprintf(stderr, "invalid_input flags: %d %d %d (expected 0 1 0)\n", flags[0], flags[1], flags[2]);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    if (!dbs_cuda_available()) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s\n", prop.name);
    const int failures = test_randomized_parity() + test_step_api_parity() + test_invalid_input_flags();
    if (failures != 0) {
        std::fprintf(stderr, "dbs_cuda_device_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("dbs_cuda_device_tests passed\n");
    return 0;
}
