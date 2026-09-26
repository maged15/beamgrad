// SPDX-License-Identifier: MIT
//
// Runs the real CUDA backend source (cuda/dbs_cuda.cu) on the CPU through the
// emulation layer and checks it against the CPU decoder in libdbs:
//   * every per-step output (tokens, parents, lengths, scores, raw scores,
//     carry-forward flags) and every final score must be bitwise identical;
//   * the NaN/+inf flags must match the CPU decoder's input validation;
//   * the CUDA backward must reproduce dbs_backward_sparse() exactly;
//   * the _ex functions (fp16/bf16 rows, logits, gradients of every score
//     output) must reproduce dbs_decode_batch_into_ex() and
//     dbs_backward_batch_into_ex() exactly.
// Cases are randomized but seeded, and cover ties, -inf, NaN and +inf entries,
// EOS carry-forward, min_length, length penalty, banned tokens, n-gram
// blocking, repetition penalty, variable batches, both scan kernels,
// multi-level tile reduction, and the maximum beam size. Every case runs with
// the block's threads scheduled forward, in reverse and shuffled, so a data
// race between barriers shows up as a mismatch.
#define DBS_CUDA_EMULATION 1
#include "../cuda/dbs_cuda.cu"

#include "check.hpp"
#include "dbs.h"

#include <cinttypes>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    int B, T, K, V;
    int eos;
    int min_length;
    float alpha;
    bool variable;
    int value_mode;  // 0 continuous, 1 heavy ties, 2 sparse -inf, 3 ties with a few NaN/+inf, 4 many NaN/+inf
    int ngram = 0;
    float penalty = 0.0f;
    bool banned = false;
};

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

struct GpuResult {
    std::vector<float> final_scores, final_raw;
    std::vector<int32_t> final_len, tokens, parents, lengths;
    std::vector<float> scores, raw_scores;
    std::vector<uint8_t> from_logprob;
    std::vector<float> row_lse;  // [B, T, K] when decoded from logits
};

size_t dtype_bytes(int dtype) { return dtype == DBS_CUDA_DTYPE_F32 ? 4 : 2; }

std::vector<float> make_log_probs(const Case& c, std::mt19937& rng) {
    std::vector<float> x(static_cast<size_t>(c.B) * c.T * c.K * c.V);
    std::uniform_real_distribution<float> u(-9.0f, 0.0f);
    std::uniform_int_distribution<int> q(0, 3);
    std::uniform_int_distribution<int> coin(0, 9);
    std::uniform_int_distribution<int> rare(0, 199);
    for (float& v : x) {
        switch (c.value_mode) {
            case 1: v = -0.5f * static_cast<float>(q(rng)); break;
            case 2: v = coin(rng) < 3 ? -std::numeric_limits<float>::infinity() : u(rng); break;
            case 3:
            case 4: {
                const int r = c.value_mode == 4 ? rare(rng) % 12 : rare(rng);
                v = r == 0 ? std::numeric_limits<float>::quiet_NaN()
                  : r == 1 ? std::numeric_limits<float>::infinity()
                  : -0.5f * static_cast<float>(q(rng));
                break;
            }
            default: v = u(rng); break;
        }
    }
    return x;
}

// Drives dbs_cuda_decode_step_ex through the same rows ([B, T, K, V] of
// dtype), one step at a time, with the prefixes tracked here as a caller
// would; every step output, the final scores, the NaN/+inf flags and (from
// logits) the rows' logsumexp must equal the full decode's.
int check_step_api(const Case& c, const void* x, int dtype, int from_logits, DBSCudaDecodeArgs args,
                   const GpuResult& g, const std::vector<uint8_t>& invalid_full,
                   const DBSCudaSearchOptions* search = nullptr) {
    args.steps = 1;
    const size_t BK = static_cast<size_t>(c.B) * c.K;
    const size_t V = static_cast<size_t>(c.V);
    const size_t elem = dtype_bytes(dtype);
    std::vector<float> raw(BK, -std::numeric_limits<float>::infinity());
    std::vector<int32_t> len(BK, 0);
    std::vector<uint8_t> finished(BK, 0);
    for (int b = 0; b < c.B; ++b) raw[static_cast<size_t>(b) * c.K] = 0.0f;
    std::vector<int32_t> paths(BK * c.T, -1), next_paths(BK * c.T, -1);
    std::vector<unsigned char> rows(BK * V * elem);
    std::vector<int32_t> tokens(BK), parents(BK), lengths(BK), final_len(BK);
    std::vector<float> scores(BK), raw_scores(BK), final_scores(BK), final_raw(BK), row_lse(BK, 7.0f);
    std::vector<uint8_t> from_logprob(BK), invalid(static_cast<size_t>(c.B)), any_invalid(static_cast<size_t>(c.B), 0);
    const unsigned char* bytes = static_cast<const unsigned char*>(x);
    int mismatches = 0;
    for (int t = 0; t < c.T; ++t) {
        for (int b = 0; b < c.B; ++b) {
            std::memcpy(&rows[static_cast<size_t>(b) * c.K * V * elem],
                        bytes + (static_cast<size_t>(b) * c.T + t) * c.K * V * elem, elem * c.K * V);
        }
        DBSCudaDecodeOutputs out{};
        out.tokens = tokens.data();
        out.parents = parents.data();
        out.lengths = lengths.data();
        out.scores = scores.data();
        out.raw_scores = raw_scores.data();
        out.from_logprob = from_logprob.data();
        out.invalid_input = invalid.data();
        const bool last = t == c.T - 1;
        out.final_scores = last ? final_scores.data() : nullptr;
        out.final_raw_scores = last ? final_raw.data() : nullptr;
        out.final_lengths = last ? final_len.data() : nullptr;
        const DBSCudaBeamState state{raw.data(), len.data(), finished.data(), paths.data(), c.T, 0};
        if (search) {
            CHECK(dbs_cuda_decode_step_ex2(rows.data(), dtype, from_logits, &args, search, &state, &out, row_lse.data(),
                                           nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
        } else if (dtype == DBS_CUDA_DTYPE_F32 && !from_logits && t % 2 == 0) {
            CHECK(dbs_cuda_decode_step(reinterpret_cast<const float*>(rows.data()), &args, &state, &out, nullptr, 0,
                                       nullptr) == DBS_CUDA_STATUS_OK);
            std::fill(row_lse.begin(), row_lse.end(), 0.0f);
        } else {
            CHECK(dbs_cuda_decode_step_ex(rows.data(), dtype, from_logits, &args, &state, &out, row_lse.data(), nullptr,
                                          0, nullptr) == DBS_CUDA_STATUS_OK);
        }
        for (int b = 0; b < c.B; ++b) {
            any_invalid[static_cast<size_t>(b)] |= invalid[static_cast<size_t>(b)];
            for (int k = 0; k < c.K; ++k) {
                const size_t i = static_cast<size_t>(b) * c.K + k;
                const size_t gs = (static_cast<size_t>(b) * c.T + t) * c.K + k;
                mismatches += tokens[i] != g.tokens[gs] || parents[i] != g.parents[gs] || lengths[i] != g.lengths[gs] ||
                              from_logprob[i] != g.from_logprob[gs] || !same_bits(scores[i], g.scores[gs]) ||
                              !same_bits(raw_scores[i], g.raw_scores[gs]);
                mismatches += !same_bits(row_lse[i], g.row_lse.empty() ? 0.0f : g.row_lse[gs]);
                // The new hypothesis: its parent's tokens, plus the one it emitted.
                int32_t* dst = &next_paths[i * c.T];
                if (parents[i] < 0) {
                    std::fill(dst, dst + c.T, -1);
                } else {
                    const int32_t* src = &paths[(static_cast<size_t>(b) * c.K + parents[i]) * c.T];
                    std::copy(src, src + c.T, dst);
                    if (from_logprob[i]) dst[t] = tokens[i];
                }
            }
        }
        paths.swap(next_paths);
    }
    for (size_t i = 0; i < BK; ++i) {
        mismatches += !same_bits(final_scores[i], g.final_scores[i]) || !same_bits(final_raw[i], g.final_raw[i]) ||
                      final_len[i] != g.final_len[i];
    }
    for (int b = 0; b < c.B; ++b) mismatches += any_invalid[static_cast<size_t>(b)] != invalid_full[static_cast<size_t>(b)];
    if (mismatches) {
        std::fprintf(stderr, "  step API mismatch (B=%d T=%d K=%d V=%d eos=%d ngram=%d penalty=%g banned=%d)\n", c.B, c.T,
                     c.K, c.V, c.eos, c.ngram, c.penalty, c.banned);
    }
    return mismatches;
}

int run_case(const Case& c, uint32_t seed) {
    std::mt19937 rng(seed);
    const std::vector<float> x = make_log_probs(c, rng);
    const size_t BK = static_cast<size_t>(c.B) * c.K;
    const size_t BTK = static_cast<size_t>(c.B) * c.T * c.K;

    std::vector<int32_t> steps(c.B, c.T), beams(c.B, c.K), eos(c.B, c.eos), min_len(c.B, c.min_length);
    if (c.variable) {
        for (int b = 0; b < c.B; ++b) {
            steps[b] = 1 + static_cast<int>(rng() % c.T);
            beams[b] = 1 + static_cast<int>(rng() % c.K);
            eos[b] = (rng() % 3 == 0) ? -1 : static_cast<int>(rng() % c.V);
            min_len[b] = static_cast<int>(rng() % 3);
        }
    }

    std::vector<uint8_t> banned(static_cast<size_t>(c.V), 0);
    for (uint8_t& v : banned) v = rng() % (c.value_mode == 4 ? 2 : 6) == 0 ? 1 : 0;

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
    args.banned_tokens = c.banned ? banned.data() : nullptr;
    if (c.variable) {
        args.steps_per_example = steps.data();
        args.beam_sizes_per_example = beams.data();
        args.eos_tokens_per_example = eos.data();
        args.min_lengths_per_example = min_len.data();
    }

    GpuResult g;
    g.final_scores.assign(BK, 1.0f);
    g.final_raw.assign(BK, 1.0f);
    g.final_len.assign(BK, 7);
    g.tokens.assign(BTK, 7);
    g.parents.assign(BTK, 7);
    g.lengths.assign(BTK, 7);
    g.scores.assign(BTK, 1.0f);
    g.raw_scores.assign(BTK, 1.0f);
    g.from_logprob.assign(BTK, 7);
    std::vector<uint8_t> invalid(static_cast<size_t>(c.B), 7);
    DBSCudaDecodeOutputs out{};
    out.invalid_input = invalid.data();
    out.final_scores = g.final_scores.data();
    out.final_raw_scores = g.final_raw.data();
    out.final_lengths = g.final_len.data();
    out.tokens = g.tokens.data();
    out.parents = g.parents.data();
    out.lengths = g.lengths.data();
    out.scores = g.scores.data();
    out.raw_scores = g.raw_scores.data();
    out.from_logprob = g.from_logprob.data();
    CHECK(dbs_cuda_decode(x.data(), &args, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);

    std::vector<float> grad_final(BK);
    std::uniform_real_distribution<float> gd(-2.0f, 2.0f);
    for (float& v : grad_final) v = gd(rng);
    std::vector<float> grad(x.size(), 0.0f);
    CHECK(dbs_cuda_backward(&args, g.parents.data(), g.tokens.data(), g.lengths.data(), g.from_logprob.data(),
                            grad_final.data(), grad.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
    // The per-slot path gradient scatters to exactly the dense gradient.
    std::vector<float> draws(BTK, 7.0f);
    CHECK(dbs_cuda_path_gradient(&args, g.parents.data(), g.tokens.data(), g.lengths.data(), g.from_logprob.data(),
                                 grad_final.data(), draws.data(), nullptr) == DBS_CUDA_STATUS_OK);
    std::vector<float> scattered(x.size(), 0.0f);
    for (size_t s = 0; s < BTK; ++s) {
        if (g.parents[s] < 0 || !g.from_logprob[s]) continue;
        const size_t bt = s / static_cast<size_t>(c.K);
        scattered[(bt * c.K + static_cast<size_t>(g.parents[s])) * c.V + static_cast<size_t>(g.tokens[s])] += draws[s];
    }
    const bool path_ok = std::memcmp(scattered.data(), grad.data(), grad.size() * sizeof(float)) == 0;

    int mismatches = 0;
    auto report = [&](const char* what, int b, int t, int k) {
        if (mismatches++ < 5) {
            std::fprintf(stderr, "  mismatch %s b=%d t=%d k=%d (B=%d T=%d K=%d V=%d eos=%d minlen=%d alpha=%g var=%d mode=%d ngram=%d penalty=%g banned=%d)\n",
                         what, b, t, k, c.B, c.T, c.K, c.V, c.eos, c.min_length, c.alpha, c.variable, c.value_mode,
                         c.ngram, c.penalty, c.banned);
        }
    };

    for (int b = 0; b < c.B; ++b) {
        const int Tb = steps[b], Kb = beams[b];
        DBSOptionsC opt{};
        opt.beam_size = Kb;
        opt.eos_token = eos[b];
        opt.min_length = min_len[b];
        opt.length_penalty_alpha = c.alpha;
        opt.validate_inputs = 0;
        DBSDecoderHandle* h = nullptr;
        CHECK(dbs_create_ex(opt, &h) == 0);

        // CPU decodes a dense [Tb, Kb, V] copy of this example.
        std::vector<float> local(static_cast<size_t>(Tb) * Kb * c.V);
        for (int t = 0; t < Tb; ++t)
            for (int k = 0; k < Kb; ++k)
                std::memcpy(&local[(static_cast<size_t>(t) * Kb + k) * c.V],
                            &x[((static_cast<size_t>(b) * c.T + t) * c.K + k) * c.V], sizeof(float) * c.V);
        DBSAdvancedConstraintsC constraints{};
        constraints.min_length = -1;
        constraints.banned_tokens = c.banned ? banned.data() : nullptr;
        constraints.no_repeat_ngram_size = c.ngram;
        constraints.repetition_penalty = c.penalty;
        DBSResultHandle* r = nullptr;
        CHECK(dbs_decode_constrained_ex(h, local.data(), Tb, c.V, &constraints, &r) == 0);

        // The NaN/+inf flag matches the CPU decoder's validation of the rows it reads.
        DBSOptionsC strict = opt;
        strict.validate_inputs = 1;
        DBSDecoderHandle* hs = nullptr;
        CHECK(dbs_create_ex(strict, &hs) == 0);
        DBSResultHandle* rs = nullptr;
        const int rc_strict = dbs_decode_constrained_ex(hs, local.data(), Tb, c.V, &constraints, &rs);
        if (rs) dbs_free_result(rs);
        dbs_destroy(hs);
        if ((rc_strict != 0) != (invalid[static_cast<size_t>(b)] != 0)) report("invalid flag", b, -1, -1);

        const int32_t* tok = dbs_result_tokens(r);
        const int32_t* par = dbs_result_parents(r);
        const int32_t* len = dbs_result_lengths(r);
        const float* sc = dbs_result_scores(r);
        const float* raw = dbs_result_raw_scores(r);
        const float* fs = dbs_result_final_scores(r);
        const float* fr = dbs_result_final_raw_scores(r);

        for (int t = 0; t < c.T; ++t) {
            for (int k = 0; k < c.K; ++k) {
                const size_t gs = (static_cast<size_t>(b) * c.T + t) * c.K + k;
                if (t >= Tb || k >= Kb) {
                    if (g.tokens[gs] != -1 || g.parents[gs] != -1 || g.lengths[gs] != 0 || g.from_logprob[gs] != 0 ||
                        !std::isinf(g.scores[gs]) || !std::isinf(g.raw_scores[gs])) report("padding", b, t, k);
                    continue;
                }
                const size_t cs = static_cast<size_t>(t) * Kb + k;
                if (g.tokens[gs] != tok[cs]) report("token", b, t, k);
                if (g.parents[gs] != par[cs]) report("parent", b, t, k);
                if (g.lengths[gs] != len[cs]) report("length", b, t, k);
                if (!same_bits(g.scores[gs], sc[cs])) report("score", b, t, k);
                if (!same_bits(g.raw_scores[gs], raw[cs])) report("raw_score", b, t, k);
                // A selected slot is a carry-forward iff its parent had already emitted EOS.
                const bool carry = t > 0 && par[cs] >= 0 && eos[b] >= 0 &&
                                   tok[static_cast<size_t>(t - 1) * Kb + par[cs]] == eos[b];
                const uint8_t expected_flp = par[cs] >= 0 && !carry ? 1 : 0;
                if (g.from_logprob[gs] != expected_flp) report("from_logprob", b, t, k);
            }
        }
        for (int k = 0; k < c.K; ++k) {
            const size_t gs = static_cast<size_t>(b) * c.K + k;
            if (k >= Kb) {
                if (!std::isinf(g.final_scores[gs]) || g.final_len[gs] != 0) report("final padding", b, -1, k);
                continue;
            }
            if (!same_bits(g.final_scores[gs], fs[k])) report("final_score", b, -1, k);
            if (!same_bits(g.final_raw[gs], fr[k])) report("final_raw", b, -1, k);
            if (g.final_len[gs] != len[static_cast<size_t>(Tb - 1) * Kb + k]) report("final_length", b, -1, k);
        }

        // Backward parity.
        DBSBackwardHandle* bw = nullptr;
        CHECK(dbs_backward_sparse(h, r, nullptr, nullptr, &grad_final[static_cast<size_t>(b) * c.K], &bw) == 0);
        std::vector<float> expected(static_cast<size_t>(c.T) * c.K * c.V, 0.0f);
        const int64_t nnz = dbs_backward_sparse_logprob_count(bw);
        const int64_t* idx = dbs_backward_sparse_logprob_indices(bw);
        const float* val = dbs_backward_sparse_logprob_values(bw);
        for (int64_t i = 0; i < nnz; ++i) {
            const int64_t flat = idx[i];
            const int64_t v = flat % c.V;
            const int64_t kk = (flat / c.V) % Kb;
            const int64_t t = flat / (static_cast<int64_t>(c.V) * Kb);
            expected[(static_cast<size_t>(t) * c.K + kk) * c.V + v] += val[i];
        }
        const float* got = &grad[static_cast<size_t>(b) * c.T * c.K * c.V];
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!same_bits(got[i], expected[i])) {
                report("grad", b, static_cast<int>(i / (static_cast<size_t>(c.K) * c.V)), static_cast<int>((i / c.V) % c.K));
                break;
            }
        }
        dbs_free_backward(bw);
        dbs_free_result(r);
        dbs_destroy(h);
    }
    if (!path_ok) {
        std::fprintf(stderr, "  path gradient differs from the dense gradient (B=%d T=%d K=%d V=%d)\n", c.B, c.T, c.K, c.V);
        ++mismatches;
    }
    // The step interface has no per-example step counts.
    if (!c.variable) mismatches += check_step_api(c, x.data(), DBS_CUDA_DTYPE_F32, 0, args, g, invalid);
    return mismatches;
}

uint16_t to_bf16_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

// Truncating float -> fp16 (NaN stays NaN; tiny values flush to zero): any
// encoding will do, the CPU and the GPU read the same bits.
uint16_t to_f16_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);
    const int32_t exp = static_cast<int32_t>((u >> 23) & 0xffu) - 127 + 15;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((u & 0x7fffffu) >> 13));
}

std::vector<unsigned char> encode(const std::vector<float>& x, int dtype) {
    std::vector<unsigned char> out(x.size() * dtype_bytes(dtype));
    for (size_t i = 0; i < x.size(); ++i) {
        if (dtype == DBS_CUDA_DTYPE_F32) {
            std::memcpy(&out[i * 4], &x[i], 4);
        } else {
            const uint16_t h = dtype == DBS_CUDA_DTYPE_F16 ? to_f16_bits(x[i]) : to_bf16_bits(x[i]);
            std::memcpy(&out[i * 2], &h, 2);
        }
    }
    return out;
}

// dbs_cuda_decode_ex and dbs_cuda_backward_ex against the CPU's batch _ex API
// for one input type. Logits are the log-prob cases shifted and scaled (ties
// stay ties), and with value_mode 2 some rows have no finite entry at all. The
// batch API has one beam size and EOS per batch, so `variable` only varies the
// steps per example.
int run_ex_case(const Case& c, int dtype, int from_logits, uint32_t seed, const std::vector<int32_t>& extra_eos = {}) {
    std::mt19937 rng(seed);
    std::vector<float> values = make_log_probs(c, rng);
    const size_t B = static_cast<size_t>(c.B), T = static_cast<size_t>(c.T), K = static_cast<size_t>(c.K);
    const size_t V = static_cast<size_t>(c.V);
    const size_t BK = B * K, BTK = B * T * K;
    if (from_logits) {
        for (float& v : values) {
            if (std::isfinite(v)) v = v * 2.0f + 5.0f;
        }
    }
    if (c.value_mode == 2) {
        for (size_t r = 0; r < BTK; ++r) {
            if (rng() % 6 == 0) std::fill(&values[r * V], &values[r * V] + V, -std::numeric_limits<float>::infinity());
        }
    }
    const std::vector<unsigned char> data = encode(values, dtype);

    std::vector<int32_t> steps(B, c.T);
    if (c.variable) {
        for (int32_t& s : steps) s = 1 + static_cast<int>(rng() % c.T);
    }
    std::vector<uint8_t> banned(V, 0);
    for (uint8_t& v : banned) v = rng() % 6 == 0 ? 1 : 0;

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
    args.banned_tokens = c.banned ? banned.data() : nullptr;
    args.steps_per_example = c.variable ? steps.data() : nullptr;

    GpuResult g;
    g.final_scores.assign(BK, 1.0f);
    g.final_raw.assign(BK, 1.0f);
    g.final_len.assign(BK, 7);
    g.tokens.assign(BTK, 7);
    g.parents.assign(BTK, 7);
    g.lengths.assign(BTK, 7);
    g.scores.assign(BTK, 1.0f);
    g.raw_scores.assign(BTK, 1.0f);
    g.from_logprob.assign(BTK, 7);
    g.row_lse.assign(BTK, 7.0f);
    std::vector<uint8_t> invalid(B, 7);
    DBSCudaDecodeOutputs out{};
    out.invalid_input = invalid.data();
    out.final_scores = g.final_scores.data();
    out.final_raw_scores = g.final_raw.data();
    out.final_lengths = g.final_len.data();
    out.tokens = g.tokens.data();
    out.parents = g.parents.data();
    out.lengths = g.lengths.data();
    out.scores = g.scores.data();
    out.raw_scores = g.raw_scores.data();
    out.from_logprob = g.from_logprob.data();
    DBSCudaSearchOptions search{};
    search.extra_eos_count = static_cast<int>(extra_eos.size());
    for (size_t i = 0; i < extra_eos.size(); ++i) search.extra_eos_tokens[i] = extra_eos[i];
    if (extra_eos.empty()) {
        CHECK(dbs_cuda_decode_ex(data.data(), dtype, from_logits, &args, &out, g.row_lse.data(), nullptr, 0, nullptr) ==
              DBS_CUDA_STATUS_OK);
    } else {
        CHECK(dbs_cuda_decode_ex2(data.data(), dtype, from_logits, &args, &search, &out, g.row_lse.data(), nullptr, 0,
                                  nullptr) == DBS_CUDA_STATUS_OK);
    }

    DBSOptionsC opt{};
    opt.beam_size = c.K;
    opt.eos_token = c.eos;
    opt.min_length = c.min_length;
    opt.length_penalty_alpha = c.alpha;
    opt.validate_inputs = 0;
    DBSDecoderHandle* h = nullptr;
    CHECK(dbs_create_ex(opt, &h) == 0);
    CHECK(dbs_set_extra_eos_tokens(h, extra_eos.data(), static_cast<int>(extra_eos.size())) == 0);
    DBSAdvancedConstraintsC constraints{};
    constraints.min_length = -1;
    constraints.banned_tokens = c.banned ? banned.data() : nullptr;
    constraints.no_repeat_ngram_size = c.ngram;
    constraints.repetition_penalty = c.penalty;
    GpuResult cpu;  // the same arrays, from the CPU
    cpu.final_scores.assign(BK, 2.0f);
    cpu.final_raw.assign(BK, 2.0f);
    cpu.final_len.assign(BK, 9);
    cpu.tokens.assign(BTK, 9);
    cpu.parents.assign(BTK, 9);
    cpu.lengths.assign(BTK, 9);
    cpu.scores.assign(BTK, 2.0f);
    cpu.raw_scores.assign(BTK, 2.0f);
    cpu.from_logprob.assign(BTK, 9);
    cpu.row_lse.assign(BTK, 2.0f);
    DBSDecodeOutputsExC ex{};
    ex.base.final_scores = cpu.final_scores.data();
    ex.base.final_raw_scores = cpu.final_raw.data();
    ex.base.final_lengths = cpu.final_len.data();
    ex.base.tokens = cpu.tokens.data();
    ex.base.parents = cpu.parents.data();
    ex.base.lengths = cpu.lengths.data();
    ex.base.scores = cpu.scores.data();
    ex.base.raw_scores = cpu.raw_scores.data();
    ex.base.from_logprob = cpu.from_logprob.data();
    ex.row_lse = cpu.row_lse.data();
    CHECK(dbs_decode_batch_into_ex(h, data.data(), dtype, from_logits, c.B, c.T, c.V, args.steps_per_example,
                                   &constraints, 1, &ex) == 0);

    int mismatches = 0;
    auto report = [&](const char* what, size_t i) {
        if (mismatches++ < 5) {
            std::fprintf(stderr, "  _ex mismatch %s at %zu (dtype=%d logits=%d B=%d T=%d K=%d V=%d eos=%d mode=%d ngram=%d penalty=%g banned=%d var=%d)\n",
                         what, i, dtype, from_logits, c.B, c.T, c.K, c.V, c.eos, c.value_mode, c.ngram, c.penalty,
                         c.banned, c.variable);
        }
    };
    for (size_t i = 0; i < BK; ++i) {
        if (!same_bits(g.final_scores[i], cpu.final_scores[i])) report("final_score", i);
        if (!same_bits(g.final_raw[i], cpu.final_raw[i])) report("final_raw", i);
        if (g.final_len[i] != cpu.final_len[i]) report("final_length", i);
    }
    for (size_t i = 0; i < BTK; ++i) {
        if (g.tokens[i] != cpu.tokens[i]) report("token", i);
        if (g.parents[i] != cpu.parents[i]) report("parent", i);
        if (g.lengths[i] != cpu.lengths[i]) report("length", i);
        if (g.from_logprob[i] != cpu.from_logprob[i]) report("from_logprob", i);
        if (!same_bits(g.scores[i], cpu.scores[i])) report("score", i);
        if (!same_bits(g.raw_scores[i], cpu.raw_scores[i])) report("raw_score", i);
        if (!same_bits(g.row_lse[i], cpu.row_lse[i])) report("row_lse", i);
    }

    // The NaN/+inf flags match the CPU's validation, example by example.
    DBSOptionsC strict = opt;
    strict.validate_inputs = 1;
    DBSDecoderHandle* hs = nullptr;
    CHECK(dbs_create_ex(strict, &hs) == 0);
    CHECK(dbs_set_extra_eos_tokens(hs, extra_eos.data(), static_cast<int>(extra_eos.size())) == 0);
    const size_t example_bytes = T * K * V * dtype_bytes(dtype);
    std::vector<float> scratch(BTK * 2);
    for (size_t b = 0; b < B; ++b) {
        DBSDecodeOutputsExC one{};
        one.base.final_scores = scratch.data();
        const int rc = dbs_decode_batch_into_ex(hs, data.data() + b * example_bytes, dtype, from_logits, 1, c.T, c.V,
                                                &steps[b], &constraints, 1, &one);
        if ((rc != 0) != (invalid[b] != 0)) report("invalid flag", b);
    }
    dbs_destroy(hs);

    // Backward: any subset of the score gradients, accumulated onto a common base.
    std::uniform_real_distribution<float> gd(-2.0f, 2.0f);
    auto maybe = [&](size_t n) {
        std::vector<float> v;
        if (rng() % 3 != 0) {
            v.resize(n);
            for (float& x : v) x = rng() % 4 == 0 ? 0.0f : gd(rng);
        }
        return v;
    };
    const std::vector<float> gf = maybe(BK), gfr = maybe(BK), gs = maybe(BTK), gr = maybe(BTK);
    auto ptr = [](const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); };
    std::vector<float> base(BTK * V);
    for (float& x : base) x = rng() % 2 ? 0.0f : gd(rng);

    std::vector<float> grad_gpu = base;
    DBSCudaBackwardInputs gin{};
    gin.parents = g.parents.data();
    gin.tokens = g.tokens.data();
    gin.lengths = g.lengths.data();
    gin.from_logprob = g.from_logprob.data();
    gin.grad_final_scores = ptr(gf);
    gin.grad_final_raw_scores = ptr(gfr);
    gin.grad_scores = ptr(gs);
    gin.grad_raw_scores = ptr(gr);
    gin.logits = from_logits ? data.data() : nullptr;
    gin.logits_type = dtype;
    gin.row_lse = from_logits ? g.row_lse.data() : nullptr;
    CHECK(dbs_cuda_backward_ex(&args, &gin, grad_gpu.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);

    std::vector<float> grad_cpu = base;
    DBSBackwardInputsC cin{};
    cin.batch_size = c.B;
    cin.steps = c.T;
    cin.vocab_size = c.V;
    cin.steps_per_example = args.steps_per_example;
    cin.parents = cpu.parents.data();
    cin.tokens = cpu.tokens.data();
    cin.lengths = cpu.lengths.data();
    cin.from_logprob = cpu.from_logprob.data();
    cin.grad_final_scores = ptr(gf);
    cin.grad_final_raw_scores = ptr(gfr);
    cin.grad_scores = ptr(gs);
    cin.grad_raw_scores = ptr(gr);
    cin.logits = from_logits ? data.data() : nullptr;
    cin.logits_type = dtype;
    cin.row_lse = from_logits ? cpu.row_lse.data() : nullptr;
    CHECK(dbs_backward_batch_into_ex(h, &cin, 1, grad_cpu.data()) == 0);
    for (size_t i = 0; i < grad_gpu.size(); ++i) {
        if (!same_bits(grad_gpu[i], grad_cpu[i])) {
            report("grad", i);
            break;
        }
    }
    dbs_destroy(h);

    if (!c.variable) {
        mismatches += check_step_api(c, data.data(), dtype, from_logits, args, g, invalid,
                                     extra_eos.empty() ? nullptr : &search);
    }
    return mismatches;
}

int run_ex_all(const Case& c, uint32_t seed) {
    int failures = 0;
    int schedule = 0;
    for (int dtype : {DBS_CUDA_DTYPE_F32, DBS_CUDA_DTYPE_F16, DBS_CUDA_DTYPE_BF16}) {
        for (int from_logits : {0, 1}) {
            // Every combination under one schedule, rotating through the three.
            const dbs_emu::Schedule schedules[] = {dbs_emu::Schedule::Forward, dbs_emu::Schedule::Reverse,
                                                   dbs_emu::Schedule::Shuffled};
            dbs_emu::set_schedule(schedules[schedule++ % 3], seed);
            failures += run_ex_case(c, dtype, from_logits, seed) != 0 ? 1 : 0;
        }
    }
    dbs_emu::set_schedule(dbs_emu::Schedule::Forward);
    return failures;
}

void test_ex_parity() {
    std::mt19937 rng(20260926);
    int cases = 0;
    int failures = 0;
    for (int i = 0; i < 40; ++i) {
        Case c{};
        c.B = 1 + static_cast<int>(rng() % 3);
        c.T = 1 + static_cast<int>(rng() % 5);
        c.K = 1 + static_cast<int>(rng() % 9);
        c.V = 1 + static_cast<int>(rng() % 60);
        c.eos = (rng() % 3 == 0) ? -1 : static_cast<int>(rng() % c.V);
        c.min_length = static_cast<int>(rng() % 3);
        c.alpha = (rng() % 3) * 0.35f;
        c.variable = rng() % 3 == 0;
        c.value_mode = static_cast<int>(rng() % 4);
        if (rng() % 3 == 0) {
            c.ngram = static_cast<int>(rng() % 4);
            const float penalties[] = {0.0f, 1.0f, 1.3f, 2.0f};
            c.penalty = penalties[rng() % 4];
            c.banned = rng() % 2 == 0;
        }
        failures += run_ex_all(c, static_cast<uint32_t>(rng()));
        ++cases;
    }
    // Rows longer than the 256 logsumexp lanes, several register-scan chunks,
    // the tile scan, many NaN/+inf entries, and the maximum beam size. (The
    // tile reduction never reads inputs, so the cases above cover it.)
    failures += run_ex_all(Case{2, 3, 4, 700, 5, 1, 0.6f, false, 0}, 11);
    failures += run_ex_all(Case{1, 3, 16, 600, 4, 0, 0.9f, false, 3, 3, 1.2f, true}, 12);
    failures += run_ex_all(Case{2, 3, 17, 300, -1, 0, 0.0f, true, 2}, 13);
    failures += run_ex_all(Case{1, 2, 64, 300, 7, 1, 0.6f, false, 1, 1, 1.4f, false}, 14);
    failures += run_ex_all(Case{3, 4, 7, 23, 4, 4, 0.3f, false, 4, 0, 0.0f, true}, 15);
    failures += run_ex_all(Case{1, 2, DBS_CUDA_MAX_BEAM, 9, 2, 0, 0.8f, false, 1}, 16);
    cases += 6;
    std::printf("cuda emulation _ex parity: %d cases x 3 types x 2 input kinds, %d failing runs\n", cases, failures);
    CHECK(failures == 0);
}

// Extra EOS tokens (DBSCudaSearchOptions) against dbs_set_extra_eos_tokens.
// Heavy ties make every token, EOS tokens included, often the best.
void test_extra_eos_parity() {
    std::mt19937 rng(20261001);
    int cases = 0;
    int failures = 0;
    for (int i = 0; i < 24; ++i) {
        Case c{};
        c.B = 1 + static_cast<int>(rng() % 3);
        c.T = 2 + static_cast<int>(rng() % 5);
        c.K = 1 + static_cast<int>(rng() % 6);
        c.V = 3 + static_cast<int>(rng() % 12);
        c.eos = static_cast<int>(rng() % c.V);
        c.min_length = static_cast<int>(rng() % 3);
        c.alpha = (rng() % 3) * 0.35f;
        c.variable = rng() % 4 == 0;
        c.value_mode = rng() % 2 ? 1 : 0;
        if (rng() % 3 == 0) {
            c.ngram = static_cast<int>(rng() % 3);
            c.penalty = rng() % 2 ? 1.5f : 0.0f;
            c.banned = rng() % 2 == 0;
        }
        std::vector<int32_t> extra;
        for (int e = 1 + static_cast<int>(rng() % 3); e > 0; --e) extra.push_back(static_cast<int32_t>(rng() % c.V));
        const int dtype = static_cast<int>(rng() % 3);
        const int from_logits = static_cast<int>(rng() % 2);
        const uint32_t seed = static_cast<uint32_t>(rng());
        for (auto schedule : {dbs_emu::Schedule::Forward, dbs_emu::Schedule::Shuffled}) {
            dbs_emu::set_schedule(schedule, seed);
            failures += run_ex_case(c, dtype, from_logits, seed, extra) != 0 ? 1 : 0;
        }
        ++cases;
    }
    // The tile scan (K > 16) and the maximum number of extra tokens.
    std::vector<int32_t> many;
    for (int v = 0; v < DBS_CUDA_MAX_EXTRA_EOS; ++v) many.push_back(v);
    failures += run_ex_case(Case{2, 4, 20, 40, 21, 1, 0.6f, false, 1, 0, 0.0f, true}, DBS_CUDA_DTYPE_BF16, 1, 7, many);
    cases += 1;
    dbs_emu::set_schedule(dbs_emu::Schedule::Forward);
    std::printf("cuda emulation extra EOS parity: %d cases, %d failing runs\n", cases, failures);
    CHECK(failures == 0);

    // Validation.
    std::vector<float> x(2 * 3, -1.0f), scores(2);
    DBSCudaDecodeOutputs out{};
    out.final_scores = scores.data();
    DBSCudaDecodeArgs a{};
    a.batch_size = 1;
    a.steps = 1;
    a.beam_size = 2;
    a.vocab_size = 3;
    a.eos_token = 0;
    DBSCudaSearchOptions o{};
    o.extra_eos_count = 1;
    o.extra_eos_tokens[0] = 3;  // outside the vocabulary
    CHECK(dbs_cuda_decode_ex2(x.data(), 0, 0, &a, &o, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    o.extra_eos_tokens[0] = 2;
    CHECK(dbs_cuda_decode_ex2(x.data(), 0, 0, &a, &o, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
    o.extra_eos_count = DBS_CUDA_MAX_EXTRA_EOS + 1;
    CHECK(dbs_cuda_decode_ex2(x.data(), 0, 0, &a, &o, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    o.extra_eos_count = 1;
    o.reserved0 = 1;
    CHECK(dbs_cuda_decode_ex2(x.data(), 0, 0, &a, &o, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    o.reserved0 = 0;
    a.eos_token = -1;  // extra EOS tokens need EOS handling
    CHECK(dbs_cuda_decode_ex2(x.data(), 0, 0, &a, &o, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
}

void test_ex_argument_validation() {
    const int B = 1, K = 2, V = 3;
    std::vector<float> x(static_cast<size_t>(B) * K * V, -1.0f);
    std::vector<float> scores(static_cast<size_t>(B) * K), row_lse(static_cast<size_t>(B) * K, 7.0f);
    DBSCudaDecodeOutputs out{};
    out.final_scores = scores.data();
    DBSCudaDecodeArgs a{};
    a.batch_size = B;
    a.steps = 1;
    a.beam_size = K;
    a.vocab_size = V;
    a.eos_token = -1;
    CHECK(dbs_cuda_decode_ex(x.data(), 3, 0, &a, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    CHECK(dbs_cuda_decode_ex(x.data(), -1, 1, &a, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    CHECK(dbs_cuda_decode_ex(nullptr, 0, 1, &a, &out, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    // Without logits, row_lse is all zeros.
    CHECK(dbs_cuda_decode_ex(x.data(), DBS_CUDA_DTYPE_F32, 0, &a, &out, row_lse.data(), nullptr, 0, nullptr) ==
          DBS_CUDA_STATUS_OK);
    for (float v : row_lse) CHECK(v == 0.0f);
    // From logits: only the one live row is read at step 0.
    CHECK(dbs_cuda_decode_ex(x.data(), DBS_CUDA_DTYPE_F32, 1, &a, &out, row_lse.data(), nullptr, 0, nullptr) ==
          DBS_CUDA_STATUS_OK);
    CHECK(row_lse[0] > -1.0f && row_lse[1] == 0.0f);
    DBSCudaBeamState state{};
    CHECK(dbs_cuda_decode_step_ex(x.data(), 5, 0, &a, &state, &out, nullptr, nullptr, 0, nullptr) ==
          DBS_CUDA_STATUS_INVALID_ARGUMENT);

    std::vector<int32_t> parents = {0, -1}, tokens = {0, -1}, lengths = {1, 0};
    std::vector<uint8_t> flp = {1, 0};
    std::vector<float> gf = {1.0f, 0.0f}, grad(x.size(), 0.0f);
    DBSCudaBackwardInputs in{};
    in.parents = parents.data();
    in.tokens = tokens.data();
    in.lengths = lengths.data();
    in.from_logprob = flp.data();
    in.grad_final_scores = gf.data();
    CHECK(dbs_cuda_backward_ex(&a, &in, grad.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
    CHECK(grad[0] == 1.0f);
    CHECK(dbs_cuda_backward_ex(&a, &in, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    DBSCudaBackwardInputs bad = in;
    bad.logits = x.data();  // logits need row_lse
    CHECK(dbs_cuda_backward_ex(&a, &bad, grad.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    bad.row_lse = row_lse.data();
    bad.logits_type = 7;
    CHECK(dbs_cuda_backward_ex(&a, &bad, grad.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    bad = in;
    bad.reserved0 = 1;
    CHECK(dbs_cuda_backward_ex(&a, &bad, grad.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    bad = in;
    bad.reserved[2] = grad.data();
    CHECK(dbs_cuda_backward_ex(&a, &bad, grad.data(), nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    // The logits correction needs the whole workspace.
    bad = in;
    bad.logits = x.data();
    bad.row_lse = row_lse.data();
    const int64_t need = dbs_cuda_backward_workspace_size(&a);
    std::vector<unsigned char> ws(static_cast<size_t>(need));
    CHECK(dbs_cuda_backward_ex(&a, &bad, grad.data(), ws.data(), need - 1, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    std::fill(grad.begin(), grad.end(), 0.0f);
    CHECK(dbs_cuda_backward_ex(&a, &bad, grad.data(), ws.data(), need, nullptr) == DBS_CUDA_STATUS_OK);
    // Row 0 of three equal logits: 1 - 1/3 for the token taken, -1/3 for the others.
    CHECK(std::fabs(grad[0] - 2.0f / 3.0f) < 1e-6f && std::fabs(grad[1] + 1.0f / 3.0f) < 1e-6f);
    CHECK(grad[3] == 0.0f && grad[4] == 0.0f && grad[5] == 0.0f);
}

int run_all_schedules(const Case& c, uint32_t seed) {
    int failures = 0;
    for (auto schedule : {dbs_emu::Schedule::Forward, dbs_emu::Schedule::Reverse, dbs_emu::Schedule::Shuffled}) {
        dbs_emu::set_schedule(schedule, seed);
        failures += run_case(c, seed) != 0 ? 1 : 0;
    }
    dbs_emu::set_schedule(dbs_emu::Schedule::Forward);
    return failures;
}

void test_randomized_parity() {
    std::mt19937 rng(20240917);
    int cases = 0;
    int failures = 0;
    for (int i = 0; i < 160; ++i) {
        Case c{};
        c.B = 1 + static_cast<int>(rng() % 3);
        c.T = 1 + static_cast<int>(rng() % 6);
        c.K = 1 + static_cast<int>(rng() % 9);
        c.V = 1 + static_cast<int>(rng() % 60);
        c.eos = (rng() % 3 == 0) ? -1 : static_cast<int>(rng() % c.V);
        c.min_length = static_cast<int>(rng() % 3);
        c.alpha = (rng() % 3) * 0.35f;
        c.variable = rng() % 3 == 0;
        c.value_mode = static_cast<int>(rng() % 4);
        if (rng() % 3 == 0) {
            c.ngram = static_cast<int>(rng() % 4);
            const float penalties[] = {0.0f, 1.0f, 1.3f, 2.0f};
            c.penalty = penalties[rng() % 4];
            c.banned = rng() % 2 == 0;
        }
        failures += run_all_schedules(c, static_cast<uint32_t>(rng()));
        ++cases;
    }
    // Both scan kernels around their boundary, with constraints.
    for (int K : {15, 16, 17, 32}) {
        failures += run_all_schedules(Case{2, 4, K, 37, 3, 2, 0.6f, false, 3, 2, 1.5f, true}, 10u + K);
        failures += run_all_schedules(Case{1, 3, K, 300, -1, 0, 0.0f, true, 1}, 20u + K);
        cases += 2;
    }
    // Tiny vocabularies over many steps, so prefixes repeat and n-gram blocking
    // and the repetition penalty act often.
    for (int V : {2, 3, 5}) {
        for (int ngram : {1, 2, 3, 4}) {
            for (int K : {1, 3, 20}) {
                const float penalty = (ngram % 2) ? 1.5f : 0.0f;
                failures += run_all_schedules(Case{2, 10, K, V, V > 2 ? V - 1 : -1, 0, 0.5f, false, 1, ngram, penalty, false}, 100u + V * 10 + ngram);
                ++cases;
            }
        }
    }
    // Many NaN/+inf entries, half the vocabulary banned, EOS masked by min_length:
    // every entry of a row that is read must be checked.
    for (int K : {2, 7, 18}) {
        failures += run_all_schedules(Case{3, 5, K, 23, 4, 4, 0.3f, false, 4, 0, 0.0f, true}, 200u + K);
        failures += run_all_schedules(Case{2, 4, K, 31, 1, 3, 0.0f, true, 4, 2, 1.3f, true}, 300u + K);
        cases += 2;
    }
    // Several register-scan chunks per example (K*V > 8192).
    failures += run_all_schedules(Case{2, 3, 4, 5000, 7, 1, 0.6f, false, 0}, 5);
    failures += run_all_schedules(Case{1, 3, 16, 2100, 4, 0, 0.9f, false, 3, 3, 1.2f, true}, 6);
    // Multi-tile scans with a reduction level (K*V spans many 4096-candidate tiles).
    failures += run_all_schedules(Case{2, 3, 64, 5000, 7, 1, 0.6f, false, 0}, 1);
    failures += run_all_schedules(Case{1, 2, 64, 4200, -1, 0, 0.0f, false, 1, 1, 1.4f, false}, 2);
    // Maximum beam size.
    failures += run_all_schedules(Case{1, 3, DBS_CUDA_MAX_BEAM, 9, 2, 0, 0.8f, false, 1}, 3);
    failures += run_all_schedules(Case{2, 2, DBS_CUDA_MAX_BEAM, 5, -1, 0, 0.0f, true, 0}, 4);
    cases += 6;
    std::printf("cuda emulation parity: %d cases x 3 schedules, %d failing runs\n", cases, failures);
    CHECK(failures == 0);
}

// NaN/+inf flags for entries placed exactly where the search must still look
// (banned tokens, EOS masked by min_length) or must not (rows it never reads).
void test_validation_flags() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (int K : {2, 20}) {  // register scan and tile scan
        const int B = 2, T = 3, V = 8;
        struct Placement {
            int b, t, k, v;
            float value;
            bool read;  // the search reads this row
        };
        const Placement placements[] = {
            {0, 0, 0, 3, nan, true},    // banned token of a live row
            {1, 0, 0, 5, inf, true},    // EOS masked by min_length
            {0, 1, 1, 6, nan, true},    // second beam once it is live
            {0, 0, 1, 2, nan, false},   // beam 1 is not live at step 0
            {1, 0, K - 1, 0, inf, false},
        };
        std::vector<uint8_t> banned(V, 0);
        banned[3] = 1;
        for (const Placement& pl : placements) {
            std::vector<float> x(static_cast<size_t>(B) * T * K * V);
            for (size_t i = 0; i < x.size(); ++i) x[i] = -0.25f * static_cast<float>((i * 7) % 5) - 0.1f * static_cast<float>(i % V == 1);
            x[((static_cast<size_t>(pl.b) * T + pl.t) * K + pl.k) * V + pl.v] = pl.value;
            DBSCudaDecodeArgs a{};
            a.batch_size = B;
            a.steps = T;
            a.beam_size = K;
            a.vocab_size = V;
            a.eos_token = 5;
            a.min_length = 3;
            a.banned_tokens = banned.data();
            std::vector<float> scores(static_cast<size_t>(B) * K);
            std::vector<uint8_t> invalid(B, 7);
            DBSCudaDecodeOutputs out{};
            out.final_scores = scores.data();
            out.invalid_input = invalid.data();
            CHECK(dbs_cuda_decode(x.data(), &a, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
            for (int b = 0; b < B; ++b) CHECK(invalid[b] == ((b == pl.b && pl.read) ? 1 : 0));
        }
    }
}

void test_argument_validation() {
    std::vector<float> x(2 * 3 * 4, -1.0f);
    std::vector<float> scores(2 * 4);
    DBSCudaDecodeOutputs out{};
    out.final_scores = scores.data();
    DBSCudaDecodeArgs a{};
    a.batch_size = 2;
    a.steps = 1;
    a.beam_size = 4;
    a.vocab_size = 3;
    a.eos_token = -1;
    CHECK(dbs_cuda_decode(x.data(), &a, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);

    DBSCudaDecodeArgs bad = a;
    bad.eos_token = 3;
    CHECK(dbs_cuda_decode(x.data(), &bad, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    bad = a;
    bad.beam_size = DBS_CUDA_MAX_BEAM + 1;
    CHECK(dbs_cuda_decode_workspace_size(&bad) < 0);
    bad = a;
    bad.length_penalty_alpha = -1.0f;
    CHECK(dbs_cuda_decode(x.data(), &bad, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    bad = a;
    bad.repetition_penalty = -2.0f;
    CHECK(dbs_cuda_decode(x.data(), &bad, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    bad = a;
    bad.reserved0 = 1;
    CHECK(dbs_cuda_decode(x.data(), &bad, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    CHECK(dbs_cuda_backward_workspace_size(&a) > 0);
    CHECK(dbs_cuda_decode(x.data(), &a, nullptr, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);

    // Too-small caller workspace is rejected; an adequate one works.
    const int64_t need = dbs_cuda_decode_workspace_size(&a);
    CHECK(need > 0);
    std::vector<unsigned char> ws(static_cast<size_t>(need));
    CHECK(dbs_cuda_decode(x.data(), &a, &out, ws.data(), need - 1, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    CHECK(dbs_cuda_decode(x.data(), &a, &out, ws.data(), need, nullptr) == DBS_CUDA_STATUS_OK);

    // Invalid per-example metadata is detected on the device.
    std::vector<int32_t> steps = {1, 2};  // 2 > T
    DBSCudaDecodeArgs var = a;
    var.steps_per_example = steps.data();
    CHECK(dbs_cuda_decode(x.data(), &var, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    std::vector<int32_t> eos = {-1, 3};  // 3 >= V
    var = a;
    var.eos_tokens_per_example = eos.data();
    CHECK(dbs_cuda_decode(x.data(), &var, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_INVALID_ARGUMENT);

    CHECK(dbs_cuda_set_synchronization(2) == DBS_CUDA_STATUS_INVALID_ARGUMENT);
    CHECK(dbs_cuda_set_synchronization(1) == DBS_CUDA_STATUS_OK);
    CHECK(dbs_cuda_get_synchronization() == 1);
    CHECK(dbs_cuda_decode(x.data(), &a, &out, nullptr, 0, nullptr) == DBS_CUDA_STATUS_OK);
    CHECK(dbs_cuda_set_synchronization(0) == DBS_CUDA_STATUS_OK);
    CHECK(std::strcmp(dbs_cuda_status_string(DBS_CUDA_STATUS_INVALID_ARGUMENT), "invalid argument") == 0);
}

}  // namespace

int main() {
    CHECK(dbs_cuda_available() == 1);
    test_argument_validation();
    test_validation_flags();
    test_randomized_parity();
    test_ex_argument_validation();
    test_extra_eos_parity();
    test_ex_parity();
    std::printf("cuda_emulation_tests passed\n");
    return 0;
}
