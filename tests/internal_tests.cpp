// SPDX-License-Identifier: MIT
//
// Tests of libdbs internals, linked against the implementation objects:
//
//   * every SIMD row scan the host supports against the scalar reference, bit
//     for bit, on random rows (ties, -inf, NaN/+inf, banned and masked tokens,
//     forced tokens, pre-filled top-k buffers);
//   * full decode + backward on every kernel path against the scalar path;
//   * constrained decoding (n-gram blocking, repetition penalty, token filter,
//     banned/forced tokens, min length) against a direct reference
//     implementation that checks every candidate token against the prefix;
//   * input validation covers exactly the rows the search reads;
//   * the length penalty matches pow() and stays defined for huge exponents.
#include "decoder.hpp"

#include "check.hpp"

#include <cstring>
#include <iostream>
#include <random>
#include <string>

#if defined(_WIN32) && !defined(DBS_STATIC)
#error dbs_internal_tests must compile with DBS_STATIC on Windows
#endif

using namespace dbs;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

std::mt19937& rng() {
    static std::mt19937 engine(20240917u);
    return engine;
}

int uniform_int(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng()); }
float uniform_float(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng()); }

uint32_t bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

struct KernelScope {
    KernelOverride previous;
    explicit KernelScope(KernelPath path) : previous(current_kernel_override()) { set_kernel_override({true, path}); }
    ~KernelScope() { set_kernel_override(previous); }
    KernelScope(const KernelScope&) = delete;
    KernelScope& operator=(const KernelScope&) = delete;
};

std::vector<KernelPath> simd_paths() {
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

bool same_candidates(const std::vector<Candidate>& a, const std::vector<Candidate>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (bits(a[i].score) != bits(b[i].score) || bits(a[i].raw_score) != bits(b[i].raw_score) ||
            a[i].parent != b[i].parent || a[i].token != b[i].token || a[i].length != b[i].length ||
            a[i].from_logprob != b[i].from_logprob) {
            return false;
        }
    }
    return true;
}

template <class A, class B>
bool same_floats(const A& a, const B& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (bits(a[i]) != bits(b[i])) return false;
    }
    return true;
}

template <class A, class B>
bool same_values(const A& a, const B& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

float special_value(int style) {
    switch (uniform_int(0, style == 0 ? 3 : 12)) {
        case 0: return -kInf;
        case 1: return -0.25f * static_cast<float>(uniform_int(0, 6));  // ties
        case 2: return uniform_float(-6.0f, 1.0f);
        case 3: return -1.0f;
        case 4: return std::numeric_limits<float>::quiet_NaN();
        case 5: return kInf;
        default: return uniform_float(-10.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Row scans
// ---------------------------------------------------------------------------

void test_row_scan_parity() {
    const std::vector<KernelPath> paths = simd_paths();
    int cases = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        const int V = uniform_int(1, 80);
        std::vector<float> row(static_cast<size_t>(V));
        const int style = uniform_int(0, 1);  // 1 includes NaN and +inf
        for (float& x : row) x = special_value(style);
        std::vector<uint8_t> banned(static_cast<size_t>(V));
        for (uint8_t& b : banned) b = static_cast<uint8_t>(uniform_int(0, 3) == 0 ? uniform_int(1, 255) : 0);

        RowScan s;
        s.row = row.data();
        s.banned = uniform_int(0, 1) ? banned.data() : nullptr;
        s.parent_raw = uniform_int(0, 3) == 0 ? 0.0f : uniform_float(-8.0f, 0.0f);
        s.new_length = uniform_int(1, 9);
        s.inv_penalty = 1.0f / gnmt_length_penalty(s.new_length, uniform_int(0, 1) ? 0.0f : uniform_float(0.1f, 2.0f));
        s.parent = uniform_int(0, 7);
        s.vocab_size = V;
        s.forced_token = uniform_int(0, 5) == 0 ? uniform_int(0, V - 1) : -1;
        s.masked_token = uniform_int(0, 2) == 0 ? uniform_int(0, V - 1) : -1;

        // A top-k buffer that may already hold candidates of other parents.
        const int top_count = uniform_int(1, 12);
        std::vector<Candidate> initial(static_cast<size_t>(top_count), Candidate{-kInf, -kInf, -1, -1, 0, 0});
        for (int i = uniform_int(0, 2 * top_count); i > 0; --i) {
            const float raw = -0.25f * static_cast<float>(uniform_int(0, 30));
            insert_topk(initial.data(), top_count,
                        Candidate{raw * s.inv_penalty, raw, uniform_int(0, 7), uniform_int(0, V - 1), s.new_length, 1});
        }

        std::vector<Candidate> expected = initial;
        const bool expected_invalid = scan_row_scalar(s, expected.data(), top_count);
        bool any_invalid = false;
        for (float x : row) any_invalid |= !(x < kInf);
        CHECK(expected_invalid == any_invalid);

        for (KernelPath path : paths) {
            KernelScope scope(path);
            std::vector<Candidate> actual = initial;
            const bool invalid = scan_row(s, actual.data(), top_count);
            if (invalid != expected_invalid || !same_candidates(actual, expected)) {
                std::cerr << "row scan mismatch on " << kernel_path_name(path) << " (trial " << trial << ", V " << V
                          << ", invalid " << invalid << " vs " << expected_invalid << ")\n";
                for (size_t i = 0; i < actual.size(); ++i) {
                    std::cerr << "  " << actual[i].score << "/" << actual[i].token << "  vs  " << expected[i].score << "/"
                              << expected[i].token << "\n";
                }
                std::abort();
            }
            ++cases;
        }
    }
    std::cout << "  row scans: " << cases << " SIMD comparisons\n";
}

// ---------------------------------------------------------------------------
// Reference constrained decoder: checks every token directly against the prefix.
// ---------------------------------------------------------------------------

bool prefix_contains(const std::vector<int32_t>& prefix, int token) {
    return std::find(prefix.begin(), prefix.end(), token) != prefix.end();
}

bool would_repeat_ngram(const std::vector<int32_t>& prefix, int token, int n) {
    if (n <= 0) return false;
    if (n == 1) return prefix_contains(prefix, token);
    const int len = static_cast<int>(prefix.size());
    if (len + 1 < n) return false;
    const int suffix_start = len - (n - 1);
    for (int i = 0; i + n <= len; ++i) {
        bool same = prefix[static_cast<size_t>(i + n - 1)] == token;
        for (int j = 0; j < n - 1 && same; ++j) {
            same = prefix[static_cast<size_t>(i + j)] == prefix[static_cast<size_t>(suffix_start + j)];
        }
        if (same) return true;
    }
    return false;
}

DecodeResult reference_decode(const BeamOptions& opt, const float* x, int T, int V, const DecodeConstraints* c) {
    const int K = opt.beam_size;
    const int eos = opt.eos_token;
    const int min_length = c && c->min_length >= 0 ? c->min_length : opt.min_length;
    DecodeResult r;
    r.steps = T;
    r.beam_size = K;
    r.vocab_size = V;
    const size_t n = static_cast<size_t>(T) * static_cast<size_t>(K);
    r.parents.assign(n, -1);
    r.tokens.assign(n, -1);
    r.lengths.assign(n, 0);
    r.scores.assign(n, -kInf);
    r.raw_scores.assign(n, -kInf);
    r.from_logprob.assign(n, 0);

    std::vector<float> raw(static_cast<size_t>(K), -kInf);
    std::vector<int32_t> len(static_cast<size_t>(K), 0);
    std::vector<uint8_t> ended(static_cast<size_t>(K), 0);
    std::vector<std::vector<int32_t>> prefix(static_cast<size_t>(K));
    raw[0] = 0.0f;

    for (int t = 0; t < T; ++t) {
        std::vector<Candidate> top(static_cast<size_t>(K), Candidate{-kInf, -kInf, -1, -1, 0, 0});
        for (int b = 0; b < K; ++b) {
            const float parent_raw = raw[static_cast<size_t>(b)];
            if (!std::isfinite(parent_raw)) continue;
            if (eos >= 0 && ended[static_cast<size_t>(b)]) {
                const int l = std::max(1, static_cast<int>(len[static_cast<size_t>(b)]));
                insert_topk(top.data(), K, Candidate{parent_raw / gnmt_length_penalty(l, opt.length_penalty_alpha), parent_raw, b, eos, l, 0});
                continue;
            }
            const int new_len = len[static_cast<size_t>(b)] + 1;
            const float inv = 1.0f / gnmt_length_penalty(new_len, opt.length_penalty_alpha);
            const int forced = c && c->forced_tokens ? c->forced_tokens[t] : -1;
            const float* row = x + (static_cast<size_t>(t) * K + static_cast<size_t>(b)) * static_cast<size_t>(V);
            const std::vector<int32_t>& p = prefix[static_cast<size_t>(b)];
            for (int v = 0; v < V; ++v) {
                if (forced >= 0 && v != forced) continue;
                if (c && c->banned_tokens && c->banned_tokens[v]) continue;
                if (eos >= 0 && v == eos && new_len < min_length) continue;
                if (c && would_repeat_ngram(p, v, c->no_repeat_ngram_size)) continue;
                if (c && c->token_filter &&
                    c->token_filter(c->token_filter_user_data, c->batch_index, t, b, p.empty() ? nullptr : p.data(),
                                    static_cast<int>(p.size()), v) == 0) {
                    continue;
                }
                float lp = row[v];
                if (!std::isfinite(lp)) continue;
                if (c && c->repetition_penalty > 1.0f && prefix_contains(p, v)) lp = lp - std::log(c->repetition_penalty);
                const float cand_raw = parent_raw + lp;
                insert_topk(top.data(), K, Candidate{cand_raw * inv, cand_raw, b, v, new_len, 1});
            }
        }
        std::vector<float> next_raw(static_cast<size_t>(K));
        std::vector<int32_t> next_len(static_cast<size_t>(K));
        std::vector<uint8_t> next_ended(static_cast<size_t>(K));
        std::vector<std::vector<int32_t>> next_prefix(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k) {
            const Candidate& cand = top[static_cast<size_t>(k)];
            const size_t idx = static_cast<size_t>(t) * K + static_cast<size_t>(k);
            r.parents[idx] = cand.parent;
            r.tokens[idx] = cand.token;
            r.lengths[idx] = cand.length;
            r.scores[idx] = cand.score;
            r.raw_scores[idx] = cand.raw_score;
            r.from_logprob[idx] = cand.from_logprob;
            next_raw[static_cast<size_t>(k)] = cand.raw_score;
            next_len[static_cast<size_t>(k)] = cand.length;
            if (cand.parent >= 0) {
                next_ended[static_cast<size_t>(k)] = ended[static_cast<size_t>(cand.parent)] || (eos >= 0 && cand.token == eos);
                next_prefix[static_cast<size_t>(k)] = prefix[static_cast<size_t>(cand.parent)];
                if (cand.from_logprob) next_prefix[static_cast<size_t>(k)].push_back(cand.token);
            }
        }
        raw.swap(next_raw);
        len.swap(next_len);
        ended.swap(next_ended);
        prefix.swap(next_prefix);
    }
    r.final_raw_scores.assign(raw.begin(), raw.end());
    r.final_scores.resize(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        const int l = std::max(1, static_cast<int>(len[static_cast<size_t>(k)]));
        r.final_scores[static_cast<size_t>(k)] = raw[static_cast<size_t>(k)] / gnmt_length_penalty(l, opt.length_penalty_alpha);
    }
    return r;
}

int filter_calls = 0;
int hash_filter(void*, int, int step, int parent, const int32_t* prefix, int prefix_len, int token) {
    ++filter_calls;
    int h = token * 7 + step * 3 + parent + prefix_len;
    for (int i = 0; i < prefix_len; ++i) h += prefix[i] * (i + 1);
    return (h % 5) != 0;
}

void test_constrained_decode_matches_reference() {
    const std::vector<KernelPath> paths = simd_paths();
    int cases = 0;
    for (int trial = 0; trial < 1500; ++trial) {
        BeamOptions opt;
        opt.beam_size = uniform_int(1, 6);
        const int V = uniform_int(2, 40);
        const int T = uniform_int(1, 8);
        opt.eos_token = uniform_int(0, 2) == 0 ? -1 : uniform_int(0, V - 1);
        opt.min_length = uniform_int(0, 3);
        opt.length_penalty_alpha = uniform_int(0, 1) ? 0.0f : uniform_float(0.1f, 1.5f);
        opt.validate_inputs = 0;

        std::vector<float> x(static_cast<size_t>(T) * opt.beam_size * V);
        for (float& v : x) v = uniform_int(0, 9) == 0 ? -kInf : -0.25f * static_cast<float>(uniform_int(0, 12));
        std::vector<uint8_t> banned(static_cast<size_t>(V));
        for (uint8_t& b : banned) b = uniform_int(0, 5) == 0;
        std::vector<int32_t> forced(static_cast<size_t>(T));
        for (int32_t& f : forced) f = uniform_int(0, 4) == 0 ? uniform_int(0, V - 1) : -1;

        DecodeConstraints c;
        c.banned_tokens = uniform_int(0, 1) ? banned.data() : nullptr;
        c.forced_tokens = uniform_int(0, 3) == 0 ? forced.data() : nullptr;
        c.min_length = uniform_int(-1, 3);
        const float penalties[] = {1.0f, 1.0f, 1.2f, 2.5f};
        c.repetition_penalty = penalties[uniform_int(0, 3)];
        c.no_repeat_ngram_size = uniform_int(0, 4);
        c.token_filter = uniform_int(0, 3) == 0 ? hash_filter : nullptr;
        c.batch_index = uniform_int(0, 3);

        filter_calls = 0;
        const DecodeResult expected = reference_decode(opt, x.data(), T, V, &c);
        const int expected_calls = filter_calls;

        std::vector<KernelPath> all = paths;
        all.push_back(KernelPath::Scalar);
        for (KernelPath path : all) {
            KernelScope scope(path);
            filter_calls = 0;
            const DecodeResult actual = BeamSearchDecoder(opt).decode_constrained(x.data(), T, V, &c);
            const bool ok = same_values(actual.parents, expected.parents) && same_values(actual.tokens, expected.tokens) &&
                            same_values(actual.lengths, expected.lengths) && same_floats(actual.scores, expected.scores) &&
                            same_floats(actual.raw_scores, expected.raw_scores) &&
                            same_values(actual.from_logprob, expected.from_logprob) &&
                            same_floats(actual.final_scores, expected.final_scores) &&
                            same_floats(actual.final_raw_scores, expected.final_raw_scores) &&
                            filter_calls == expected_calls;
            if (!ok) {
                std::cerr << "constrained decode differs from the reference on " << kernel_path_name(path)
                          << " (trial " << trial << ")\n";
                std::abort();
            }
            ++cases;
        }
    }
    std::cout << "  constrained decodes: " << cases << " comparisons with the reference\n";
}

// ---------------------------------------------------------------------------
// Whole decode + backward on every path, bit for bit
// ---------------------------------------------------------------------------

struct Bundle {
    DecodeResult forward;
    BackwardResult dense;
    BackwardResult sparse;
};

Bundle run_case(const BeamOptions& opt, const std::vector<float>& x, int T, int V, const DecodeConstraints* c) {
    const BeamSearchDecoder decoder(opt);
    Bundle out;
    out.forward = decoder.decode_constrained(x.data(), T, V, c);
    const size_t n = static_cast<size_t>(T) * static_cast<size_t>(opt.beam_size);
    const size_t np = static_cast<size_t>(T) * static_cast<size_t>(out.forward.relaxed_pool_size);
    std::vector<float> gs(n), gr(np), gf(static_cast<size_t>(opt.beam_size));
    for (size_t i = 0; i < gs.size(); ++i) gs[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    for (size_t i = 0; i < gr.size(); ++i) gr[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.03125f;
    for (size_t i = 0; i < gf.size(); ++i) gf[i] = static_cast<float>(i + 1) * 0.2f;
    const float* gr_ptr = np > 0 ? gr.data() : nullptr;
    out.dense = decoder.backward(out.forward, gs.data(), gr_ptr, gf.data());
    out.sparse = decoder.backward_sparse(out.forward, gs.data(), gr_ptr, gf.data());
    return out;
}

bool same_bundle(const Bundle& a, const Bundle& b) {
    const DecodeResult& x = a.forward;
    const DecodeResult& y = b.forward;
    return same_values(x.parents, y.parents) && same_values(x.tokens, y.tokens) && same_values(x.lengths, y.lengths) &&
           same_floats(x.scores, y.scores) && same_floats(x.raw_scores, y.raw_scores) && same_floats(x.weights, y.weights) &&
           same_floats(x.final_scores, y.final_scores) && same_floats(x.final_raw_scores, y.final_raw_scores) &&
           same_values(x.pool_parents, y.pool_parents) && same_values(x.pool_tokens, y.pool_tokens) &&
           same_floats(x.pool_scores, y.pool_scores) && same_floats(x.relaxed_weights, y.relaxed_weights) &&
           same_floats(a.dense.grad_log_probs, b.dense.grad_log_probs) &&
           same_floats(a.dense.grad_initial_scores, b.dense.grad_initial_scores) &&
           same_values(a.sparse.sparse_logprob_indices, b.sparse.sparse_logprob_indices) &&
           same_floats(a.sparse.sparse_logprob_values, b.sparse.sparse_logprob_values) &&
           same_floats(a.sparse.grad_initial_scores, b.sparse.grad_initial_scores);
}

void test_decode_backward_parity() {
    const std::vector<KernelPath> paths = simd_paths();
    int cases = 0;
    for (int trial = 0; trial < 400; ++trial) {
        BeamOptions opt;
        opt.beam_size = uniform_int(1, 9);
        const int V = uniform_int(1, 70);
        const int T = uniform_int(1, 6);
        opt.eos_token = uniform_int(0, 2) == 0 ? -1 : uniform_int(0, V - 1);
        opt.min_length = uniform_int(0, 3);
        opt.length_penalty_alpha = uniform_int(0, 1) ? 0.0f : uniform_float(0.1f, 1.5f);
        opt.relaxed_pool_multiplier = uniform_int(0, 3);
        opt.selected_temperature = 0.85f;
        opt.soft_topk_temperature = 0.4f;
        opt.validate_inputs = 0;
        std::vector<float> x(static_cast<size_t>(T) * opt.beam_size * V);
        for (float& v : x) v = special_value(uniform_int(0, 1));
        std::vector<uint8_t> banned(static_cast<size_t>(V));
        for (uint8_t& b : banned) b = uniform_int(0, 5) == 0;
        DecodeConstraints c;
        c.banned_tokens = banned.data();
        const DecodeConstraints* cp = uniform_int(0, 2) == 0 ? &c : nullptr;

        Bundle expected;
        {
            KernelScope scope(KernelPath::Scalar);
            expected = run_case(opt, x, T, V, cp);
        }
        for (KernelPath path : paths) {
            KernelScope scope(path);
            if (!same_bundle(run_case(opt, x, T, V, cp), expected)) {
                std::cerr << "decode/backward mismatch on " << kernel_path_name(path) << " (trial " << trial << ")\n";
                std::abort();
            }
            ++cases;
        }
    }
    std::cout << "  decode + backward: " << cases << " SIMD comparisons\n";
}

// ---------------------------------------------------------------------------
// Validation, variable beams, model steps
// ---------------------------------------------------------------------------

bool decode_throws(const BeamOptions& opt, const std::vector<float>& x, int T, int V) {
    try {
        (void)BeamSearchDecoder(opt).decode(x.data(), T, V);
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("NaN or +inf") != std::string::npos);
        return true;
    }
    return false;
}

void test_validation_covers_the_rows_read() {
    BeamOptions opt;
    opt.beam_size = 3;
    opt.eos_token = 1;
    opt.validate_inputs = 1;
    const int T = 3, V = 5;
    std::vector<float> clean(static_cast<size_t>(T) * 3 * V, -1.0f);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < 3; ++k) clean[(static_cast<size_t>(t) * 3 + static_cast<size_t>(k)) * V + static_cast<size_t>(k + 2)] = -0.1f;
    }
    CHECK(!decode_throws(opt, clean, T, V));

    for (KernelPath path : [] { auto p = simd_paths(); p.push_back(KernelPath::Scalar); return p; }()) {
        KernelScope scope(path);
        // Beams 1 and 2 are not live at t = 0: their rows are never read.
        std::vector<float> x = clean;
        x[static_cast<size_t>(1) * V + 3] = std::numeric_limits<float>::quiet_NaN();
        x[static_cast<size_t>(2) * V + 0] = kInf;
        CHECK(!decode_throws(opt, x, T, V));
        // Every element of a row that is read counts, including banned or masked ones.
        for (int v = 0; v < V; ++v) {
            std::vector<float> y = clean;
            y[static_cast<size_t>(v)] = (v % 2) ? kInf : std::numeric_limits<float>::quiet_NaN();
            CHECK(decode_throws(opt, y, T, V));
        }
        // Without validation the entry is skipped instead.
        std::vector<float> y = clean;
        y[0] = std::numeric_limits<float>::quiet_NaN();
        BeamOptions lax = opt;
        lax.validate_inputs = 0;
        CHECK(!decode_throws(lax, y, T, V));
    }
}

void test_backward_uses_result_beam_size() {
    // A result decoded with beam 2 goes through a decoder configured for beam 4.
    BeamOptions small;
    small.beam_size = 2;
    BeamOptions big;
    big.beam_size = 4;
    const int T = 3, V = 6;
    std::vector<float> x(static_cast<size_t>(T) * 2 * V);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -0.1f * static_cast<float>((i * 7) % 11);
    const DecodeResult r = BeamSearchDecoder(small).decode(x.data(), T, V);
    const float g[2] = {1.0f, 0.5f};
    const BackwardResult a = BeamSearchDecoder(small).backward_sparse(r, nullptr, nullptr, g);
    const BackwardResult b = BeamSearchDecoder(big).backward_sparse(r, nullptr, nullptr, g);
    CHECK(same_values(a.sparse_logprob_indices, b.sparse_logprob_indices));
    CHECK(same_floats(a.sparse_logprob_values, b.sparse_logprob_values));
    const BackwardResult d = BeamSearchDecoder(big).backward(r, nullptr, nullptr, g);
    CHECK(d.grad_log_probs.size() == x.size());
}

void test_model_steps_match_tensor_decode() {
    // A model whose rows depend on each beam's full prefix: decoding with the
    // callback must equal decoding the rows it produced for the chosen beams.
    for (int trial = 0; trial < 200; ++trial) {
        BeamOptions opt;
        opt.beam_size = uniform_int(1, 5);
        opt.eos_token = uniform_int(0, 1) ? -1 : 0;
        opt.length_penalty_alpha = uniform_int(0, 1) ? 0.0f : 0.7f;
        const int K = opt.beam_size;
        const int T = uniform_int(1, 7);
        const int V = uniform_int(2, 12);
        std::vector<float> rows_seen(static_cast<size_t>(T) * K * V);
        const ModelStepFunction model = [&](const ModelStepInfo& info, float* rows) {
            for (int k = 0; k < K; ++k) {
                uint32_t h = 2166136261u;
                for (int s = 0; s < info.step; ++s) h = (h ^ static_cast<uint32_t>(info.prefixes[static_cast<size_t>(k) * info.step + s] + 7)) * 16777619u;
                if (info.step > 0 && info.parents[k] >= 0) {
                    // The prefix of slot k ends with the token it emitted last step.
                    CHECK(info.prefixes[static_cast<size_t>(k) * info.step + info.step - 1] == info.tokens[k]);
                }
                for (int v = 0; v < V; ++v) {
                    h = (h ^ static_cast<uint32_t>(v)) * 16777619u;
                    rows[static_cast<size_t>(k) * V + v] = -static_cast<float>(h % 97u) / 16.0f;
                }
            }
            std::copy(rows, rows + static_cast<size_t>(K) * V, rows_seen.begin() + static_cast<ptrdiff_t>(info.step) * K * V);
        };
        const BeamSearchDecoder decoder(opt);
        const DecodeResult a = decoder.decode_model_steps(T, V, model, nullptr, nullptr);
        const DecodeResult b = decoder.decode(rows_seen.data(), T, V);
        CHECK(same_values(a.tokens, b.tokens) && same_values(a.parents, b.parents));
        CHECK(same_floats(a.final_scores, b.final_scores) && same_floats(a.scores, b.scores));
    }
}

void test_step_matches_decode() {
    // Stepping through the rows of a tensor, with the prefixes tracked by the
    // caller, must select exactly what decode() selects.
    int cases = 0;
    for (int trial = 0; trial < 600; ++trial) {
        BeamOptions opt;
        opt.beam_size = uniform_int(1, 6);
        const int K = opt.beam_size;
        const int V = uniform_int(2, 40);
        const int T = uniform_int(1, 8);
        opt.eos_token = uniform_int(0, 2) == 0 ? -1 : uniform_int(0, V - 1);
        opt.min_length = uniform_int(0, 3);
        opt.length_penalty_alpha = uniform_int(0, 1) ? 0.0f : uniform_float(0.1f, 1.5f);
        opt.validate_inputs = 0;
        std::vector<float> x(static_cast<size_t>(T) * K * V);
        for (float& v : x) v = uniform_int(0, 9) == 0 ? -kInf : -0.25f * static_cast<float>(uniform_int(0, 12));
        std::vector<uint8_t> banned(static_cast<size_t>(V));
        for (uint8_t& b : banned) b = uniform_int(0, 5) == 0;
        DecodeConstraints c;
        c.banned_tokens = uniform_int(0, 1) ? banned.data() : nullptr;
        const float penalties[] = {1.0f, 1.0f, 1.2f, 2.5f};
        c.repetition_penalty = penalties[uniform_int(0, 3)];
        c.no_repeat_ngram_size = uniform_int(0, 4);
        c.token_filter = uniform_int(0, 3) == 0 ? hash_filter : nullptr;

        for (KernelPath path : {KernelPath::Scalar, simd_paths().empty() ? KernelPath::Scalar : simd_paths().front()}) {
            KernelScope scope(path);
            const BeamSearchDecoder decoder(opt);
            const DecodeResult expected = decoder.decode_constrained(x.data(), T, V, &c);

            std::vector<float> raw(static_cast<size_t>(K), -kInf);
            raw[0] = 0.0f;
            std::vector<int32_t> lengths(static_cast<size_t>(K), 0);
            std::vector<uint8_t> finished(static_cast<size_t>(K), 0);
            std::vector<int32_t> paths(static_cast<size_t>(K) * T, -1), next(paths.size());
            std::vector<int32_t> tok(static_cast<size_t>(K)), par(tok.size()), len(tok.size());
            std::vector<float> sc(tok.size()), rs(tok.size());
            std::vector<uint8_t> flp(tok.size());
            std::vector<float> final_scores(tok.size());
            bool ok = true;
            for (int t = 0; t < T; ++t) {
                TraceOutputs out;
                out.tokens = tok.data();
                out.parents = par.data();
                out.lengths = len.data();
                out.scores = sc.data();
                out.raw_scores = rs.data();
                out.from_logprob = flp.data();
                decoder.step(x.data() + static_cast<size_t>(t) * K * V, V, t, &c, raw.data(), lengths.data(), finished.data(),
                             paths.data(), T, out, t == T - 1 ? final_scores.data() : nullptr);
                for (int k = 0; k < K; ++k) {
                    const size_t e = static_cast<size_t>(t) * K + k;
                    ok = ok && tok[k] == expected.tokens[e] && par[k] == expected.parents[e] && len[k] == expected.lengths[e] &&
                         bits(sc[k]) == bits(expected.scores[e]) && bits(rs[k]) == bits(expected.raw_scores[e]) &&
                         flp[k] == expected.from_logprob[e];
                    int32_t* dst = &next[static_cast<size_t>(k) * T];
                    if (par[k] < 0) {
                        std::fill(dst, dst + T, -1);
                    } else {
                        std::copy_n(&paths[static_cast<size_t>(par[k]) * T], T, dst);
                        if (flp[k]) dst[t] = tok[k];
                    }
                }
                paths.swap(next);
            }
            ok = ok && same_floats(raw, expected.final_raw_scores) && same_floats(final_scores, expected.final_scores);
            if (!ok) {
                std::cerr << "step() differs from decode() on " << kernel_path_name(path) << " (trial " << trial << ")\n";
                std::abort();
            }
            ++cases;
        }
    }
    std::cout << "  step interface: " << cases << " stepped decodes equal decode()\n";
}

void test_length_penalty() {
    // Within one float ulp of pow() on the exponents and lengths decoders use.
    for (float alpha : {0.1f, 0.25f, 0.5f, 0.6f, 0.7f, 1.0f, 1.3f, 2.0f, 3.7f}) {
        for (int len = 0; len <= 2000; ++len) {
            const double base = (5.0 + static_cast<double>(std::max(len, 1))) / 6.0;
            const float expected = static_cast<float>(std::pow(base, static_cast<double>(alpha)));
            const float actual = gnmt_length_penalty(len, alpha);
            const uint32_t a = bits(actual), e = bits(expected);
            CHECK((a > e ? a - e : e - a) <= 1u);
        }
    }
    // Exponents far beyond int range: no float-to-int overflow (the sanitizer
    // build checks the conversion), and the result saturates to +inf.
    for (float alpha : {64.5f, 1.0e4f, 3.0e9f, 1.0e30f, std::numeric_limits<float>::max()}) {
        CHECK(gnmt_length_penalty(1, alpha) == 1.0f);
        CHECK(gnmt_length_penalty(100, alpha) == kInf);
    }
}

} // namespace

int main() {
    std::cout << "kernel paths:";
    for (KernelPath p : simd_paths()) std::cout << ' ' << kernel_path_name(p);
    std::cout << " (plus scalar)\n";
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    CHECK(DBS_CAN_COMPILE_AVX2 && DBS_CAN_COMPILE_SSE42);
#endif
    test_row_scan_parity();
    test_constrained_decode_matches_reference();
    test_decode_backward_parity();
    test_validation_covers_the_rows_read();
    test_backward_uses_result_beam_size();
    test_model_steps_match_tensor_decode();
    test_length_penalty();
    test_step_matches_decode();
    std::cout << "dbs_internal_tests passed\n";
    return 0;
}
