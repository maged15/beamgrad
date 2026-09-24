// SPDX-License-Identifier: MIT
//
// The CPU beam-search decoder: hard top-k forward with deterministic ordering,
// and the sparse/dense surrogate backward passes.
#include "decoder.hpp"

#include <string>

namespace dbs {

namespace {

bool prefix_contains_token(const std::vector<int32_t>& prefix, int token) {
    return std::find(prefix.begin(), prefix.end(), token) != prefix.end();
}

bool would_repeat_ngram(const std::vector<int32_t>& prefix, int token, int n) {
    if (n <= 0) return false;
    if (n == 1) return prefix_contains_token(prefix, token);
    const int prefix_len = static_cast<int>(prefix.size());
    if (prefix_len + 1 < n) return false;

    const int suffix_start = prefix_len - (n - 1);

    for (int i = 0; i + n <= prefix_len; ++i) {
        bool same = true;
        for (int j = 0; j < n - 1; ++j) {
            if (prefix[static_cast<size_t>(i + j)] != prefix[static_cast<size_t>(suffix_start + j)]) {
                same = false;
                break;
            }
        }
        if (same && prefix[static_cast<size_t>(i + n - 1)] != token) same = false;
        if (same) return true;
    }
    return false;
}

bool token_allowed_by_advanced_constraints(
    const DecodeConstraints* constraints,
    int step,
    int parent,
    const std::vector<int32_t>& prefix,
    int token
) {
    if (!constraints) return true;
    if (constraints->no_repeat_ngram_size > 0 &&
        would_repeat_ngram(prefix, token, constraints->no_repeat_ngram_size)) {
        return false;
    }
    if (constraints->token_filter) {
        const int rc = constraints->token_filter(
            constraints->token_filter_user_data,
            constraints->batch_index,
            step,
            parent,
            prefix.empty() ? nullptr : prefix.data(),
            static_cast<int>(prefix.size()),
            token);
        if (rc == 0) return false;
    }
    return true;
}

float apply_repetition_penalty(float lp, const DecodeConstraints* constraints, const std::vector<int32_t>& prefix, int token) {
    if (!constraints || !(constraints->repetition_penalty > 1.0f)) return lp;
    if (!prefix_contains_token(prefix, token)) return lp;
    return lp - std::log(constraints->repetition_penalty);
}

void validate_logprob_tensor(const float* x, int steps, int beam_size, int vocab_size) {
    if (!x) throw std::invalid_argument("log_probs cannot be null");
    const size_t st = static_cast<size_t>(steps);
    const size_t k = static_cast<size_t>(beam_size);
    const size_t v = static_cast<size_t>(vocab_size);
    const size_t count = checked_mul_size(checked_mul_size(st, k, "log-prob tensor size overflow"), v, "log-prob tensor size overflow");

    // Sample up to 1000 evenly-spaced elements to keep validation O(1) in tensor size.
    constexpr size_t kMaxSamples = 1000;
    const size_t stride = count > kMaxSamples ? count / kMaxSamples : 1;

    for (size_t i = 0; i < count; i += stride) {
        const float value = x[i];
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
            throw std::invalid_argument("log_probs contains NaN or +Inf");
        }
    }
}

void scan_parent_row_scalar_advanced(
    const float* row,
    float parent_raw,
    int parent_length,
    int parent,
    int step,
    int vocab_size,
    Candidate* top,
    int top_count,
    int vocab_block,
    float length_penalty_alpha,
    const DecodeConstraints* constraints,
    const std::vector<int32_t>& prefix,
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
            if (constraints && constraints->banned_tokens && constraints->banned_tokens[v]) continue;
            if (eos_token >= 0 && v == eos_token && new_len < min_length) continue;
            if (!token_allowed_by_advanced_constraints(constraints, step, parent, prefix, v)) continue;

            float lp = row[v];
            if (!std::isfinite(lp)) continue;
            lp = apply_repetition_penalty(lp, constraints, prefix, v);

            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}

void scatter_raw_candidate_grad(
    const DecodeResult& fwd,
    float* grad_log_probs,
    float* prev_raw_grad,
    int t,
    int parent,
    int token,
    uint8_t from_logprob,
    float draw
) {
    if (parent < 0 || draw == 0.0f) return;

    const int K = fwd.beam_size;
    const int V = fwd.vocab_size;

    prev_raw_grad[parent] += draw;

    if (from_logprob && token >= 0) {
        const size_t grad_idx =
            (static_cast<size_t>(t) * K + parent) * V + token;

        grad_log_probs[grad_idx] += draw;
    }
}

void scatter_raw_candidate_grad_sparse(
    const DecodeResult& fwd,
    std::vector<SparseGradEntry>& entries,
    float* prev_raw_grad,
    int t,
    int parent,
    int token,
    uint8_t from_logprob,
    float draw
) {
    if (parent < 0 || draw == 0.0f) return;

    const int K = fwd.beam_size;
    const int V = fwd.vocab_size;

    prev_raw_grad[parent] += draw;

    if (from_logprob && token >= 0) {
        const int64_t grad_idx =
            (static_cast<int64_t>(t) * K + parent) * static_cast<int64_t>(V) + token;
        entries.push_back(SparseGradEntry{grad_idx, draw});
    }
}

void finalize_sparse_entries(
    std::vector<SparseGradEntry>& entries,
    BackwardResult& out
) {
    if (entries.empty()) return;

    std::sort(
        entries.begin(),
        entries.end(),
        [](const SparseGradEntry& a, const SparseGradEntry& b) {
            return a.index < b.index;
        }
    );

    out.sparse_logprob_indices.reserve(entries.size());
    out.sparse_logprob_values.reserve(entries.size());

    int64_t cur = entries[0].index;
    float sum = 0.0f;

    for (const SparseGradEntry& e : entries) {
        if (e.index == cur) {
            sum += e.value;
        } else {
            out.sparse_logprob_indices.push_back(cur);
            out.sparse_logprob_values.push_back(sum);
            cur = e.index;
            sum = e.value;
        }
    }

    out.sparse_logprob_indices.push_back(cur);
    out.sparse_logprob_values.push_back(sum);
}

void backward_selected(
    const DecodeResult& fwd,
    const float* grad_selected_weights,
    const float* grad_final_scores,
    const float* next_raw_grad,
    float* prev_raw_grad,
    float* grad_log_probs,
    int t
) {
    const int K = fwd.beam_size;

    const float* weights =
        fwd.weights.data() + static_cast<size_t>(t) * K;

    const float* gw =
        grad_selected_weights
            ? grad_selected_weights + static_cast<size_t>(t) * K
            : nullptr;

    const float weighted_dot =
        gw ? dot(weights, gw, K) : 0.0f;

    for (int k = 0; k < K; ++k) {
        const size_t idx = static_cast<size_t>(t) * K + k;

        const int parent = fwd.parents[idx];
        const int token = fwd.tokens[idx];

        if (parent < 0) continue;

        float drank = 0.0f;

        if (gw) {
            const float w = weights[k];
            drank +=
                (w * (gw[k] - weighted_dot)) /
                fwd.selected_temperature;
        }

        if (grad_final_scores && t == fwd.steps - 1) {
            drank += grad_final_scores[k];
        }

        const int len = std::max(1, static_cast<int>(fwd.lengths[idx]));
        const float inv_penalty =
            1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

        const float draw = next_raw_grad[k] + drank * inv_penalty;

        scatter_raw_candidate_grad(
            fwd,
            grad_log_probs,
            prev_raw_grad,
            t,
            parent,
            token,
            fwd.from_logprob[idx],
            draw
        );
    }
}

void backward_relaxed_topk(
    const DecodeResult& fwd,
    const float* grad_relaxed_weights,
    float* prev_raw_grad,
    float* grad_log_probs,
    int t
) {
    if (!grad_relaxed_weights) return;

    constexpr float EPS = 1.0e-12f;

    const int P = fwd.relaxed_pool_size;

    const float* w =
        fwd.relaxed_weights.data() + static_cast<size_t>(t) * P;

    const float* gw =
        grad_relaxed_weights + static_cast<size_t>(t) * P;

    float denom = 0.0f;
    float numer = 0.0f;

    for (int p = 0; p < P; ++p) {
        const float a = w[p] * (1.0f - w[p]);
        denom += a;
        numer += gw[p] * a;
    }

    if (denom <= EPS) return;

    const float center = numer / denom;
    const float inv_temp = 1.0f / fwd.soft_topk_temperature;

    for (int p = 0; p < P; ++p) {
        const size_t idx = static_cast<size_t>(t) * P + p;

        const int parent = fwd.pool_parents[idx];
        const int token = fwd.pool_tokens[idx];

        if (parent < 0) continue;

        const float a = w[p] * (1.0f - w[p]);
        const float drank = a * inv_temp * (gw[p] - center);

        if (drank == 0.0f) continue;

        const int len = std::max(1, static_cast<int>(fwd.pool_lengths[idx]));
        const float inv_penalty =
            1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

        scatter_raw_candidate_grad(
            fwd,
            grad_log_probs,
            prev_raw_grad,
            t,
            parent,
            token,
            fwd.pool_from_logprob[idx],
            drank * inv_penalty
        );
    }
}

void backward_selected_sparse(
    const DecodeResult& fwd,
    const float* grad_selected_weights,
    const float* grad_final_scores,
    const float* next_raw_grad,
    float* prev_raw_grad,
    std::vector<SparseGradEntry>& entries,
    int t
) {
    const int K = fwd.beam_size;

    const float* weights =
        fwd.weights.data() + static_cast<size_t>(t) * K;

    const float* gw =
        grad_selected_weights
            ? grad_selected_weights + static_cast<size_t>(t) * K
            : nullptr;

    const float weighted_dot = gw ? dot(weights, gw, K) : 0.0f;

    for (int k = 0; k < K; ++k) {
        const size_t idx = static_cast<size_t>(t) * K + k;

        const int parent = fwd.parents[idx];
        const int token = fwd.tokens[idx];

        if (parent < 0) continue;

        float drank = 0.0f;

        if (gw) {
            const float w = weights[k];
            drank += (w * (gw[k] - weighted_dot)) / fwd.selected_temperature;
        }

        if (grad_final_scores && t == fwd.steps - 1) {
            drank += grad_final_scores[k];
        }

        const int len = std::max(1, static_cast<int>(fwd.lengths[idx]));
        const float inv_penalty = 1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

        const float draw = next_raw_grad[k] + drank * inv_penalty;

        scatter_raw_candidate_grad_sparse(
            fwd,
            entries,
            prev_raw_grad,
            t,
            parent,
            token,
            fwd.from_logprob[idx],
            draw
        );
    }
}

void backward_relaxed_topk_sparse(
    const DecodeResult& fwd,
    const float* grad_relaxed_weights,
    float* prev_raw_grad,
    std::vector<SparseGradEntry>& entries,
    int t
) {
    if (!grad_relaxed_weights) return;

    constexpr float EPS = 1.0e-12f;

    const int P = fwd.relaxed_pool_size;

    const float* w =
        fwd.relaxed_weights.data() + static_cast<size_t>(t) * P;

    const float* gw =
        grad_relaxed_weights + static_cast<size_t>(t) * P;

    float denom = 0.0f;
    float numer = 0.0f;

    for (int p = 0; p < P; ++p) {
        const float a = w[p] * (1.0f - w[p]);
        denom += a;
        numer += gw[p] * a;
    }

    if (denom <= EPS) return;

    const float center = numer / denom;
    const float inv_temp = 1.0f / fwd.soft_topk_temperature;

    for (int p = 0; p < P; ++p) {
        const size_t idx = static_cast<size_t>(t) * P + p;

        const int parent = fwd.pool_parents[idx];
        const int token = fwd.pool_tokens[idx];

        if (parent < 0) continue;

        const float a = w[p] * (1.0f - w[p]);
        const float drank = a * inv_temp * (gw[p] - center);

        if (drank == 0.0f) continue;

        const int len = std::max(1, static_cast<int>(fwd.pool_lengths[idx]));
        const float inv_penalty = 1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

        scatter_raw_candidate_grad_sparse(
            fwd,
            entries,
            prev_raw_grad,
            t,
            parent,
            token,
            fwd.pool_from_logprob[idx],
            drank * inv_penalty
        );
    }
}

std::vector<std::vector<int32_t>> build_sequences(const DecodeResult& r) {
    const int T = r.steps;
    const int K = r.beam_size;

    std::vector<std::vector<int32_t>> seqs(K);

    for (int k = 0; k < K; ++k) {
        std::vector<int32_t> seq(T, -1);

        int cur = k;

        for (int t = T - 1; t >= 0; --t) {
            if (cur < 0) break;

            const size_t idx = static_cast<size_t>(t) * K + cur;

            seq[t] = r.tokens[idx];
            cur = r.parents[idx];
        }

        if (r.eos_token >= 0) {
            auto it = std::find(seq.begin(), seq.end(), r.eos_token);

            if (it != seq.end()) {
                seq.erase(it, seq.end());
            }
        }
        seq.erase(
            seq.begin(),
            std::find_if(seq.begin(), seq.end(), [](int32_t token) { return token >= 0; })
        );

        seqs[k] = std::move(seq);
    }

    return seqs;
}

} // namespace

BeamSearchDecoder::BeamSearchDecoder(BeamOptions options)
    : opt_(options) {
    if (opt_.beam_size <= 0) {
        throw std::invalid_argument("beam_size must be positive");
    }

    if (!(opt_.selected_temperature > 0.0f)) {
        throw std::invalid_argument("selected_temperature must be positive");
    }

    if (!(opt_.soft_topk_temperature > 0.0f)) {
        throw std::invalid_argument("soft_topk_temperature must be positive");
    }

    if (opt_.soft_topk_max_iters <= 0) {
        throw std::invalid_argument("soft_topk_max_iters must be positive");
    }

    if (!(opt_.soft_topk_tolerance > 0.0f)) {
        throw std::invalid_argument("soft_topk_tolerance must be positive");
    }

    if (opt_.length_penalty_alpha < 0.0f) {
        throw std::invalid_argument("length_penalty_alpha cannot be negative");
    }

    if (opt_.min_length < 0) {
        throw std::invalid_argument("min_length cannot be negative");
    }

    if (opt_.max_dense_gradient_elements <= 0) {
        throw std::invalid_argument("max_dense_gradient_elements must be positive");
    }

    opt_.vocab_block = std::max(16, opt_.vocab_block);
    opt_.relaxed_pool_multiplier = std::max(1, opt_.relaxed_pool_multiplier);
}

DecodeResult BeamSearchDecoder::decode(const float* log_probs, int steps, int vocab_size) const {
    return decode_constrained(log_probs, steps, vocab_size, nullptr);
}

DecodeResult BeamSearchDecoder::decode_constrained(const float* log_probs, int steps, int vocab_size, const DecodeConstraints* constraints) const {
    if (!log_probs) throw std::invalid_argument("log_probs cannot be null");
    if (steps <= 0) throw std::invalid_argument("steps must be positive");
    if (vocab_size <= 0) throw std::invalid_argument("vocab_size must be positive");

    if (opt_.eos_token >= vocab_size) {
        throw std::invalid_argument("eos_token is outside vocab");
    }

    if (constraints && constraints->forced_tokens) {
        for (int t = 0; t < steps; ++t) {
            const int tok = constraints->forced_tokens[t];
            if (tok >= vocab_size) throw std::invalid_argument("forced token is outside vocab");
        }
    }

    if (opt_.validate_inputs) {
        validate_logprob_tensor(log_probs, steps, opt_.beam_size, vocab_size);
    }

    const int effective_min_length = constraints && constraints->min_length >= 0 ? constraints->min_length : opt_.min_length;

    const int K = opt_.beam_size;
    const int V = vocab_size;
    const size_t relaxed_pool_size = checked_mul_size(static_cast<size_t>(K), static_cast<size_t>(opt_.relaxed_pool_multiplier), "relaxed pool size overflow");
    if (relaxed_pool_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("relaxed pool size exceeds int range");
    }
    const int P = std::max(K, static_cast<int>(relaxed_pool_size));

    const size_t selected_count = checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(K), "selected result size overflow");
    const size_t pool_count = checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(P), "pool result size overflow");
    (void)checked_mul_size(selected_count, static_cast<size_t>(V), "log-prob tensor size overflow");

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    DecodeResult result;
    result.steps = steps;
    result.beam_size = K;
    result.vocab_size = V;
    result.eos_token = opt_.eos_token;
    result.relaxed_pool_size = P;
    result.selected_temperature = opt_.selected_temperature;
    result.soft_topk_temperature = opt_.soft_topk_temperature;
    result.length_penalty_alpha = opt_.length_penalty_alpha;

    result.parents.assign(selected_count, -1);
    result.tokens.assign(selected_count, -1);
    result.lengths.assign(selected_count, 0);

    result.raw_scores.assign(selected_count, NEG_INF);
    result.scores.assign(selected_count, NEG_INF);
    result.weights.assign(selected_count, 0.0f);

    result.final_raw_scores.assign(K, NEG_INF);
    result.final_scores.assign(K, NEG_INF);

    result.from_logprob.assign(selected_count, 0);

    result.pool_parents.assign(pool_count, -1);
    result.pool_tokens.assign(pool_count, -1);
    result.pool_lengths.assign(pool_count, 0);

    result.pool_raw_scores.assign(pool_count, NEG_INF);
    result.pool_scores.assign(pool_count, NEG_INF);
    result.relaxed_weights.assign(pool_count, 0.0f);

    result.pool_from_logprob.assign(pool_count, 0);

    AlignedFloatVector prev_raw_scores(K, NEG_INF);
    AlignedFloatVector next_raw_scores(K, NEG_INF);

    AlignedIntVector prev_lengths(K, 0);
    AlignedIntVector next_lengths(K, 0);

    std::vector<uint8_t> ended_prev(K, 0);
    std::vector<uint8_t> ended_next(K, 0);
    std::vector<std::vector<int32_t>> prev_sequences(K);
    std::vector<std::vector<int32_t>> next_sequences(K);

    prev_raw_scores[0] = 0.0f;

    const bool has_advanced_constraints = constraints && (
        constraints->repetition_penalty > 1.0f ||
        constraints->no_repeat_ngram_size > 0 ||
        constraints->token_filter != nullptr
    );

    std::vector<Candidate> top(P);

    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < P; ++i) {
            top[i] = Candidate{NEG_INF, NEG_INF, -1, -1, 0, 0};
        }

        const float* step_base = log_probs + static_cast<size_t>(t) * K * V;

        for (int b = 0; b < K; ++b) {
            const float parent_raw = prev_raw_scores[b];

            if (!std::isfinite(parent_raw)) continue;

            if (opt_.eos_token >= 0 && ended_prev[b]) {
                const int len = std::max(1, static_cast<int>(prev_lengths[b]));
                const float rank =
                    parent_raw / gnmt_length_penalty(len, opt_.length_penalty_alpha);

                insert_topk(
                    top.data(),
                    P,
                    Candidate{rank, parent_raw, b, opt_.eos_token, len, 0}
                );

                continue;
            }

            const float* row = step_base + static_cast<size_t>(b) * V;

            const int forced_token = constraints && constraints->forced_tokens ? constraints->forced_tokens[t] : -1;

            if (has_advanced_constraints) {
                scan_parent_row_scalar_advanced(
                    row,
                    parent_raw,
                    prev_lengths[b],
                    b,
                    t,
                    V,
                    top.data(),
                    P,
                    opt_.vocab_block,
                    opt_.length_penalty_alpha,
                    constraints,
                    prev_sequences[b],
                    forced_token,
                    opt_.eos_token,
                    effective_min_length
                );
            } else {
                scan_parent_row(
                    row,
                    parent_raw,
                    prev_lengths[b],
                    b,
                    V,
                    top.data(),
                    P,
                    opt_.vocab_block,
                    opt_.length_penalty_alpha,
                    constraints ? constraints->banned_tokens : nullptr,
                    forced_token,
                    opt_.eos_token,
                    effective_min_length
                );
            }
        }

        for (int p = 0; p < P; ++p) {
            const size_t idx = static_cast<size_t>(t) * P + p;

            result.pool_parents[idx] = top[p].parent;
            result.pool_tokens[idx] = top[p].token;
            result.pool_lengths[idx] = top[p].length;

            result.pool_raw_scores[idx] = top[p].raw_score;
            result.pool_scores[idx] = top[p].score;

            result.pool_from_logprob[idx] = top[p].from_logprob;
        }

        soft_topk_inclusion(
            result.pool_scores.data() + static_cast<size_t>(t) * P,
            result.relaxed_weights.data() + static_cast<size_t>(t) * P,
            P,
            K,
            opt_.soft_topk_temperature,
            opt_.soft_topk_tolerance,
            opt_.soft_topk_max_iters
        );

        for (int k = 0; k < K; ++k) {
            const size_t idx = static_cast<size_t>(t) * K + k;

            result.parents[idx] = top[k].parent;
            result.tokens[idx] = top[k].token;
            result.lengths[idx] = top[k].length;

            result.raw_scores[idx] = top[k].raw_score;
            result.scores[idx] = top[k].score;

            result.from_logprob[idx] = top[k].from_logprob;

            next_raw_scores[k] = top[k].raw_score;
            next_lengths[k] = top[k].length;

            next_sequences[k].clear();
            if (top[k].parent >= 0) {
                next_sequences[k] = prev_sequences[static_cast<size_t>(top[k].parent)];
                const bool parent_ended = ended_prev[top[k].parent] != 0;
                const bool token_is_eos =
                    opt_.eos_token >= 0 && top[k].token == opt_.eos_token;

                if (top[k].from_logprob && top[k].token >= 0) {
                    next_sequences[k].push_back(top[k].token);
                }

                ended_next[k] = static_cast<uint8_t>(parent_ended || token_is_eos);
            } else {
                ended_next[k] = 0;
            }
        }

        softmax_selected(
            result.scores.data() + static_cast<size_t>(t) * K,
            result.weights.data() + static_cast<size_t>(t) * K,
            K,
            opt_.selected_temperature
        );

        std::swap(prev_raw_scores, next_raw_scores);
        std::fill(next_raw_scores.begin(), next_raw_scores.end(), NEG_INF);

        std::swap(prev_lengths, next_lengths);
        std::fill(next_lengths.begin(), next_lengths.end(), 0);

        ended_prev.swap(ended_next);
        std::fill(ended_next.begin(), ended_next.end(), uint8_t{0});

        prev_sequences.swap(next_sequences);
        for (auto& seq : next_sequences) seq.clear();
    }

    result.final_raw_scores = prev_raw_scores;

    for (int k = 0; k < K; ++k) {
        const int len = std::max(1, static_cast<int>(prev_lengths[k]));
        result.final_scores[k] =
            prev_raw_scores[k] / gnmt_length_penalty(len, opt_.length_penalty_alpha);
    }

    result.sequences = build_sequences(result);

    return result;
}

BackwardResult BeamSearchDecoder::backward(
    const DecodeResult& fwd,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores
) const {
    const int T = fwd.steps;
    const int K = fwd.beam_size;
    const int V = fwd.vocab_size;
    const int P = fwd.relaxed_pool_size;

    if (K != opt_.beam_size) {
        throw std::invalid_argument("forward result beam_size does not match decoder");
    }

    if (P < K) {
        throw std::invalid_argument("invalid relaxed pool size");
    }

    const size_t selected_count = checked_mul_size(static_cast<size_t>(T), static_cast<size_t>(K), "selected gradient size overflow");
    const size_t dense_grad_count = checked_mul_size(selected_count, static_cast<size_t>(V), "dense gradient size overflow");
    if (opt_.max_dense_gradient_elements >= 0 &&
        dense_grad_count > static_cast<size_t>(opt_.max_dense_gradient_elements)) {
        throw std::length_error("dense gradient allocation exceeds max_dense_gradient_elements; use sparse backward");
    }

    BackwardResult out;
    out.sparse = false;
    out.grad_log_probs.assign(dense_grad_count, 0.0f);
    out.grad_initial_scores.assign(K, 0.0f);

    AlignedFloatVector next_raw_grad(K, 0.0f);
    AlignedFloatVector prev_raw_grad(K, 0.0f);

    for (int t = T - 1; t >= 0; --t) {
        std::fill(prev_raw_grad.begin(), prev_raw_grad.end(), 0.0f);

        backward_selected(
            fwd,
            grad_selected_weights,
            grad_final_scores,
            next_raw_grad.data(),
            prev_raw_grad.data(),
            out.grad_log_probs.data(),
            t
        );

        backward_relaxed_topk(
            fwd,
            grad_relaxed_weights,
            prev_raw_grad.data(),
            out.grad_log_probs.data(),
            t
        );

        next_raw_grad.swap(prev_raw_grad);
    }

    out.grad_initial_scores = next_raw_grad;
    return out;
}

BackwardResult BeamSearchDecoder::backward_sparse(
    const DecodeResult& fwd,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores
) const {
    const int T = fwd.steps;
    const int K = fwd.beam_size;
    const int P = fwd.relaxed_pool_size;

    if (K != opt_.beam_size) {
        throw std::invalid_argument("forward result beam_size does not match decoder");
    }

    if (P < K) {
        throw std::invalid_argument("invalid relaxed pool size");
    }

    BackwardResult out;
    out.sparse = true;
    out.grad_initial_scores.assign(K, 0.0f);

    std::vector<SparseGradEntry> entries;
    entries.reserve(static_cast<size_t>(T) * static_cast<size_t>(K + P));

    AlignedFloatVector next_raw_grad(K, 0.0f);
    AlignedFloatVector prev_raw_grad(K, 0.0f);

    for (int t = T - 1; t >= 0; --t) {
        std::fill(prev_raw_grad.begin(), prev_raw_grad.end(), 0.0f);

        backward_selected_sparse(
            fwd,
            grad_selected_weights,
            grad_final_scores,
            next_raw_grad.data(),
            prev_raw_grad.data(),
            entries,
            t
        );

        backward_relaxed_topk_sparse(
            fwd,
            grad_relaxed_weights,
            prev_raw_grad.data(),
            entries,
            t
        );

        next_raw_grad.swap(prev_raw_grad);
    }

    out.grad_initial_scores = next_raw_grad;
    finalize_sparse_entries(entries, out);
    return out;
}

} // namespace dbs
