// SPDX-License-Identifier: MIT
//
// The CPU beam-search decoder: hard top-k forward with deterministic ordering,
// and the sparse/dense surrogate backward passes.
#include "decoder.hpp"

#include <string>

namespace dbs {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Constraint scratch bits, per token.
constexpr uint8_t kBanned = 1;     // banned_tokens
constexpr uint8_t kNgram = 2;      // would repeat an n-gram of the beam's prefix
constexpr uint8_t kPenalised = 4;  // already in the beam's prefix (repetition penalty)

std::string location(int step, int beam) {
    return " (step " + std::to_string(step) + ", beam " + std::to_string(beam) + ")";
}

// Hard beam search over one example, one step at a time.
class BeamSearch {
public:
    BeamSearch(
        const BeamOptions& opt,
        int vocab_size,
        const DecodeConstraints* constraints,
        const TraceOutputs& out,
        bool with_pool)
        : opt_(opt),
          c_(constraints),
          out_(out),
          K_(opt.beam_size),
          V_(vocab_size),
          P_(with_pool && opt.relaxed_pool_multiplier > 0 ? opt.beam_size * opt.relaxed_pool_multiplier : opt.beam_size),
          with_pool_(with_pool && opt.relaxed_pool_multiplier > 0),
          alpha_(opt.length_penalty_alpha),
          eos_(opt.eos_token),
          min_length_(constraints && constraints->min_length >= 0 ? constraints->min_length : opt.min_length),
          raw_(K_, kNegInf),
          next_raw_(K_, kNegInf),
          len_(K_, 0),
          next_len_(K_, 0),
          ended_(K_, 0),
          next_ended_(K_, 0),
          parents_(K_, -1),
          tokens_(K_, -1),
          selected_scores_(K_, kNegInf),
          top_(static_cast<size_t>(P_)) {
        raw_[0] = 0.0f;
        if (with_pool_) pool_scores_.assign(static_cast<size_t>(P_), kNegInf);
        if (c_) {
            penalise_ = c_->repetition_penalty > 1.0f;
            log_penalty_ = penalise_ ? std::log(c_->repetition_penalty) : 0.0f;
            constrained_ = penalise_ || c_->no_repeat_ngram_size > 0 || c_->token_filter != nullptr;
        }
        if (constrained_) {
            mask_.assign(static_cast<size_t>(V_), 0);
            if (c_->banned_tokens) {
                for (int v = 0; v < V_; ++v) mask_[static_cast<size_t>(v)] = c_->banned_tokens[v] ? kBanned : 0;
            }
            prefix_.resize(static_cast<size_t>(K_));
            next_prefix_.resize(static_cast<size_t>(K_));
        }
    }

    int step_index() const noexcept { return t_; }

    // Resumes a search at `step` from an explicit state (see
    // BeamSearchDecoder::step); the trace of that step is written at index 0.
    void restore(int step, const float* raw, const int32_t* lengths, const uint8_t* finished,
                 const int32_t* prefixes, int prefix_stride) {
        t_ = step;
        origin_ = step;
        for (int k = 0; k < K_; ++k) {
            const size_t i = static_cast<size_t>(k);
            raw_[i] = raw[k];
            len_[i] = lengths[k];
            ended_[i] = finished[k] != 0 ? 1 : 0;
            if (constrained_) {
                const int n = std::max(0, std::min(lengths[k], prefix_stride));
                const int32_t* row = prefixes + i * static_cast<size_t>(prefix_stride);
                prefix_[i].assign(row, row + n);
            }
        }
    }

    void save(float* raw, int32_t* lengths, uint8_t* finished) const {
        for (int k = 0; k < K_; ++k) {
            const size_t i = static_cast<size_t>(k);
            raw[k] = raw_[i];
            lengths[k] = len_[i];
            finished[k] = ended_[i];
        }
    }
    const float* raw_scores() const noexcept { return raw_.data(); }
    const int32_t* lengths() const noexcept { return len_.data(); }
    const uint8_t* finished() const noexcept { return ended_.data(); }
    const int32_t* parents() const noexcept { return parents_.data(); }
    const int32_t* tokens() const noexcept { return tokens_.data(); }

    // rows: this step's [K, V] log-probabilities (row b extends beam b).
    void step(const float* rows) {
        const int t = t_;
        std::fill(top_.begin(), top_.end(), Candidate{kNegInf, kNegInf, -1, -1, 0, 0});
        const int forced = c_ && c_->forced_tokens ? c_->forced_tokens[t] : -1;

        for (int b = 0; b < K_; ++b) {
            const float parent_raw = raw_[static_cast<size_t>(b)];
            if (!std::isfinite(parent_raw)) continue;

            if (eos_ >= 0 && ended_[static_cast<size_t>(b)]) {
                // A finished beam competes with its unchanged score.
                const int len = std::max(1, static_cast<int>(len_[static_cast<size_t>(b)]));
                const float rank = parent_raw / gnmt_length_penalty(len, alpha_);
                insert_topk(top_.data(), P_, Candidate{rank, parent_raw, b, eos_, len, 0});
                continue;
            }

            RowScan s;
            s.row = rows + static_cast<size_t>(b) * static_cast<size_t>(V_);
            s.banned = c_ ? c_->banned_tokens : nullptr;
            s.parent_raw = parent_raw;
            s.new_length = len_[static_cast<size_t>(b)] + 1;
            s.inv_penalty = 1.0f / gnmt_length_penalty(s.new_length, alpha_);
            s.parent = b;
            s.vocab_size = V_;
            s.forced_token = forced;
            s.masked_token = eos_ >= 0 && s.new_length < min_length_ ? eos_ : -1;

            const bool invalid = constrained_ ? scan_constrained(s, t) : scan_row(s, top_.data(), P_);
            if (invalid && opt_.validate_inputs) {
                throw std::invalid_argument("log_probs contains NaN or +inf" + location(t, b));
            }
        }

        if (with_pool_) record_pool(t);
        select(t);
        ++t_;
    }

    void finish(float* final_scores, float* final_raw_scores, int32_t* final_lengths) const {
        for (int k = 0; k < K_; ++k) {
            const size_t i = static_cast<size_t>(k);
            const int len = std::max(1, static_cast<int>(len_[i]));
            if (final_scores) final_scores[k] = raw_[i] / gnmt_length_penalty(len, alpha_);
            if (final_raw_scores) final_raw_scores[k] = raw_[i];
            if (final_lengths) final_lengths[k] = len_[i];
        }
    }

private:
    // Prefix tokens outside the vocabulary (only possible in a caller-supplied
    // state, see restore) are ignored.
    void mark(int token, uint8_t bit) {
        if (token < 0 || token >= V_) return;
        uint8_t& m = mask_[static_cast<size_t>(token)];
        if ((m & (kNgram | kPenalised)) == 0) touched_.push_back(token);
        m = static_cast<uint8_t>(m | bit);
    }

    // Scan with n-gram blocking, repetition penalty and/or a token filter. The
    // blocked and penalised tokens of the beam are computed once per step, so
    // the row itself is scanned by the SIMD kernel unless a filter is set.
    bool scan_constrained(RowScan s, int t) {
        const std::vector<int32_t>& prefix = prefix_[static_cast<size_t>(s.parent)];
        const int L = static_cast<int>(prefix.size());
        const int n = c_->no_repeat_ngram_size;
        touched_.clear();
        penalised_.clear();

        if (n == 1) {
            for (int32_t token : prefix) mark(token, kNgram);
        } else if (n > 1 && L >= n - 1) {
            // Tokens that would complete an n-gram whose first n - 1 tokens equal
            // the prefix's last n - 1 tokens.
            const int suffix_start = L - (n - 1);
            for (int i = 0; i + n <= L; ++i) {
                bool same = true;
                for (int j = 0; j < n - 1 && same; ++j) {
                    same = prefix[static_cast<size_t>(i + j)] == prefix[static_cast<size_t>(suffix_start + j)];
                }
                if (same) mark(prefix[static_cast<size_t>(i + n - 1)], kNgram);
            }
        }
        if (penalise_) {
            for (int32_t token : prefix) {
                if (token < 0 || token >= V_) continue;
                if ((mask_[static_cast<size_t>(token)] & kPenalised) == 0) {
                    mark(token, kPenalised);
                    penalised_.push_back(token);
                }
            }
        }

        bool invalid = false;
        if (c_->token_filter) {
            invalid = scan_filtered(s, t, prefix);
        } else {
            // Blocked and penalised tokens are masked out of the fast scan; the
            // penalised ones are then added with their penalty.
            s.banned = mask_.data();
            invalid = scan_row(s, top_.data(), P_);
            for (int32_t v : penalised_) {
                if (mask_[static_cast<size_t>(v)] & (kBanned | kNgram)) continue;
                if (s.forced_token >= 0 && v != s.forced_token) continue;
                if (v == s.masked_token) continue;
                const float lp = s.row[v];
                if (!(lp < kInf) || lp == kNegInf) continue;
                const float raw = s.parent_raw + (lp - log_penalty_);
                insert_topk(top_.data(), P_, Candidate{raw * s.inv_penalty, raw, s.parent, v, s.new_length, 1});
            }
        }

        for (int32_t v : touched_) mask_[static_cast<size_t>(v)] &= kBanned;
        return invalid;
    }

    // Per-token path for a user token filter, called in the documented order:
    // after the forced/banned/min-length/n-gram checks, before the value check.
    bool scan_filtered(const RowScan& s, int t, const std::vector<int32_t>& prefix) {
        bool invalid = false;
        for (int v = 0; v < V_; ++v) {
            const float lp = s.row[v];
            if (!(lp < kInf)) invalid = true;
            if (s.forced_token >= 0 && v != s.forced_token) continue;
            const uint8_t m = mask_[static_cast<size_t>(v)];
            if (m & kBanned) continue;
            if (v == s.masked_token) continue;
            if (m & kNgram) continue;
            const int allowed = c_->token_filter(
                c_->token_filter_user_data, c_->batch_index, t, s.parent,
                prefix.empty() ? nullptr : prefix.data(), static_cast<int>(prefix.size()), v);
            if (allowed == 0) continue;
            if (!(lp < kInf) || lp == kNegInf) continue;
            const float value = (m & kPenalised) ? lp - log_penalty_ : lp;
            const float raw = s.parent_raw + value;
            insert_topk(top_.data(), P_, Candidate{raw * s.inv_penalty, raw, s.parent, v, s.new_length, 1});
        }
        return invalid;
    }

    void record_pool(int t) {
        const size_t base = static_cast<size_t>(t - origin_) * static_cast<size_t>(P_);
        for (int p = 0; p < P_; ++p) {
            const Candidate& c = top_[static_cast<size_t>(p)];
            const size_t idx = base + static_cast<size_t>(p);
            if (out_.pool_parents) out_.pool_parents[idx] = c.parent;
            if (out_.pool_tokens) out_.pool_tokens[idx] = c.token;
            if (out_.pool_lengths) out_.pool_lengths[idx] = c.length;
            if (out_.pool_raw_scores) out_.pool_raw_scores[idx] = c.raw_score;
            if (out_.pool_scores) out_.pool_scores[idx] = c.score;
            if (out_.pool_from_logprob) out_.pool_from_logprob[idx] = c.from_logprob;
            pool_scores_[static_cast<size_t>(p)] = c.score;
        }
        if (out_.relaxed_weights) {
            soft_topk_inclusion(
                pool_scores_.data(), out_.relaxed_weights + base, P_, K_,
                opt_.soft_topk_temperature, opt_.soft_topk_tolerance, opt_.soft_topk_max_iters);
        }
    }

    void select(int t) {
        const size_t base = static_cast<size_t>(t - origin_) * static_cast<size_t>(K_);
        for (int k = 0; k < K_; ++k) {
            const Candidate& c = top_[static_cast<size_t>(k)];
            const size_t i = static_cast<size_t>(k);
            const size_t idx = base + i;
            if (out_.parents) out_.parents[idx] = c.parent;
            if (out_.tokens) out_.tokens[idx] = c.token;
            if (out_.lengths) out_.lengths[idx] = c.length;
            if (out_.raw_scores) out_.raw_scores[idx] = c.raw_score;
            if (out_.scores) out_.scores[idx] = c.score;
            if (out_.from_logprob) out_.from_logprob[idx] = c.from_logprob;

            selected_scores_[i] = c.score;
            next_raw_[i] = c.raw_score;
            next_len_[i] = c.length;
            parents_[i] = c.parent;
            tokens_[i] = c.token;
            if (c.parent >= 0) {
                const size_t parent = static_cast<size_t>(c.parent);
                next_ended_[i] = static_cast<uint8_t>(ended_[parent] != 0 || (eos_ >= 0 && c.token == eos_));
                if (constrained_) {
                    next_prefix_[i] = prefix_[parent];
                    if (c.from_logprob && c.token >= 0) next_prefix_[i].push_back(c.token);
                }
            } else {
                next_ended_[i] = 0;
                if (constrained_) next_prefix_[i].clear();
            }
        }
        if (out_.weights) softmax_selected(selected_scores_.data(), out_.weights + base, K_, opt_.selected_temperature);

        raw_.swap(next_raw_);
        len_.swap(next_len_);
        ended_.swap(next_ended_);
        if (constrained_) prefix_.swap(next_prefix_);
    }

    const BeamOptions& opt_;
    const DecodeConstraints* c_;
    TraceOutputs out_;
    int K_;
    int V_;
    int P_;  // candidates kept per step: K, or the relaxed pool size
    bool with_pool_;
    float alpha_;
    int eos_;
    int min_length_;
    bool constrained_ = false;  // n-gram blocking, repetition penalty or a token filter
    bool penalise_ = false;
    float log_penalty_ = 0.0f;
    int t_ = 0;
    int origin_ = 0;  // the step whose trace is written at index 0

    std::vector<float> raw_, next_raw_;
    std::vector<int32_t> len_, next_len_;
    std::vector<uint8_t> ended_, next_ended_;
    std::vector<int32_t> parents_, tokens_;
    std::vector<float> selected_scores_;
    std::vector<float> pool_scores_;
    std::vector<Candidate> top_;

    std::vector<uint8_t> mask_;
    std::vector<int32_t> touched_;
    std::vector<int32_t> penalised_;
    std::vector<std::vector<int32_t>> prefix_, next_prefix_;
};

void check_decode_args(const BeamOptions& opt, int steps, int vocab_size, const DecodeConstraints* constraints) {
    if (steps <= 0) throw std::invalid_argument("steps must be positive");
    if (vocab_size <= 0) throw std::invalid_argument("vocab_size must be positive");
    if (opt.eos_token >= vocab_size) throw std::invalid_argument("eos_token is outside the vocabulary");
    if (constraints) {
        if (constraints->forced_tokens) {
            for (int t = 0; t < steps; ++t) {
                if (constraints->forced_tokens[t] >= vocab_size) {
                    throw std::invalid_argument("forced token is outside the vocabulary");
                }
            }
        }
        if (!(constraints->repetition_penalty > 0.0f) || !std::isfinite(constraints->repetition_penalty)) {
            throw std::invalid_argument("repetition_penalty must be finite and positive (1 disables it)");
        }
    }
    const size_t selected = checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(opt.beam_size), "result size overflow");
    (void)checked_mul_size(selected, static_cast<size_t>(vocab_size), "log-prob tensor size overflow");
}

// Allocates a result and points a trace at it.
TraceOutputs allocate_result(DecodeResult& r, const BeamOptions& opt, int steps, int vocab_size) {
    const int K = opt.beam_size;
    int P = 0;
    if (opt.relaxed_pool_multiplier > 0) {
        const size_t pool = checked_mul_size(static_cast<size_t>(K), static_cast<size_t>(opt.relaxed_pool_multiplier), "relaxed pool size overflow");
        if (pool > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::overflow_error("relaxed pool size exceeds int range");
        P = static_cast<int>(pool);
    }
    const size_t n = static_cast<size_t>(steps) * static_cast<size_t>(K);
    const size_t np = checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(P), "pool result size overflow");

    r.steps = steps;
    r.beam_size = K;
    r.vocab_size = vocab_size;
    r.eos_token = opt.eos_token;
    r.relaxed_pool_size = P;
    r.selected_temperature = opt.selected_temperature;
    r.soft_topk_temperature = opt.soft_topk_temperature;
    r.length_penalty_alpha = opt.length_penalty_alpha;

    r.parents.assign(n, -1);
    r.tokens.assign(n, -1);
    r.lengths.assign(n, 0);
    r.raw_scores.assign(n, kNegInf);
    r.scores.assign(n, kNegInf);
    r.weights.assign(n, 0.0f);
    r.from_logprob.assign(n, 0);
    r.final_raw_scores.assign(static_cast<size_t>(K), kNegInf);
    r.final_scores.assign(static_cast<size_t>(K), kNegInf);

    TraceOutputs out;
    out.parents = r.parents.data();
    out.tokens = r.tokens.data();
    out.lengths = r.lengths.data();
    out.scores = r.scores.data();
    out.raw_scores = r.raw_scores.data();
    out.from_logprob = r.from_logprob.data();
    out.weights = r.weights.data();
    if (P > 0) {
        r.pool_parents.assign(np, -1);
        r.pool_tokens.assign(np, -1);
        r.pool_lengths.assign(np, 0);
        r.pool_raw_scores.assign(np, kNegInf);
        r.pool_scores.assign(np, kNegInf);
        r.relaxed_weights.assign(np, 0.0f);
        r.pool_from_logprob.assign(np, 0);
        out.pool_parents = r.pool_parents.data();
        out.pool_tokens = r.pool_tokens.data();
        out.pool_lengths = r.pool_lengths.data();
        out.pool_scores = r.pool_scores.data();
        out.pool_raw_scores = r.pool_raw_scores.data();
        out.relaxed_weights = r.relaxed_weights.data();
        out.pool_from_logprob = r.pool_from_logprob.data();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

struct BackwardInputs {
    int T = 0;
    int K = 0;
    int V = 0;
    int P = 0;
    float selected_temperature = 1.0f;
    float soft_topk_temperature = 1.0f;
    float alpha = 0.0f;
    const int32_t* parents = nullptr;
    const int32_t* tokens = nullptr;
    const int32_t* lengths = nullptr;
    const uint8_t* from_logprob = nullptr;
    const float* weights = nullptr;
    const int32_t* pool_parents = nullptr;
    const int32_t* pool_tokens = nullptr;
    const int32_t* pool_lengths = nullptr;
    const float* relaxed_weights = nullptr;
    const uint8_t* pool_from_logprob = nullptr;
};

BackwardInputs inputs_from(const DecodeResult& fwd) {
    BackwardInputs in;
    in.T = fwd.steps;
    in.K = fwd.beam_size;
    in.V = fwd.vocab_size;
    in.P = fwd.relaxed_pool_size;
    in.selected_temperature = fwd.selected_temperature;
    in.soft_topk_temperature = fwd.soft_topk_temperature;
    in.alpha = fwd.length_penalty_alpha;
    in.parents = fwd.parents.data();
    in.tokens = fwd.tokens.data();
    in.lengths = fwd.lengths.data();
    in.from_logprob = fwd.from_logprob.data();
    in.weights = fwd.weights.empty() ? nullptr : fwd.weights.data();
    if (in.P > 0) {
        in.pool_parents = fwd.pool_parents.data();
        in.pool_tokens = fwd.pool_tokens.data();
        in.pool_lengths = fwd.pool_lengths.data();
        in.relaxed_weights = fwd.relaxed_weights.data();
        in.pool_from_logprob = fwd.pool_from_logprob.data();
    }
    return in;
}

// Where a backward pass puts the gradient of a log-prob entry: `index` is its
// flattened [T, K, V] position, `slot` the selected beam (t * K + k) it came
// from, or -1 for a relaxed-pool candidate.
struct DenseSink {
    float* grad;
    void add(int64_t index, int64_t, float value) const { grad[index] += value; }
};

struct SparseSink {
    std::vector<SparseGradEntry>* entries;
    void add(int64_t index, int64_t, float value) const { entries->push_back(SparseGradEntry{index, value}); }
};

// One value per selected beam: distinct beams of a step have distinct entries,
// so this is the dense gradient restricted to the entries the beams used.
struct SlotSink {
    float* draws;
    void add(int64_t, int64_t slot, float value) const {
        if (slot >= 0) draws[slot] = value;
    }
};

// A candidate's raw-score gradient flows to its parent beam and, unless it was
// carried forward, to the log-prob entry that produced it.
template <class Sink>
void scatter(const BackwardInputs& in, const Sink& sink, float* prev_raw_grad, int t, int parent, int token,
             uint8_t from_logprob, float draw, int64_t slot) {
    if (parent < 0 || draw == 0.0f) return;
    prev_raw_grad[parent] += draw;
    if (from_logprob && token >= 0) {
        sink.add((static_cast<int64_t>(t) * in.K + parent) * static_cast<int64_t>(in.V) + token, slot, draw);
    }
}

template <class Sink>
void run_backward(
    const BackwardInputs& in,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    const Sink& sink,
    AlignedFloatVector& grad_initial_scores) {
    const int T = in.T;
    const int K = in.K;
    const int P = in.P;
    if (grad_selected_weights && !in.weights) {
        throw std::invalid_argument("grad_selected_weights needs the selected-beam weights of the forward result");
    }
    if (grad_relaxed_weights && P <= 0) {
        throw std::invalid_argument(
            "grad_relaxed_weights needs a relaxed pool: create the decoder with relaxed_pool_multiplier >= 1");
    }

    AlignedFloatVector next_raw_grad(static_cast<size_t>(K), 0.0f);
    AlignedFloatVector prev_raw_grad(static_cast<size_t>(K), 0.0f);
    constexpr float kEps = 1.0e-12f;

    for (int t = T - 1; t >= 0; --t) {
        std::fill(prev_raw_grad.begin(), prev_raw_grad.end(), 0.0f);
        const size_t base = static_cast<size_t>(t) * static_cast<size_t>(K);

        // Selected beams: final-score and softmax-weight gradients.
        const float* w = in.weights ? in.weights + base : nullptr;
        const float* gw = grad_selected_weights ? grad_selected_weights + base : nullptr;
        const float weighted_dot = gw ? dot(w, gw, K) : 0.0f;
        for (int k = 0; k < K; ++k) {
            const size_t idx = base + static_cast<size_t>(k);
            const int parent = in.parents[idx];
            if (parent < 0) continue;
            float drank = 0.0f;
            if (gw) drank += (w[k] * (gw[k] - weighted_dot)) / in.selected_temperature;
            if (grad_final_scores && t == T - 1) drank += grad_final_scores[k];
            const int len = std::max(1, static_cast<int>(in.lengths[idx]));
            const float inv_penalty = 1.0f / gnmt_length_penalty(len, in.alpha);
            const float draw = next_raw_grad[static_cast<size_t>(k)] + drank * inv_penalty;
            scatter(in, sink, prev_raw_grad.data(), t, parent, in.tokens[idx], in.from_logprob[idx], draw,
                    static_cast<int64_t>(idx));
        }

        // Relaxed pool: implicit differentiation through the bisection threshold.
        if (grad_relaxed_weights) {
            const size_t pbase = static_cast<size_t>(t) * static_cast<size_t>(P);
            const float* r = in.relaxed_weights + pbase;
            const float* gr = grad_relaxed_weights + pbase;
            float denom = 0.0f;
            float numer = 0.0f;
            for (int p = 0; p < P; ++p) {
                const float a = r[p] * (1.0f - r[p]);
                denom += a;
                numer += gr[p] * a;
            }
            if (denom > kEps) {
                const float center = numer / denom;
                const float inv_temp = 1.0f / in.soft_topk_temperature;
                for (int p = 0; p < P; ++p) {
                    const size_t idx = pbase + static_cast<size_t>(p);
                    const int parent = in.pool_parents[idx];
                    if (parent < 0) continue;
                    const float a = r[p] * (1.0f - r[p]);
                    const float drank = a * inv_temp * (gr[p] - center);
                    if (drank == 0.0f) continue;
                    const int len = std::max(1, static_cast<int>(in.pool_lengths[idx]));
                    const float inv_penalty = 1.0f / gnmt_length_penalty(len, in.alpha);
                    scatter(in, sink, prev_raw_grad.data(), t, parent, in.pool_tokens[idx], in.pool_from_logprob[idx], -1,
                            drank * inv_penalty);
                }
            }
        }

        next_raw_grad.swap(prev_raw_grad);
    }
    grad_initial_scores = next_raw_grad;
}

void finalize_sparse_entries(std::vector<SparseGradEntry>& entries, BackwardResult& out) {
    if (entries.empty()) return;
    // Stable, so entries for one index are summed in the order they were
    // produced (the dense order) with every standard library.
    std::stable_sort(entries.begin(), entries.end(),
                     [](const SparseGradEntry& a, const SparseGradEntry& b) { return a.index < b.index; });
    out.sparse_logprob_indices.reserve(entries.size());
    out.sparse_logprob_values.reserve(entries.size());
    int64_t current = entries[0].index;
    float sum = 0.0f;
    for (const SparseGradEntry& e : entries) {
        if (e.index == current) {
            sum += e.value;
        } else {
            out.sparse_logprob_indices.push_back(current);
            out.sparse_logprob_values.push_back(sum);
            current = e.index;
            sum = e.value;
        }
    }
    out.sparse_logprob_indices.push_back(current);
    out.sparse_logprob_values.push_back(sum);
}

} // namespace

BeamSearchDecoder::BeamSearchDecoder(BeamOptions options)
    : opt_(options) {
    if (opt_.beam_size <= 0) throw std::invalid_argument("beam_size must be positive");
    if (opt_.eos_token < -1) throw std::invalid_argument("eos_token must be -1 (disabled) or a token id");
    if (!(opt_.selected_temperature > 0.0f) || !std::isfinite(opt_.selected_temperature)) {
        throw std::invalid_argument("selected_temperature must be finite and positive");
    }
    if (!(opt_.soft_topk_temperature > 0.0f) || !std::isfinite(opt_.soft_topk_temperature)) {
        throw std::invalid_argument("soft_topk_temperature must be finite and positive");
    }
    if (opt_.soft_topk_max_iters <= 0) throw std::invalid_argument("soft_topk_max_iters must be positive");
    if (!(opt_.soft_topk_tolerance > 0.0f)) throw std::invalid_argument("soft_topk_tolerance must be positive");
    if (!(opt_.length_penalty_alpha >= 0.0f) || !std::isfinite(opt_.length_penalty_alpha)) {
        throw std::invalid_argument("length_penalty_alpha must be finite and non-negative");
    }
    if (opt_.min_length < 0) throw std::invalid_argument("min_length cannot be negative");
    if (opt_.relaxed_pool_multiplier < 0) throw std::invalid_argument("relaxed_pool_multiplier cannot be negative");
    if (opt_.max_dense_gradient_elements <= 0) throw std::invalid_argument("max_dense_gradient_elements must be positive");
}

DecodeResult BeamSearchDecoder::decode(const float* log_probs, int steps, int vocab_size) const {
    return decode_constrained(log_probs, steps, vocab_size, nullptr);
}

DecodeResult BeamSearchDecoder::decode_constrained(
    const float* log_probs,
    int steps,
    int vocab_size,
    const DecodeConstraints* constraints,
    int64_t step_stride) const {
    if (!log_probs) throw std::invalid_argument("log_probs cannot be null");
    check_decode_args(opt_, steps, vocab_size, constraints);
    const int64_t row_block = static_cast<int64_t>(opt_.beam_size) * vocab_size;
    if (step_stride == 0) step_stride = row_block;
    if (step_stride < row_block) throw std::invalid_argument("step stride is smaller than beam_size * vocab_size");

    DecodeResult result;
    const TraceOutputs out = allocate_result(result, opt_, steps, vocab_size);
    BeamSearch search(opt_, vocab_size, constraints, out, /*with_pool=*/true);
    for (int t = 0; t < steps; ++t) search.step(log_probs + static_cast<size_t>(t) * static_cast<size_t>(step_stride));
    search.finish(result.final_scores.data(), result.final_raw_scores.data(), nullptr);
    return result;
}

void BeamSearchDecoder::decode_into(
    const float* log_probs,
    int steps,
    int vocab_size,
    const DecodeConstraints* constraints,
    const TraceOutputs& trace,
    float* final_scores,
    float* final_raw_scores,
    int32_t* final_lengths) const {
    if (!log_probs) throw std::invalid_argument("log_probs cannot be null");
    check_decode_args(opt_, steps, vocab_size, constraints);
    // The relaxed pool never changes which beams are selected; it is not kept here.
    BeamSearch search(opt_, vocab_size, constraints, trace, /*with_pool=*/false);
    const size_t row_block = static_cast<size_t>(opt_.beam_size) * static_cast<size_t>(vocab_size);
    for (int t = 0; t < steps; ++t) search.step(log_probs + static_cast<size_t>(t) * row_block);
    search.finish(final_scores, final_raw_scores, final_lengths);
}

void BeamSearchDecoder::step(
    const float* rows,
    int vocab_size,
    int step,
    const DecodeConstraints* constraints,
    float* raw_scores,
    int32_t* lengths,
    uint8_t* finished,
    const int32_t* prefixes,
    int prefix_stride,
    const TraceOutputs& out,
    float* final_scores) const {
    if (!rows) throw std::invalid_argument("log_probs cannot be null");
    if (!raw_scores || !lengths || !finished) throw std::invalid_argument("the beam state cannot be null");
    if (step < 0) throw std::invalid_argument("step cannot be negative");
    if (prefix_stride < 0) throw std::invalid_argument("prefix_stride cannot be negative");
    if (constraints && constraints->forced_tokens) {
        throw std::invalid_argument("forced tokens are not supported by the step interface");
    }
    check_decode_args(opt_, 1, vocab_size, constraints);
    const bool needs_prefixes =
        constraints && (constraints->repetition_penalty > 1.0f || constraints->no_repeat_ngram_size > 0 ||
                        constraints->token_filter != nullptr);
    if (needs_prefixes && prefix_stride > 0 && !prefixes) {
        throw std::invalid_argument("prefixes cannot be null with n-gram blocking, a repetition penalty or a token filter");
    }
    if (out.pool_parents || out.pool_tokens || out.pool_lengths || out.pool_scores || out.pool_raw_scores ||
        out.relaxed_weights || out.pool_from_logprob) {
        throw std::invalid_argument("the step interface has no relaxed pool");
    }
    BeamSearch search(opt_, vocab_size, constraints, out, /*with_pool=*/false);
    search.restore(step, raw_scores, lengths, finished, prefixes, prefix_stride);
    search.step(rows);
    search.save(raw_scores, lengths, finished);
    if (final_scores) search.finish(final_scores, nullptr, nullptr);
}

DecodeResult BeamSearchDecoder::decode_model_steps(
    int steps,
    int vocab_size,
    const ModelStepFunction& step_fn,
    const DecodeConstraints* constraints,
    std::vector<float>* rows_buffer) const {
    check_decode_args(opt_, steps, vocab_size, constraints);
    const int K = opt_.beam_size;
    const size_t row_block = static_cast<size_t>(K) * static_cast<size_t>(vocab_size);

    DecodeResult result;
    const TraceOutputs out = allocate_result(result, opt_, steps, vocab_size);
    BeamSearch search(opt_, vocab_size, constraints, out, /*with_pool=*/true);

    std::vector<float> local_rows;
    std::vector<float>& rows = rows_buffer ? *rows_buffer : local_rows;
    std::vector<float> scores(static_cast<size_t>(K));
    std::vector<int32_t> prefixes(checked_mul_size(static_cast<size_t>(K), static_cast<size_t>(steps), "prefix size overflow"));

    for (int t = 0; t < steps; ++t) {
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(k);
            const int len = std::max(1, static_cast<int>(search.lengths()[i]));
            scores[i] = search.raw_scores()[i] / gnmt_length_penalty(len, opt_.length_penalty_alpha);
            // Row k of the [K, t] prefix matrix: beam k's path, walked back from step t - 1.
            int beam = k;
            for (int s = t - 1; s >= 0; --s) {
                int32_t token = -1;
                if (beam >= 0) {
                    const size_t idx = static_cast<size_t>(s) * static_cast<size_t>(K) + static_cast<size_t>(beam);
                    token = result.tokens[idx];
                    beam = result.parents[idx];
                }
                prefixes[i * static_cast<size_t>(t) + static_cast<size_t>(s)] = token;
            }
        }
        ModelStepInfo info;
        info.step = t;
        info.beam_size = K;
        info.vocab_size = vocab_size;
        info.parents = search.parents();
        info.tokens = search.tokens();
        info.lengths = search.lengths();
        info.scores = scores.data();
        info.raw_scores = search.raw_scores();
        info.finished = search.finished();
        info.prefixes = prefixes.data();

        rows.assign(row_block, kNegInf);
        step_fn(info, rows.data());
        search.step(rows.data());
    }
    search.finish(result.final_scores.data(), result.final_raw_scores.data(), nullptr);
    return result;
}

BackwardResult BeamSearchDecoder::backward(
    const DecodeResult& fwd,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores) const {
    const size_t selected = checked_mul_size(static_cast<size_t>(fwd.steps), static_cast<size_t>(fwd.beam_size), "selected gradient size overflow");
    const size_t dense_count = checked_mul_size(selected, static_cast<size_t>(fwd.vocab_size), "dense gradient size overflow");
    if (dense_count > static_cast<size_t>(opt_.max_dense_gradient_elements)) {
        throw std::length_error("dense gradient allocation exceeds max_dense_gradient_elements; use sparse backward");
    }

    BackwardResult out;
    out.sparse = false;
    out.grad_log_probs.assign(dense_count, 0.0f);
    run_backward(inputs_from(fwd), grad_selected_weights, grad_relaxed_weights, grad_final_scores,
                 DenseSink{out.grad_log_probs.data()}, out.grad_initial_scores);
    return out;
}

BackwardResult BeamSearchDecoder::backward_sparse(
    const DecodeResult& fwd,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores) const {
    BackwardResult out;
    out.sparse = true;
    std::vector<SparseGradEntry> entries;
    entries.reserve(static_cast<size_t>(fwd.steps) * static_cast<size_t>(fwd.beam_size + fwd.relaxed_pool_size));
    run_backward(inputs_from(fwd), grad_selected_weights, grad_relaxed_weights, grad_final_scores,
                 SparseSink{&entries}, out.grad_initial_scores);
    finalize_sparse_entries(entries, out);
    return out;
}

namespace {

// Backward inputs of a trace, after checking that its indexes stay in range.
BackwardInputs trusted_trace(const TraceView& trace, const float* grad_final_scores, const void* out) {
    const int T = trace.steps;
    const int K = trace.beam_size;
    const int V = trace.vocab_size;
    if (T <= 0 || K <= 0 || V <= 0) throw std::invalid_argument("trace dimensions must be positive");
    if (!trace.parents || !trace.tokens || !trace.lengths || !trace.from_logprob || !grad_final_scores || !out) {
        throw std::invalid_argument("trace, grad_final_scores and the output cannot be null");
    }
    const size_t n = static_cast<size_t>(T) * static_cast<size_t>(K);
    for (size_t i = 0; i < n; ++i) {
        // The trace indexes the gradient buffer, so it is only trusted in range.
        if (trace.parents[i] < -1 || trace.parents[i] >= K || trace.tokens[i] < -1 || trace.tokens[i] >= V ||
            trace.lengths[i] < 0) {
            throw std::invalid_argument("decode trace is out of range (parents, tokens or lengths)");
        }
    }
    BackwardInputs in;
    in.T = T;
    in.K = K;
    in.V = V;
    in.alpha = trace.length_penalty_alpha;
    in.parents = trace.parents;
    in.tokens = trace.tokens;
    in.lengths = trace.lengths;
    in.from_logprob = trace.from_logprob;
    return in;
}

} // namespace

void final_scores_backward_into(const TraceView& trace, const float* grad_final_scores, float* grad_log_probs) {
    const BackwardInputs in = trusted_trace(trace, grad_final_scores, grad_log_probs);
    AlignedFloatVector grad_initial;
    run_backward(in, nullptr, nullptr, grad_final_scores, DenseSink{grad_log_probs}, grad_initial);
}

void final_scores_path_gradient(const TraceView& trace, const float* grad_final_scores, float* draws) {
    const BackwardInputs in = trusted_trace(trace, grad_final_scores, draws);
    std::fill(draws, draws + static_cast<size_t>(in.T) * static_cast<size_t>(in.K), 0.0f);
    AlignedFloatVector grad_initial;
    run_backward(in, nullptr, nullptr, grad_final_scores, SlotSink{draws}, grad_initial);
}

} // namespace dbs
