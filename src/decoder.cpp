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
        bool with_pool,
        bool from_logits = false)
        : opt_(opt),
          c_(constraints),
          out_(out),
          K_(opt.beam_size),
          V_(vocab_size),
          P_(with_pool && opt.relaxed_pool_multiplier > 0 ? opt.beam_size * opt.relaxed_pool_multiplier : opt.beam_size),
          with_pool_(with_pool && opt.relaxed_pool_multiplier > 0),
          from_logits_(from_logits),
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
        if (eos_ >= 0) {
            // Every end-of-sequence token, distinct and in increasing order.
            eos_set_.push_back(eos_);
            eos_set_.insert(eos_set_.end(), opt.extra_eos_tokens.begin(), opt.extra_eos_tokens.end());
            std::sort(eos_set_.begin(), eos_set_.end());
            eos_set_.erase(std::unique(eos_set_.begin(), eos_set_.end()), eos_set_.end());
        }
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
            // A finished beam's last token (its EOS), from its prefix when given.
            tokens_[i] = prefixes && lengths[k] >= 1 && lengths[k] <= prefix_stride
                             ? prefixes[i * static_cast<size_t>(prefix_stride) + static_cast<size_t>(lengths[k] - 1)]
                             : -1;
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

    // rows: this step's [K, V] rows of `type` (row b extends beam b):
    // log-probabilities, or logits when decoding from logits. A 16-bit row is
    // converted only when its beam is expanded.
    void step(const void* rows, DType type = DType::F32) {
        const int t = t_;
        std::fill(top_.begin(), top_.end(), Candidate{kNegInf, kNegInf, -1, -1, 0, 0});
        const int forced = c_ && c_->forced_tokens ? c_->forced_tokens[t] : -1;

        for (int b = 0; b < K_; ++b) {
            const float parent_raw = raw_[static_cast<size_t>(b)];
            if (!std::isfinite(parent_raw)) continue;

            if (eos_ >= 0 && ended_[static_cast<size_t>(b)]) {
                // A finished beam competes with its unchanged score, and carries
                // the EOS token it ended with.
                const int len = std::max(1, static_cast<int>(len_[static_cast<size_t>(b)]));
                const float rank = parent_raw / gnmt_length_penalty(len, alpha_);
                const int32_t last = tokens_[static_cast<size_t>(b)];
                insert_topk(top_.data(), P_, Candidate{rank, parent_raw, b, is_eos(last) ? last : eos_, len, 0});
                continue;
            }

            RowScan s;
            s.row = row_as_float(rows, type, b);
            if (from_logits_) {
                // lp = x - logsumexp(row), computed on the fly by the scan.
                const LogitStats stats = logit_stats(s.row, V_);
                if (stats.invalid && opt_.validate_inputs) {
                    throw std::invalid_argument("logits contain NaN or +inf" + location(t, b));
                }
                if (out_.row_lse) {
                    out_.row_lse[static_cast<size_t>(t - origin_) * static_cast<size_t>(K_) + static_cast<size_t>(b)] = stats.lse;
                }
                if (stats.lse == kNegInf) continue;  // no finite logit: nothing to select
                s.has_offset = true;
                s.offset = stats.lse;
            }
            s.banned = c_ ? c_->banned_tokens : nullptr;
            s.parent_raw = parent_raw;
            s.new_length = len_[static_cast<size_t>(b)] + 1;
            s.inv_penalty = 1.0f / gnmt_length_penalty(s.new_length, alpha_);
            s.parent = b;
            s.vocab_size = V_;
            s.forced_token = forced;
            if (s.new_length < min_length_) {
                s.masked_tokens = eos_set_.data();
                s.masked_count = static_cast<int>(eos_set_.size());
            }

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
    bool is_eos(int32_t token) const noexcept {
        return std::binary_search(eos_set_.begin(), eos_set_.end(), token);
    }

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
                if (is_masked(s, v)) continue;
                const float lp = row_log_prob(s, v);
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
            const float lp = row_log_prob(s, v);
            if (!(lp < kInf)) invalid = true;
            if (s.forced_token >= 0 && v != s.forced_token) continue;
            const uint8_t m = mask_[static_cast<size_t>(v)];
            if (m & kBanned) continue;
            if (is_masked(s, v)) continue;
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

    // Row b of a step's rows as floats: a pointer into the input for float32,
    // otherwise converted into row_buffer_.
    const float* row_as_float(const void* rows, DType type, int b) {
        const size_t offset = static_cast<size_t>(b) * static_cast<size_t>(V_);
        if (type == DType::F32) return static_cast<const float*>(rows) + offset;
        row_buffer_.resize(static_cast<size_t>(V_));
        convert_to_float(static_cast<const uint16_t*>(rows) + offset, type, static_cast<size_t>(V_), row_buffer_.data());
        return row_buffer_.data();
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
                next_ended_[i] = static_cast<uint8_t>(ended_[parent] != 0 || is_eos(c.token));
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
    bool from_logits_;
    float alpha_;
    int eos_;
    std::vector<int32_t> eos_set_;  // every EOS token, sorted; empty when eos_ < 0
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
    std::vector<float> row_buffer_;  // one converted 16-bit row

    std::vector<uint8_t> mask_;
    std::vector<int32_t> touched_;
    std::vector<int32_t> penalised_;
    std::vector<std::vector<int32_t>> prefix_, next_prefix_;
};

void check_decode_args(const BeamOptions& opt, int steps, int vocab_size, const DecodeConstraints* constraints) {
    if (steps <= 0) throw std::invalid_argument("steps must be positive");
    if (vocab_size <= 0) throw std::invalid_argument("vocab_size must be positive");
    if (opt.eos_token >= vocab_size) throw std::invalid_argument("eos_token is outside the vocabulary");
    for (int32_t token : opt.extra_eos_tokens) {
        if (token >= vocab_size) throw std::invalid_argument("an extra EOS token is outside the vocabulary");
    }
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

// The selected beam (t * K + k) a gradient came from, or none for a
// relaxed-pool candidate. A type of its own, so that it cannot be swapped
// with the float gradient next to it.
struct Slot {
    int64_t value;
};
constexpr Slot kPoolCandidate{-1};

// A candidate's raw-score gradient flows to its parent beam and, unless it was
// carried forward, to the log-prob entry that produced it.
template <class Sink>
void scatter(const BackwardInputs& in, const Sink& sink, float* prev_raw_grad, int t, int parent, int token,
             uint8_t from_logprob, float draw, Slot slot) {
    if (parent < 0 || draw == 0.0f) return;
    prev_raw_grad[parent] += draw;
    if (from_logprob && token >= 0) {
        sink.add((static_cast<int64_t>(t) * in.K + parent) * static_cast<int64_t>(in.V) + token, slot.value, draw);
    }
}

// The surrogate backward of one decode: every output is differentiated along
// the path of tokens that produced it, holding the selection fixed. Per step,
// from the last to the first, a selected beam's raw-score gradient is
//     draw = next + drank * inv_penalty + g_raw [+ g_final_raw at the last step]
// with drank = softmax-weight term + g_scores [+ g_final at the last step], and
// then a pool candidate's is
//     d = (relaxed term + g_pool_scores) * inv_penalty + g_pool_raw.
// Terms whose gradient is not given are skipped, so any subset of gradients
// gives the same bits as before the others existed.
template <class Sink>
void run_backward(
    const BackwardInputs& in,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const OutputGradients& g,
    const Sink& sink,
    AlignedFloatVector& grad_initial_scores) {
    const int T = in.T;
    const int K = in.K;
    const int P = in.P;
    if (grad_selected_weights && !in.weights) {
        throw std::invalid_argument("grad_selected_weights needs the selected-beam weights of the forward result");
    }
    const bool pool_grads = grad_relaxed_weights || g.pool_scores || g.pool_raw_scores;
    if (pool_grads && P <= 0) {
        throw std::invalid_argument(
            "relaxed-pool gradients need a relaxed pool: create the decoder with relaxed_pool_multiplier >= 1");
    }
    if (grad_relaxed_weights && !in.relaxed_weights) {
        throw std::invalid_argument("grad_relaxed_weights needs the relaxed weights of the forward result");
    }

    AlignedFloatVector next_raw_grad(static_cast<size_t>(K), 0.0f);
    AlignedFloatVector prev_raw_grad(static_cast<size_t>(K), 0.0f);
    constexpr float kEps = 1.0e-12f;

    for (int t = T - 1; t >= 0; --t) {
        std::fill(prev_raw_grad.begin(), prev_raw_grad.end(), 0.0f);
        const size_t base = static_cast<size_t>(t) * static_cast<size_t>(K);
        const bool last = t == T - 1;

        // Selected beams.
        const float* w = in.weights ? in.weights + base : nullptr;
        const float* gw = grad_selected_weights ? grad_selected_weights + base : nullptr;
        const float weighted_dot = gw ? dot(w, gw, K) : 0.0f;
        for (int k = 0; k < K; ++k) {
            const size_t idx = base + static_cast<size_t>(k);
            const int parent = in.parents[idx];
            if (parent < 0) continue;
            float drank = 0.0f;
            if (gw) drank += (w[k] * (gw[k] - weighted_dot)) / in.selected_temperature;
            if (g.scores) drank += g.scores[idx];
            if (g.final_scores && last) drank += g.final_scores[k];
            const int len = std::max(1, static_cast<int>(in.lengths[idx]));
            const float inv_penalty = 1.0f / gnmt_length_penalty(len, in.alpha);
            float draw = next_raw_grad[static_cast<size_t>(k)] + drank * inv_penalty;
            if (g.raw_scores) draw += g.raw_scores[idx];
            if (g.final_raw_scores && last) draw += g.final_raw_scores[k];
            scatter(in, sink, prev_raw_grad.data(), t, parent, in.tokens[idx], in.from_logprob[idx], draw,
                    Slot{static_cast<int64_t>(idx)});
        }

        // Pool candidates: the relaxed weights' implicit gradient through the
        // bisection threshold, and the pool scores' own gradients.
        if (pool_grads) {
            const size_t pbase = static_cast<size_t>(t) * static_cast<size_t>(P);
            const float* r = grad_relaxed_weights ? in.relaxed_weights + pbase : nullptr;
            const float* gr = grad_relaxed_weights ? grad_relaxed_weights + pbase : nullptr;
            bool relaxed = false;
            float center = 0.0f;
            float inv_temp = 0.0f;
            if (r) {
                float denom = 0.0f;
                float numer = 0.0f;
                for (int p = 0; p < P; ++p) {
                    const float a = r[p] * (1.0f - r[p]);
                    denom += a;
                    numer += gr[p] * a;
                }
                if (denom > kEps) {
                    relaxed = true;
                    center = numer / denom;
                    inv_temp = 1.0f / in.soft_topk_temperature;
                }
            }
            for (int p = 0; p < P; ++p) {
                const size_t idx = pbase + static_cast<size_t>(p);
                const int parent = in.pool_parents[idx];
                if (parent < 0) continue;
                float drank = 0.0f;
                if (relaxed) {
                    const float a = r[p] * (1.0f - r[p]);
                    drank += a * inv_temp * (gr[p] - center);
                }
                if (g.pool_scores) drank += g.pool_scores[idx];
                const int len = std::max(1, static_cast<int>(in.pool_lengths[idx]));
                const float inv_penalty = 1.0f / gnmt_length_penalty(len, in.alpha);
                float d = drank * inv_penalty;
                if (g.pool_raw_scores) d += g.pool_raw_scores[idx];
                scatter(in, sink, prev_raw_grad.data(), t, parent, in.pool_tokens[idx], in.pool_from_logprob[idx], d,
                        kPoolCandidate);
            }
        }

        next_raw_grad.swap(prev_raw_grad);
    }
    grad_initial_scores = next_raw_grad;
}

// Sorts entries by index and sums those with the same index, in the order they
// were produced (stable sort, starting from 0 as a dense gradient does), so
// the values are exactly the dense backward's.
std::vector<SparseGradEntry> merge_sparse_entries(std::vector<SparseGradEntry>& entries) {
    std::vector<SparseGradEntry> merged;
    if (entries.empty()) return merged;
    std::stable_sort(entries.begin(), entries.end(),
                     [](const SparseGradEntry& a, const SparseGradEntry& b) { return a.index < b.index; });
    SparseGradEntry current{entries[0].index, 0.0f};
    for (const SparseGradEntry& e : entries) {
        if (e.index != current.index) {
            merged.push_back(current);
            current = SparseGradEntry{e.index, 0.0f};
        }
        current.value += e.value;
    }
    merged.push_back(current);
    return merged;
}

void finalize_sparse_entries(std::vector<SparseGradEntry>& entries, BackwardResult& out) {
    const std::vector<SparseGradEntry> merged = merge_sparse_entries(entries);
    out.sparse_logprob_indices.reserve(merged.size());
    out.sparse_logprob_values.reserve(merged.size());
    for (const SparseGradEntry& e : merged) {
        out.sparse_logprob_indices.push_back(e.index);
        out.sparse_logprob_values.push_back(e.value);
    }
}

} // namespace

BeamSearchDecoder::BeamSearchDecoder(BeamOptions options)
    : opt_(options) {
    if (opt_.beam_size <= 0) throw std::invalid_argument("beam_size must be positive");
    if (opt_.eos_token < -1) throw std::invalid_argument("eos_token must be -1 (disabled) or a token id");
    for (int32_t token : opt_.extra_eos_tokens) {
        if (token < 0) throw std::invalid_argument("extra EOS tokens must be token ids");
    }
    if (!opt_.extra_eos_tokens.empty() && opt_.eos_token < 0) {
        throw std::invalid_argument("extra EOS tokens need EOS handling: set eos_token to one of the EOS tokens");
    }
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

DecodeResult BeamSearchDecoder::decode_typed(
    const void* log_probs,
    DType type,
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
    const char* base = static_cast<const char*>(log_probs);
    const size_t step_bytes = static_cast<size_t>(step_stride) * dtype_size(type);
    for (int t = 0; t < steps; ++t) search.step(base + static_cast<size_t>(t) * step_bytes, type);
    search.finish(result.final_scores.data(), result.final_raw_scores.data(), nullptr);
    return result;
}

void BeamSearchDecoder::decode_into(
    const void* log_probs,
    DType type,
    bool from_logits,
    int steps,
    int vocab_size,
    const DecodeConstraints* constraints,
    const TraceOutputs& trace,
    float* final_scores,
    float* final_raw_scores,
    int32_t* final_lengths) const {
    if (!log_probs) throw std::invalid_argument("log_probs cannot be null");
    if (trace.weights || trace.relaxed_weights) {
        throw std::invalid_argument("decode_into does not compute selected-beam or relaxed weights");
    }
    check_decode_args(opt_, steps, vocab_size, constraints);
    // The pool never changes which beams are selected; it is kept only when
    // the caller asks for it.
    const bool pool = trace.pool_parents || trace.pool_tokens || trace.pool_lengths || trace.pool_scores ||
                      trace.pool_raw_scores || trace.pool_from_logprob;
    if (pool && opt_.relaxed_pool_multiplier <= 0) {
        throw std::invalid_argument("pool outputs need a relaxed pool: set relaxed_pool_multiplier >= 1");
    }
    BeamSearch search(opt_, vocab_size, constraints, trace, pool, from_logits);
    const char* base = static_cast<const char*>(log_probs);
    const size_t step_bytes = static_cast<size_t>(opt_.beam_size) * static_cast<size_t>(vocab_size) * dtype_size(type);
    for (int t = 0; t < steps; ++t) search.step(base + static_cast<size_t>(t) * step_bytes, type);
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
    // The [K, t] prefix matrix of step t (row k: beam k's tokens at steps
    // 0..t-1), built from step t - 1's by copying each beam's parent row and
    // appending the token it emitted; a dead slot's row is all -1. Two buffers
    // alternate, so each step costs one pass over the matrix.
    const size_t prefix_size = checked_mul_size(static_cast<size_t>(K), static_cast<size_t>(steps), "prefix size overflow");
    std::vector<int32_t> prefixes(prefix_size);
    std::vector<int32_t> next_prefixes(prefix_size);

    for (int t = 0; t < steps; ++t) {
        if (t > 0) {
            const size_t width = static_cast<size_t>(t - 1);  // the previous rows' length
            for (int k = 0; k < K; ++k) {
                const size_t idx = width * static_cast<size_t>(K) + static_cast<size_t>(k);  // step t - 1, slot k
                const int32_t parent = result.parents[idx];
                int32_t* row = next_prefixes.data() + static_cast<size_t>(k) * static_cast<size_t>(t);
                if (parent >= 0) {
                    std::copy_n(prefixes.data() + static_cast<size_t>(parent) * width, width, row);
                } else {
                    std::fill_n(row, width, -1);
                }
                row[width] = result.tokens[idx];
            }
            prefixes.swap(next_prefixes);
        }
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(k);
            const int len = std::max(1, static_cast<int>(search.lengths()[i]));
            scores[i] = search.raw_scores()[i] / gnmt_length_penalty(len, opt_.length_penalty_alpha);
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
    OutputGradients g;
    g.final_scores = grad_final_scores;
    run_backward(inputs_from(fwd), grad_selected_weights, grad_relaxed_weights, g,
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
    OutputGradients g;
    g.final_scores = grad_final_scores;
    run_backward(inputs_from(fwd), grad_selected_weights, grad_relaxed_weights, g,
                 SparseSink{&entries}, out.grad_initial_scores);
    finalize_sparse_entries(entries, out);
    return out;
}

namespace {

// Backward inputs of a trace, after checking that its indexes stay in range
// (they index the gradient buffer, so they are only trusted in range).
BackwardInputs trusted_trace(const TraceView& trace) {
    const int T = trace.steps;
    const int K = trace.beam_size;
    const int V = trace.vocab_size;
    const int P = trace.pool_size;
    if (T <= 0 || K <= 0 || V <= 0 || P < 0) throw std::invalid_argument("trace dimensions must be positive");
    if (!trace.parents || !trace.tokens || !trace.lengths || !trace.from_logprob) {
        throw std::invalid_argument("the trace cannot be null");
    }
    const size_t n = static_cast<size_t>(T) * static_cast<size_t>(K);
    for (size_t i = 0; i < n; ++i) {
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
    if (P > 0) {
        if (!trace.pool_parents || !trace.pool_tokens || !trace.pool_lengths || !trace.pool_from_logprob) {
            throw std::invalid_argument("the pool trace cannot be null when pool_size > 0");
        }
        const size_t np = static_cast<size_t>(T) * static_cast<size_t>(P);
        for (size_t i = 0; i < np; ++i) {
            if (trace.pool_parents[i] < -1 || trace.pool_parents[i] >= K || trace.pool_tokens[i] < -1 ||
                trace.pool_tokens[i] >= V || trace.pool_lengths[i] < 0) {
                throw std::invalid_argument("pool trace is out of range (parents, tokens or lengths)");
            }
        }
        in.P = P;
        in.pool_parents = trace.pool_parents;
        in.pool_tokens = trace.pool_tokens;
        in.pool_lengths = trace.pool_lengths;
        in.pool_from_logprob = trace.pool_from_logprob;
    }
    return in;
}

OutputGradients final_only(const float* grad_final_scores) {
    OutputGradients g;
    g.final_scores = grad_final_scores;
    return g;
}

} // namespace

std::vector<SparseGradEntry> path_gradient(const TraceView& trace, const OutputGradients& grads) {
    const BackwardInputs in = trusted_trace(trace);
    std::vector<SparseGradEntry> entries;
    entries.reserve(static_cast<size_t>(in.T) * static_cast<size_t>(in.K + in.P));
    AlignedFloatVector grad_initial;
    run_backward(in, nullptr, nullptr, grads, SparseSink{&entries}, grad_initial);
    return merge_sparse_entries(entries);
}

void final_scores_backward_into(const TraceView& trace, const float* grad_final_scores, float* grad_log_probs) {
    if (!grad_final_scores || !grad_log_probs) throw std::invalid_argument("grad_final_scores and the output cannot be null");
    const BackwardInputs in = trusted_trace(trace);
    AlignedFloatVector grad_initial;
    run_backward(in, nullptr, nullptr, final_only(grad_final_scores), DenseSink{grad_log_probs}, grad_initial);
}

void final_scores_path_gradient(const TraceView& trace, const float* grad_final_scores, float* draws) {
    if (!grad_final_scores || !draws) throw std::invalid_argument("grad_final_scores and the output cannot be null");
    const BackwardInputs in = trusted_trace(trace);
    std::fill(draws, draws + static_cast<size_t>(in.T) * static_cast<size_t>(in.K), 0.0f);
    AlignedFloatVector grad_initial;
    run_backward(in, nullptr, nullptr, final_only(grad_final_scores), SlotSink{draws}, grad_initial);
}

} // namespace dbs
