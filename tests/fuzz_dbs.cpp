// SPDX-License-Identifier: MIT
//
// libFuzzer harness for the C ABI. Every choice (entry point, shapes, options,
// constraints, values and corruptions) is read from the input, so each input
// replays deterministically. Besides "no crash and no sanitizer report", it
// checks that:
//   * every call returns DBS_OK or DBS_ERROR_INVALID_ARGUMENT (a runtime or
//     unknown error means an exception escaped from inside the library);
//   * the sparse backward equals the dense one, entry for entry;
//   * F16/BF16 input decodes exactly like the same values as float32;
//   * dbs_backward_batch_into rejects a trace corrupted out of range with
//     DBS_ERROR_INVALID_ARGUMENT, and accepts one corrupted within range.
#include "dbs.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// The fuzz input as a stream of choices; past its end every byte reads as 0.
class Input {
  public:
    Input(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    uint8_t byte() { return pos_ < size_ ? data_[pos_++] : 0; }
    int pick(int n) { return n <= 1 ? 0 : static_cast<int>(byte() % static_cast<unsigned>(n)); }  // [0, n)
    bool coin() { return (byte() & 1) != 0; }
    uint16_t u16() {
        const unsigned lo = byte();
        return static_cast<uint16_t>(lo | (static_cast<unsigned>(byte()) << 8));
    }
    // A log-probability in [-10, 0], or -inf (0xFFFF).
    float log_prob() {
        const uint16_t raw = u16();
        return raw == 0xFFFF ? -std::numeric_limits<float>::infinity() : -static_cast<float>(raw % 10001) / 1000.0f;
    }

  private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

void expect(bool ok) {
    if (!ok) __builtin_trap();
}

void expect_status(int rc) { expect(rc == DBS_OK || rc == DBS_ERROR_INVALID_ARGUMENT); }

struct Shape {
    int T;
    int K;
    int V;
};

Shape shape_from(Input& in) { return Shape{1 + in.pick(6), 1 + in.pick(4), 1 + in.pick(32)}; }

DBSOptionsC options_from(Input& in, const Shape& s, int pool_multiplier) {
    DBSOptionsC opt{};
    opt.beam_size = s.K;
    const int eos = in.pick(s.V + 1);
    opt.eos_token = eos == s.V ? -1 : eos;
    opt.selected_temperature = 0.1f + static_cast<float>(in.pick(20)) / 10.0f;
    opt.soft_topk_temperature = 0.1f + static_cast<float>(in.pick(20)) / 10.0f;
    opt.relaxed_pool_multiplier = pool_multiplier;
    opt.length_penalty_alpha = static_cast<float>(in.pick(20)) / 10.0f;
    opt.soft_topk_tolerance = 1.0e-4f;
    opt.soft_topk_max_iters = 16 + in.pick(33);
    opt.min_length = in.pick(4);
    opt.validate_inputs = 1;
    opt.max_dense_gradient_elements = 1000000;
    return opt;
}

std::vector<float> values_from(Input& in, size_t n) {
    std::vector<float> x(n);
    for (float& v : x) v = in.log_prob();
    return x;
}

std::vector<float> gradients_from(Input& in, size_t n) {
    std::vector<float> g(n);
    for (float& v : g) v = static_cast<float>(in.pick(201) - 100) / 50.0f;  // [-2, 2]
    return g;
}

// Constraints for a [T, V] search; the vectors own the arrays it points to.
struct Constraints {
    std::vector<uint8_t> banned;
    std::vector<int32_t> forced;
    DBSAdvancedConstraintsC c{};
};

void constraints_from(Input& in, const Shape& s, Constraints& out) {
    if (in.coin()) {
        out.banned.resize(static_cast<size_t>(s.V));
        for (uint8_t& b : out.banned) b = in.pick(4) == 0 ? 1 : 0;  // about a quarter banned
    }
    if (in.coin()) {
        out.forced.resize(static_cast<size_t>(s.T));
        for (int32_t& f : out.forced) f = in.pick(3) == 0 ? in.pick(s.V) : -1;
    }
    out.c.banned_tokens = out.banned.empty() ? nullptr : out.banned.data();
    out.c.forced_tokens = out.forced.empty() ? nullptr : out.forced.data();
    out.c.min_length = in.pick(5) - 1;  // -1 keeps the decoder's min_length
    out.c.repetition_penalty = in.coin() ? 1.3f : 1.0f;
    out.c.no_repeat_ngram_size = in.pick(5);
}

DBSDecoderHandle* create(const DBSOptionsC& opt) {
    DBSDecoderHandle* h = nullptr;
    const int rc = dbs_create_ex(opt, &h);
    expect_status(rc);
    return rc == DBS_OK ? h : nullptr;
}

// Sparse and dense backward with every gradient input; the sparse entries must
// equal the dense gradient (both sum each entry in the same order), and the
// dense gradient is zero everywhere else.
void check_backward(Input& in, DBSDecoderHandle* h, const DBSResultHandle* r) {
    const int T = dbs_result_steps(r);
    const int K = dbs_result_beam_size(r);
    const int P = dbs_result_pool_size(r);
    const std::vector<float> grad_final = gradients_from(in, static_cast<size_t>(K));
    const std::vector<float> grad_selected = gradients_from(in, static_cast<size_t>(T) * K);
    const std::vector<float> grad_relaxed = gradients_from(in, static_cast<size_t>(T) * (P > 0 ? P : 1));
    const float* gs = in.coin() ? grad_selected.data() : nullptr;
    const float* gr = P > 0 && in.coin() ? grad_relaxed.data() : nullptr;
    const float* gf = in.coin() ? grad_final.data() : nullptr;

    DBSBackwardHandle* sparse = nullptr;
    DBSBackwardHandle* dense = nullptr;
    const int rs = dbs_backward_sparse(h, r, gs, gr, gf, &sparse);
    const int rd = dbs_backward_dense(h, r, gs, gr, gf, &dense);
    expect(rs == DBS_OK && rd == DBS_OK);  // valid inputs by construction
    const int64_t n = dbs_backward_grad_log_probs_count(r);
    const float* grad = dbs_backward_grad_log_probs(dense);
    const int64_t count = dbs_backward_sparse_logprob_count(sparse);
    const int64_t* index = dbs_backward_sparse_logprob_indices(sparse);
    const float* value = dbs_backward_sparse_logprob_values(sparse);
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < count; ++i) {
        expect(index[i] >= 0 && index[i] < n && (i == 0 || index[i] > index[i - 1]));
        expect(grad[index[i]] == value[i]);
        seen[static_cast<size_t>(index[i])] = 1;
    }
    for (int64_t i = 0; i < n; ++i) expect(seen[static_cast<size_t>(i)] || grad[i] == 0.0f);
    dbs_free_backward(sparse);
    dbs_free_backward(dense);
}

// dbs_decode with a relaxed pool, then both backward passes.
void fuzz_decode(Input& in) {
    const Shape s = shape_from(in);
    DBSDecoderHandle* h = create(options_from(in, s, 1 + in.pick(4)));
    if (!h) return;
    const std::vector<float> x = values_from(in, static_cast<size_t>(s.T) * s.K * s.V);
    DBSResultHandle* r = nullptr;
    const int rc = dbs_decode(h, x.data(), s.T, s.V, &r);
    expect_status(rc);
    if (rc == DBS_OK) {
        check_backward(in, h, r);
        dbs_free_result(r);
    }
    dbs_destroy(h);
}

// dbs_decode_constrained_ex: banned and forced tokens, min_length, n-gram
// blocking and the repetition penalty.
void fuzz_constrained(Input& in) {
    const Shape s = shape_from(in);
    DBSDecoderHandle* h = create(options_from(in, s, in.pick(3)));
    if (!h) return;
    Constraints c;
    constraints_from(in, s, c);
    const std::vector<float> x = values_from(in, static_cast<size_t>(s.T) * s.K * s.V);
    DBSResultHandle* r = nullptr;
    const int rc = dbs_decode_constrained_ex(h, x.data(), s.T, s.V, in.coin() ? &c.c : nullptr, &r);
    expect_status(rc);
    if (rc == DBS_OK) {
        check_backward(in, h, r);
        dbs_free_result(r);
    }
    dbs_destroy(h);
}

float half_to_float(uint16_t h) {
    const int exponent = (h >> 10) & 0x1F;
    const int mantissa = h & 0x3FF;
    float v;
    if (exponent == 0) v = std::ldexp(static_cast<float>(mantissa), -24);
    else if (exponent == 31) v = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else v = std::ldexp(static_cast<float>(mantissa | 0x400), exponent - 25);
    return (h & 0x8000) ? -v : v;
}

float bf16_to_float(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float v;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

// dbs_decode_typed with F16 or BF16 input must decode exactly as the same
// values in float32.
void fuzz_typed(Input& in) {
    const Shape s = shape_from(in);
    DBSDecoderHandle* h = create(options_from(in, s, in.pick(3)));
    if (!h) return;
    const bool f16 = in.coin();
    const size_t n = static_cast<size_t>(s.T) * s.K * s.V;
    std::vector<uint16_t> bits(n);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        uint16_t b = in.u16();
        if (in.pick(16) != 0) {
            // Mostly finite negative values; otherwise any pattern (NaN, +-inf, positive).
            b = static_cast<uint16_t>(b | 0x8000);
            const uint16_t exponent_mask = f16 ? 0x7C00 : 0x7F80;
            if ((b & exponent_mask) == exponent_mask) b = static_cast<uint16_t>(b & ~(exponent_mask & (exponent_mask >> 1)));
        }
        bits[i] = b;
        x[i] = f16 ? half_to_float(b) : bf16_to_float(b);
    }
    DBSResultHandle* typed = nullptr;
    DBSResultHandle* plain = nullptr;
    const int rt = dbs_decode_typed(h, bits.data(), f16 ? DBS_DTYPE_F16 : DBS_DTYPE_BF16, s.T, s.V, &typed);
    const int rp = dbs_decode(h, x.data(), s.T, s.V, &plain);
    expect_status(rt);
    expect(rt == rp);
    if (rt == DBS_OK) {
        const size_t k = static_cast<size_t>(s.K);
        const size_t tk = static_cast<size_t>(s.T) * k;
        expect(std::memcmp(dbs_result_final_scores(typed), dbs_result_final_scores(plain), k * sizeof(float)) == 0);
        expect(std::memcmp(dbs_result_tokens(typed), dbs_result_tokens(plain), tk * sizeof(int32_t)) == 0);
        expect(std::memcmp(dbs_result_parents(typed), dbs_result_parents(plain), tk * sizeof(int32_t)) == 0);
        dbs_free_result(typed);
        dbs_free_result(plain);
    }
    dbs_destroy(h);
}

// dbs_decode_batch_into, then dbs_backward_batch_into on its trace, which the
// fuzzer may corrupt first.
void fuzz_batch_into(Input& in) {
    const Shape s = shape_from(in);
    const int B = 1 + in.pick(4);
    DBSDecoderHandle* h = create(options_from(in, s, 0));
    if (!h) return;
    const size_t bk = static_cast<size_t>(B) * s.K;
    const size_t btk = bk * s.T;
    const std::vector<float> x = values_from(in, btk * s.V);
    std::vector<int32_t> steps(static_cast<size_t>(B));
    for (int32_t& t : steps) t = 1 + in.pick(s.T + (in.pick(8) == 0 ? 1 : 0));  // occasionally T + 1: rejected
    const int32_t* steps_ptr = in.coin() ? steps.data() : nullptr;
    Constraints c;
    constraints_from(in, s, c);
    const DBSAdvancedConstraintsC* constraints = in.coin() ? &c.c : nullptr;
    const int threads = 1 + in.pick(2);

    std::vector<float> final_scores(bk), final_raw(bk), scores(btk), raw(btk);
    std::vector<int32_t> final_lengths(bk), tokens(btk), parents(btk), lengths(btk);
    std::vector<uint8_t> from_logprob(btk);
    DBSDecodeOutputsC out{final_scores.data(), final_raw.data(), final_lengths.data(), tokens.data(), parents.data(),
                          lengths.data(), scores.data(), raw.data(), from_logprob.data()};
    const int rc = dbs_decode_batch_into(h, x.data(), B, s.T, s.V, steps_ptr, constraints, threads, &out);
    expect_status(rc);
    if (rc != DBS_OK) {
        dbs_destroy(h);
        return;
    }

    // Corrupt a few trace entries, in range or not.
    constexpr int32_t kMin = std::numeric_limits<int32_t>::min();
    constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
    for (int n = in.pick(4); n > 0; --n) {
        const size_t i = static_cast<size_t>(in.pick(static_cast<int>(btk)));
        switch (in.pick(4)) {
            case 0: {
                const int32_t choices[] = {-2, -1, s.K, s.K + 1, kMin, kMax, in.pick(s.K)};
                parents[i] = choices[in.pick(7)];
                break;
            }
            case 1: {
                const int32_t choices[] = {-2, -1, s.V, kMin, kMax, in.pick(s.V)};
                tokens[i] = choices[in.pick(6)];
                break;
            }
            case 2: {
                const int32_t choices[] = {-1, kMin, 0, 1 + in.pick(s.T), kMax};
                lengths[i] = choices[in.pick(5)];
                break;
            }
            default:
                from_logprob[i] = in.byte();  // any byte is a valid flag
                break;
        }
    }
    // The backward reads, and must range-check, the first steps[b] steps of
    // each example: parents in [-1, K), tokens in [-1, V), lengths >= 0.
    bool out_of_range = false;
    for (int b = 0; b < B; ++b) {
        const int read = steps_ptr ? steps[static_cast<size_t>(b)] : s.T;
        for (size_t i = static_cast<size_t>(b) * s.T * s.K, end = i + static_cast<size_t>(read) * s.K; i < end; ++i) {
            out_of_range |= parents[i] < -1 || parents[i] >= s.K || tokens[i] < -1 || tokens[i] >= s.V || lengths[i] < 0;
        }
    }

    const std::vector<float> grad_final = gradients_from(in, bk);
    std::vector<float> grad(btk * s.V, 0.0f);
    const int rb = dbs_backward_batch_into(h, B, s.T, s.V, steps_ptr, parents.data(), tokens.data(), lengths.data(),
                                           from_logprob.data(), grad_final.data(), threads, grad.data());
    expect(rb == (out_of_range ? DBS_ERROR_INVALID_ARGUMENT : DBS_OK));
    dbs_destroy(h);
}

// dbs_decode_batch_variable: per-example steps, beam sizes, EOS tokens, minimum
// lengths, banned and forced tokens.
void fuzz_batch_variable(Input& in) {
    const Shape s = shape_from(in);
    const int B = 1 + in.pick(4);
    DBSDecoderHandle* h = create(options_from(in, s, in.pick(3)));
    if (!h) return;
    const std::vector<float> x = values_from(in, static_cast<size_t>(B) * s.T * s.K * s.V);
    std::vector<int32_t> steps(static_cast<size_t>(B)), beams(steps.size()), eos(steps.size()), min_lengths(steps.size());
    for (size_t b = 0; b < steps.size(); ++b) {
        steps[b] = 1 + in.pick(s.T);
        beams[b] = 1 + in.pick(s.K + (in.pick(8) == 0 ? 1 : 0));  // occasionally K + 1: rejected
        eos[b] = in.pick(s.V + 1) - 1;
        min_lengths[b] = in.pick(4);
    }
    std::vector<uint8_t> banned(static_cast<size_t>(B) * s.V);
    for (uint8_t& v : banned) v = in.pick(4) == 0 ? 1 : 0;
    std::vector<int32_t> forced(static_cast<size_t>(B) * s.T);
    for (int32_t& f : forced) f = in.pick(3) == 0 ? in.pick(s.V) : -1;

    DBSBatchResultHandle* r = nullptr;
    const int rc = dbs_decode_batch_variable(
        h, x.data(), B, s.T, s.K, s.V, in.coin() ? steps.data() : nullptr, in.coin() ? beams.data() : nullptr,
        in.coin() ? eos.data() : nullptr, in.coin() ? min_lengths.data() : nullptr, in.coin() ? banned.data() : nullptr,
        in.coin() ? forced.data() : nullptr, 1 + in.pick(2), &r);
    expect_status(rc);
    if (rc == DBS_OK) {
        expect(dbs_batch_result_size(r) == B);
        for (int b = 0; b < B; ++b) {
            const DBSResultHandle* e = dbs_batch_result_at(r, b);
            expect(e != nullptr && dbs_result_beam_size(e) >= 1 && dbs_result_beam_size(e) <= s.K);
            expect(dbs_result_validate_deterministic_order(e) == DBS_OK);
        }
        dbs_free_batch_result(r);
    }
    dbs_destroy(h);
}

// The original harness: dbs_decode and a sparse final-score backward.
void fuzz_plain(Input& in) {
    const Shape s = shape_from(in);
    DBSDecoderHandle* h = create(options_from(in, s, in.pick(4)));
    if (!h) return;
    const std::vector<float> x = values_from(in, static_cast<size_t>(s.T) * s.K * s.V);
    DBSResultHandle* r = nullptr;
    const int rc = dbs_decode(h, x.data(), s.T, s.V, &r);
    expect_status(rc);
    if (rc == DBS_OK) {
        std::vector<float> grad_final(static_cast<size_t>(s.K), 0.0f);
        grad_final[0] = 1.0f;
        DBSBackwardHandle* b = nullptr;
        expect(dbs_backward_sparse(h, r, nullptr, nullptr, grad_final.data(), &b) == DBS_OK);
        dbs_free_backward(b);
        dbs_free_result(r);
    }
    dbs_destroy(h);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    Input in(data, size);
    switch (in.pick(6)) {
        case 0: fuzz_decode(in); break;
        case 1: fuzz_constrained(in); break;
        case 2: fuzz_typed(in); break;
        case 3: fuzz_batch_into(in); break;
        case 4: fuzz_batch_variable(in); break;
        default: fuzz_plain(in); break;
    }
    return 0;
}
