// SPDX-License-Identifier: MIT
//
// beamgrad._C: defines the beamgrad operators with torch.library and registers
// their CPU kernels. The libdbs sources are compiled into this module, so the
// decoder writes straight into the output tensors and examples of a batch are
// decoded in parallel on PyTorch's thread pool. beamgrad._C_cuda registers the
// CUDA kernels of the same operators (cuda_ops.cpp).
//
//   beamgrad::decode(log_probs[B,T,K,V] f32, steps[B]?, eos_token, min_length,
//                    length_penalty_alpha, banned_tokens[V] u8?,
//                    no_repeat_ngram_size, repetition_penalty, validate)
//     -> final_scores[B,K], final_raw_scores[B,K], final_lengths[B,K] i32,
//        tokens, parents, lengths [B,T,K] i32, scores, raw_scores [B,T,K] f32,
//        from_logprob [B,T,K] u8
//   beamgrad::final_scores_backward(grad_final[B,K], parents, tokens, lengths,
//                    from_logprob, steps[B]?, vocab_size, length_penalty_alpha)
//     -> grad_log_probs[B,T,K,V] f32
//   beamgrad::final_scores_path_gradient(<same arguments as final_scores_backward>)
//     -> draws[B,T,K] f32: that gradient at the entry each selected beam used
//        (0 for carried-forward and dead slots); scattering it gives the dense
//        gradient. Lets beamgrad.beam_search hand each step its own gradient.
//   beamgrad::decode_step(log_probs[B,K,V] f32, raw_scores[B,K] f32,
//                    lengths[B,K] i32, finished[B,K] u8, prefixes[B,K,L] i32,
//                    eos_token, min_length, length_penalty_alpha,
//                    banned_tokens[V] u8?, no_repeat_ngram_size,
//                    repetition_penalty, validate)
//     -> tokens, parents [B,K] i32, lengths [B,K] i32, scores [B,K] f32,
//        raw_scores [B,K] f32, from_logprob [B,K] u8, finished [B,K] u8,
//        final_scores [B,K] f32: one search step from an explicit state (the
//        beams after the previous step and their token prefixes), returning
//        the step's beams, which are also the new state, and their final
//        scores were the search to end here. Used by beamgrad.beam_search.
//        On CUDA, every live, unfinished beam of an example must have the
//        same length, as in any state the search produced (see
//        DBSCudaBeamState in include/dbs_cuda.h); with validate and a length
//        penalty, a state that does not raises ValueError.
//
// With validate, a NaN or +inf in any row the search reads raises ValueError.
//
// The module itself is empty: importing it runs the TORCH_LIBRARY
// registrations below. It uses only CPython's limited API (no pybind11), so
// one build works with every CPython >= 3.10 (an abi3 wheel).
#include <Python.h>

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/Parallel.h>

#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "decoder.hpp"

namespace {

using Tensor = at::Tensor;
using DecodeOutputs = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using StepOutputs = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

constexpr int64_t kIntMax = std::numeric_limits<int>::max();

void check_dim(int64_t value, const char* name) {
    TORCH_CHECK_VALUE(value > 0 && value <= kIntMax, name, " must be in [1, INT_MAX], got ", value);
}

std::vector<int32_t> per_example_steps(const c10::optional<Tensor>& steps, int64_t B, int64_t T) {
    std::vector<int32_t> out(static_cast<size_t>(B), static_cast<int32_t>(T));
    if (!steps.has_value()) return out;
    TORCH_CHECK_VALUE(!steps->is_floating_point() && !steps->is_complex() && steps->scalar_type() != torch::kBool,
                      "steps must be an integer tensor, got ", steps->scalar_type());
    const Tensor s = steps->to(torch::kCPU, torch::kInt64).contiguous();
    TORCH_CHECK_VALUE(s.dim() == 1 && s.size(0) == B, "steps must have shape [B] = [", B, "]");
    const int64_t* p = s.data_ptr<int64_t>();
    for (int64_t b = 0; b < B; ++b) {
        TORCH_CHECK_VALUE(p[b] >= 1 && p[b] <= T, "steps[", b, "] = ", p[b], " must be in [1, ", T, "]");
        out[static_cast<size_t>(b)] = static_cast<int32_t>(p[b]);
    }
    return out;
}

// Keeps the failure of the lowest-numbered example of a parallel loop.
class FirstError {
public:
    void record(int64_t b, const std::string& message, bool invalid_argument) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failed_ || b < example_) {
            failed_ = true;
            example_ = b;
            message_ = message;
            invalid_argument_ = invalid_argument;
        }
    }
    void raise_if_failed() const {
        if (!failed_) return;
        TORCH_CHECK_VALUE(!invalid_argument_, "example ", example_, ": ", message_);
        TORCH_CHECK(false, "example ", example_, ": ", message_);
    }

private:
    std::mutex mutex_;
    bool failed_ = false;
    bool invalid_argument_ = false;
    int64_t example_ = 0;
    std::string message_;
};

DecodeOutputs decode_cpu(
    const Tensor& log_probs,
    const c10::optional<Tensor>& steps,
    int64_t eos_token,
    int64_t min_length,
    double length_penalty_alpha,
    const c10::optional<Tensor>& banned_tokens,
    int64_t no_repeat_ngram_size,
    double repetition_penalty,
    bool validate) {
    TORCH_CHECK(log_probs.device().is_cpu(), "beamgrad::decode (CPU) expects a CPU tensor");
    TORCH_CHECK_VALUE(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK_VALUE(log_probs.dim() == 4, "log_probs must have shape [B, T, K, V]");
    const Tensor x = log_probs.contiguous();
    const int64_t B = x.size(0), T = x.size(1), K = x.size(2), V = x.size(3);
    check_dim(B, "B");
    check_dim(T, "T");
    check_dim(K, "K");
    check_dim(V, "V");
    TORCH_CHECK_VALUE(eos_token >= -1 && eos_token < V, "eos_token must be -1 or a token id < ", V);
    TORCH_CHECK_VALUE(min_length >= 0 && min_length <= kIntMax, "min_length must be non-negative");
    TORCH_CHECK_VALUE(std::isfinite(length_penalty_alpha) && length_penalty_alpha >= 0.0,
                      "length_penalty_alpha must be finite and non-negative");
    TORCH_CHECK_VALUE(std::isfinite(repetition_penalty) && repetition_penalty > 0.0,
                      "repetition_penalty must be finite and positive");

    const std::vector<int32_t> steps_b = per_example_steps(steps, B, T);
    Tensor banned;
    if (banned_tokens.has_value()) {
        banned = banned_tokens->to(torch::kCPU, torch::kUInt8).contiguous();
        TORCH_CHECK_VALUE(banned.dim() == 1 && banned.size(0) == V, "banned_tokens must be a [V] mask");
    }

    dbs::BeamOptions opt;
    opt.beam_size = static_cast<int>(K);
    opt.eos_token = static_cast<int>(eos_token);
    opt.min_length = static_cast<int>(min_length);
    opt.length_penalty_alpha = static_cast<float>(length_penalty_alpha);
    opt.validate_inputs = validate ? 1 : 0;
    const dbs::BeamSearchDecoder decoder(opt);

    dbs::DecodeConstraints constraints;
    constraints.banned_tokens = banned.defined() ? banned.data_ptr<uint8_t>() : nullptr;
    constraints.no_repeat_ngram_size = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(no_repeat_ngram_size, kIntMax)));
    constraints.repetition_penalty = static_cast<float>(repetition_penalty);
    const bool constrained =
        banned.defined() || constraints.no_repeat_ngram_size > 0 || constraints.repetition_penalty > 1.0f;

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor final_scores = torch::full({B, K}, neg_inf, f32);
    Tensor final_raw = torch::full({B, K}, neg_inf, f32);
    Tensor final_lengths = torch::zeros({B, K}, i32);
    Tensor tokens = torch::full({B, T, K}, -1, i32);
    Tensor parents = torch::full({B, T, K}, -1, i32);
    Tensor lengths = torch::zeros({B, T, K}, i32);
    Tensor scores = torch::full({B, T, K}, neg_inf, f32);
    Tensor raw_scores = torch::full({B, T, K}, neg_inf, f32);
    Tensor from_logprob = torch::zeros({B, T, K}, torch::TensorOptions().dtype(torch::kUInt8));

    const float* xp = x.data_ptr<float>();
    FirstError error;
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            const int64_t trace = b * T * K;
            dbs::TraceOutputs out;
            out.tokens = tokens.data_ptr<int32_t>() + trace;
            out.parents = parents.data_ptr<int32_t>() + trace;
            out.lengths = lengths.data_ptr<int32_t>() + trace;
            out.scores = scores.data_ptr<float>() + trace;
            out.raw_scores = raw_scores.data_ptr<float>() + trace;
            out.from_logprob = from_logprob.data_ptr<uint8_t>() + trace;
            dbs::DecodeConstraints c = constraints;
            c.batch_index = static_cast<int>(b);
            try {
                decoder.decode_into(xp + b * T * K * V, dbs::DType::F32, /*from_logits=*/false,
                                    steps_b[static_cast<size_t>(b)], static_cast<int>(V),
                                    constrained ? &c : nullptr, out, final_scores.data_ptr<float>() + b * K,
                                    final_raw.data_ptr<float>() + b * K, final_lengths.data_ptr<int32_t>() + b * K);
            } catch (const std::invalid_argument& e) {
                error.record(b, e.what(), true);
            } catch (const std::exception& e) {
                error.record(b, e.what(), false);
            }
        }
    });
    error.raise_if_failed();
    return {final_scores, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob};
}

void check_step_inputs(const Tensor& log_probs, const Tensor& raw_scores, const Tensor& lengths, const Tensor& finished,
                       const Tensor& prefixes) {
    TORCH_CHECK_VALUE(log_probs.scalar_type() == torch::kFloat32 && log_probs.dim() == 3,
                      "log_probs must be a float32 [B, K, V] tensor");
    const auto B = log_probs.size(0), K = log_probs.size(1);
    check_dim(B, "B");
    check_dim(K, "K");
    check_dim(log_probs.size(2), "V");
    TORCH_CHECK_VALUE(raw_scores.scalar_type() == torch::kFloat32 && raw_scores.sizes() == torch::IntArrayRef({B, K}),
                      "raw_scores must be a float32 [B, K] tensor");
    TORCH_CHECK_VALUE(lengths.scalar_type() == torch::kInt32 && lengths.sizes() == torch::IntArrayRef({B, K}),
                      "lengths must be an int32 [B, K] tensor");
    TORCH_CHECK_VALUE(finished.scalar_type() == torch::kUInt8 && finished.sizes() == torch::IntArrayRef({B, K}),
                      "finished must be a uint8 [B, K] tensor");
    TORCH_CHECK_VALUE(prefixes.scalar_type() == torch::kInt32 && prefixes.dim() == 3 && prefixes.size(0) == B &&
                          prefixes.size(1) == K && prefixes.size(2) <= kIntMax,
                      "prefixes must be an int32 [B, K, L] tensor");
}

StepOutputs decode_step_cpu(
    const Tensor& log_probs,
    const Tensor& raw_scores,
    const Tensor& lengths,
    const Tensor& finished,
    const Tensor& prefixes,
    int64_t eos_token,
    int64_t min_length,
    double length_penalty_alpha,
    const c10::optional<Tensor>& banned_tokens,
    int64_t no_repeat_ngram_size,
    double repetition_penalty,
    bool validate) {
    TORCH_CHECK(log_probs.device().is_cpu(), "beamgrad::decode_step (CPU) expects CPU tensors");
    check_step_inputs(log_probs, raw_scores, lengths, finished, prefixes);
    const Tensor x = log_probs.contiguous();
    const Tensor pre = prefixes.contiguous();
    const int64_t B = x.size(0), K = x.size(1), V = x.size(2), L = pre.size(2);
    TORCH_CHECK_VALUE(eos_token >= -1 && eos_token < V, "eos_token must be -1 or a token id < ", V);
    TORCH_CHECK_VALUE(min_length >= 0 && min_length <= kIntMax, "min_length must be non-negative");
    TORCH_CHECK_VALUE(std::isfinite(length_penalty_alpha) && length_penalty_alpha >= 0.0,
                      "length_penalty_alpha must be finite and non-negative");
    TORCH_CHECK_VALUE(std::isfinite(repetition_penalty) && repetition_penalty > 0.0,
                      "repetition_penalty must be finite and positive");
    Tensor banned;
    if (banned_tokens.has_value()) {
        banned = banned_tokens->to(torch::kCPU, torch::kUInt8).contiguous();
        TORCH_CHECK_VALUE(banned.dim() == 1 && banned.size(0) == V, "banned_tokens must be a [V] mask");
    }

    dbs::BeamOptions opt;
    opt.beam_size = static_cast<int>(K);
    opt.eos_token = static_cast<int>(eos_token);
    opt.min_length = static_cast<int>(min_length);
    opt.length_penalty_alpha = static_cast<float>(length_penalty_alpha);
    opt.validate_inputs = validate ? 1 : 0;
    const dbs::BeamSearchDecoder decoder(opt);
    dbs::DecodeConstraints constraints;
    constraints.banned_tokens = banned.defined() ? banned.data_ptr<uint8_t>() : nullptr;
    constraints.no_repeat_ngram_size = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(no_repeat_ngram_size, kIntMax)));
    constraints.repetition_penalty = static_cast<float>(repetition_penalty);

    // The state is advanced in copies, so the inputs are left untouched.
    Tensor raw = raw_scores.contiguous().clone();
    Tensor len = lengths.contiguous().clone();
    Tensor fin = finished.contiguous().clone();
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32);
    Tensor tokens = torch::empty({B, K}, i32);
    Tensor parents = torch::empty({B, K}, i32);
    Tensor scores = torch::empty({B, K}, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor final_scores = torch::empty({B, K}, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor from_logprob = torch::empty({B, K}, torch::TensorOptions().dtype(torch::kUInt8));

    FirstError error;
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            dbs::TraceOutputs out;
            out.tokens = tokens.data_ptr<int32_t>() + b * K;
            out.parents = parents.data_ptr<int32_t>() + b * K;
            out.scores = scores.data_ptr<float>() + b * K;
            out.from_logprob = from_logprob.data_ptr<uint8_t>() + b * K;
            dbs::DecodeConstraints c = constraints;
            c.batch_index = static_cast<int>(b);
            try {
                decoder.step(x.data_ptr<float>() + b * K * V, static_cast<int>(V), static_cast<int>(L), &c,
                             raw.data_ptr<float>() + b * K, len.data_ptr<int32_t>() + b * K,
                             fin.data_ptr<uint8_t>() + b * K, pre.data_ptr<int32_t>() + b * K * L, static_cast<int>(L), out,
                             final_scores.data_ptr<float>() + b * K);
            } catch (const std::invalid_argument& e) {
                error.record(b, e.what(), true);
            } catch (const std::exception& e) {
                error.record(b, e.what(), false);
            }
        }
    });
    error.raise_if_failed();
    // The new state's lengths and raw scores are the selected beams'.
    return {tokens, parents, len, scores, raw, from_logprob, fin, final_scores};
}

Tensor final_scores_backward_cpu(
    const Tensor& grad_final,
    const Tensor& parents,
    const Tensor& tokens,
    const Tensor& lengths,
    const Tensor& from_logprob,
    const c10::optional<Tensor>& steps,
    int64_t vocab_size,
    double length_penalty_alpha) {
    TORCH_CHECK_VALUE(parents.dim() == 3 && parents.scalar_type() == torch::kInt32,
                      "parents must be an int32 [B, T, K] tensor");
    const int64_t B = parents.size(0), T = parents.size(1), K = parents.size(2), V = vocab_size;
    check_dim(V, "vocab_size");
    for (const Tensor* t : {&tokens, &lengths}) {
        TORCH_CHECK_VALUE(t->sizes() == parents.sizes() && t->scalar_type() == torch::kInt32,
                          "tokens and lengths must be int32 [B, T, K] tensors");
    }
    TORCH_CHECK_VALUE(from_logprob.sizes() == parents.sizes() && from_logprob.scalar_type() == torch::kUInt8,
                      "from_logprob must be a uint8 [B, T, K] tensor");
    TORCH_CHECK_VALUE(grad_final.scalar_type() == torch::kFloat32 && grad_final.dim() == 2 && grad_final.size(0) == B &&
                          grad_final.size(1) == K,
                      "grad_final must be a float32 [B, K] tensor");
    const Tensor par = parents.contiguous(), tok = tokens.contiguous(), len = lengths.contiguous();
    const Tensor flp = from_logprob.contiguous(), g = grad_final.contiguous();

    const std::vector<int32_t> steps_b = per_example_steps(steps, B, T);
    Tensor grad = torch::zeros({B, T, K, V}, torch::TensorOptions().dtype(torch::kFloat32));
    FirstError error;
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            const int64_t off = b * T * K;
            dbs::TraceView trace;
            trace.steps = steps_b[static_cast<size_t>(b)];
            trace.beam_size = static_cast<int>(K);
            trace.vocab_size = static_cast<int>(V);
            trace.length_penalty_alpha = static_cast<float>(length_penalty_alpha);
            trace.parents = par.data_ptr<int32_t>() + off;
            trace.tokens = tok.data_ptr<int32_t>() + off;
            trace.lengths = len.data_ptr<int32_t>() + off;
            trace.from_logprob = flp.data_ptr<uint8_t>() + off;
            try {
                dbs::final_scores_backward_into(trace, g.data_ptr<float>() + b * K, grad.data_ptr<float>() + off * V);
            } catch (const std::invalid_argument& e) {
                error.record(b, e.what(), true);
            }
        }
    });
    error.raise_if_failed();
    return grad;
}

Tensor final_scores_path_gradient_cpu(
    const Tensor& grad_final,
    const Tensor& parents,
    const Tensor& tokens,
    const Tensor& lengths,
    const Tensor& from_logprob,
    const c10::optional<Tensor>& steps,
    int64_t vocab_size,
    double length_penalty_alpha) {
    TORCH_CHECK_VALUE(parents.dim() == 3 && parents.scalar_type() == torch::kInt32,
                      "parents must be an int32 [B, T, K] tensor");
    const int64_t B = parents.size(0), T = parents.size(1), K = parents.size(2), V = vocab_size;
    check_dim(V, "vocab_size");
    for (const Tensor* t : {&tokens, &lengths}) {
        TORCH_CHECK_VALUE(t->sizes() == parents.sizes() && t->scalar_type() == torch::kInt32,
                          "tokens and lengths must be int32 [B, T, K] tensors");
    }
    TORCH_CHECK_VALUE(from_logprob.sizes() == parents.sizes() && from_logprob.scalar_type() == torch::kUInt8,
                      "from_logprob must be a uint8 [B, T, K] tensor");
    TORCH_CHECK_VALUE(grad_final.scalar_type() == torch::kFloat32 && grad_final.dim() == 2 && grad_final.size(0) == B &&
                          grad_final.size(1) == K,
                      "grad_final must be a float32 [B, K] tensor");
    const Tensor par = parents.contiguous(), tok = tokens.contiguous(), len = lengths.contiguous();
    const Tensor flp = from_logprob.contiguous(), g = grad_final.contiguous();
    const std::vector<int32_t> steps_b = per_example_steps(steps, B, T);
    Tensor draws = torch::zeros({B, T, K}, torch::TensorOptions().dtype(torch::kFloat32));
    FirstError error;
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            const int64_t off = b * T * K;
            dbs::TraceView trace;
            trace.steps = steps_b[static_cast<size_t>(b)];
            trace.beam_size = static_cast<int>(K);
            trace.vocab_size = static_cast<int>(V);
            trace.length_penalty_alpha = static_cast<float>(length_penalty_alpha);
            trace.parents = par.data_ptr<int32_t>() + off;
            trace.tokens = tok.data_ptr<int32_t>() + off;
            trace.lengths = len.data_ptr<int32_t>() + off;
            trace.from_logprob = flp.data_ptr<uint8_t>() + off;
            try {
                dbs::final_scores_path_gradient(trace, g.data_ptr<float>() + b * K, draws.data_ptr<float>() + off);
            } catch (const std::invalid_argument& e) {
                error.record(b, e.what(), true);
            }
        }
    });
    error.raise_if_failed();
    return draws;
}

Tensor length_penalty_cpu(const Tensor& lengths, double alpha) {
    TORCH_CHECK_VALUE(!lengths.is_floating_point() && !lengths.is_complex() && lengths.scalar_type() != torch::kBool,
                      "lengths must be an integer tensor");
    TORCH_CHECK_VALUE(std::isfinite(alpha) && alpha >= 0.0, "length_penalty_alpha must be finite and non-negative");
    const Tensor len = lengths.to(torch::kInt64).contiguous();
    Tensor out = torch::empty(len.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
    const int64_t* l = len.data_ptr<int64_t>();
    float* o = out.data_ptr<float>();
    for (int64_t i = 0; i < len.numel(); ++i) {
        const int64_t clamped = std::max<int64_t>(0, std::min<int64_t>(l[i], kIntMax));
        o[i] = dbs::gnmt_length_penalty(static_cast<int>(clamped), static_cast<float>(alpha));
    }
    return out;
}

} // namespace

TORCH_LIBRARY(beamgrad, m) {
    m.def(
        "decode(Tensor log_probs, Tensor? steps, int eos_token, int min_length, float length_penalty_alpha, "
        "Tensor? banned_tokens, int no_repeat_ngram_size, float repetition_penalty, bool validate) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "final_scores_backward(Tensor grad_final, Tensor parents, Tensor tokens, Tensor lengths, "
        "Tensor from_logprob, Tensor? steps, int vocab_size, float length_penalty_alpha) -> Tensor");
    m.def("length_penalty(Tensor lengths, float length_penalty_alpha) -> Tensor");
    m.def(
        "final_scores_path_gradient(Tensor grad_final, Tensor parents, Tensor tokens, Tensor lengths, "
        "Tensor from_logprob, Tensor? steps, int vocab_size, float length_penalty_alpha) -> Tensor");
    m.def(
        "decode_step(Tensor log_probs, Tensor raw_scores, Tensor lengths, Tensor finished, Tensor prefixes, "
        "int eos_token, int min_length, float length_penalty_alpha, Tensor? banned_tokens, int no_repeat_ngram_size, "
        "float repetition_penalty, bool validate) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(beamgrad, CPU, m) {
    m.impl("decode", &decode_cpu);
    m.impl("final_scores_backward", &final_scores_backward_cpu);
    m.impl("decode_step", &decode_step_cpu);
    m.impl("final_scores_path_gradient", &final_scores_path_gradient_cpu);
    m.impl("length_penalty", &length_penalty_cpu);
}

PyMODINIT_FUNC PyInit__C(void) {
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_C", "beamgrad CPU operators, registered as torch.ops.beamgrad.*", -1, nullptr,
    };
    return PyModule_Create(&module);
}
