// SPDX-License-Identifier: MIT
//
// beamgrad._C_cuda: CUDA kernels for the beamgrad operators defined in
// beamgrad._C (cpu_ops.cpp), backed by the native engine in cuda/dbs_cuda.cu.
// Work runs on PyTorch's current stream and scratch memory comes from
// PyTorch's caching allocator. Two things read back from the device and so
// synchronize: per-example steps (validated on the device) and validate=true
// (the engine's NaN/+inf flags, one byte per example).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <tuple>

#include "dbs_cuda.h"

namespace {

using Tensor = at::Tensor;
using DecodeOutputs = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using StepOutputs = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

constexpr int64_t kIntMax = std::numeric_limits<int>::max();

void check_dim(int64_t value, const char* name) {
    TORCH_CHECK_VALUE(value > 0 && value <= kIntMax, name, " must be in [1, INT_MAX], got ", value);
}

void check_status(int status, const char* what) {
    TORCH_CHECK_VALUE(status != DBS_CUDA_STATUS_INVALID_ARGUMENT, "beamgrad CUDA ", what, ": invalid argument");
    TORCH_CHECK(status == DBS_CUDA_STATUS_OK, "beamgrad CUDA ", what, " failed: ", dbs_cuda_status_string(status));
}

c10::optional<Tensor> device_steps(const c10::optional<Tensor>& steps, const Tensor& like, int64_t B, int64_t T) {
    if (!steps.has_value()) return c10::nullopt;
    TORCH_CHECK_VALUE(steps->dim() == 1 && steps->size(0) == B, "steps must have shape [B] = [", B, "]");
    TORCH_CHECK_VALUE(!steps->is_floating_point() && !steps->is_complex() && steps->scalar_type() != torch::kBool,
                      "steps must be an integer tensor, got ", steps->scalar_type());
    // Range-checked before the values are narrowed to int32, where 2^32 + 1
    // would pass as 1. Reads one flag back (per-example steps synchronize anyway).
    const Tensor s = steps->to(like.device(), torch::kInt64);
    if (!s.ge(1).logical_and(s.le(T)).all().item<bool>()) {
        const Tensor host = s.cpu();
        const int64_t* p = host.data_ptr<int64_t>();
        for (int64_t b = 0; b < B; ++b) {
            TORCH_CHECK_VALUE(p[b] >= 1 && p[b] <= T, "steps[", b, "] = ", p[b], " must be in [1, ", T, "]");
        }
    }
    return s.to(torch::kInt32).contiguous();
}

DBSCudaDecodeArgs make_args(int64_t B, int64_t T, int64_t K, int64_t V, int64_t eos_token, int64_t min_length,
                            double alpha, const c10::optional<Tensor>& steps) {
    check_dim(B, "B");
    check_dim(T, "T");
    check_dim(K, "K");
    check_dim(V, "V");
    TORCH_CHECK_VALUE(K <= DBS_CUDA_MAX_BEAM, "beam_size ", K, " exceeds the CUDA backend maximum of ", DBS_CUDA_MAX_BEAM);
    TORCH_CHECK_VALUE(eos_token >= -1 && eos_token < V, "eos_token must be -1 or a token id < ", V);
    TORCH_CHECK_VALUE(min_length >= 0 && min_length <= kIntMax, "min_length must be non-negative");
    DBSCudaDecodeArgs a{};
    a.batch_size = static_cast<int>(B);
    a.steps = static_cast<int>(T);
    a.beam_size = static_cast<int>(K);
    a.vocab_size = static_cast<int>(V);
    a.eos_token = static_cast<int>(eos_token);
    a.min_length = static_cast<int>(min_length);
    a.length_penalty_alpha = static_cast<float>(alpha);
    a.steps_per_example = steps.has_value() ? steps->data_ptr<int32_t>() : nullptr;
    return a;
}

Tensor workspace(int64_t bytes, const Tensor& like) {
    TORCH_CHECK_VALUE(bytes >= 0, "beamgrad CUDA: unsupported problem size (see DBS_CUDA_MAX_BEAM and the limits in dbs_cuda.h)");
    return torch::empty({bytes > 0 ? bytes : 1}, like.options().dtype(torch::kUInt8));
}

DecodeOutputs decode_cuda(
    const Tensor& log_probs,
    const c10::optional<Tensor>& steps,
    int64_t eos_token,
    int64_t min_length,
    double length_penalty_alpha,
    const c10::optional<Tensor>& banned_tokens,
    int64_t no_repeat_ngram_size,
    double repetition_penalty,
    bool validate) {
    TORCH_CHECK(log_probs.is_cuda(), "beamgrad::decode (CUDA) expects a CUDA tensor");
    TORCH_CHECK_VALUE(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK_VALUE(log_probs.dim() == 4, "log_probs must have shape [B, T, K, V]");
    TORCH_CHECK_VALUE(std::isfinite(length_penalty_alpha) && length_penalty_alpha >= 0.0,
                      "length_penalty_alpha must be finite and non-negative");
    TORCH_CHECK_VALUE(std::isfinite(repetition_penalty) && repetition_penalty > 0.0,
                      "repetition_penalty must be finite and positive");
    const c10::cuda::CUDAGuard guard(log_probs.device());
    const Tensor x = log_probs.contiguous();
    const int64_t B = x.size(0), T = x.size(1), K = x.size(2), V = x.size(3);
    const c10::optional<Tensor> steps_i32 = device_steps(steps, x, B, T);
    DBSCudaDecodeArgs args = make_args(B, T, K, V, eos_token, min_length, length_penalty_alpha, steps_i32);
    args.no_repeat_ngram_size = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(no_repeat_ngram_size, kIntMax)));
    args.repetition_penalty = static_cast<float>(repetition_penalty);
    Tensor banned;
    if (banned_tokens.has_value()) {
        banned = banned_tokens->to(x.device(), torch::kUInt8).contiguous();
        TORCH_CHECK_VALUE(banned.dim() == 1 && banned.size(0) == V, "banned_tokens must be a [V] mask");
        args.banned_tokens = banned.data_ptr<uint8_t>();
    }

    const auto f32 = x.options();
    const auto i32 = x.options().dtype(torch::kInt32);
    const auto u8 = x.options().dtype(torch::kUInt8);
    Tensor final_scores = torch::empty({B, K}, f32);
    Tensor final_raw = torch::empty({B, K}, f32);
    Tensor final_lengths = torch::empty({B, K}, i32);
    Tensor tokens = torch::empty({B, T, K}, i32);
    Tensor parents = torch::empty({B, T, K}, i32);
    Tensor lengths = torch::empty({B, T, K}, i32);
    Tensor scores = torch::empty({B, T, K}, f32);
    Tensor raw_scores = torch::empty({B, T, K}, f32);
    Tensor from_logprob = torch::empty({B, T, K}, u8);
    Tensor invalid = validate ? torch::empty({B}, u8) : Tensor();

    DBSCudaDecodeOutputs out{};
    out.final_scores = final_scores.data_ptr<float>();
    out.final_raw_scores = final_raw.data_ptr<float>();
    out.final_lengths = final_lengths.data_ptr<int32_t>();
    out.tokens = tokens.data_ptr<int32_t>();
    out.parents = parents.data_ptr<int32_t>();
    out.lengths = lengths.data_ptr<int32_t>();
    out.scores = scores.data_ptr<float>();
    out.raw_scores = raw_scores.data_ptr<float>();
    out.from_logprob = from_logprob.data_ptr<uint8_t>();
    out.invalid_input = validate ? invalid.data_ptr<uint8_t>() : nullptr;

    const int64_t ws_bytes = dbs_cuda_decode_workspace_size(&args);
    Tensor ws = workspace(ws_bytes, x);
    check_status(dbs_cuda_decode(x.data_ptr<float>(), &args, &out, ws.data_ptr(), ws_bytes,
                                 at::cuda::getCurrentCUDAStream().stream()),
                 "decode");
    if (validate) {
        const Tensor flags = invalid.cpu();  // one byte per example; synchronizes the stream
        const uint8_t* f = flags.data_ptr<uint8_t>();
        for (int64_t b = 0; b < B; ++b) {
            TORCH_CHECK_VALUE(f[b] == 0, "example ", b, ": log_probs contains NaN or +inf in a row the search reads");
        }
    }
    return {final_scores, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob};
}

StepOutputs decode_step_cuda(
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
    TORCH_CHECK(log_probs.is_cuda(), "beamgrad::decode_step (CUDA) expects CUDA tensors");
    TORCH_CHECK_VALUE(log_probs.scalar_type() == torch::kFloat32 && log_probs.dim() == 3,
                      "log_probs must be a float32 [B, K, V] tensor");
    const c10::cuda::CUDAGuard guard(log_probs.device());
    const Tensor x = log_probs.contiguous();
    const int64_t B = x.size(0), K = x.size(1), V = x.size(2);
    for (const Tensor* t : {&raw_scores, &lengths, &finished, &prefixes}) {
        TORCH_CHECK_VALUE(t->device() == x.device(), "the beam state must be on the device of log_probs");
    }
    TORCH_CHECK_VALUE(raw_scores.scalar_type() == torch::kFloat32 && raw_scores.sizes() == torch::IntArrayRef({B, K}),
                      "raw_scores must be a float32 [B, K] tensor");
    TORCH_CHECK_VALUE(lengths.scalar_type() == torch::kInt32 && lengths.sizes() == torch::IntArrayRef({B, K}),
                      "lengths must be an int32 [B, K] tensor");
    TORCH_CHECK_VALUE(finished.scalar_type() == torch::kUInt8 && finished.sizes() == torch::IntArrayRef({B, K}),
                      "finished must be a uint8 [B, K] tensor");
    TORCH_CHECK_VALUE(prefixes.scalar_type() == torch::kInt32 && prefixes.dim() == 3 && prefixes.size(0) == B &&
                          prefixes.size(1) == K && prefixes.size(2) <= kIntMax,
                      "prefixes must be an int32 [B, K, L] tensor");
    TORCH_CHECK_VALUE(std::isfinite(length_penalty_alpha) && length_penalty_alpha >= 0.0,
                      "length_penalty_alpha must be finite and non-negative");
    TORCH_CHECK_VALUE(std::isfinite(repetition_penalty) && repetition_penalty > 0.0,
                      "repetition_penalty must be finite and positive");
    DBSCudaDecodeArgs args = make_args(B, 1, K, V, eos_token, min_length, length_penalty_alpha, c10::nullopt);
    args.no_repeat_ngram_size = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(no_repeat_ngram_size, kIntMax)));
    args.repetition_penalty = static_cast<float>(repetition_penalty);
    Tensor banned;
    if (banned_tokens.has_value()) {
        banned = banned_tokens->to(x.device(), torch::kUInt8).contiguous();
        TORCH_CHECK_VALUE(banned.dim() == 1 && banned.size(0) == V, "banned_tokens must be a [V] mask");
        args.banned_tokens = banned.data_ptr<uint8_t>();
    }

    // The state is advanced in copies, so the inputs are left untouched.
    Tensor raw = raw_scores.contiguous().clone();
    Tensor len = lengths.contiguous().clone();
    Tensor fin = finished.contiguous().clone();
    const Tensor pre = prefixes.contiguous();
    Tensor tokens = torch::empty({B, K}, x.options().dtype(torch::kInt32));
    Tensor parents = torch::empty({B, K}, x.options().dtype(torch::kInt32));
    Tensor scores = torch::empty({B, K}, x.options());
    Tensor final_scores = torch::empty({B, K}, x.options());
    Tensor from_logprob = torch::empty({B, K}, x.options().dtype(torch::kUInt8));
    Tensor invalid = validate ? torch::empty({B}, x.options().dtype(torch::kUInt8)) : Tensor();

    DBSCudaDecodeOutputs out{};
    out.tokens = tokens.data_ptr<int32_t>();
    out.parents = parents.data_ptr<int32_t>();
    out.scores = scores.data_ptr<float>();
    out.from_logprob = from_logprob.data_ptr<uint8_t>();
    out.final_scores = final_scores.data_ptr<float>();
    out.invalid_input = validate ? invalid.data_ptr<uint8_t>() : nullptr;
    const DBSCudaBeamState state{raw.data_ptr<float>(), len.data_ptr<int32_t>(), fin.data_ptr<uint8_t>(),
                                 pre.numel() > 0 ? pre.data_ptr<int32_t>() : nullptr, static_cast<int>(pre.size(2)), 0};
    const int64_t ws_bytes = dbs_cuda_decode_step_workspace_size(&args, state.prefix_stride);
    Tensor ws = workspace(ws_bytes, x);
    check_status(dbs_cuda_decode_step(x.data_ptr<float>(), &args, &state, &out, ws.data_ptr(), ws_bytes,
                                      at::cuda::getCurrentCUDAStream().stream()),
                 "decode_step");
    if (validate) {
        const Tensor flags = invalid.cpu();  // one byte per example; synchronizes the stream
        const uint8_t* f = flags.data_ptr<uint8_t>();
        for (int64_t b = 0; b < B; ++b) {
            TORCH_CHECK_VALUE(f[b] == 0, "example ", b, ": log_probs contains NaN or +inf in a row the search reads");
        }
    }
    // The new state's lengths and raw scores are the selected beams'.
    return {tokens, parents, len, scores, raw, from_logprob, fin, final_scores};
}

Tensor final_scores_backward_cuda(
    const Tensor& grad_final,
    const Tensor& parents,
    const Tensor& tokens,
    const Tensor& lengths,
    const Tensor& from_logprob,
    const c10::optional<Tensor>& steps,
    int64_t vocab_size,
    double length_penalty_alpha) {
    TORCH_CHECK(parents.is_cuda(), "beamgrad::final_scores_backward (CUDA) expects CUDA tensors");
    TORCH_CHECK_VALUE(parents.dim() == 3 && parents.scalar_type() == torch::kInt32, "parents must be an int32 [B, T, K] tensor");
    const c10::cuda::CUDAGuard guard(parents.device());
    const int64_t B = parents.size(0), T = parents.size(1), K = parents.size(2);
    for (const Tensor* t : {&tokens, &lengths}) {
        TORCH_CHECK_VALUE(t->device() == parents.device() && t->sizes() == parents.sizes() && t->scalar_type() == torch::kInt32,
                          "tokens and lengths must be int32 [B, T, K] tensors on the same device");
    }
    TORCH_CHECK_VALUE(from_logprob.device() == parents.device() && from_logprob.sizes() == parents.sizes() &&
                          from_logprob.scalar_type() == torch::kUInt8,
                      "from_logprob must be a uint8 [B, T, K] tensor on the same device");
    TORCH_CHECK_VALUE(grad_final.device() == parents.device() && grad_final.scalar_type() == torch::kFloat32 &&
                          grad_final.dim() == 2 && grad_final.size(0) == B && grad_final.size(1) == K,
                      "grad_final must be a float32 [B, K] tensor on the same device");
    const Tensor par = parents.contiguous(), tok = tokens.contiguous(), len = lengths.contiguous();
    const Tensor flp = from_logprob.contiguous(), g = grad_final.contiguous();
    const c10::optional<Tensor> steps_i32 = device_steps(steps, par, B, T);
    const DBSCudaDecodeArgs args = make_args(B, T, K, vocab_size, -1, 0, length_penalty_alpha, steps_i32);

    Tensor grad = torch::zeros({B, T, K, vocab_size}, g.options());
    check_status(dbs_cuda_backward(&args, par.data_ptr<int32_t>(), tok.data_ptr<int32_t>(), len.data_ptr<int32_t>(),
                                   flp.data_ptr<uint8_t>(), g.data_ptr<float>(), grad.data_ptr<float>(), nullptr, 0,
                                   at::cuda::getCurrentCUDAStream().stream()),
                 "backward");
    return grad;
}

Tensor final_scores_path_gradient_cuda(
    const Tensor& grad_final,
    const Tensor& parents,
    const Tensor& tokens,
    const Tensor& lengths,
    const Tensor& from_logprob,
    const c10::optional<Tensor>& steps,
    int64_t vocab_size,
    double length_penalty_alpha) {
    TORCH_CHECK(parents.is_cuda(), "beamgrad::final_scores_path_gradient (CUDA) expects CUDA tensors");
    TORCH_CHECK_VALUE(parents.dim() == 3 && parents.scalar_type() == torch::kInt32, "parents must be an int32 [B, T, K] tensor");
    const c10::cuda::CUDAGuard guard(parents.device());
    const int64_t B = parents.size(0), T = parents.size(1), K = parents.size(2);
    for (const Tensor* t : {&tokens, &lengths}) {
        TORCH_CHECK_VALUE(t->device() == parents.device() && t->sizes() == parents.sizes() && t->scalar_type() == torch::kInt32,
                          "tokens and lengths must be int32 [B, T, K] tensors on the same device");
    }
    TORCH_CHECK_VALUE(from_logprob.device() == parents.device() && from_logprob.sizes() == parents.sizes() &&
                          from_logprob.scalar_type() == torch::kUInt8,
                      "from_logprob must be a uint8 [B, T, K] tensor on the same device");
    TORCH_CHECK_VALUE(grad_final.device() == parents.device() && grad_final.scalar_type() == torch::kFloat32 &&
                          grad_final.dim() == 2 && grad_final.size(0) == B && grad_final.size(1) == K,
                      "grad_final must be a float32 [B, K] tensor on the same device");
    const Tensor par = parents.contiguous(), tok = tokens.contiguous(), len = lengths.contiguous();
    const Tensor flp = from_logprob.contiguous(), g = grad_final.contiguous();
    const c10::optional<Tensor> steps_i32 = device_steps(steps, par, B, T);
    const DBSCudaDecodeArgs args = make_args(B, T, K, vocab_size, -1, 0, length_penalty_alpha, steps_i32);

    Tensor draws = torch::empty({B, T, K}, g.options());
    check_status(dbs_cuda_path_gradient(&args, par.data_ptr<int32_t>(), tok.data_ptr<int32_t>(), len.data_ptr<int32_t>(),
                                        flp.data_ptr<uint8_t>(), g.data_ptr<float>(), draws.data_ptr<float>(),
                                        at::cuda::getCurrentCUDAStream().stream()),
                 "path gradient");
    return draws;
}

Tensor length_penalty_cuda(const Tensor& lengths, double alpha) {
    TORCH_CHECK(lengths.is_cuda(), "beamgrad::length_penalty (CUDA) expects a CUDA tensor");
    TORCH_CHECK_VALUE(!lengths.is_floating_point() && !lengths.is_complex() && lengths.scalar_type() != torch::kBool,
                      "lengths must be an integer tensor");
    TORCH_CHECK_VALUE(std::isfinite(alpha) && alpha >= 0.0, "length_penalty_alpha must be finite and non-negative");
    const c10::cuda::CUDAGuard guard(lengths.device());
    const Tensor len = lengths.clamp(0, kIntMax).to(torch::kInt32).contiguous();
    Tensor out = torch::empty(len.sizes(), len.options().dtype(torch::kFloat32));
    check_status(dbs_cuda_length_penalty(len.data_ptr<int32_t>(), len.numel(), static_cast<float>(alpha),
                                         out.data_ptr<float>(), at::cuda::getCurrentCUDAStream().stream()),
                 "length penalty");
    return out;
}

} // namespace

TORCH_LIBRARY_IMPL(beamgrad, CUDA, m) {
    m.impl("decode", &decode_cuda);
    m.impl("final_scores_backward", &final_scores_backward_cuda);
    m.impl("final_scores_path_gradient", &final_scores_path_gradient_cuda);
    m.impl("length_penalty", &length_penalty_cuda);
    m.impl("decode_step", &decode_step_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "beamgrad CUDA operators, registered as torch.ops.beamgrad.*";
    m.def("device_available", []() { return dbs_cuda_available() != 0; },
          "Whether the CUDA runtime reports a usable device");
    m.def("max_beam_size", []() { return DBS_CUDA_MAX_BEAM; });
}
