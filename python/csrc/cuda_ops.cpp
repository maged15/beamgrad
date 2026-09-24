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

constexpr int64_t kIntMax = std::numeric_limits<int>::max();

void check_dim(int64_t value, const char* name) {
    TORCH_CHECK_VALUE(value > 0 && value <= kIntMax, name, " must be in [1, INT_MAX], got ", value);
}

void check_status(int status, const char* what) {
    TORCH_CHECK_VALUE(status != DBS_CUDA_STATUS_INVALID_ARGUMENT, "beamgrad CUDA ", what, ": invalid argument");
    TORCH_CHECK(status == DBS_CUDA_STATUS_OK, "beamgrad CUDA ", what, " failed: ", dbs_cuda_status_string(status));
}

c10::optional<Tensor> device_steps(const c10::optional<Tensor>& steps, const Tensor& like, int64_t B) {
    if (!steps.has_value()) return c10::nullopt;
    TORCH_CHECK_VALUE(steps->dim() == 1 && steps->size(0) == B, "steps must have shape [B] = [", B, "]");
    return steps->to(like.device(), torch::kInt32).contiguous();
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
    const c10::optional<Tensor> steps_i32 = device_steps(steps, x, B);
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
    const c10::optional<Tensor> steps_i32 = device_steps(steps, par, B);
    const DBSCudaDecodeArgs args = make_args(B, T, K, vocab_size, -1, 0, length_penalty_alpha, steps_i32);

    Tensor grad = torch::zeros({B, T, K, vocab_size}, g.options());
    check_status(dbs_cuda_backward(&args, par.data_ptr<int32_t>(), tok.data_ptr<int32_t>(), len.data_ptr<int32_t>(),
                                   flp.data_ptr<uint8_t>(), g.data_ptr<float>(), grad.data_ptr<float>(), nullptr, 0,
                                   at::cuda::getCurrentCUDAStream().stream()),
                 "backward");
    return grad;
}

} // namespace

TORCH_LIBRARY_IMPL(beamgrad, CUDA, m) {
    m.impl("decode", &decode_cuda);
    m.impl("final_scores_backward", &final_scores_backward_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "beamgrad CUDA operators, registered as torch.ops.beamgrad.*";
    m.def("device_available", []() { return dbs_cuda_available() != 0; },
          "Whether the CUDA runtime reports a usable device");
    m.def("max_beam_size", []() { return DBS_CUDA_MAX_BEAM; });
}
