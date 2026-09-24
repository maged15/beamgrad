// SPDX-License-Identifier: MIT
//
// beamgrad._C_cuda: CUDA operators backed by the native engine in
// cuda/dbs_cuda.cu. Same contract as beamgrad._C (see cpu_ops.cpp). Work runs
// on PyTorch's current stream, scratch memory comes from PyTorch's caching
// allocator, and nothing synchronizes unless per-example steps are supplied
// (their validation reads one flag back from the device).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <vector>

#include "dbs_cuda.h"

namespace {

constexpr int64_t kIntMax = std::numeric_limits<int>::max();

void check_dim(int64_t value, const char* name) {
    TORCH_CHECK(value > 0 && value <= kIntMax, name, " must be in [1, INT_MAX], got ", value);
}

void check_status(int status, const char* what) {
    TORCH_CHECK(status == DBS_CUDA_STATUS_OK, "beamgrad CUDA ", what, " failed: ", dbs_cuda_status_string(status));
}

c10::optional<torch::Tensor> device_steps(const c10::optional<torch::Tensor>& steps, const torch::Tensor& like, int64_t B) {
    if (!steps.has_value()) return c10::nullopt;
    TORCH_CHECK(steps->dim() == 1 && steps->size(0) == B, "steps must have shape [B] = [", B, "]");
    return steps->to(like.device(), torch::kInt32).contiguous();
}

DBSCudaDecodeArgs make_args(int64_t B, int64_t T, int64_t K, int64_t V, int64_t eos_token, int64_t min_length,
                            double alpha, const c10::optional<torch::Tensor>& steps) {
    check_dim(B, "B");
    check_dim(T, "T");
    check_dim(K, "K");
    check_dim(V, "V");
    TORCH_CHECK(K <= DBS_CUDA_MAX_BEAM, "beam_size ", K, " exceeds the CUDA backend maximum of ", DBS_CUDA_MAX_BEAM);
    TORCH_CHECK(eos_token >= -1 && eos_token < V, "eos_token must be -1 or < vocab size (", V, ")");
    TORCH_CHECK(min_length >= 0 && min_length <= kIntMax, "min_length must be non-negative");
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

torch::Tensor workspace(int64_t bytes, const torch::Tensor& like) {
    TORCH_CHECK(bytes >= 0, "beamgrad CUDA: unsupported problem size (see DBS_CUDA_MAX_BEAM and limits in dbs_cuda.h)");
    return torch::empty({bytes > 0 ? bytes : 1}, like.options().dtype(torch::kUInt8));
}

std::vector<torch::Tensor> decode(
    const torch::Tensor& log_probs,
    const c10::optional<torch::Tensor>& steps,
    int64_t eos_token,
    int64_t min_length,
    double length_penalty_alpha) {
    TORCH_CHECK(log_probs.is_cuda(), "beamgrad._C_cuda.decode expects a CUDA tensor");
    TORCH_CHECK(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(log_probs.dim() == 4, "log_probs must have shape [B, T, K, V]");
    TORCH_CHECK(log_probs.is_contiguous(), "log_probs must be contiguous");
    const c10::cuda::CUDAGuard guard(log_probs.device());
    const int64_t B = log_probs.size(0), T = log_probs.size(1), K = log_probs.size(2), V = log_probs.size(3);
    const c10::optional<torch::Tensor> steps_i32 = device_steps(steps, log_probs, B);
    const DBSCudaDecodeArgs args = make_args(B, T, K, V, eos_token, min_length, length_penalty_alpha, steps_i32);

    const auto f32 = log_probs.options();
    const auto i32 = log_probs.options().dtype(torch::kInt32);
    torch::Tensor final_scores = torch::empty({B, K}, f32);
    torch::Tensor final_raw = torch::empty({B, K}, f32);
    torch::Tensor final_lengths = torch::empty({B, K}, i32);
    torch::Tensor tokens = torch::empty({B, T, K}, i32);
    torch::Tensor parents = torch::empty({B, T, K}, i32);
    torch::Tensor lengths = torch::empty({B, T, K}, i32);
    torch::Tensor scores = torch::empty({B, T, K}, f32);
    torch::Tensor raw_scores = torch::empty({B, T, K}, f32);
    torch::Tensor from_logprob = torch::empty({B, T, K}, log_probs.options().dtype(torch::kUInt8));

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

    const int64_t ws_bytes = dbs_cuda_decode_workspace_size(&args);
    torch::Tensor ws = workspace(ws_bytes, log_probs);
    check_status(dbs_cuda_decode(log_probs.data_ptr<float>(), &args, &out, ws.data_ptr(), ws_bytes,
                                 at::cuda::getCurrentCUDAStream().stream()),
                 "decode");
    return {final_scores, final_raw, final_lengths, tokens, parents, lengths, scores, raw_scores, from_logprob};
}

torch::Tensor backward(
    const torch::Tensor& grad_final,
    const torch::Tensor& parents,
    const torch::Tensor& tokens,
    const torch::Tensor& lengths,
    const torch::Tensor& from_logprob,
    const c10::optional<torch::Tensor>& steps,
    int64_t vocab_size,
    double length_penalty_alpha) {
    TORCH_CHECK(parents.is_cuda() && parents.dim() == 3 && parents.scalar_type() == torch::kInt32 && parents.is_contiguous(),
                "parents must be a contiguous int32 [B, T, K] CUDA tensor");
    const c10::cuda::CUDAGuard guard(parents.device());
    const int64_t B = parents.size(0), T = parents.size(1), K = parents.size(2);
    for (const torch::Tensor* t : {&tokens, &lengths}) {
        TORCH_CHECK(t->device() == parents.device() && t->sizes() == parents.sizes() &&
                        t->scalar_type() == torch::kInt32 && t->is_contiguous(),
                    "tokens and lengths must be contiguous int32 [B, T, K] tensors on the same device");
    }
    TORCH_CHECK(from_logprob.device() == parents.device() && from_logprob.sizes() == parents.sizes() &&
                    from_logprob.scalar_type() == torch::kUInt8 && from_logprob.is_contiguous(),
                "from_logprob must be a contiguous uint8 [B, T, K] tensor on the same device");
    TORCH_CHECK(grad_final.device() == parents.device() && grad_final.scalar_type() == torch::kFloat32 &&
                    grad_final.is_contiguous() && grad_final.dim() == 2 && grad_final.size(0) == B &&
                    grad_final.size(1) == K,
                "grad_final must be a contiguous float32 [B, K] tensor on the same device");
    const c10::optional<torch::Tensor> steps_i32 = device_steps(steps, parents, B);
    const DBSCudaDecodeArgs args = make_args(B, T, K, vocab_size, -1, 0, length_penalty_alpha, steps_i32);

    torch::Tensor grad = torch::zeros({B, T, K, vocab_size}, grad_final.options());
    const int64_t ws_bytes = dbs_cuda_backward_workspace_size(&args);
    torch::Tensor ws = workspace(ws_bytes, parents);
    check_status(dbs_cuda_backward(&args, parents.data_ptr<int32_t>(), tokens.data_ptr<int32_t>(),
                                   lengths.data_ptr<int32_t>(), from_logprob.data_ptr<uint8_t>(),
                                   grad_final.data_ptr<float>(), grad.data_ptr<float>(), ws.data_ptr(), ws_bytes,
                                   at::cuda::getCurrentCUDAStream().stream()),
                 "backward");
    return grad;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "beamgrad CUDA operators";
    m.def("decode", &decode, "Hard beam search on CUDA (see cuda/dbs_cuda.cu)",
          py::arg("log_probs"), py::arg("steps"), py::arg("eos_token"), py::arg("min_length"),
          py::arg("length_penalty_alpha"));
    m.def("backward", &backward, "Surrogate gradient of the final scores on CUDA",
          py::arg("grad_final"), py::arg("parents"), py::arg("tokens"), py::arg("lengths"),
          py::arg("from_logprob"), py::arg("steps"), py::arg("vocab_size"), py::arg("length_penalty_alpha"));
    m.def("device_available", []() { return dbs_cuda_available() != 0; },
          "Whether the CUDA runtime reports a usable device");
    m.def("max_beam_size", []() { return DBS_CUDA_MAX_BEAM; });
}
