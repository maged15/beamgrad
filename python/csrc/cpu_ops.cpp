// SPDX-License-Identifier: MIT
//
// beamgrad._C: CPU operators. The libdbs sources are compiled into this module,
// so the decoder is called directly and batches are decoded in parallel.
//
// Both operators share their contract with beamgrad._C_cuda (cuda_ops.cpp):
//   decode(log_probs[B,T,K,V] f32, steps[B] i32 | None, eos, min_length, alpha)
//     -> final_scores[B,K], final_raw_scores[B,K], final_lengths[B,K],
//        tokens, parents, lengths [B,T,K] i32, scores, raw_scores [B,T,K] f32,
//        from_logprob [B,T,K] u8
//   backward(grad_final[B,K], parents, tokens, lengths, from_logprob, steps,
//            vocab_size, alpha) -> grad_log_probs[B,T,K,V]
#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <cstring>
#include <limits>
#include <vector>

#include "decoder.hpp"

namespace {

constexpr int64_t kIntMax = std::numeric_limits<int>::max();

void check_dim(int64_t value, const char* name) {
    TORCH_CHECK(value > 0 && value <= kIntMax, name, " must be in [1, INT_MAX], got ", value);
}

std::vector<int32_t> per_example_steps(const c10::optional<torch::Tensor>& steps, int64_t B, int64_t T) {
    std::vector<int32_t> out(static_cast<size_t>(B), static_cast<int32_t>(T));
    if (!steps.has_value()) return out;
    const torch::Tensor s = steps->to(torch::kCPU, torch::kInt64).contiguous();
    TORCH_CHECK(s.dim() == 1 && s.size(0) == B, "steps must have shape [B] = [", B, "]");
    const int64_t* p = s.data_ptr<int64_t>();
    for (int64_t b = 0; b < B; ++b) {
        TORCH_CHECK(p[b] >= 1 && p[b] <= T, "steps[", b, "] = ", p[b], " must be in [1, ", T, "]");
        out[static_cast<size_t>(b)] = static_cast<int32_t>(p[b]);
    }
    return out;
}

dbs::BeamOptions make_options(int64_t beam_size, int64_t eos_token, int64_t min_length, double alpha) {
    TORCH_CHECK(eos_token >= -1 && eos_token <= kIntMax, "eos_token must be -1 or a token id");
    TORCH_CHECK(min_length >= 0 && min_length <= kIntMax, "min_length must be non-negative");
    dbs::BeamOptions opt;
    opt.beam_size = static_cast<int>(beam_size);
    opt.eos_token = static_cast<int>(eos_token);
    opt.min_length = static_cast<int>(min_length);
    opt.length_penalty_alpha = static_cast<float>(alpha);
    // The relaxed pool never changes which beams are selected; keep it minimal.
    opt.relaxed_pool_multiplier = 1;
    opt.validate_inputs = 0;
    return opt;
}

std::vector<torch::Tensor> decode(
    const torch::Tensor& log_probs,
    const c10::optional<torch::Tensor>& steps,
    int64_t eos_token,
    int64_t min_length,
    double length_penalty_alpha) {
    TORCH_CHECK(log_probs.device().is_cpu(), "beamgrad._C.decode expects a CPU tensor");
    TORCH_CHECK(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(log_probs.dim() == 4, "log_probs must have shape [B, T, K, V]");
    TORCH_CHECK(log_probs.is_contiguous(), "log_probs must be contiguous");
    const int64_t B = log_probs.size(0), T = log_probs.size(1), K = log_probs.size(2), V = log_probs.size(3);
    check_dim(B, "B");
    check_dim(T, "T");
    check_dim(K, "K");
    check_dim(V, "V");
    TORCH_CHECK(eos_token < V, "eos_token (", eos_token, ") must be < vocab size (", V, ")");

    const std::vector<int32_t> steps_b = per_example_steps(steps, B, T);
    const dbs::BeamSearchDecoder decoder(make_options(K, eos_token, min_length, length_penalty_alpha));

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    torch::Tensor final_scores = torch::full({B, K}, neg_inf, f32);
    torch::Tensor final_raw = torch::full({B, K}, neg_inf, f32);
    torch::Tensor final_lengths = torch::zeros({B, K}, i32);
    torch::Tensor tokens = torch::full({B, T, K}, -1, i32);
    torch::Tensor parents = torch::full({B, T, K}, -1, i32);
    torch::Tensor lengths = torch::zeros({B, T, K}, i32);
    torch::Tensor scores = torch::full({B, T, K}, neg_inf, f32);
    torch::Tensor raw_scores = torch::full({B, T, K}, neg_inf, f32);
    torch::Tensor from_logprob = torch::zeros({B, T, K}, torch::TensorOptions().dtype(torch::kUInt8));

    const float* x = log_probs.data_ptr<float>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            const int Tb = steps_b[static_cast<size_t>(b)];
            const dbs::DecodeResult r = decoder.decode(x + b * T * K * V, Tb, static_cast<int>(V));
            const size_t n = static_cast<size_t>(Tb) * static_cast<size_t>(K);
            const int64_t step_off = b * T * K;
            std::memcpy(tokens.data_ptr<int32_t>() + step_off, r.tokens.data(), n * sizeof(int32_t));
            std::memcpy(parents.data_ptr<int32_t>() + step_off, r.parents.data(), n * sizeof(int32_t));
            std::memcpy(lengths.data_ptr<int32_t>() + step_off, r.lengths.data(), n * sizeof(int32_t));
            std::memcpy(scores.data_ptr<float>() + step_off, r.scores.data(), n * sizeof(float));
            std::memcpy(raw_scores.data_ptr<float>() + step_off, r.raw_scores.data(), n * sizeof(float));
            std::memcpy(from_logprob.data_ptr<uint8_t>() + step_off, r.from_logprob.data(), n);
            std::memcpy(final_scores.data_ptr<float>() + b * K, r.final_scores.data(), static_cast<size_t>(K) * sizeof(float));
            std::memcpy(final_raw.data_ptr<float>() + b * K, r.final_raw_scores.data(), static_cast<size_t>(K) * sizeof(float));
            std::memcpy(final_lengths.data_ptr<int32_t>() + b * K,
                        r.lengths.data() + static_cast<size_t>(Tb - 1) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(int32_t));
        }
    });
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
    TORCH_CHECK(parents.dim() == 3 && parents.scalar_type() == torch::kInt32 && parents.is_contiguous(),
                "parents must be a contiguous int32 [B, T, K] tensor");
    const int64_t B = parents.size(0), T = parents.size(1), K = parents.size(2), V = vocab_size;
    check_dim(V, "vocab_size");
    for (const torch::Tensor* t : {&tokens, &lengths}) {
        TORCH_CHECK(t->sizes() == parents.sizes() && t->scalar_type() == torch::kInt32 && t->is_contiguous(),
                    "tokens and lengths must be contiguous int32 [B, T, K] tensors");
    }
    TORCH_CHECK(from_logprob.sizes() == parents.sizes() && from_logprob.scalar_type() == torch::kUInt8 &&
                    from_logprob.is_contiguous(),
                "from_logprob must be a contiguous uint8 [B, T, K] tensor");
    TORCH_CHECK(grad_final.device().is_cpu() && grad_final.scalar_type() == torch::kFloat32 &&
                    grad_final.is_contiguous() && grad_final.dim() == 2 && grad_final.size(0) == B &&
                    grad_final.size(1) == K,
                "grad_final must be a contiguous float32 [B, K] CPU tensor");
    // The trace is only trusted after a range check: it indexes memory below.
    TORCH_CHECK(parents.ge(-1).logical_and(parents.lt(K)).all().item<bool>(), "parents out of range");
    TORCH_CHECK(tokens.ge(-1).logical_and(tokens.lt(V)).all().item<bool>(), "tokens out of range");

    const std::vector<int32_t> steps_b = per_example_steps(steps, B, T);
    const dbs::BeamSearchDecoder decoder(make_options(K, -1, 0, length_penalty_alpha));
    torch::Tensor grad = torch::zeros({B, T, K, V}, torch::TensorOptions().dtype(torch::kFloat32));

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            const int Tb = steps_b[static_cast<size_t>(b)];
            const size_t n = static_cast<size_t>(Tb) * static_cast<size_t>(K);
            const int64_t off = b * T * K;
            // Rebuild the parts of the forward result the final-score backward reads.
            dbs::DecodeResult fwd;
            fwd.steps = Tb;
            fwd.beam_size = static_cast<int>(K);
            fwd.vocab_size = static_cast<int>(V);
            fwd.relaxed_pool_size = static_cast<int>(K);
            fwd.length_penalty_alpha = static_cast<float>(length_penalty_alpha);
            fwd.parents.assign(parents.data_ptr<int32_t>() + off, parents.data_ptr<int32_t>() + off + n);
            fwd.tokens.assign(tokens.data_ptr<int32_t>() + off, tokens.data_ptr<int32_t>() + off + n);
            fwd.lengths.assign(lengths.data_ptr<int32_t>() + off, lengths.data_ptr<int32_t>() + off + n);
            fwd.from_logprob.assign(from_logprob.data_ptr<uint8_t>() + off, from_logprob.data_ptr<uint8_t>() + off + n);
            fwd.weights.assign(n, 0.0f);
            const dbs::BackwardResult r =
                decoder.backward_sparse(fwd, nullptr, nullptr, grad_final.data_ptr<float>() + b * K);
            float* g = grad.data_ptr<float>() + b * T * K * V;
            for (size_t i = 0; i < r.sparse_logprob_indices.size(); ++i) {
                g[r.sparse_logprob_indices[i]] += r.sparse_logprob_values[i];
            }
        }
    });
    return grad;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "beamgrad CPU operators";
    m.def("decode", &decode, "Hard beam search on CPU (see python/csrc/cpu_ops.cpp)",
          py::arg("log_probs"), py::arg("steps"), py::arg("eos_token"), py::arg("min_length"),
          py::arg("length_penalty_alpha"));
    m.def("backward", &backward, "Surrogate gradient of the final scores on CPU",
          py::arg("grad_final"), py::arg("parents"), py::arg("tokens"), py::arg("lengths"),
          py::arg("from_logprob"), py::arg("steps"), py::arg("vocab_size"), py::arg("length_penalty_alpha"));
}
