// SPDX-License-Identifier: MIT
//
// CPU beam-search decoder with surrogate gradients.
//
// Forward: hard beam search with a deterministic total order over candidates
// (score, raw score, parent, token, length, origin), GNMT length penalty, EOS
// carry-forward, and optional decoding constraints. Alongside the selected
// beams, each step records a relaxed candidate pool (top K * multiplier) with
// sigmoid k-hot inclusion weights.
//
// Backward: gradients flow from final scores, selected-beam softmax weights,
// and relaxed-pool weights back to the log-prob entries that produced each
// selected/pooled candidate. The sparse variant is the default; the dense
// variant materialises [T, K, V] and is capped by max_dense_gradient_elements.
#pragma once

#include "kernels.hpp"

namespace dbs {

class BeamSearchDecoder {
public:
    explicit BeamSearchDecoder(BeamOptions options);

    const BeamOptions& options() const noexcept { return opt_; }

    // log_probs: contiguous [steps, beam_size, vocab_size] float32.
    DecodeResult decode(const float* log_probs, int steps, int vocab_size) const;
    DecodeResult decode_constrained(
        const float* log_probs,
        int steps,
        int vocab_size,
        const DecodeConstraints* constraints) const;

    // Any gradient pointer may be null. Shapes: grad_selected_weights [T*K],
    // grad_relaxed_weights [T*P], grad_final_scores [K].
    BackwardResult backward(
        const DecodeResult& fwd,
        const float* grad_selected_weights,
        const float* grad_relaxed_weights,
        const float* grad_final_scores) const;
    BackwardResult backward_sparse(
        const DecodeResult& fwd,
        const float* grad_selected_weights,
        const float* grad_relaxed_weights,
        const float* grad_final_scores) const;

private:
    BeamOptions opt_;
};

} // namespace dbs
