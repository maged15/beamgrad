// SPDX-License-Identifier: MIT
//
// CPU beam-search decoder with surrogate gradients.
//
// Forward: hard beam search with a deterministic total order over candidates
// (score, raw score, parent, token, length, origin), GNMT length penalty, EOS
// carry-forward, and optional decoding constraints. Optionally, each step also
// records a relaxed candidate pool (top K * multiplier) with sigmoid k-hot
// inclusion weights.
//
// Backward: gradients flow from final scores, selected-beam softmax weights,
// and relaxed-pool weights back to the log-prob entries that produced each
// selected/pooled candidate. The sparse variant is the default; the dense
// variant materialises [T, K, V] and is capped by max_dense_gradient_elements.
//
// Every backward reads the beam size, vocabulary, temperatures and length
// penalty from the forward result, so it accepts results decoded with other
// options (for example per-example beam sizes from a variable batch).
#pragma once

#include "kernels.hpp"

#include <functional>

namespace dbs {

// Where a search writes its per-step trace. Selected-beam arrays are [T * K],
// pool arrays [T * P]; any pointer may be null when the output is not needed
// (pool pointers must be null when the relaxed pool is disabled).
struct TraceOutputs {
    int32_t* parents = nullptr;
    int32_t* tokens = nullptr;
    int32_t* lengths = nullptr;
    float* scores = nullptr;
    float* raw_scores = nullptr;
    uint8_t* from_logprob = nullptr;
    float* weights = nullptr;  // selected-beam softmax weights

    int32_t* pool_parents = nullptr;
    int32_t* pool_tokens = nullptr;
    int32_t* pool_lengths = nullptr;
    float* pool_scores = nullptr;
    float* pool_raw_scores = nullptr;
    float* relaxed_weights = nullptr;
    uint8_t* pool_from_logprob = nullptr;
};

// The beams entering a model step, as passed to a model-step callback.
struct ModelStepInfo {
    int step = 0;
    int beam_size = 0;
    int vocab_size = 0;
    const int32_t* parents = nullptr;     // [K] parent slot at step - 1 (-1 at step 0 and for dead beams)
    const int32_t* tokens = nullptr;      // [K] token emitted at step - 1 (-1 at step 0 and for dead beams)
    const int32_t* lengths = nullptr;     // [K]
    const float* scores = nullptr;        // [K] raw / length penalty (-inf for dead beams)
    const float* raw_scores = nullptr;    // [K] cumulative log-probabilities
    const uint8_t* finished = nullptr;    // [K] non-zero after EOS
    const int32_t* prefixes = nullptr;    // [K * step] tokens of each beam's path at steps 0..step-1
};

// Fills rows[K * V] (pre-filled with -inf) for the step described by `info`.
// Throws to abort the decode.
using ModelStepFunction = std::function<void(const ModelStepInfo& info, float* rows)>;

// A decode trace as flat arrays, for the final-score backward.
struct TraceView {
    int steps = 0;
    int beam_size = 0;
    int vocab_size = 0;
    float length_penalty_alpha = 0.0f;
    const int32_t* parents = nullptr;       // [T * K]
    const int32_t* tokens = nullptr;        // [T * K]
    const int32_t* lengths = nullptr;       // [T * K]
    const uint8_t* from_logprob = nullptr;  // [T * K]
};

class BeamSearchDecoder {
public:
    explicit BeamSearchDecoder(BeamOptions options);

    const BeamOptions& options() const noexcept { return opt_; }

    // log_probs: [steps, beam_size, vocab_size] float32, where step t starts at
    // log_probs + t * step_stride (step_stride = beam_size * vocab_size when
    // contiguous; rows of one step are always contiguous).
    DecodeResult decode(const float* log_probs, int steps, int vocab_size) const;
    DecodeResult decode_constrained(
        const float* log_probs,
        int steps,
        int vocab_size,
        const DecodeConstraints* constraints,
        int64_t step_stride = 0) const;

    // Decodes straight into caller buffers: the trace (weights and pool outputs
    // must be null) and optional final arrays [K].
    void decode_into(
        const float* log_probs,
        int steps,
        int vocab_size,
        const DecodeConstraints* constraints,
        const TraceOutputs& trace,
        float* final_scores,
        float* final_raw_scores,
        int32_t* final_lengths) const;

    // Runs the search one step at a time, asking `step_fn` for each step's
    // [K, V] rows given the current beams. O(T) model calls.
    DecodeResult decode_model_steps(
        int steps,
        int vocab_size,
        const ModelStepFunction& step_fn,
        const DecodeConstraints* constraints,
        std::vector<float>* rows_buffer) const;

    // Any gradient pointer may be null. Shapes: grad_selected_weights [T*K],
    // grad_relaxed_weights [T*P], grad_final_scores [K], using the result's
    // own T, K and P.
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

// Final-score surrogate gradient of one example, accumulated (+=) into
// grad_log_probs [T, K, V]. Checks that the trace indexes stay in range.
void final_scores_backward_into(const TraceView& trace, const float* grad_final_scores, float* grad_log_probs);

} // namespace dbs
