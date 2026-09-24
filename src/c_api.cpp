// SPDX-License-Identifier: MIT
//
// The exported C ABI (include/dbs.h): opaque handles, error reporting, stats,
// and thin exception-safe wrappers around dbs::BeamSearchDecoder.
#include "dbs.h"

#include "decoder.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct DBSDecoderHandle {
    std::unique_ptr<dbs::BeamSearchDecoder> decoder;
    dbs::BeamOptions options;
    int beam_size = 0;
    mutable std::mutex error_mutex;
    std::string last_error;
    mutable std::mutex stats_mutex;
    DBSStatsC stats{};
    uint64_t deterministic_seed = 0;
};

struct DBSResultHandle {
    dbs::DecodeResult result;
};

struct DBSBackwardHandle {
    dbs::BackwardResult result;
};

struct DBSBatchResultHandle {
    std::vector<DBSResultHandle> results;
};

struct DBSWorkspaceHandle {
    std::mutex mutex;
    std::vector<float> f32;
    std::vector<int32_t> i32;
};

static thread_local std::string g_dbs_last_error;

static void dbs_set_error(DBSDecoderHandle* handle, const std::string& message) {
    g_dbs_last_error = message;
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        handle->last_error = message;
    }
}

static void dbs_clear_error(DBSDecoderHandle* handle) {
    g_dbs_last_error.clear();
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        handle->last_error.clear();
    }
}

static int classify_exception(const std::exception& e) {
    if (dynamic_cast<const std::invalid_argument*>(&e)) return 1;
    if (dynamic_cast<const std::bad_alloc*>(&e)) return 3;
    if (dynamic_cast<const std::overflow_error*>(&e)) return 4;
    if (dynamic_cast<const std::length_error*>(&e)) return 4;
    return 2;
}

static int64_t now_ns_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
}

static void dbs_mark_error(DBSDecoderHandle* handle, int category) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.last_error_category = category;
}

static void dbs_record_decode_stats(DBSDecoderHandle* handle, const dbs::DecodeResult& r, int64_t elapsed_ns, int threads, bool model_step) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
    handle->stats.used_batch_threads = threads;
    handle->stats.used_model_step_callback = model_step ? 1 : 0;
    handle->stats.last_decode_ns = elapsed_ns;
    handle->stats.last_selected_count = static_cast<int64_t>(r.steps) * r.beam_size;
    handle->stats.last_pool_count = static_cast<int64_t>(r.steps) * r.relaxed_pool_size;
    handle->stats.last_logprob_count = static_cast<int64_t>(r.steps) * r.beam_size * r.vocab_size;
    handle->stats.last_allocation_bytes =
        static_cast<int64_t>(r.tokens.size() * sizeof(int32_t) + r.parents.size() * sizeof(int32_t) +
                             r.scores.size() * sizeof(float) + r.raw_scores.size() * sizeof(float) +
                             r.pool_tokens.size() * sizeof(int32_t) + r.pool_scores.size() * sizeof(float) +
                             r.relaxed_weights.size() * sizeof(float));
    handle->stats.last_error_category = 0;
}

static void dbs_record_backward_stats(DBSDecoderHandle* handle, const dbs::BackwardResult& r, int64_t elapsed_ns) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.used_sparse_backward = r.sparse ? 1 : 0;
    handle->stats.used_dense_backward = r.sparse ? 0 : 1;
    handle->stats.last_backward_ns = elapsed_ns;
    handle->stats.last_sparse_grad_count = static_cast<int64_t>(r.sparse_logprob_values.size());
    handle->stats.last_error_category = 0;
}

// Zero-initialised fields select the documented defaults (so `DBSOptionsC opt = {0}`
// is valid); explicit out-of-range values are rejected rather than silently replaced.
static dbs::BeamOptions from_c_options(const DBSOptionsC& c) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::invalid_argument(message);
    };
    auto finite_non_negative = [](float x) { return std::isfinite(x) && x >= 0.0f; };

    require(c.beam_size >= 0, "beam_size cannot be negative");
    require(c.eos_token >= -1, "eos_token must be -1 (disabled) or a token id");
    require(finite_non_negative(c.selected_temperature), "selected_temperature must be finite and non-negative (0 selects the default)");
    require(finite_non_negative(c.soft_topk_temperature), "soft_topk_temperature must be finite and non-negative (0 selects the default)");
    require(c.relaxed_pool_multiplier >= 0, "relaxed_pool_multiplier cannot be negative");
    require(c.vocab_block >= 0, "vocab_block cannot be negative");
    require(finite_non_negative(c.length_penalty_alpha), "length_penalty_alpha must be finite and non-negative");
    require(finite_non_negative(c.soft_topk_tolerance), "soft_topk_tolerance must be finite and non-negative (0 selects the default)");
    require(c.soft_topk_max_iters >= 0, "soft_topk_max_iters cannot be negative");
    require(c.min_length >= 0, "min_length cannot be negative");
    require(c.max_dense_gradient_elements >= 0, "max_dense_gradient_elements cannot be negative");

    const dbs::BeamOptions defaults;
    dbs::BeamOptions o;
    o.beam_size = c.beam_size > 0 ? c.beam_size : defaults.beam_size;
    o.eos_token = c.eos_token;
    o.selected_temperature = c.selected_temperature > 0.0f ? c.selected_temperature : defaults.selected_temperature;
    o.soft_topk_temperature = c.soft_topk_temperature > 0.0f ? c.soft_topk_temperature : defaults.soft_topk_temperature;
    o.relaxed_pool_multiplier = c.relaxed_pool_multiplier > 0 ? c.relaxed_pool_multiplier : defaults.relaxed_pool_multiplier;
    o.vocab_block = c.vocab_block > 0 ? c.vocab_block : defaults.vocab_block;
    o.length_penalty_alpha = c.length_penalty_alpha;
    o.soft_topk_tolerance = c.soft_topk_tolerance > 0.0f ? c.soft_topk_tolerance : defaults.soft_topk_tolerance;
    o.soft_topk_max_iters = c.soft_topk_max_iters > 0 ? c.soft_topk_max_iters : defaults.soft_topk_max_iters;
    o.min_length = c.min_length;
    o.validate_inputs = c.validate_inputs == 0 ? 0 : 1;
    o.max_dense_gradient_elements = c.max_dense_gradient_elements > 0 ? c.max_dense_gradient_elements : defaults.max_dense_gradient_elements;
    return o;
}

extern "C" DBS_EXPORT int dbs_abi_version() {
    return DBS_ABI_VERSION;
}

#define DBS_STRINGIFY_IMPL(x) #x
#define DBS_STRINGIFY(x) DBS_STRINGIFY_IMPL(x)

extern "C" DBS_EXPORT const char* dbs_version_string() {
    return DBS_STRINGIFY(DBS_VERSION_MAJOR) "." DBS_STRINGIFY(DBS_VERSION_MINOR) "." DBS_STRINGIFY(DBS_VERSION_PATCH);
}

extern "C" DBS_EXPORT const char* dbs_last_global_error() {
    return g_dbs_last_error.empty() ? "" : g_dbs_last_error.c_str();
}


extern "C" DBS_EXPORT int dbs_workspace_create(DBSWorkspaceHandle** out_workspace) {
    if (out_workspace) *out_workspace = nullptr;
    if (!out_workspace) {
        dbs_set_error(nullptr, "out_workspace cannot be null");
        return -1;
    }
    try {
        *out_workspace = new DBSWorkspaceHandle;
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(nullptr, e.what());
        return -2;
    }
}

extern "C" DBS_EXPORT void dbs_workspace_destroy(DBSWorkspaceHandle* workspace) {
    delete workspace;
}

extern "C" DBS_EXPORT int dbs_workspace_reserve(DBSWorkspaceHandle* workspace, int64_t float_count, int64_t int_count) {
    if (!workspace || float_count < 0 || int_count < 0) return -1;
    try {
        std::lock_guard<std::mutex> lock(workspace->mutex);
        workspace->f32.reserve(static_cast<size_t>(float_count));
        workspace->i32.reserve(static_cast<size_t>(int_count));
        return 0;
    } catch (...) {
        return -2;
    }
}

extern "C" DBS_EXPORT int64_t dbs_workspace_allocated_bytes(DBSWorkspaceHandle* workspace) {
    if (!workspace) return 0;
    std::lock_guard<std::mutex> lock(workspace->mutex);
    return static_cast<int64_t>(workspace->f32.capacity() * sizeof(float) + workspace->i32.capacity() * sizeof(int32_t));
}

extern "C" DBS_EXPORT int dbs_create_ex(DBSOptionsC options, DBSDecoderHandle** out_handle) {
    if (out_handle) *out_handle = nullptr;
    if (!out_handle) {
        dbs_set_error(nullptr, "out_handle cannot be null");
        return -1;
    }

    try {
        auto h = std::make_unique<DBSDecoderHandle>();
        dbs::BeamOptions parsed = from_c_options(options);
        h->options = parsed;
        h->beam_size = parsed.beam_size;
        h->stats.abi_version = DBS_ABI_VERSION;
        h->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
        h->decoder =
            std::make_unique<dbs::BeamSearchDecoder>(parsed);

        DBSDecoderHandle* raw = h.get();
        *out_handle = h.release();
        dbs_clear_error(raw);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(nullptr, e.what());
        return -2;
    } catch (...) {
        dbs_set_error(nullptr, "unknown exception");
        return -3;
    }
}

extern "C" DBS_EXPORT DBSDecoderHandle* dbs_create(DBSOptionsC options) {
    DBSDecoderHandle* h = nullptr;
    return dbs_create_ex(options, &h) == 0 ? h : nullptr;
}

extern "C" DBS_EXPORT void dbs_destroy(DBSDecoderHandle* handle) {
    delete handle;
}

extern "C" DBS_EXPORT const char* dbs_last_error(DBSDecoderHandle* handle) {
    if (!handle) return dbs_last_global_error();
    // Copy into thread-local storage so the returned pointer stays valid even if
    // another thread records a new error on the same handle.
    static thread_local std::string snapshot;
    {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        snapshot = handle->last_error;
    }
    return snapshot.c_str();
}


static float dbs_fp16_to_float(uint16_t h) noexcept {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 113u;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03ffu;
            out = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static float dbs_bf16_to_float(uint16_t h) noexcept {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static std::vector<float> dbs_convert_to_f32_checked(const void* data, int data_type, int64_t count) {
    if (!data) throw std::invalid_argument("typed log_probs cannot be null");
    if (count < 0) throw std::overflow_error("negative element count");
    std::vector<float> out(static_cast<size_t>(count));
    if (data_type == DBS_DTYPE_F32) {
        const float* p = static_cast<const float*>(data);
        std::copy(p, p + count, out.begin());
    } else if (data_type == DBS_DTYPE_F16) {
        const uint16_t* p = static_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < count; ++i) out[static_cast<size_t>(i)] = dbs_fp16_to_float(p[i]);
    } else if (data_type == DBS_DTYPE_BF16) {
        const uint16_t* p = static_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < count; ++i) out[static_cast<size_t>(i)] = dbs_bf16_to_float(p[i]);
    } else {
        throw std::invalid_argument("unsupported DBS data type");
    }
    return out;
}

extern "C" DBS_EXPORT int dbs_decode(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) return -1;

    try {
        const auto start = std::chrono::steady_clock::now();
        auto r = std::make_unique<DBSResultHandle>();

        r->result = handle->decoder->decode(log_probs, steps, vocab_size);
        dbs_record_decode_stats(handle, r->result, now_ns_since(start), 1, false);

        *out_result = r.release();
        dbs_clear_error(handle);

        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}


extern "C" DBS_EXPORT int dbs_decode_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) return -1;
    try {
        if (steps <= 0 || vocab_size <= 0) throw std::invalid_argument("typed decode dimensions must be positive");
        const int64_t count = static_cast<int64_t>(dbs::checked_mul_size(
            dbs::checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(handle->beam_size), "typed decode size overflow"),
            static_cast<size_t>(vocab_size), "typed decode size overflow"));
        if (data_type == DBS_DTYPE_F32) {
            return dbs_decode(handle, static_cast<const float*>(log_probs), steps, vocab_size, out_result);
        }
        std::vector<float> f32 = dbs_convert_to_f32_checked(log_probs, data_type, count);
        return dbs_decode(handle, f32.data(), steps, vocab_size, out_result);
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_decode_batch_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) return -1;
    try {
        if (batch_size <= 0 || steps <= 0 || vocab_size <= 0) throw std::invalid_argument("typed batch decode dimensions must be positive");
        size_t n = dbs::checked_mul_size(static_cast<size_t>(batch_size), static_cast<size_t>(steps), "typed batch decode size overflow");
        n = dbs::checked_mul_size(n, static_cast<size_t>(handle->beam_size), "typed batch decode size overflow");
        const int64_t count = static_cast<int64_t>(dbs::checked_mul_size(n, static_cast<size_t>(vocab_size), "typed batch decode size overflow"));
        if (data_type == DBS_DTYPE_F32) {
            return dbs_decode_batch(handle, static_cast<const float*>(log_probs), batch_size, steps, vocab_size, num_threads, out_result);
        }
        std::vector<float> f32 = dbs_convert_to_f32_checked(log_probs, data_type, count);
        return dbs_decode_batch(handle, f32.data(), batch_size, steps, vocab_size, num_threads, out_result);
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_decode_constrained(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const uint8_t* banned_tokens,
    const int32_t* forced_tokens,
    int min_length,
    DBSResultHandle** out_result
) {
    DBSAdvancedConstraintsC c{};
    c.banned_tokens = banned_tokens;
    c.forced_tokens = forced_tokens;
    c.min_length = min_length;
    c.repetition_penalty = 1.0f;
    c.no_repeat_ngram_size = 0;
    return dbs_decode_constrained_ex(handle, log_probs, steps, vocab_size, &c, out_result);
}

extern "C" DBS_EXPORT int dbs_decode_constrained_ex(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints_c,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        dbs::DecodeConstraints constraints;
        if (constraints_c) {
            constraints.banned_tokens = constraints_c->banned_tokens;
            constraints.forced_tokens = constraints_c->forced_tokens;
            constraints.min_length = constraints_c->min_length;
            constraints.repetition_penalty = constraints_c->repetition_penalty > 0.0f ? constraints_c->repetition_penalty : 1.0f;
            constraints.no_repeat_ngram_size = constraints_c->no_repeat_ngram_size;
            constraints.token_filter = constraints_c->token_filter;
            constraints.token_filter_user_data = constraints_c->token_filter_user_data;
            constraints.batch_index = constraints_c->batch_index;
        }

        auto r = std::make_unique<DBSResultHandle>();
        r->result = handle->decoder->decode_constrained(log_probs, steps, vocab_size, &constraints);
        dbs_record_decode_stats(handle, r->result, now_ns_since(start_time), 1, false);
        *out_result = r.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}


extern "C" DBS_EXPORT int dbs_decode_model_steps(
    DBSDecoderHandle* handle,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    DBSWorkspaceHandle local_workspace;
    return dbs_decode_model_steps_with_workspace(
        handle,
        &local_workspace,
        step_fn,
        user_data,
        batch_index,
        steps,
        vocab_size,
        out_result);
}

extern "C" DBS_EXPORT int dbs_decode_model_steps_with_workspace(
    DBSDecoderHandle* handle,
    DBSWorkspaceHandle* workspace,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !workspace || !step_fn || !out_result) {
        dbs_set_error(handle, "invalid decoder, workspace, model step callback, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }
    if (steps <= 0 || vocab_size <= 0) {
        dbs_set_error(handle, "steps and vocab_size must be positive");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        const int K = handle->beam_size;
        const size_t row_count = static_cast<size_t>(K) * static_cast<size_t>(vocab_size);
        const size_t total = static_cast<size_t>(steps) * row_count;

        std::lock_guard<std::mutex> workspace_lock(workspace->mutex);
        workspace->f32.assign(total, -std::numeric_limits<float>::infinity());
        workspace->i32.assign(static_cast<size_t>(K), -1);
        std::vector<float> prev_scores(static_cast<size_t>(K), 0.0f);

        for (int t = 0; t < steps; ++t) {
            float* out_row = workspace->f32.data() + static_cast<size_t>(t) * row_count;
            const int rc = step_fn(
                user_data,
                batch_index,
                t,
                workspace->i32.data(),
                prev_scores.data(),
                K,
                vocab_size,
                out_row);
            if (rc != 0) {
                throw std::runtime_error("model step callback returned non-zero status");
            }

            if (t + 1 < steps) {
                dbs::DecodeResult partial = handle->decoder->decode(workspace->f32.data(), t + 1, vocab_size);
                const int32_t* tok = partial.tokens.data() + static_cast<size_t>(t) * K;
                const float* scores = partial.final_scores.data();
                for (int k = 0; k < K; ++k) {
                    workspace->i32[static_cast<size_t>(k)] = tok[k];
                    prev_scores[static_cast<size_t>(k)] = scores[k];
                }
            }
        }

        auto r = std::make_unique<DBSResultHandle>();
        r->result = handle->decoder->decode(workspace->f32.data(), steps, vocab_size);
        dbs_record_decode_stats(handle, r->result, now_ns_since(start_time), 1, true);
        {
            const int64_t workspace_bytes = static_cast<int64_t>(
                workspace->f32.capacity() * sizeof(float) + workspace->i32.capacity() * sizeof(int32_t));
            std::lock_guard<std::mutex> lock(handle->stats_mutex);
            handle->stats.last_allocation_bytes += workspace_bytes;
        }
        *out_result = r.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_decode_batch(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }
    if (!log_probs || batch_size <= 0 || steps <= 0 || vocab_size <= 0) {
        dbs_set_error(handle, "invalid batch decode arguments");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));

        const size_t step_beam = dbs::checked_mul_size(
            static_cast<size_t>(steps),
            static_cast<size_t>(handle->beam_size),
            "batch stride overflow");
        const size_t stride = dbs::checked_mul_size(
            step_beam,
            static_cast<size_t>(vocab_size),
            "batch stride overflow");

        const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        const int threads = std::max(1, std::min(batch_size, num_threads > 0 ? num_threads : std::max(1, hw_threads)));
        std::atomic<int> index{0};
        std::mutex error_mutex;
        std::exception_ptr first_exception = nullptr;

        // Worker threads inherit the calling thread's kernel override.
        const dbs::KernelOverride captured_override = dbs::current_kernel_override();

        auto worker = [&, captured_override]() {
            dbs::set_kernel_override(captured_override);
            for (;;) {
                const int b = index.fetch_add(1);
                if (b >= batch_size) break;
                try {
                    br->results[static_cast<size_t>(b)].result =
                        handle->decoder->decode(log_probs + static_cast<size_t>(b) * stride, steps, vocab_size);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_exception) first_exception = std::current_exception();
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        if (!br->results.empty()) {
            dbs_record_decode_stats(handle, br->results.front().result, now_ns_since(start_time), threads, false);
        }
        *out_result = br.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}


extern "C" DBS_EXPORT int dbs_decode_batch_variable(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int max_steps,
    int max_beam_size,
    int vocab_size,
    const int32_t* steps_per_example,
    const int32_t* beam_sizes_per_example,
    const int32_t* eos_tokens_per_example,
    const int32_t* min_lengths_per_example,
    const uint8_t* banned_tokens_per_example,
    const int32_t* forced_tokens_per_example,
    int num_threads,
    DBSBatchResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !log_probs || !out_result || batch_size <= 0 || max_steps <= 0 || max_beam_size <= 0 || vocab_size <= 0) {
        dbs_set_error(handle, "invalid variable batch decode arguments");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));
        const size_t input_stride = dbs::checked_mul_size(
            dbs::checked_mul_size(static_cast<size_t>(max_steps), static_cast<size_t>(max_beam_size), "variable batch stride overflow"),
            static_cast<size_t>(vocab_size), "variable batch stride overflow");
        (void)dbs::checked_mul_size(input_stride, static_cast<size_t>(batch_size), "variable batch size overflow");

        const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        const int threads = std::max(1, std::min(batch_size, num_threads > 0 ? num_threads : std::max(1, hw_threads)));
        std::atomic<int> index{0};
        std::mutex error_mutex;
        std::exception_ptr first_exception = nullptr;

        // Worker threads inherit the calling thread's kernel override.
        const dbs::KernelOverride captured_override = dbs::current_kernel_override();

        auto worker = [&, captured_override]() {
            dbs::set_kernel_override(captured_override);
            for (;;) {
                const int b = index.fetch_add(1);
                if (b >= batch_size) break;
                try {
                    const int steps = steps_per_example ? steps_per_example[b] : max_steps;
                    const int beam = beam_sizes_per_example ? beam_sizes_per_example[b] : handle->beam_size;
                    if (steps <= 0 || steps > max_steps) throw std::invalid_argument("invalid per-example steps");
                    if (beam <= 0 || beam > max_beam_size) throw std::invalid_argument("invalid per-example beam size");

                    dbs::BeamOptions opt = handle->options;
                    opt.beam_size = beam;
                    if (eos_tokens_per_example) opt.eos_token = eos_tokens_per_example[b];
                    if (min_lengths_per_example) opt.min_length = std::max(0, min_lengths_per_example[b]);
                    dbs::BeamSearchDecoder local_decoder(opt);

                    std::vector<float> local(static_cast<size_t>(steps) * static_cast<size_t>(beam) * static_cast<size_t>(vocab_size));
                    const float* base = log_probs + static_cast<size_t>(b) * input_stride;
                    for (int t = 0; t < steps; ++t) {
                        for (int k = 0; k < beam; ++k) {
                            const float* src = base + (static_cast<size_t>(t) * max_beam_size + static_cast<size_t>(k)) * static_cast<size_t>(vocab_size);
                            float* dst = local.data() + (static_cast<size_t>(t) * beam + static_cast<size_t>(k)) * static_cast<size_t>(vocab_size);
                            std::copy(src, src + vocab_size, dst);
                        }
                    }

                    dbs::DecodeConstraints constraints;
                    if (banned_tokens_per_example) constraints.banned_tokens = banned_tokens_per_example + static_cast<size_t>(b) * static_cast<size_t>(vocab_size);
                    if (forced_tokens_per_example) constraints.forced_tokens = forced_tokens_per_example + static_cast<size_t>(b) * static_cast<size_t>(max_steps);
                    constraints.min_length = min_lengths_per_example ? min_lengths_per_example[b] : -1;
                    constraints.batch_index = b;

                    br->results[static_cast<size_t>(b)].result =
                        local_decoder.decode_constrained(local.data(), steps, vocab_size,
                            (banned_tokens_per_example || forced_tokens_per_example || min_lengths_per_example) ? &constraints : nullptr);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_exception) first_exception = std::current_exception();
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        if (!br->results.empty()) dbs_record_decode_stats(handle, br->results.front().result, now_ns_since(start_time), threads, false);
        *out_result = br.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_backward(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    return dbs_backward_sparse(
        handle,
        result,
        grad_selected_weights,
        grad_relaxed_weights,
        grad_final_scores,
        out_backward
    );
}

extern "C" DBS_EXPORT int dbs_backward_dense(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    if (out_backward) *out_backward = nullptr;
    if (!handle || !handle->decoder || !result || !out_backward) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto b = std::make_unique<DBSBackwardHandle>();
        b->result = handle->decoder->backward(
            result->result,
            grad_selected_weights,
            grad_relaxed_weights,
            grad_final_scores);
        dbs_record_backward_stats(handle, b->result, now_ns_since(start_time));
        *out_backward = b.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_backward_sparse(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    if (out_backward) *out_backward = nullptr;
    if (!handle || !handle->decoder || !result || !out_backward) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto b = std::make_unique<DBSBackwardHandle>();
        b->result = handle->decoder->backward_sparse(
            result->result,
            grad_selected_weights,
            grad_relaxed_weights,
            grad_final_scores);
        dbs_record_backward_stats(handle, b->result, now_ns_since(start_time));
        *out_backward = b.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_backward_default(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    return dbs_backward_sparse(
        handle,
        result,
        grad_selected_weights,
        grad_relaxed_weights,
        grad_final_scores,
        out_backward
    );
}

extern "C" DBS_EXPORT void dbs_free_result(DBSResultHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT void dbs_free_batch_result(DBSBatchResultHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT int dbs_batch_result_size(const DBSBatchResultHandle* result) {
    return result ? static_cast<int>(result->results.size()) : 0;
}

extern "C" DBS_EXPORT const DBSResultHandle* dbs_batch_result_at(const DBSBatchResultHandle* result, int batch_index) {
    if (!result || batch_index < 0 || batch_index >= static_cast<int>(result->results.size())) return nullptr;
    return &result->results[static_cast<size_t>(batch_index)];
}

extern "C" DBS_EXPORT void dbs_free_backward(DBSBackwardHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT int dbs_result_steps(const DBSResultHandle* result) {
    return result ? result->result.steps : 0;
}

extern "C" DBS_EXPORT int dbs_result_beam_size(const DBSResultHandle* result) {
    return result ? result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int dbs_result_vocab_size(const DBSResultHandle* result) {
    return result ? result->result.vocab_size : 0;
}

extern "C" DBS_EXPORT int dbs_result_pool_size(const DBSResultHandle* result) {
    return result ? result->result.relaxed_pool_size : 0;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_tokens(const DBSResultHandle* result) {
    return result ? result->result.tokens.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_parents(const DBSResultHandle* result) {
    return result ? result->result.parents.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_scores(const DBSResultHandle* result) {
    return result ? result->result.scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_raw_scores(const DBSResultHandle* result) {
    return result ? result->result.raw_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_relaxed_weights(const DBSResultHandle* result) {
    return result ? result->result.relaxed_weights.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_tokens(const DBSResultHandle* result) {
    return result ? result->result.pool_tokens.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_parents(const DBSResultHandle* result) {
    return result ? result->result.pool_parents.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_pool_scores(const DBSResultHandle* result) {
    return result ? result->result.pool_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_grad_log_probs(const DBSBackwardHandle* result) {
    return result ? result->result.grad_log_probs.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_grad_initial_scores(const DBSBackwardHandle* result) {
    return result ? result->result.grad_initial_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const int64_t* dbs_backward_sparse_logprob_indices(const DBSBackwardHandle* result) {
    return result ? result->result.sparse_logprob_indices.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_sparse_logprob_values(const DBSBackwardHandle* result) {
    return result ? result->result.sparse_logprob_values.data() : nullptr;
}

extern "C" DBS_EXPORT int64_t dbs_backward_sparse_logprob_count(const DBSBackwardHandle* result) {
    return result ? static_cast<int64_t>(result->result.sparse_logprob_values.size()) : 0;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_lengths(const DBSResultHandle* result) {
    return result ? result->result.lengths.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_lengths(const DBSResultHandle* result) {
    return result ? result->result.pool_lengths.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_weights(const DBSResultHandle* result) {
    return result ? result->result.weights.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_final_scores(const DBSResultHandle* result) {
    return result ? result->result.final_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_final_raw_scores(const DBSResultHandle* result) {
    return result ? result->result.final_raw_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_pool_raw_scores(const DBSResultHandle* result) {
    return result ? result->result.pool_raw_scores.data() : nullptr;
}

extern "C" DBS_EXPORT int64_t dbs_result_selected_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_pool_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.relaxed_pool_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_logprob_count(const DBSResultHandle* result) {
    return result
        ? static_cast<int64_t>(result->result.steps) * result->result.beam_size * result->result.vocab_size
        : 0;
}

extern "C" DBS_EXPORT int64_t dbs_backward_grad_log_probs_count(const DBSResultHandle* result) {
    return dbs_result_logprob_count(result);
}

extern "C" DBS_EXPORT int64_t dbs_backward_grad_initial_scores_count(const DBSResultHandle* result) {
    return result ? result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int dbs_backward_is_sparse(const DBSBackwardHandle* result) {
    return result && result->result.sparse ? 1 : 0;
}


extern "C" DBS_EXPORT int64_t dbs_result_eos_count(const DBSResultHandle* result, int eos_token) {
    if (!result || eos_token < 0) return 0;
    int64_t count = 0;
    for (int32_t token : result->result.tokens) {
        if (token == eos_token) ++count;
    }
    return count;
}

extern "C" DBS_EXPORT int dbs_result_validate_deterministic_order(const DBSResultHandle* result) {
    if (!result) return -1;
    const int T = result->result.steps;
    const int K = result->result.beam_size;
    if (T < 0 || K < 0) return -2;
    for (int t = 0; t < T; ++t) {
        for (int k = 1; k < K; ++k) {
            const size_t prev = static_cast<size_t>(t) * K + (k - 1);
            const size_t cur = static_cast<size_t>(t) * K + k;
            const float a = result->result.scores[prev];
            const float b = result->result.scores[cur];
            if (std::isfinite(a) && std::isfinite(b) && b > a) return -3;
            if (a == b) {
                const int32_t pa = result->result.parents[prev];
                const int32_t pb = result->result.parents[cur];
                const int32_t ta = result->result.tokens[prev];
                const int32_t tb = result->result.tokens[cur];
                if (pb < pa || (pb == pa && tb < ta)) return -4;
            }
        }
    }
    return 0;
}

extern "C" DBS_EXPORT int dbs_result_summary_json(const DBSResultHandle* result, int eos_token, char* out_json, int64_t out_json_capacity) {
    if (!result || !out_json || out_json_capacity <= 0) return -1;
    const auto& r = result->result;
    int min_len = r.lengths.empty() ? 0 : std::numeric_limits<int>::max();
    int max_len = 0;
    for (int32_t len : r.lengths) {
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
    }
    if (r.lengths.empty()) min_len = 0;
    float min_final = r.final_scores.empty() ? 0.0f : r.final_scores[0];
    float max_final = r.final_scores.empty() ? 0.0f : r.final_scores[0];
    for (float score : r.final_scores) {
        if (score < min_final) min_final = score;
        if (score > max_final) max_final = score;
    }
    const int64_t eos_count = dbs_result_eos_count(result, eos_token);
    const int order_ok = dbs_result_validate_deterministic_order(result) == 0 ? 1 : 0;
    const int n = std::snprintf(out_json, static_cast<size_t>(out_json_capacity),
        "{\"abi_version\":%d,\"steps\":%d,\"beam_size\":%d,\"vocab_size\":%d,\"selected_count\":%lld,\"pool_count\":%lld,\"eos_token\":%d,\"eos_count\":%lld,\"min_length\":%d,\"max_length\":%d,\"min_final_score\":%.9g,\"max_final_score\":%.9g,\"deterministic_order\":%d}",
        DBS_ABI_VERSION,
        r.steps,
        r.beam_size,
        r.vocab_size,
        static_cast<long long>(static_cast<int64_t>(r.steps) * r.beam_size),
        static_cast<long long>(static_cast<int64_t>(r.steps) * r.relaxed_pool_size),
        eos_token,
        static_cast<long long>(eos_count),
        min_len,
        max_len,
        static_cast<double>(min_final),
        static_cast<double>(max_final),
        order_ok);
    if (n < 0) return -2;
    return n < out_json_capacity ? 0 : 1;
}

extern "C" DBS_EXPORT int dbs_validate_production_gate_manifest(const char* manifest_json, char* out_error, int64_t out_error_capacity) {
    (void)manifest_json;
    if (out_error && out_error_capacity > 0) {
        std::snprintf(
            out_error,
            static_cast<size_t>(out_error_capacity),
            "%s",
            "deprecated: release evidence cannot be validated by the C ABI; run tests and attach raw benchmark/artifact logs");
    }
    return -1;
}

extern "C" DBS_EXPORT int dbs_has_avx512() {
    return dbs::runtime_has_avx512() ? 1 : 0;
}

extern "C" DBS_EXPORT int dbs_has_avx2() {
    return dbs::runtime_has_avx2() ? 1 : 0;
}

extern "C" DBS_EXPORT int dbs_has_sse42() {
    return dbs::runtime_has_sse42() ? 1 : 0;
}

extern "C" DBS_EXPORT int dbs_has_neon() {
    return dbs::runtime_has_neon() ? 1 : 0;
}

extern "C" DBS_EXPORT const char* dbs_selected_kernel_name() {
    return dbs::kernel_path_name(dbs::selected_kernel_path());
}

extern "C" DBS_EXPORT int dbs_get_stats(DBSDecoderHandle* handle, DBSStatsC* out_stats) {
    if (!handle || !out_stats) return -1;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    *out_stats = handle->stats;
    out_stats->total_allocator_calls = dbs::g_allocator_calls.load(std::memory_order_relaxed);
    out_stats->total_allocator_bytes = dbs::g_allocator_bytes.load(std::memory_order_relaxed);
    return 0;
}


extern "C" DBS_EXPORT int dbs_get_stats_json(DBSDecoderHandle* handle, char* out_json, int64_t out_json_capacity) {
    if (!handle || !out_json || out_json_capacity <= 0) return -1;
    DBSStatsC s{};
    if (dbs_get_stats(handle, &s) != 0) return -1;
    const int n = std::snprintf(
        out_json,
        static_cast<size_t>(out_json_capacity),
        "{\"abi_version\":%d,\"kernel\":\"%s\",\"used_sparse_backward\":%d,\"used_dense_backward\":%d,\"batch_threads\":%d,\"model_step\":%d,\"last_error_category\":%d,\"decode_ns\":%lld,\"backward_ns\":%lld,\"allocation_bytes\":%lld,\"selected_count\":%lld,\"pool_count\":%lld,\"logprob_count\":%lld,\"sparse_grad_count\":%lld,\"allocator_calls\":%lld,\"allocator_bytes\":%lld}",
        s.abi_version,
        dbs::kernel_path_name(static_cast<dbs::KernelPath>(s.last_kernel)),
        s.used_sparse_backward,
        s.used_dense_backward,
        s.used_batch_threads,
        s.used_model_step_callback,
        s.last_error_category,
        static_cast<long long>(s.last_decode_ns),
        static_cast<long long>(s.last_backward_ns),
        static_cast<long long>(s.last_allocation_bytes),
        static_cast<long long>(s.last_selected_count),
        static_cast<long long>(s.last_pool_count),
        static_cast<long long>(s.last_logprob_count),
        static_cast<long long>(s.last_sparse_grad_count),
        static_cast<long long>(s.total_allocator_calls),
        static_cast<long long>(s.total_allocator_bytes));
    if (n < 0) return -2;
    return n < out_json_capacity ? 0 : 1;
}

extern "C" DBS_EXPORT int dbs_is_deterministic() {
    // Reports the implementation contract: hard decode uses deterministic
    // score/raw-score/parent/token/length/from-logprob ordering and no RNG.
    // This is not a runtime proof over arbitrary caller-provided inputs.
    return 1;
}

extern "C" DBS_EXPORT int dbs_set_deterministic_seed(DBSDecoderHandle* handle, uint64_t seed) {
    if (!handle) return -1;
    handle->deterministic_seed = seed;
    return 0;
}

extern "C" DBS_EXPORT uint64_t dbs_get_deterministic_seed(DBSDecoderHandle* handle) {
    return handle ? handle->deterministic_seed : 0;
}

extern "C" DBS_EXPORT void dbs_allocator_counters_reset() {
    dbs::g_allocator_calls.store(0, std::memory_order_relaxed);
    dbs::g_allocator_bytes.store(0, std::memory_order_relaxed);
}

extern "C" DBS_EXPORT int64_t dbs_allocator_call_count() {
    return dbs::g_allocator_calls.load(std::memory_order_relaxed);
}

extern "C" DBS_EXPORT int64_t dbs_allocator_byte_count() {
    return dbs::g_allocator_bytes.load(std::memory_order_relaxed);
}

extern "C" DBS_EXPORT void dbs_reset_stats(DBSDecoderHandle* handle) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats = DBSStatsC{};
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
}
