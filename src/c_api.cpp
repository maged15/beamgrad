// SPDX-License-Identifier: MIT
//
// The exported C ABI (include/dbs.h): opaque handles, error reporting, stats,
// and exception-safe wrappers around dbs::BeamSearchDecoder.
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
    mutable std::mutex error_mutex;
    std::string last_error;
    mutable std::mutex stats_mutex;
    DBSStatsC stats{};
    std::atomic<uint64_t> deterministic_seed{0};
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

namespace {

thread_local std::string g_last_error;

void set_error(DBSDecoderHandle* handle, const std::string& message) {
    g_last_error = message;
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        handle->last_error = message;
    }
}

void clear_error(DBSDecoderHandle* handle) {
    g_last_error.clear();
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        handle->last_error.clear();
    }
}

// DBSStatsC::last_error_category: 1 invalid argument, 2 runtime, 3 allocation, 4 overflow.
void mark_error(DBSDecoderHandle* handle, int category) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.last_error_category = category;
}

int fail(DBSDecoderHandle* handle, const std::string& message, int category, int status) {
    set_error(handle, message);
    mark_error(handle, category);
    return status;
}

// Runs `body`, translating exceptions into the documented status codes. The
// handle may be null (the error is then only recorded per thread).
template <class Body>
int guarded(DBSDecoderHandle* handle, Body&& body) {
    try {
        body();
        clear_error(handle);
        return DBS_OK;
    } catch (const std::invalid_argument& e) {
        return fail(handle, e.what(), 1, DBS_ERROR_INVALID_ARGUMENT);
    } catch (const std::length_error& e) {
        return fail(handle, e.what(), 4, DBS_ERROR_INVALID_ARGUMENT);
    } catch (const std::overflow_error& e) {
        return fail(handle, e.what(), 4, DBS_ERROR_INVALID_ARGUMENT);
    } catch (const std::bad_alloc&) {
        return fail(handle, "out of memory", 3, DBS_ERROR_RUNTIME);
    } catch (const std::exception& e) {
        return fail(handle, e.what(), 2, DBS_ERROR_RUNTIME);
    } catch (...) {
        return fail(handle, "unknown exception", 2, DBS_ERROR_UNKNOWN);
    }
}

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

const dbs::BeamSearchDecoder& decoder_of(DBSDecoderHandle* handle) {
    require(handle != nullptr && handle->decoder != nullptr, "decoder handle cannot be null");
    return *handle->decoder;
}

int64_t now_ns_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

void record_decode_stats(DBSDecoderHandle* handle, const dbs::DecodeResult& r, int64_t elapsed_ns, int threads, bool model_step) {
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
    handle->stats.used_batch_threads = threads;
    handle->stats.used_model_step_callback = model_step ? 1 : 0;
    handle->stats.last_decode_ns = elapsed_ns;
    handle->stats.last_selected_count = static_cast<int64_t>(r.steps) * r.beam_size;
    handle->stats.last_pool_count = static_cast<int64_t>(r.steps) * r.relaxed_pool_size;
    handle->stats.last_logprob_count = static_cast<int64_t>(r.steps) * r.beam_size * r.vocab_size;
    handle->stats.last_allocation_bytes = static_cast<int64_t>(
        r.tokens.size() * sizeof(int32_t) + r.parents.size() * sizeof(int32_t) + r.lengths.size() * sizeof(int32_t) +
        r.scores.size() * sizeof(float) + r.raw_scores.size() * sizeof(float) + r.weights.size() * sizeof(float) +
        r.pool_tokens.size() * sizeof(int32_t) + r.pool_parents.size() * sizeof(int32_t) +
        r.pool_lengths.size() * sizeof(int32_t) + r.pool_scores.size() * sizeof(float) +
        r.pool_raw_scores.size() * sizeof(float) + r.relaxed_weights.size() * sizeof(float));
    handle->stats.last_error_category = 0;
}

void record_batch_stats(DBSDecoderHandle* handle, int64_t selected, int64_t logprobs, int64_t elapsed_ns, int threads) {
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
    handle->stats.used_batch_threads = threads;
    handle->stats.used_model_step_callback = 0;
    handle->stats.last_decode_ns = elapsed_ns;
    handle->stats.last_selected_count = selected;
    handle->stats.last_pool_count = 0;
    handle->stats.last_logprob_count = logprobs;
    handle->stats.last_allocation_bytes = 0;
    handle->stats.last_error_category = 0;
}

void record_backward_stats(DBSDecoderHandle* handle, const dbs::BackwardResult& r, int64_t elapsed_ns) {
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
dbs::BeamOptions from_c_options(const DBSOptionsC& c) {
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
    o.relaxed_pool_multiplier = c.relaxed_pool_multiplier;
    o.length_penalty_alpha = c.length_penalty_alpha;
    o.soft_topk_tolerance = c.soft_topk_tolerance > 0.0f ? c.soft_topk_tolerance : defaults.soft_topk_tolerance;
    o.soft_topk_max_iters = c.soft_topk_max_iters > 0 ? c.soft_topk_max_iters : defaults.soft_topk_max_iters;
    o.min_length = c.min_length;
    o.validate_inputs = c.validate_inputs == 0 ? 0 : 1;
    o.max_dense_gradient_elements = c.max_dense_gradient_elements > 0 ? c.max_dense_gradient_elements : defaults.max_dense_gradient_elements;
    return o;
}

dbs::DecodeConstraints from_c_constraints(const DBSAdvancedConstraintsC& c) {
    dbs::DecodeConstraints out;
    out.banned_tokens = c.banned_tokens;
    out.forced_tokens = c.forced_tokens;
    out.min_length = c.min_length;
    // Zero-initialised structs leave the penalty at 0, which means "off".
    out.repetition_penalty = c.repetition_penalty > 0.0f ? c.repetition_penalty : 1.0f;
    out.no_repeat_ngram_size = c.no_repeat_ngram_size;
    out.token_filter = c.token_filter;
    out.token_filter_user_data = c.token_filter_user_data;
    out.batch_index = c.batch_index;
    return out;
}

// A C data type as the decoder's element type.
dbs::DType dtype_of(int data_type) {
    require(data_type == DBS_DTYPE_F32 || data_type == DBS_DTYPE_F16 || data_type == DBS_DTYPE_BF16,
            "unsupported DBS data type");
    return static_cast<dbs::DType>(data_type);
}

// Below this much work, starting threads costs more than it saves. Work is the
// candidates a decode scans (B * T * K * V) or the trace entries a backward
// walks (B * T * K). On an 8-thread x86-64 machine one thread is faster up to
// about 256k candidates and slower from about 1M (`dbs_bench batch-threads`).
constexpr size_t kMinThreadedWork = size_t{1} << 19;

// Threads for `items` independent examples: an explicit request, capped at the
// number of examples; or, for requested <= 0, one per hardware thread, except
// a single one when the work is small.
int thread_count(int requested, int items, size_t work) {
    if (requested <= 0 && work < kMinThreadedWork) return 1;
    const int hardware = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    return std::max(1, std::min(items, requested > 0 ? requested : hardware));
}

// Runs fn(i) for i in [0, n) on `threads` threads, the calling thread included.
// Workers inherit the caller's kernel override. The first exception is rethrown.
template <class Fn>
void parallel_for(int n, int threads, Fn&& fn) {
    if (threads <= 1 || n <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<int> next{0};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    const dbs::KernelOverride kernel_override = dbs::current_kernel_override();
    auto work = [&]() {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= n) break;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
        pool.emplace_back([&, kernel_override]() {
            dbs::set_kernel_override(kernel_override);
            work();
        });
    }
    work();
    for (auto& th : pool) th.join();
    if (first_error) std::rethrow_exception(first_error);
}

// Prefixes an input error with the example it came from.
[[noreturn]] void rethrow_for_example(int b, const std::invalid_argument& e) {
    throw std::invalid_argument("example " + std::to_string(b) + ": " + e.what());
}

std::vector<int32_t> steps_for(const int32_t* steps_per_example, int batch_size, int steps) {
    std::vector<int32_t> out(static_cast<size_t>(batch_size), steps);
    if (!steps_per_example) return out;
    for (int b = 0; b < batch_size; ++b) {
        if (steps_per_example[b] < 1 || steps_per_example[b] > steps) {
            throw std::invalid_argument("steps_per_example[" + std::to_string(b) + "] must be in [1, steps]");
        }
        out[static_cast<size_t>(b)] = steps_per_example[b];
    }
    return out;
}

void decode_model_steps_impl(
    DBSDecoderHandle* handle,
    std::vector<float>* rows,
    const dbs::ModelStepFunction& fn,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints_c,
    DBSResultHandle** out_result) {
    require(out_result != nullptr, "out_result cannot be null");
    *out_result = nullptr;
    const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
    const auto start = std::chrono::steady_clock::now();
    dbs::DecodeConstraints constraints;
    if (constraints_c) constraints = from_c_constraints(*constraints_c);
    auto r = std::make_unique<DBSResultHandle>();
    r->result = decoder.decode_model_steps(steps, vocab_size, fn, constraints_c ? &constraints : nullptr, rows);
    record_decode_stats(handle, r->result, now_ns_since(start), 1, true);
    {
        std::lock_guard<std::mutex> lock(handle->stats_mutex);
        handle->stats.last_allocation_bytes += static_cast<int64_t>(rows->capacity() * sizeof(float));
    }
    *out_result = r.release();
}

template <class T, class A>
const T* data_or_null(const std::vector<T, A>& v) {
    return v.empty() ? nullptr : v.data();
}

} // namespace

extern "C" DBS_EXPORT int dbs_abi_version() {
    return DBS_ABI_VERSION;
}

#define DBS_STRINGIFY_IMPL(x) #x
#define DBS_STRINGIFY(x) DBS_STRINGIFY_IMPL(x)

extern "C" DBS_EXPORT const char* dbs_version_string() {
    return DBS_STRINGIFY(DBS_VERSION_MAJOR) "." DBS_STRINGIFY(DBS_VERSION_MINOR) "." DBS_STRINGIFY(DBS_VERSION_PATCH);
}

extern "C" DBS_EXPORT const char* dbs_last_global_error() {
    return g_last_error.c_str();
}

extern "C" DBS_EXPORT int dbs_workspace_create(DBSWorkspaceHandle** out_workspace) {
    if (out_workspace) *out_workspace = nullptr;
    return guarded(nullptr, [&] {
        require(out_workspace != nullptr, "out_workspace cannot be null");
        *out_workspace = new DBSWorkspaceHandle;
    });
}

extern "C" DBS_EXPORT void dbs_workspace_destroy(DBSWorkspaceHandle* workspace) {
    delete workspace;
}

extern "C" DBS_EXPORT int dbs_workspace_reserve(DBSWorkspaceHandle* workspace, int64_t float_count, int64_t int_count) {
    return guarded(nullptr, [&] {
        require(workspace != nullptr, "workspace cannot be null");
        require(float_count >= 0 && int_count >= 0, "reserve counts cannot be negative");
        std::lock_guard<std::mutex> lock(workspace->mutex);
        workspace->f32.reserve(static_cast<size_t>(float_count));
        workspace->i32.reserve(static_cast<size_t>(int_count));
    });
}

extern "C" DBS_EXPORT int64_t dbs_workspace_allocated_bytes(DBSWorkspaceHandle* workspace) {
    if (!workspace) return 0;
    std::lock_guard<std::mutex> lock(workspace->mutex);
    return static_cast<int64_t>(workspace->f32.capacity() * sizeof(float) + workspace->i32.capacity() * sizeof(int32_t));
}

extern "C" DBS_EXPORT int dbs_create_ex(DBSOptionsC options, DBSDecoderHandle** out_handle) {
    if (out_handle) *out_handle = nullptr;
    return guarded(nullptr, [&] {
        require(out_handle != nullptr, "out_handle cannot be null");
        auto h = std::make_unique<DBSDecoderHandle>();
        h->options = from_c_options(options);
        h->decoder = std::make_unique<dbs::BeamSearchDecoder>(h->options);
        h->stats.abi_version = DBS_ABI_VERSION;
        h->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
        *out_handle = h.release();
    });
}

extern "C" DBS_EXPORT int dbs_set_extra_eos_tokens(DBSDecoderHandle* handle, const int32_t* tokens, int count) {
    return guarded(handle, [&] {
        decoder_of(handle);
        require(count >= 0 && (count == 0 || tokens != nullptr), "tokens cannot be null when count > 0");
        dbs::BeamOptions options = handle->options;
        options.extra_eos_tokens.assign(tokens, tokens + count);
        // Validated by the decoder (token ids, and eos_token >= 0 when there are any).
        auto decoder = std::make_unique<dbs::BeamSearchDecoder>(options);
        handle->options = std::move(options);
        handle->decoder = std::move(decoder);
    });
}

extern "C" DBS_EXPORT DBSDecoderHandle* dbs_create(DBSOptionsC options) {
    DBSDecoderHandle* h = nullptr;
    return dbs_create_ex(options, &h) == DBS_OK ? h : nullptr;
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

extern "C" DBS_EXPORT int dbs_decode(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    return dbs_decode_constrained_ex(handle, log_probs, steps, vocab_size, nullptr, out_result);
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
    return guarded(handle, [&] {
        require(out_result != nullptr, "out_result cannot be null");
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        const dbs::DType type = dtype_of(data_type);
        const auto start = std::chrono::steady_clock::now();
        auto r = std::make_unique<DBSResultHandle>();
        // 16-bit rows are converted one at a time, for the beams that are expanded.
        r->result = decoder.decode_typed(log_probs, type, steps, vocab_size, nullptr);
        record_decode_stats(handle, r->result, now_ns_since(start), 1, false);
        *out_result = r.release();
    });
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
    return guarded(handle, [&] {
        require(out_result != nullptr, "out_result cannot be null");
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        const dbs::DType type = dtype_of(data_type);
        require(log_probs != nullptr, "log_probs cannot be null");
        require(batch_size > 0 && steps > 0 && vocab_size > 0, "batch_size, steps and vocab_size must be positive");
        const auto start = std::chrono::steady_clock::now();
        const size_t stride = dbs::checked_mul_size(
            dbs::checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(decoder.options().beam_size), "batch stride overflow"),
            static_cast<size_t>(vocab_size), "batch stride overflow");
        const size_t work = dbs::checked_mul_size(stride, static_cast<size_t>(batch_size), "batch size overflow");
        const char* base = static_cast<const char*>(log_probs);
        const size_t stride_bytes = dbs::checked_mul_size(stride, dbs::dtype_size(type), "batch stride overflow");

        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));
        const int threads = thread_count(num_threads, batch_size, work);
        parallel_for(batch_size, threads, [&](int b) {
            try {
                br->results[static_cast<size_t>(b)].result =
                    decoder.decode_typed(base + static_cast<size_t>(b) * stride_bytes, type, steps, vocab_size, nullptr);
            } catch (const std::invalid_argument& e) {
                rethrow_for_example(b, e);
            }
        });
        record_decode_stats(handle, br->results.front().result, now_ns_since(start), threads, false);
        *out_result = br.release();
    });
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
    return guarded(handle, [&] {
        require(out_result != nullptr, "out_result cannot be null");
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        const auto start = std::chrono::steady_clock::now();
        dbs::DecodeConstraints constraints;
        if (constraints_c) constraints = from_c_constraints(*constraints_c);
        auto r = std::make_unique<DBSResultHandle>();
        r->result = decoder.decode_constrained(log_probs, steps, vocab_size, constraints_c ? &constraints : nullptr);
        record_decode_stats(handle, r->result, now_ns_since(start), 1, false);
        *out_result = r.release();
    });
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
    return dbs_decode_model_steps_with_workspace(handle, &local_workspace, step_fn, user_data, batch_index, steps,
                                                 vocab_size, out_result);
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
    return guarded(handle, [&] {
        require(workspace != nullptr, "workspace cannot be null");
        require(step_fn != nullptr, "model step callback cannot be null");
        std::lock_guard<std::mutex> lock(workspace->mutex);
        std::vector<float> zeros;
        auto fn = [&](const dbs::ModelStepInfo& info, float* rows) {
            // This callback only sees the previous tokens and scores, with
            // scores of 0 at the first step.
            const float* scores = info.scores;
            if (info.step == 0) {
                zeros.assign(static_cast<size_t>(info.beam_size), 0.0f);
                scores = zeros.data();
            }
            if (step_fn(user_data, batch_index, info.step, info.tokens, scores, info.beam_size, info.vocab_size, rows) != 0) {
                throw std::runtime_error("model step callback returned non-zero status");
            }
        };
        decode_model_steps_impl(handle, &workspace->f32, fn, steps, vocab_size, nullptr, out_result);
    });
}

extern "C" DBS_EXPORT int dbs_decode_model_steps_ex(
    DBSDecoderHandle* handle,
    DBSModelStepExFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    return guarded(handle, [&] {
        require(step_fn != nullptr, "model step callback cannot be null");
        std::vector<float> rows;
        auto fn = [&](const dbs::ModelStepInfo& info, float* out_rows) {
            DBSModelStepInfoC c{};
            c.batch_index = batch_index;
            c.step = info.step;
            c.beam_size = info.beam_size;
            c.vocab_size = info.vocab_size;
            c.parents = info.parents;
            c.tokens = info.tokens;
            c.lengths = info.lengths;
            c.scores = info.scores;
            c.raw_scores = info.raw_scores;
            c.finished = info.finished;
            c.prefixes = info.prefixes;
            if (step_fn(user_data, &c, out_rows) != 0) {
                throw std::runtime_error("model step callback returned non-zero status");
            }
        };
        DBSAdvancedConstraintsC with_index{};
        if (constraints) {
            with_index = *constraints;
            with_index.batch_index = batch_index;
        }
        decode_model_steps_impl(handle, &rows, fn, steps, vocab_size, constraints ? &with_index : nullptr, out_result);
    });
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
    return guarded(handle, [&] {
        require(out_result != nullptr, "out_result cannot be null");
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        require(log_probs != nullptr, "log_probs cannot be null");
        require(batch_size > 0 && steps > 0 && vocab_size > 0, "batch_size, steps and vocab_size must be positive");
        const auto start = std::chrono::steady_clock::now();
        const size_t stride = dbs::checked_mul_size(
            dbs::checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(decoder.options().beam_size), "batch stride overflow"),
            static_cast<size_t>(vocab_size), "batch stride overflow");
        const size_t work = dbs::checked_mul_size(stride, static_cast<size_t>(batch_size), "batch size overflow");

        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));
        const int threads = thread_count(num_threads, batch_size, work);
        parallel_for(batch_size, threads, [&](int b) {
            try {
                br->results[static_cast<size_t>(b)].result =
                    decoder.decode(log_probs + static_cast<size_t>(b) * stride, steps, vocab_size);
            } catch (const std::invalid_argument& e) {
                rethrow_for_example(b, e);
            }
        });
        record_decode_stats(handle, br->results.front().result, now_ns_since(start), threads, false);
        *out_result = br.release();
    });
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
    return guarded(handle, [&] {
        require(out_result != nullptr, "out_result cannot be null");
        decoder_of(handle);
        require(log_probs != nullptr, "log_probs cannot be null");
        require(batch_size > 0 && max_steps > 0 && max_beam_size > 0 && vocab_size > 0,
                "batch_size, max_steps, max_beam_size and vocab_size must be positive");
        const auto start = std::chrono::steady_clock::now();
        const size_t step_stride = dbs::checked_mul_size(static_cast<size_t>(max_beam_size), static_cast<size_t>(vocab_size), "variable batch stride overflow");
        const size_t input_stride = dbs::checked_mul_size(step_stride, static_cast<size_t>(max_steps), "variable batch stride overflow");
        const size_t work = dbs::checked_mul_size(input_stride, static_cast<size_t>(batch_size), "variable batch size overflow");

        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));
        const int threads = thread_count(num_threads, batch_size, work);
        parallel_for(batch_size, threads, [&](int b) {
            try {
                const int steps = steps_per_example ? steps_per_example[b] : max_steps;
                const int beam = beam_sizes_per_example ? beam_sizes_per_example[b] : handle->options.beam_size;
                require(steps > 0 && steps <= max_steps, "per-example steps must be in [1, max_steps]");
                require(beam > 0 && beam <= max_beam_size, "per-example beam size must be in [1, max_beam_size]");

                dbs::BeamOptions opt = handle->options;
                opt.beam_size = beam;
                if (eos_tokens_per_example) opt.eos_token = eos_tokens_per_example[b];
                if (min_lengths_per_example) opt.min_length = std::max(0, min_lengths_per_example[b]);
                const dbs::BeamSearchDecoder local_decoder(opt);

                dbs::DecodeConstraints constraints;
                if (banned_tokens_per_example) constraints.banned_tokens = banned_tokens_per_example + static_cast<size_t>(b) * static_cast<size_t>(vocab_size);
                if (forced_tokens_per_example) constraints.forced_tokens = forced_tokens_per_example + static_cast<size_t>(b) * static_cast<size_t>(max_steps);
                constraints.batch_index = b;
                // Rows of a smaller beam are read in place (the first `beam`
                // rows of each [max_beam_size, V] step).
                br->results[static_cast<size_t>(b)].result = local_decoder.decode_constrained(
                    log_probs + static_cast<size_t>(b) * input_stride, steps, vocab_size, &constraints,
                    static_cast<int64_t>(step_stride));
            } catch (const std::invalid_argument& e) {
                rethrow_for_example(b, e);
            }
        });
        record_decode_stats(handle, br->results.front().result, now_ns_since(start), threads, false);
        *out_result = br.release();
    });
}

namespace {

// dbs_decode_batch_into and dbs_decode_batch_into_ex (ex may be null).
void decode_batch_into(
    DBSDecoderHandle* handle,
    const void* inputs,
    int data_type,
    bool from_logits,
    int batch_size,
    int steps,
    int vocab_size,
    const int32_t* steps_per_example,
    const DBSAdvancedConstraintsC* constraints_c,
    int num_threads,
    const DBSDecodeOutputsC* outputs,
    const DBSDecodeOutputsExC* ex) {
    const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
    const dbs::DType type = dtype_of(data_type);
    require(inputs != nullptr, "log_probs cannot be null");
    require(outputs != nullptr && outputs->final_scores != nullptr, "outputs and outputs->final_scores cannot be null");
    require(batch_size > 0 && steps > 0 && vocab_size > 0, "batch_size, steps and vocab_size must be positive");
    const bool pool = ex && (ex->pool_parents || ex->pool_tokens || ex->pool_lengths || ex->pool_scores ||
                             ex->pool_raw_scores || ex->pool_from_logprob);
    if (ex) {
        require(!ex->reserved[0] && !ex->reserved[1] && !ex->reserved[2] && !ex->reserved[3], "reserved fields must be NULL");
        require(!pool || decoder.options().relaxed_pool_multiplier > 0,
                "pool outputs need a relaxed pool: set relaxed_pool_multiplier >= 1");
    }
    const auto start = std::chrono::steady_clock::now();
    const int K = decoder.options().beam_size;
    const size_t P = pool ? static_cast<size_t>(K) * static_cast<size_t>(decoder.options().relaxed_pool_multiplier) : 0;
    const size_t trace_stride = dbs::checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(K), "batch stride overflow");
    const size_t pool_stride = dbs::checked_mul_size(static_cast<size_t>(steps), P, "batch stride overflow");
    const size_t input_stride = dbs::checked_mul_size(trace_stride, static_cast<size_t>(vocab_size), "batch stride overflow");
    const size_t work = dbs::checked_mul_size(input_stride, static_cast<size_t>(batch_size), "batch size overflow");
    const size_t input_bytes = dbs::checked_mul_size(input_stride, dbs::dtype_size(type), "batch stride overflow");
    const std::vector<int32_t> steps_b = steps_for(steps_per_example, batch_size, steps);
    const dbs::DecodeConstraints shared = constraints_c ? from_c_constraints(*constraints_c) : dbs::DecodeConstraints{};
    const char* base = static_cast<const char*>(inputs);

    const int threads = thread_count(num_threads, batch_size, work);
    parallel_for(batch_size, threads, [&](int b) {
        const size_t bs = static_cast<size_t>(b);
        const size_t off = bs * trace_stride;
        const size_t poff = bs * pool_stride;
        dbs::TraceOutputs trace;
        trace.parents = outputs->parents ? outputs->parents + off : nullptr;
        trace.tokens = outputs->tokens ? outputs->tokens + off : nullptr;
        trace.lengths = outputs->lengths ? outputs->lengths + off : nullptr;
        trace.scores = outputs->scores ? outputs->scores + off : nullptr;
        trace.raw_scores = outputs->raw_scores ? outputs->raw_scores + off : nullptr;
        trace.from_logprob = outputs->from_logprob ? outputs->from_logprob + off : nullptr;
        if (ex) {
            trace.pool_parents = ex->pool_parents ? ex->pool_parents + poff : nullptr;
            trace.pool_tokens = ex->pool_tokens ? ex->pool_tokens + poff : nullptr;
            trace.pool_lengths = ex->pool_lengths ? ex->pool_lengths + poff : nullptr;
            trace.pool_scores = ex->pool_scores ? ex->pool_scores + poff : nullptr;
            trace.pool_raw_scores = ex->pool_raw_scores ? ex->pool_raw_scores + poff : nullptr;
            trace.pool_from_logprob = ex->pool_from_logprob ? ex->pool_from_logprob + poff : nullptr;
            trace.row_lse = from_logits && ex->row_lse ? ex->row_lse + off : nullptr;
            // Rows the search does not read get 0.
            if (ex->row_lse) std::fill(ex->row_lse + off, ex->row_lse + off + trace_stride, 0.0f);
        }

        // Padding for the steps this example does not decode.
        const size_t used = static_cast<size_t>(steps_b[bs]) * static_cast<size_t>(K);
        const size_t pool_used = static_cast<size_t>(steps_b[bs]) * P;
        const float neg_inf = -std::numeric_limits<float>::infinity();
        if (trace.parents) std::fill(trace.parents + used, trace.parents + trace_stride, -1);
        if (trace.tokens) std::fill(trace.tokens + used, trace.tokens + trace_stride, -1);
        if (trace.lengths) std::fill(trace.lengths + used, trace.lengths + trace_stride, 0);
        if (trace.scores) std::fill(trace.scores + used, trace.scores + trace_stride, neg_inf);
        if (trace.raw_scores) std::fill(trace.raw_scores + used, trace.raw_scores + trace_stride, neg_inf);
        if (trace.from_logprob) std::fill(trace.from_logprob + used, trace.from_logprob + trace_stride, uint8_t{0});
        if (trace.pool_parents) std::fill(trace.pool_parents + pool_used, trace.pool_parents + pool_stride, -1);
        if (trace.pool_tokens) std::fill(trace.pool_tokens + pool_used, trace.pool_tokens + pool_stride, -1);
        if (trace.pool_lengths) std::fill(trace.pool_lengths + pool_used, trace.pool_lengths + pool_stride, 0);
        if (trace.pool_scores) std::fill(trace.pool_scores + pool_used, trace.pool_scores + pool_stride, neg_inf);
        if (trace.pool_raw_scores) std::fill(trace.pool_raw_scores + pool_used, trace.pool_raw_scores + pool_stride, neg_inf);
        if (trace.pool_from_logprob) {
            std::fill(trace.pool_from_logprob + pool_used, trace.pool_from_logprob + pool_stride, uint8_t{0});
        }

        dbs::DecodeConstraints constraints = shared;
        constraints.batch_index = b;
        const size_t final_off = bs * static_cast<size_t>(K);
        try {
            decoder.decode_into(
                base + bs * input_bytes, type, from_logits, steps_b[bs], vocab_size,
                constraints_c ? &constraints : nullptr, trace,
                outputs->final_scores + final_off,
                outputs->final_raw_scores ? outputs->final_raw_scores + final_off : nullptr,
                outputs->final_lengths ? outputs->final_lengths + final_off : nullptr);
        } catch (const std::invalid_argument& e) {
            rethrow_for_example(b, e);
        }
    });
    record_batch_stats(handle, static_cast<int64_t>(batch_size) * static_cast<int64_t>(trace_stride),
                       static_cast<int64_t>(batch_size) * static_cast<int64_t>(input_stride),
                       now_ns_since(start), threads);
}

} // namespace

extern "C" DBS_EXPORT int dbs_decode_batch_into(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int steps,
    int vocab_size,
    const int32_t* steps_per_example,
    const DBSAdvancedConstraintsC* constraints_c,
    int num_threads,
    const DBSDecodeOutputsC* outputs
) {
    return guarded(handle, [&] {
        decode_batch_into(handle, log_probs, DBS_DTYPE_F32, false, batch_size, steps, vocab_size, steps_per_example,
                          constraints_c, num_threads, outputs, nullptr);
    });
}

extern "C" DBS_EXPORT int dbs_decode_batch_into_ex(
    DBSDecoderHandle* handle,
    const void* inputs,
    int data_type,
    int from_logits,
    int batch_size,
    int steps,
    int vocab_size,
    const int32_t* steps_per_example,
    const DBSAdvancedConstraintsC* constraints_c,
    int num_threads,
    const DBSDecodeOutputsExC* outputs
) {
    return guarded(handle, [&] {
        require(outputs != nullptr, "outputs cannot be null");
        decode_batch_into(handle, inputs, data_type, from_logits != 0, batch_size, steps, vocab_size, steps_per_example,
                          constraints_c, num_threads, &outputs->base, outputs);
    });
}

extern "C" DBS_EXPORT int dbs_backward_batch_into(
    DBSDecoderHandle* handle,
    int batch_size,
    int steps,
    int vocab_size,
    const int32_t* steps_per_example,
    const int32_t* parents,
    const int32_t* tokens,
    const int32_t* lengths,
    const uint8_t* from_logprob,
    const float* grad_final_scores,
    int num_threads,
    float* grad_log_probs
) {
    return guarded(handle, [&] {
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        require(parents && tokens && lengths && from_logprob, "parents, tokens, lengths and from_logprob cannot be null");
        require(grad_final_scores && grad_log_probs, "grad_final_scores and grad_log_probs cannot be null");
        require(batch_size > 0 && steps > 0 && vocab_size > 0, "batch_size, steps and vocab_size must be positive");
        const auto start = std::chrono::steady_clock::now();
        const int K = decoder.options().beam_size;
        const size_t trace_stride = dbs::checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(K), "batch stride overflow");
        const size_t grad_stride = dbs::checked_mul_size(trace_stride, static_cast<size_t>(vocab_size), "batch stride overflow");
        (void)dbs::checked_mul_size(grad_stride, static_cast<size_t>(batch_size), "batch size overflow");
        const size_t work = trace_stride * static_cast<size_t>(batch_size);  // no larger than the product above
        const std::vector<int32_t> steps_b = steps_for(steps_per_example, batch_size, steps);

        parallel_for(batch_size, thread_count(num_threads, batch_size, work), [&](int b) {
            const size_t bs = static_cast<size_t>(b);
            dbs::TraceView trace;
            trace.steps = steps_b[bs];
            trace.beam_size = K;
            trace.vocab_size = vocab_size;
            trace.length_penalty_alpha = decoder.options().length_penalty_alpha;
            trace.parents = parents + bs * trace_stride;
            trace.tokens = tokens + bs * trace_stride;
            trace.lengths = lengths + bs * trace_stride;
            trace.from_logprob = from_logprob + bs * trace_stride;
            try {
                dbs::final_scores_backward_into(trace, grad_final_scores + bs * static_cast<size_t>(K), grad_log_probs + bs * grad_stride);
            } catch (const std::invalid_argument& e) {
                rethrow_for_example(b, e);
            }
        });
        std::lock_guard<std::mutex> lock(handle->stats_mutex);
        handle->stats.used_sparse_backward = 0;
        handle->stats.used_dense_backward = 1;
        handle->stats.last_backward_ns = now_ns_since(start);
        handle->stats.last_sparse_grad_count = 0;
        handle->stats.last_error_category = 0;
    });
}

extern "C" DBS_EXPORT int dbs_backward_batch_into_ex(
    DBSDecoderHandle* handle,
    const DBSBackwardInputsC* in,
    int num_threads,
    float* grad_inputs
) {
    return guarded(handle, [&] {
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        require(in != nullptr && grad_inputs != nullptr, "inputs and grad_inputs cannot be null");
        require(!in->reserved[0] && !in->reserved[1] && !in->reserved[2] && !in->reserved[3], "reserved fields must be NULL");
        require(in->parents && in->tokens && in->lengths && in->from_logprob, "parents, tokens, lengths and from_logprob cannot be null");
        require(in->batch_size > 0 && in->steps > 0 && in->vocab_size > 0 && in->pool_size >= 0,
                "batch_size, steps and vocab_size must be positive and pool_size non-negative");
        const bool pool_grads = in->grad_pool_scores || in->grad_pool_raw_scores;
        require(!pool_grads || (in->pool_size > 0 && in->pool_parents && in->pool_tokens && in->pool_lengths && in->pool_from_logprob),
                "pool gradients need pool_size > 0 and the pool trace");
        dbs::DType logits_type = dbs::DType::F32;
        if (in->logits) {
            logits_type = dtype_of(in->logits_type);
            require(in->row_lse != nullptr, "row_lse cannot be null with logits");
        }
        const auto start = std::chrono::steady_clock::now();
        const int B = in->batch_size, T = in->steps, V = in->vocab_size;
        const int K = decoder.options().beam_size;
        const int P = pool_grads ? in->pool_size : 0;
        const size_t trace_stride = dbs::checked_mul_size(static_cast<size_t>(T), static_cast<size_t>(K), "batch stride overflow");
        const size_t pool_stride = dbs::checked_mul_size(static_cast<size_t>(T), static_cast<size_t>(in->pool_size), "batch stride overflow");
        const size_t grad_stride = dbs::checked_mul_size(trace_stride, static_cast<size_t>(V), "batch stride overflow");
        (void)dbs::checked_mul_size(grad_stride, static_cast<size_t>(B), "batch size overflow");
        const std::vector<int32_t> steps_b = steps_for(in->steps_per_example, B, T);
        // With logits every row with a path gradient is written in full.
        const size_t work = trace_stride * static_cast<size_t>(B) * (in->logits ? static_cast<size_t>(V) : 1);
        const size_t logits_row_bytes = static_cast<size_t>(V) * dbs::dtype_size(logits_type);
        const char* logits = static_cast<const char*>(in->logits);

        parallel_for(B, thread_count(num_threads, B, work), [&](int b) {
            const size_t bs = static_cast<size_t>(b);
            const size_t off = bs * trace_stride;
            const size_t poff = bs * pool_stride;
            dbs::TraceView trace;
            trace.steps = steps_b[bs];
            trace.beam_size = K;
            trace.vocab_size = V;
            trace.pool_size = P;
            trace.length_penalty_alpha = decoder.options().length_penalty_alpha;
            trace.parents = in->parents + off;
            trace.tokens = in->tokens + off;
            trace.lengths = in->lengths + off;
            trace.from_logprob = in->from_logprob + off;
            if (P > 0) {
                trace.pool_parents = in->pool_parents + poff;
                trace.pool_tokens = in->pool_tokens + poff;
                trace.pool_lengths = in->pool_lengths + poff;
                trace.pool_from_logprob = in->pool_from_logprob + poff;
            }
            dbs::OutputGradients g;
            g.final_scores = in->grad_final_scores ? in->grad_final_scores + bs * static_cast<size_t>(K) : nullptr;
            g.final_raw_scores = in->grad_final_raw_scores ? in->grad_final_raw_scores + bs * static_cast<size_t>(K) : nullptr;
            g.scores = in->grad_scores ? in->grad_scores + off : nullptr;
            g.raw_scores = in->grad_raw_scores ? in->grad_raw_scores + off : nullptr;
            g.pool_scores = in->grad_pool_scores ? in->grad_pool_scores + poff : nullptr;
            g.pool_raw_scores = in->grad_pool_raw_scores ? in->grad_pool_raw_scores + poff : nullptr;
            float* out = grad_inputs + bs * grad_stride;
            try {
                const std::vector<dbs::SparseGradEntry> entries = dbs::path_gradient(trace, g);
                const auto add = [out](int64_t index, float value) { out[index] += value; };
                if (!logits) {
                    dbs::apply_path_gradient(entries, V, nullptr, [](int64_t, float*) -> const float* { return nullptr; }, add);
                } else {
                    const char* rows = logits + bs * trace_stride * logits_row_bytes;
                    const auto read_row = [&](int64_t r, float* buffer) -> const float* {
                        const char* row = rows + static_cast<size_t>(r) * logits_row_bytes;
                        if (logits_type == dbs::DType::F32) return reinterpret_cast<const float*>(row);
                        dbs::convert_to_float(row, logits_type, static_cast<size_t>(V), buffer);
                        return buffer;
                    };
                    dbs::apply_path_gradient(entries, V, in->row_lse + off, read_row, add);
                }
            } catch (const std::invalid_argument& e) {
                rethrow_for_example(b, e);
            }
        });
        std::lock_guard<std::mutex> lock(handle->stats_mutex);
        handle->stats.used_sparse_backward = 0;
        handle->stats.used_dense_backward = 1;
        handle->stats.last_backward_ns = now_ns_since(start);
        handle->stats.last_sparse_grad_count = 0;
        handle->stats.last_error_category = 0;
    });
}

extern "C" DBS_EXPORT int dbs_backward(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    return dbs_backward_sparse(handle, result, grad_selected_weights, grad_relaxed_weights, grad_final_scores, out_backward);
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
    return guarded(handle, [&] {
        require(out_backward != nullptr, "out_backward cannot be null");
        require(result != nullptr, "result cannot be null");
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        const auto start = std::chrono::steady_clock::now();
        auto b = std::make_unique<DBSBackwardHandle>();
        b->result = decoder.backward(result->result, grad_selected_weights, grad_relaxed_weights, grad_final_scores);
        record_backward_stats(handle, b->result, now_ns_since(start));
        *out_backward = b.release();
    });
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
    return guarded(handle, [&] {
        require(out_backward != nullptr, "out_backward cannot be null");
        require(result != nullptr, "result cannot be null");
        const dbs::BeamSearchDecoder& decoder = decoder_of(handle);
        const auto start = std::chrono::steady_clock::now();
        auto b = std::make_unique<DBSBackwardHandle>();
        b->result = decoder.backward_sparse(result->result, grad_selected_weights, grad_relaxed_weights, grad_final_scores);
        record_backward_stats(handle, b->result, now_ns_since(start));
        *out_backward = b.release();
    });
}

extern "C" DBS_EXPORT int dbs_backward_default(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    return dbs_backward_sparse(handle, result, grad_selected_weights, grad_relaxed_weights, grad_final_scores, out_backward);
}

extern "C" DBS_EXPORT void dbs_free_result(DBSResultHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT void dbs_free_batch_result(DBSBatchResultHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT void dbs_free_backward(DBSBackwardHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT int dbs_batch_result_size(const DBSBatchResultHandle* result) {
    return result ? static_cast<int>(result->results.size()) : 0;
}

extern "C" DBS_EXPORT const DBSResultHandle* dbs_batch_result_at(const DBSBatchResultHandle* result, int batch_index) {
    if (!result || batch_index < 0 || batch_index >= static_cast<int>(result->results.size())) return nullptr;
    return &result->results[static_cast<size_t>(batch_index)];
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
    return result ? data_or_null(result->result.tokens) : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_parents(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.parents) : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_lengths(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.lengths) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_scores(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.scores) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_raw_scores(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.raw_scores) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_weights(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.weights) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_final_scores(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.final_scores) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_final_raw_scores(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.final_raw_scores) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_relaxed_weights(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.relaxed_weights) : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_tokens(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.pool_tokens) : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_parents(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.pool_parents) : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_lengths(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.pool_lengths) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_pool_scores(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.pool_scores) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_pool_raw_scores(const DBSResultHandle* result) {
    return result ? data_or_null(result->result.pool_raw_scores) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_grad_log_probs(const DBSBackwardHandle* result) {
    return result ? data_or_null(result->result.grad_log_probs) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_grad_initial_scores(const DBSBackwardHandle* result) {
    return result ? data_or_null(result->result.grad_initial_scores) : nullptr;
}

extern "C" DBS_EXPORT const int64_t* dbs_backward_sparse_logprob_indices(const DBSBackwardHandle* result) {
    return result ? data_or_null(result->result.sparse_logprob_indices) : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_sparse_logprob_values(const DBSBackwardHandle* result) {
    return result ? data_or_null(result->result.sparse_logprob_values) : nullptr;
}

extern "C" DBS_EXPORT int64_t dbs_backward_sparse_logprob_count(const DBSBackwardHandle* result) {
    return result ? static_cast<int64_t>(result->result.sparse_logprob_values.size()) : 0;
}

extern "C" DBS_EXPORT int dbs_backward_is_sparse(const DBSBackwardHandle* result) {
    return result && result->result.sparse ? 1 : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_selected_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_pool_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.relaxed_pool_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_logprob_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.beam_size * result->result.vocab_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_backward_grad_log_probs_count(const DBSResultHandle* result) {
    return dbs_result_logprob_count(result);
}

extern "C" DBS_EXPORT int64_t dbs_backward_grad_initial_scores_count(const DBSResultHandle* result) {
    return result ? result->result.beam_size : 0;
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
    const dbs::DecodeResult& r = result->result;
    const int T = r.steps;
    const int K = r.beam_size;
    for (int t = 0; t < T; ++t) {
        for (int k = 1; k < K; ++k) {
            const size_t prev = static_cast<size_t>(t) * static_cast<size_t>(K) + static_cast<size_t>(k - 1);
            const size_t cur = prev + 1;
            const dbs::Candidate a{r.scores[prev], r.raw_scores[prev], r.parents[prev], r.tokens[prev], r.lengths[prev], r.from_logprob[prev]};
            const dbs::Candidate b{r.scores[cur], r.raw_scores[cur], r.parents[cur], r.tokens[cur], r.lengths[cur], r.from_logprob[cur]};
            if (!dbs::candidate_better(b, a)) continue;  // a precedes (or equals) b
            return b.score > a.score ? -3 : -4;
        }
    }
    return 0;
}

extern "C" DBS_EXPORT int dbs_result_summary_json(const DBSResultHandle* result, int eos_token, char* out_json, int64_t out_json_capacity) {
    if (!result || !out_json || out_json_capacity <= 0) {
        return fail(nullptr, "result and out_json cannot be null, and the capacity must be positive", 1, DBS_ERROR_INVALID_ARGUMENT);
    }
    const auto& r = result->result;
    int min_len = r.lengths.empty() ? 0 : std::numeric_limits<int>::max();
    int max_len = 0;
    for (int32_t len : r.lengths) {
        min_len = std::min(min_len, static_cast<int>(len));
        max_len = std::max(max_len, static_cast<int>(len));
    }
    float min_final = r.final_scores.empty() ? 0.0f : r.final_scores[0];
    float max_final = min_final;
    for (float score : r.final_scores) {
        min_final = std::min(min_final, score);
        max_final = std::max(max_final, score);
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
    if (n < 0) return fail(nullptr, "formatting the summary failed", 2, DBS_ERROR_RUNTIME);
    return n < out_json_capacity ? 0 : 1;
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
    return guarded(nullptr, [&] {
        require(handle != nullptr && out_stats != nullptr, "handle and out_stats cannot be null");
        std::lock_guard<std::mutex> lock(handle->stats_mutex);
        *out_stats = handle->stats;
        out_stats->total_allocator_calls = dbs::g_allocator_calls.load(std::memory_order_relaxed);
        out_stats->total_allocator_bytes = dbs::g_allocator_bytes.load(std::memory_order_relaxed);
    });
}

extern "C" DBS_EXPORT int dbs_get_stats_json(DBSDecoderHandle* handle, char* out_json, int64_t out_json_capacity) {
    if (!handle || !out_json || out_json_capacity <= 0) {
        return fail(nullptr, "handle and out_json cannot be null, and the capacity must be positive", 1, DBS_ERROR_INVALID_ARGUMENT);
    }
    DBSStatsC s{};
    const int rc = dbs_get_stats(handle, &s);
    if (rc != DBS_OK) return rc;
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
    if (n < 0) return fail(nullptr, "formatting stats failed", 2, DBS_ERROR_RUNTIME);
    return n < out_json_capacity ? 0 : 1;
}

extern "C" DBS_EXPORT int dbs_is_deterministic() {
    return 1;
}

extern "C" DBS_EXPORT int dbs_set_deterministic_seed(DBSDecoderHandle* handle, uint64_t seed) {
    return guarded(nullptr, [&] {
        require(handle != nullptr, "handle cannot be null");
        handle->deterministic_seed.store(seed, std::memory_order_relaxed);
    });
}

extern "C" DBS_EXPORT uint64_t dbs_get_deterministic_seed(DBSDecoderHandle* handle) {
    return handle ? handle->deterministic_seed.load(std::memory_order_relaxed) : 0;
}

extern "C" DBS_EXPORT void dbs_allocator_counters_reset() {
    // The byte count is a gauge of live allocations: resetting it would make it
    // go negative as those allocations are freed.
    dbs::g_allocator_calls.store(0, std::memory_order_relaxed);
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
