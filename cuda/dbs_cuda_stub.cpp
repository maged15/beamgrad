// SPDX-License-Identifier: MIT
//
// libdbs_cuda for builds without CUDA. It exports the same symbols as the real
// backend so downstream code links everywhere; every entry point reports
// DBS_CUDA_STATUS_UNAVAILABLE.
#include "dbs_cuda.h"

#include <atomic>

namespace {
std::atomic<int> g_synchronize{0};
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_available(void) { return 0; }

extern "C" DBS_CUDA_EXPORT const char* dbs_cuda_status_string(int status) {
    switch (status) {
        case DBS_CUDA_STATUS_OK: return "ok";
        case DBS_CUDA_STATUS_UNAVAILABLE: return "CUDA backend unavailable (library built without CUDA)";
        case DBS_CUDA_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DBS_CUDA_STATUS_LAUNCH_FAILED: return "CUDA launch failed";
        case DBS_CUDA_STATUS_OUT_OF_MEMORY: return "CUDA out of memory";
        default: return "unknown CUDA status";
    }
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_set_synchronization(int synchronize) {
    if (synchronize != 0 && synchronize != 1) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    g_synchronize.store(synchronize);
    return DBS_CUDA_STATUS_OK;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_get_synchronization(void) { return g_synchronize.load(); }

extern "C" DBS_CUDA_EXPORT int64_t dbs_cuda_decode_workspace_size(const DBSCudaDecodeArgs*) { return -1; }

extern "C" DBS_CUDA_EXPORT int64_t dbs_cuda_backward_workspace_size(const DBSCudaDecodeArgs*) { return -1; }

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode(
    const float*, const DBSCudaDecodeArgs*, const DBSCudaDecodeOutputs*, void*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int64_t dbs_cuda_decode_step_workspace_size(const DBSCudaDecodeArgs*, int) { return -1; }

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_step(
    const float*, const DBSCudaDecodeArgs*, const DBSCudaBeamState*, const DBSCudaDecodeOutputs*, void*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_length_penalty(const int32_t*, int64_t, float, float*, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_path_gradient(
    const DBSCudaDecodeArgs*, const int32_t*, const int32_t*, const int32_t*, const uint8_t*, const float*, float*, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_backward(
    const DBSCudaDecodeArgs*, const int32_t*, const int32_t*, const int32_t*, const uint8_t*,
    const float*, float*, void*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_ex(
    const void*, int, int, const DBSCudaDecodeArgs*, const DBSCudaDecodeOutputs*, float*, void*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_step_ex(
    const void*, int, int, const DBSCudaDecodeArgs*, const DBSCudaBeamState*, const DBSCudaDecodeOutputs*, float*,
    void*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_backward_ex(
    const DBSCudaDecodeArgs*, const DBSCudaBackwardInputs*, float*, void*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}
