// SPDX-License-Identifier: MIT
//
// Runtime CPU feature detection and kernel-path selection.
#include "kernels.hpp"

namespace dbs {

bool runtime_has_avx512() noexcept {
#if DBS_CAN_COMPILE_AVX512
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
    }();
    return supported;
#else
    return false;
#endif
}

bool runtime_has_avx2() noexcept {
#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return supported;
#else
    return false;
#endif
}

bool runtime_has_sse42() noexcept {
#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("sse4.2");
    }();
    return supported;
#else
    return false;
#endif
}

bool runtime_has_neon() noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

namespace {

thread_local KernelOverride g_kernel_override;

bool kernel_path_runtime_available(KernelPath k) noexcept {
    switch (k) {
        case KernelPath::AVX512: return runtime_has_avx512();
        case KernelPath::AVX2: return runtime_has_avx2();
        case KernelPath::SSE42: return runtime_has_sse42();
        case KernelPath::NEON: return runtime_has_neon();
        case KernelPath::Scalar: return true;
    }
    return false;
}

} // namespace

KernelOverride current_kernel_override() noexcept {
    return g_kernel_override;
}

void set_kernel_override(KernelOverride value) noexcept {
    g_kernel_override = value;
}

bool kernel_path_enabled(KernelPath k) noexcept {
    return !g_kernel_override.enabled || g_kernel_override.path == k;
}

KernelPath selected_kernel_path() noexcept {
    if (g_kernel_override.enabled) {
        return kernel_path_runtime_available(g_kernel_override.path) ? g_kernel_override.path : KernelPath::Scalar;
    }
#if DBS_CAN_COMPILE_AVX512
    if (runtime_has_avx512()) return KernelPath::AVX512;
#endif
#if DBS_CAN_COMPILE_AVX2
    if (runtime_has_avx2()) return KernelPath::AVX2;
#endif
#if DBS_CAN_COMPILE_SSE42
    if (runtime_has_sse42()) return KernelPath::SSE42;
#endif
#if DBS_ARM_NEON
    if (runtime_has_neon()) return KernelPath::NEON;
#endif
    return KernelPath::Scalar;
}

const char* kernel_path_name(KernelPath k) noexcept {
    switch (k) {
        case KernelPath::AVX512: return "avx512";
        case KernelPath::AVX2: return "avx2";
        case KernelPath::SSE42: return "sse4.2";
        case KernelPath::NEON: return "neon";
        case KernelPath::Scalar: return "scalar";
    }
    return "scalar";
}

} // namespace dbs
