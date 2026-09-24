// SPDX-License-Identifier: MIT
//
// Runtime CPU feature detection and kernel-path selection.
#include "kernels.hpp"

namespace dbs {

#if DBS_X86_SIMD && defined(_MSC_VER) && !defined(__clang__)
namespace {

struct X86Features {
    bool sse42 = false;
    bool avx2 = false;
    bool avx512f = false;
};

// CPUID plus the OS's XSAVE state: AVX needs the YMM state enabled, AVX-512
// additionally the opmask and ZMM states.
X86Features detect_x86_features() noexcept {
    X86Features f;
    int regs[4] = {};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    __cpuid(regs, 1);
    f.sse42 = (regs[2] & (1 << 20)) != 0;
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    const bool ymm_state = (xcr0 & 0x6) == 0x6;
    const bool zmm_state = (xcr0 & 0xe6) == 0xe6;
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        f.avx2 = avx && ymm_state && (regs[1] & (1 << 5)) != 0;
        f.avx512f = avx && zmm_state && (regs[1] & (1 << 16)) != 0;
    }
    return f;
}

const X86Features& x86_features() noexcept {
    static const X86Features features = detect_x86_features();
    return features;
}

} // namespace

bool runtime_has_avx512() noexcept { return x86_features().avx512f; }
bool runtime_has_avx2() noexcept { return x86_features().avx2; }
bool runtime_has_sse42() noexcept { return x86_features().sse42; }

#elif DBS_X86_SIMD

bool runtime_has_avx512() noexcept {
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") != 0;
    }();
    return supported;
}

bool runtime_has_avx2() noexcept {
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0;
    }();
    return supported;
}

bool runtime_has_sse42() noexcept {
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("sse4.2") != 0;
    }();
    return supported;
}

#else

bool runtime_has_avx512() noexcept { return false; }
bool runtime_has_avx2() noexcept { return false; }
bool runtime_has_sse42() noexcept { return false; }

#endif

bool runtime_has_neon() noexcept {
    return DBS_ARM_NEON != 0;
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
