// SPDX-License-Identifier: MIT
//
// Log-softmax pieces shared by the CPU decoder and the CUDA engine, for
// decoding directly from logits (x - logsumexp(row) is computed on the fly
// instead of materialising log_softmax).
//
// Everything here uses basic IEEE float operations only (add, subtract,
// multiply, floor, exponent bits) in a fixed order, with no fused
// multiply-add, so every CPU kernel path and the GPU produce the same bits.
// The row sum uses a fixed reduction order: lane j (of kLogitLanes) sums the
// entries v = j, j + kLogitLanes, ... in increasing v, then the lanes are
// combined by a pairwise tree. The CUDA engine maps lanes to threads.
#pragma once

#include "penalty.hpp"

#include <stdint.h>
#include <string.h>

namespace dbs {

constexpr int kLogitLanes = 256;

namespace det {

#if defined(__CUDA_ARCH__)
DBS_HOST_DEVICE inline float add(float a, float b) { return __fadd_rn(a, b); }
DBS_HOST_DEVICE inline float sub(float a, float b) { return __fsub_rn(a, b); }
DBS_HOST_DEVICE inline float mul(float a, float b) { return __fmul_rn(a, b); }
DBS_HOST_DEVICE inline float as_float(uint32_t u) { return __uint_as_float(u); }
#else
DBS_HOST_DEVICE inline float add(float a, float b) { return a + b; }
DBS_HOST_DEVICE inline float sub(float a, float b) { return a - b; }
DBS_HOST_DEVICE inline float mul(float a, float b) { return a * b; }
DBS_HOST_DEVICE inline float as_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
#endif

// Constants of exp_nonpositive (the Cephes expf split and polynomial).
constexpr float kLog2e = 1.44269504088896341f;
constexpr float kLn2Hi = 0.693359375f;
constexpr float kLn2Lo = -2.12194440e-4f;
constexpr float kExpP0 = 1.9875691500e-4f;
constexpr float kExpP1 = 1.3981999507e-3f;
constexpr float kExpP2 = 8.3334519073e-3f;
constexpr float kExpP3 = 4.1665795894e-2f;
constexpr float kExpP4 = 1.6666665459e-1f;
constexpr float kExpP5 = 5.0000001201e-1f;
// Below this the result is (about) the smallest normal float or less: 0 is
// returned, which keeps every result a normal float.
constexpr float kExpMin = -87.0f;

// exp(x) for x <= 0 (within about 2 ulp); exactly 1 at 0, and 0 below
// kExpMin (and for -inf and NaN). The SIMD kernels evaluate the same
// operations in the same order.
DBS_HOST_DEVICE inline float exp_nonpositive(float x) {
    if (!(x >= kExpMin)) return 0.0f;
    const float k = ::floorf(add(mul(x, kLog2e), 0.5f));
    const float r = sub(sub(x, mul(k, kLn2Hi)), mul(k, kLn2Lo));
    const float r2 = mul(r, r);
    float p = kExpP0;
    p = add(mul(p, r), kExpP1);
    p = add(mul(p, r), kExpP2);
    p = add(mul(p, r), kExpP3);
    p = add(mul(p, r), kExpP4);
    p = add(mul(p, r), kExpP5);
    p = add(add(mul(p, r2), r), 1.0f);
    const int e = static_cast<int>(k) + 127;  // k in [-126, 0] for x in [kExpMin, 0]
    return mul(p, as_float(static_cast<uint32_t>(e) << 23));
}

} // namespace det

// logsumexp of a row from its maximum over finite entries and the combined
// lane sum of exp(x - max) (which is at least 1: the maximum contributes 1).
DBS_HOST_DEVICE inline float logsumexp_from(float max, float sum) {
    return det::add(max, static_cast<float>(penalty_detail::log_ge1(static_cast<double>(sum))));
}

// Softmax probability of x in a row with logsumexp lse; 0 for non-finite x.
DBS_HOST_DEVICE inline float softmax_probability(float x, float lse) {
    const bool finite = x < det::as_float(0x7f800000u) && x > det::as_float(0xff800000u);
    return finite ? det::exp_nonpositive(det::sub(x, lse)) : 0.0f;
}

} // namespace dbs
