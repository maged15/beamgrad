// SPDX-License-Identifier: MIT
//
// GNMT length penalty ((5 + length) / 6)^alpha, shared by the CPU decoder and
// the CUDA engine.
//
// The power is evaluated in double precision from basic IEEE operations only
// (add, subtract, multiply, divide, and the exact frexp/ldexp), with no libm
// pow/log/exp and no fused multiply-add, then rounded to float once. Basic IEEE
// operations are correctly rounded on every CPU and GPU, so the host and the
// device produce the same bits. Host code that includes this header must be
// compiled without floating-point contraction (-ffp-contract=off; the build
// sets it); device code uses the explicitly rounded intrinsics.
#pragma once

#include <math.h>

#if defined(__CUDACC__)
#define DBS_HOST_DEVICE __host__ __device__
#else
#define DBS_HOST_DEVICE
#endif

namespace dbs {
namespace penalty_detail {

#if defined(__CUDA_ARCH__)
DBS_HOST_DEVICE inline double add(double a, double b) { return __dadd_rn(a, b); }
DBS_HOST_DEVICE inline double sub(double a, double b) { return __dsub_rn(a, b); }
DBS_HOST_DEVICE inline double mul(double a, double b) { return __dmul_rn(a, b); }
DBS_HOST_DEVICE inline double div(double a, double b) { return __ddiv_rn(a, b); }
#else
DBS_HOST_DEVICE inline double add(double a, double b) { return a + b; }
DBS_HOST_DEVICE inline double sub(double a, double b) { return a - b; }
DBS_HOST_DEVICE inline double mul(double a, double b) { return a * b; }
DBS_HOST_DEVICE inline double div(double a, double b) { return a / b; }
#endif

// ln(x) for x >= 1: x = m * 2^e with m in [sqrt(1/2), sqrt(2)), and
// ln(m) = 2 atanh(s) with s = (m - 1) / (m + 1), |s| <= 0.172.
DBS_HOST_DEVICE inline double log_ge1(double x) {
    int e = 0;
    double m = ::frexp(x, &e);  // m in [0.5, 1)
    if (m < 0.70710678118654752440) {
        m = mul(m, 2.0);
        e -= 1;
    }
    const double s = div(sub(m, 1.0), add(m, 1.0));
    const double s2 = mul(s, s);
    // 1 + s2/3 + s2^2/5 + ... + s2^11/23 (truncation error below 1e-18).
    double p = 1.0 / 23.0;
    p = add(mul(p, s2), 1.0 / 21.0);
    p = add(mul(p, s2), 1.0 / 19.0);
    p = add(mul(p, s2), 1.0 / 17.0);
    p = add(mul(p, s2), 1.0 / 15.0);
    p = add(mul(p, s2), 1.0 / 13.0);
    p = add(mul(p, s2), 1.0 / 11.0);
    p = add(mul(p, s2), 1.0 / 9.0);
    p = add(mul(p, s2), 1.0 / 7.0);
    p = add(mul(p, s2), 1.0 / 5.0);
    p = add(mul(p, s2), 1.0 / 3.0);
    p = add(mul(p, s2), 1.0);
    const double log_m = mul(mul(2.0, s), p);
    // ln 2 split so that e * kLn2Hi is exact.
    constexpr double kLn2Hi = 6.93147180369123816490e-01;
    constexpr double kLn2Lo = 1.90821492927058770002e-10;
    const double de = static_cast<double>(e);
    return add(mul(de, kLn2Hi), add(mul(de, kLn2Lo), log_m));
}

// exp(y) for y >= 0: y = k ln 2 + r with |r| <= ln(2) / 2, exp(r) by its
// Taylor series (truncation error below 1e-20).
DBS_HOST_DEVICE inline double exp_ge0(double y) {
    if (y > 709.0) return HUGE_VAL;  // beyond the float range long before this
    constexpr double kInvLn2 = 1.44269504088896338700e+00;
    constexpr double kLn2Hi = 6.93147180369123816490e-01;
    constexpr double kLn2Lo = 1.90821492927058770002e-10;
    const int k = static_cast<int>(add(mul(y, kInvLn2), 0.5));  // round half up; y >= 0
    const double dk = static_cast<double>(k);
    const double r = sub(sub(y, mul(dk, kLn2Hi)), mul(dk, kLn2Lo));
    double p = 1.0 / 1307674368000.0;          // 1/15!
    p = add(mul(p, r), 1.0 / 87178291200.0);   // 1/14!
    p = add(mul(p, r), 1.0 / 6227020800.0);    // 1/13!
    p = add(mul(p, r), 1.0 / 479001600.0);     // 1/12!
    p = add(mul(p, r), 1.0 / 39916800.0);      // 1/11!
    p = add(mul(p, r), 1.0 / 3628800.0);       // 1/10!
    p = add(mul(p, r), 1.0 / 362880.0);        // 1/9!
    p = add(mul(p, r), 1.0 / 40320.0);         // 1/8!
    p = add(mul(p, r), 1.0 / 5040.0);          // 1/7!
    p = add(mul(p, r), 1.0 / 720.0);           // 1/6!
    p = add(mul(p, r), 1.0 / 120.0);           // 1/5!
    p = add(mul(p, r), 1.0 / 24.0);            // 1/4!
    p = add(mul(p, r), 1.0 / 6.0);             // 1/3!
    p = add(mul(p, r), 0.5);                   // 1/2!
    p = add(mul(p, r), 1.0);
    p = add(mul(p, r), 1.0);
    return ::ldexp(p, k);
}

} // namespace penalty_detail

DBS_HOST_DEVICE inline float gnmt_length_penalty(int length, float alpha) {
    if (alpha == 0.0f) return 1.0f;
    const int l = length > 1 ? length : 1;
    const double base = penalty_detail::div(5.0 + static_cast<double>(l), 6.0);
    // Small integer exponents by repeated squaring, which is exact whenever the
    // result fits in a double (e.g. alpha = 1, 2, 3 on short hypotheses). The
    // range check comes first: converting a float beyond INT_MAX to int is
    // undefined behaviour.
    if (alpha <= 64.0f && alpha == static_cast<float>(static_cast<int>(alpha))) {
        int n = static_cast<int>(alpha);
        double result = 1.0;
        double b = base;
        while (n > 0) {
            if (n & 1) result = penalty_detail::mul(result, b);
            b = penalty_detail::mul(b, b);
            n >>= 1;
        }
        return static_cast<float>(result);
    }
    const double y = penalty_detail::mul(static_cast<double>(alpha), penalty_detail::log_ge1(base));
    return static_cast<float>(penalty_detail::exp_ge0(y));
}

} // namespace dbs
