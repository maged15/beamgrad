#include <cstdlib>
#include <iostream>

#if defined(_WIN32) && !defined(DBS_STATIC)
#error dbs_simd_internal_tests must compile with DBS_STATIC on Windows
#endif

#include "internal_test_hooks.hpp"

static void require_ok(int rc, const dbs::internal_test::ParityReport& report, const char* label) {
    if (rc != 0 || report.failures != 0) {
        std::cerr << label << " failed";
        if (report.message[0] != '\0') {
            std::cerr << ": " << report.message;
        }
        std::cerr << "\n";
        std::abort();
    }
}

int main() {
    const dbs::internal_test::SimdCapabilities caps = dbs::internal_test::simd_capabilities();

#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
    if (!caps.can_compile_avx2 || !caps.can_compile_sse42) {
        std::cerr << "GCC/Clang x86 builds must compile AVX2 and SSE4.2 internal paths\n";
        return 1;
    }
#endif
#endif

    dbs::internal_test::ParityReport vector_report{};
    require_ok(
        dbs::internal_test::run_avx512_vector_math_parity(&vector_report),
        vector_report,
        "avx512_vector_math_parity");

    dbs::internal_test::ParityReport decode_report{};
    require_ok(
        dbs::internal_test::run_scalar_vs_simd_decode_backward_parity(&decode_report),
        decode_report,
        "scalar_vs_simd_decode_backward_parity");

    std::cout << "dbs_simd_internal_tests passed"
              << " compile_avx512=" << caps.can_compile_avx512
              << " compile_avx2=" << caps.can_compile_avx2
              << " compile_sse42=" << caps.can_compile_sse42
              << " runtime_avx512=" << caps.runtime_avx512
              << " runtime_avx2=" << caps.runtime_avx2
              << " runtime_sse42=" << caps.runtime_sse42
              << " vector_cases=" << vector_report.cases_run
              << " decode_cases=" << decode_report.cases_run
              << "\n";
    return 0;
}
