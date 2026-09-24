// SPDX-License-Identifier: MIT
#pragma once

namespace dbs::internal_test {

struct SimdCapabilities {
    int can_compile_avx512;
    int can_compile_avx2;
    int can_compile_sse42;
    int can_compile_neon;
    int runtime_avx512;
    int runtime_avx2;
    int runtime_sse42;
    int runtime_neon;
};

struct ParityReport {
    int cases_run;
    int simd_paths_run;
    int failures;
    char message[512];
};

SimdCapabilities simd_capabilities();
int run_avx512_vector_math_parity(ParityReport* report);
int run_scalar_vs_simd_decode_backward_parity(ParityReport* report);

} // namespace dbs::internal_test
