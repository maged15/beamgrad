// SPDX-License-Identifier: MIT
//
// IEEE fp16 and bfloat16 to float32, exactly, on the host and the device, so
// the CPU decoder and the CUDA engine read 16-bit input the same way.
#pragma once

#include "logits.hpp"

#include <stdint.h>

namespace dbs {

// Element types of the input rows (values match DBSDataTypeC in dbs.h).
enum class DType : int { F32 = 0, F16 = 1, BF16 = 2 };

DBS_HOST_DEVICE inline float f16_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {  // subnormal: normalise the mantissa
            uint32_t m = mant;
            uint32_t e = 113u;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            out = sign | (e << 23) | ((m & 0x03ffu) << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return det::as_float(out);
}

DBS_HOST_DEVICE inline float bf16_bits_to_float(uint16_t h) { return det::as_float(static_cast<uint32_t>(h) << 16); }

// Element i of a buffer of `type`, as float.
DBS_HOST_DEVICE inline float load_as_float(const void* data, DType type, int64_t i) {
    if (type == DType::F32) return static_cast<const float*>(data)[i];
    const uint16_t h = static_cast<const uint16_t*>(data)[i];
    return type == DType::F16 ? f16_bits_to_float(h) : bf16_bits_to_float(h);
}

} // namespace dbs
