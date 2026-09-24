// SPDX-License-Identifier: MIT
#include "dbs.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) return 0;
    const int T = 1 + (data[0] % 4);
    const int K = 1 + (data[1] % 4);
    const int V = 1 + (data[2] % 32);
    const int eos = data[3] % (V + 1);

    DBSOptionsC opt{};
    opt.beam_size = K;
    opt.eos_token = eos == V ? -1 : eos;
    opt.selected_temperature = 0.1f + static_cast<float>(data[4] % 20) / 10.0f;
    opt.soft_topk_temperature = 0.1f + static_cast<float>(data[5] % 20) / 10.0f;
    opt.relaxed_pool_multiplier = data[6] % 4;  // 0 disables the relaxed pool
    opt.length_penalty_alpha = static_cast<float>(data[7] % 20) / 10.0f;
    opt.soft_topk_tolerance = 1.0e-4f;
    opt.soft_topk_max_iters = 16;
    opt.validate_inputs = 1;
    opt.max_dense_gradient_elements = 1000000;

    DBSDecoderHandle* h = nullptr;
    if (dbs_create_ex(opt, &h) != 0 || !h) return 0;

    std::vector<float> x(static_cast<size_t>(T) * K * V, -10.0f);
    size_t pos = 8;
    for (float& v : x) {
        if (pos + 4 <= size) {
            uint32_t raw = 0;
            std::memcpy(&raw, data + pos, 4);
            pos += 4;
            v = -static_cast<float>(raw % 10000) / 1000.0f;
        }
    }

    DBSResultHandle* r = nullptr;
    if (dbs_decode(h, x.data(), T, V, &r) == 0 && r) {
        std::vector<float> grad_final(static_cast<size_t>(K), 0.0f);
        grad_final[0] = 1.0f;
        DBSBackwardHandle* b = nullptr;
        if (dbs_backward_sparse(h, r, nullptr, nullptr, grad_final.data(), &b) == 0 && b) {
            dbs_free_backward(b);
        }
        dbs_free_result(r);
    }

    dbs_destroy(h);
    return 0;
}
