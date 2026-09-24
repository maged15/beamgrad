// SPDX-License-Identifier: MIT
#include "dbs.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static DBSOptionsC options(int beam_size, int max_dense = 100000000) {
    DBSOptionsC opt{};
    opt.beam_size = beam_size;
    opt.eos_token = -1;
    opt.selected_temperature = 1.0f;
    opt.soft_topk_temperature = 0.25f;
    opt.relaxed_pool_multiplier = 0;  // the relaxed pool is opt-in
    opt.length_penalty_alpha = 1.0f;
    opt.soft_topk_tolerance = 1.0e-4f;
    opt.soft_topk_max_iters = 48;
    opt.min_length = 0;
    opt.validate_inputs = 0;
    opt.max_dense_gradient_elements = max_dense;
    return opt;
}

static std::vector<float> make_logits(int T, int K, int V, int seed) {
    std::vector<float> x(static_cast<size_t>(T) * K * V);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-12.0f, 0.0f);
    for (float& v : x) v = dist(rng);
    return x;
}

static long long us_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();
}

static void run_one(int T, int K, int V, int B, int repeats) {
    DBSDecoderHandle* h = nullptr;
    if (dbs_create_ex(options(K), &h) != 0) {
        std::cerr << dbs_last_global_error() << "\n";
        std::exit(1);
    }

    auto x = make_logits(T, K, V, 123 + T + K + V);
    std::vector<float> batched(static_cast<size_t>(B) * x.size());
    for (int b = 0; b < B; ++b) std::copy(x.begin(), x.end(), batched.begin() + static_cast<ptrdiff_t>(b) * static_cast<ptrdiff_t>(x.size()));

    long long fwd_us = 0;
    long long sparse_us = 0;
    long long dense_us = 0;
    long long batch_us = 0;
    int64_t sparse_nnz = 0;

    for (int i = 0; i < repeats; ++i) {
        DBSResultHandle* r = nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();
        int rc = dbs_decode(h, x.data(), T, V, &r);
        fwd_us += us_since(t0);
        if (rc != 0) {
            std::cerr << dbs_last_error(h) << "\n";
            std::exit(2);
        }

        std::vector<float> grad_final(static_cast<size_t>(K), 0.0f);
        grad_final[0] = 1.0f;

        DBSBackwardHandle* sparse = nullptr;
        t0 = std::chrono::high_resolution_clock::now();
        rc = dbs_backward(h, r, nullptr, nullptr, grad_final.data(), &sparse);
        sparse_us += us_since(t0);
        if (rc != 0) {
            std::cerr << dbs_last_error(h) << "\n";
            std::exit(3);
        }
        sparse_nnz += dbs_backward_sparse_logprob_count(sparse);
        dbs_free_backward(sparse);

        if (static_cast<int64_t>(T) * K * V <= 1000000) {
            DBSBackwardHandle* dense = nullptr;
            t0 = std::chrono::high_resolution_clock::now();
            rc = dbs_backward_dense(h, r, nullptr, nullptr, grad_final.data(), &dense);
            dense_us += us_since(t0);
            if (rc != 0) {
                std::cerr << dbs_last_error(h) << "\n";
                std::exit(4);
            }
            dbs_free_backward(dense);
        } else {
            dense_us = -repeats;
        }

        dbs_free_result(r);

        DBSBatchResultHandle* br = nullptr;
        t0 = std::chrono::high_resolution_clock::now();
        rc = dbs_decode_batch(h, batched.data(), B, T, V, 0, &br);
        batch_us += us_since(t0);
        if (rc != 0) {
            std::cerr << dbs_last_error(h) << "\n";
            std::exit(5);
        }
        dbs_free_batch_result(br);
    }

    DBSStatsC stats{};
    dbs_get_stats(h, &stats);

    std::cout << T << ',' << K << ',' << V << ',' << B << ',' << repeats << ','
              << (fwd_us / repeats) << ','
              << (sparse_us / repeats) << ','
              << (dense_us < 0 ? -1 : dense_us / repeats) << ','
              << (batch_us / repeats) << ','
              << (sparse_nnz / repeats) << ','
              << dbs_selected_kernel_name() << ','
              << dbs_has_avx512() << ',' << dbs_has_avx2() << ',' << dbs_has_sse42() << ',' << dbs_has_neon() << '\n';
    dbs_destroy(h);
}

int main(int argc, char** argv) {
    if (argc == 6) {
        std::cout << "T,K,V,B,repeats,forward_us,sparse_backward_us,dense_backward_us,batch_forward_us,sparse_nnz,kernel,avx512,avx2,sse42,neon\n";
        run_one(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
        return 0;
    }

    const int repeats = argc > 1 ? std::atoi(argv[1]) : 10;
    std::cout << "T,K,V,B,repeats,forward_us,sparse_backward_us,dense_backward_us,batch_forward_us,sparse_nnz,kernel,avx512,avx2,sse42,neon\n";
    for (int T : {4, 8, 16}) {
        for (int K : {2, 4, 8}) {
            for (int V : {1000, 8000, 32000}) {
                for (int B : {1, 4}) {
                    run_one(T, K, V, B, repeats);
                }
            }
        }
    }
    return 0;
}
