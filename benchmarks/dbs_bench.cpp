// SPDX-License-Identifier: MIT
#include "dbs.h"

#include <algorithm>
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

// dbs_decode_model_steps_ex with a callback that only copies precomputed rows,
// so the time is the search's own work per step, including the [K, t] prefix
// matrix it hands to the model.
static int copy_rows(void* user_data, const DBSModelStepInfoC* info, float* out_log_probs) {
    const std::vector<float>& x = *static_cast<const std::vector<float>*>(user_data);
    const size_t block = static_cast<size_t>(info->beam_size) * static_cast<size_t>(info->vocab_size);
    std::copy(x.begin() + static_cast<ptrdiff_t>(info->step * block),
              x.begin() + static_cast<ptrdiff_t>((info->step + 1) * block), out_log_probs);
    return 0;
}

static void run_model_steps(int T, int K, int V, int repeats) {
    DBSDecoderHandle* h = nullptr;
    if (dbs_create_ex(options(K), &h) != 0) {
        std::cerr << dbs_last_global_error() << "\n";
        std::exit(1);
    }
    const std::vector<float> x = make_logits(T, K, V, 321 + T + K + V);
    long long best_us = -1;
    for (int i = 0; i < repeats; ++i) {
        DBSResultHandle* r = nullptr;
        const auto t0 = std::chrono::high_resolution_clock::now();
        const int rc = dbs_decode_model_steps_ex(h, copy_rows, const_cast<std::vector<float>*>(&x), 0, T, V, nullptr, &r);
        const long long us = us_since(t0);
        if (rc != 0) {
            std::cerr << dbs_last_error(h) << "\n";
            std::exit(6);
        }
        dbs_free_result(r);
        if (best_us < 0 || us < best_us) best_us = us;
    }
    std::cout << T << ',' << K << ',' << V << ',' << repeats << ',' << best_us << '\n';
    dbs_destroy(h);
}

// dbs_decode_batch_into with automatic threading (num_threads = 0) and with one
// thread, best of `repeats` each: where starting threads costs more than it saves.
static void run_batch_threads(int B, int T, int K, int V, int repeats) {
    DBSDecoderHandle* h = nullptr;
    if (dbs_create_ex(options(K), &h) != 0) {
        std::cerr << dbs_last_global_error() << "\n";
        std::exit(1);
    }
    const std::vector<float> one = make_logits(T, K, V, 99 + T + K + V);
    std::vector<float> x(static_cast<size_t>(B) * one.size());
    for (int b = 0; b < B; ++b) std::copy(one.begin(), one.end(), x.begin() + static_cast<ptrdiff_t>(b) * static_cast<ptrdiff_t>(one.size()));
    const size_t bk = static_cast<size_t>(B) * K;
    std::vector<float> final_scores(bk);
    DBSDecodeOutputsC out{};
    out.final_scores = final_scores.data();
    long long best[2] = {-1, -1};
    for (int i = 0; i < repeats; ++i) {
        for (int mode = 0; mode < 2; ++mode) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            if (dbs_decode_batch_into(h, x.data(), B, T, V, nullptr, nullptr, mode == 0 ? 0 : 1, &out) != 0) {
                std::cerr << dbs_last_error(h) << "\n";
                std::exit(7);
            }
            const long long us = us_since(t0);
            if (best[mode] < 0 || us < best[mode]) best[mode] = us;
        }
    }
    std::cout << B << ',' << T << ',' << K << ',' << V << ',' << static_cast<long long>(B) * T * K * V << ','
              << best[0] << ',' << best[1] << '\n';
    dbs_destroy(h);
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "batch-threads") {
        // dbs_bench batch-threads [repeats]: best-of-repeats microseconds, automatic vs one thread.
        const int repeats = argc > 2 ? std::atoi(argv[2]) : 50;
        std::cout << "B,T,K,V,work,auto_threads_best_us,one_thread_best_us\n";
        const int shapes[][4] = {{2, 4, 2, 64},   {4, 4, 4, 256},   {8, 8, 4, 256},    {8, 8, 4, 1000},
                                 {8, 16, 4, 2000}, {8, 16, 8, 4000}, {16, 16, 8, 8000}, {32, 32, 8, 32000}};
        for (const auto& s : shapes) run_batch_threads(s[0], s[1], s[2], s[3], repeats);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "model-steps") {
        // dbs_bench model-steps [repeats]: best-of-repeats microseconds per search.
        const int repeats = argc > 2 ? std::atoi(argv[2]) : 20;
        std::cout << "T,K,V,repeats,model_steps_best_us\n";
        for (int T : {64, 256, 1024}) {
            for (int K : {4, 16}) {
                run_model_steps(T, K, 32, repeats);
            }
        }
        return 0;
    }
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
