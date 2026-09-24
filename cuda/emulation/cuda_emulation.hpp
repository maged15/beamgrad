// SPDX-License-Identifier: MIT
//
// A minimal, deterministic CPU emulation of the CUDA features used by
// cuda/dbs_cuda.cu, so the real kernel source can be executed and tested on
// machines without an NVIDIA GPU (including hosted CI runners).
//
// Each thread block runs as a set of cooperative fibers (ucontext) on the
// calling OS thread. __syncthreads() switches fibers; a block whose threads do
// not all reach the same barrier is reported as an error, as is any launch
// configuration CUDA would reject. Blocks run one after another, so
// `__shared__` variables can be emulated with function-local statics.
//
// Between two barriers the threads of a block run one after another, in an
// order set by dbs_emu::set_schedule(): forward, reverse, or shuffled (seeded,
// reshuffled at every barrier). A kernel whose result changes with the order
// has a data race between barriers; running the tests under several orders
// exposes it.
//
// This is a correctness tool only: it says nothing about performance, memory
// coalescing, or warp-level behaviour, and it is never part of a release build.
#pragma once

#if !defined(__linux__)
#error "The CUDA emulation layer requires Linux (POSIX ucontext)."
#endif

#include <ucontext.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
    constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

inline thread_local dim3 threadIdx;
inline thread_local dim3 blockIdx;
inline thread_local dim3 blockDim;
inline thread_local dim3 gridDim;

// ---------------------------------------------------------------------------
// Runtime API subset
// ---------------------------------------------------------------------------

using cudaStream_t = struct CUstream_st*;

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorLaunchFailure = 719,
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaDeviceAttr {
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
};

namespace dbs_emu {

inline cudaError_t& last_error() {
    static cudaError_t error = cudaSuccess;
    return error;
}

[[noreturn]] inline void fatal(const std::string& message) {
    std::fprintf(stderr, "cuda emulation: %s\n", message.c_str());
    std::abort();
}

} // namespace dbs_emu

inline const char* cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidConfiguration: return "invalid configuration argument";
        default: return "unspecified launch failure";
    }
}

inline cudaError_t cudaGetLastError() {
    const cudaError_t e = dbs_emu::last_error();
    dbs_emu::last_error() = cudaSuccess;
    return e;
}

inline cudaError_t cudaPeekAtLastError() { return dbs_emu::last_error(); }

inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = 1;
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int) {
    *value = attr == cudaDevAttrMaxGridDimX ? 2147483647 : 65535;
    return cudaSuccess;
}

template <class T>
inline cudaError_t cudaMalloc(T** ptr, size_t bytes) {
    void* p = std::calloc(1, bytes == 0 ? 1 : bytes);
    if (!p) return cudaErrorMemoryAllocation;
    *ptr = static_cast<T*>(p);
    return cudaSuccess;
}

template <class T>
inline cudaError_t cudaMallocAsync(T** ptr, size_t bytes, cudaStream_t) {
    return cudaMalloc(ptr, bytes);
}

inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t) { return cudaFree(ptr); }

inline cudaError_t cudaMemsetAsync(void* ptr, int value, size_t bytes, cudaStream_t = nullptr) {
    std::memset(ptr, value, bytes);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind, cudaStream_t = nullptr) {
    std::memmove(dst, src, bytes);
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

// ---------------------------------------------------------------------------
// Device intrinsics
// ---------------------------------------------------------------------------

inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }

inline unsigned int __float_as_uint(float x) {
    unsigned int u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

inline float __uint_as_float(unsigned int u) {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

// Fibers of a block never run concurrently, so plain read-modify-write is atomic.
inline int atomicOr(int* address, int value) {
    const int old = *address;
    *address = old | value;
    return old;
}

inline int atomicCAS(int* address, int compare, int value) {
    const int old = *address;
    if (old == compare) *address = value;
    return old;
}

namespace dbs_emu {

enum class Schedule { Forward, Reverse, Shuffled };

class BlockScheduler {
public:
    static constexpr size_t kStackBytes = 256 * 1024;

    void set_schedule(Schedule schedule, uint32_t seed) {
        schedule_ = schedule;
        seed_ = seed == 0 ? 1u : seed;
    }

    static BlockScheduler& instance() {
        static BlockScheduler scheduler;
        return scheduler;
    }

    void run(unsigned int threads, const std::function<void()>& body) {
        body_ = &body;
        ensure_fibers(threads);
        for (unsigned int i = 0; i < threads; ++i) {
            Fiber& f = fibers_[i];
            f.done = false;
            f.at_barrier = false;
            getcontext(&f.ctx);
            f.ctx.uc_stack.ss_sp = f.stack.get();
            f.ctx.uc_stack.ss_size = kStackBytes;
            f.ctx.uc_link = &scheduler_ctx_;
            makecontext(&f.ctx, reinterpret_cast<void (*)()>(&BlockScheduler::trampoline), 0);
        }
        order_.resize(threads);
        for (unsigned int i = 0; i < threads; ++i) order_[i] = i;
        for (;;) {
            unsigned int finished = 0;
            unsigned int waiting = 0;
            arrange(threads);
            for (unsigned int n = 0; n < threads; ++n) {
                const unsigned int i = order_[n];
                Fiber& f = fibers_[i];
                if (f.done) {
                    ++finished;
                    continue;
                }
                f.at_barrier = false;
                current_ = i;
                threadIdx = dim3(i, 0, 0);
                swapcontext(&scheduler_ctx_, &f.ctx);
                if (f.done) {
                    ++finished;
                } else {
                    ++waiting;
                }
            }
            if (finished == threads) break;
            if (finished != 0 && waiting != 0) {
                fatal("__syncthreads() reached by only part of a block (divergent barrier)");
            }
        }
        body_ = nullptr;
    }

    void barrier() {
        Fiber& f = fibers_[current_];
        f.at_barrier = true;
        swapcontext(&f.ctx, &scheduler_ctx_);
    }

private:
    struct Fiber {
        ucontext_t ctx{};
        std::unique_ptr<char[]> stack;
        bool done = false;
        bool at_barrier = false;
    };

    // The order in which the block's threads run until their next barrier.
    void arrange(unsigned int threads) {
        for (unsigned int i = 0; i < threads; ++i) {
            order_[i] = schedule_ == Schedule::Reverse ? threads - 1 - i : i;
        }
        if (schedule_ == Schedule::Shuffled) {
            for (unsigned int i = threads; i > 1; --i) {
                seed_ = seed_ * 1664525u + 1013904223u;
                const unsigned int j = (seed_ >> 8) % i;
                const unsigned int tmp = order_[i - 1];
                order_[i - 1] = order_[j];
                order_[j] = tmp;
            }
        }
    }

    static void trampoline() {
        BlockScheduler& s = instance();
        (*s.body_)();
        s.fibers_[s.current_].done = true;
        // Returning resumes uc_link (the scheduler).
    }

    void ensure_fibers(unsigned int threads) {
        while (fibers_.size() < threads) {
            Fiber f;
            f.stack.reset(new char[kStackBytes]);
            fibers_.push_back(std::move(f));
        }
    }

    std::vector<Fiber> fibers_;
    std::vector<unsigned int> order_;
    Schedule schedule_ = Schedule::Forward;
    uint32_t seed_ = 1;
    ucontext_t scheduler_ctx_{};
    const std::function<void()>* body_ = nullptr;
    unsigned int current_ = 0;
};

// Runs every block of a 1-D or 2-D grid of 1-D blocks, in order.
template <class Kernel, class... Args>
void launch(Kernel kernel, dim3 grid, dim3 block, cudaStream_t, Args... args) {
    if (block.x == 0 || block.x > 1024 || block.y != 1 || block.z != 1 ||
        grid.x == 0 || grid.y == 0 || grid.y > 65535 || grid.z != 1) {
        last_error() = cudaErrorInvalidConfiguration;
        return;
    }
    gridDim = grid;
    blockDim = block;
    const std::function<void()> body = [&]() { kernel(args...); };
    for (unsigned int by = 0; by < grid.y; ++by) {
        for (unsigned int bx = 0; bx < grid.x; ++bx) {
            blockIdx = dim3(bx, by, 0);
            BlockScheduler::instance().run(block.x, body);
        }
    }
}

inline void set_schedule(Schedule schedule, uint32_t seed = 1) { BlockScheduler::instance().set_schedule(schedule, seed); }

} // namespace dbs_emu

inline void __syncthreads() { dbs_emu::BlockScheduler::instance().barrier(); }
