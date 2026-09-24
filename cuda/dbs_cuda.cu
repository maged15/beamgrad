// SPDX-License-Identifier: MIT
//
// Native CUDA beam search (see include/dbs_cuda.h for the contract).
//
// Algorithm, per decode step t:
//
//   1. scan_tiles_kernel: the K*V candidate space of every example is split
//      into tiles of kTile candidates, one thread block per (example, tile).
//      Each candidate (parent beam p, token v) gets a 64-bit key
//          ordered(raw score) << 32 | ~(p * V + v)
//      and the block keeps the K largest keys of its tile (bitonic sort).
//   2. reduce_tiles_kernel: tile winners are reduced the same way until at
//      most kTile keys per example remain.
//   3. select_step_kernel: one block per example sorts the survivors, merges
//      them with the EOS carry-forward candidates, and writes the step's beams.
//
// Why a 64-bit key reproduces the CPU order exactly: the CPU ranks candidates
// by (score, raw, parent, token, length, origin). Every scanned candidate of a
// step extends a live beam, and all live beams have length t, so all of them
// share one positive length-penalty factor. The score is therefore a
// monotone function of the raw score, and ordering by (raw desc, index asc) is
// the same total order. Carry-forward candidates (finished beams, other
// lengths) are few and are merged with the full comparator in step 3.
//
// The backward pass walks each example's selected paths from the last step to
// the first, exactly like the CPU sparse backward, with one thread per example
// so accumulation order (and therefore every bit of the result) is fixed.
#include "dbs_cuda.h"

#if defined(DBS_CUDA_EMULATION)
#include "emulation/cuda_emulation.hpp"
#define DBS_LAUNCH(kernel, grid, block, stream, ...) ::dbs_emu::launch(kernel, grid, block, stream, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#define DBS_LAUNCH(kernel, grid, block, stream, ...) kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 256;                 // threads per block for tile/select kernels
constexpr int kTile = 4096;                   // candidates sorted per block
constexpr int kMaxBeam = DBS_CUDA_MAX_BEAM;
constexpr int kBackwardThreads = 128;         // examples per backward block
constexpr int64_t kAlign = 256;               // workspace sub-buffer alignment

static_assert(kTile % kThreads == 0, "tile must be a multiple of the block size");
static_assert(2 * kMaxBeam <= kTile, "tile reduction must shrink the candidate set");

std::atomic<int> g_synchronize{0};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000u); }

__device__ __forceinline__ bool is_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}

// Same formula as dbs::gnmt_length_penalty in src/common.hpp: computed in double
// and rounded once, so it matches the host bit for bit (a float powf would not).
// Called a handful of times per beam and step, so double throughput is irrelevant.
__device__ __forceinline__ float length_penalty(int length, float alpha) {
    if (alpha == 0.0f) return 1.0f;
    const int l = length > 1 ? length : 1;
    return static_cast<float>(pow((5.0 + static_cast<double>(l)) / 6.0, static_cast<double>(alpha)));
}

__device__ __forceinline__ uint64_t make_key(float raw, uint32_t index) {
    const uint32_t u = __float_as_uint(__fadd_rn(raw, 0.0f));  // -0.0 -> +0.0
    const uint32_t ordered = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<uint64_t>(ordered) << 32) | static_cast<uint64_t>(0xffffffffu - index);
}

__device__ __forceinline__ float key_raw(uint64_t key) {
    const uint32_t ordered = static_cast<uint32_t>(key >> 32);
    return __uint_as_float((ordered & 0x80000000u) ? (ordered & 0x7fffffffu) : ~ordered);
}

__device__ __forceinline__ uint32_t key_index(uint64_t key) {
    return 0xffffffffu - static_cast<uint32_t>(key);
}

__host__ __device__ __forceinline__ int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

struct Meta {
    int steps;
    int beam;
    int eos;
    int min_length;
};

__device__ __forceinline__ Meta load_meta(const DBSCudaDecodeArgs& a, int b) {
    Meta m;
    m.steps = a.steps_per_example ? a.steps_per_example[b] : a.steps;
    m.beam = a.beam_sizes_per_example ? a.beam_sizes_per_example[b] : a.beam_size;
    m.eos = a.eos_tokens_per_example ? a.eos_tokens_per_example[b] : a.eos_token;
    m.min_length = a.min_lengths_per_example ? a.min_lengths_per_example[b] : a.min_length;
    return m;
}

// Descending bitonic sort of n (a power of two) keys in shared memory.
// Must be called by every thread of the block.
__device__ void bitonic_sort_desc(uint64_t* keys, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
                const int partner = i ^ j;
                if (partner > i) {
                    const uint64_t a = keys[i];
                    const uint64_t b = keys[partner];
                    const bool descending = (i & k) == 0;
                    if (descending ? (a < b) : (a > b)) {
                        keys[i] = b;
                        keys[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Candidate as ranked by the CPU decoder (dbs::candidate_better).
struct Cand {
    float score;
    float raw;
    int parent;
    int token;
    int length;
    int from_logprob;
};

__device__ __forceinline__ bool better(const Cand& a, const Cand& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.raw != b.raw) return a.raw > b.raw;
    if (a.parent != b.parent) return a.parent < b.parent;
    if (a.token != b.token) return a.token < b.token;
    if (a.length != b.length) return a.length < b.length;
    return a.from_logprob > b.from_logprob;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void validate_meta_kernel(DBSCudaDecodeArgs a, int* status) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= a.batch_size) return;
    const Meta m = load_meta(a, b);
    const bool ok = m.steps >= 1 && m.steps <= a.steps &&
                    m.beam >= 1 && m.beam <= a.beam_size &&
                    m.eos >= -1 && m.eos < a.vocab_size &&
                    m.min_length >= 0;
    if (!ok) atomicCAS(status, DBS_CUDA_STATUS_OK, DBS_CUDA_STATUS_INVALID_ARGUMENT);
}

__global__ void init_state_kernel(DBSCudaDecodeArgs a, float* beam_raw, int32_t* beam_len, uint8_t* beam_ended) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(a.batch_size) * a.beam_size;
    if (i >= total) return;
    const int b = static_cast<int>(i / a.beam_size);
    const int k = static_cast<int>(i % a.beam_size);
    const Meta m = load_meta(a, b);
    beam_raw[i] = (k == 0 && k < m.beam) ? 0.0f : neg_inf();
    beam_len[i] = 0;
    beam_ended[i] = 0;
}

// grid = (B, tiles). Writes the K best keys of each tile (0 = no candidate).
__global__ void scan_tiles_kernel(
    const float* __restrict__ log_probs,
    DBSCudaDecodeArgs a,
    int t,
    const float* __restrict__ beam_raw,
    const int32_t* __restrict__ beam_len,
    const uint8_t* __restrict__ beam_ended,
    uint64_t* __restrict__ out_keys,
    int tiles) {
    __shared__ uint64_t keys[kTile];
    __shared__ int any_valid;

    const int b = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int K = a.beam_size;
    const int V = a.vocab_size;
    const Meta m = load_meta(a, b);
    const int64_t base = static_cast<int64_t>(tile) * kTile;
    const int64_t remaining = static_cast<int64_t>(K) * V - base;
    const int count = static_cast<int>(remaining < kTile ? remaining : kTile);
    const int n = next_pow2(count);
    const bool step_active = t < m.steps;

    if (threadIdx.x == 0) any_valid = 0;
    __syncthreads();

    int mine = 0;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        uint64_t key = 0;
        if (step_active && i < count) {
            const int64_t c = base + i;
            const int parent = static_cast<int>(c / V);
            const int token = static_cast<int>(c - static_cast<int64_t>(parent) * V);
            if (parent < m.beam) {
                const int64_t s = static_cast<int64_t>(b) * K + parent;
                const float parent_raw = beam_raw[s];
                const bool ended = m.eos >= 0 && beam_ended[s] != 0;
                const bool eos_masked = m.eos >= 0 && token == m.eos && beam_len[s] + 1 < m.min_length;
                if (is_finite(parent_raw) && !ended && !eos_masked) {
                    const float lp = log_probs[((static_cast<int64_t>(b) * a.steps + t) * K + parent) * V + token];
                    if (is_finite(lp)) {
                        const float raw = __fadd_rn(parent_raw, lp);
                        if (raw != neg_inf()) key = make_key(raw, static_cast<uint32_t>(c));
                    }
                }
            }
        }
        keys[i] = key;
        mine |= key != 0 ? 1 : 0;
    }
    if (mine) atomicOr(&any_valid, 1);
    __syncthreads();

    const bool active = any_valid != 0;
    if (active) bitonic_sort_desc(keys, n);

    uint64_t* out = out_keys + (static_cast<int64_t>(b) * tiles + tile) * K;
    for (int i = static_cast<int>(threadIdx.x); i < K; i += static_cast<int>(blockDim.x)) {
        out[i] = (active && i < n) ? keys[i] : 0;
    }
}

// grid = (B, tiles_out). Keeps the K best of each kTile slice of in_keys.
__global__ void reduce_tiles_kernel(
    const uint64_t* __restrict__ in_keys,
    int n_in,
    int K,
    uint64_t* __restrict__ out_keys,
    int tiles_out) {
    __shared__ uint64_t keys[kTile];
    __shared__ int any_valid;

    const int b = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int base = tile * kTile;
    const int count = (n_in - base) < kTile ? (n_in - base) : kTile;
    const int n = next_pow2(count);
    const uint64_t* in = in_keys + static_cast<int64_t>(b) * n_in + base;

    if (threadIdx.x == 0) any_valid = 0;
    __syncthreads();

    int mine = 0;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        const uint64_t key = i < count ? in[i] : 0;
        keys[i] = key;
        mine |= key != 0 ? 1 : 0;
    }
    if (mine) atomicOr(&any_valid, 1);
    __syncthreads();

    const bool active = any_valid != 0;
    if (active) bitonic_sort_desc(keys, n);

    uint64_t* out = out_keys + (static_cast<int64_t>(b) * tiles_out + tile) * K;
    for (int i = static_cast<int>(threadIdx.x); i < K; i += static_cast<int>(blockDim.x)) {
        out[i] = (active && i < n) ? keys[i] : 0;
    }
}

struct StepOutputs {
    int32_t* tokens;
    int32_t* parents;
    int32_t* lengths;
    float* scores;
    float* raw_scores;
    uint8_t* from_logprob;
};

__device__ __forceinline__ void write_slot(const StepOutputs& o, int64_t slot, const Cand& c) {
    if (o.tokens) o.tokens[slot] = c.token;
    if (o.parents) o.parents[slot] = c.parent;
    if (o.lengths) o.lengths[slot] = c.length;
    if (o.scores) o.scores[slot] = c.score;
    if (o.raw_scores) o.raw_scores[slot] = c.raw;
    if (o.from_logprob) o.from_logprob[slot] = static_cast<uint8_t>(c.from_logprob);
}

__device__ __forceinline__ Cand empty_slot() {
    Cand c;
    c.score = neg_inf();
    c.raw = neg_inf();
    c.parent = -1;
    c.token = -1;
    c.length = 0;
    c.from_logprob = 0;
    return c;
}

struct SelectShared {
    float state_raw[kMaxBeam];
    int32_t state_len[kMaxBeam];
    uint8_t state_ended[kMaxBeam];
    int n_scanned;
    int n_carry;
    union {
        uint64_t keys[kTile];
        struct {
            float s_score[kMaxBeam];
            float s_raw[kMaxBeam];
            int32_t s_index[kMaxBeam];
            float c_score[kMaxBeam];
            float c_raw[kMaxBeam];
            int32_t c_parent[kMaxBeam];
            int32_t c_len[kMaxBeam];
        } m;
    } u;
};

// grid = (B). Selects the step's beams and advances the beam state.
__global__ void select_step_kernel(
    DBSCudaDecodeArgs a,
    int t,
    const uint64_t* __restrict__ in_keys,
    int n_in,
    float* __restrict__ beam_raw,
    int32_t* __restrict__ beam_len,
    uint8_t* __restrict__ beam_ended,
    StepOutputs out) {
    __shared__ SelectShared sh;

    const int b = static_cast<int>(blockIdx.x);
    const int K = a.beam_size;
    const int V = a.vocab_size;
    const Meta m = load_meta(a, b);
    const int64_t slot_base = (static_cast<int64_t>(b) * a.steps + t) * K;

    if (t >= m.steps) {
        // Finished example: no state change, empty step outputs.
        for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
            write_slot(out, slot_base + k, empty_slot());
        }
        return;
    }

    for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
        const int64_t s = static_cast<int64_t>(b) * K + k;
        sh.state_raw[k] = beam_raw[s];
        sh.state_len[k] = beam_len[s];
        sh.state_ended[k] = beam_ended[s];
    }
    const int n = next_pow2(n_in);
    const uint64_t* in = in_keys + static_cast<int64_t>(b) * n_in;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        sh.u.keys[i] = i < n_in ? in[i] : 0;
    }
    __syncthreads();
    bitonic_sort_desc(sh.u.keys, n);

    // Keep this thread's share of the top m.beam keys in registers, then reuse
    // the key buffer for the merge arrays.
    constexpr int kPerThread = kMaxBeam / kThreads;
    uint64_t mine[kPerThread];
    for (int r = 0; r < kPerThread; ++r) {
        const int i = static_cast<int>(threadIdx.x) + r * kThreads;
        mine[r] = (i < m.beam && i < n) ? sh.u.keys[i] : 0;
    }
    __syncthreads();

    for (int r = 0; r < kPerThread; ++r) {
        const int i = static_cast<int>(threadIdx.x) + r * kThreads;
        if (i >= m.beam) continue;
        if (mine[r] == 0) {
            sh.u.m.s_index[i] = -1;
            continue;
        }
        const float raw = key_raw(mine[r]);
        const int index = static_cast<int>(key_index(mine[r]));
        const int parent = index / V;
        const float inv_penalty = __fdiv_rn(1.0f, length_penalty(sh.state_len[parent] + 1, a.length_penalty_alpha));
        sh.u.m.s_score[i] = __fmul_rn(raw, inv_penalty);
        sh.u.m.s_raw[i] = raw;
        sh.u.m.s_index[i] = index;
    }
    if (threadIdx.x == 0) {
        int n_carry = 0;
        if (m.eos >= 0) {
            for (int p = 0; p < m.beam; ++p) {
                const float raw = sh.state_raw[p];
                if (sh.state_ended[p] && is_finite(raw)) {
                    const int len = sh.state_len[p] > 1 ? sh.state_len[p] : 1;
                    sh.u.m.c_score[n_carry] = __fdiv_rn(raw, length_penalty(len, a.length_penalty_alpha));
                    sh.u.m.c_raw[n_carry] = raw;
                    sh.u.m.c_parent[n_carry] = p;
                    sh.u.m.c_len[n_carry] = len;
                    ++n_carry;
                }
            }
        }
        sh.n_carry = n_carry;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n_scanned = 0;
        while (n_scanned < m.beam && sh.u.m.s_index[n_scanned] >= 0) ++n_scanned;
        sh.n_scanned = n_scanned;
    }
    __syncthreads();

    const int n_scanned = sh.n_scanned;
    const int n_carry = sh.n_carry;
    const int total = n_scanned + n_carry;

    auto scanned = [&](int i) {
        Cand c;
        const int index = sh.u.m.s_index[i];
        c.score = sh.u.m.s_score[i];
        c.raw = sh.u.m.s_raw[i];
        c.parent = index / V;
        c.token = index - c.parent * V;
        c.length = sh.state_len[c.parent] + 1;
        c.from_logprob = 1;
        return c;
    };
    auto carried = [&](int j) {
        Cand c;
        c.score = sh.u.m.c_score[j];
        c.raw = sh.u.m.c_raw[j];
        c.parent = sh.u.m.c_parent[j];
        c.token = m.eos;
        c.length = sh.u.m.c_len[j];
        c.from_logprob = 0;
        return c;
    };
    auto emit = [&](int pos, const Cand& c) {
        if (pos >= m.beam) return;
        write_slot(out, slot_base + pos, c);
        const int64_t s = static_cast<int64_t>(b) * K + pos;
        beam_raw[s] = c.raw;
        beam_len[s] = c.length;
        beam_ended[s] = static_cast<uint8_t>(c.from_logprob ? (c.token == m.eos && m.eos >= 0) : 1);
    };

    // Scanned candidates are already in rank order; carry-forward candidates are
    // ranked by counting. Every merged position is unique.
    for (int i = static_cast<int>(threadIdx.x); i < n_scanned; i += static_cast<int>(blockDim.x)) {
        const Cand c = scanned(i);
        int pos = i;
        for (int j = 0; j < n_carry; ++j) pos += better(carried(j), c) ? 1 : 0;
        emit(pos, c);
    }
    for (int j = static_cast<int>(threadIdx.x); j < n_carry; j += static_cast<int>(blockDim.x)) {
        const Cand c = carried(j);
        int pos = 0;
        for (int jj = 0; jj < n_carry; ++jj) pos += (jj != j && better(carried(jj), c)) ? 1 : 0;
        for (int i = 0; i < n_scanned; ++i) pos += better(scanned(i), c) ? 1 : 0;
        emit(pos, c);
    }
    const int filled = total < m.beam ? total : m.beam;
    for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
        if (k < filled) continue;
        write_slot(out, slot_base + k, empty_slot());
        const int64_t s = static_cast<int64_t>(b) * K + k;
        beam_raw[s] = neg_inf();
        beam_len[s] = 0;
        beam_ended[s] = 0;
    }
}

__global__ void finalize_kernel(
    DBSCudaDecodeArgs a,
    const float* __restrict__ beam_raw,
    const int32_t* __restrict__ beam_len,
    float* __restrict__ final_scores,
    float* __restrict__ final_raw_scores,
    int32_t* __restrict__ final_lengths) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(a.batch_size) * a.beam_size;
    if (i >= total) return;
    const int b = static_cast<int>(i / a.beam_size);
    const int k = static_cast<int>(i % a.beam_size);
    const Meta m = load_meta(a, b);
    float score = neg_inf();
    float raw = neg_inf();
    int length = 0;
    if (k < m.beam) {
        raw = beam_raw[i];
        length = beam_len[i];
        score = __fdiv_rn(raw, length_penalty(length, a.length_penalty_alpha));
    }
    final_scores[i] = score;
    if (final_raw_scores) final_raw_scores[i] = raw;
    if (final_lengths) final_lengths[i] = length;
}

// One thread per example; mirrors the CPU sparse backward with only
// grad_final_scores supplied.
__global__ void backward_kernel(
    DBSCudaDecodeArgs a,
    const int32_t* __restrict__ parents,
    const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ lengths,
    const uint8_t* __restrict__ from_logprob,
    const float* __restrict__ grad_final_scores,
    float* __restrict__ grad_log_probs,
    float* __restrict__ scratch) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= a.batch_size) return;
    const int K = a.beam_size;
    const int V = a.vocab_size;
    const Meta m = load_meta(a, b);
    float* next = scratch + static_cast<int64_t>(b) * 2 * K;
    float* prev = next + K;
    for (int k = 0; k < m.beam; ++k) next[k] = 0.0f;

    for (int t = m.steps - 1; t >= 0; --t) {
        for (int k = 0; k < m.beam; ++k) prev[k] = 0.0f;
        for (int k = 0; k < m.beam; ++k) {
            const int64_t slot = (static_cast<int64_t>(b) * a.steps + t) * K + k;
            const int parent = parents[slot];
            if (parent < 0 || parent >= m.beam) continue;  // also rejects malformed traces
            const float drank = (t == m.steps - 1) ? grad_final_scores[static_cast<int64_t>(b) * K + k] : 0.0f;
            const float inv_penalty = __fdiv_rn(1.0f, length_penalty(lengths[slot], a.length_penalty_alpha));
            const float draw = __fadd_rn(next[k], __fmul_rn(drank, inv_penalty));
            if (draw == 0.0f) continue;
            prev[parent] = __fadd_rn(prev[parent], draw);
            const int token = tokens[slot];
            if (from_logprob[slot] && token >= 0 && token < V) {
                const int64_t g = ((static_cast<int64_t>(b) * a.steps + t) * K + parent) * V + token;
                grad_log_probs[g] = __fadd_rn(grad_log_probs[g], draw);
            }
        }
        float* tmp = next;
        next = prev;
        prev = tmp;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

int64_t align_up(int64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

bool sync_requested() {
    if (g_synchronize.load(std::memory_order_relaxed) != 0) return true;
    const char* v = std::getenv("DBS_CUDA_SYNC_CHECK");
    const char* legacy = std::getenv("DBS_CUDA_DEBUG_SYNC");
    return (v && v[0] == '1') || (legacy && legacy[0] == '1');
}

int finish(cudaStream_t stream) {
    if (cudaGetLastError() != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    if (sync_requested()) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
        if (cudaGetLastError() != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    }
    return DBS_CUDA_STATUS_OK;
}

bool has_per_example(const DBSCudaDecodeArgs& a) {
    return a.steps_per_example || a.beam_sizes_per_example || a.eos_tokens_per_example || a.min_lengths_per_example;
}

bool valid_args(const DBSCudaDecodeArgs* a) {
    if (!a) return false;
    if (a->batch_size <= 0 || a->steps <= 0 || a->beam_size <= 0 || a->vocab_size <= 0) return false;
    if (a->beam_size > kMaxBeam) return false;
    if (a->eos_token < -1 || a->eos_token >= a->vocab_size) return false;
    if (a->min_length < 0) return false;
    if (!std::isfinite(a->length_penalty_alpha) || a->length_penalty_alpha < 0.0f) return false;
    if (a->reserved0 != 0) return false;
    // Candidate indices p * V + v must fit in int32, and the tile count must fit
    // in gridDim.y (65535 tiles of kTile candidates, about 268M candidates).
    const int64_t candidates = static_cast<int64_t>(a->beam_size) * a->vocab_size;
    if (candidates > std::numeric_limits<int32_t>::max()) return false;
    if (ceil_div(candidates, kTile) > 65535) return false;
    // All tensor element counts must fit in int64.
    const double elements = static_cast<double>(a->batch_size) * a->steps * a->beam_size * a->vocab_size;
    return elements < 9.0e18;
}

struct DecodePlan {
    int tiles0;
    int64_t keys0_bytes;
    int64_t keys1_bytes;
    int64_t state_raw_offset;
    int64_t state_len_offset;
    int64_t state_ended_offset;
    int64_t keys0_offset;
    int64_t keys1_offset;
    int64_t status_offset;
    int64_t total_bytes;
};

DecodePlan make_plan(const DBSCudaDecodeArgs& a) {
    DecodePlan p{};
    const int64_t B = a.batch_size;
    const int64_t K = a.beam_size;
    p.tiles0 = static_cast<int>(ceil_div(K * a.vocab_size, kTile));
    const int64_t n0 = static_cast<int64_t>(p.tiles0) * K;
    const int64_t n1 = n0 > kTile ? ceil_div(n0, kTile) * K : 0;
    p.keys0_bytes = B * n0 * static_cast<int64_t>(sizeof(uint64_t));
    p.keys1_bytes = B * n1 * static_cast<int64_t>(sizeof(uint64_t));
    int64_t off = 0;
    p.state_raw_offset = off;
    off = align_up(off + B * K * static_cast<int64_t>(sizeof(float)));
    p.state_len_offset = off;
    off = align_up(off + B * K * static_cast<int64_t>(sizeof(int32_t)));
    p.state_ended_offset = off;
    off = align_up(off + B * K);
    p.keys0_offset = off;
    off = align_up(off + p.keys0_bytes);
    p.keys1_offset = off;
    off = align_up(off + p.keys1_bytes);
    p.status_offset = off;
    off = align_up(off + static_cast<int64_t>(sizeof(int)));
    p.total_bytes = off;
    return p;
}

int validate_per_example(const DBSCudaDecodeArgs& a, int* device_status, cudaStream_t stream) {
    if (!has_per_example(a)) return DBS_CUDA_STATUS_OK;
    if (cudaMemsetAsync(device_status, 0, sizeof(int), stream) != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    const int blocks = static_cast<int>(ceil_div(a.batch_size, kThreads));
    DBS_LAUNCH(validate_meta_kernel, dim3(blocks), dim3(kThreads), stream, a, device_status);
    int host_status = DBS_CUDA_STATUS_LAUNCH_FAILED;
    if (cudaMemcpyAsync(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return DBS_CUDA_STATUS_LAUNCH_FAILED;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        return DBS_CUDA_STATUS_LAUNCH_FAILED;
    }
    return host_status == DBS_CUDA_STATUS_OK ? DBS_CUDA_STATUS_OK : DBS_CUDA_STATUS_INVALID_ARGUMENT;
}

// Owns scratch memory when the caller did not provide a workspace.
class Workspace {
public:
    Workspace(void* user, int64_t user_bytes, int64_t needed, cudaStream_t stream) : stream_(stream) {
        if (user) {
            ptr_ = static_cast<char*>(user);
            ok_ = user_bytes >= needed;
            status_ = ok_ ? DBS_CUDA_STATUS_OK : DBS_CUDA_STATUS_INVALID_ARGUMENT;
            return;
        }
        void* p = nullptr;
        if (cudaMallocAsync(&p, static_cast<size_t>(needed > 0 ? needed : 1), stream) != cudaSuccess) {
            cudaGetLastError();
            status_ = DBS_CUDA_STATUS_OUT_OF_MEMORY;
            return;
        }
        ptr_ = static_cast<char*>(p);
        owned_ = true;
        ok_ = true;
    }
    ~Workspace() {
        if (owned_) cudaFreeAsync(ptr_, stream_);
    }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    bool ok() const { return ok_; }
    int status() const { return status_; }
    template <class T>
    T* at(int64_t offset) const { return reinterpret_cast<T*>(ptr_ + offset); }

private:
    cudaStream_t stream_;
    char* ptr_ = nullptr;
    bool owned_ = false;
    bool ok_ = false;
    int status_ = DBS_CUDA_STATUS_OK;
};

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" DBS_CUDA_EXPORT int dbs_cuda_available(void) {
    int count = 0;
    const bool ok = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    if (!ok) cudaGetLastError();
    return ok ? 1 : 0;
}

extern "C" DBS_CUDA_EXPORT const char* dbs_cuda_status_string(int status) {
    switch (status) {
        case DBS_CUDA_STATUS_OK: return "ok";
        case DBS_CUDA_STATUS_UNAVAILABLE: return "CUDA backend unavailable";
        case DBS_CUDA_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DBS_CUDA_STATUS_LAUNCH_FAILED: return "CUDA launch failed";
        case DBS_CUDA_STATUS_OUT_OF_MEMORY: return "CUDA out of memory";
        default: return "unknown CUDA status";
    }
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_set_synchronization(int synchronize) {
    if (synchronize != 0 && synchronize != 1) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    g_synchronize.store(synchronize, std::memory_order_relaxed);
    return DBS_CUDA_STATUS_OK;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_get_synchronization(void) {
    return g_synchronize.load(std::memory_order_relaxed);
}

extern "C" DBS_CUDA_EXPORT int64_t dbs_cuda_decode_workspace_size(const DBSCudaDecodeArgs* args) {
    if (!valid_args(args)) return -1;
    return make_plan(*args).total_bytes;
}

extern "C" DBS_CUDA_EXPORT int64_t dbs_cuda_backward_workspace_size(const DBSCudaDecodeArgs* args) {
    if (!valid_args(args)) return -1;
    return align_up(static_cast<int64_t>(args->batch_size) * 2 * args->beam_size * static_cast<int64_t>(sizeof(float)));
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode(
    const float* log_probs,
    const DBSCudaDecodeArgs* args,
    const DBSCudaDecodeOutputs* outputs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream_ptr) {
    if (!log_probs || !valid_args(args) || !outputs || !outputs->final_scores) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    const DBSCudaDecodeArgs& a = *args;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const DecodePlan plan = make_plan(a);
    Workspace ws(workspace, workspace_bytes, plan.total_bytes, stream);
    if (!ws.ok()) return ws.status();

    float* beam_raw = ws.at<float>(plan.state_raw_offset);
    int32_t* beam_len = ws.at<int32_t>(plan.state_len_offset);
    uint8_t* beam_ended = ws.at<uint8_t>(plan.state_ended_offset);
    uint64_t* keys0 = ws.at<uint64_t>(plan.keys0_offset);
    uint64_t* keys1 = ws.at<uint64_t>(plan.keys1_offset);

    const int rc = validate_per_example(a, ws.at<int>(plan.status_offset), stream);
    if (rc != DBS_CUDA_STATUS_OK) return rc;

    const int64_t beams = static_cast<int64_t>(a.batch_size) * a.beam_size;
    const dim3 beam_grid(static_cast<unsigned int>(ceil_div(beams, kThreads)));
    DBS_LAUNCH(init_state_kernel, beam_grid, dim3(kThreads), stream, a, beam_raw, beam_len, beam_ended);

    StepOutputs step_out;
    step_out.tokens = outputs->tokens;
    step_out.parents = outputs->parents;
    step_out.lengths = outputs->lengths;
    step_out.scores = outputs->scores;
    step_out.raw_scores = outputs->raw_scores;
    step_out.from_logprob = outputs->from_logprob;

    const unsigned int B = static_cast<unsigned int>(a.batch_size);
    for (int t = 0; t < a.steps; ++t) {
        DBS_LAUNCH(scan_tiles_kernel, dim3(B, static_cast<unsigned int>(plan.tiles0)), dim3(kThreads), stream,
                   log_probs, a, t, beam_raw, beam_len, beam_ended, keys0, plan.tiles0);
        int n = plan.tiles0 * a.beam_size;
        uint64_t* src = keys0;
        uint64_t* dst = keys1;
        while (n > kTile) {
            const int tiles = static_cast<int>(ceil_div(n, kTile));
            DBS_LAUNCH(reduce_tiles_kernel, dim3(B, static_cast<unsigned int>(tiles)), dim3(kThreads), stream,
                       src, n, a.beam_size, dst, tiles);
            n = tiles * a.beam_size;
            uint64_t* tmp = src;
            src = dst;
            dst = tmp;
        }
        DBS_LAUNCH(select_step_kernel, dim3(B), dim3(kThreads), stream,
                   a, t, src, n, beam_raw, beam_len, beam_ended, step_out);
    }

    DBS_LAUNCH(finalize_kernel, beam_grid, dim3(kThreads), stream,
               a, beam_raw, beam_len, outputs->final_scores, outputs->final_raw_scores, outputs->final_lengths);
    return finish(stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_backward(
    const DBSCudaDecodeArgs* args,
    const int32_t* parents,
    const int32_t* tokens,
    const int32_t* lengths,
    const uint8_t* from_logprob,
    const float* grad_final_scores,
    float* grad_log_probs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream_ptr) {
    if (!valid_args(args) || !parents || !tokens || !lengths || !from_logprob || !grad_final_scores || !grad_log_probs) {
        return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    }
    const DBSCudaDecodeArgs& a = *args;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    Workspace ws(workspace, workspace_bytes, dbs_cuda_backward_workspace_size(args), stream);
    if (!ws.ok()) return ws.status();
    const int blocks = static_cast<int>(ceil_div(a.batch_size, kBackwardThreads));
    DBS_LAUNCH(backward_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(kBackwardThreads), stream,
               a, parents, tokens, lengths, from_logprob, grad_final_scores, grad_log_probs, ws.at<float>(0));
    return finish(stream);
}
