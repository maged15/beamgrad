// SPDX-License-Identifier: MIT
//
// Native CUDA beam search (see include/dbs_cuda.h for the contract).
//
// Algorithm, per decode step t:
//
//   0. constraint_kernel (only with n-gram blocking or a repetition penalty):
//      rebuilds, for every live beam, bitmaps of the tokens its prefix blocks
//      and penalises. (From logits, row_lse_kernel has computed the
//      logsumexp of every row before the first step, in the CPU's fixed order.)
//   1. A scan kernel turns the K*V candidates of every example into 64-bit keys
//          ordered(raw score) << 32 | ~(p * V + v)
//      and keeps the K largest per block, in no particular order: for K <= 16,
//      every thread keeps its own top K in registers over a chunk of
//      candidates (scan_small_kernel, with chunks sized so that even a single
//      example fills the GPU); for larger beams every thread holds kTile /
//      kThreads candidates (scan_tiles_kernel). The block then finds its K
//      largest keys with a radix select (block_top_k), which sorts nothing.
//      Both scans check every element of the rows they read for NaN and +inf,
//      reading 16-bit inputs as float32 and subtracting the row's logsumexp
//      from logits (candidate_key is the only place inputs are read).
//   2. reduce_tiles_kernel: block winners are reduced the same way until at
//      most kTile keys per example remain.
//   3. select_step_kernel: one block per example selects the K best survivors
//      the same way and sorts only those, merges them with the EOS
//      carry-forward candidates, writes the step's beams and advances the beam
//      state (and the beams' token prefixes).
//
// Keys are unique, so "the K largest keys" is a set that does not depend on
// how candidates are split into blocks or in which order a block emits them;
// sorting the K winners in step 3 makes the result deterministic.
//
// Why a 64-bit key reproduces the CPU order exactly: the CPU ranks candidates
// by (score, raw, parent, token, length, origin). Every scanned candidate of a
// step extends a live beam, and all live beams have length t, so all of them
// share one positive length-penalty factor. The score is therefore a
// monotone function of the raw score, and ordering by (raw desc, index asc) is
// the same total order. Carry-forward candidates (finished beams, other
// lengths) are few and are merged with the full comparator in step 3.
//
// All floating-point operations that feed results use explicitly rounded
// intrinsics (no fused multiply-add), and the length penalty comes from
// src/penalty.hpp, which the CPU decoder shares, so both backends produce the
// same bits.
//
// The backward pass walks each example's selected paths from the last step to
// the first with one block per example. Within a step every beam is handled by
// its own thread (their gradient entries are distinct), and each parent sums
// its children in slot order, the same order as the CPU backward. From logits,
// the path gradients are then grouped by row in token order (logit_rows_kernel)
// and every such row gets its log-softmax correction (logit_correction_kernel).
#include "dbs_cuda.h"

#if defined(DBS_CUDA_EMULATION)
#include "emulation/cuda_emulation.hpp"
#define DBS_LAUNCH(kernel, grid, block, stream, ...) ::dbs_emu::launch(kernel, grid, block, stream, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#define DBS_LAUNCH(kernel, grid, block, stream, ...) kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#endif

#include "half.hpp"
#include "penalty.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 256;                 // threads per block for scan/select kernels
constexpr int kTile = 4096;                   // candidates per block (large beams) and per reduction slice
constexpr int kTileItems = kTile / kThreads;  // keys per thread in the tile scan and the reduction
constexpr int kSmallBeam = 16;                // largest beam handled by the register top-k scan
constexpr int kSmallItemsMin = 4;             // candidates per thread in the register top-k scan ...
constexpr int kSmallItemsMax = 32;            // ... chosen in this range from the problem size
constexpr int kTargetBlocks = 512;            // scan blocks worth launching to fill a large GPU
constexpr int kMaxGridY = 65535;
constexpr int kMaxBeam = DBS_CUDA_MAX_BEAM;
constexpr int kBackwardThreads = 256;
constexpr int64_t kMaxStepBlocks = 65535;     // blocks of the backward's per-step kernels (each loops over steps)
constexpr int kCorrectionItems = 8;           // vocabulary entries per thread in logit_correction_kernel
constexpr int64_t kMaxLseBlocks = 1 << 20;    // blocks of row_lse_kernel (each loops over rows)
// With extra EOS tokens, the engine's internal copy of the arguments points
// banned_tokens at a [V] array of these flags and sets reserved0 (which callers
// must leave 0), so the scans read one array and ScanParams keeps its size.
constexpr uint8_t kFlagBanned = 1;
constexpr uint8_t kFlagEos = 2;
constexpr int64_t kAlign = 256;               // workspace sub-buffer alignment

static_assert(kTile % kThreads == 0, "tile must be a multiple of the block size");
static_assert(2 * kMaxBeam <= kTile, "tile reduction must shrink the candidate set");
static_assert(kThreads >= 16, "block_top_k clears its 16-bin histogram with the first 16 threads");
static_assert((kMaxBeam & (kMaxBeam - 1)) == 0, "select_step_kernel sorts next_pow2(beam) <= kMaxBeam keys");
static_assert(DBS_CUDA_DTYPE_F32 == static_cast<int>(dbs::DType::F32) &&
                  DBS_CUDA_DTYPE_F16 == static_cast<int>(dbs::DType::F16) &&
                  DBS_CUDA_DTYPE_BF16 == static_cast<int>(dbs::DType::BF16),
              "DBS_CUDA_DTYPE_* are dbs::DType values");

std::atomic<int> g_synchronize{0};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000u); }
__device__ __forceinline__ float pos_inf() { return __uint_as_float(0x7f800000u); }

__device__ __forceinline__ bool is_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ float length_penalty(int length, float alpha) {
    return dbs::gnmt_length_penalty(length, alpha);
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

// Whether `token` is one of the extra EOS tokens (see kFlagEos).
__device__ __forceinline__ bool extra_eos(const DBSCudaDecodeArgs& a, int token) {
    return a.reserved0 != 0 && a.banned_tokens && (a.banned_tokens[token] & kFlagEos) != 0;
}

// Bitonic sort of n (a power of two) keys in shared memory, descending or
// ascending. Must be called by every thread of the block.
template <class Key, bool kDescending>
__device__ void bitonic_sort(Key* keys, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
                const int partner = i ^ j;
                if (partner > i) {
                    const Key a = keys[i];
                    const Key b = keys[partner];
                    const bool first_half = (i & k) == 0;
                    const bool swap = kDescending == first_half ? (a < b) : (a > b);
                    if (swap) {
                        keys[i] = b;
                        keys[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Shared state of block_top_k.
struct TopKShared {
    int hist[16];
    int count;
    int remaining;
    int done;
    uint64_t prefix;
    uint64_t mask;
};

// Writes the k largest keys held by the block to out[0, k), in no particular
// order, padded with 0 (no candidate) when fewer than k keys are non-zero.
// Every thread holds ITEMS keys; non-zero keys must be unique (candidate keys
// are), so the k largest form one well-defined set. A radix select over 4-bit
// digits, most significant first, narrows down the k-th largest key until the
// keys above it plus its digit bucket are exactly k; nothing is sorted. Must
// be called by every thread of the block.
template <int ITEMS>
__device__ void block_top_k(const uint64_t (&keys)[ITEMS], int k, uint64_t* __restrict__ out, TopKShared& s) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) s.count = 0;
    __syncthreads();
    int nonzero = 0;
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) nonzero += keys[j] != 0 ? 1 : 0;
    if (nonzero) atomicAdd(&s.count, nonzero);
    __syncthreads();
    const bool take_all = s.count <= k;
    __syncthreads();  // every thread has read the count before it is reused below

    uint64_t prefix = 0;
    uint64_t mask = 0;
    if (!take_all) {
        if (tid == 0) {
            s.prefix = 0;
            s.mask = 0;
            s.remaining = k;
        }
        for (int shift = 60; shift >= 0; shift -= 4) {
            if (tid < 16) s.hist[tid] = 0;
            __syncthreads();
            prefix = s.prefix;
            mask = s.mask;
            // Count the digits of the keys still in the running; neighbouring
            // keys of a thread often share a digit, so runs are added at once.
            int digit = -1;
            int run = 0;
#pragma unroll
            for (int j = 0; j < ITEMS; ++j) {
                if ((keys[j] & mask) != prefix) continue;
                const int d = static_cast<int>((keys[j] >> shift) & 15u);
                if (d != digit) {
                    if (run) atomicAdd(&s.hist[digit], run);
                    digit = d;
                    run = 0;
                }
                ++run;
            }
            if (run) atomicAdd(&s.hist[digit], run);
            __syncthreads();
            if (tid == 0) {
                // The digit of the k-th largest key, and its rank within that digit's bucket.
                int remaining = s.remaining;
                int d = 15;
                while (d > 0 && s.hist[d] < remaining) remaining -= s.hist[d--];
                s.prefix |= static_cast<uint64_t>(d) << shift;
                s.mask |= static_cast<uint64_t>(15) << shift;
                s.remaining = remaining;
                s.done = s.hist[d] == remaining;  // the whole bucket belongs to the top k
            }
            __syncthreads();
            if (s.done) break;
        }
        prefix = s.prefix;
        mask = s.mask;
    }

    // Keys above the k-th largest key's bucket, and that bucket: exactly k keys.
    if (tid == 0) s.count = 0;
    __syncthreads();
    int take = 0;
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) take += (take_all ? keys[j] != 0 : (keys[j] & mask) >= prefix) ? 1 : 0;
    int pos = take ? atomicAdd(&s.count, take) : 0;
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const bool selected = take_all ? keys[j] != 0 : (keys[j] & mask) >= prefix;
        if (selected && pos < k) out[pos++] = keys[j];
    }
    __syncthreads();
    const int written = s.count < k ? s.count : k;
    for (int i = written + tid; i < k; i += static_cast<int>(blockDim.x)) out[i] = 0;
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

// Element i of inputs of type kType (a DBS_CUDA_DTYPE_*), as float32. A
// template parameter rather than a runtime switch, which the compiler does not
// always resolve in the unrolled loops.
template <int kType>
__device__ __forceinline__ float load_input(const void* inputs, int64_t i) {
    if constexpr (kType == DBS_CUDA_DTYPE_F16) {
        return dbs::f16_bits_to_float(static_cast<const uint16_t*>(inputs)[i]);
    } else if constexpr (kType == DBS_CUDA_DTYPE_BF16) {
        return dbs::bf16_bits_to_float(static_cast<const uint16_t*>(inputs)[i]);
    } else {
        return static_cast<const float*>(inputs)[i];
    }
}

// Everything a scan kernel needs besides the beam state. At most 128 bytes:
// a larger kernel parameter costs the scans about 30 registers (and the tile
// scan 50% of its speed) with CUDA 12.4. The input type is a template
// parameter of the scans instead (see candidate_key).
struct ScanParams {
    DBSCudaDecodeArgs a;
    const void* inputs;         // [B, T, K, V] of the scan's input type
    const float* row_lse;       // [B, steps, K] logsumexp of the rows (row_lse_kernel) from logits, or null
    const uint32_t* blocked;    // [B, K, words] n-gram bitmap, or null
    const uint32_t* penalised;  // [B, K, words] repetition bitmap, or null
    int words;                  // bitmap words per beam
    float log_penalty;          // logf(repetition_penalty), computed on the host
    uint8_t* invalid;           // [B] NaN/+inf flags, or null
};

// The key of candidate (parent, token) of example b at step t, or 0 if it is
// not a candidate. Sets `invalid` if the input is NaN or +inf. Only called
// for live, unfinished parents, so every element of their rows is checked.
// The inputs are of type kType (a DBS_CUDA_DTYPE_*); with kLogits they are
// logits and lp = x - logsumexp(row), as in dbs::row_log_prob (a row without
// a finite logit has no candidates). Both are template parameters so that
// each scan kernel is compiled for exactly one kind of input: a runtime switch
// in the unrolled scans costs registers and, for float32 log-probs, speed.
template <int kType, bool kLogits>
__device__ __forceinline__ uint64_t candidate_key(
    const ScanParams& p, int b, int t, int parent, int token, float parent_raw, int parent_len, const Meta& m,
    uint32_t index, bool& invalid) {
    const int K = p.a.beam_size;
    const int V = p.a.vocab_size;
    const int64_t i = ((static_cast<int64_t>(b) * p.a.steps + t) * K + parent) * V + token;
    const float x = load_input<kType>(p.inputs, i);
    if (!(x < pos_inf())) {
        invalid = true;
        return 0;
    }
    float lp = x;
    if constexpr (kLogits) {
        const float lse = p.row_lse[(static_cast<int64_t>(b) * p.a.steps + t) * K + parent];
        if (lse == neg_inf()) return 0;
        lp = __fsub_rn(x, lse);
    }
    if (lp == neg_inf()) return 0;
    const uint8_t flags = p.a.banned_tokens ? p.a.banned_tokens[token] : 0;
    // flags is the caller's banned mask, or kFlag* bits with extra EOS tokens.
    if (p.a.reserved0 != 0 ? (flags & kFlagBanned) != 0 : flags != 0) return 0;
    const bool eos = m.eos >= 0 && (token == m.eos || (p.a.reserved0 != 0 && (flags & kFlagEos) != 0));
    if (eos && parent_len + 1 < m.min_length) return 0;
    float value = lp;
    if (p.blocked) {
        const int64_t word = (static_cast<int64_t>(b) * K + parent) * p.words + (token >> 5);
        const uint32_t bit = 1u << (token & 31);
        if (p.blocked[word] & bit) return 0;
        if (p.penalised[word] & bit) value = __fsub_rn(lp, p.log_penalty);
    }
    const float raw = __fadd_rn(parent_raw, value);
    if (raw == neg_inf()) return 0;
    return make_key(raw, index);
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

__global__ void init_state_kernel(DBSCudaDecodeArgs a, float* beam_raw, int32_t* beam_len, uint8_t* beam_ended,
                                  int32_t* beam_last, uint8_t* invalid) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(a.batch_size) * a.beam_size;
    if (i >= total) return;
    const int b = static_cast<int>(i / a.beam_size);
    const int k = static_cast<int>(i % a.beam_size);
    const Meta m = load_meta(a, b);
    beam_raw[i] = (k == 0 && k < m.beam) ? 0.0f : neg_inf();
    beam_len[i] = 0;
    beam_ended[i] = 0;
    if (beam_last) beam_last[i] = -1;
    if (invalid && k == 0) invalid[b] = 0;
}

// Extra EOS tokens by value, so that no host memory has to outlive the launch.
struct ExtraEos {
    int count;
    int32_t tokens[DBS_CUDA_MAX_EXTRA_EOS];
};

// One thread per token: the kFlag* flags of the scans (see kFlagBanned).
__global__ void token_flags_kernel(int V, const uint8_t* __restrict__ banned, ExtraEos extra, uint8_t* __restrict__ flags) {
    const int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= V) return;
    uint8_t f = banned && banned[v] ? kFlagBanned : 0;
    for (int i = 0; i < extra.count; ++i) {
        if (extra.tokens[i] == v) f = static_cast<uint8_t>(f | kFlagEos);
    }
    flags[v] = f;
}

// Step API with extra EOS tokens: the last token of each beam's prefix (-1 when
// the prefix does not cover the hypothesis), which a finished beam carries.
__global__ void last_token_kernel(int64_t beams, const int32_t* __restrict__ prefixes, int stride,
                                  const int32_t* __restrict__ lengths, int32_t* __restrict__ last) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= beams) return;
    const int len = lengths[i];
    last[i] = prefixes && len >= 1 && len <= stride ? prefixes[i * stride + len - 1] : -1;
}

// Constraint state of one beam slot: its token prefix (double-buffered across
// steps), and the tokens whose bits are set in its bitmaps.
struct ConstraintState {
    const int32_t* prefix;      // [B, K, stride]
    const int32_t* prefix_len;  // [B, K]
    uint32_t* blocked;          // [B, K, words]
    uint32_t* penalised;        // [B, K, words]
    int32_t* marked;            // [B, K, 2 * stride] tokens with bits set
    int32_t* marked_count;      // [B, K]
    int stride;                 // prefix capacity per beam
    int words;
    int ngram;
    int penalise;
};

// grid = (B). Rebuilds each live beam's blocked and penalised token bitmaps
// from its prefix, with the same rules as the CPU decoder. Prefix tokens
// outside the vocabulary (only possible in a caller-supplied state, see
// dbs_cuda_decode_step) are ignored.
__global__ void constraint_kernel(DBSCudaDecodeArgs a, int t, ConstraintState c, const float* beam_raw) {
    const int b = static_cast<int>(blockIdx.x);
    const int K = a.beam_size;
    const int V = a.vocab_size;
    const Meta m = load_meta(a, b);
    if (t >= m.steps) return;
    for (int p = static_cast<int>(threadIdx.x); p < m.beam; p += static_cast<int>(blockDim.x)) {
        const int64_t slot = static_cast<int64_t>(b) * K + p;
        uint32_t* blocked = c.blocked + slot * c.words;
        uint32_t* penalised = c.penalised + slot * c.words;
        int32_t* marked = c.marked + slot * 2 * c.stride;
        // Clear the bits of the hypothesis that held this slot at the previous step.
        for (int i = 0; i < c.marked_count[slot]; ++i) {
            blocked[marked[i] >> 5] = 0;
            penalised[marked[i] >> 5] = 0;
        }
        int count = 0;
        if (is_finite(beam_raw[slot])) {
            const int32_t* prefix = c.prefix + slot * c.stride;
            const int length = c.prefix_len[slot];
            const int L = length < 0 ? 0 : (length > c.stride ? c.stride : length);
            const int n = c.ngram;
            if (n == 1) {
                for (int i = 0; i < L; ++i) {
                    if (prefix[i] < 0 || prefix[i] >= V) continue;
                    blocked[prefix[i] >> 5] |= 1u << (prefix[i] & 31);
                    marked[count++] = prefix[i];
                }
            } else if (n > 1 && L >= n - 1) {
                const int suffix_start = L - (n - 1);
                for (int i = 0; i + n <= L; ++i) {
                    bool same = true;
                    for (int j = 0; j < n - 1 && same; ++j) same = prefix[i + j] == prefix[suffix_start + j];
                    const int token = prefix[i + n - 1];
                    if (same && token >= 0 && token < V) {
                        blocked[token >> 5] |= 1u << (token & 31);
                        marked[count++] = token;
                    }
                }
            }
            if (c.penalise) {
                for (int i = 0; i < L; ++i) {
                    if (prefix[i] < 0 || prefix[i] >= V) continue;
                    penalised[prefix[i] >> 5] |= 1u << (prefix[i] & 31);
                    marked[count++] = prefix[i];
                }
            }
        }
        c.marked_count[slot] = count;
    }
}

// grid = (row blocks), block = dbs::kLogitLanes threads. The logsumexp of
// every row [b, t, k] of an example's active steps and beams, into lse
// [B, steps, K] (-inf for a row without a finite entry, and for the other
// rows), as dbs::logit_stats computes it: the maximum over finite entries,
// then thread j sums exp(x - max) over the entries v = j, j + kLogitLanes, ...
// in increasing v (lane j of the CPU's reduction), and the lanes are combined
// by the same pairwise tree. All rows at once: the rows a search step reads
// are only known once the previous step has run, but one block per row of a
// single step leaves most of the GPU idle. select_step_kernel reports the
// values of the rows the search read.
template <int kType>
__global__ void row_lse_kernel(DBSCudaDecodeArgs a, const void* __restrict__ inputs, float* __restrict__ lse) {
    __shared__ float lanes[dbs::kLogitLanes];

    const int tid = static_cast<int>(threadIdx.x);
    const int K = a.beam_size;
    const int V = a.vocab_size;
    const int64_t n_rows = static_cast<int64_t>(a.batch_size) * a.steps * K;
    for (int64_t r = blockIdx.x; r < n_rows; r += gridDim.x) {
        const int k = static_cast<int>(r % K);
        const int t = static_cast<int>((r / K) % a.steps);
        const Meta m = load_meta(a, static_cast<int>(r / (static_cast<int64_t>(K) * a.steps)));
        if (t >= m.steps || k >= m.beam) {
            if (tid == 0) lse[r] = neg_inf();
            continue;
        }
        const int64_t row = r * V;

        float max = neg_inf();
        for (int v = tid; v < V; v += dbs::kLogitLanes) {
            const float x = load_input<kType>(inputs, row + v);
            if (x < pos_inf() && x > max) max = x;
        }
        lanes[tid] = max;
        __syncthreads();
        for (int stride = dbs::kLogitLanes / 2; stride > 0; stride >>= 1) {
            if (tid < stride && lanes[tid + stride] > lanes[tid]) lanes[tid] = lanes[tid + stride];
            __syncthreads();
        }
        max = lanes[0];
        __syncthreads();  // every thread has read the maximum before the lanes are reused

        float value = neg_inf();
        if (max != neg_inf()) {
            float lane = 0.0f;
            for (int v = tid; v < V; v += dbs::kLogitLanes) {
                const float x = load_input<kType>(inputs, row + v);
                if (x < pos_inf() && x != neg_inf()) lane = dbs::det::add(lane, dbs::det::exp_nonpositive(dbs::det::sub(x, max)));
            }
            lanes[tid] = lane;
            __syncthreads();
            for (int stride = dbs::kLogitLanes / 2; stride > 0; stride >>= 1) {
                if (tid < stride) lanes[tid] = dbs::det::add(lanes[tid], lanes[tid + stride]);
                __syncthreads();
            }
            value = dbs::logsumexp_from(max, lanes[0]);
        }
        if (tid == 0) lse[r] = value;
        __syncthreads();  // lanes[0] has been read before the next row reuses the lanes
    }
}

// grid = (B, chunks), for beam_size <= KMAX <= kSmallBeam. Every thread keeps
// the KMAX best keys of its `items` candidates in registers, and the block
// writes the K best keys of its chunk (0 = no candidate). kType and kLogits:
// see candidate_key.
template <int KMAX, int kType, bool kLogits>
__global__ void scan_small_kernel(
    ScanParams p,
    int t,
    const float* __restrict__ beam_raw,
    const int32_t* __restrict__ beam_len,
    const uint8_t* __restrict__ beam_ended,
    uint64_t* __restrict__ out_keys,
    int chunks,
    int items) {
    __shared__ float parent_raw[KMAX];
    __shared__ int32_t parent_len[KMAX];
    __shared__ int expand[KMAX];
    __shared__ TopKShared topk;

    const int b = static_cast<int>(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    const int K = p.a.beam_size;
    const int V = p.a.vocab_size;
    const Meta m = load_meta(p.a, b);
    const bool step_active = t < m.steps;

    for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
        const int64_t s = static_cast<int64_t>(b) * K + k;
        parent_raw[k] = beam_raw[s];
        parent_len[k] = beam_len[s];
        expand[k] = step_active && k < m.beam && is_finite(beam_raw[s]) && !(m.eos >= 0 && beam_ended[s] != 0);
    }
    __syncthreads();

    uint64_t best[KMAX];
#pragma unroll
    for (int j = 0; j < KMAX; ++j) best[j] = 0;
    bool invalid = false;

    if (step_active) {
        const int64_t total = static_cast<int64_t>(K) * V;
        int64_t c = static_cast<int64_t>(chunk) * kThreads * items + threadIdx.x;
        int parent = static_cast<int>(c / V);
        int token = static_cast<int>(c - static_cast<int64_t>(parent) * V);
        for (int i = 0; i < items && c < total; ++i) {
            if (expand[parent]) {
                const uint64_t key = candidate_key<kType, kLogits>(p, b, t, parent, token, parent_raw[parent],
                                                                   parent_len[parent], m, static_cast<uint32_t>(c),
                                                                   invalid);
                if (key > best[KMAX - 1]) {
                    best[KMAX - 1] = key;
#pragma unroll
                    for (int j = KMAX - 1; j > 0; --j) {
                        if (best[j] > best[j - 1]) {
                            const uint64_t tmp = best[j];
                            best[j] = best[j - 1];
                            best[j - 1] = tmp;
                        }
                    }
                }
            }
            c += kThreads;
            token += kThreads;
            while (token >= V) {
                token -= V;
                ++parent;
            }
        }
    }
    if (invalid && p.invalid) p.invalid[b] = 1;

    // The chunk's K best keys are among the threads' K best.
    block_top_k(best, K, out_keys + (static_cast<int64_t>(b) * chunks + chunk) * K, topk);
}

// grid = (B, tiles). Writes the K best keys of each tile (0 = no candidate).
template <int kType, bool kLogits>
__global__ void scan_tiles_kernel(
    ScanParams p,
    int t,
    const float* __restrict__ beam_raw,
    const int32_t* __restrict__ beam_len,
    const uint8_t* __restrict__ beam_ended,
    uint64_t* __restrict__ out_keys,
    int tiles) {
    __shared__ TopKShared topk;

    const int b = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int K = p.a.beam_size;
    const int V = p.a.vocab_size;
    const Meta m = load_meta(p.a, b);
    const int64_t base = static_cast<int64_t>(tile) * kTile;
    const int64_t remaining = static_cast<int64_t>(K) * V - base;
    const int count = static_cast<int>(remaining < kTile ? remaining : kTile);
    const bool step_active = t < m.steps;

    uint64_t keys[kTileItems];
    bool invalid = false;
#pragma unroll
    for (int j = 0; j < kTileItems; ++j) {
        const int i = static_cast<int>(threadIdx.x) + j * kThreads;
        uint64_t key = 0;
        if (step_active && i < count) {
            const int64_t c = base + i;
            const int parent = static_cast<int>(c / V);
            const int token = static_cast<int>(c - static_cast<int64_t>(parent) * V);
            if (parent < m.beam) {
                const int64_t s = static_cast<int64_t>(b) * K + parent;
                const float parent_raw = beam_raw[s];
                const bool ended = m.eos >= 0 && beam_ended[s] != 0;
                if (is_finite(parent_raw) && !ended) {
                    key = candidate_key<kType, kLogits>(p, b, t, parent, token, parent_raw, beam_len[s], m,
                                                        static_cast<uint32_t>(c), invalid);
                }
            }
        }
        keys[j] = key;
    }
    if (invalid && p.invalid) p.invalid[b] = 1;

    block_top_k(keys, K, out_keys + (static_cast<int64_t>(b) * tiles + tile) * K, topk);
}

// grid = (B, tiles_out). Keeps the K best of each kTile slice of in_keys.
__global__ void reduce_tiles_kernel(
    const uint64_t* __restrict__ in_keys,
    int n_in,
    int K,
    uint64_t* __restrict__ out_keys,
    int tiles_out) {
    __shared__ TopKShared topk;

    const int b = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int base = tile * kTile;
    const int count = (n_in - base) < kTile ? (n_in - base) : kTile;
    const uint64_t* in = in_keys + static_cast<int64_t>(b) * n_in + base;

    uint64_t keys[kTileItems];
#pragma unroll
    for (int j = 0; j < kTileItems; ++j) {
        const int i = static_cast<int>(threadIdx.x) + j * kThreads;
        keys[j] = i < count ? in[i] : 0;
    }
    block_top_k(keys, K, out_keys + (static_cast<int64_t>(b) * tiles_out + tile) * K, topk);
}

struct StepOutputs {
    int32_t* tokens;
    int32_t* parents;
    int32_t* lengths;
    float* scores;
    float* raw_scores;
    uint8_t* from_logprob;
    const float* lse;  // [B, T, K] every row's logsumexp (from logits), or null
    float* row_lse;    // [B, T, K] output: lse of the rows the step reads, 0 for the others; or null
};

// Token prefixes of the beams (only with n-gram blocking or a repetition
// penalty): read from `prefix`, written for the next step to `next_prefix`.
struct PrefixBuffers {
    const int32_t* prefix;
    const int32_t* prefix_len;
    int32_t* next_prefix;
    int32_t* next_prefix_len;
    int stride;  // prefix capacity per beam
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
    TopKShared topk;
    union {
        uint64_t keys[kMaxBeam];  // the step's best m.beam keys, sorted
        struct {
            float s_score[kMaxBeam];
            float s_raw[kMaxBeam];
            int32_t s_index[kMaxBeam];
            float c_score[kMaxBeam];
            float c_raw[kMaxBeam];
            int32_t c_parent[kMaxBeam];
            int32_t c_len[kMaxBeam];
            int32_t c_token[kMaxBeam];
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
    int32_t* __restrict__ beam_last,
    StepOutputs out,
    PrefixBuffers prefixes) {
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
            if (out.row_lse) out.row_lse[slot_base + k] = 0.0f;
        }
        return;
    }

    for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
        const int64_t s = static_cast<int64_t>(b) * K + k;
        sh.state_raw[k] = beam_raw[s];
        sh.state_len[k] = beam_len[s];
        sh.state_ended[k] = beam_ended[s];
        if (out.row_lse) {
            // The logsumexp of the rows the step read (live, unfinished beams), as the CPU reports it.
            const bool read = k < m.beam && is_finite(beam_raw[s]) && !(m.eos >= 0 && beam_ended[s] != 0);
            out.row_lse[slot_base + k] = read && out.lse ? out.lse[slot_base + k] : 0.0f;
        }
    }
    // The m.beam best of the n_in (<= kTile) surviving keys, then sorted.
    const uint64_t* in = in_keys + static_cast<int64_t>(b) * n_in;
    uint64_t survivors[kTileItems];
#pragma unroll
    for (int j = 0; j < kTileItems; ++j) {
        const int i = static_cast<int>(threadIdx.x) + j * kThreads;
        survivors[j] = i < n_in ? in[i] : 0;
    }
    block_top_k(survivors, m.beam, sh.u.keys, sh.topk);
    const int n = next_pow2(m.beam);
    for (int i = m.beam + static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) sh.u.keys[i] = 0;
    __syncthreads();
    bitonic_sort<uint64_t, true>(sh.u.keys, n);

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
                    // The EOS token the beam ended with (beam_last is kept only with extra EOS tokens).
                    int token = m.eos;
                    if (beam_last) {
                        const int last = beam_last[static_cast<int64_t>(b) * K + p];
                        if (last >= 0 && last < V && (last == m.eos || extra_eos(a, last))) token = last;
                    }
                    sh.u.m.c_token[n_carry] = token;
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
        c.token = sh.u.m.c_token[j];
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
        beam_ended[s] = static_cast<uint8_t>(c.from_logprob ? (m.eos >= 0 && (c.token == m.eos || extra_eos(a, c.token))) : 1);
        if (beam_last) beam_last[s] = c.token;
        if (prefixes.next_prefix) {
            // The new hypothesis' prefix: its parent's, plus the token it emitted.
            const int64_t from = static_cast<int64_t>(b) * K + c.parent;
            const int L = prefixes.prefix_len[from];
            const int64_t stride = prefixes.stride;
            for (int i = 0; i < L; ++i) prefixes.next_prefix[s * stride + i] = prefixes.prefix[from * stride + i];
            const bool append = c.from_logprob && c.token >= 0;
            if (append) prefixes.next_prefix[s * stride + L] = c.token;
            prefixes.next_prefix_len[s] = L + (append ? 1 : 0);
        }
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
        if (beam_last) beam_last[s] = -1;
        if (prefixes.next_prefix) prefixes.next_prefix_len[s] = 0;
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

// One thread per entry: the GNMT length penalty of each length, as the search uses it.
__global__ void length_penalty_kernel(const int32_t* __restrict__ lengths, int64_t count, float alpha,
                                      float* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = length_penalty(lengths[i], alpha);
}

// Gradients of a loss with respect to the decode's score outputs; any may be
// null (a zero gradient).
struct OutputGrads {
    const float* final_scores;      // [B, K]
    const float* final_raw_scores;  // [B, K]
    const float* scores;            // [B, T, K]
    const float* raw_scores;        // [B, T, K]
};

// grid = (B). Mirrors the CPU backward (dbs::run_backward without a relaxed
// pool). Per step, each thread handles some beams: it computes the beam's
// raw-score gradient
//     draw = next + drank * inv_penalty + g_raw [+ g_final_raw at the last step]
// with drank = g_scores [+ g_final at the last step], adding the terms in the
// CPU's order, writes its log-prob entry (distinct beams of a step have
// distinct entries), and files (parent, beam) as a sort key; after sorting,
// each parent sums its children's gradients in beam order, like the CPU.
__global__ void backward_kernel(
    DBSCudaDecodeArgs a,
    const int32_t* __restrict__ parents,
    const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ lengths,
    const uint8_t* __restrict__ from_logprob,
    OutputGrads grads,
    float* __restrict__ grad_log_probs,
    float* __restrict__ draws) {
    __shared__ float grad_a[kMaxBeam];
    __shared__ float grad_b[kMaxBeam];
    __shared__ float draw[kMaxBeam];
    __shared__ uint32_t order[kMaxBeam];

    const int b = static_cast<int>(blockIdx.x);
    const int K = a.beam_size;
    const int V = a.vocab_size;
    const Meta m = load_meta(a, b);
    const int Kb = m.beam;
    const int n = next_pow2(Kb);
    constexpr uint32_t kNone = 0xffffffffu;

    float* next = grad_a;  // raw-score gradient of each beam after step t
    float* prev = grad_b;  // ... after step t - 1
    for (int k = static_cast<int>(threadIdx.x); k < Kb; k += static_cast<int>(blockDim.x)) next[k] = 0.0f;
    __syncthreads();

    for (int t = m.steps - 1; t >= 0; --t) {
        for (int k = static_cast<int>(threadIdx.x); k < n; k += static_cast<int>(blockDim.x)) {
            uint32_t key = kNone;
            float d = 0.0f;
            if (k < Kb) {
                prev[k] = 0.0f;
                const int64_t slot = (static_cast<int64_t>(b) * a.steps + t) * K + k;
                const int parent = parents[slot];
                if (parent >= 0 && parent < Kb) {  // also rejects malformed traces
                    const bool last = t == m.steps - 1;
                    const int64_t beam = static_cast<int64_t>(b) * K + k;
                    float drank = 0.0f;
                    if (grads.scores) drank = __fadd_rn(drank, grads.scores[slot]);
                    if (last && grads.final_scores) drank = __fadd_rn(drank, grads.final_scores[beam]);
                    const float inv_penalty = __fdiv_rn(1.0f, length_penalty(lengths[slot], a.length_penalty_alpha));
                    d = __fadd_rn(next[k], __fmul_rn(drank, inv_penalty));
                    if (grads.raw_scores) d = __fadd_rn(d, grads.raw_scores[slot]);
                    if (last && grads.final_raw_scores) d = __fadd_rn(d, grads.final_raw_scores[beam]);
                    if (d != 0.0f) {
                        const int token = tokens[slot];
                        if (from_logprob[slot] && token >= 0 && token < V) {
                            if (draws) {
                                draws[slot] = d;  // the entry's gradient, kept per slot
                            } else {
                                const int64_t g = ((static_cast<int64_t>(b) * a.steps + t) * K + parent) * V + token;
                                grad_log_probs[g] = __fadd_rn(grad_log_probs[g], d);
                            }
                        }
                        key = static_cast<uint32_t>(parent) * static_cast<uint32_t>(Kb) + static_cast<uint32_t>(k);
                    }
                }
                draw[k] = d;
            }
            order[k] = key;
        }
        __syncthreads();
        bitonic_sort<uint32_t, false>(order, n);

        // Children of one parent are now contiguous and in beam order: the
        // first entry of each run sums the run.
        for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
            const uint32_t key = order[i];
            if (key == kNone) continue;
            const uint32_t parent = key / static_cast<uint32_t>(Kb);
            if (i > 0 && order[i - 1] != kNone && order[i - 1] / static_cast<uint32_t>(Kb) == parent) continue;
            float sum = 0.0f;
            for (int j = i; j < n && order[j] != kNone && order[j] / static_cast<uint32_t>(Kb) == parent; ++j) {
                sum = __fadd_rn(sum, draw[order[j] % static_cast<uint32_t>(Kb)]);
            }
            prev[parent] = sum;
        }
        __syncthreads();
        float* tmp = next;
        next = prev;
        prev = tmp;
    }
}

// The path gradients of each decode step, grouped by the row they belong to.
// Row j of step (b, t) (j < n_rows[b, t]) is row parent[.., j] of the step's
// inputs; its entries are token / value[.., first .. first + count), in token
// order, and sum is their sum in that order.
struct LogitRows {
    int32_t* n_rows;  // [B, T]
    int32_t* parent;  // [B, T, K]
    int32_t* first;   // [B, T, K]
    int32_t* count;   // [B, T, K]
    float* sum;       // [B, T, K]
    int32_t* token;   // [B, T, K] entries, sorted by (parent, token)
    float* value;     // [B, T, K]
};

// grid = (steps blocks), one decode step (b, t) per block at a time. Groups
// the step's path gradients (draws, one per selected slot, 0 for none) by row
// in token order, the order of the CPU's merged entries (dbs::path_gradient):
// sorting the keys (parent * V + token, slot) puts every row's entries
// together in token order, and one thread sums each row in that order.
__global__ void logit_rows_kernel(
    DBSCudaDecodeArgs a,
    const int32_t* __restrict__ parents,
    const int32_t* __restrict__ tokens,
    const float* __restrict__ draws,
    LogitRows rows) {
    __shared__ uint64_t order[kMaxBeam];
    constexpr uint64_t kNone = ~static_cast<uint64_t>(0);
    constexpr int kSlotBits = 16;
    static_assert(kMaxBeam <= (1 << kSlotBits), "slots must fit in the key's low bits");

    const int K = a.beam_size;
    const int64_t V = a.vocab_size;
    const int64_t n_steps = static_cast<int64_t>(a.batch_size) * a.steps;
    for (int64_t step = blockIdx.x; step < n_steps; step += gridDim.x) {
        const int b = static_cast<int>(step / a.steps);
        const int t = static_cast<int>(step - static_cast<int64_t>(b) * a.steps);
        const Meta m = load_meta(a, b);
        if (t >= m.steps) {
            if (threadIdx.x == 0) rows.n_rows[step] = 0;
            continue;
        }
        const int64_t base = step * K;
        const int n = next_pow2(m.beam);
        for (int k = static_cast<int>(threadIdx.x); k < n; k += static_cast<int>(blockDim.x)) {
            uint64_t key = kNone;
            // A non-zero draw belongs to a slot with a valid parent and token (backward_kernel).
            if (k < m.beam && draws[base + k] != 0.0f) {
                const uint64_t entry = static_cast<uint64_t>(parents[base + k]) * static_cast<uint64_t>(V) +
                                       static_cast<uint64_t>(tokens[base + k]);
                key = (entry << kSlotBits) | static_cast<uint64_t>(k);
            }
            order[k] = key;
        }
        __syncthreads();
        bitonic_sort<uint64_t, false>(order, n);
        if (threadIdx.x == 0) {
            int n_rows = 0;
            int i = 0;
            while (i < n && order[i] != kNone) {
                const uint64_t row = (order[i] >> kSlotBits) / static_cast<uint64_t>(V);
                float sum = 0.0f;
                int j = i;
                for (; j < n && order[j] != kNone && (order[j] >> kSlotBits) / static_cast<uint64_t>(V) == row; ++j) {
                    const int slot = static_cast<int>(order[j] & ((1u << kSlotBits) - 1));
                    const float d = draws[base + slot];
                    rows.token[base + j] = tokens[base + slot];
                    rows.value[base + j] = d;
                    sum = __fadd_rn(sum, d);
                }
                rows.parent[base + n_rows] = static_cast<int32_t>(row);
                rows.first[base + n_rows] = i;
                rows.count[base + n_rows] = j - i;
                rows.sum[base + n_rows] = sum;
                ++n_rows;
                i = j;
            }
            rows.n_rows[step] = n_rows;
        }
        __syncthreads();  // `order` is rewritten for the block's next step
    }
}

// grid = (steps blocks, vocabulary chunks). Adds the log-softmax gradient of
// every row with path gradients to grad, as dbs::apply_path_gradient does:
// entry v of such a row gets g_v - softmax(row)_v * S, where g_v is its path
// gradient (0 for tokens without one) and S the row's sum. The logits are of
// type kType.
template <int kType>
__global__ void logit_correction_kernel(
    DBSCudaDecodeArgs a,
    const void* __restrict__ logits,
    const float* __restrict__ row_lse,
    LogitRows rows,
    float* __restrict__ grad) {
    const int K = a.beam_size;
    const int64_t V = a.vocab_size;
    const int64_t n_steps = static_cast<int64_t>(a.batch_size) * a.steps;
    const int64_t v0 = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    const int64_t v_stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
    for (int64_t step = blockIdx.x; step < n_steps; step += gridDim.x) {
        const int64_t base = step * K;
        const int n_rows = rows.n_rows[step];
        for (int j = 0; j < n_rows; ++j) {
            const int64_t r = base + rows.parent[base + j];
            const int32_t* tok = rows.token + base + rows.first[base + j];
            const float* val = rows.value + base + rows.first[base + j];
            const int count = rows.count[base + j];
            const float sum = rows.sum[base + j];
            const float lse = row_lse[r];
            for (int64_t v = v0; v < V; v += v_stride) {
                const float x = load_input<kType>(logits, r * V + v);
                int lo = 0;
                int hi = count;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (tok[mid] < v) lo = mid + 1;
                    else hi = mid;
                }
                const float g = lo < count && tok[lo] == v ? val[lo] : 0.0f;
                const float d = __fsub_rn(g, __fmul_rn(dbs::softmax_probability(x, lse), sum));
                grad[r * V + v] = __fadd_rn(grad[r * V + v], d);
            }
        }
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

bool constrained(const DBSCudaDecodeArgs& a) {
    return a.no_repeat_ngram_size > 0 || a.repetition_penalty > 1.0f;
}

bool valid_dtype(int dtype) {
    return dtype == DBS_CUDA_DTYPE_F32 || dtype == DBS_CUDA_DTYPE_F16 || dtype == DBS_CUDA_DTYPE_BF16;
}

bool valid_args(const DBSCudaDecodeArgs* a) {
    if (!a) return false;
    if (a->batch_size <= 0 || a->steps <= 0 || a->beam_size <= 0 || a->vocab_size <= 0) return false;
    if (a->beam_size > kMaxBeam) return false;
    if (a->eos_token < -1 || a->eos_token >= a->vocab_size) return false;
    if (a->min_length < 0) return false;
    if (!std::isfinite(a->length_penalty_alpha) || a->length_penalty_alpha < 0.0f) return false;
    if (!std::isfinite(a->repetition_penalty) || a->repetition_penalty < 0.0f) return false;
    if (a->reserved0 != 0) return false;
    // Candidate indices p * V + v must fit in int32, and the tile count must fit
    // in gridDim.y (65535 tiles of kTile candidates, about 268M candidates).
    const int64_t candidates = static_cast<int64_t>(a->beam_size) * a->vocab_size;
    if (candidates > std::numeric_limits<int32_t>::max()) return false;
    if (ceil_div(candidates, kTile) > kMaxGridY) return false;
    // All tensor element counts must fit in int64.
    const double elements = static_cast<double>(a->batch_size) * a->steps * a->beam_size * a->vocab_size;
    return elements < 9.0e18;
}

// Candidates per thread of the register top-k scan: few enough that the whole
// batch spans about kTargetBlocks blocks (so one example still fills a large
// GPU), but at least kSmallItemsMin, and enough to keep the grid within
// kMaxGridY blocks per example. Depends only on the arguments, so workspace
// sizes do not depend on the device.
int small_scan_items(const DBSCudaDecodeArgs& a) {
    const int64_t per_example = static_cast<int64_t>(a.beam_size) * a.vocab_size;
    int64_t items = ceil_div(per_example * a.batch_size, static_cast<int64_t>(kTargetBlocks) * kThreads);
    items = std::max<int64_t>(items, ceil_div(per_example, static_cast<int64_t>(kMaxGridY) * kThreads));
    return static_cast<int>(std::min<int64_t>(kSmallItemsMax, std::max<int64_t>(kSmallItemsMin, items)));
}

static_assert(sizeof(ScanParams) <= 128, "see ScanParams");

struct DecodePlan {
    bool small;          // register top-k scan (beam_size <= kSmallBeam)
    int small_items;     // candidates per thread in the register top-k scan
    int blocks0;         // scan blocks per example
    int words;           // constraint bitmap words per beam
    int prefix_stride;   // prefix capacity per beam
    int64_t state_raw_offset;
    int64_t state_len_offset;
    int64_t state_ended_offset;
    int64_t keys0_offset;
    int64_t keys1_offset;
    int64_t status_offset;
    int64_t lse_offset;  // [B, T, K] logsumexp of the rows (decoding from logits)
    int64_t flags_offset;  // [V] kFlag* token flags (extra EOS tokens)
    int64_t last_offset;   // [B, K] each beam's last token (extra EOS tokens)
    // Constraint buffers (only when n-gram blocking or a repetition penalty is on).
    int64_t prefix_offset[2];
    int64_t prefix_len_offset[2];
    int64_t blocked_offset;
    int64_t penalised_offset;
    int64_t marked_offset;
    int64_t marked_count_offset;
    int64_t constraint_bytes;  // bytes to zero before decoding (bitmaps and counts)
    int64_t total_bytes;
};

// prefix_stride < 0 plans a full decode, which keeps the beams' prefixes itself
// (a.steps tokens each); otherwise one step whose prefixes the caller supplies.
DecodePlan make_plan(const DBSCudaDecodeArgs& a, int prefix_stride = -1) {
    DecodePlan p{};
    const int64_t B = a.batch_size;
    const int64_t K = a.beam_size;
    const int64_t T = a.steps;
    const int64_t candidates = K * a.vocab_size;
    p.small = K <= kSmallBeam;
    p.small_items = small_scan_items(a);
    p.blocks0 = static_cast<int>(ceil_div(candidates, p.small ? static_cast<int64_t>(kThreads) * p.small_items : kTile));
    const int64_t n0 = static_cast<int64_t>(p.blocks0) * K;
    const int64_t n1 = n0 > kTile ? ceil_div(n0, kTile) * K : 0;
    int64_t off = 0;
    auto take = [&off](int64_t bytes) {
        const int64_t at = off;
        off = align_up(off + bytes);
        return at;
    };
    p.state_raw_offset = take(B * K * static_cast<int64_t>(sizeof(float)));
    p.state_len_offset = take(B * K * static_cast<int64_t>(sizeof(int32_t)));
    p.state_ended_offset = take(B * K);
    p.keys0_offset = take(B * n0 * static_cast<int64_t>(sizeof(uint64_t)));
    p.keys1_offset = take(B * n1 * static_cast<int64_t>(sizeof(uint64_t)));
    p.status_offset = take(static_cast<int64_t>(sizeof(int)));
    p.lse_offset = take(B * T * K * static_cast<int64_t>(sizeof(float)));
    p.flags_offset = take(a.vocab_size);
    p.last_offset = take(B * K * static_cast<int64_t>(sizeof(int32_t)));
    if (constrained(a)) {
        p.words = static_cast<int>(ceil_div(a.vocab_size, 32));
        p.prefix_stride = prefix_stride < 0 ? a.steps : prefix_stride;
        if (prefix_stride < 0) {
            for (int i = 0; i < 2; ++i) {
                p.prefix_offset[i] = take(B * K * T * static_cast<int64_t>(sizeof(int32_t)));
                p.prefix_len_offset[i] = take(B * K * static_cast<int64_t>(sizeof(int32_t)));
            }
        }
        p.marked_offset = take(B * K * 2 * p.prefix_stride * static_cast<int64_t>(sizeof(int32_t)));
        // Zeroed together: bitmaps and marked counts.
        const int64_t zero_begin = off;
        p.blocked_offset = take(B * K * p.words * static_cast<int64_t>(sizeof(uint32_t)));
        p.penalised_offset = take(B * K * p.words * static_cast<int64_t>(sizeof(uint32_t)));
        p.marked_count_offset = take(B * K * static_cast<int64_t>(sizeof(int32_t)));
        p.constraint_bytes = off - zero_begin;
    }
    p.total_bytes = off;
    return p;
}

// Scratch of dbs_cuda_backward_ex's logits correction: the path gradient of
// every slot, and the rows they belong to (LogitRows).
struct BackwardPlan {
    int64_t draws_offset;
    int64_t n_rows_offset;
    int64_t parent_offset;
    int64_t first_offset;
    int64_t count_offset;
    int64_t sum_offset;
    int64_t token_offset;
    int64_t value_offset;
    int64_t total_bytes;
};

BackwardPlan make_backward_plan(const DBSCudaDecodeArgs& a) {
    BackwardPlan p{};
    const int64_t steps = static_cast<int64_t>(a.batch_size) * a.steps;
    const int64_t slots = steps * a.beam_size;
    int64_t off = 0;
    auto take = [&off](int64_t bytes) {
        const int64_t at = off;
        off = align_up(off + bytes);
        return at;
    };
    p.draws_offset = take(slots * static_cast<int64_t>(sizeof(float)));
    p.n_rows_offset = take(steps * static_cast<int64_t>(sizeof(int32_t)));
    p.parent_offset = take(slots * static_cast<int64_t>(sizeof(int32_t)));
    p.first_offset = take(slots * static_cast<int64_t>(sizeof(int32_t)));
    p.count_offset = take(slots * static_cast<int64_t>(sizeof(int32_t)));
    p.sum_offset = take(slots * static_cast<int64_t>(sizeof(float)));
    p.token_offset = take(slots * static_cast<int64_t>(sizeof(int32_t)));
    p.value_offset = take(slots * static_cast<int64_t>(sizeof(float)));
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

template <int kType, bool kLogits>
void launch_scan_as(const DecodePlan& plan, const ScanParams& params, int t, const float* beam_raw,
                    const int32_t* beam_len, const uint8_t* beam_ended, uint64_t* keys, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(params.a.batch_size), static_cast<unsigned int>(plan.blocks0));
    if (!plan.small) {
        const auto tiles = scan_tiles_kernel<kType, kLogits>;
        DBS_LAUNCH(tiles, grid, dim3(kThreads), stream, params, t, beam_raw, beam_len, beam_ended, keys, plan.blocks0);
        return;
    }
    const int K = params.a.beam_size;
    const auto small = K <= 1   ? scan_small_kernel<1, kType, kLogits>
                       : K <= 2 ? scan_small_kernel<2, kType, kLogits>
                       : K <= 4 ? scan_small_kernel<4, kType, kLogits>
                       : K <= 8 ? scan_small_kernel<8, kType, kLogits>
                                : scan_small_kernel<16, kType, kLogits>;
    DBS_LAUNCH(small, grid, dim3(kThreads), stream, params, t, beam_raw, beam_len, beam_ended, keys, plan.blocks0,
               plan.small_items);
}

void launch_scan(const DecodePlan& plan, const ScanParams& params, int dtype, int t, const float* beam_raw,
                 const int32_t* beam_len, const uint8_t* beam_ended, uint64_t* keys, cudaStream_t stream) {
    const bool logits = params.row_lse != nullptr;
    switch (dtype) {
        case DBS_CUDA_DTYPE_F16:
            if (logits) return launch_scan_as<DBS_CUDA_DTYPE_F16, true>(plan, params, t, beam_raw, beam_len, beam_ended, keys, stream);
            return launch_scan_as<DBS_CUDA_DTYPE_F16, false>(plan, params, t, beam_raw, beam_len, beam_ended, keys, stream);
        case DBS_CUDA_DTYPE_BF16:
            if (logits) return launch_scan_as<DBS_CUDA_DTYPE_BF16, true>(plan, params, t, beam_raw, beam_len, beam_ended, keys, stream);
            return launch_scan_as<DBS_CUDA_DTYPE_BF16, false>(plan, params, t, beam_raw, beam_len, beam_ended, keys, stream);
        default:
            if (logits) return launch_scan_as<DBS_CUDA_DTYPE_F32, true>(plan, params, t, beam_raw, beam_len, beam_ended, keys, stream);
            return launch_scan_as<DBS_CUDA_DTYPE_F32, false>(plan, params, t, beam_raw, beam_len, beam_ended, keys, stream);
    }
}

// Clears the constraint scratch (bitmaps, marked tokens) and points `cons` and
// the scan parameters at it; the prefixes are set per step. False if the
// memset could not be queued.
bool prepare_constraints(const DecodePlan& plan, const DBSCudaDecodeArgs& a, const Workspace& ws, cudaStream_t stream,
                         ConstraintState& cons, ScanParams& params) {
    if (cudaMemsetAsync(ws.at<char>(plan.blocked_offset), 0, static_cast<size_t>(plan.constraint_bytes), stream) != cudaSuccess) {
        return false;
    }
    cons.blocked = ws.at<uint32_t>(plan.blocked_offset);
    cons.penalised = ws.at<uint32_t>(plan.penalised_offset);
    cons.marked = ws.at<int32_t>(plan.marked_offset);
    cons.marked_count = ws.at<int32_t>(plan.marked_count_offset);
    cons.stride = plan.prefix_stride;
    cons.words = plan.words;
    cons.ngram = a.no_repeat_ngram_size;
    cons.penalise = a.repetition_penalty > 1.0f ? 1 : 0;
    params.blocked = cons.blocked;
    params.penalised = cons.penalised;
    params.words = plan.words;
    // Computed on the host exactly as the CPU decoder does.
    params.log_penalty = cons.penalise ? std::log(a.repetition_penalty) : 0.0f;
    return true;
}

// Queues the kernels of decode step t, whose inputs are of type dtype; `cons`
// is null without n-gram blocking or a repetition penalty. Advances the beam
// state in place.
void launch_step(const DecodePlan& plan, const ScanParams& params, int dtype, const ConstraintState* cons,
                 const PrefixBuffers& prefixes, int t, float* beam_raw, int32_t* beam_len, uint8_t* beam_ended,
                 int32_t* beam_last, uint64_t* keys0, uint64_t* keys1, const StepOutputs& step_out, cudaStream_t stream) {
    const DBSCudaDecodeArgs& a = params.a;
    const unsigned int B = static_cast<unsigned int>(a.batch_size);
    if (cons) {
        DBS_LAUNCH(constraint_kernel, dim3(B), dim3(kThreads), stream, a, t, *cons, static_cast<const float*>(beam_raw));
    }
    launch_scan(plan, params, dtype, t, beam_raw, beam_len, beam_ended, keys0, stream);
    int n = plan.blocks0 * a.beam_size;
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
               a, t, src, n, beam_raw, beam_len, beam_ended, beam_last, step_out, prefixes);
}

StepOutputs step_outputs(const DBSCudaDecodeOutputs& o, const float* lse, float* row_lse) {
    StepOutputs s;
    s.tokens = o.tokens;
    s.parents = o.parents;
    s.lengths = o.lengths;
    s.scores = o.scores;
    s.raw_scores = o.raw_scores;
    s.from_logprob = o.from_logprob;
    s.lse = lse;
    s.row_lse = row_lse;
    return s;
}

bool valid_search_options(const DBSCudaSearchOptions* o, const DBSCudaDecodeArgs& a) {
    if (!o) return true;
    if (o->reserved0 != 0 || o->reserved[0] || o->reserved[1] || o->reserved[2] || o->reserved[3]) return false;
    if (o->extra_eos_count < 0 || o->extra_eos_count > DBS_CUDA_MAX_EXTRA_EOS) return false;
    for (int i = 0; i < o->extra_eos_count; ++i) {
        if (o->extra_eos_tokens[i] < 0 || o->extra_eos_tokens[i] >= a.vocab_size) return false;
    }
    // Extra EOS tokens need EOS handling.
    return o->extra_eos_count == 0 || a.eos_token >= 0 || a.eos_tokens_per_example;
}

// With extra EOS tokens: builds the token flags in the workspace and points the
// engine's copy of the arguments at them (see kFlagBanned). Returns the [B, K]
// last-token state, or null without extra EOS tokens.
int32_t* use_extra_eos(const DBSCudaSearchOptions* o, const DecodePlan& plan, const Workspace& ws, DBSCudaDecodeArgs& a,
                       cudaStream_t stream) {
    if (!o || o->extra_eos_count == 0) return nullptr;
    ExtraEos extra{};
    extra.count = o->extra_eos_count;
    for (int i = 0; i < extra.count; ++i) extra.tokens[i] = o->extra_eos_tokens[i];
    uint8_t* flags = ws.at<uint8_t>(plan.flags_offset);
    DBS_LAUNCH(token_flags_kernel, dim3(static_cast<unsigned int>(ceil_div(a.vocab_size, kThreads))), dim3(kThreads),
               stream, a.vocab_size, a.banned_tokens, extra, flags);
    a.banned_tokens = flags;
    a.reserved0 = 1;
    return ws.at<int32_t>(plan.last_offset);
}

// From logits: the logsumexp of every row of `inputs`, into lse [B, steps, K].
void launch_row_lse(const DBSCudaDecodeArgs& a, const void* inputs, int dtype, float* lse, cudaStream_t stream) {
    const int64_t rows = static_cast<int64_t>(a.batch_size) * a.steps * a.beam_size;
    const auto kernel = dtype == DBS_CUDA_DTYPE_F16    ? row_lse_kernel<DBS_CUDA_DTYPE_F16>
                        : dtype == DBS_CUDA_DTYPE_BF16 ? row_lse_kernel<DBS_CUDA_DTYPE_BF16>
                                                       : row_lse_kernel<DBS_CUDA_DTYPE_F32>;
    DBS_LAUNCH(kernel, dim3(static_cast<unsigned int>(std::min(rows, kMaxLseBlocks))), dim3(dbs::kLogitLanes), stream,
               a, inputs, lse);
}

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
    // The backward keeps its state in shared memory; only the logits
    // correction of dbs_cuda_backward_ex needs scratch.
    return make_backward_plan(*args).total_bytes;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_ex2(
    const void* inputs,
    int data_type,
    int from_logits,
    const DBSCudaDecodeArgs* args,
    const DBSCudaSearchOptions* options,
    const DBSCudaDecodeOutputs* outputs,
    float* row_lse,
    void* workspace,
    int64_t workspace_bytes,
    void* stream_ptr) {
    if (!inputs || !valid_dtype(data_type) || !valid_args(args) || !outputs || !outputs->final_scores) {
        return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (!valid_search_options(options, *args)) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    DBSCudaDecodeArgs a = *args;
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
    int32_t* beam_last = use_extra_eos(options, plan, ws, a, stream);
    DBS_LAUNCH(init_state_kernel, beam_grid, dim3(kThreads), stream, a, beam_raw, beam_len, beam_ended, beam_last,
               outputs->invalid_input);

    ScanParams params{};
    params.a = a;
    params.inputs = inputs;
    params.row_lse = from_logits ? ws.at<float>(plan.lse_offset) : nullptr;
    params.invalid = outputs->invalid_input;
    if (from_logits) launch_row_lse(a, inputs, data_type, ws.at<float>(plan.lse_offset), stream);

    ConstraintState cons{};
    PrefixBuffers prefixes{};
    int32_t* prefix[2] = {nullptr, nullptr};
    int32_t* prefix_len[2] = {nullptr, nullptr};
    const bool with_constraints = constrained(a);
    if (with_constraints) {
        if (!prepare_constraints(plan, a, ws, stream, cons, params)) return DBS_CUDA_STATUS_LAUNCH_FAILED;
        for (int i = 0; i < 2; ++i) {
            prefix[i] = ws.at<int32_t>(plan.prefix_offset[i]);
            prefix_len[i] = ws.at<int32_t>(plan.prefix_len_offset[i]);
        }
        if (cudaMemsetAsync(prefix_len[0], 0, static_cast<size_t>(beams) * sizeof(int32_t), stream) != cudaSuccess) {
            return DBS_CUDA_STATUS_LAUNCH_FAILED;
        }
        prefixes.stride = plan.prefix_stride;
    }

    const StepOutputs step_out = step_outputs(*outputs, params.row_lse, row_lse);
    for (int t = 0; t < a.steps; ++t) {
        if (with_constraints) {
            cons.prefix = prefix[t & 1];
            cons.prefix_len = prefix_len[t & 1];
            prefixes.prefix = prefix[t & 1];
            prefixes.prefix_len = prefix_len[t & 1];
            prefixes.next_prefix = prefix[(t + 1) & 1];
            prefixes.next_prefix_len = prefix_len[(t + 1) & 1];
        }
        launch_step(plan, params, data_type, with_constraints ? &cons : nullptr, prefixes, t, beam_raw, beam_len, beam_ended,
                    beam_last, keys0, keys1, step_out, stream);
    }

    DBS_LAUNCH(finalize_kernel, beam_grid, dim3(kThreads), stream,
               a, beam_raw, beam_len, outputs->final_scores, outputs->final_raw_scores, outputs->final_lengths);
    return finish(stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_ex(
    const void* inputs,
    int data_type,
    int from_logits,
    const DBSCudaDecodeArgs* args,
    const DBSCudaDecodeOutputs* outputs,
    float* row_lse,
    void* workspace,
    int64_t workspace_bytes,
    void* stream) {
    return dbs_cuda_decode_ex2(inputs, data_type, from_logits, args, nullptr, outputs, row_lse, workspace,
                               workspace_bytes, stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode(
    const float* log_probs,
    const DBSCudaDecodeArgs* args,
    const DBSCudaDecodeOutputs* outputs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream) {
    return dbs_cuda_decode_ex(log_probs, DBS_CUDA_DTYPE_F32, 0, args, outputs, nullptr, workspace, workspace_bytes, stream);
}

extern "C" DBS_CUDA_EXPORT int64_t dbs_cuda_decode_step_workspace_size(const DBSCudaDecodeArgs* args, int prefix_stride) {
    if (!valid_args(args) || args->steps != 1 || args->steps_per_example || prefix_stride < 0) return -1;
    return make_plan(*args, prefix_stride).total_bytes;
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_step_ex2(
    const void* inputs,
    int data_type,
    int from_logits,
    const DBSCudaDecodeArgs* args,
    const DBSCudaSearchOptions* options,
    const DBSCudaBeamState* state,
    const DBSCudaDecodeOutputs* outputs,
    float* row_lse,
    void* workspace,
    int64_t workspace_bytes,
    void* stream_ptr) {
    if (!inputs || !valid_dtype(data_type) || !valid_args(args) || !state || !outputs) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (!valid_search_options(options, *args)) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    DBSCudaDecodeArgs a = *args;
    if (a.steps != 1 || a.steps_per_example) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (!state->raw_scores || !state->lengths || !state->finished || state->prefix_stride < 0 || state->reserved0 != 0) {
        return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    }
    const bool with_constraints = constrained(a);
    if (with_constraints && state->prefix_stride > 0 && !state->prefixes) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const DecodePlan plan = make_plan(a, state->prefix_stride);
    Workspace ws(workspace, workspace_bytes, plan.total_bytes, stream);
    if (!ws.ok()) return ws.status();

    const int rc = validate_per_example(a, ws.at<int>(plan.status_offset), stream);
    if (rc != DBS_CUDA_STATUS_OK) return rc;
    if (outputs->invalid_input &&
        cudaMemsetAsync(outputs->invalid_input, 0, static_cast<size_t>(a.batch_size), stream) != cudaSuccess) {
        return DBS_CUDA_STATUS_LAUNCH_FAILED;
    }
    const int64_t beams = static_cast<int64_t>(a.batch_size) * a.beam_size;
    int32_t* beam_last = use_extra_eos(options, plan, ws, a, stream);
    if (beam_last) {
        // The token each finished beam carries forward: the last of its prefix.
        DBS_LAUNCH(last_token_kernel, dim3(static_cast<unsigned int>(ceil_div(beams, kThreads))), dim3(kThreads), stream,
                   beams, state->prefixes, state->prefix_stride, static_cast<const int32_t*>(state->lengths), beam_last);
    }

    ScanParams params{};
    params.a = a;
    params.inputs = inputs;
    params.row_lse = from_logits ? ws.at<float>(plan.lse_offset) : nullptr;
    params.invalid = outputs->invalid_input;
    if (from_logits) launch_row_lse(a, inputs, data_type, ws.at<float>(plan.lse_offset), stream);
    ConstraintState cons{};
    if (with_constraints) {
        if (!prepare_constraints(plan, a, ws, stream, cons, params)) return DBS_CUDA_STATUS_LAUNCH_FAILED;
        cons.prefix = state->prefixes;
        cons.prefix_len = state->lengths;  // live beams' prefixes are exactly their hypotheses
    }
    const PrefixBuffers no_prefixes{};  // the caller tracks the prefixes
    launch_step(plan, params, data_type, with_constraints ? &cons : nullptr, no_prefixes, 0, state->raw_scores, state->lengths,
                state->finished, beam_last, ws.at<uint64_t>(plan.keys0_offset), ws.at<uint64_t>(plan.keys1_offset),
                step_outputs(*outputs, params.row_lse, row_lse), stream);
    if (outputs->final_scores) {
        DBS_LAUNCH(finalize_kernel, dim3(static_cast<unsigned int>(ceil_div(beams, kThreads))), dim3(kThreads), stream,
                   a, state->raw_scores, state->lengths, outputs->final_scores, outputs->final_raw_scores,
                   outputs->final_lengths);
    }
    return finish(stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_step_ex(
    const void* inputs,
    int data_type,
    int from_logits,
    const DBSCudaDecodeArgs* args,
    const DBSCudaBeamState* state,
    const DBSCudaDecodeOutputs* outputs,
    float* row_lse,
    void* workspace,
    int64_t workspace_bytes,
    void* stream) {
    return dbs_cuda_decode_step_ex2(inputs, data_type, from_logits, args, nullptr, state, outputs, row_lse, workspace,
                                    workspace_bytes, stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_decode_step(
    const float* log_probs,
    const DBSCudaDecodeArgs* args,
    const DBSCudaBeamState* state,
    const DBSCudaDecodeOutputs* outputs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream) {
    return dbs_cuda_decode_step_ex(log_probs, DBS_CUDA_DTYPE_F32, 0, args, state, outputs, nullptr, workspace,
                                   workspace_bytes, stream);
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
    DBSCudaBackwardInputs in{};
    in.parents = parents;
    in.tokens = tokens;
    in.lengths = lengths;
    in.from_logprob = from_logprob;
    in.grad_final_scores = grad_final_scores;
    return dbs_cuda_backward_ex(args, &in, grad_log_probs, workspace, workspace_bytes, stream_ptr);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_backward_ex(
    const DBSCudaDecodeArgs* args,
    const DBSCudaBackwardInputs* in,
    float* grad_inputs,
    void* workspace,
    int64_t workspace_bytes,
    void* stream_ptr) {
    if (!valid_args(args) || !in || !grad_inputs) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (!in->parents || !in->tokens || !in->lengths || !in->from_logprob) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (in->reserved0 != 0 || in->reserved[0] || in->reserved[1] || in->reserved[2] || in->reserved[3]) {
        return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (in->logits && (!in->row_lse || !valid_dtype(in->logits_type))) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    const DBSCudaDecodeArgs& a = *args;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    OutputGrads grads{};
    grads.final_scores = in->grad_final_scores;
    grads.final_raw_scores = in->grad_final_raw_scores;
    grads.scores = in->grad_scores;
    grads.raw_scores = in->grad_raw_scores;
    const dim3 grid(static_cast<unsigned int>(a.batch_size));
    if (!in->logits) {
        DBS_LAUNCH(backward_kernel, grid, dim3(kBackwardThreads), stream,
                   a, in->parents, in->tokens, in->lengths, in->from_logprob, grads, grad_inputs, static_cast<float*>(nullptr));
        return finish(stream);
    }

    // From logits: the path gradients per slot, grouped by row, then every
    // row with one gets its log-softmax gradient.
    const BackwardPlan plan = make_backward_plan(a);
    Workspace ws(workspace, workspace_bytes, plan.total_bytes, stream);
    if (!ws.ok()) return ws.status();
    float* draws = ws.at<float>(plan.draws_offset);
    const size_t slots = static_cast<size_t>(a.batch_size) * a.steps * a.beam_size;
    if (cudaMemsetAsync(draws, 0, slots * sizeof(float), stream) != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    DBS_LAUNCH(backward_kernel, grid, dim3(kBackwardThreads), stream,
               a, in->parents, in->tokens, in->lengths, in->from_logprob, grads, static_cast<float*>(nullptr), draws);
    LogitRows rows{};
    rows.n_rows = ws.at<int32_t>(plan.n_rows_offset);
    rows.parent = ws.at<int32_t>(plan.parent_offset);
    rows.first = ws.at<int32_t>(plan.first_offset);
    rows.count = ws.at<int32_t>(plan.count_offset);
    rows.sum = ws.at<float>(plan.sum_offset);
    rows.token = ws.at<int32_t>(plan.token_offset);
    rows.value = ws.at<float>(plan.value_offset);
    const int64_t n_steps = static_cast<int64_t>(a.batch_size) * a.steps;
    const unsigned int step_blocks = static_cast<unsigned int>(std::min<int64_t>(n_steps, kMaxStepBlocks));
    DBS_LAUNCH(logit_rows_kernel, dim3(step_blocks), dim3(kBackwardThreads), stream,
               a, in->parents, in->tokens, static_cast<const float*>(draws), rows);
    const unsigned int vocab_blocks = static_cast<unsigned int>(
        std::min<int64_t>(ceil_div(a.vocab_size, static_cast<int64_t>(kThreads) * kCorrectionItems), kMaxGridY));
    const auto correction = in->logits_type == DBS_CUDA_DTYPE_F16    ? logit_correction_kernel<DBS_CUDA_DTYPE_F16>
                            : in->logits_type == DBS_CUDA_DTYPE_BF16 ? logit_correction_kernel<DBS_CUDA_DTYPE_BF16>
                                                                     : logit_correction_kernel<DBS_CUDA_DTYPE_F32>;
    DBS_LAUNCH(correction, dim3(step_blocks, vocab_blocks), dim3(kThreads), stream,
               a, in->logits, in->row_lse, rows, grad_inputs);
    return finish(stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_length_penalty(
    const int32_t* lengths, int64_t count, float alpha, float* out, void* stream_ptr) {
    if (count < 0 || (count > 0 && (!lengths || !out)) || !std::isfinite(alpha) || alpha < 0.0f) {
        return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0) return DBS_CUDA_STATUS_OK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    DBS_LAUNCH(length_penalty_kernel, dim3(static_cast<unsigned int>(ceil_div(count, kThreads))), dim3(kThreads), stream,
               lengths, count, alpha, out);
    return finish(stream);
}

extern "C" DBS_CUDA_EXPORT int dbs_cuda_path_gradient(
    const DBSCudaDecodeArgs* args,
    const int32_t* parents,
    const int32_t* tokens,
    const int32_t* lengths,
    const uint8_t* from_logprob,
    const float* grad_final_scores,
    float* draws,
    void* stream_ptr) {
    if (!valid_args(args) || !parents || !tokens || !lengths || !from_logprob || !grad_final_scores || !draws) {
        return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    }
    const DBSCudaDecodeArgs& a = *args;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const size_t slots = static_cast<size_t>(a.batch_size) * a.steps * a.beam_size;
    if (cudaMemsetAsync(draws, 0, slots * sizeof(float), stream) != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    OutputGrads grads{};
    grads.final_scores = grad_final_scores;
    DBS_LAUNCH(backward_kernel, dim3(static_cast<unsigned int>(a.batch_size)), dim3(kBackwardThreads), stream,
               a, parents, tokens, lengths, from_logprob, grads, static_cast<float*>(nullptr), draws);
    return finish(stream);
}
