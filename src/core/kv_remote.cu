// src/core/kv_remote.cu - S-31: the main QSA layers' K/V on a second GPU (kv_remote.hpp).
#include "strata/core/kv_remote.hpp"

#include "strata/core/keep_warm.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/pin_budget.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace {

using strata::kernels::kStepCount;
using strata::kernels::kStepWidth;

std::atomic<uint64_t> g_allocated{0};

// the mapped header's words (uint32 index): one 128-byte line each
constexpr int kWin = 0, kAbort = 32;
constexpr uint64_t kReq = 0, kDone = 128, kPayload = 256;   // byte offsets in a slot

bool ready_device(int dev) {
    static std::atomic<int> state{0};   // 0 not tried, 1 ok, -1 refused
    const int s = state.load();
    if (s != 0) return s > 0;
    int n = 0, primary = -1;
    if (cudaGetDeviceCount(&n) != cudaSuccess || dev < 0 || dev >= n || cudaGetDevice(&primary) != cudaSuccess ||
        dev == primary) {
        cudaGetLastError();
        std::fprintf(stderr, "strata: S-31: STRATA_S31_KV_DEV=%d is not a visible second CUDA device (%d visible, "
                             "primary %d)\n", dev, n, primary);
        state.store(-1);
        return false;
    }
    // the host waits on this card's stream between windows: spin, as the helper caches do (remote_experts.cpp); the
    // mailbox is mapped into its context.  Only possible before the context exists; an existing one keeps its flags.
    cudaInitDevice(dev, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess)
        std::fprintf(stderr, "strata: S-31: the main layers' K/V and their decode attention on CUDA%d (%s, sm_%d%d)\n", dev,
                     p.name, p.major, p.minor);
    state.store(1);
    return true;
}

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <bool H> struct Elem;
template <> struct Elem<false> {
    using T = float;
    __device__ static T to(float x) { return x; }
    __device__ static float from(const T* p) { return __ldcv(p); }
};
template <> struct Elem<true> {
    using T = __half;
    __device__ static T to(float x) { return __float2half_rn(x); }
    __device__ static float from(const T* p) {
        const unsigned short b = __ldcv((const unsigned short*) p);
        return __half2float(__ushort_as_half(b));
    }
};

struct Geo {
    int nq, nkv;          // floats per row: n_head * head_dim, n_head_kv * head_dim
    int64_t cap;
    uint64_t off_k, off_v, off_sel, off_step, off_out;   // byte offsets in a slot
};

// the last block of a grid to finish raises `word` to the window number (after every block's stores are out)
__device__ __forceinline__ void raise_last(uint32_t* cnt, volatile uint32_t* word, const volatile uint32_t* hdr) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prev = atomicAdd(cnt, 1u);
        if (prev == gridDim.x - 1) {
            *(volatile uint32_t*) cnt = 0u;
            __threadfence_system();
            *word = hdr[kWin];
            __threadfence_system();
        }
    }
}

template <bool H>
__global__ void publish_kernel(uint8_t* slot, const volatile uint32_t* hdr, Geo g, const float* __restrict__ q,
                               const float* __restrict__ k, const float* __restrict__ v, const int32_t* __restrict__ sel,
                               const int32_t* __restrict__ step, int n, uint32_t* cnt) {
    using E = Elem<H>;
    typename E::T* pq = (typename E::T*) (slot + kPayload);
    typename E::T* pk = (typename E::T*) (slot + g.off_k);
    typename E::T* pv = (typename E::T*) (slot + g.off_v);
    int32_t* ps = (int32_t*) (slot + g.off_sel);
    int32_t* pt = (int32_t*) (slot + g.off_step);
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, nt = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < (int64_t) n * g.nq; i += nt) pq[i] = E::to(q[i]);
    for (int64_t i = tid; i < (int64_t) n * g.nkv; i += nt) { pk[i] = E::to(k[i]); pv[i] = E::to(v[i]); }
    for (int64_t i = tid; i < (int64_t) n * kStepCount; i += nt) pt[i] = step[i];
    for (int64_t i = tid; i < (int64_t) n * g.cap; i += nt) {   // only the cells each row selected cross
        const int64_t t = i / g.cap, c = i - t * g.cap;
        if (c < step[t * kStepCount + kStepWidth]) ps[i] = sel[i];
    }
    raise_last(cnt, (volatile uint32_t*) (slot + kReq), hdr);
}

// one thread: wait until `word` holds this window's number (or the window is abandoned)
__global__ void wait_kernel(const volatile uint32_t* word, const volatile uint32_t* hdr) {
    const uint32_t want = hdr[kWin];
    while (*word != want) {
        if (hdr[kAbort] != 0u) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    __threadfence_system();
}

template <bool H>
__global__ void fetch_kernel(const uint8_t* slot, Geo g, float* __restrict__ q, float* __restrict__ k,
                             float* __restrict__ v, int32_t* __restrict__ sel, int32_t* __restrict__ step, int n) {
    using E = Elem<H>;
    const typename E::T* pq = (const typename E::T*) (slot + kPayload);
    const typename E::T* pk = (const typename E::T*) (slot + g.off_k);
    const typename E::T* pv = (const typename E::T*) (slot + g.off_v);
    const int32_t* ps = (const int32_t*) (slot + g.off_sel);
    const int32_t* pt = (const int32_t*) (slot + g.off_step);
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, nt = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < (int64_t) n * g.nq; i += nt) q[i] = E::from(pq + i);
    for (int64_t i = tid; i < (int64_t) n * g.nkv; i += nt) { k[i] = E::from(pk + i); v[i] = E::from(pv + i); }
    for (int64_t i = tid; i < (int64_t) n * kStepCount; i += nt) step[i] = __ldcv(pt + i);
    for (int64_t i = tid; i < (int64_t) n * g.cap; i += nt) {
        const int64_t t = i / g.cap, c = i - t * g.cap;
        if (c < __ldcv(pt + t * kStepCount + kStepWidth)) sel[i] = __ldcv(ps + i);
    }
}

template <bool H>
__global__ void reply_kernel(uint8_t* slot, const volatile uint32_t* hdr, Geo g, const float* __restrict__ attn, int n,
                             uint32_t* cnt) {
    using E = Elem<H>;
    typename E::T* po = (typename E::T*) (slot + g.off_out);
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, nt = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < (int64_t) n * g.nq; i += nt) po[i] = E::to(attn[i]);
    raise_last(cnt, (volatile uint32_t*) (slot + kDone), hdr);
}

// STRATA_S31_PROF=1: one thread charges the time since the previous stamp to phase k ([0] last stamp, [1..8] sums,
// [9] windows)
__global__ void stamp_kernel(unsigned long long* p, int k) {
    const unsigned long long t = now_ns();
    if (k > 0) p[k] += t - p[0];
    else p[9] += 1;
    p[0] = t;
}

template <bool H>
__global__ void collect_kernel(const uint8_t* slot, Geo g, float* __restrict__ attn, int n) {
    using E = Elem<H>;
    const typename E::T* po = (const typename E::T*) (slot + g.off_out);
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, nt = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < (int64_t) n * g.nq; i += nt) attn[i] = E::from(po + i);
}

uint64_t up256(uint64_t n) { return (n + 255) & ~255ull; }

// a state's K/V arrays (on the remote card) beside the staging pool's, and their bytes per (cell, head) row
struct Arr {
    uint8_t* remote;
    uint8_t* stage;
    uint64_t row;
};
int arrays(const QsaState& st, const strata::kernels::KvHostPools& sp, const strata::kernels::QsaShapes& s, Arr* a) {
    const uint64_t hd = (uint64_t) s.head_dim, sc = (hd / strata::kernels::KV_Q8_GROUP) * 2;
    const uint64_t q4 = strata::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    if (st.kv_hybrid) {
        a[0] = {(uint8_t*) st.k_q, (uint8_t*) sp.k_q, hd};
        a[1] = {(uint8_t*) st.k_scale, (uint8_t*) sp.k_scale, sc};
        a[2] = {(uint8_t*) st.v_q4, (uint8_t*) sp.v_q4, q4};
        return 3;
    }
    if (st.kv_q4) {
        a[0] = {(uint8_t*) st.k_q4, (uint8_t*) sp.k_q4, q4};
        a[1] = {(uint8_t*) st.v_q4, (uint8_t*) sp.v_q4, q4};
        return 2;
    }
    if (st.kv_int8) {
        a[0] = {(uint8_t*) st.k_q, (uint8_t*) sp.k_q, hd};
        a[1] = {(uint8_t*) st.v_q, (uint8_t*) sp.v_q, hd};
        a[2] = {(uint8_t*) st.k_scale, (uint8_t*) sp.k_scale, sc};
        a[3] = {(uint8_t*) st.v_scale, (uint8_t*) sp.v_scale, sc};
        return 4;
    }
    a[0] = {(uint8_t*) st.k_pool, (uint8_t*) sp.k_pool, hd * 2};
    a[1] = {(uint8_t*) st.v_pool, (uint8_t*) sp.v_pool, hd * 2};
    return 2;
}

// The prompt path's copies go through a ring of pinned host chunks: the primary's side of each piece is a DMA at its
// x16 speed on the caller's stream, the second card's side (PCIe x1, ~0.75 GB/s) runs on that card's own stream, so
// the slow half is off the primary's compute stream (cudaMemcpyPeerAsync without P2P held the stream for both halves:
// +0.9 s of a 20K-token prompt, 10-24).  Each chunk carries the last event of either side that used it; a piece waits
// for both before touching it.  STRATA_S31_PEER=1: the old peer copies (the A/B).
struct Bounce {
    static constexpr int kN = 16;
    static constexpr uint64_t kChunk = 4ull << 20;
    bool tried = false, ok = false;
    int dev = -1;
    cudaStream_t rs = nullptr;   // the second card's copy stream
    uint8_t* buf = nullptr;
    cudaEvent_t ev_p[kN] = {}, ev_r[kN] = {};   // primary-side / remote-side last use
    int next = 0, last_r = -1;
};
Bounce g_bounce;

bool peer_copies() {
    static const bool v = [] { const char* e = std::getenv("STRATA_S31_PEER"); return e != nullptr && e[0] == '1'; }();
    return v;
}

bool bounce_ready(int dev) {
    Bounce& b = g_bounce;
    if (b.tried) return b.ok;
    b.tried = true;
    b.dev = dev;
    if (host_alloc_guarded((void**) &b.buf, (size_t) (Bounce::kChunk * Bounce::kN), cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        b.buf = nullptr;
        std::fprintf(stderr, "strata: S-31: no pinned RAM for the prompt path's K/V ring: peer copies\n");
        return false;
    }
    for (int i = 0; i < Bounce::kN; ++i)
        if (cudaEventCreateWithFlags(&b.ev_p[i], cudaEventDisableTiming) != cudaSuccess) return false;
    {
        const OnDevice on(dev);
        if (cudaStreamCreateWithFlags(&b.rs, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); return false; }
        for (int i = 0; i < Bounce::kN; ++i)
            if (cudaEventCreateWithFlags(&b.ev_r[i], cudaEventDisableTiming) != cudaSuccess) { cudaGetLastError(); return false; }
    }
    b.ok = true;
    std::fprintf(stderr, "strata: S-31: the prompt path's K/V crosses through %d pinned chunks of %llu MiB\n", Bounce::kN,
                 (unsigned long long) (Bounce::kChunk >> 20));
    return true;
}

void fail(const char* what, cudaError_t e) {
    std::fprintf(stderr, "strata: S-31: the K/V %s copy failed: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}

void stage_copy(const QsaState& st, const strata::kernels::KvHostPools& sp, int64_t p0, int64_t p1, bool in,
                const strata::kernels::QsaShapes& s, void* stream) {
    if (p1 <= p0 || st.kv_dev < 0) return;
    int primary = 0;
    cudaGetDevice(&primary);
    Arr a[4];
    const int na = arrays(st, sp, s, a);
    const cudaStream_t ps = (cudaStream_t) stream;
    const bool ring = !peer_copies() && bounce_ready(st.kv_dev);
    Bounce& b = g_bounce;
    for (int i = 0; i < na; ++i) {
        const uint64_t page = a[i].row * (uint64_t) s.n_head_kv * (uint64_t) s.page_size;
        const uint64_t off = (uint64_t) p0 * page, bytes = (uint64_t) (p1 - p0) * page;
        if (!ring) {
            const cudaError_t e = in ? cudaMemcpyPeerAsync(a[i].stage + off, primary, a[i].remote + off, st.kv_dev, bytes, ps)
                                     : cudaMemcpyPeerAsync(a[i].remote + off, st.kv_dev, a[i].stage + off, primary, bytes, ps);
            if (e != cudaSuccess) fail(in ? "stage-in" : "stage-out", e);
            continue;
        }
        for (uint64_t o = 0; o < bytes; o += Bounce::kChunk) {
            const uint64_t len = std::min<uint64_t>(Bounce::kChunk, bytes - o);
            const int c = b.next;
            b.next = (b.next + 1) % Bounce::kN;
            uint8_t* h = b.buf + (uint64_t) c * Bounce::kChunk;
            uint8_t* rem = a[i].remote + off + o;
            uint8_t* stg = a[i].stage + off + o;
            cudaError_t e = cudaSuccess;
            if (!in) {   // primary -> chunk (fast), chunk -> second card (its stream)
                if ((e = cudaStreamWaitEvent(ps, b.ev_r[c], 0)) != cudaSuccess ||
                    (e = cudaStreamWaitEvent(ps, b.ev_p[c], 0)) != cudaSuccess ||
                    (e = cudaMemcpyAsync(h, stg, len, cudaMemcpyDeviceToHost, ps)) != cudaSuccess ||
                    (e = cudaEventRecord(b.ev_p[c], ps)) != cudaSuccess)
                    fail("stage-out", e);
                const OnDevice on(b.dev);
                if ((e = cudaStreamWaitEvent(b.rs, b.ev_p[c], 0)) != cudaSuccess ||
                    (e = cudaMemcpyAsync(rem, h, len, cudaMemcpyHostToDevice, b.rs)) != cudaSuccess ||
                    (e = cudaEventRecord(b.ev_r[c], b.rs)) != cudaSuccess)
                    fail("stage-out", e);
            } else {     // second card -> chunk (its stream, after every earlier stage-out), chunk -> primary
                {
                    const OnDevice on(b.dev);
                    if ((e = cudaStreamWaitEvent(b.rs, b.ev_p[c], 0)) != cudaSuccess ||
                        (e = cudaMemcpyAsync(h, rem, len, cudaMemcpyDeviceToHost, b.rs)) != cudaSuccess ||
                        (e = cudaEventRecord(b.ev_r[c], b.rs)) != cudaSuccess)
                        fail("stage-in", e);
                }
                if ((e = cudaStreamWaitEvent(ps, b.ev_r[c], 0)) != cudaSuccess ||
                    (e = cudaStreamWaitEvent(ps, b.ev_p[c], 0)) != cudaSuccess ||
                    (e = cudaMemcpyAsync(stg, h, len, cudaMemcpyHostToDevice, ps)) != cudaSuccess ||
                    (e = cudaEventRecord(b.ev_p[c], ps)) != cudaSuccess)
                    fail("stage-in", e);
            }
            b.last_r = c;
        }
    }
}

// the second card's stream `s` (a decode window's graph) after every prompt-path copy queued so far
void bounce_fence(cudaStream_t s) {
    const Bounce& b = g_bounce;
    if (b.ok && b.last_r >= 0) cudaStreamWaitEvent(s, b.ev_r[b.last_r], 0);
}

}  // namespace

int kv_remote_device() {
    static const int d = [] {
        const char* e = std::getenv("STRATA_S31_KV_DEV");
        return (e != nullptr && *e != '\0') ? std::atoi(e) : -1;
    }();
    return d;
}

bool kv_remote_half() {
    static const bool h = [] { const char* e = std::getenv("STRATA_S31_HALF"); return e != nullptr && e[0] == '1'; }();
    return h;
}

void* kv_remote_alloc(uint64_t bytes) {
    const int dev = kv_remote_device();
    if (dev < 0 || !ready_device(dev)) return nullptr;
    const OnDevice on(dev);
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        cudaGetLastError();
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        std::fprintf(stderr, "strata: S-31: CUDA%d cannot hold %.2f GiB more K/V (%.2f GiB placed, %.2f GiB free)\n", dev,
                     (double) bytes / 1073741824.0, (double) g_allocated.load() / 1073741824.0, (double) f / 1073741824.0);
        return nullptr;
    }
    g_allocated += bytes;
    return p;
}

uint64_t kv_remote_allocated() { return g_allocated.load(); }

void kv_remote_stage_in(const QsaState& st, const strata::kernels::KvHostPools& stage, int64_t n_pages,
                        const strata::kernels::QsaShapes& s, void* stream) {
    stage_copy(st, stage, 0, n_pages, true, s, stream);
}

void kv_remote_stage_out(const QsaState& st, const strata::kernels::KvHostPools& stage, int64_t p0, int64_t p1,
                         const strata::kernels::QsaShapes& s, void* stream) {
    stage_copy(st, stage, p0, p1, false, s, stream);
}

// ================================ the decode window's exchange ================================

KvRemoteWindow::~KvRemoteWindow() {
    if (dev_ < 0) return;
    release();
    const OnDevice on(dev_);
    if (stream_ != nullptr) cudaStreamSynchronize((cudaStream_t) stream_);
    for (void*& e : exec_)
        if (e != nullptr) { cudaGraphExecDestroy((cudaGraphExec_t) e); e = nullptr; }
    // the buffers live as long as the process (the session's do too)
}

bool KvRemoteWindow::init(int n_slots, int max_t, int64_t cap, int64_t attn_scratch_floats,
                          const strata::kernels::QsaShapes& s, std::string& err) {
    const int dev = kv_remote_device();
    if (dev < 0) return true;
    if (!ready_device(dev)) { err = "S-31: the K/V device is not usable"; return false; }
    if (max_t >= kMaxT) { err = "S-31: the window is longer than the K/V exchange holds"; return false; }
    n_slots_ = n_slots;
    max_t_ = max_t;
    cap_ = cap;
    scratch_floats_ = attn_scratch_floats;
    s_ = s;
    half_ = kv_remote_half();
    const uint64_t el = half_ ? 2 : 4, T = (uint64_t) max_t;
    const uint64_t nq = (uint64_t) s.n_head * s.head_dim, nkv = (uint64_t) s.n_head_kv * s.head_dim;
    off_k_ = kPayload + up256(T * nq * el);
    off_v_ = off_k_ + up256(T * nkv * el);
    off_sel_ = off_v_ + up256(T * nkv * el);
    off_step_ = off_sel_ + up256(T * (uint64_t) cap * 4);
    off_out_ = off_step_ + up256(T * kStepCount * 4);
    slot_bytes_ = off_out_ + up256(T * nq * el);
    if (host_alloc_guarded((void**) &box_, (size_t) (slot_bytes_ * (uint64_t) n_slots), cudaHostAllocMapped |
                           cudaHostAllocPortable) != cudaSuccess ||
        host_alloc_guarded((void**) &hdr_, 4096, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        err = "S-31: the K/V exchange's pinned mailbox could not be allocated";
        return false;
    }
    std::memset(box_, 0, (size_t) (slot_bytes_ * (uint64_t) n_slots));
    std::memset(hdr_, 0, 4096);
    // (one virtual address space: the mailbox's device address is its host address on both cards - checked)
    void *d0 = nullptr, *d1 = nullptr;
    if (cudaMalloc((void**) &cnt0_, 64) != cudaSuccess || cudaMemset(cnt0_, 0, 64) != cudaSuccess ||
        cudaHostGetDevicePointer(&d0, box_, 0) != cudaSuccess || d0 != (void*) box_) {
        cudaGetLastError();
        err = "S-31: the primary's side of the K/V exchange could not be set up";
        return false;
    }
    {
        const OnDevice on(dev);
        cudaStream_t st = nullptr;
        const uint64_t TF = T * nq;
        bool ok = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
                  cudaHostGetDevicePointer(&d1, box_, 0) == cudaSuccess && d1 == (void*) box_ &&
                  cudaMalloc((void**) &cnt1_, 64) == cudaSuccess && cudaMemset(cnt1_, 0, 64) == cudaSuccess &&
                  cudaMalloc((void**) &q1_, TF * 4) == cudaSuccess && cudaMalloc((void**) &attn1_, TF * 4) == cudaSuccess &&
                  cudaMalloc((void**) &k1_, T * nkv * 4) == cudaSuccess && cudaMalloc((void**) &v1_, T * nkv * 4) == cudaSuccess &&
                  cudaMalloc((void**) &sel1_, T * (uint64_t) cap * 4) == cudaSuccess &&
                  cudaMalloc((void**) &step1_, T * kStepCount * 4) == cudaSuccess &&
                  cudaMalloc((void**) &scratch1_, T * (uint64_t) attn_scratch_floats * 4 + 64) == cudaSuccess &&
                  cudaMemset(sel1_, 0, T * (uint64_t) cap * 4) == cudaSuccess &&
                  cudaMalloc((void**) &in1_, slot_bytes_) == cudaSuccess;
        if (ok && std::getenv("STRATA_S31_PROF") != nullptr)
            ok = cudaMalloc((void**) &prof1_, 128) == cudaSuccess && cudaMemset(prof1_, 0, 128) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            err = "S-31: the K/V device's side of the exchange could not be set up";
            return false;
        }
        stream_ = st;
    }
    dev_ = dev;
    std::fprintf(stderr, "strata: S-31: decode exchange: %d slots of %.0f KiB pinned, rows as %s\n", n_slots,
                 (double) slot_bytes_ / 1024.0, half_ ? "fp16" : "fp32");
    return true;
}

namespace {
// a T-token window's slot layout: its inputs packed from kPayload (one copy reads them), the output where init put it
Geo geo_of(int T, bool half, const strata::kernels::QsaShapes& s, int64_t cap, uint64_t off_out, uint64_t* in_end) {
    const uint64_t el = half ? 2 : 4, t = (uint64_t) T;
    const uint64_t nq = (uint64_t) s.n_head * s.head_dim, nkv = (uint64_t) s.n_head_kv * s.head_dim;
    Geo g{(int) nq, (int) nkv, cap, 0, 0, 0, 0, off_out};
    g.off_k = kPayload + up256(t * nq * el);
    g.off_v = g.off_k + up256(t * nkv * el);
    g.off_step = g.off_v + up256(t * nkv * el);
    g.off_sel = g.off_step + up256(t * kStepCount * 4);
    *in_end = g.off_sel + t * (uint64_t) cap * 4;
    return g;
}
}  // namespace

void KvRemoteWindow::publish(int slot, const float* q, const float* k, const float* v, const int32_t* sel,
                             const int32_t* step, int n, void* stream) {
    uint64_t in_end = 0;
    const Geo g = geo_of(n, half_, s_, cap_, off_out_, &in_end);
    uint8_t* p = box_ + (uint64_t) slot * slot_bytes_;
    if (half_) publish_kernel<true><<<16, 256, 0, (cudaStream_t) stream>>>(p, hdr_, g, q, k, v, sel, step, n, cnt0_);
    else publish_kernel<false><<<16, 256, 0, (cudaStream_t) stream>>>(p, hdr_, g, q, k, v, sel, step, n, cnt0_);
}

void KvRemoteWindow::collect(int slot, float* attn, int n, void* stream) {
    const Geo g{(int) (s_.n_head * s_.head_dim), (int) (s_.n_head_kv * s_.head_dim), cap_, off_k_, off_v_, off_sel_,
                off_step_, off_out_};
    uint8_t* p = box_ + (uint64_t) slot * slot_bytes_;
    wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>((const volatile uint32_t*) (p + kDone), hdr_);
    if (half_) collect_kernel<true><<<16, 256, 0, (cudaStream_t) stream>>>(p, g, attn, n);
    else collect_kernel<false><<<16, 256, 0, (cudaStream_t) stream>>>(p, g, attn, n);
}

bool KvRemoteWindow::capture(int T, const std::vector<std::pair<int, const QsaState*>>& layers, std::string& err) {
    using namespace strata::kernels;
    if (dev_ < 0 || captured(T)) return true;
    if (T < 1 || T > max_t_) { err = "S-31: window size out of range"; return false; }
    const OnDevice on(dev_);
    cudaStream_t cs = (cudaStream_t) stream_;
    uint64_t in_end = 0;
    const Geo g = geo_of(T, half_, s_, cap_, off_out_, &in_end);
    const int NKVHD = (int) (s_.n_head_kv * s_.head_dim);
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "S-31: the K/V device's capture could not begin";
        return false;
    }
    unsigned long long* pf = prof1_;
    if (pf) stamp_kernel<<<1, 1, 0, cs>>>(pf, 0);
    for (const auto& [slot, stp] : layers) {
        const QsaState& st = *stp;
        uint8_t* p = box_ + (uint64_t) slot * slot_bytes_;
        wait_kernel<<<1, 1, 0, cs>>>((const volatile uint32_t*) (p + kReq), hdr_);
        if (pf) stamp_kernel<<<1, 1, 0, cs>>>(pf, 1);
        // the payload in one DMA (the copy engine reads PCIe in large requests; the fetch kernel's own 2..4-byte loads
        // from mapped memory cost 2.4 ms a window on x1, 10-24), then unpacked from the card's own memory
        if (cudaMemcpyAsync(in1_ + kPayload, p + kPayload, in_end - kPayload, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            cudaStreamEndCapture(cs, nullptr);
            cudaGetLastError();
            err = "S-31: the K/V device's payload copy could not be captured";
            return false;
        }
        if (half_) fetch_kernel<true><<<8, 256, 0, cs>>>(in1_, g, q1_, k1_, v1_, sel1_, step1_, T);
        else fetch_kernel<false><<<8, 256, 0, cs>>>(in1_, g, q1_, k1_, v1_, sel1_, step1_, T);
        if (pf) stamp_kernel<<<1, 1, 0, cs>>>(pf, 2);
        // the window's cells into the pools, as verify.cpp's batched append writes them on the primary
        if (st.kv_hybrid) {
            kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.rtable, step1_, kStepCount, k1_, k1_, NKVHD, T,
                               s_, cs, nullptr);
            kv_append_q4_steps(st.v_q4, st.v_q4, st.rtable, step1_, kStepCount, T, v1_, v1_, s_, cs, nullptr);
        } else if (st.kv_q4) {
            kv_append_q4_steps(st.k_q4, st.v_q4, st.rtable, step1_, kStepCount, T, k1_, v1_, s_, cs, nullptr);
        } else if (st.kv_int8) {
            kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.rtable, step1_, kStepCount, k1_, v1_, NKVHD, T,
                               s_, cs, nullptr);
        } else {
            for (int t = 0; t < T; ++t)
                kv_append_step(st.k_pool, st.v_pool, st.rtable, step1_ + t * kStepCount, k1_ + (size_t) t * NKVHD,
                               v1_ + (size_t) t * NKVHD, s_, cs, nullptr);
        }
        if (pf) stamp_kernel<<<1, 1, 0, cs>>>(pf, 3);
        QsaAttnPools pools = qsa_attn_pools(st);
        pools.page_table = st.rtable;
        qsa_decode_attn_batch(q1_, pools, sel1_, step1_, cap_, s_, scratch1_, attn1_, T, cs);
        if (pf) stamp_kernel<<<1, 1, 0, cs>>>(pf, 4);
        if (half_) reply_kernel<true><<<8, 256, 0, cs>>>(p, hdr_, g, attn1_, T, cnt1_);
        else reply_kernel<false><<<8, 256, 0, cs>>>(p, hdr_, g, attn1_, T, cnt1_);
        if (pf) stamp_kernel<<<1, 1, 0, cs>>>(pf, 5);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    cudaGraphExec_t ex = nullptr;
    if (ce != cudaSuccess || graph == nullptr || cudaGraphInstantiate(&ex, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();
        err = std::string("S-31: the K/V device's window graph: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(ex, cs);
    exec_[T] = ex;
    return true;
}

bool KvRemoteWindow::launch(int T, std::string& err) {
    if (dev_ < 0) return true;
    if (!captured(T)) { err = "S-31: the K/V device's window graph was not captured"; return false; }
    const OnDevice on(dev_);
    cudaStream_t cs = (cudaStream_t) stream_;
    if (launched_ && cudaStreamQuery(cs) == cudaErrorNotReady) {   // an abandoned window's graph: let it out
        ((volatile uint32_t*) hdr_)[kAbort] = 1u;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        cudaStreamSynchronize(cs);
    }
    const cudaError_t pe = cudaGetLastError();
    if (pe != cudaSuccess) { err = std::string("S-31: the K/V device: ") + cudaGetErrorString(pe); return false; }
    if (prof1_ != nullptr && launched_ && (win_ & 255u) == 0u) {   // STRATA_S31_PROF: the last 256 windows
        unsigned long long h[10] = {};
        cudaStreamSynchronize(cs);
        if (cudaMemcpy(h, prof1_, sizeof h, cudaMemcpyDeviceToHost) == cudaSuccess && h[9] > 0) {
            const double w = (double) h[9] * 1e6;
            std::fprintf(stderr, "strata: S-31 prof: %llu windows, ms/window on CUDA%d: waiting %.2f fetch %.2f append %.2f "
                                 "attention %.2f reply %.2f\n", h[9], dev_, h[1] / w, h[2] / w, h[3] / w, h[4] / w, h[5] / w);
        }
        cudaMemset(prof1_, 0, sizeof h);
    }
    ((volatile uint32_t*) hdr_)[kAbort] = 0u;
    ((volatile uint32_t*) hdr_)[kWin] = ++win_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    bounce_fence(cs);   // the prompt's last pages are in the pools before this window reads them
    const cudaError_t le = cudaGraphLaunch((cudaGraphExec_t) exec_[T], cs);
    if (le != cudaSuccess) { err = std::string("S-31: the K/V device's launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs);   // WDDM: submit now, the primary's graph will wait for it
    launched_ = true;
    keep_warm_poke(dev_);
    return true;
}

void KvRemoteWindow::release() {
    if (hdr_ == nullptr) return;
    ((volatile uint32_t*) hdr_)[kAbort] = 1u;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

}  // namespace strata::core
