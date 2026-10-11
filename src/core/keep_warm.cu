// src/core/keep_warm.cu - a sleeping kernel that keeps a lightly loaded card at its compute clocks (keep_warm.hpp).
#include "strata/core/keep_warm.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace strata::core {

#if defined(STRATA_USE_HIP)
void keep_warm_poke(int) {}
#else
namespace {

using Clock = std::chrono::steady_clock;

constexpr auto KEEP = std::chrono::milliseconds(1500);   // after the last window: a turn's gaps are shorter
// One launch sleeps 2 ms, then the thread looks again: a cudaDeviceSynchronize on the card (a checkpoint, a request's
// end) waits for it, so it is short.
constexpr unsigned long long SPIN_NS = 2000000ull;
constexpr int MAX_DEV = 64;

// Sleeps on the card for `ns`: one thread, no memory traffic, but the governor counts the card busy.
__global__ void keep_warm_kernel(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100000);
#endif
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

struct Warm {
    std::mutex m;
    std::condition_variable cv;
    Clock::time_point until{};
};

// never freed: the detached thread may still wait on its condition variable while the process exits
Warm* g_warm[MAX_DEV];
std::atomic<int> g_state[MAX_DEV];   // 0 not decided, 1 on (thread started), -1 off
std::mutex g_decide;

bool wanted(int dev, std::string* name) {
    const char* env = std::getenv("STRATA_KEEP_WARM");
    if (env != nullptr && env[0] == '0') return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    *name = p.name;
    if (env != nullptr && env[0] == '1') return true;
    return p.major == 7 && p.minor == 5 && std::strstr(p.name, "GTX 16") != nullptr;
}

void run(int dev, Warm* w) {
    cudaSetDevice(dev);
    int lo = 0, hi = 0;
    cudaStream_t s = nullptr;
    cudaEvent_t done = nullptr;   // a blocking-sync event: the thread sleeps in the wait instead of spinning a core
    if (cudaDeviceGetStreamPriorityRange(&lo, &hi) != cudaSuccess ||
        cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, lo) != cudaSuccess ||
        cudaEventCreateWithFlags(&done, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "strata keep-warm: CUDA%d: no stream of its own, left alone\n", dev);
        return;
    }
    std::unique_lock<std::mutex> lk(w->m);
    for (;;) {
        w->cv.wait(lk, [&] { return Clock::now() < w->until; });
        lk.unlock();
        keep_warm_kernel<<<1, 1, 0, s>>>(SPIN_NS);
        cudaEventRecord(done, s);
        const cudaError_t e = cudaEventSynchronize(done);
        if (e != cudaSuccess) {
            cudaGetLastError();
            std::fprintf(stderr, "strata keep-warm: CUDA%d: %s, stopped\n", dev, cudaGetErrorString(e));
            return;
        }
        lk.lock();
    }
}

}  // namespace

void keep_warm_poke(int dev) {
    if (dev < 0 || dev >= MAX_DEV) return;
    int st = g_state[dev].load(std::memory_order_acquire);
    if (st == 0) {
        std::lock_guard<std::mutex> g(g_decide);
        st = g_state[dev].load(std::memory_order_relaxed);
        if (st == 0) {
            std::string name;
            if (wanted(dev, &name)) {
                g_warm[dev] = new Warm();
                std::thread(run, dev, g_warm[dev]).detach();
                std::fprintf(stderr, "strata keep-warm: CUDA%d (%s) is kept at its compute clocks while windows run "
                             "(STRATA_KEEP_WARM=0: off)\n", dev, name.c_str());
                st = 1;
            } else {
                st = -1;
            }
            g_state[dev].store(st, std::memory_order_release);
        }
    }
    if (st != 1) return;
    Warm* w = g_warm[dev];
    {
        std::lock_guard<std::mutex> lk(w->m);
        w->until = Clock::now() + KEEP;
    }
    w->cv.notify_one();
}
#endif

}  // namespace strata::core
