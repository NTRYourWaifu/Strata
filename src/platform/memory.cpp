// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstring>
#else
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    uint64_t total = 0;
    for (const auto& g : regions) total += g.second;
    if (total == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // same growth as lock_resident, but for the sum of the regions and with ONE margin
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) total + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    for (const auto& g : regions) {
        uint8_t* base = (uint8_t*) g.first;
        for (uint64_t off = 0; off < g.second; off += chunk) {
            const uint64_t n = g.second - off < chunk ? g.second - off : chunk;
            if (!VirtualLock(base + off, (SIZE_T) n)) {
                r.note = "VirtualLock stopped after " + std::to_string((unsigned long long) (r.locked_bytes >> 20)) +
                         " of " + std::to_string((unsigned long long) (total >> 20)) + " MiB (error " +
                         std::to_string(GetLastError()) + ")";
                r.ok = r.locked_bytes > 0;
                return r;
            }
            r.locked_bytes += n;
        }
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (total >> 20)) + " MiB in " +
             std::to_string(regions.size()) + " regions via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

void soften_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    // Only VirtualUnlock: the minimum lock_resident raised stays, and it is a soft one (QUOTA_LIMITS_HARDWS_MIN_DISABLE),
    // so the memory manager trims below it when another program needs the RAM - oldest pages first, which are these.
    // Lowering it instead made relock_resident's raise fail (error 1450) once the others had the RAM (10-13).
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

void evict_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 256ull << 20;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));   // ERROR_NOT_LOCKED is expected
}

uint64_t missing_resident(const void* p, uint64_t bytes, uint32_t stride) {
    if (p == nullptr || bytes == 0) return 0;
    if (stride == 0) stride = 1;
    const uint64_t step = (uint64_t) stride * 4096;
    const size_t n = (size_t) ((bytes + step - 1) / step);
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info(n);
    for (size_t i = 0; i < n; ++i) info[i].VirtualAddress = (PVOID) ((const uint8_t*) p + i * step);
    if (!K32QueryWorkingSetEx(GetCurrentProcess(), info.data(), (DWORD) (n * sizeof(info[0])))) return 0;
    uint64_t out = 0;
    for (const auto& e : info) if (!e.VirtualAttributes.Valid) ++out;
    out *= step;
    return out < bytes ? out : bytes;
}

LockResult relock_resident(void* p, uint64_t bytes, const std::function<void(uint64_t, uint64_t)>& progress) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    uint8_t* base = (uint8_t*) p;
    const uint64_t chunk = 1ull << 30;
    // the trimmed pages back from the page file in large reads first (VirtualLock alone faults them one cluster at a
    // time); skipped when (almost) nothing was trimmed - then only the locking is left
    using Prefetch = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const Prefetch prefetch = (Prefetch) (void*) GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    const bool fetch = prefetch != nullptr && missing_resident(p, bytes) >= (64ull << 20);
    if (fetch) {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        for (uint64_t off = 0; off < bytes; off += chunk)
            ranges.push_back({base + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk)});
        (void) prefetch(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
    }
    // the minimum is still raised (soften_resident keeps it): only the pages
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " + std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
        if (progress) progress(r.locked_bytes, bytes);
    }
    r.ok = true;
    r.note = std::string("locked again via VirtualLock") + (fetch ? " after PrefetchVirtualMemory" : "");
    return r;
}

uint64_t make_room(uint64_t want, std::string& note) {
    MEMORYSTATUSEX ms{};
    auto avail = [&]() -> uint64_t {
        ms.dwLength = sizeof ms;
        return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPhys : 0;
    };
    const uint64_t before = avail();
    const uint64_t keep = 3ull << 30;
    if (ms.ullTotalPhys > keep && want > ms.ullTotalPhys - keep) want = ms.ullTotalPhys - keep;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t now = before;
    int rounds = 0;
    // The block is the whole target, not the shortfall: its first pages come from the free and standby lists (which
    // already count as available), and only the rest makes the memory manager trim the other working sets.  Freed,
    // all of it is free.  A few rounds: a program that is running pulls part of its pages back at once.
    for (; rounds < 3 && now < want; ++rounds) {
        const uint64_t need = want + (512ull << 20);
        uint8_t* b = (uint8_t*) VirtualAlloc(nullptr, (SIZE_T) need, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (b == nullptr) {
            note = "VirtualAlloc of " + std::to_string((unsigned long long) (need >> 20)) + " MiB failed (error " +
                   std::to_string(GetLastError()) + "); ";
            break;
        }
        for (uint64_t off = 0; off < need; off += 4096) ((volatile uint8_t*) b)[off] = 1;
        VirtualFree(b, 0, MEM_RELEASE);
        now = avail();
    }
    char msg[200];
    std::snprintf(msg, sizeof msg, "available RAM %.2f -> %.2f GiB (wanted %.2f) in %.1f s, %d round%s",
                  (double) before / 1073741824.0, (double) now / 1073741824.0, (double) want / 1073741824.0,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), rounds,
                  rounds == 1 ? "" : "s");
    note += msg;
    return now;
}

uint64_t process_working_set() {
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof pmc;
    return K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? (uint64_t) pmc.WorkingSetSize : 0;
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    for (const auto& g : regions) {
        if (g.first == nullptr || g.second == 0) continue;
        if (mlock(g.first, g.second) != 0) { r.note = "mlock failed (raise ulimit -l)"; r.ok = r.locked_bytes > 0; return r; }
        r.locked_bytes += g.second;
    }
    r.ok = r.locked_bytes > 0;
    r.note = r.ok ? "mlock of " + std::to_string(regions.size()) + " regions" : "nothing to lock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

void soften_resident(void* p, uint64_t bytes) { unlock_resident(p, bytes); }

LockResult relock_resident(void* p, uint64_t bytes, const std::function<void(uint64_t, uint64_t)>&) {
    return lock_resident(p, bytes);
}

void evict_resident(void*, uint64_t) {}

uint64_t missing_resident(const void*, uint64_t, uint32_t) { return 0; }

uint64_t make_room(uint64_t, std::string& note) {
    note = "make_room is Windows-only";
    const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

uint64_t process_working_set() { return 0; }

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
#if defined(__APPLE__)   // macOS: F_RDADVISE is the read-ahead hint
    {
        radvisory ra{(off_t) (offset + at), (int) std::min(kAdviseStep, bytes - at)};
        (void) fcntl(fd, F_RDADVISE, &ra);
    }
#else
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
#endif
}
#endif

ProcIo proc_io_sample() {
    ProcIo r;
#if defined(__linux__)
    if (std::FILE* f = std::fopen("/proc/self/io", "r")) {
        char key[64];
        unsigned long long v;
        while (std::fscanf(f, "%63[^:]: %llu\n", key, &v) == 2)
            if (std::strcmp(key, "read_bytes") == 0) { r.read_bytes = v; r.valid = true; }
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        if (const char* p = std::strrchr(buf, ')')) {   // fields after the command: state ppid ... minflt cminflt majflt
            unsigned long long minflt = 0, cminflt = 0, majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %llu %llu %llu", &minflt, &cminflt, &majflt) == 3)
                r.major_faults = majflt;
        }
    }
#endif
    return r;
}

}  // namespace strata::platform
