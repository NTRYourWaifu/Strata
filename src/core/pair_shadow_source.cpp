// src/core/pair_shadow_source.cpp - S-15 (fork): see the header.
//
// THE WINDOW is S-16's (ssd_shadow_source.cpp), over a cycle of pairs instead of whole layers: place p of the cycle is
// the p-th of the helper's (layer, expert) pairs in (layer, expert) order, the order the prompt path asks them in.  Slot
// s % W holds sequence number s (place s % P), the drive the manifest names reads it, and a slot is given back when the
// prompt path took its blob or passed it by (an expert no token of the chunk routed to: dropped once the asks are SLACK
// blobs past it).  An ask the window does not hold restarts the cycle there or is read on the spot; either way the
// bytes are the copy's, the same as experts.bin's (make_shadow.py compares them).
#include "strata/core/pair_shadow_source.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace {
constexpr int64_t kSlack = 48;   // asks this far past a blob drop it (the stager's threads ask a few out of order)

#if defined(_WIN32)
void* open_unbuffered(const std::string& path) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) std::max(n, 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    return h == INVALID_HANDLE_VALUE ? nullptr : (void*) h;
}
void close_handle(void* h) { if (h) CloseHandle((HANDLE) h); }
#endif
}  // namespace

bool PairShadowSource::read_at(void* h, uint64_t off, void* dst, uint64_t bytes) const {
#if defined(_WIN32)
    uint8_t* p = (uint8_t*) dst;
    while (bytes > 0) {
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        const DWORD want = (DWORD) std::min<uint64_t>(bytes, 64ull << 20);
        if (!ReadFile((HANDLE) h, p, want, &got, &ov) || got == 0) return false;
        p += got; off += got; bytes -= got;
    }
    return true;
#else
    (void) h; (void) off; (void) dst; (void) bytes;
    return false;
#endif
}

std::unique_ptr<PairShadowSource> PairShadowSource::open(ExpertSource* base, const std::string& manifest,
                                                         const std::function<bool(int64_t, int64_t)>& wanted,
                                                         int64_t window_mib, int threads, std::string& err) {
#if !defined(_WIN32)
    (void) base; (void) manifest; (void) wanted; (void) window_mib; (void) threads;
    err = "the SSD shadow source is Windows only (unbuffered ReadFile)";
    return nullptr;
#else
    std::unique_ptr<PairShadowSource> s(new PairShadowSource());
    s->base_ = base;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    s->E_ = lay.n_expert;
    std::ifstream f(manifest);
    if (!f) { err = "cannot read " + manifest; return nullptr; }
    std::string line;
    struct X { int64_t l, e; int d; uint64_t off; };
    std::vector<X> xs;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        std::string k;
        is >> k;
        if (k == "#") {
            std::string w;
            while (is >> w) if (w == "layers") is >> s->L_;
            if (s->L_ <= 0) { err = manifest + ": no layer count"; return nullptr; }
            s->bb_.assign((size_t) s->L_, 0);
            s->stride_.assign((size_t) s->L_, 0);
        } else if (k == "drive") {
            int i = 0;
            double wt = 0;
            is >> i >> wt;
            std::string p;
            std::getline(is, p);
            p.erase(0, p.find_first_not_of(' '));
            if (i != (int) s->drives_.size()) { err = "drive lines out of order in " + manifest; return nullptr; }
            auto d = std::make_unique<Drive>();
            d->path = p;
            d->weight = wt;
            s->drives_.push_back(std::move(d));
        } else if (k == "layer") {
            int64_t l = 0, bb = 0, st = 0;
            uint64_t off = 0;
            is >> l >> bb >> st >> off;
            if (l < 0 || l >= s->L_) continue;
            if ((uint64_t) bb != lay.blob_bytes(l) || off != lay.layer_offset(l)) {
                err = manifest + ": layer " + std::to_string(l) + " is not this pack's (blob or offset differs)";
                return nullptr;
            }
            if (st < bb || st % 4096 != 0) {
                err = manifest + ": layer " + std::to_string(l) + "'s stride is not 4 KiB aligned";
                return nullptr;
            }
            s->bb_[(size_t) l] = (uint64_t) bb;
            s->stride_[(size_t) l] = (uint64_t) st;
        } else if (k == "x") {
            X x{};
            is >> x.l >> x.e >> x.d >> x.off;
            xs.push_back(x);
        }
    }
    if (s->L_ <= 0 || s->drives_.empty()) { err = manifest + " has no layers or drives"; return nullptr; }
    std::sort(xs.begin(), xs.end(), [](const X& a, const X& b) { return a.l != b.l ? a.l < b.l : a.e < b.e; });
    s->idx_of_.assign((size_t) (s->L_ * s->E_), -1);
    for (const X& x : xs) {
        if (x.l < 0 || x.l >= s->L_ || x.e < 0 || x.e >= s->E_ || s->stride_[(size_t) x.l] == 0) {
            err = manifest + ": expert " + std::to_string(x.l) + "." + std::to_string(x.e) + " is out of range";
            return nullptr;
        }
        if (x.d < 0 || x.d >= (int) s->drives_.size() || x.off % 4096 != 0) {
            err = manifest + ": expert " + std::to_string(x.l) + "." + std::to_string(x.e) + " has no aligned place";
            return nullptr;
        }
        if (!wanted(x.l, x.e)) { ++s->skipped_; continue; }
        if (s->idx_of_[(size_t) (x.l * s->E_ + x.e)] >= 0) continue;
        s->idx_of_[(size_t) (x.l * s->E_ + x.e)] = (int32_t) s->layer_of_.size();
        s->layer_of_.push_back((int32_t) x.l);
        s->drive_of_.push_back((uint8_t) x.d);
        s->off_of_.push_back(x.off);
        s->max_stride_ = std::max(s->max_stride_, s->stride_[(size_t) x.l]);
    }
    s->P_ = (int64_t) s->layer_of_.size();
    if (s->P_ == 0) { err = manifest + ": none of its pairs is the helper's in this run"; return nullptr; }
    for (auto& d : s->drives_) {
        d->fallback = open_unbuffered(d->path);
        if (!d->fallback) { err = "cannot open " + d->path + " unbuffered"; return nullptr; }
    }
    // at least two slacks and a bit: an ask past a full window can then always drop the window's oldest blobs
    s->W_ = std::max<int64_t>(2 * kSlack + 16, (window_mib << 20) / (int64_t) s->max_stride_);
    // reserved only: prompt_begin commits it, prompt_end gives the pages back
    s->arena_ = (uint8_t*) VirtualAlloc(nullptr, (size_t) (s->W_ * (int64_t) s->max_stride_), MEM_RESERVE, PAGE_READWRITE);
    if (!s->arena_) { err = "the SSD window's buffers"; return nullptr; }
    s->win_.assign((size_t) s->W_, Entry{});
    for (size_t d = 0; d < s->drives_.size(); ++d) {
        const int n = threads > 0 ? threads : s->drives_[d]->weight < 2.0 ? 2 : 4;
        for (int t = 0; t < n; ++t) s->drives_[d]->workers.emplace_back([p = s.get(), d] { p->worker((int) d); });
    }
    return s;
#endif
}

PairShadowSource::~PairShadowSource() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_work_.notify_all();
    for (auto& d : drives_)
        for (auto& t : d->workers) t.join();
#if defined(_WIN32)
    for (auto& d : drives_) close_handle(d->fallback);
    if (arena_) VirtualFree(arena_, 0, MEM_RELEASE);
#endif
}

void PairShadowSource::prompt_begin() {
#if defined(_WIN32)
    std::lock_guard<std::mutex> lk(mu_);
    if (active_) return;
    if (!VirtualAlloc(arena_, (size_t) (W_ * (int64_t) max_stride_), MEM_COMMIT, PAGE_READWRITE)) return;   // asks read on the spot
    win_.assign((size_t) W_, Entry{});
    lo_ = next_ = 0;
    active_ = true;
    pump_locked();   // the first chunk's first layer
#endif
}

void PairShadowSource::prompt_end() {
#if defined(_WIN32)
    std::unique_lock<std::mutex> lk(mu_);
    if (!active_) return;
    active_ = false;
    for (auto& d : drives_) {   // not yet picked up: never read
        for (int64_t s : d->queue) {
            Entry& en = win_[(size_t) (s % W_)];
            en.state = 0;
            en.s = -1;
            --inflight_;
        }
        d->queue.clear();
    }
    for (Entry& en : win_)
        if (en.state == 1) en.state = 3;
        else if (en.state == 2) { en.state = 0; en.s = -1; }
    cv_ready_.wait(lk, [&] { return inflight_ == 0; });
    VirtualFree(arena_, (size_t) (W_ * (int64_t) max_stride_), MEM_DECOMMIT);
    lk.unlock();
    if (st_blobs_.load() + st_direct_.load() > 0) std::fprintf(stderr, "strata s15: %s\n", take_stats().c_str());
#endif
}

void PairShadowSource::pump_locked() {
    if (!active_) return;
    bool any = false;
    while (next_ - lo_ < W_) {
        Entry& en = win_[(size_t) (next_ % W_)];
        if (en.state != 0) break;   // its previous blob is still being read (dropped by a reset)
        en.s = next_;
        en.state = 1;
        en.ok = true;
        drives_[drive_of_[(size_t) (next_ % P_)]]->queue.push_back(next_);
        ++inflight_;
        ++next_;
        any = true;
    }
    if (any) cv_work_.notify_all();
}

void PairShadowSource::advance_locked() {
    while (lo_ < next_) {
        const Entry& en = win_[(size_t) (lo_ % W_)];
        if (en.s == lo_ && en.state != 0) break;
        ++lo_;
    }
    pump_locked();
}

void PairShadowSource::worker(int d) {
#if defined(_WIN32)
    Drive& dr = *drives_[(size_t) d];
    void* h = open_unbuffered(dr.path);
    for (;;) {
        int64_t s = -1;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return quit_ || !dr.queue.empty(); });
            if (quit_) break;
            s = dr.queue.front();
            dr.queue.pop_front();
        }
        const int64_t p = s % P_;
        const uint64_t st = stride_[(size_t) layer_of_[(size_t) p]];
        uint8_t* dst = arena_ + (size_t) (s % W_) * max_stride_;
        const bool ok = h != nullptr && read_at(h, off_of_[(size_t) p], dst, st);
        if (ok) dr.bytes.fetch_add(st, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            Entry& en = win_[(size_t) (s % W_)];
            --inflight_;
            if (en.state == 3) {   // dropped while it was read: the slot is free now
                en.state = 0;
                en.s = -1;
                advance_locked();
            } else {
                en.state = 2;
                en.ok = ok;
            }
        }
        cv_ready_.notify_all();
    }
    close_handle(h);
#else
    (void) d;
#endif
}

bool PairShadowSource::read_direct(int64_t p, uint8_t* dst) {
#if defined(_WIN32)
    const int64_t l = layer_of_[(size_t) p];
    const uint64_t st = stride_[(size_t) l];
    uint8_t* tmp = (uint8_t*) VirtualAlloc(nullptr, (size_t) st, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!tmp) return false;
    bool ok;
    {
        std::lock_guard<std::mutex> lk(mu_);   // the drive's one fallback handle
        ok = read_at(drives_[drive_of_[(size_t) p]]->fallback, off_of_[(size_t) p], tmp, st);
    }
    if (ok) std::memcpy(dst, tmp, (size_t) bb_[(size_t) l]);
    VirtualFree(tmp, 0, MEM_RELEASE);
    st_direct_.fetch_add(1, std::memory_order_relaxed);
    return ok;
#else
    (void) p; (void) dst;
    return false;
#endif
}

bool PairShadowSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (!mine(layer, expert)) return base_->copy_blob(layer, expert, dst);
    if (dst == nullptr) return false;
    const int64_t p = idx_of_[(size_t) (layer * E_ + expert)];
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    if (!active_) {   // outside a prompt the window holds no RAM
        lk.unlock();
        return read_direct(p, dst);
    }
    // the sequence number this ask means: the cycle's next occurrence of p at or after the window's start, less the
    // slack an out-of-order ask may trail by
    int64_t s = lo_ - (lo_ % P_) + p;
    while (s < lo_ - kSlack) s += P_;
    if (s < lo_) {   // already passed by: read it on the spot
        lk.unlock();
        return read_direct(p, dst);
    }
    if (s >= next_ + W_) {
        // far ahead of the window (a prompt that stopped in the middle of a layer, the first ask in a long while):
        // drop the window and restart the cycle at p
        for (Entry& en : win_)
            if (en.state == 1) en.state = 3;
            else if (en.state == 2) { en.state = 0; en.s = -1; }
        lo_ = next_ = s;
        st_resets_.fetch_add(1, std::memory_order_relaxed);
        pump_locked();   // slots still being read stay out of use until their reads land
    }
    // the asks have moved past these: give their slots back
    for (int64_t q = lo_; q < std::min(next_, s - kSlack); ++q) {
        Entry& en = win_[(size_t) (q % W_)];
        if (en.s != q) continue;
        if (en.state == 1) en.state = 3;
        else if (en.state == 2) { en.state = 0; en.s = -1; }
    }
    advance_locked();
    // issue up to s (a full window waits for the asks before s to give slots back)
    for (;;) {
        if (s < next_) {
            Entry& en = win_[(size_t) (s % W_)];
            if (en.s != s || en.state == 3) {   // its slot was dropped meanwhile
                lk.unlock();
                return read_direct(p, dst);
            }
            if (en.state == 2) break;
        } else {
            pump_locked();
        }
        cv_ready_.wait(lk);
    }
    Entry& en = win_[(size_t) (s % W_)];
    const bool ok = en.ok;
    const uint8_t* src = arena_ + (size_t) (s % W_) * max_stride_;
    en.state = 4;   // being copied out: no one frees or refills it
    lk.unlock();
    if (ok) std::memcpy(dst, src, (size_t) bb_[(size_t) layer]);
    lk.lock();
    en.state = 0;
    en.s = -1;
    advance_locked();
    lk.unlock();
    cv_ready_.notify_all();
    st_blobs_.fetch_add(1, std::memory_order_relaxed);
    st_wait_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
    return ok || read_direct(p, dst);
}

bool PairShadowSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    // the base's runs stay runs (it may merge neighbouring blobs into one read); this source's go one by one
    std::vector<int32_t> bl, be;
    std::vector<uint8_t*> bd;
    for (size_t i = 0; i < n; ++i) {
        if (mine(layers[i], experts[i])) {
            if (!copy_blob(layers[i], experts[i], dst[i])) return false;
        } else {
            bl.push_back(layers[i]);
            be.push_back(experts[i]);
            bd.push_back(dst[i]);
        }
    }
    return bl.empty() || base_->copy_blobs(bl.data(), be.data(), bd.data(), bl.size());
}

const uint8_t* PairShadowSource::blob(int64_t layer, int64_t expert) {
    if (!mine(layer, expert)) return base_->blob(layer, expert);
    // the prompt path copies this source's blobs (transient); a short-lived copy for anyone else
    thread_local std::vector<uint8_t> tmp;
    tmp.resize((size_t) bb_[(size_t) layer]);
    return read_direct(idx_of_[(size_t) (layer * E_ + expert)], tmp.data()) ? tmp.data() : nullptr;
}

const uint8_t* PairShadowSource::blob_stable(int64_t layer, int64_t expert) {
    return mine(layer, expert) ? nullptr : base_->blob_stable(layer, expert);
}

int PairShadowSource::blob_parts(int64_t layer, int64_t expert, BlobPart* out) {
    return mine(layer, expert) ? 0 : base_->blob_parts(layer, expert, out);
}

void PairShadowSource::others(int64_t layer, const int64_t* experts, int64_t n, std::vector<int64_t>& ids,
                              std::vector<int64_t>& at) const {
    ids.clear();
    at.clear();
    for (int64_t i = 0; i < n; ++i)
        if (!mine(layer, experts[i])) { ids.push_back(experts[i]); at.push_back(i); }
}

void PairShadowSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    std::vector<int64_t> ids, at;
    others(layer, experts, n, ids, at);
    if (!ids.empty()) base_->prefetch(layer, ids.data(), (int64_t) ids.size());
}

int64_t PairShadowSource::prefetch_begin(int64_t layer, const int64_t* experts, int64_t n, uint8_t* pending) {
    std::vector<int64_t> ids, at;
    others(layer, experts, n, ids, at);
    for (int64_t i = 0; i < n; ++i) pending[i] = 0;   // this source's: copy_blob waits on the window
    if (ids.empty()) return 0;
    std::vector<uint8_t> sub(ids.size(), 0);
    const int64_t got = base_->prefetch_begin(layer, ids.data(), (int64_t) ids.size(), sub.data());
    for (size_t j = 0; j < at.size(); ++j) pending[at[j]] = sub[j];
    return got;
}

int64_t PairShadowSource::wait_landed(int64_t layer, const int64_t* experts, int64_t n, uint8_t* state) {
    std::vector<int64_t> ids, at;
    others(layer, experts, n, ids, at);
    if (ids.empty()) return 0;
    std::vector<uint8_t> sub(ids.size());
    for (size_t j = 0; j < at.size(); ++j) sub[j] = state[at[j]];
    const int64_t got = base_->wait_landed(layer, ids.data(), (int64_t) ids.size(), sub.data());
    for (size_t j = 0; j < at.size(); ++j) state[at[j]] = sub[j];
    return got;
}

void PairShadowSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    std::vector<int64_t> ids, at;
    others(layer, experts, n, ids, at);
    if (!ids.empty()) base_->warm(layer, ids.data(), (int64_t) ids.size());
}

void PairShadowSource::warm_scored(int64_t layer, const int64_t* experts, const float* conf, int64_t n, int ahead) {
    std::vector<int64_t> ids, at;
    others(layer, experts, n, ids, at);
    if (ids.empty()) return;
    std::vector<float> c(ids.size());
    for (size_t j = 0; j < at.size(); ++j) c[j] = conf[at[j]];
    base_->warm_scored(layer, ids.data(), c.data(), (int64_t) ids.size(), ahead);
}

std::string PairShadowSource::take_stats() {
    const int64_t n = st_blobs_.exchange(0), dir = st_direct_.exchange(0), rs = st_resets_.exchange(0);
    const uint64_t wus = st_wait_us_.exchange(0);
    uint64_t tot = 0;
    std::vector<uint64_t> by;
    for (auto& d : drives_) { by.push_back(d->bytes.exchange(0)); tot += by.back(); }
    char b[256];
    std::snprintf(b, sizeof b, "%lld blobs from the window (%.2f GiB read in all), asks waited %.0f ms in all, %lld read on the "
                               "spot, %lld restarts; drives", (long long) n, (double) tot / 1073741824.0, (double) wus / 1000.0,
                  (long long) dir, (long long) rs);
    std::string s = b;
    for (size_t i = 0; i < by.size(); ++i) {
        std::snprintf(b, sizeof b, " %c:%.0f%%", drives_[i]->path.empty() ? '?' : drives_[i]->path[0],
                      tot ? 100.0 * (double) by[i] / (double) tot : 0.0);
        s += b;
    }
    return s;
}

}  // namespace strata::core
