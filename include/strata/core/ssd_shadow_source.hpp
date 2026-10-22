// include/strata/core/ssd_shadow_source.hpp - S-16 (fork): the first layers' experts read from SSD copies.
//
// A layer split whose first card is slow at prompts (a GTX 1660 SUPER: no tensor cores, a x1 link) can have the
// next card read those layers too, for long prompts only (Prefill::set_offload).  That card holds none of their
// experts, and RAM need not hold them either: this source reads them from copies of their blobs spread over
// several drives (tools: 測試\S16_SSD讀題\make_shadow.py), in the drives' speed ratio, unbuffered, many reads in
// flight, into a small window of host buffers that runs ahead of the prompt path - while layer N is computed the
// window already holds layer N + 1's first experts, and at the end of a chunk the next chunk's first layer.  Every
// other layer is the base source's, unchanged.
#pragma once

#include "strata/core/expert_source.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class SsdShadowSource : public ExpertSource {
public:
    /// `manifest`: the s16_shadow.txt that make_shadow.py wrote; layers [0, `layers`) are read from the copies (the
    /// manifest must hold them).  `window_mib`: the read-ahead window; `threads`: reads in flight per drive (0: 4 per
    /// drive, 2 for a drive of weight under 2).  Null with `err` set when a file is missing or does not match.
    static std::unique_ptr<SsdShadowSource> open(ExpertSource* base, const std::string& manifest, int64_t layers,
                                                 int64_t window_mib, int threads, std::string& err);
    ~SsdShadowSource() override;

    int64_t shadow_layers() const { return L_; }
    /// The window holds RAM only while an offloaded prompt reads: begin_prompt commits its buffers and starts reading
    /// the cycle from its start, end_prompt waits for the reads in flight and gives the pages back to Windows (the
    /// decode then has them: the engine leaves only a few hundred MiB free on a squeezed PC).
    void begin_prompt();
    void end_prompt();
    /// One line: blobs read, GiB, the prompt path's waits on the window, the drives' shares (and resets the counters).
    std::string take_stats();

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override;
    int blob_parts(int64_t layer, int64_t expert, BlobPart* out) override;
    void note_async_read(const uint8_t* blob, void* stream) override { base_->note_async_read(blob, stream); }
    int64_t reads() const override { return base_->reads(); }
    bool pinned(int64_t layer, int64_t expert) const override { return layer >= L_ && base_->pinned(layer, expert); }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override {
        if (layer >= L_) base_->begin_layer(layer, ids, k);
    }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return layer >= L_ ? base_->device_alias(layer, expert) : nullptr;
    }
    void prefetch(int64_t layer, int64_t expert) override { if (layer >= L_) base_->prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override { return layer >= L_ ? base_->release(layer, expert) : 0; }
    bool pcie_layer(int64_t layer) const override { return layer >= L_ && base_->pcie_layer(layer); }
    bool transient(int64_t layer, int64_t expert) const override { return layer < L_ || base_->transient(layer, expert); }
    /// The prompt path's stager picks its SSD profile (32 threads) when a source is transient; the layers this source
    /// adds bring their own reads in flight, and the base's blobs are memcpy from RAM: keep the RAM profile.
    double cached_share(int64_t samples) const override { (void) samples; return 1.0; }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override { return base_->advise_pairs(pairs, n); }
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override { if (layer >= L_) base_->prefetch(layer, experts, n); }
    int64_t prefetch_begin(int64_t layer, const int64_t* experts, int64_t n, uint8_t* pending) override {
        if (layer >= L_) return base_->prefetch_begin(layer, experts, n, pending);
        return ExpertSource::prefetch_begin(layer, experts, n, pending);
    }
    int64_t wait_landed(int64_t layer, const int64_t* experts, int64_t n, uint8_t* state) override {
        if (layer >= L_) return base_->wait_landed(layer, experts, n, state);
        return ExpertSource::wait_landed(layer, experts, n, state);
    }
    void prefetch_end() override { base_->prefetch_end(); }
    bool read_ahead_open() const override { return base_->read_ahead_open(); }
    bool linux_file_defaults() const override { return base_->linux_file_defaults(); }
    bool zc_roles(int64_t layer, int64_t expert, const uint8_t** p) override {
        return layer >= L_ && base_->zc_roles(layer, expert, p);
    }
    void warm(int64_t layer, const int64_t* experts, int64_t n) override { if (layer >= L_) base_->warm(layer, experts, n); }
    void warm_scored(int64_t layer, const int64_t* experts, const float* conf, int64_t n, int ahead) override {
        if (layer >= L_) base_->warm_scored(layer, experts, conf, n, ahead);
    }
    bool wants_scores() const override { return base_->wants_scores(); }
    bool warms() const override { return base_->warms(); }

private:
    SsdShadowSource() = default;
    struct Drive {
        std::string path;
        double weight = 0;
        std::deque<int64_t> queue;          // window sequence numbers to read, in order
        std::vector<std::thread> workers;
        void* fallback = nullptr;           // HANDLE for the out-of-window reads (under mu_)
        std::atomic<uint64_t> bytes{0};
    };
    struct Entry {
        int64_t s = -1;                     // window sequence number held (-1: none)
        int state = 0;                      // 0 free, 1 reading, 2 ready, 3 dropped while reading
        bool ok = true;
    };
    bool read_at(void* h, uint64_t off, void* dst, uint64_t bytes) const;
    bool read_direct(int64_t layer, int64_t expert, uint8_t* dst);   // out of the window, synchronous
    void pump_locked();                     // issue reads while the window has room
    void advance_locked();                  // lo_ past the consumed / dropped entries
    void worker(int d);

    ExpertSource* base_ = nullptr;
    int64_t L_ = 0, E_ = 0, P_ = 0;         // shadow layers, experts per layer, blobs per cycle (L_ x E_)
    std::vector<uint64_t> bb_, stride_;     // per layer
    std::vector<uint8_t> drive_of_;         // per (layer, expert)
    std::vector<uint64_t> off_of_;
    std::vector<std::unique_ptr<Drive>> drives_;
    uint64_t max_stride_ = 0;
    int64_t W_ = 0;                         // window slots
    uint8_t* arena_ = nullptr;              // W_ x max_stride_, page aligned
    std::vector<Entry> win_;
    int64_t lo_ = 0, next_ = 0;             // the window holds sequence numbers [lo_, next_)
    int64_t inflight_ = 0;
    bool quit_ = false;
    bool active_ = false;                   // between begin_prompt and end_prompt (the arena committed)
    std::mutex mu_;
    std::condition_variable cv_work_, cv_ready_;
    // stats (since take_stats)
    std::atomic<int64_t> st_blobs_{0}, st_direct_{0}, st_resets_{0};
    std::atomic<uint64_t> st_wait_us_{0};
};

}  // namespace strata::core
