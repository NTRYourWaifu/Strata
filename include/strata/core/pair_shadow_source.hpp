// include/strata/core/pair_shadow_source.hpp - S-15 (fork): the helper cache's experts read from SSD copies.
//
// A helper cache (--expert-cache-device1) holds its experts in a second card's VRAM only: the resident RAM mode leaves
// them out of the RAM copy, so every prompt chunk reads them from experts.bin - about 4.5 GiB a 512-token chunk with a
// 6 GB GTX 1660 SUPER as the helper, which one drive reads at ~4 GB/s, slower than the chunk's other work.  This source
// reads them from copies spread over several drives (tools: 測試\S15_副卡專家\make_shadow.py, from the pairs the engine
// wrote with STRATA_S15_HELPER_DUMP), in the drives' speed ratio, unbuffered, many reads in flight, into a small
// window of host buffers that runs ahead of the prompt path.  The prompt path asks for the helper's experts layer after
// layer, in id order, chunk after chunk: one fixed cycle of the pairs, which the window reads ahead (as S-16's
// SsdShadowSource does for whole layers).  Every other pair is the base source's, unchanged.
#pragma once

#include "strata/core/expert_source.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class PairShadowSource : public ExpertSource {
public:
    /// `manifest`: the s15_shadow.txt that make_shadow.py wrote; the pairs it lists that `wanted` accepts form the
    /// cycle (the helper's experts: the others are in RAM or VRAM, and the prompt path never asks the drive for them).
    /// `window_mib`: the read-ahead window; `threads`: reads in flight per drive (0: 4 per drive, 2 for a drive of
    /// weight under 2).  Null with `err` set when a file is missing or does not match.
    static std::unique_ptr<PairShadowSource> open(ExpertSource* base, const std::string& manifest,
                                                  const std::function<bool(int64_t, int64_t)>& wanted,
                                                  int64_t window_mib, int threads, std::string& err);
    ~PairShadowSource() override;

    int64_t pairs() const { return P_; }
    int64_t listed_not_wanted() const { return skipped_; }
    /// The window holds RAM only while a prompt reads (Prefill::run): the decode then has the pages (the engine leaves
    /// only a few hundred MiB free on a squeezed PC).
    void prompt_begin() override;
    void prompt_end() override;
    /// One line: blobs read, GiB, the prompt path's waits on the window, the drives' shares (and resets the counters).
    std::string take_stats();

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override;
    int blob_parts(int64_t layer, int64_t expert, BlobPart* out) override;
    void note_async_read(const uint8_t* blob, void* stream) override { base_->note_async_read(blob, stream); }
    int64_t reads() const override { return base_->reads(); }
    bool pinned(int64_t layer, int64_t expert) const override { return !mine(layer, expert) && base_->pinned(layer, expert); }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override { base_->begin_layer(layer, ids, k); }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return mine(layer, expert) ? nullptr : base_->device_alias(layer, expert);
    }
    void prefetch(int64_t layer, int64_t expert) override { if (!mine(layer, expert)) base_->prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override { return mine(layer, expert) ? 0 : base_->release(layer, expert); }
    bool pcie_layer(int64_t layer) const override { return base_->pcie_layer(layer); }
    bool transient(int64_t layer, int64_t expert) const override { return mine(layer, expert) || base_->transient(layer, expert); }
    /// The prompt path's stager picks its SSD profile (32 threads) when a source is transient; the pairs this source
    /// adds bring their own reads in flight, and the base's blobs are memcpy from RAM: keep the RAM profile.
    double cached_share(int64_t samples) const override { (void) samples; return 1.0; }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override { return base_->advise_pairs(pairs, n); }
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    int64_t prefetch_begin(int64_t layer, const int64_t* experts, int64_t n, uint8_t* pending) override;
    int64_t wait_landed(int64_t layer, const int64_t* experts, int64_t n, uint8_t* state) override;
    void prefetch_end() override { base_->prefetch_end(); }
    bool read_ahead_open() const override { return base_->read_ahead_open(); }
    bool linux_file_defaults() const override { return base_->linux_file_defaults(); }
    bool zc_roles(int64_t layer, int64_t expert, const uint8_t** p) override {
        return !mine(layer, expert) && base_->zc_roles(layer, expert, p);
    }
    void warm(int64_t layer, const int64_t* experts, int64_t n) override;
    void warm_scored(int64_t layer, const int64_t* experts, const float* conf, int64_t n, int ahead) override;
    bool wants_scores() const override { return base_->wants_scores(); }
    bool warms() const override { return base_->warms(); }

private:
    PairShadowSource() = default;
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
        int state = 0;                      // 0 free, 1 reading, 2 ready, 3 dropped while reading, 4 being copied out
        bool ok = true;
    };
    bool mine(int64_t layer, int64_t expert) const {
        return layer >= 0 && layer < L_ && expert >= 0 && expert < E_ && idx_of_[(size_t) (layer * E_ + expert)] >= 0;
    }
    /// the experts of `layer` that are not this source's, with their positions in the caller's list
    void others(int64_t layer, const int64_t* experts, int64_t n, std::vector<int64_t>& ids, std::vector<int64_t>& at) const;
    bool read_at(void* h, uint64_t off, void* dst, uint64_t bytes) const;
    bool read_direct(int64_t p, uint8_t* dst);   // out of the window, synchronous
    void pump_locked();                     // issue reads while the window has room
    void advance_locked();                  // lo_ past the consumed / dropped entries
    void worker(int d);

    ExpertSource* base_ = nullptr;
    int64_t L_ = 0, E_ = 0, P_ = 0;         // layers, experts per layer, pairs in the cycle
    int64_t skipped_ = 0;                   // listed in the manifest, not wanted (not the helper's in this run)
    std::vector<uint64_t> bb_, stride_;     // per layer
    std::vector<int32_t> idx_of_;           // per (layer, expert): its place in the cycle, -1 if not this source's
    std::vector<int32_t> layer_of_;         // per place
    std::vector<uint8_t> drive_of_;         // per place
    std::vector<uint64_t> off_of_;          // per place
    std::vector<std::unique_ptr<Drive>> drives_;
    uint64_t max_stride_ = 0;
    int64_t W_ = 0;                         // window slots
    uint8_t* arena_ = nullptr;              // W_ x max_stride_, page aligned
    std::vector<Entry> win_;
    int64_t lo_ = 0, next_ = 0;             // the window holds sequence numbers [lo_, next_)
    int64_t inflight_ = 0;
    bool active_ = false;
    bool quit_ = false;
    std::mutex mu_;
    std::condition_variable cv_work_, cv_ready_;
    std::atomic<int64_t> st_blobs_{0}, st_direct_{0}, st_resets_{0};
    std::atomic<uint64_t> st_wait_us_{0};
};

}  // namespace strata::core
