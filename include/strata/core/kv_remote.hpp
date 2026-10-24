// include/strata/core/kv_remote.hpp - S-31: the main QSA layers' K/V on a second GPU, which also computes their decode
// attention (STRATA_S31_KV_DEV=N, the CUDA device; unset: off and nothing here runs).
//
// Why: the K/V of a 131K context is ~1.6 GiB of the primary card that the expert cache can use instead.  The second
// card has no P2P path to the primary here (GTX 1660 SUPER on PCIe x1 beside an RTX 3080 Ti), so nothing on either card
// reads the other's memory: the two talk through mapped pinned host memory.
//
//   decode (verify window): the primary's window graph, at each QSA layer, writes the layer's queries, new K/V rows,
//   selection and step records to that layer's mailbox slot and raises `req`; a graph on the second card (launched just
//   before the window's own, one per window size) waits for it, appends the rows to its pools, runs the same attention
//   kernel over the same selection, writes the result back and raises `done`; the primary waits for that and goes on.
//   The values are the ones the primary would have computed (STRATA_S31_HALF=1: the rows cross as fp16, not bitwise).
//
//   prompt (prefill): the layer's cells [0, p0) are copied into a staging pool on the primary (the KV streaming path's,
//   kv_mode 1), the chunk is appended and attended there, and the chunk's pages go back to the second card.
//
// Only main layers in mode 0 (no --kv-resident, no --kv-grow), one sequence (no --batch), no window split.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

namespace strata::core {

struct QsaState;

/// The device the K/V goes to, -1 when off (STRATA_S31_KV_DEV, read once).
int kv_remote_device();
/// STRATA_S31_HALF=1: queries, K/V rows and the attention output cross PCIe as fp16 (half the bytes; not bitwise).
bool kv_remote_half();
/// `bytes` of the remote device's memory (its context first set to spin and map host memory); nullptr when refused.
void* kv_remote_alloc(uint64_t bytes);
uint64_t kv_remote_allocated();

/// The prompt path: pages [0, n_pages) of `st`'s pools (on the remote card) into `stage` (the same arrays, identity
/// layout, on the primary), and pages [p0, p1) of `stage` back to `st`.  The primary's half of each copy is queued on
/// `stream`, the second card's half on that card's own copy stream (through pinned chunks; STRATA_S31_PEER=1: peer copies
/// on `stream`).  The next decode window's graph on the second card waits for all of them.
void kv_remote_stage_in(const QsaState& st, const strata::kernels::KvHostPools& stage, int64_t n_pages,
                        const strata::kernels::QsaShapes& s, void* stream);
void kv_remote_stage_out(const QsaState& st, const strata::kernels::KvHostPools& stage, int64_t p0, int64_t p1,
                         const strata::kernels::QsaShapes& s, void* stream);

/// The decode window's exchange (one per Verifier).
class KvRemoteWindow {
public:
    ~KvRemoteWindow();
    bool on() const { return dev_ >= 0; }
    bool init(int n_slots, int max_t, int64_t cap, int64_t attn_scratch_floats, const strata::kernels::QsaShapes& s,
              std::string& err);
    /// Inside the primary's capture, on its stream: slot `slot`'s payload out and `req` raised; later its result in.
    void publish(int slot, const float* q, const float* k, const float* v, const int32_t* sel, const int32_t* step, int n,
                 void* stream);
    void collect(int slot, float* attn, int n, void* stream);
    /// The remote card's graph for a T-token window: for each (slot, state), in the order the primary publishes them,
    /// wait, append the rows, attend, reply.
    bool capture(int T, const std::vector<std::pair<int, const QsaState*>>& layers, std::string& err);
    bool captured(int T) const { return T >= 0 && T < kMaxT && exec_[T] != nullptr; }
    /// Before the window's own graph: the previous window's remote graph drained, the window number raised, this one
    /// launched (and submitted).
    bool launch(int T, std::string& err);
    /// An abandoned window: every wait on both cards returns.
    void release();

private:
    static constexpr int kMaxT = 16;
    int dev_ = -1, n_slots_ = 0, max_t_ = 0;
    int64_t cap_ = 0, scratch_floats_ = 0;
    strata::kernels::QsaShapes s_{};
    bool half_ = false;
    uint64_t slot_bytes_ = 0, off_k_ = 0, off_v_ = 0, off_sel_ = 0, off_step_ = 0, off_out_ = 0;
    uint8_t* box_ = nullptr;         // the mailbox (mapped, portable): n_slots slots
    uint32_t* hdr_ = nullptr;        // [0] window number, [32] abort, [64..] error words (mapped, portable)
    uint32_t* cnt0_ = nullptr;       // the publish kernel's block counter (primary)
    // remote card
    void* stream_ = nullptr;
    uint32_t* cnt1_ = nullptr;
    float *q1_ = nullptr, *k1_ = nullptr, *v1_ = nullptr, *attn1_ = nullptr, *scratch1_ = nullptr;
    int32_t *sel1_ = nullptr, *step1_ = nullptr;
    uint8_t* in1_ = nullptr;         // a slot's inputs, copied in by DMA before they are unpacked
    unsigned long long* prof1_ = nullptr;   // STRATA_S31_PROF=1: the phase sums (kv_remote.cu stamp_kernel)
    void* exec_[kMaxT] = {};
    uint32_t win_ = 0;
    bool launched_ = false;
};

}  // namespace strata::core
