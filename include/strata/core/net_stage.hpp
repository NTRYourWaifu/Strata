// include/strata/core/net_stage.hpp - a layer split across two PCs: the later stage runs in another process, on
// another machine, and the hand-off of each verify window crosses a TCP connection instead of mapped pinned memory.
//
// The first PC runs layers [0, K) and the server loop as always; its verifier's next stage is a NetLink.  The second
// PC runs `strata --net-listen PORT --net-split K ...`: layers [K, n_layers), the head, its own expert cache, resident
// RAM copy of its own layers' experts and CPU pool.  Per window the first PC sends the window's tokens, positions and
// hand-off rows (T x handoff_floats(g) floats: R, the pending write, inject), the second runs its stage and answers
// with the T picks.  The hand-off is the same bytes the in-process split hands over, so the split stays bit-exact.
#pragma once

#include "strata/kernels/sampler.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

/// Commit flags (a second int32; a second PC that reads only n ignores it).  NetCommitIdle: the next window is at least
/// one of this PC's stages away (the serial loop, or a pipelined guess that missed), so the second PC may run its
/// adaptive round now instead of right after its reply, where the next window may already be waiting (H-42).
constexpr int32_t NetCommitIdle = 1;

/// One message: [u32 magic][u32 op][u64 bytes] then the payload.
enum class NetOp : uint32_t {
    Hello = 1,    ///< first -> second: geometry; reply: vocab
    Run = 2,      ///< a verify window
    Commit = 3,   ///< keep the first n tokens of the last window (int32 n[, int32 flags: NetCommitIdle])
    Zero = 4,     ///< session_zero: a fresh sequence
    Logits = 5,   ///< row t of the last window's head logits
    Bye = 6,
    Save = 7,      ///< a conversation checkpoint of this PC's layers at L tokens (int64 L, int32 n, int64 keep[n])
    Restore = 8,   ///< put the checkpoint at L tokens back (int64 L)
    Prefill = 9,   ///< a batched prompt chunk: NetPrefillHead, int32 tokens[T], float rows[T x hc*n_embd]
    Cvec = 10,     ///< the request's control vector switch (int32 0|1) for this PC's layers
    Reply = 100,
};

struct NetHello {
    uint32_t version = 4;   // 2: Save / Restore (conversation checkpoints); 3: Prefill (the batched prompt path); 4: Cvec
    int32_t split = 0;          ///< K: the second PC's first layer
    int32_t n_layers = 0, n_embd = 0, hc = 0;
    int32_t max_t = 0;
    int64_t max_context = 0;
    int32_t sampler_bytes = (int32_t) sizeof(strata::kernels::SamplerParams);
    int32_t handoff_floats = 0;
};

/// The fixed head of a Run message; tokens[T], the history rows (T x hist_len int32, when hist_len > 0), the spec q
/// rows (nq x q_stride int32) and the hand-off rows (T x handoff_floats float) follow.
struct NetRunHead {
    int32_t T = 0;
    int32_t head_sampling = 1;
    int64_t pos0 = 0;
    strata::kernels::SamplerParams sp;
    int32_t hist_len = 0;
    int32_t nq = 0;
    int32_t q_stride = 0;
    int32_t want_R = 0;   ///< 1: the reply carries the window's final residual rows too (the first PC's draft layer)
};

/// A batched prompt chunk for the second PC's layers (the reply: its last layer's rows when want_R).
struct NetPrefillHead {
    int64_t pos0 = 0;
    int32_t T = 0;
    int32_t single_chunk = 0;
    int32_t ple_prev[2] = {-1, -1};   ///< the two tokens before pos0, oldest first (-1: none)
    int32_t want_R = 0;
    int32_t pad = 0;
};

/// A blocking TCP connection (Winsock on Windows, BSD sockets elsewhere), TCP_NODELAY.
class NetConn {
public:
    NetConn() = default;
    ~NetConn();
    NetConn(const NetConn&) = delete;
    NetConn& operator=(const NetConn&) = delete;
    bool connect_to(const std::string& host_port, std::string& err);
    /// Listen on `port` (all interfaces) and accept one connection.
    bool accept_one(int port, std::string& err);
    bool send_msg(NetOp op, const void* a, size_t na, const void* b = nullptr, size_t nb = 0, const void* c = nullptr,
                  size_t nc = 0, const void* d = nullptr, size_t nd = 0, const void* e = nullptr, size_t ne = 0);
    /// The next message: its op and payload.
    bool recv_msg(NetOp& op, std::vector<uint8_t>& payload, std::string& err);
    void close();
    bool open() const { return fd_ != ~0ull; }

private:
    uint64_t fd_ = ~0ull;
    uint64_t lfd_ = ~0ull;
    bool send_all(const void* p, size_t n);
    bool recv_all(void* p, size_t n);
};

/// The first PC's side: the later stage as a verifier's next stage (Verifier::set_link).
class NetLink {
public:
    bool connect(const std::string& host_port, const NetHello& h, std::string& err);
    /// One window.  `hand` is the host view of the stage's hand-off (T rows written); `hist_dev` the device history
    /// rows (T x hist_len) or null; `q` the drafter's q rows (host) or null.  `out[T]`: the picks.
    bool run(int T, const int32_t* tokens, int64_t pos0, const float* hand, int handoff_floats,
             const strata::kernels::SamplerParams& sp, bool head_sampling, const int32_t* hist_dev, int hist_len,
             const int32_t* q, int nq, int q_stride, int32_t* out, std::string& err);
    bool commit(int n_keep, std::string& err, int32_t flags = NetCommitIdle);
    bool zero(std::string& err);
    /// The second PC saves its layers' running state at `L` tokens and keeps only the checkpoints at `keep` (and L).
    bool save(int64_t L, const std::vector<int64_t>& keep, std::string& err);
    bool restore(int64_t L, std::string& err);
    /// The request's control vector switch for the second PC's layers (it answers an error when it has none loaded).
    bool cvec(bool on, std::string& err);
    /// A batched prompt chunk: this PC's rows of layer K go over, the other PC reads its layers; with `back` its last
    /// layer's rows come back (T x row_floats).  Thread: one at a time, while nothing else uses the link.
    bool prefill(const NetPrefillHead& h, const int32_t* tokens, const float* rows, int64_t row_floats,
                 std::vector<float>* back, std::string& err);
    double ms_prefill = 0;          ///< the prompt chunks' round trips, summed
    int64_t prefill_chunks = 0;
    bool logits(int t, float* host, std::string& err);
    int64_t vocab() const { return vocab_; }
    /// The draft layer (MTP) runs on the first PC from the last layer's residual rows: when `R_dev` is set, every
    /// window's reply carries its T final residual rows (R_floats each = hc * n_embd) and run() puts them here (device).
    float* R_dev = nullptr;
    int R_floats = 0;
    /// The stream the rows are copied on (null: cudaMemcpy).  A non-blocking stream keeps the copy from waiting for
    /// this PC's windows in flight (--pipeline-windows across two PCs: NetPipe's thread copies while stage 0 runs).
    void* R_stream = nullptr;
    /// Milliseconds spent waiting for the second PC (send + its stage + receive), summed.
    double ms_remote = 0;
    double ms_rcopy = 0;            ///< ... of which the residual rows' copy to the device
    int64_t windows = 0;
    uint64_t bytes_sent = 0;

private:
    NetConn c_;
    int64_t vocab_ = 0;
    std::vector<uint8_t> buf_;
    std::vector<int32_t> hist_h_;
    bool reply(std::string& err, std::vector<uint8_t>* payload = nullptr);
};

/// --pipeline-windows across two PCs: the other PC (the last stage) driven without blocking the decode loop, so this
/// PC's stage runs the next window on a guess while the other PC verifies this one.  The last stage only ever runs
/// verified windows, so one window at a time is in flight here and the other PC needs nothing new: `launch` hands the
/// window to a thread that sends it and waits for the reply (NetLink::run), `done` polls, `finish` reads the picks.
/// The final residual rows land in R[par] (the window's parity): the drafter reads one window's rows while the next
/// window's reply comes in.  R[0] is the link's own buffer (the serial loop's), R[1] is allocated here.
class NetPipe {
public:
    ~NetPipe();
    /// `device`: this PC's drafter's device (R[1] and the copy stream go there).
    bool start(NetLink* link, int device, std::string& err);
    /// The window: `hand` is the host view of its hand-off (T rows), copied before launch returns, as are `tokens`.
    void launch(int par, int T, const int32_t* tokens, int64_t pos0, const float* hand, int handoff_floats,
                const strata::kernels::SamplerParams& sp, bool head_sampling);
    bool in_flight() const { return busy_; }
    /// The reply (or an error) is in.
    bool done() const;
    /// After `done`: the picks (`out` may be null: a drained window).  The link's rows buffer is R[0] again.
    bool finish(int32_t* out, std::string& err);
    /// The commit of the window just finished (nothing in flight; no reply: see NetLink::commit).
    bool commit(int n_keep, std::string& err, int32_t flags) { return link_->commit(n_keep, err, flags); }
    /// The window in flight to its end, its picks discarded.
    void drain();
    float* R[2] = {nullptr, nullptr};

private:
    NetLink* link_ = nullptr;
    int dev_ = 0;
    void* stream_ = nullptr;
    bool busy_ = false;
    std::thread th_;
    std::atomic<bool> done_{false};
    bool ok_ = false;
    std::string err_;
    int T_ = 0;
    int32_t out_[16] = {};
    std::vector<int32_t> tok_;
    std::vector<float> hand_;
};

/// Reply payload: [i32 ok][u32 len][err bytes][data].
bool net_reply_ok(NetConn& c, const void* data, size_t n);
bool net_reply_err(NetConn& c, const std::string& err);

}  // namespace strata::core
