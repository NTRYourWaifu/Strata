// src/core/net_stage.cpp - the TCP side of a layer split across two PCs (see net_stage.hpp).
#include "strata/core/net_stage.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
static const sock_t kBad = INVALID_SOCKET;
static void sock_close(sock_t s) { closesocket(s); }
static std::string sock_err() { return "socket error " + std::to_string(WSAGetLastError()); }
#else
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
using sock_t = int;
static const sock_t kBad = -1;
static void sock_close(sock_t s) { ::close(s); }
static std::string sock_err() { return std::string("socket error ") + std::strerror(errno); }
#endif

namespace strata::core {

namespace {
constexpr uint32_t kMagic = 0x314E5353;   // "SSN1"
struct MsgHead {
    uint32_t magic;
    uint32_t op;
    uint64_t bytes;
};
bool net_init() {
#if defined(_WIN32)
    static const bool ok = [] {
        WSADATA w;
        return WSAStartup(MAKEWORD(2, 2), &w) == 0;
    }();
    return ok;
#else
    return true;
#endif
}
void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int big = 8 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &big, sizeof big);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &big, sizeof big);
}
}  // namespace

void net_to_bf16(const float* in, uint16_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, in + i, 4);
        u += 0x7FFFu + ((u >> 16) & 1u);   // round to nearest even (the rows carry no NaN)
        out[i] = (uint16_t) (u >> 16);
    }
}

void net_from_bf16(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t u = (uint32_t) in[i] << 16;
        std::memcpy(out + i, &u, 4);
    }
}

NetConn::~NetConn() { close(); }

void NetConn::close() {
    if (fd_ != ~0ull) sock_close((sock_t) fd_);
    if (lfd_ != ~0ull) sock_close((sock_t) lfd_);
    fd_ = lfd_ = ~0ull;
}

bool NetConn::connect_to(const std::string& host_port, std::string& err) {
    if (!net_init()) { err = "WSAStartup failed"; return false; }
    const size_t c = host_port.rfind(':');
    if (c == std::string::npos) { err = "expected HOST:PORT, got " + host_port; return false; }
    const std::string host = host_port.substr(0, c), port = host_port.substr(c + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host_port;
        return false;
    }
    sock_t s = kBad;
    for (int attempt = 0; attempt < 600 && s == kBad; ++attempt) {   // the second PC may still be loading: 10 minutes
        s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (s == kBad) break;
        if (::connect(s, res->ai_addr, (int) res->ai_addrlen) != 0) {
            sock_close(s);
            s = kBad;
#if defined(_WIN32)
            Sleep(1000);
#else
            sleep(1);
#endif
        }
    }
    freeaddrinfo(res);
    if (s == kBad) { err = "cannot connect to " + host_port + " (" + sock_err() + ")"; return false; }
    tune(s);
    fd_ = (uint64_t) s;
    return true;
}

bool NetConn::accept_one(int port, std::string& err) {
    if (!net_init()) { err = "WSAStartup failed"; return false; }
    if (lfd_ == ~0ull) {
        sock_t l = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (l == kBad) { err = sock_err(); return false; }
        int one = 1;
        setsockopt(l, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons((uint16_t) port);
        if (bind(l, (sockaddr*) &a, sizeof a) != 0 || listen(l, 1) != 0) {
            err = "cannot listen on port " + std::to_string(port) + " (" + sock_err() + ")";
            sock_close(l);
            return false;
        }
        lfd_ = (uint64_t) l;
    }
    if (fd_ != ~0ull) { sock_close((sock_t) fd_); fd_ = ~0ull; }
    sock_t s = accept((sock_t) lfd_, nullptr, nullptr);
    if (s == kBad) { err = sock_err(); return false; }
    tune(s);
    fd_ = (uint64_t) s;
    return true;
}

bool NetConn::send_all(const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const int k = ::send((sock_t) fd_, c, (int) std::min<size_t>(n, 1u << 30), 0);
        if (k <= 0) return false;
        c += k;
        n -= (size_t) k;
    }
    return true;
}

bool NetConn::recv_all(void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const int k = ::recv((sock_t) fd_, c, (int) std::min<size_t>(n, 1u << 30), 0);
        if (k <= 0) return false;
        c += k;
        n -= (size_t) k;
    }
    return true;
}

bool NetConn::send_msg(NetOp op, const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc,
                       const void* d, size_t nd, const void* e, size_t ne) {
    // one buffer, one send: the window's message leaves in as few segments as the stack makes of it
    thread_local std::vector<uint8_t> out;
    const MsgHead h{kMagic, (uint32_t) op, (uint64_t) (na + nb + nc + nd + ne)};
    out.resize(sizeof h + na + nb + nc + nd + ne);
    uint8_t* w = out.data();
    std::memcpy(w, &h, sizeof h);
    w += sizeof h;
    for (auto [p, n] : {std::pair<const void*, size_t>{a, na}, {b, nb}, {c, nc}, {d, nd}, {e, ne}})
        if (n > 0) { std::memcpy(w, p, n); w += n; }
    return send_all(out.data(), out.size());
}

bool NetConn::recv_msg(NetOp& op, std::vector<uint8_t>& payload, std::string& err) {
    MsgHead h{};
    if (!recv_all(&h, sizeof h)) { err = "the connection closed"; return false; }
    if (h.magic != kMagic) { err = "net stage: a bad message (magic)"; return false; }
    if (h.bytes > (1ull << 32)) { err = "net stage: a message too large"; return false; }
    payload.resize((size_t) h.bytes);
    if (h.bytes > 0 && !recv_all(payload.data(), (size_t) h.bytes)) { err = "the connection closed mid-message"; return false; }
    op = (NetOp) h.op;
    return true;
}

bool net_reply_ok(NetConn& c, const void* data, size_t n) {
    const int32_t ok = 1;
    const uint32_t len = 0;
    return c.send_msg(NetOp::Reply, &ok, 4, &len, 4, data, n);
}

bool net_reply_err(NetConn& c, const std::string& err) {
    const int32_t ok = 0;
    const uint32_t len = (uint32_t) err.size();
    return c.send_msg(NetOp::Reply, &ok, 4, &len, 4, err.data(), err.size());
}

bool NetLink::reply(std::string& err, std::vector<uint8_t>* payload) {
    NetOp op{};
    if (!c_.recv_msg(op, buf_, err)) { err = "net stage: " + err; return false; }
    if (op != NetOp::Reply || buf_.size() < 8) { err = "net stage: an unexpected reply"; return false; }
    int32_t ok = 0;
    uint32_t len = 0;
    std::memcpy(&ok, buf_.data(), 4);
    std::memcpy(&len, buf_.data() + 4, 4);
    if (!ok) {
        err = "the second PC: " + std::string((const char*) buf_.data() + 8, std::min<size_t>(len, buf_.size() - 8));
        return false;
    }
    if (payload) payload->assign(buf_.begin() + 8 + len, buf_.end());
    return true;
}

bool NetLink::connect(const std::string& host_port, const NetHello& h, std::string& err) {
    if (const char* v = std::getenv("STRATA_NET_HALF")) half = v[0] == '1';
    if (!c_.connect_to(host_port, err)) return false;
    if (!c_.send_msg(NetOp::Hello, &h, sizeof h)) { err = "net stage: send failed"; return false; }
    std::vector<uint8_t> p;
    if (!reply(err, &p)) return false;
    if (p.size() < 8) { err = "net stage: a short hello reply"; return false; }
    std::memcpy(&vocab_, p.data(), 8);
    return true;
}

bool NetLink::run(int T, const int32_t* tokens, int64_t pos0, const float* hand, int handoff_floats,
                  const strata::kernels::SamplerParams& sp, bool head_sampling, const int32_t* hist_dev, int hist_len,
                  const int32_t* q, int nq, int q_stride, int32_t* out, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    NetRunHead h;
    h.T = T;
    h.pos0 = pos0;
    h.head_sampling = head_sampling ? 1 : 0;
    h.sp = sp;
    h.hist_len = hist_dev != nullptr ? hist_len : 0;
    h.nq = q != nullptr ? nq : 0;
    h.q_stride = q_stride;
    h.want_R = (R_dev != nullptr || R_host != nullptr ? 1 : 0) | (half ? NetRowsBf16 : 0);
    if (h.hist_len > 0) {   // the penalty rows live on this PC's GPU: the window's T rows go along
        hist_h_.resize((size_t) T * (size_t) hist_len);
        if (cudaMemcpy(hist_h_.data(), hist_dev, hist_h_.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "net stage: the penalty history copy failed";
            return false;
        }
    }
    const size_t nf = (size_t) T * (size_t) handoff_floats;
    const void* hp = hand;
    if (half) {
        half_out_.resize(nf);
        net_to_bf16(hand, half_out_.data(), nf);
        hp = half_out_.data();
    }
    const size_t nh = nf * (half ? 2 : sizeof(float));
    if (!c_.send_msg(NetOp::Run, &h, sizeof h, tokens, (size_t) T * 4, hist_h_.data(), (size_t) h.hist_len * T * 4, q,
                     (size_t) h.nq * (size_t) q_stride * 4, hp, nh)) {
        err = "net stage: send failed";
        return false;
    }
    bytes_sent += sizeof h + (size_t) T * 4 + nh;
    std::vector<uint8_t> p;
    if (!reply(err, &p)) return false;
    const size_t nRf = (h.want_R & 1) ? (size_t) T * (size_t) R_floats : 0;
    const size_t nR = nRf * sizeof(float);
    if (p.size() < (size_t) T * 4 + nRf * (half ? 2 : 4)) { err = "net stage: a short window reply"; return false; }
    std::memcpy(out, p.data(), (size_t) T * 4);
    if (nR > 0 && R_host != nullptr) {   // NetPipe's thread: host memory only, no CUDA call (see NetPipe)
        const auto tc = std::chrono::steady_clock::now();
        const uint8_t* src = p.data() + (size_t) T * 4;
        if (half) net_from_bf16((const uint16_t*) src, R_host, nRf);
        else std::memcpy(R_host, src, nR);
        ms_rcopy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    } else if (nR > 0) {
        const auto tc = std::chrono::steady_clock::now();
        const void* src = p.data() + (size_t) T * 4;
        if (half) {
            half_in_.resize(nRf);
            net_from_bf16((const uint16_t*) src, half_in_.data(), nRf);
            src = half_in_.data();
        }
        cudaStream_t rs = (cudaStream_t) R_stream;
        const bool ok = rs == nullptr
                            ? cudaMemcpy(R_dev, src, nR, cudaMemcpyHostToDevice) == cudaSuccess
                            : cudaMemcpyAsync(R_dev, src, nR, cudaMemcpyHostToDevice, rs) == cudaSuccess &&
                                  cudaStreamSynchronize(rs) == cudaSuccess;
        if (!ok) {
            err = "net stage: the final residual rows copy failed";
            return false;
        }
        ms_rcopy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    ms_remote += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++windows;
    return true;
}

// No reply: the second PC commits before it reads the next message, and a commit that failed there fails the
// next message's reply.  One round trip per window instead of two.
bool NetLink::commit(int n_keep, std::string& err, int32_t flags) {
    const int32_t m[2] = {n_keep, flags};
    if (!c_.send_msg(NetOp::Commit, m, 8)) { err = "net stage: send failed"; return false; }
    return true;
}

bool NetLink::zero(std::string& err) {
    if (!c_.send_msg(NetOp::Zero, nullptr, 0)) { err = "net stage: send failed"; return false; }
    return reply(err);
}

bool NetLink::save(int64_t L, const std::vector<int64_t>& keep, std::string& err) {
    std::vector<uint8_t> p(12 + keep.size() * 8);
    const int32_t n = (int32_t) keep.size();
    std::memcpy(p.data(), &L, 8);
    std::memcpy(p.data() + 8, &n, 4);
    if (!keep.empty()) std::memcpy(p.data() + 12, keep.data(), keep.size() * 8);
    if (!c_.send_msg(NetOp::Save, p.data(), p.size())) { err = "net stage: send failed"; return false; }
    return reply(err);
}

bool NetLink::cvec(bool on, std::string& err) {
    const int32_t v = on ? 1 : 0;
    if (!c_.send_msg(NetOp::Cvec, &v, 4)) { err = "net stage: send failed"; return false; }
    return reply(err);
}

bool NetLink::prefill(const NetPrefillHead& h0, const int32_t* tokens, const float* rows, int64_t row_floats,
                      std::vector<float>* back, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    NetPrefillHead h = h0;
    if (half) h.want_R |= NetRowsBf16;
    const size_t nf = (size_t) h.T * (size_t) row_floats;
    const void* rp = rows;
    if (half) {
        half_out_.resize(nf);
        net_to_bf16(rows, half_out_.data(), nf);
        rp = half_out_.data();
    }
    const size_t nr = nf * (half ? 2 : sizeof(float));
    if (!c_.send_msg(NetOp::Prefill, &h, sizeof h, tokens, (size_t) h.T * 4, rp, nr)) {
        err = "net stage: send failed";
        return false;
    }
    bytes_sent += sizeof h + (size_t) h.T * 4 + nr;
    std::vector<uint8_t> p;
    if (!reply(err, back != nullptr ? &p : nullptr)) return false;
    if (back != nullptr) {
        if (p.size() < nr) { err = "net stage: a short prompt chunk reply"; return false; }
        back->resize(nf);
        if (half) net_from_bf16((const uint16_t*) p.data(), back->data(), nf);
        else std::memcpy(back->data(), p.data(), nr);
    }
    ms_prefill += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++prefill_chunks;
    return true;
}

bool NetLink::restore(int64_t L, std::string& err) {
    if (!c_.send_msg(NetOp::Restore, &L, 8)) { err = "net stage: send failed"; return false; }
    return reply(err);
}

NetPipe::~NetPipe() {
    if (th_.joinable()) {
        if (done_.load()) th_.join();
        else th_.detach();   // still waiting on the other PC (an engine ending on an error): not waited for
    }
}

bool NetPipe::start(NetLink* link, int device, std::string& err) {
    link_ = link;
    dev_ = device;
    if (cudaSetDevice(device) != cudaSuccess) { err = "net pipe: cudaSetDevice failed"; return false; }
    cudaStream_t s = nullptr;   // the serial loop's row copies (the decode loop's own thread) stay off the busy streams
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) { err = "net pipe: stream creation failed"; return false; }
    stream_ = s;
    link->R_stream = stream_;
    if (link->R_dev != nullptr) {   // the drafter wants the rows: two page-locked host buffers, one per parity
        const size_t rb = (size_t) 8 * (size_t) link->R_floats * sizeof(float);   // kVerifyMaxT rows
        for (float*& r : R)
            if (cudaHostAlloc((void**) &r, rb, cudaHostAllocPortable) != cudaSuccess) {
                err = "net pipe: the residual rows' page-locked buffers";
                return false;
            }
    }
    return true;
}

void NetPipe::launch(int par, int T, const int32_t* tokens, int64_t pos0, const float* hand, int handoff_floats,
                     const strata::kernels::SamplerParams& sp, bool head_sampling) {
    tok_.assign(tokens, tokens + T);
    hand_.assign(hand, hand + (size_t) T * (size_t) handoff_floats);
    T_ = T;
    busy_ = true;
    done_.store(false);
    ok_ = false;
    err_.clear();
    float* rd = R[par & 1];
    th_ = std::thread([this, pos0, handoff_floats, sp, head_sampling, rd] {
        link_->R_host = rd;   // no CUDA call on this thread (see the class comment)
        ok_ = link_->run(T_, tok_.data(), pos0, hand_.data(), handoff_floats, sp, head_sampling, nullptr, 0, nullptr, 0,
                         0, out_, err_);
        link_->R_host = nullptr;
        done_.store(true, std::memory_order_release);
    });
}

bool NetPipe::done() const { return done_.load(std::memory_order_acquire); }

bool NetPipe::finish(int32_t* out, std::string& err) {
    if (th_.joinable()) th_.join();
    busy_ = false;
    if (!ok_) { err = err_; return false; }
    if (out != nullptr) std::memcpy(out, out_, (size_t) T_ * 4);
    return true;
}

void NetPipe::drain() {
    if (!busy_) return;
    std::string e;
    (void) finish(nullptr, e);
}

bool NetLink::logits(int t, float* host, std::string& err) {
    const int32_t r = t;
    if (!c_.send_msg(NetOp::Logits, &r, 4)) { err = "net stage: send failed"; return false; }
    std::vector<uint8_t> p;
    if (!reply(err, &p)) return false;
    if (p.size() < (size_t) vocab_ * 4) { err = "net stage: a short logits reply"; return false; }
    std::memcpy(host, p.data(), (size_t) vocab_ * 4);
    return true;
}

}  // namespace strata::core
