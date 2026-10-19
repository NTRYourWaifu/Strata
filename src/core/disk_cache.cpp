#include "strata/core/disk_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <system_error>

namespace strata::core {
namespace {
namespace fs = std::filesystem;

constexpr char kMagic[8] = {'S', 'T', 'R', 'D', 'C', 'M', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kFnvBasis = 1469598103934665603ull;

uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

fs::path u8(const std::string& s) { return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size())); }
std::string str(const fs::path& p) {
    const std::u8string s = p.u8string();
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

// the key of a file name: what comes before its first dot (16 hex digits for this cache's files)
std::string key_of(const fs::path& p) {
    const std::string n = str(p.filename());
    const size_t dot = n.find('.');
    return dot == std::string::npos ? std::string() : n.substr(0, dot);
}

bool is_key(const std::string& k) {
    return k.size() == 16 && std::all_of(k.begin(), k.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

struct Group {
    uint64_t bytes = 0;
    fs::file_time_type newest{};
    std::vector<fs::path> files;
};

std::map<std::string, Group> groups(const std::string& dir) {
    std::map<std::string, Group> g;
    std::error_code ec;
    for (fs::directory_iterator it(u8(dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const std::string k = key_of(it->path());
        if (!is_key(k)) continue;
        Group& x = g[k];
        x.bytes += (uint64_t) it->file_size(ec);
        const auto t = it->last_write_time(ec);
        if (x.files.empty() || t > x.newest) x.newest = t;
        x.files.push_back(it->path());
    }
    return g;
}

class Writer {
public:
    void raw(const void* p, size_t n) { const auto* b = (const uint8_t*) p; buf.insert(buf.end(), b, b + n); }
    template <class T> void put(T v) { raw(&v, sizeof v); }
    std::vector<uint8_t> buf;
};

class Reader {
public:
    Reader(const std::vector<uint8_t>& b) : b_(b) {}
    bool raw(void* p, size_t n) {
        if (n > b_.size() - at_) return false;
        std::memcpy(p, b_.data() + at_, n);
        at_ += n;
        return true;
    }
    template <class T> bool get(T& v) { return raw(&v, sizeof v); }
    size_t at() const { return at_; }
    size_t left() const { return b_.size() - at_; }
private:
    const std::vector<uint8_t>& b_;
    size_t at_ = 0;
};

bool read_meta(const fs::path& p, DiskSnapshot& s) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 16 || std::memcmp(b.data(), kMagic, 8) != 0) return false;
    uint64_t want = 0;
    std::memcpy(&want, b.data() + b.size() - 8, 8);
    if (fnv(b.data(), b.size() - 8, kFnvBasis) != want) return false;
    Reader r(b);
    char magic[8];
    uint32_t ver = 0, ll = 0;
    int32_t cvec = 1;
    uint64_t n = 0;
    if (!r.raw(magic, 8) || !r.get(ver) || ver != kVersion || !r.get(s.model) || !r.get(s.config) || !r.get(ll) ||
        ll > r.left())
        return false;
    s.layout.assign(ll, '\0');
    if (!r.raw(s.layout.data(), ll) || !r.get(s.kind) || !r.get(cvec) || !r.get(s.bytes) || !r.get(s.remote_bytes) ||
        !r.get(s.used) || !r.get(n) || n > r.left() / 4)
        return false;
    s.cvec = cvec != 0;
    s.ids.resize((size_t) n);
    if (!r.raw(s.ids.data(), (size_t) n * 4) || !r.get(n) || n > r.left() / 8) return false;
    s.cks.resize((size_t) n);
    if (!r.raw(s.cks.data(), (size_t) n * 8) || r.left() != 8) return false;
    for (int64_t c : s.cks)
        if (c < 1 || c > (int64_t) s.ids.size()) return false;
    s.key = key_of(p);
    return is_key(s.key);
}
}  // namespace

const char* disk_kind_name(int32_t kind) {
    return kind == kDiskRoot ? "system prompt root" : kind == kDiskSwitch ? "left for another conversation" : "turn end";
}

std::string disk_snapshot_key(const std::vector<int32_t>& ids, bool cvec) {
    uint64_t h = fnv(ids.data(), ids.size() * sizeof(int32_t), kFnvBasis);
    const uint8_t c = cvec ? 1 : 0;
    h = fnv(&c, 1, h);
    char k[17];
    std::snprintf(k, sizeof k, "%016llx", (unsigned long long) h);
    return k;
}

int64_t disk_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string DiskCache::path(const std::string& key, const std::string& ext) const {
    return str(u8(dir_) / u8(key + "." + ext));
}

bool DiskCache::open(const std::string& dir, uint64_t cap_bytes, std::string& err) {
    std::error_code ec;
    fs::create_directories(u8(dir), ec);
    if (!fs::is_directory(u8(dir), ec)) {
        err = "disk cache: cannot create the folder " + dir;
        return false;
    }
    dir_ = dir;
    cap_ = cap_bytes;
    e_.clear();
    for (fs::directory_iterator it(u8(dir), ec), end; !ec && it != end; it.increment(ec)) {
        const fs::path p = it->path();
        const std::string name = str(p.filename());
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {   // a write that did not finish
            std::error_code rec;   // its own: a file it cannot delete does not end the scan
            fs::remove(p, rec);
            continue;
        }
        if (p.extension() != ".meta") continue;
        DiskSnapshot s;
        if (!read_meta(p, s) || !fs::exists(u8(path(s.key, "s0")), ec)) {
            std::fprintf(stderr, "strata serve: disk cache: %s is broken or incomplete; deleted\n", name.c_str());
            disk_drop_key(dir_, key_of(p));
            continue;
        }
        e_.push_back(std::move(s));
    }
    return true;
}

int DiskCache::best(const std::vector<int64_t>& prompt, bool cvec, uint64_t model, uint64_t config,
                    const std::string& layout, int64_t& resume) const {
    int pick = -1;
    resume = 0;
    for (size_t i = 0; i < e_.size(); ++i) {
        const DiskSnapshot& s = e_[i];
        if (s.cvec != cvec || s.model != model || s.config != config || s.layout != layout) continue;
        const size_t lim = std::min(s.ids.size(), prompt.size());
        size_t lcp = 0;
        while (lcp < lim && (int64_t) s.ids[lcp] == prompt[lcp]) ++lcp;
        int64_t cand = 0;
        // the last prompt token always starts the first verify window: a snapshot must stop before it
        if (lcp == s.ids.size() && s.ids.size() < prompt.size()) {
            cand = (int64_t) s.ids.size();
        } else {
            for (auto it = s.cks.rbegin(); it != s.cks.rend(); ++it)
                if (*it <= (int64_t) lcp && *it < (int64_t) prompt.size()) { cand = *it; break; }
        }
        if (cand <= 0) continue;
        if (cand > resume || (cand == resume && pick >= 0 &&
                              s.bytes + s.remote_bytes < e_[(size_t) pick].bytes + e_[(size_t) pick].remote_bytes)) {
            resume = cand;
            pick = (int) i;
        }
    }
    return pick;
}

int DiskCache::find(const std::vector<int32_t>& ids, bool cvec) const {
    for (size_t i = 0; i < e_.size(); ++i)
        if (e_[i].cvec == cvec && e_[i].ids == ids) return (int) i;
    return -1;
}

bool DiskCache::write_meta(const DiskSnapshot& s, std::string& err) const {
    Writer w;
    w.raw(kMagic, 8);
    w.put(kVersion);
    w.put(s.model);
    w.put(s.config);
    w.put((uint32_t) s.layout.size());
    w.raw(s.layout.data(), s.layout.size());
    w.put(s.kind);
    w.put((int32_t) (s.cvec ? 1 : 0));
    w.put(s.bytes);
    w.put(s.remote_bytes);
    w.put(s.used);
    w.put((uint64_t) s.ids.size());
    w.raw(s.ids.data(), s.ids.size() * 4);
    w.put((uint64_t) s.cks.size());
    w.raw(s.cks.data(), s.cks.size() * 8);
    w.put(fnv(w.buf.data(), w.buf.size(), kFnvBasis));
    const fs::path tmp = u8(path(s.key, "meta.tmp")), dst = u8(path(s.key, "meta"));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write((const char*) w.buf.data(), (std::streamsize) w.buf.size());
        if (!f) { err = "disk cache: writing " + str(tmp) + " failed"; return false; }
    }
    std::error_code ec;
    fs::rename(tmp, dst, ec);
    if (ec) { err = "disk cache: renaming " + str(tmp) + ": " + ec.message(); return false; }
    return true;
}

bool DiskCache::put(DiskSnapshot s, std::string& err) {
    if (!write_meta(s, err)) return false;
    for (DiskSnapshot& x : e_)
        if (x.key == s.key) { x = std::move(s); return true; }
    e_.push_back(std::move(s));
    return true;
}

void DiskCache::touch(size_t i) {
    e_[i].used = disk_now_ms();
    std::string err;
    if (!write_meta(e_[i], err)) std::fprintf(stderr, "strata serve: %s\n", err.c_str());
}

void DiskCache::drop(size_t i) {
    disk_drop_key(dir_, e_[i].key);
    e_.erase(e_.begin() + (std::ptrdiff_t) i);
}

uint64_t DiskCache::dir_bytes() const {
    uint64_t n = 0;
    for (const auto& [k, g] : groups(dir_)) n += g.bytes;
    return n;
}

std::vector<std::string> DiskCache::trim(const std::string& keep) {
    std::vector<std::string> out;
    auto g = groups(dir_);
    uint64_t total = 0;
    for (const auto& [k, x] : g) total += x.bytes;
    // the order: files without an entry first (oldest first), then the entries least recently used first
    std::vector<std::pair<int64_t, std::string>> order;
    for (const auto& [k, x] : g) {
        const auto it = std::find_if(e_.begin(), e_.end(), [&](const DiskSnapshot& s) { return s.key == k; });
        order.emplace_back(it == e_.end() ? INT64_MIN + 1 : it->used, k);
    }
    std::sort(order.begin(), order.end());
    for (const auto& [used, k] : order) {
        if (total <= cap_) break;
        if (k == keep) continue;
        total -= g[k].bytes;
        disk_drop_key(dir_, k);
        e_.erase(std::remove_if(e_.begin(), e_.end(), [&](const DiskSnapshot& s) { return s.key == k; }), e_.end());
        out.push_back(k);
    }
    return out;
}

std::vector<std::string> disk_trim_dir(const std::string& dir, uint64_t cap_bytes, const std::string& keep) {
    std::vector<std::string> out;
    auto g = groups(dir);
    uint64_t total = 0;
    for (const auto& [k, x] : g) total += x.bytes;
    std::vector<std::pair<fs::file_time_type, std::string>> order;
    for (const auto& [k, x] : g) order.emplace_back(x.newest, k);
    std::sort(order.begin(), order.end());
    for (const auto& [t, k] : order) {
        if (total <= cap_bytes) break;
        if (k == keep) continue;
        total -= g[k].bytes;
        disk_drop_key(dir, k);
        out.push_back(k);
    }
    return out;
}

void disk_drop_key(const std::string& dir, const std::string& key) {
    if (!is_key(key)) return;
    std::error_code ec;
    // the map held by name: a range-for over groups(dir)[key].files would read a destroyed temporary (C++20)
    const auto g = groups(dir);
    const auto it = g.find(key);
    if (it == g.end()) return;
    for (const fs::path& p : it->second.files) fs::remove(p, ec);
}

bool disk_make_dir(const std::string& dir) {
    std::error_code ec;
    fs::create_directories(u8(dir), ec);
    return fs::is_directory(u8(dir), ec);
}

void disk_touch_file(const std::string& path) {
    std::error_code ec;
    fs::last_write_time(u8(path), fs::file_time_type::clock::now(), ec);
}

}  // namespace strata::core
