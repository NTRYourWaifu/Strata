// include/strata/core/disk_cache.hpp - the disk cache of conversation snapshots (H-69, --disk-cache DIR): a turn's
// end (the last reply was not a tool call), the system prompt's root and a long conversation left for another one
// are written to the SSD, and a later request that starts with one of them reads it back instead of the prompt.
//
// One snapshot is a set of files named <key>.<ext> in DIR: <key>.meta (this index's record: the token ids, the
// checkpoint lengths the files hold, the identity it was written with) and one session file per GPU stage on this PC
// (<key>.s0, <key>.s1, ...; conversation_file.hpp's format).  A split across two PCs keeps the other PC's layers in
// <key>.net on that PC.  Pure host code: no CUDA.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct DiskSnapshot {
    std::string key;                 ///< 16 hex digits: the files' name
    uint64_t model = 0, config = 0;  ///< SessionFileIdentity it was written with
    std::string layout;              ///< the stages' layer ranges (a snapshot of another split is not read)
    int32_t kind = 0;                ///< DiskKind
    bool cvec = true;                ///< the control vector's state the snapshot was read with
    std::vector<int32_t> ids;        ///< the snapshot's tokens (its live state is at ids.size())
    std::vector<int64_t> cks;        ///< the checkpoints the files hold (prefixes of ids), shortest first
    uint64_t bytes = 0;              ///< this PC's files
    uint64_t remote_bytes = 0;       ///< the other PC's file (0: none)
    int64_t used = 0;                ///< last written or read (ms since the epoch): the eviction order
};

enum DiskKind : int32_t { kDiskTurn = 0, kDiskRoot = 1, kDiskSwitch = 2 };
const char* disk_kind_name(int32_t kind);

/// The snapshot's name: the token ids and the control vector's state, hashed.
std::string disk_snapshot_key(const std::vector<int32_t>& ids, bool cvec);
int64_t disk_now_ms();

class DiskCache {
public:
    /// Creates DIR when missing and reads every <key>.meta in it (a broken one is skipped and its files deleted).
    bool open(const std::string& dir, uint64_t cap_bytes, std::string& err);
    bool on() const { return !dir_.empty(); }
    const std::string& dir() const { return dir_; }
    uint64_t cap() const { return cap_; }
    std::string path(const std::string& key, const std::string& ext) const;
    const std::vector<DiskSnapshot>& entries() const { return e_; }
    const DiskSnapshot& at(size_t i) const { return e_[i]; }

    /// The snapshot that gives `prompt` the most tokens: its live state when its ids are a prefix of the prompt
    /// shorter than it, else its deepest checkpoint that is.  Only snapshots of this identity, layout and control
    /// vector state count.  -1 when none; `resume` is the tokens it gives.  Ties: the smaller files.
    int best(const std::vector<int64_t>& prompt, bool cvec, uint64_t model, uint64_t config, const std::string& layout,
             int64_t& resume) const;
    /// The snapshot with exactly these ids and control vector state (any identity), or -1.
    int find(const std::vector<int32_t>& ids, bool cvec) const;
    /// Writes <key>.meta (through a temporary file) and adds or replaces the entry.
    bool put(DiskSnapshot s, std::string& err);
    /// Marks it used now (its .meta rewritten).
    void touch(size_t i);
    /// Deletes its files and the entry.
    void drop(size_t i);
    /// While all files in DIR together are above the cap, deletes the least recently used snapshot (files without a
    /// .meta first), never `keep`.  Returns the keys deleted.
    std::vector<std::string> trim(const std::string& keep);
    /// Every file in DIR, summed.
    uint64_t dir_bytes() const;

private:
    std::string dir_;
    uint64_t cap_ = 0;
    std::vector<DiskSnapshot> e_;
    bool write_meta(const DiskSnapshot& s, std::string& err) const;
};

/// A PC that keeps only its files (the second PC: <key>.net): deletes <key>.* groups, the oldest by their newest
/// file's time first, never `keep`, until DIR holds at most cap_bytes.  Returns the keys deleted.
std::vector<std::string> disk_trim_dir(const std::string& dir, uint64_t cap_bytes, const std::string& keep);
/// Creates DIR when missing (the second PC's folder: it has no DiskCache::open).  False when it is not a folder.
bool disk_make_dir(const std::string& dir);
/// Deletes every <key>.* file in DIR.
void disk_drop_key(const std::string& dir, const std::string& key);
/// Sets the file's time to now (the second PC's eviction order).
void disk_touch_file(const std::string& path);

}  // namespace strata::core
