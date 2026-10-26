"""S-50 (fork): which reply each disk cache snapshot continues, for a front-end's conversation list.

The engine writes a snapshot (--disk-cache DIR, <key>.meta + <key>.s0 ...) at a turn's end, at a stop and when a
conversation is left for another one; its .meta holds the token ids.  It does not know which chat the ids are, and a
front-end does not know token ids.  This links the two by the OpenAI response id (`chatcmpl-...`, which Pi keeps as
the reply's responseId): every chat request's prompt ids are kept for a while, and a .meta written after the request's
first token whose ids start with its prompt and hold about its reply after it (prompt + reply - 1 .. + 1 measured) is
that reply's snapshot.  "Written after the first token" keeps out the snapshot the same request read back: the engine
reads one back (and rewrites its .meta's last-used time) before the first token and writes after the last.  Without it
the same prompt sent again (the launcher's warm-up, a retry) would take the older snapshot that holds another reply.
One JSON line per link goes to DIR/replies.jsonl:

    {"rid": "chatcmpl-...", "key": "<16 hex>", "kind": "turn", "tokens": 9696, "ck": 9675, "bytes": 323700000, "t": <ms>}

"ck" is the deepest checkpoint the snapshot holds (its end when it holds none).  The next prompt usually leaves the
reply's thinking out, so it does not start with the whole snapshot: the engine reads back up to that checkpoint (the
prompt's end) and reads the reply again.

A reader (Pi Web's /api/checkpoints) checks that DIR/<key>.meta still exists before it believes a line: the engine
deletes old snapshots when DIR is full and a user may move files away.  The file is not a snapshot (its name is not
16 hex digits), so the engine's open and trim leave it alone.

Nothing here touches the engine: it reads the folder from a thread, every 2 s while a request was seen in the last
10 minutes, and does nothing at all otherwise.
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
from array import array
from pathlib import Path

MAGIC = b"STRDCM01"
KIND_NAMES = {0: "turn", 1: "root", 2: "switch", 3: "stop"}
PENDING_MAX = 12            # requests kept (a tool loop sends one per call)
WATCH_S = 600.0             # the folder is read while a request was seen in this many seconds
POLL_S = 2.0
UNMATCHED_S = 180.0         # a .meta no request explains yet is looked at again for this long
REPLY_TAIL = (-3, 3)        # snapshot tokens - prompt - reply: -1 and +1 measured (10-26)
COMPACT_BYTES = 256 << 10   # replies.jsonl above this is rewritten without the lines whose snapshot is gone


def read_meta_head(path: Path):
    """(kind, bytes, used_ms, ids as raw little-endian int32 bytes, deepest checkpoint) of a .meta, or None when it is
    not one (yet)."""
    try:
        b = path.read_bytes()
    except OSError:
        return None
    if len(b) < 16 or b[:8] != MAGIC:
        return None
    try:
        at = 8
        ver, = struct.unpack_from("<I", b, at); at += 4
        if ver not in (1, 2):
            return None
        at += 16                                           # model, config
        ll, = struct.unpack_from("<I", b, at); at += 4
        at += ll                                           # layout
        kind, _cvec = struct.unpack_from("<ii", b, at); at += 8
        nbytes, remote, used, n = struct.unpack_from("<QQqQ", b, at); at += 32
        if at + 4 * n > len(b) - 8:
            return None
        ids = b[at:at + 4 * n]
        at += 4 * n
        nck, = struct.unpack_from("<Q", b, at); at += 8
        cks = struct.unpack_from(f"<{nck}q", b, at) if nck and at + 8 * nck <= len(b) - 8 else ()
        return kind, nbytes + remote, used, ids, max(cks, default=n)
    except struct.error:
        return None


class CkptLedger:
    def __init__(self, folder: str | os.PathLike, log=None):
        self.dir = Path(folder)
        self.path = self.dir / "replies.jsonl"
        self.log = log or (lambda msg: None)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending: list[dict] = []   # {"rid", "q": bytes, "n": int, "t", "first": time|None, "done": int|None, "end"}
        self.seen: dict[str, int] = {}                     # key -> the .meta's mtime_ns already looked at
        self.unmatched: dict[str, float] = {}              # key -> when it was first looked at without a match
        self.linked: set[tuple[str, str]] = set()
        self.last_request = 0.0
        self.thread: threading.Thread | None = None
        # the snapshots already there are not this run's replies
        for p in self._metas():
            try:
                self.seen[p.stem] = p.stat().st_mtime_ns
            except OSError:
                pass

    # ---- the request side (the server's threads) ----
    def begin(self, rid: str, ids) -> None:
        """A chat request is about to run with these prompt ids."""
        q = array("i", ids).tobytes()
        with self.lock:
            self.pending.append({"rid": rid, "q": q, "n": len(q) // 4, "t": time.time(), "first": None, "done": None,
                                 "end": None})
            del self.pending[:-PENDING_MAX]
            self.last_request = time.time()
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="ckpt-ledger", daemon=True)
                self.thread.start()
        self.wake.set()

    def first(self, rid: str) -> None:
        """Its first token came: the prompt is read, so a snapshot read back for it was read back before now."""
        now = time.time()
        with self.lock:
            for p in self.pending:
                if p["rid"] == rid and p["first"] is None:
                    p["first"] = now

    def end(self, rid: str, completion_tokens: int) -> None:
        """It ended (normally or at a stop) after this many tokens of reply."""
        now = time.time()
        with self.lock:
            for p in self.pending:
                if p["rid"] == rid:
                    p["done"] = max(0, int(completion_tokens or 0))
                    p["end"] = now
            self.last_request = time.time()
        self.wake.set()

    # ---- the folder side (its own thread) ----
    def _metas(self):
        try:
            with os.scandir(self.dir) as it:
                return [Path(e.path) for e in it if e.name.endswith(".meta") and len(e.name) == 21]
        except OSError:
            return []

    def _run(self) -> None:
        while True:
            with self.lock:
                idle = time.time() - self.last_request > WATCH_S and not self.unmatched
            if idle:
                self.wake.wait()
            else:
                self.wake.wait(POLL_S)
            self.wake.clear()
            try:
                self.scan()
            except Exception as e:                         # a cache view must never take the server down
                self.log(f"[strata] S-50 replies ledger: {type(e).__name__}: {e}")

    def scan(self) -> list[dict]:
        """Looks at the new or changed .meta files once; returns the lines written (tests call it directly)."""
        now = time.time()
        out = []
        for p in self._metas():
            key = p.stem
            try:
                mt = p.stat().st_mtime_ns
            except OSError:
                continue
            if self.seen.get(key) == mt and key not in self.unmatched:
                continue
            head = read_meta_head(p)
            if head is None:                               # being written: next time
                continue
            kind, nbytes, used, ids, ck = head
            line = self._match(key, kind, nbytes, used, ids, ck, mt / 1e9) if kind in (0, 2, 3) else None
            if line is None and kind in (0, 2, 3):
                first = self.unmatched.setdefault(key, now)
                if now - first < UNMATCHED_S:
                    self.seen[key] = mt
                    continue
            self.unmatched.pop(key, None)
            self.seen[key] = mt
            if line is not None:
                out.append(line)
        if out:
            self._append(out)
        return out

    def _match(self, key, kind, nbytes, used, ids: bytes, ck: int, mtime: float):
        n = len(ids) // 4
        best = None
        with self.lock:
            pend = list(self.pending)
        for p in pend:
            if p["done"] is None or not REPLY_TAIL[0] <= n - p["n"] - p["done"] <= REPLY_TAIL[1]:
                continue
            after = p["first"] if p["first"] is not None else p["end"]   # no token at all: its end
            if mtime < after - 0.05:                     # older than its first token: read back, not written for it
                continue
            if ids[:len(p["q"])] != p["q"]:
                continue
            if best is None or p["t"] >= best["t"]:      # the same prompt twice: the newer request
                best = p
        if best is None or (best["rid"], key) in self.linked:
            return None
        self.linked.add((best["rid"], key))
        return {"rid": best["rid"], "key": key, "kind": KIND_NAMES.get(kind, str(kind)), "tokens": n, "ck": int(ck),
                "bytes": int(nbytes), "t": int(used) if used > 0 else int(time.time() * 1000)}

    def _append(self, lines: list[dict]) -> None:
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)
            if self.path.stat().st_size > COMPACT_BYTES:
                self.compact()
        except OSError as e:
            self.log(f"[strata] S-50 replies ledger: cannot write {self.path}: {e}")

    def compact(self) -> None:
        """Rewrites replies.jsonl without the lines whose snapshot is gone."""
        keep = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for raw in f:
                    try:
                        x = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(x, dict) and (self.dir / f"{x.get('key')}.meta").exists():
                        keep.append(raw if raw.endswith("\n") else raw + "\n")
        except OSError:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        os.replace(tmp, self.path)
