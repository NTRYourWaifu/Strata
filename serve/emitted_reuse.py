"""S-84 (fork): a continuation is tokenized the way the model wrote it.

The engine keeps a conversation as token ids and reuses it only while the next prompt's ids continue it.  The next
prompt is the client's TEXT tokenized again, and the tokenizer does not always give back the ids the model emitted:
the model writes `收` `成` where the tokenizer writes `收成`, ` new` `Text` where it writes ` newText`.  One such token
anywhere in a reply and the engine falls back to the checkpoint before that reply and reads the whole reply again
(measured on the daily log, 11-14..11-15: 161 of 1652 continuations, 43% of the replies over 2000 tokens; 18 of the 19
snapshots that diverged on the SSD diverged this way, none at the reply's last tokens).

EmittedReuse remembers the ids of what went through the engine (the prompt sent and the tokens that came back) and
reads the ids in the disk cache's snapshots.  A new prompt that says the same TEXT as one of them, in other ids, gets
that one's ids for as long as the text is the same: `resync`.  The text the model sees is byte for byte what the client
sent; only which ids spell it changes.  Special tokens are never matched as text: they must be the same id.
"""
from __future__ import annotations

import threading
from array import array
from pathlib import Path


def common_prefix(a, b) -> int:
    """Leading items two sequences of one type share (slices compare at C speed)."""
    n = min(len(a), len(b))
    lo, step = 0, 4096
    while lo < n:                         # whole chunks first, doubling
        hi = min(n, lo + step)
        if a[lo:hi] != b[lo:hi]:
            break
        lo, step = hi, min(step * 2, 1 << 20)
    else:
        return n
    while hi - lo > 1:                    # the first difference is in [lo, hi)
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def align(kept, new, at, token_bytes, is_special):
    """Walk `kept` and `new` from index `at` (their common prefix) while they spell the same bytes.
    -> (k, j): kept[:k] and new[:j] are the same text and both end on a token boundary; the furthest such pair."""
    i = j = at
    ba = bb = b""
    last = (at, at)
    nk, nn = len(kept), len(new)
    while True:
        if not ba and not bb:
            while i < nk and j < nn and kept[i] == new[j]:
                i += 1
                j += 1
            last = (i, j)
            if i >= nk or j >= nn or is_special(kept[i]) or is_special(new[j]):
                break
            ba, bb = token_bytes(kept[i]), token_bytes(new[j])
            i += 1
            j += 1
            if not ba or not bb:
                break
        m = min(len(ba), len(bb))
        if ba[:m] != bb[:m]:
            break
        ba, bb = ba[m:], bb[m:]
        if ba and not bb:
            if j >= nn or is_special(new[j]):
                break
            bb = token_bytes(new[j])
            j += 1
            if not bb:
                break
        elif bb and not ba:
            if i >= nk or is_special(kept[i]):
                break
            ba = token_bytes(kept[i])
            i += 1
            if not ba:
                break
    return last


class _Tracked:
    """The engine's iterator of token ids, every id noted as it passes.  close() and anything else go to the engine's
    own iterator (the server STOPs the engine through close())."""

    def __init__(self, gen, live):
        self._gen, self._it, self._live = gen, iter(gen), live

    def __iter__(self):
        return self

    def __next__(self):
        t = next(self._it)
        if isinstance(t, int):          # (None is the engine's heartbeat)
            self._live.append(t)
        return t

    def close(self):
        close = getattr(self._gen, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name):
        return getattr(self._gen, name)


class EmittedReuse:
    MIN_SHARED = 8         # a kept conversation is looked at only when it shares at least this many leading ids
    DISK_TRY = 4            # snapshots walked per pass: the ones sharing the longest prefix

    def __init__(self, tok, keep: int = 12, disk_dir=None, log=None):
        self.tok = tok
        self.keep = keep
        self.seqs: list[array] = []             # newest last; array('i'): 4 bytes an id
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self.disk: dict[str, tuple[int, array]] = {}        # snapshot key -> (mtime_ns, ids)
        self.log = log or (lambda msg: None)
        self.lock = threading.Lock()
        self.stats = {"prompts": 0, "resynced": 0, "tokens": 0, "from_disk": 0}
        specials = set((getattr(tok, "special_tokens", None) or {}).values())
        nvocab = len(getattr(tok, "tokens", ())) or (1 << 62)
        raw = tok.token_bytes

        def special(t):
            return t in specials or t < 0 or t >= nvocab
        self._special = special

        def tb(t):
            try:
                return raw(t)
            except (KeyError, IndexError):
                return b""              # ends the walk there
        self._bytes = tb

    # ------------------------------------------------------------------ what went through the engine
    def track(self, prompt_ids, gen):
        """Wrap one engine pass: the prompt it was given and every token it yields are one kept conversation."""
        live = array("i", prompt_ids)
        with self.lock:
            # a pass that continues a kept conversation replaces it (the same ids, longer)
            self.seqs = [s for s in self.seqs if not (len(s) <= len(live) and live[:len(s)] == s)]
            self.seqs.append(live)
            del self.seqs[:-self.keep]
        return _Tracked(gen, live)

    # ------------------------------------------------------------------ the disk cache's snapshots
    def _disk_seqs(self):
        if self.disk_dir is None or not self.disk_dir.is_dir():
            return []
        try:
            from serve.ckpt_ledger import read_meta_head
        except ImportError:
            from ckpt_ledger import read_meta_head
        seen = set()
        for meta in self.disk_dir.glob("*.meta"):
            try:
                mt = meta.stat().st_mtime_ns
            except OSError:
                continue
            seen.add(meta.stem)
            if self.disk.get(meta.stem, (0, None))[0] == mt:
                continue
            head = read_meta_head(meta)
            if not head or len(head[3]) < 4:
                self.disk.pop(meta.stem, None)
                continue
            a = array("i")
            a.frombytes(head[3][:len(head[3]) // 4 * 4])
            self.disk[meta.stem] = (mt, a)
        for k in [k for k in self.disk if k not in seen]:
            del self.disk[k]
        return [v[1] for v in self.disk.values()]

    # ------------------------------------------------------------------ a new prompt
    def _best(self, ids, seqs, floor, limit=None):
        """The kept conversation that spells the most of `ids` beyond their common prefix.  `floor`: ids[:floor] was
        taken from a kept conversation already and is not given to another one."""
        cand = []
        for s in seqs:
            c = common_prefix(ids, s)
            if c >= len(s) or c >= len(ids) or c < max(self.MIN_SHARED, floor):
                continue
            cand.append((c, s))
        if limit is not None and len(cand) > limit:
            cand.sort(key=lambda x: -x[0])
            del cand[limit:]
        best = None
        for c, s in cand:
            k, j = align(s, ids, c, self._bytes, self._special)
            if k > c and (best is None or k > best[0]):
                best = (k, j, c, s)
        return best

    def resync(self, ids: list[int]) -> list[int]:
        """`ids` with the stretch a kept conversation spells in other ids replaced by that conversation's ids."""
        with self.lock:
            self.stats["prompts"] += 1
            out, rounds, saved, disk_used, floor = array("i", ids), 0, 0, False, 0
            while rounds < 64:
                best = self._best(out, reversed(self.seqs), floor)
                if best is None or best[0] < len(best[3]):   # memory does not carry the whole of it (a restart, an older conversation)
                    d = self._best(out, self._disk_seqs(), floor, self.DISK_TRY)
                    if d is not None and (best is None or d[0] > best[0]):
                        best, disk_used = d, True
                if best is None:
                    break
                k, j, c, s = best
                # the guarantee, checked and not assumed: the same bytes
                if b"".join(map(self._bytes, s[c:k])) != b"".join(map(self._bytes, out[c:j])):
                    self.log(f"[strata] S-84: a resync at {c} was not the same text: left as tokenized")
                    break
                out = s[:k] + out[j:]
                saved += k - c
                floor = k
                rounds += 1
            if not rounds:
                return ids
            self.stats["resynced"] += 1
            self.stats["tokens"] += saved
            self.stats["from_disk"] += int(disk_used)
            self.log(f"[strata] S-84: the prompt continues a kept conversation in the model's own tokens "
                     f"({saved} tokens of it{', from the disk cache' if disk_used else ''}; "
                     f"{len(ids)} -> {len(out)} tokens)")
            return out.tolist()
