import time
import heapq
from typing import Dict, Any


class BasePolicy:
    """Interface definition for external cache policies used by MultiCacheGateway."""
    def on_put(self, key: str, answer: str, meta: Dict[str, Any]):
        """Called after a miss is backfilled for this policy."""
        ...

    def on_hit(self, key: str, meta: Dict[str, Any]):
        """Called when this policy had a hit for the prompt."""
        ...

    def maybe_evict(self):
        """Called after put; evict if over budget."""
        ...


class NoopPolicy(BasePolicy):
    """Baseline policy: do nothing beyond GPTCache default behavior."""
    def __init__(self):
        self.evictions = 0

    def on_put(self, key: str, answer: str, meta: Dict[str, Any]):
        pass

    def on_hit(self, key: str, meta: Dict[str, Any]):
        pass

    def maybe_evict(self):
        pass


class LruBytePolicy(BasePolicy):
    """
    Bounded LRU by bytes:
      - Tracks per-key size and last_access.
      - On put, if over byte_budget, evict least-recently-used items
        until bytes_used <= byte_budget.
      - Uses logical eviction (denylist) to avoid FAISS delete issues.
    """
    def __init__(self, byte_budget: int):
        self.byte_budget = int(byte_budget)
        self.bytes_used = 0
        self.evictions = 0

        self.meta: Dict[str, Dict[str, Any]] = {}  # key -> {size_bytes, last_access, hits}
        self.deny = set()

        # (optional) counters
        self.puts = 0
        self.hits_seen = 0

    def on_put(self, key: str, answer: str, meta: Dict[str, Any]):
        self.puts += 1
        now = time.time()
        sz = int(meta.get("size_bytes", len(answer.encode("utf-8", errors="ignore"))))

        if key in self.meta:
            prev = int(self.meta[key].get("size_bytes", 0))
            if sz != prev:
                self.bytes_used += (sz - prev)
            self.meta[key]["size_bytes"] = sz
            self.meta[key]["last_access"] = now
        else:
            self.meta[key] = {"size_bytes": sz, "last_access": now, "hits": 0}
            self.bytes_used += sz

        # re-allow if previously denylisted
        if key in self.deny:
            self.deny.discard(key)

    def on_hit(self, key: str, meta: Dict[str, Any]):
        self.hits_seen += 1
        m = self.meta.get(key)
        if m:
            m["hits"] = int(m.get("hits", 0)) + 1
            m["last_access"] = time.time()

    def maybe_evict(self):
        if self.bytes_used <= self.byte_budget:
            return

        # Build list sorted by oldest last_access first (true LRU)
        cands = []
        for k, m in self.meta.items():
            if k in self.deny:
                continue
            cands.append((float(m.get("last_access", 0.0)), k, int(m.get("size_bytes", 0))))
        if not cands:
            print(f"[LRU] check: bytes={self.bytes_used} budget={self.byte_budget} (no candidates)")
            return

        # sort by last_access ascending => oldest first
        cands.sort(key=lambda t: t[0])

        before = self.bytes_used
        victims = 0
        for _, victim, sz in cands:
            if self.bytes_used <= self.byte_budget:
                break
            if victim in self.deny:
                continue
            self.deny.add(victim)
            self.evictions += 1
            self.bytes_used = max(0, self.bytes_used - sz)
            victims += 1

        if victims:
            print(f"[LRU] evicted {victims} item(s); bytes: {before} -> {self.bytes_used} (budget {self.byte_budget})")
        else:
            print(f"[LRU] check: bytes={before} > budget={self.byte_budget}, but no victims selected")

    def is_denied(self, key: str) -> bool:
        return key in self.deny


class CostAwarePolicy(BasePolicy):
    """
    Logical-eviction, byte-budget-aware policy:
      - Tracks per-key size, hits, recency, and 'saved_latency_ms' (benefit proxy).
      - When over 'byte_budget', evicts the lowest-score entries (denylist).
      - We DO NOT physically delete from FAISS/SQLite to avoid Windows/FAISS ID issues.
      - MultiCacheGateway must respect denylist by treating denylisted keys as misses.
    """
    def __init__(self, byte_budget: int = 128 * 1024):
        self.byte_budget = int(byte_budget)
        self.bytes_used = 0
        self.evictions = 0

        # key -> metadata
        self.meta: Dict[str, Dict[str, Any]] = {}
        # logical evictions
        self.deny = set()

        # Simple counters for debugging/visibility
        self.puts = 0
        self.hits_seen = 0

    # ---------- Policy hooks ----------

    def on_put(self, key: str, answer: str, meta: Dict[str, Any]):
        self.puts += 1
        now = time.time()
        size_bytes = int(meta.get("size_bytes", len(answer.encode("utf-8", errors="ignore"))))
        saved_latency_ms = float(meta.get("saved_latency_ms", 0.0))

        # Update or insert metadata
        m = self.meta.get(key)
        if not m:
            self.meta[key] = {
                "size_bytes": size_bytes,
                "hits": 0,
                "last_access": now,
                "saved_latency_ms": saved_latency_ms,
            }
            self.bytes_used += size_bytes
        else:
            # refresh size/recency/benefit; adjust bytes_used if size changes
            prev = int(m.get("size_bytes", 0))
            if size_bytes != prev:
                self.bytes_used += (size_bytes - prev)
            m["size_bytes"] = size_bytes
            m["last_access"] = now
            m["saved_latency_ms"] = max(m.get("saved_latency_ms", 0.0), saved_latency_ms)

        # If this key was previously evicted logically, re-allow it
        if key in self.deny:
            self.deny.discard(key)

    def on_hit(self, key: str, meta: Dict[str, Any]):
        self.hits_seen += 1
        m = self.meta.get(key)
        if m:
            m["hits"] = int(m.get("hits", 0)) + 1
            m["last_access"] = time.time()

    def maybe_evict(self):
        before = self.bytes_used
        if self.bytes_used <= self.byte_budget:
            # Uncomment if you want to see checks even when nothing to do
            # print(f"[COSTAWARE] check: bytes={self.bytes_used} budget={self.byte_budget} (no evict needed)")
            return

        heap = []
        now = time.time()
        for k, m in self.meta.items():
            if k in self.deny:
                continue
            sz = int(m.get("size_bytes", 0))
            hits = int(m.get("hits", 0))
            saved = float(m.get("saved_latency_ms", 0.0))
            age_s = max(1.0, now - float(m.get("last_access", now)))
            recency = 1.0 / (1.0 + (age_s / 300.0))
            benefit = saved * (1.0 + 0.25 * hits) * recency
            score = benefit / max(1, sz)
            heap.append((score, k, sz))

        if not heap:
            print(f"[COSTAWARE] check: bytes={self.bytes_used} budget={self.byte_budget} (no candidates)")
            return

        heap.sort()  # worst first (lowest score)
        victims = []
        i = 0
        while self.bytes_used > self.byte_budget and i < len(heap):
            _score, victim, sz = heap[i];
            i += 1
            if victim in self.deny:
                continue
            self.deny.add(victim)
            self.evictions += 1
            self.bytes_used = max(0, self.bytes_used - sz)
            victims.append((victim, sz))

        if victims:
            print(
                f"[COSTAWARE] evicted {len(victims)} item(s); bytes: {before} -> {self.bytes_used} (budget {self.byte_budget})")
            # If you still want per-item logs:
            # for v, sz in victims:
            #     print(f"  - {v[:80]!r} size={sz}")
        else:
            print(f"[COSTAWARE] check: bytes={before} > budget={self.byte_budget}, but no victims selected")

    # ---------- Helper for gateway ----------

    def is_denied(self, key: str) -> bool:
        """Return True if this key was logically evicted and should be treated as a miss."""
        return key in self.deny
