import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Mapping

from gptcache.adapter.api import get as cache_get, put as cache_put
from cache_switch import switch_cache
from policy import BasePolicy

# ----------------- Ollama -----------------
OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate(model: str, prompt: str, timeout_s: int = 300) -> str:
    """
    Call local Ollama. Set stream=False so we measure a single latency figure.
    """
    r = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

# ----------------- Policy state -----------------
@dataclass
class PolicyState:
    name: str
    run_tag: str
    policy_impl: BasePolicy
    hits: int = 0
    misses: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

# ----------------- Gateway -----------------
class MultiCacheGateway:
    """
    Compare multiple cache policies fairly on the same prompt stream.

    For each prompt:
      1) Perform lookup for ALL policies (switching cache files per policy).
         - Respect policy logical-evictions (denylist) by forcing miss when denied.
      2) If ANY policy missed, call the model ONCE (shared).
      3) Backfill ONLY the miss policies; attribute model latency only to those.
      4) Policy.on_put -> Policy.maybe_evict (logical-eviction; no FAISS delete).
      5) Return any hit answer; otherwise return the model answer.
    """

    def __init__(
        self,
        model_name: str,
        policies: List[PolicyState],
        similarity_max_distance: float = 0.6,
        use_synthetic_answers: bool = False,
        synthetic_sizes: Optional[List[int]] = None,            # e.g., [2048, 4096, 8192]
        synthetic_latency_map_ms: Optional[Mapping[int, int]] = None,  # {size: sleep_ms}
        synthetic_sleep_ms: int = 0,                            # fallback latency if map missing
        ollama_timeout_s: int = 300,
    ):
        self.model_name = model_name
        self.policies = policies
        self.similarity_max_distance = similarity_max_distance

        # Synthetic control (Option A)
        self.use_synthetic_answers = use_synthetic_answers
        self.synthetic_sizes = synthetic_sizes or [2048, 4096, 8192]

        # default variable latency per size
        self.synthetic_latency_map_ms = synthetic_latency_map_ms or {2048: 60, 4096: 110, 8192: 220}
        self.synthetic_sleep_ms = synthetic_sleep_ms   # used only if size not in map
        self.ollama_timeout_s = ollama_timeout_s

        # tiny exact-key front map (debug aid; not used for metrics)
        self._exact_map: Dict[str, str] = {}

    # -------- internal helpers --------
    def _ensure_active(self, run_tag: str):
        """
        Ensure GPTCache global singleton points to this policy's files.
        switch_cache() is memoized; it only rebuilds when run_tag/sim changes.
        """
        switch_cache(run_tag, self.similarity_max_distance)

    # -------- public API --------
    def ask(self, prompt: str) -> Dict[str, Any]:
        lookups: List[Dict[str, Any]] = []
        any_miss = False

        # track the most recent active cache to avoid redundant switches
        active_tag: Optional[str] = None
        def ensure_active(tag: str):
            nonlocal active_tag
            if active_tag != tag:
                self._ensure_active(tag)
                active_tag = tag

        # 1) Lookups for all policies
        for pol in self.policies:
            ensure_active(pol.run_tag)
            t0 = time.perf_counter()
            ans = cache_get(prompt)
            # Respect logical eviction (denylist): force a miss if denied by policy
            if hasattr(pol.policy_impl, "is_denied") and pol.policy_impl.is_denied(prompt):
                ans = None
            lookup_ms = (time.perf_counter() - t0) * 1000.0
            hit = ans is not None
            if hit:
                pol.hits += 1
                try:
                    pol.policy_impl.on_hit(prompt, meta={})
                except Exception:
                    pass
            else:
                pol.misses += 1
                any_miss = True
            lookups.append(dict(pol=pol, hit=hit, lookup_ms=lookup_ms, ans=ans))

        # 2) If any policy missed, call model ONCE
        answer: Optional[str] = None
        model_ms: float = 0.0
        if any_miss:
            t0 = time.perf_counter()
            try:
                if self.use_synthetic_answers:
                    # ---- Option A: variable size + variable latency tied to size ----
                    idx = hash(prompt) % len(self.synthetic_sizes)
                    size = self.synthetic_sizes[idx]                 # e.g., 2048 / 4096 / 8192
                    sleep_ms = self.synthetic_latency_map_ms.get(size, self.synthetic_sleep_ms)
                    if sleep_ms and sleep_ms > 0:
                        time.sleep(sleep_ms / 1000.0)
                    answer = "x" * size
                else:
                    answer = ollama_generate(self.model_name, prompt, timeout_s=self.ollama_timeout_s)
            finally:
                model_ms = (time.perf_counter() - t0) * 1000.0  # measured "cost" saved on a future hit

        # 3) Attribute per-policy latency, backfill only misses
        ret_hit_answer: Optional[str] = next((r["ans"] for r in lookups if r["hit"]), None)

        for rec in lookups:
            pol: PolicyState = rec["pol"]
            if rec["hit"]:
                pol.latencies_ms.append(rec["lookup_ms"])
                pol.events.append({
                    "prompt": prompt,
                    "ts": time.time(),
                    "lookup_ms": rec["lookup_ms"],
                    "model_ms": 0.0,
                    "e2e_ms": rec["lookup_ms"],
                    "hit": 1,
                    "path": "lookup_hit",
                })
            else:
                # Switch to this policy's files, write the model answer into this policy's cache
                ensure_active(pol.run_tag)
                try:
                    cache_put(prompt, answer or "")
                except Exception as e:
                    # Log but keep going; we still want metrics even if a put fails
                    print(f"[ERROR] cache_put failed for policy={pol.name}: {e}")

                # Policy bookkeeping + eviction using measured model_ms as saved cost
                try:
                    size_bytes = len((answer or "").encode("utf-8", errors="ignore"))
                    pol.policy_impl.on_put(
                        prompt,
                        answer or "",
                        meta={"size_bytes": size_bytes, "saved_latency_ms": model_ms},
                    )
                    pol.policy_impl.maybe_evict()
                except Exception as e:
                    print(f"[ERROR] policy.on_put/maybe_evict failed for policy={pol.name}: {e}")

                e2e = rec["lookup_ms"] + model_ms
                pol.latencies_ms.append(e2e)
                pol.events.append({
                    "prompt": prompt,
                    "ts": time.time(),
                    "lookup_ms": rec["lookup_ms"],
                    "model_ms": model_ms,
                    "e2e_ms": e2e,
                    "hit": 0,
                    "path": "lookup_miss+model",
                })

        # 4) Return an answer (prefer any hit; else model answer)
        final_answer = ret_hit_answer if ret_hit_answer is not None else (answer or "")
        return {"answer": final_answer}
