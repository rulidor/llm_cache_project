import time, requests
from typing import List, Dict, Any
from gptcache.adapter.api import get as cache_get, put as cache_put
from cache_setup import build_cache
from policy import NoopPolicy, CostAwarePolicy

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate(model: str, prompt: str, timeout_s=300) -> str:
    r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("response", "")

class CacheGateway:
    def __init__(self, model_name="phi3:mini",
                 policy="noop",
                 byte_budget=50_000,          # small default so evictions happen fast
                 run_tag="baseline",
                 similarity_max_distance=0.6):
        self.model_name = model_name
        self.cache, self.paths = build_cache(run_tag=run_tag,
                                             similarity_max_distance=similarity_max_distance,
                                             reset_files=True)
        self.policy = NoopPolicy() if policy == "noop" else CostAwarePolicy(byte_budget=byte_budget)
        # inline metrics (simple & fast)
        self.latencies_ms: List[float] = []
        self.hits = 0
        self.misses = 0
        self.events: List[Dict[str, Any]] = []  # for CSV/plots

    def ask(self, prompt: str) -> str:
        t0 = time.perf_counter()

        cached = cache_get(prompt)  # singleton cache on your fork
        if cached is not None:
            self.policy.on_hit(prompt, meta={})
            dt = (time.perf_counter() - t0) * 1000
            self.latencies_ms.append(dt)
            self.hits += 1
            self.events.append(dict(prompt=prompt, latency_ms=dt, hit=1))
            return cached

        # miss path
        answer = ollama_generate(self.model_name, prompt)
        miss_ms = (time.perf_counter() - t0) * 1000

        cache_put(prompt, answer)
        self.policy.on_put(prompt, answer, meta={
            "size_bytes": len(answer.encode("utf-8")),
            "saved_latency_ms": miss_ms
        })
        self.policy.maybe_evict(self.cache)

        self.latencies_ms.append(miss_ms)
        self.misses += 1
        self.events.append(dict(prompt=prompt, latency_ms=miss_ms, hit=0))
        return answer
