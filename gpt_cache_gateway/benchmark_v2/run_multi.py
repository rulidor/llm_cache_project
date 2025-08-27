import csv
import json
import time
import os
from pathlib import Path
from typing import List

from cache_setup import build_cache, preproc
from multi_gateway import MultiCacheGateway, PolicyState
from policy import LruBytePolicy, CostAwarePolicy
from requests.exceptions import ReadTimeout

# ============================
# Global knobs (as requested)
# ============================
NUMBER_OF_PROMPTS = 100
BYTE_BUDGET = 16 * 1024  # KB

MODEL_NAME = "phi3:mini"
SIMILARITY_MAX_DISTANCE = 0.85

# Synthetic answer settings (for clear eviction pressure & deterministic runs)
USE_SYNTHETIC_ANSWERS = True
SYNTHETIC_SIZES = [2048, 4096, 8192]  # 2 KB, 4 KB, 8 KB
# SYNTHETIC_SIZES = [204, 409, 819]
SYNTHETIC_SLEEP_MS = 100              # simulate model latency on misses (ms)


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return float(xs[f]) if f == c else float(xs[f] + (xs[c] - xs[f]) * (k - f))


def virtual_throughput_req_per_s(latencies_ms: List[float]) -> float:
    """Policy-virtual throughput: n / (sum(e2e_ms)/1000)."""
    if not latencies_ms:
        return 0.0
    total_s = sum(latencies_ms) / 1000.0
    n = len(latencies_ms)
    return (n / total_s) if total_s > 0 else 0.0


def load_prompts(path: str, limit: int) -> List[str]:
    """Load prompts from JSON (list of {prompt: ...}); if missing, synthesize with ~20% exact duplicates."""
    p = Path(path)
    if p.exists():
        # robust utf-8 loader
        def _read_utf8(fn):
            for enc in ("utf-8", "utf-8-sig"):
                try:
                    with open(fn, "r", encoding=enc) as f:
                        return json.load(f)
                except UnicodeDecodeError:
                    continue
            with open(fn, "rb") as f:
                t = f.read().decode("utf-8", errors="replace")
            return json.loads(t)

        data = _read_utf8(path)
        prompts = [it["prompt"] for it in data if "prompt" in it]
        return prompts[:limit] if limit else prompts

    # Fallback: synthetic with ~20% duplicates after first occurrence
    base = [f"what did NAME_{i % 7} do to his sister" for i in range(max(2, limit))]
    out = []
    for i in range(limit):
        out.append(base[i % len(base)])
        if i % 5 == 4 and i >= 6:  # ~20% duplicates, and only after earlier entries exist
            out.append(base[(i - 6) % len(base)])
            if len(out) >= limit:
                break
    return out[:limit]


def write_csv(name: str, pol: PolicyState):
    out_path = Path(f"results_{name}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "prompt", "hit", "lookup_ms", "model_ms", "e2e_ms", "path", "policy"])
        for e in pol.events:
            w.writerow([
                f'{e.get("ts", 0):.6f}',
                e.get("prompt", ""),
                e.get("hit", 0),
                f'{e.get("lookup_ms", 0.0):.3f}',
                f'{e.get("model_ms", 0.0):.3f}',
                f'{e.get("e2e_ms", 0.0):.3f}',
                e.get("path", ""),
                pol.name,
            ])
        # trailer stats
        w.writerow([])
        w.writerow(["hits", pol.hits])
        w.writerow(["misses", pol.misses])
        w.writerow(["evictions", getattr(pol.policy_impl, "evictions", 0)])


def summarize(name: str, pol: PolicyState):
    n = len(pol.latencies_ms)
    p50 = percentile(pol.latencies_ms, 50)
    p95 = percentile(pol.latencies_ms, 95)
    p99 = percentile(pol.latencies_ms, 99)
    hit_rate = (pol.hits / n) if n else 0.0
    v_tput = virtual_throughput_req_per_s(pol.latencies_ms)
    evictions = getattr(pol.policy_impl, "evictions", 0)
    print(f"{name:>10}: n={n:3d} hits={pol.hits} misses={pol.misses} "
          f"hit_rate={hit_rate:.2%}  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  "
          f"evictions={evictions}  virtual_throughput={v_tput:.2f} req/s")


if __name__ == "__main__":
    # Quick canonicalization sanity check (should print equal strings)
    print("[check] canon A:", preproc("Some  text…  “quotes” \n"))
    print("[check] canon B:", preproc("Some text... \"quotes\""))

    # ----- Policies (fair A/B with identical byte budget) -----
    lru = PolicyState(
        name="lru",
        run_tag="lru-20k",
        policy_impl=LruBytePolicy(byte_budget=BYTE_BUDGET),
    )
    costaware = PolicyState(
        name="costaware",
        run_tag="costaware-20k",
        policy_impl=CostAwarePolicy(byte_budget=BYTE_BUDGET),
    )
    policies = [lru, costaware]

    # ----- Fresh per-policy caches BEFORE the run -----
    for pol in policies:
        build_cache(
            run_tag=pol.run_tag,
            similarity_max_distance=SIMILARITY_MAX_DISTANCE,
            reset_files=True,  # wipe old files for clean experiment
        )

    # ----- Gateway -----
    gw = MultiCacheGateway(
        model_name=MODEL_NAME,
        policies=policies,
        similarity_max_distance=SIMILARITY_MAX_DISTANCE,
        use_synthetic_answers=USE_SYNTHETIC_ANSWERS,
        synthetic_sizes=SYNTHETIC_SIZES,
        synthetic_sleep_ms=SYNTHETIC_SLEEP_MS,
    )

    # ----- Prompts -----
    prompts = load_prompts("prompt_stream.json", NUMBER_OF_PROMPTS)
    print(f"Prompts loaded: {len(prompts)}")

    # ----- Run -----
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        try:
            gw.ask(p)
        except ReadTimeout:
            print("[WARN] Ollama timed out")
            continue
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            continue
    wall_s = time.perf_counter() - t0
    overall_tput = (len(prompts) / wall_s) if wall_s > 0 else 0.0

    # ----- Results -----
    print(f"\nOverall: requests={len(prompts)} duration={wall_s:.2f}s "
          f"throughput={overall_tput:.2f} req/s")

    for pol in policies:
        summarize(pol.name, pol)
        write_csv(pol.name, pol)

    print("\nCSVs -> results_lru.csv, results_costaware.csv")
