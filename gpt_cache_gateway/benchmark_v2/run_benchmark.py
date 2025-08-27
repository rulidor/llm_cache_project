import csv, time
from gateway import CacheGateway

def percentile(xs, p):
    xs = sorted(xs)
    if not xs: return None
    k = (len(xs)-1) * p/100.0
    f, c = int(k), min(int(k)+1, len(xs)-1)
    return xs[f] if f == c else xs[f] + (xs[c]-xs[f]) * (k - f)

def build_prompts(n=100):
    base = "Explain BLE advertising briefly."
    prompts = []
    for i in range(n):
        if i % 3 == 0:
            prompts.append(base)  # duplicates → should hit
        else:
            prompts.append(f"Prompt {i}: summarize caching policy tradeoffs in two sentences. {i}")
    return prompts

def summarize(tag, gw, duration):
    n = len(gw.latencies_ms)
    hit_rate = (gw.hits / n) if n else 0.0
    print(f"\n=== {tag} ===")
    print(f"requests={n} hits={gw.hits} misses={gw.misses} hit_rate={hit_rate:.2%} evictions={getattr(gw.policy,'evictions',0)}")
    print(f"p50={percentile(gw.latencies_ms,50):.1f} ms  "
          f"p95={percentile(gw.latencies_ms,95):.1f} ms  "
          f"p99={percentile(gw.latencies_ms,99):.1f} ms")
    print(f"throughput={(n/duration) if duration>0 else 0.0:.2f} req/s over {duration:.2f}s")

def write_csv(filename, events, summary):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt","latency_ms","hit"])
        for e in events:
            w.writerow([e["prompt"], f'{e["latency_ms"]:.3f}', e["hit"]])
        # summary footer
        w.writerow([])
        for k, v in summary.items():
            w.writerow([k, v])

if __name__ == "__main__":
    prompts = build_prompts(100)

    # --- Baseline: Noop/LRU, separate files via run_tag
    gw = CacheGateway(model_name="phi3:mini",
                      policy="noop",
                      byte_budget=32 * 1024,
                      run_tag="baseline",
                      similarity_max_distance=0.6)
    t0 = time.perf_counter()
    for p in prompts:
        _ = gw.ask(p)
    dur = time.perf_counter() - t0
    summarize("Baseline / Noop policy", gw, dur)
    write_csv("results_baseline.csv", gw.events, {
        "requests": len(gw.latencies_ms),
        "hits": gw.hits,
        "misses": gw.misses,
        "evictions": getattr(gw.policy, "evictions", 0),
        "duration_s": f"{dur:.3f}"
    })

    # --- Cost-aware: tight byte budget to force evictions; separate files via run_tag
    gw2 = CacheGateway(model_name="phi3:mini",
                       policy="cost",
                       byte_budget=32*1024,      # 32 KB → will trigger evictions quickly
                       run_tag="costaware-32k",
                       similarity_max_distance=0.6)
    t0 = time.perf_counter()
    for p in prompts:
        _ = gw2.ask(p)
    dur2 = time.perf_counter() - t0
    summarize("Cost-aware policy (32KB budget)", gw2, dur2)
    write_csv("results_costaware_32k.csv", gw2.events, {
        "requests": len(gw2.latencies_ms),
        "hits": gw2.hits,
        "misses": gw2.misses,
        "evictions": getattr(gw2.policy, "evictions", 0),
        "duration_s": f"{dur2:.3f}"
    })

    print("\nCSV written: results_baseline.csv, results_costaware_32k.csv")
