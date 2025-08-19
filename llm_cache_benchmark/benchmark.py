
import json
import time
import statistics
from llm_cache_benchmark.cache import LRUCache, SLRUCache
from llm_cache_benchmark.model import query_model
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from llm_cache_benchmark.config import PROMPT_FILE, MAX_PROMPTS

def load_prompts(filepath: str, max_prompts=100):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    prompts = []
    for entry in raw_data:
        if "prompt" in entry:
            prompts.append(entry["prompt"])
        elif "conversation" in entry:
            for turn in entry["conversation"]:
                if "user" in turn:
                    prompts.append(turn["user"])
    return prompts[:max_prompts]

def run_cache_test(cache_type: str, prompts: list):
    if cache_type == "lru":
        cache = LRUCache(max_size=100)
    elif cache_type == "slru":
        cache = SLRUCache(protected_size=70, probationary_size=30)
    else:
        cache = None  # No cache

    latencies = []
    hits = 0
    start_time = time.perf_counter()
    for prompt in prompts:
        start = time.perf_counter()
        if cache is not None and prompt in cache:
            # Hit
            hits += 1
            _ = cache[prompt]
        else:
            # Miss
            response = query_model(prompt)
            if cache is not None:
                cache[prompt] = response
        end = time.perf_counter()
        latencies.append(end - start)
    total_time = time.perf_counter() - start_time
    throughput = len(prompts) / total_time
    return {
        "cache_type": cache_type or "no_cache",
        "hits": hits,
        "total": len(prompts),
        "latencies": latencies,
        "throughput": throughput
    }

def plot_latency_distributions(results):
    plt.figure(figsize=(10, 6))
    for result in results:
        label = result["cache_type"]
        plt.hist([l * 1000 for l in result["latencies"]], bins=30, alpha=0.5, label=label)
    plt.title("Latency Distribution (ms)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_comparison.png")

def save_results_csv(results, filename="cache_results.csv"):
    rows = []
    for result in results:
        for latency in result["latencies"]:
            rows.append({
                "cache_type": result["cache_type"],
                "latency_ms": latency * 1000
            })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df

def print_summary_table(results):
    table_data = []
    for result in results:
        mean_latency = statistics.mean(result["latencies"]) * 1000
        p95_latency = statistics.quantiles(result["latencies"], n=100)[94] * 1000
        table_data.append({
            "Cache Type": result["cache_type"],
            "Hit Rate": f"{result['hits']}/{result['total']} = {result['hits']/result['total']:.2%}",
            "Mean Latency (ms)": f"{mean_latency:.2f}",
            "P95 Latency (ms)": f"{p95_latency:.2f}",
            "Throughput (prompts/sec)": f"{result['throughput']:.2f}"
        })
    df = pd.DataFrame(table_data)
    df.to_csv("cache_performance_table.csv", index=False)
    print(df)

def main():
    path = Path(PROMPT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Please provide a JSON file with prompts named '{PROMPT_FILE}'.")

    prompts = load_prompts(str(path), max_prompts=MAX_PROMPTS)

    results = []
    for cache_type in [None, "lru", "slru"]:
        label = cache_type or "No Cache"
        print(f"Running benchmark for: {label}")
        result = run_cache_test(cache_type, prompts)
        results.append(result)
        print(f"  Hit rate: {result['hits']}/{result['total']} = {result['hits']/result['total']:.2%}")
        print(f"  Mean latency: {statistics.mean(result['latencies'])*1000:.2f} ms")
        print(f"  95th percentile: {statistics.quantiles(result['latencies'], n=100)[94]*1000:.2f} ms")
        print(f"  Throughput: {result['throughput']:.2f} prompts/sec")

    plot_latency_distributions(results)
    save_results_csv(results)
    print("Results saved to 'cache_results.csv' and 'latency_comparison.png'.")
    print_summary_table(results)

if __name__ == "__main__":
    main()
