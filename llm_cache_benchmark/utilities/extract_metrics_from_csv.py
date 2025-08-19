import pandas as pd
import statistics
from collections import defaultdict

# Load the CSV
df = pd.read_csv("../results/cache_results.csv")

# Initialize metrics
metrics = defaultdict(dict)

# Group by cache type
for cache_type, group in df.groupby("cache_type"):
    latencies = group["latency_ms"].tolist()
    total = len(latencies)
    hits = group[group["latency_ms"] < 200].shape[0]  # Approximate hits as fast responses (<200ms)

    metrics[cache_type]["Hit Rate"] = f"{hits}/{total} = {hits / total:.2%}"
    metrics[cache_type]["Mean Latency (ms)"] = round(statistics.mean(latencies), 2)
    metrics[cache_type]["P95 Latency (ms)"] = round(statistics.quantiles(latencies, n=100)[94], 2)
    total_time_sec = sum(latencies) / 1000  # Convert to seconds
    metrics[cache_type]["Throughput (prompts/sec)"] = round(total / total_time_sec, 2)

# Display results
for cache_type, vals in metrics.items():
    print(f"\nMetrics for {cache_type}:")
    for k, v in vals.items():
        print(f"  {k}: {v}")
