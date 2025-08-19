import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("../results/cache_results.csv")

# Plot histogram
plt.figure(figsize=(10, 6))

ordered_cache_types = ["no_cache", "lru", "slru"]

for cache_type in ordered_cache_types:
    group = df[df["cache_type"] == cache_type]
    plt.hist(group["latency_ms"], bins=100, alpha=0.5, label=cache_type)

# Formatting
plt.title("Latency Distribution by Cache Type")
plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

# Save to file
plt.savefig("latency_histogram_from_csv.png")
plt.show()
