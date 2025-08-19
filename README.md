
# LLM Cache Benchmark

Benchmark and compare caching strategies (LRU, SLRU, or no cache)
for LLM inference using trace datasets.

## Quick Start

1. Place your trace dataset (e.g. `lmsys_subset.json`) in the project root.
2. Configure mode in `config.py`:
   - USE_SIMULATION = True → Fast
   - USE_MOCK_API = True → Simulated latency
   - Otherwise → Uses OpenAI API

Run the benchmark:
```bash
python benchmark.py
```

Outputs:
- `cache_results.csv`
- `latency_comparison.png`
