import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- helpers ----------
def safe_float_series(df, col):
    if col not in df.columns:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.dropna()

def hit_rate(df):
    if "hit" not in df.columns or len(df) == 0:
        return 0.0
    s = pd.to_numeric(df["hit"], errors="coerce").fillna(0)
    return float(s.sum()) / max(1, len(s))

def virtual_throughput(df):
    e2e = safe_float_series(df, "e2e_ms")
    total_s = e2e.sum() / 1000.0
    n = len(e2e)
    return (n / total_s) if total_s > 0 else 0.0

def percentile(vals, p):
    xs = np.sort(np.array(vals, dtype=float))
    if xs.size == 0:
        return 0.0
    k = (xs.size - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, xs.size - 1)
    return float(xs[f]) if f == c else float(xs[f] + (xs[c] - xs[f]) * (k - f))

def load_csv_labeled(path, label=None):
    df = pd.read_csv(path)
    name = label or Path(path).stem.replace("results_", "")
    return name, df

# ---------- plotting ----------
def plot_latency_cdf(datasets, outdir):
    plt.figure()
    for name, df in datasets:
        e2e = safe_float_series(df, "e2e_ms")
        if e2e.empty:
            continue
        x = np.sort(e2e.values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=name)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cumulative probability")
    plt.title("Latency CDF")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "latency_cdf.png", dpi=160)
    plt.close()

def plot_hit_rate(datasets, outdir):
    labels, vals = [], []
    for name, df in datasets:
        labels.append(name)
        vals.append(hit_rate(df) * 100.0)
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Hit rate (%)")
    plt.title("Cache Hit Rate")
    plt.tight_layout()
    plt.savefig(outdir / "hit_rate.png", dpi=160)
    plt.close()

def plot_throughput(datasets, outdir):
    labels, vals = [], []
    for name, df in datasets:
        labels.append(name)
        vals.append(virtual_throughput(df))
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Requests / second")
    plt.title("Virtual Throughput (per policy)")
    plt.tight_layout()
    plt.savefig(outdir / "throughput.png", dpi=160)
    plt.close()

def plot_percentiles(datasets, outdir):
    labels = [name for name, _ in datasets]
    p50 = []
    p95 = []
    p99 = []
    for _, df in datasets:
        e2e = safe_float_series(df, "e2e_ms").tolist()
        p50.append(percentile(e2e, 50))
        p95.append(percentile(e2e, 95))
        p99.append(percentile(e2e, 99))
    x = np.arange(len(labels))
    width = 0.25
    plt.figure()
    plt.bar(x - width, p50, width, label="p50")
    plt.bar(x,         p95, width, label="p95")
    plt.bar(x + width, p99, width, label="p99")
    plt.xticks(x, labels)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Percentiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "percentiles.png", dpi=160)
    plt.close()

def print_summary(datasets):
    print("\n=== Summary ===")
    for name, df in datasets:
        e2e = safe_float_series(df, "e2e_ms").tolist()
        hr  = hit_rate(df) * 100.0
        vt  = virtual_throughput(df)
        p50 = percentile(e2e, 50)
        p95 = percentile(e2e, 95)
        p99 = percentile(e2e, 99)
        print(f"{name:>12}: n={len(e2e):4d}  hit_rate={hr:6.2f}%  "
              f"virt_tput={vt:6.2f} req/s  p50={p50:7.1f} ms  p95={p95:7.1f} ms  p99={p99:7.1f} ms")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot benchmark CSVs (baseline vs cost-aware).")
    ap.add_argument("--csv", nargs="+", default=["results_lru.csv", "results_costaware.csv"],
                    help="List of CSV files to compare.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional labels matching the CSV order.")
    ap.add_argument("--out", default="plots", help="Output directory for images.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for i, path in enumerate(args.csv):
        label = (args.labels[i] if args.labels and i < len(args.labels) else None)
        datasets.append(load_csv_labeled(path, label))

    # Make plots
    plot_latency_cdf(datasets, outdir)
    plot_hit_rate(datasets, outdir)
    plot_throughput(datasets, outdir)
    plot_percentiles(datasets, outdir)

    # Console summary
    print_summary(datasets)
    print(f"\nSaved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
