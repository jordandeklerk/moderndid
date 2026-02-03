# Benchmark Suite

Comprehensive benchmark suite comparing computational performance of the Python `moderndid` package against R equivalents.

## Quick Start

```bash
# att_gt benchmarks
python -m benchmark.did.run_benchmark --suite quick --python-only

# ddd benchmarks
python -m benchmark.didtriple.run_benchmark --suite quick --python-only
```

See module-specific READMEs for detailed CLI options:
- [`benchmark/did/README.md`](did/README.md) — att_gt vs R did
- [`benchmark/didtriple/README.md`](didtriple/README.md) — ddd vs R triplediff

## Methodology

The benchmark suite ensures fair comparisons between Python and R:

- **Same Data** — Both receive identical datasets (R via CSV export)
- **Same Configuration** — Equivalent parameters for both implementations
- **Same Timing Protocol** — Warmup runs, garbage collection, multiple timed runs
- **Same Computation** — Verified via matching number of estimates

## Requirements

**Python:**
- Python 3.10+
- moderndid, polars, numpy

**R (optional):**
- R with `did`, `triplediff`, and `jsonlite` packages

## Results

![Python vs R Performance](output/benchmark_scaling.png)
