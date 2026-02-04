# Continuous DiD Benchmark

Benchmark suite comparing the Python `cont_did` estimator against the R `contdid` package.

## Quick Start

```bash
# Quick test (100-1000 units)
python -m benchmark.didcont.run_benchmark --suite quick --python-only

# Large scale (up to 20M observations)
python -m benchmark.didcont.run_benchmark --suite large_scale --python-only

# With R comparison (limited to <900 units due to R bug)
python -m benchmark.didcont.run_benchmark --n-units 500 --n-periods 4
```

## Available Suites

| Suite | Description |
|-------|-------------|
| `quick` | Fast test: 100, 500, 1000 units |
| `scaling_units` | Scale from 100 to 10,000 units |
| `scaling_periods` | Scale from 3 to 8 time periods |
| `target_parameters` | Compare level (ATT) vs slope (ACRT) |
| `aggregation_types` | Compare dose vs eventstudy aggregation |
| `dose_methods` | Compare B-spline configurations |
| `cck_method` | Non-parametric CCK estimator |
| `bootstrap` | Bootstrap inference overhead |
| `large_scale` | Scale up to 5M units (20M observations) |

## CLI Options

```bash
python -m benchmark.didcont.run_benchmark [OPTIONS]

Options:
  --suite SUITE          Predefined benchmark suite
  --n-units INT          Number of units (default: 500)
  --n-periods INT        Number of time periods (default: 4)
  --target-parameter     level or slope (default: level)
  --aggregation          dose or eventstudy (default: dose)
  --dose-method          parametric or cck (default: parametric)
  --degree INT           B-spline degree (default: 3)
  --num-knots INT        Interior knots (default: 0)
  --boot                 Enable bootstrap
  --biters INT           Bootstrap iterations (default: 100)
  --python-only          Skip R benchmarks
  --runs INT             Timed runs per config (default: 5)
  --warmup INT           Warmup runs (default: 1)
```

## Results Summary

### Scaling Performance (Parametric Method)

| Units | Observations | Time (s) | Throughput |
|------:|-------------:|---------:|-----------:|
| 100 | 400 | 0.05 | ~8.5K obs/sec |
| 1,000 | 4,000 | 0.09 | ~42K obs/sec |
| 10,000 | 40,000 | 0.45 | ~88K obs/sec |
| 100,000 | 400,000 | 4.15 | ~96K obs/sec |
| 1,000,000 | 4,000,000 | 41.85 | ~96K obs/sec |
| 5,000,000 | 20,000,000 | 239.60 | ~83K obs/sec |

### Python vs R Comparison

| Observations | Python | R | Speedup |
|-------------:|-------:|--:|--------:|
| 500K | 6.2s | 132.8s | **21x** |
| 1M | 12.3s | 278.0s | **23x** |
| 2M | 24.9s | 826.3s | **33x** |

Python scales to 10M+ observations while R fails beyond 2M observations.
