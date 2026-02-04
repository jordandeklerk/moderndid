# Continuous DiD Benchmark

Benchmark suite comparing the Python `cont_did` estimator against the R `contdid` package.

## Quick Start

```bash
# Run quick validation suite
python -m benchmark.didcont.run_benchmark --suite quick --python-only

# Run with R comparison (requires R + contdid package)
python -m benchmark.didcont.run_benchmark --suite quick

# Custom configuration
python -m benchmark.didcont.run_benchmark --n-units 5000 --n-periods 4 --runs 5
```

## Available Suites

| Suite | Description |
| ----- | ----------- |
| `quick` | Fast validation (100-1000 units) |
| `scaling_units` | Scale from 100 to 10,000 units |
| `scaling_periods` | Scale from 3 to 8 time periods |
| `target_parameters` | Compare level (ATT) vs slope (ACRT) |
| `aggregation_types` | Compare dose vs eventstudy aggregation |
| `dose_methods` | Compare B-spline configurations |
| `cck_method` | Non-parametric CCK estimator |
| `bootstrap` | Bootstrap inference overhead |
| `large_scale` | Scale up to 5M units (20M observations) |

## CLI Options

```text
--suite SUITE         Predefined benchmark suite
--n-units N           Number of units (default: 500)
--n-periods N         Number of time periods (default: 4)
--target-parameter    level or slope (default: level)
--aggregation         dose or eventstudy (default: dose)
--dose-method         parametric or cck (default: parametric)
--degree N            B-spline degree (default: 3)
--num-knots N         Interior knots (default: 0)
--boot                Enable bootstrap
--biters N            Bootstrap iterations (default: 100)
--warmup N            Warmup runs (default: 1)
--runs N              Timed runs (default: 5)
--python-only         Skip R benchmarks
--output-dir DIR      Output directory (default: benchmark/output)
--quiet               Suppress verbose output
```

## Python vs R Comparison

| Observations | Python | R | Speedup |
| -----------: | -----: | ------: | ------: |
| 100K | 1.49s | 36.1s | 24x |
| 500K | 6.21s | 132.8s | 21x |
| 1M | 12.33s | 278.0s | 23x |
| 2M | 24.85s | 826.3s | **33x** |

Python scales to 10M+ observations.

## R Requirements

For R comparison benchmarks:

- R installation
- `contdid` package: `install.packages("contdid")`
- `jsonlite` package: `install.packages("jsonlite")`
