# Staggered DiD Benchmark

Benchmark suite comparing the Python `att_gt` estimator against the R `did` package.

## Quick Start

```bash
# Run quick validation suite
python -m benchmark.did.run_benchmark --suite quick --python-only

# Run with R comparison (requires R + did package)
python -m benchmark.did.run_benchmark --suite quick

# Custom configuration
python -m benchmark.did.run_benchmark --n-units 5000 --n-periods 10 --runs 5
```

## Available Suites

| Suite | Description |
| ----- | ----------- |
| `quick` | Fast validation (100-1000 units) |
| `scaling_units` | Unit scaling (100 to 100,000 units) |
| `scaling_periods` | Period scaling (5 to 20 periods) |
| `scaling_groups` | Group scaling (3 to 10 groups) |
| `est_methods` | Compare dr, ipw, reg methods |
| `bootstrap` | Bootstrap iteration scaling |
| `large_scale` | Million+ observations |

## CLI Options

```text
--suite SUITE         Predefined benchmark suite
--n-units N           Number of units (default: 1000)
--n-periods N         Number of time periods (default: 5)
--n-groups N          Number of treatment groups (default: 3)
--n-covariates N      Number of covariates (default: 0)
--est-method METHOD   Estimation method: dr, ipw, reg (default: dr)
--boot                Enable bootstrap
--biters N            Bootstrap iterations (default: 100)
--warmup N            Warmup runs (default: 1)
--runs N              Timed runs (default: 5)
--seed N              Random seed (default: 42)
--python-only         Skip R benchmarks
--output-dir DIR      Output directory (default: benchmark/output)
--quiet               Suppress verbose output
```

## Python vs R Comparison

| Observations | Python | R | Speedup |
| -----------: | -----: | -: | ------: |
| 100K | 0.08s | 1.4s | 17x |
| 500K | 0.30s | 18.6s | 61x |
| 1M | 0.79s | 70.6s | 89x |
| 2M | 1.02s | 233.7s | **229x** |

Python scales to 10M+ observations while R times out beyond 2M observations.

## R Requirements

For R comparison benchmarks:

- R installation
- `did` package: `install.packages("did")`
- `jsonlite` package: `install.packages("jsonlite")`
