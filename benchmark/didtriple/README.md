# Triple DiD Benchmark

Benchmark suite comparing the Python `ddd` estimator against the R `triplediff` package.

## Quick Start

```bash
# Run quick validation suite
python -m benchmark.didtriple.run_benchmark --suite quick --python-only

# Run with R comparison (requires R + triplediff package)
python -m benchmark.didtriple.run_benchmark --suite quick

# Custom configuration
python -m benchmark.didtriple.run_benchmark --n-units 5000 --multi-period --runs 5
```

## Available Suites

| Suite | Description |
| ----- | ----------- |
| `quick` | Fast validation (100-1000 units) |
| `scaling_units_2period` | Unit scaling with 2-period DGP |
| `scaling_units_multiperiod` | Unit scaling with multi-period DGP |
| `est_methods` | Compare dr, ipw, reg methods |
| `bootstrap` | Bootstrap iteration scaling |
| `panel_vs_rcs` | Panel vs repeated cross-section |

## CLI Options

```text
--suite SUITE         Predefined benchmark suite
--n-units N           Number of units (default: 1000)
--dgp-type N          DGP type: 1, 2, 3, 4 (default: 1)
--panel               Use panel data (default)
--rcs                 Use repeated cross-section data
--multi-period        Use multi-period DGP
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
| -----------: | -----: | -----: | ------: |
| 100K | 0.10s | 0.6s | 5x |
| 500K | 0.37s | 3.2s | 9x |
| 1M | 0.75s | 6.0s | 8x |
| 2M | 2.33s | 12.2s | **5x** |

Python scales to 10M+ observations. R scales well but is consistently slower.

## R Requirements

For R comparison benchmarks:

- R installation
- `triplediff` package: `install.packages("triplediff")`
- `jsonlite` package: `install.packages("jsonlite")`
