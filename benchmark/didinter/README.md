# Intertemporal DiD Benchmark

Benchmark suite comparing the Python `did_multiplegt` estimator against the R `DIDmultiplegtDYN` package.

## Quick Start

```bash
# Run quick validation suite
python -m benchmark.didinter.run_benchmark --suite quick --python-only

# Run with R comparison (requires R + DIDmultiplegtDYN package)
python -m benchmark.didinter.run_benchmark --suite quick

# Custom configuration
python -m benchmark.didinter.run_benchmark --n-units 1000 --n-periods 10 --effects 5 --runs 5
```

## Available Suites

| Suite | Description |
|-------|-------------|
| `quick` | Fast validation (100-1000 units) |
| `scaling_units` | Unit scaling (100 to 10,000 units) |
| `scaling_periods` | Period scaling (5 to 20 periods) |
| `scaling_effects` | Effects/placebo scaling |
| `normalized` | Normalized vs non-normalized effects |
| `bootstrap` | Bootstrap iteration scaling |
| `large_scale` | Large scale (10K to 100K units) |

## CLI Options

```
--suite SUITE         Predefined benchmark suite
--n-units N           Number of units (default: 500)
--n-periods N         Number of time periods (default: 10)
--effects N           Number of effects to estimate (default: 3)
--placebo N           Number of placebos to estimate (default: 2)
--normalized          Use normalized effects
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
|-------------:|-------:|--:|--------:|
| 10K | 0.17s | 3.1s | 18x |
| 100K | 0.49s | 19.4s | 40x |
| 200K | 0.79s | 72.4s | 91x |
| 300K | 1.07s | 114.0s | **106x** |

## R Limitations

The R `DIDmultiplegtDYN` package runs out of memory at larger scales. R benchmarks are only available up to ~300K observations.

## R Requirements

For R comparison benchmarks:
- R installation
- `DIDmultiplegtDYN` package: `install.packages("DIDmultiplegtDYN")`
- `jsonlite` package: `install.packages("jsonlite")`
- `data.table` package: `install.packages("data.table")`
