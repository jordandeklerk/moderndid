# ddd Benchmark

Benchmark suite comparing Python `ddd` (triple difference) estimator against the R `triplediff` package.

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
|-------|-------------|
| `quick` | Fast validation (100-1000 units) |
| `scaling_units_2period` | Unit scaling with 2-period DGP |
| `scaling_units_multiperiod` | Unit scaling with multi-period DGP |
| `est_methods` | Compare dr, ipw, reg methods |
| `bootstrap` | Bootstrap iteration scaling |
| `panel_vs_rcs` | Panel vs repeated cross-section |

## CLI Options

```
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

## Programmatic Usage

```python
from benchmark.didtriple import DDDPythonRunner
from moderndid import gen_dgp_2periods

# Generate data
dgp_result = gen_dgp_2periods(n=1000, dgp_type=1, panel=True)

# Run benchmark
runner = DDDPythonRunner()
result = runner.time_ddd(dgp_result["data"], n_runs=5)
print(f"Mean time: {result.mean_time:.4f}s")
```

## R Requirements

For R comparison benchmarks:
- R installation
- `triplediff` package: `install.packages("triplediff")`
- `jsonlite` package: `install.packages("jsonlite")`
