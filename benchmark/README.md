# Benchmark Suite

Comprehensive benchmark suite comparing computational performance of the Python `moderndid` package against R equivalents (`did` and `triplediff` packages).

## Requirements

- Python 3.10+
- moderndid package installed
- polars
- numpy

For R comparison:
- R with `did`, `triplediff`, and `jsonlite` packages

## Results

![Python vs R Performance](output/benchmark_scaling.png)

- **att_gt**: Up to **117x faster** than R at 2M observations.
- **ddd**: Up to **8x faster** than R at 5M units.

## Methodology

The benchmark suite ensures fair apples-to-apples comparisons between Python and R implementations.

### Same Data

Both implementations receive the identical dataset. R receives the data via CSV export from the same Polars DataFrame that Python uses directly.

### Same Configuration

Both implementations are called with equivalent parameters:

- `est_method` — Estimation method (dr, ipw, reg)
- `control_group` — Control group specification (nevertreated)
- `boot` — Bootstrap enabled/disabled
- `biters` — Number of bootstrap iterations
- `xformla` — Covariate formula
- `yname`, `tname`, `idname`, `gname` — Column name mappings

### Same Timing Protocol

1. **Warmup runs** — Both run `n_warmup` untimed iterations to allow JIT compilation and caching
2. **Garbage collection** — `gc.collect()` (Python) and `gc()` (R) called before each timed run
3. **Multiple timed runs** — Same `n_runs` for both, reporting mean, std, min, and max times
4. **High-resolution timing** — Python uses `time.perf_counter()`, R uses `Sys.time()` with `difftime()`

### Same Computation

Both call their respective `att_gt()` functions with equivalent arguments and produce the same number of group-time ATT estimates (verified via `n_estimates`).

## Quick Start

```bash
python -m benchmark.run_benchmark --suite quick --python-only

python -m benchmark.run_benchmark --n-units 1000 --n-periods 5 --runs 3

python -m benchmark.run_benchmark --suite scaling_units
```

## Available Benchmark Suites

| Suite | Description |
|-------|-------------|
| `quick` | Fast validation (100-1000 units) |
| `scaling_units` | Unit scaling (100 to 100,000 units) |
| `scaling_periods` | Period scaling (5 to 20 periods) |
| `scaling_groups` | Group scaling (3 to 10 groups) |
| `est_methods` | Compare dr, ipw, reg methods |
| `bootstrap` | Bootstrap iteration scaling |
| `large_scale` | Million+ observations |

## CLI Options

```
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

## Output

Results are saved to `benchmark/output/` in both CSV and JSON formats with timestamps.

## Programmatic Usage

```python
from benchmark import (
    BenchmarkConfig,
    StaggeredDIDDGP,
    PythonBenchmarkRunner,
    RBenchmarkRunner,
)

dgp = StaggeredDIDDGP(n_units=1000, n_periods=5, n_groups=3)
data = dgp.generate_data()

runner = PythonBenchmarkRunner()
result = runner.time_att_gt(data["df"], n_runs=5)
print(f"Mean time: {result.mean_time:.4f}s")
```
