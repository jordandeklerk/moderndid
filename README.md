<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png#gh-light-mode-only" width="250" align="left" alt="moderndid logo"></img>
<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-dark.png#gh-dark-mode-only" width="250" align="left" alt="moderndid logo"></img>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/moderndid/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/moderndid)
[![Documentation](https://readthedocs.org/projects/moderndid/badge/?version=latest)](https://moderndid.readthedocs.io/en/latest/)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/moderndid)](https://github.com/jordandeklerk/moderndid/graphs/commit-activity)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/moderndid)](https://github.com/jordandeklerk/moderndid/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)


__ModernDiD__ is a scalable, GPU-accelerated difference-in-differences library for Python. It consolidates modern DiD estimators from leading econometric research and various R and Stata packages into a single framework with a consistent API. Runs on a single machine, NVIDIA GPUs, and distributed Dask and Spark clusters.

> [!WARNING]
> This package is currently in active development with core estimators and some sensitivity analysis implemented. The API is subject to change.

## Features

- **DiD Estimators** - [Staggered DiD](moderndid/did), [Doubly Robust DiD](moderndid/drdid), [Continuous DiD](moderndid/didcont), [Triple DiD](moderndid/didtriple), [Intertemporal DiD](moderndid/didinter), [Honest DiD](moderndid/didhonest)
- **Dataframe agnostic** - Pass any [Arrow-compatible](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) DataFrame such as [polars](https://pola.rs/), [pandas](https://pandas.pydata.org/), [pyarrow](https://arrow.apache.org/docs/python/), [duckdb](https://duckdb.org/), and more powered by [narwhals](https://narwhals-dev.github.io/narwhals/)
- **Distributed computing** - Scale DiD estimators to billions of observations across multi-node [Dask](https://www.dask.org/) and [Spark](https://spark.apache.org/) clusters with automatic dispatch. Simply pass a Dask or Spark DataFrame to supported estimators and the distributed backend activates transparently
- **Fast computation** - [Polars](https://pola.rs/) for internal data wrangling, [NumPy](https://numpy.org/) vectorization, [Numba](https://numba.pydata.org/) JIT compilation, and threaded parallel compute
- **GPU acceleration** - Optional [CuPy](https://cupy.dev/)-accelerated regression and propensity score estimation across all doubly robust and IPW estimators on NVIDIA GPUs, with multi-GPU scaling in distributed environments
- **Native plots** - Built on [plotnine](https://plotnine.org/) with full plotting customization support with the `ggplot` object
- **Robust inference** - Analytical standard errors, bootstrap (weighted and multiplier), and simultaneous confidence bands

For detailed documentation, including user guides and API reference, see [moderndid.readthedocs.io](https://moderndid.readthedocs.io/en/latest/).

## Installation

The base installation includes core DiD estimators that share the same dependencies (`did`, `drdid`, `didinter`, `didtriple`):

```bash
uv pip install moderndid
```

For full functionality including all estimators, plotting, and performance optimizations:

```bash
uv pip install moderndid[all]
```

### Optional Extras

Extras are additive. They add functionality to the base install, so you always get the core estimators plus whatever extras you specify.

- **`didcont`** - Base + continuous treatment DiD (`cont_did`)
- **`didhonest`** - Base + sensitivity analysis (`honest_did`)
- **`plots`** - Base + visualization (`plot_gt`, `plot_event_study`, ...)
- **`numba`** - Base + faster bootstrap inference
- **`dask`** - Base + distributed estimation via Dask
- **`spark`** - Base + distributed estimation via PySpark
- **`gpu`** - Base + GPU-accelerated estimation (requires CUDA)
- **`all`** - Everything (except `gpu`, which requires specific infrastructure)

```bash
uv pip install moderndid[didcont]     # Base estimators + cont_did
uv pip install moderndid[didhonest]   # Base estimators + sensitivity analysis
uv pip install moderndid[numba]       # Base estimators with faster computations
uv pip install moderndid[dask]        # Base estimators with Dask distributed
uv pip install moderndid[spark]       # Base estimators with Spark distributed
uv pip install moderndid[gpu]         # Base estimators with GPU acceleration
uv pip install moderndid[gpu,dask]    # Combine multiple extras
```

Or install from source:

```bash
uv pip install git+https://github.com/jordandeklerk/moderndid.git
```

### Distributed Computing

For datasets that exceed single-machine memory, pass a Dask or Spark dataFrame to [`att_gt()`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.att_gt.html#moderndid.att_gt) or [`ddd()`](https://moderndid.readthedocs.io/en/latest/api/generated/didtriple/moderndid.ddd.html#moderndid.ddd) and the distributed backend activates automatically. All computation happens on workers via partition-level sufficient statistics. Only small summary matrices return to the driver. Results are numerically identical to the local estimators.

**Dask**

```python
import dask.dataframe as dd
from dask.distributed import Client
import moderndid as did

ddf = dd.read_parquet("panel_data.parquet")
client = Client()

result = did.att_gt(
    data=ddf,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    est_method="dr",
    n_partitions=64,         # partitions per cell (default: total cluster threads)
    max_cohorts=4,           # cohorts to process in parallel
    backend="cupy",          # run worker linear algebra on GPUs (optional)
)

event_study = did.aggte(result, type="dynamic")
```

Add `backend="cupy"` to run worker-side linear algebra on GPUs. For multi-GPU machines, use `dask-cuda` with a `LocalCUDACluster` to pin one worker per GPU.

**Spark**

```python
from pyspark.sql import SparkSession
import moderndid as did

spark = SparkSession.builder.master("local[*]").getOrCreate()
sdf = spark.read.parquet("panel_data.parquet")

result = did.att_gt(
    data=sdf,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    est_method="dr",
    n_partitions=64,         # partitions per cell (default: Spark parallelism)
    max_cohorts=4,           # cohorts to process in parallel
    backend="cupy",          # run partition linear algebra on GPUs (optional)
)

event_study = did.aggte(result, type="dynamic")
```

See the [Distributed Estimation guide](https://moderndid.readthedocs.io/en/latest/user_guide/distributed.html) for usage and the [Distributed Backend Architecture](https://moderndid.readthedocs.io/en/latest/dev/distributed_architecture.html) for details on the design.

### GPU Acceleration

On machines with NVIDIA GPUs, install the `gpu` extra and pass `backend="cupy"` to offload regression and propensity score estimation to the GPU. The backend activates only for that call and reverts automatically. See the [GPU troubleshooting section](#common-troubleshooting-for-gpu) below for guidance on common issues:

```python
import moderndid as did

result = did.att_gt(data,
                    yname="lemp",
                    tname="year",
                    idname="countyreal",
                    gname="first.treat",
                    backend="cupy")
```

You can also set the backend globally with `did.set_backend("cupy")` and revert with `did.set_backend("numpy")`. For multi-GPU scaling, combine with a Dask DataFrame as shown above.

See the [GPU guide](https://moderndid.readthedocs.io/en/latest/user_guide/gpu.html) for details and [GPU benchmark results](scripts/README.md) for performance comparisons across several NVIDIA GPUs.

### Consistent API

All estimators share a unified interface for core parameters, making it easy to switch between methods:

```python
# Staggered DiD
result = did.att_gt(data, yname="y", tname="t", idname="id", gname="g", ...)
# Triple DiD
result = did.ddd(data, yname="y", tname="t", idname="id", gname="g", pname="p", ...)
# Continuous DiD
result = did.cont_did(data, yname="y", tname="t", idname="id", gname="g", dname="dose", ...)
# Doubly robust 2-period DiD
result = did.drdid(data, yname="y", tname="t", idname="id", treatname="treat", ...)
# Intertemporal DiD
result = did.did_multiplegt(data, yname="y", tname="t", idname="id", dname="treat", ...)
```

### Example Datasets

Several classic datasets from the DiD literature are included for experimentation:

```python
did.load_mpdta()       # County teen employment
did.load_nsw()         # NSW job training program
did.load_ehec()        # Medicaid expansion
did.load_engel()       # Household expenditure
did.load_favara_imbs() # Bank lending
did.load_cai2016()     # Crop insurance
```

Synthetic data generators are also available for simulations and benchmarking:

```python
did.gen_did_scalable()           # Staggered DiD panel
did.simulate_cont_did_data()     # Continuous treatment DiD
did.gen_dgp_2periods()           # Two-period triple DiD
did.gen_dgp_mult_periods()       # Staggered triple DiD
did.gen_dgp_scalable()           # Large-scale triple DiD
```

## Quick Start

This example uses county-level teen employment data to estimate the effect of minimum wage increases. States adopted higher minimum wages at different times (2004, 2006, or 2007), making this a staggered adoption design.

The [`att_gt()`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.att_gt.html#moderndid.att_gt) function is a core __ModernDiD__ estimator that estimates the average treatment effect for each group $g$ (defined by when units were first treated) at each time period $t$ in multi-period, staggered adoption designs. We use the doubly robust estimator, which combines outcome regression and propensity score weighting to provide consistent estimates if either model is correctly specified.

```python
import moderndid as did

# County teen employment data
data = did.load_mpdta()

# Estimate group-time average treatment effects
attgt_result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    est_method="dr",
)
print(attgt_result)
```

The output shows treatment effects for each group-time pair, along with pointwise confidence bands that account for multiple testing:

```
==============================================================================
 Group-Time Average Treatment Effects
==============================================================================

┌───────┬──────┬──────────┬────────────┬────────────────────────────┐
│ Group │ Time │ ATT(g,t) │ Std. Error │ [95% Pointwise Conf. Band] │
├───────┼──────┼──────────┼────────────┼────────────────────────────┤
│  2004 │ 2004 │  -0.0105 │     0.0233 │ [-0.0561,  0.0351]         │
│  2004 │ 2005 │  -0.0704 │     0.0310 │ [-0.1312, -0.0097] *       │
│  2004 │ 2006 │  -0.1373 │     0.0364 │ [-0.2087, -0.0658] *       │
│  2004 │ 2007 │  -0.1008 │     0.0344 │ [-0.1682, -0.0335] *       │
│  2006 │ 2004 │   0.0065 │     0.0233 │ [-0.0392,  0.0522]         │
│  2006 │ 2005 │  -0.0028 │     0.0196 │ [-0.0411,  0.0356]         │
│  2006 │ 2006 │  -0.0046 │     0.0178 │ [-0.0394,  0.0302]         │
│  2006 │ 2007 │  -0.0412 │     0.0202 │ [-0.0809, -0.0016] *       │
│  2007 │ 2004 │   0.0305 │     0.0150 │ [ 0.0010,  0.0600] *       │
│  2007 │ 2005 │  -0.0027 │     0.0164 │ [-0.0349,  0.0294]         │
│  2007 │ 2006 │  -0.0311 │     0.0179 │ [-0.0661,  0.0040]         │
│  2007 │ 2007 │  -0.0261 │     0.0167 │ [-0.0587,  0.0066]         │
└───────┴──────┴──────────┴────────────┴────────────────────────────┘

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0

 P-value for pre-test of parallel trends assumption:  0.1681

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Control Group:  Never Treated
 Anticipation Periods:  0

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Estimation Method:  Doubly Robust

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 Reference: Callaway and Sant'Anna (2021)
```

Rows where the confidence band excludes zero are marked with `*`. The pre-test p-value tests whether pre-treatment effects are jointly zero, providing a diagnostic for the parallel trends assumption.

We can plot these results using the `plot_gt()` functionality:

```python
did.plot_gt(attgt_result)
```

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/att.png" alt="ATT plot">

While group-time effects are useful, they can be difficult to summarize when there are many groups and time periods. The `aggte` function aggregates these into more interpretable summaries. Setting `type="dynamic"` produces an event study that shows how effects evolve relative to treatment timing:

```python
event_study = did.aggte(attgt_result, type="dynamic")
print(event_study)
```

```
==============================================================================
 Aggregate Treatment Effects (Event Study)
==============================================================================

 Overall summary of ATT's based on event-study/dynamic aggregation:

┌─────────┬────────────┬────────────────────────┐
│     ATT │ Std. Error │ [95% Conf. Interval]   │
├─────────┼────────────┼────────────────────────┤
│ -0.0772 │     0.0200 │ [ -0.1164,  -0.0381] * │
└─────────┴────────────┴────────────────────────┘


 Dynamic Effects:

┌────────────┬──────────┬────────────┬────────────────────────────┐
│ Event time │ Estimate │ Std. Error │ [95% Pointwise Conf. Band] │
├────────────┼──────────┼────────────┼────────────────────────────┤
│         -3 │   0.0305 │     0.0150 │ [-0.0078,  0.0688]         │
│         -2 │  -0.0006 │     0.0133 │ [-0.0344,  0.0333]         │
│         -1 │  -0.0245 │     0.0142 │ [-0.0607,  0.0118]         │
│          0 │  -0.0199 │     0.0118 │ [-0.0501,  0.0102]         │
│          1 │  -0.0510 │     0.0169 │ [-0.0940, -0.0079] *       │
│          2 │  -0.1373 │     0.0364 │ [-0.2301, -0.0444] *       │
│          3 │  -0.1008 │     0.0344 │ [-0.1883, -0.0133] *       │
└────────────┴──────────┴────────────┴────────────────────────────┘

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Control Group: Never Treated
 Anticipation Periods: 0

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Estimation Method: Doubly Robust

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 Reference: Callaway and Sant'Anna (2021)
```

Event time 0 is the period of first treatment, e.g., the on-impact effect, negative event times are pre-treatment periods, and positive event times are post-treatment periods. Pre-treatment effects near zero lean in support of the parallel trends assumption (but do not confirm it), while post-treatment effects reveal how the treatment impact evolves over time. The overall ATT at the top provides a single summary measure across all post-treatment periods.

We can also use built-in plotting functionality to plot the event study results with `plot_event_study()`:

```python
did.plot_event_study(event_study)
```

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/event_study.png" alt="Event study plot">

## Common Troubleshooting for GPU

If `set_backend("cupy")` raises **`CuPy is not installed`**, the most common cause is installing the generic `cupy` package, which tries to compile from source. Instead, install a prebuilt wheel that matches your CUDA driver version:

```bash
uv pip install cupy-cuda12x   # CUDA 12.x
uv pip install cupy-cuda11x   # CUDA 11.x
```

Run `nvidia-smi` to check which CUDA version your driver supports. After installing, restart your Python process (or notebook runtime) before importing ModernDiD (CuPy availability is checked once at import time).

If you see **`cudaErrorInsufficientDriver`**, the installed CuPy wheel expects a newer CUDA version than your driver provides. Check `nvidia-smi` and switch to the matching wheel.

If you see **`No CUDA GPU is available`**, make sure `nvidia-smi` shows a device. In cloud notebooks, verify that a GPU runtime is selected.

## Available Methods

Each core module includes a dedicated walkthrough covering methodology background, API usage, and guidance on interpreting results.

### Core Implementations

| Module | Description | Reference |
|--------|-------------|-----------|
| [`moderndid.did`](moderndid/did) | Staggered DiD with group-time effects | [Callaway & Sant'Anna (2021)](https://arxiv.org/pdf/1803.09015) |
| [`moderndid.drdid`](moderndid/drdid) | Doubly robust 2-period estimators | [Sant'Anna & Zhao (2020)](https://arxiv.org/pdf/1812.01723) |
| [`moderndid.didhonest`](moderndid/didhonest) | Sensitivity analysis for parallel trends | [Rambachan & Roth (2023)](https://asheshrambachan.github.io/assets/files/hpt-draft.pdf) |
| [`moderndid.didcont`](moderndid/didcont) | Continuous/multi-valued treatments | [Callaway et al. (2024)](https://arxiv.org/pdf/2107.02637) |
| [`moderndid.didtriple`](moderndid/didtriple) | Triple difference-in-differences | [Ortiz-Villavicencio & Sant'Anna (2025)](https://arxiv.org/pdf/2505.09942) |
| [`moderndid.didinter`](moderndid/didinter) | Intertemporal DiD with non-absorbing treatment | [Chaisemartin & D'Haultfœuille (2024)](https://arxiv.org/pdf/2007.04267) |

### Planned Development

| Module | Description | Reference |
|--------|-------------|-----------|
| `moderndid.didml` | Machine learning approaches to DiD | [Hatamyar et al. (2023)](https://arxiv.org/pdf/2310.11962) |
| `moderndid.drdidweak` | Robust to weak overlap | [Ma et al. (2023)](https://arxiv.org/pdf/2304.08974) |
| `moderndid.didcomp` | Compositional changes in repeated cross-sections | [Sant'Anna & Xu (2025)](https://arxiv.org/pdf/2304.13925) |
| `moderndid.didimpute` | Imputation-based estimators | [Borusyak, Jaravel, & Spiess (2024)](https://arxiv.org/pdf/2108.12419) |
| `moderndid.didbacon` | Goodman-Bacon decomposition | [Goodman-Bacon (2019)](https://cdn.vanderbilt.edu/vu-my/wp-content/uploads/sites/2318/2019/07/29170757/ddtiming_7_29_2019.pdf) |
| `moderndid.didlocal` | Local projections DiD | [Dube et al. (2025)](https://www.nber.org/system/files/working_papers/w31184/w31184.pdf) |
| `moderndid.did2s` | Two-stage DiD | [Gardner (2021)](https://jrgcmu.github.io/2sdd_current.pdf) |
| `moderndid.etwfe` | Extended two-way fixed effects | [Wooldridge (2021)](https://ssrn.com/abstract=3906345), [Wooldridge (2023)](https://doi.org/10.1093/ectj/utad016) |
| `moderndid.functional` | Specification tests | [Roth & Sant'Anna (2023)](https://arxiv.org/pdf/2010.04814) |
