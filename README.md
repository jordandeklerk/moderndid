<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png#gh-light-mode-only" width="250" align="left" alt="moderndid logo"></img>
<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-dark.png#gh-dark-mode-only" width="250" align="left" alt="moderndid logo"></img>

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)
[![PyPI -Version](https://img.shields.io/pypi/v/moderndid.svg)](https://pypi.org/project/moderndid/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/moderndid/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/moderndid)
[![Build Status](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/moderndid/badge/?version=latest)](https://moderndid.readthedocs.io/en/latest/)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/moderndid)](https://github.com/jordandeklerk/moderndid/graphs/commit-activity)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/moderndid)](https://github.com/jordandeklerk/moderndid/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
<!-- [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) -->

__ModernDiD__ is a scalable, GPU-accelerated difference-in-differences library for Python. It consolidates modern DiD estimators from leading econometric research and various R and Stata packages into a single framework with a consistent API. Runs on a single machine, NVIDIA GPUs, and distributed Spark and Dask clusters.

## Features

- **DiD Estimators** - [Staggered DiD](https://moderndid.readthedocs.io/en/latest/api/multiperiod.html), [Doubly Robust DiD](https://moderndid.readthedocs.io/en/latest/api/drdid.html), [Continuous DiD](https://moderndid.readthedocs.io/en/latest/api/didcont.html), [Triple DiD](https://moderndid.readthedocs.io/en/latest/api/didtriple.html), [Intertemporal DiD](https://moderndid.readthedocs.io/en/latest/api/didinter.html), [Honest DiD](https://moderndid.readthedocs.io/en/latest/api/honestdid.html).
- **Dataframe agnostic** - Pass any [Arrow-compatible](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) DataFrame such as [polars](https://pola.rs/), [pandas](https://pandas.pydata.org/), [pyarrow](https://arrow.apache.org/docs/python/), [duckdb](https://duckdb.org/), and more powered by [narwhals](https://narwhals-dev.github.io/narwhals/).
- **Distributed computing** - Scale to billions of observations across [Spark](https://spark.apache.org/) and [Dask](https://www.dask.org/) clusters. Pass a distributed DataFrame and the backend activates transparently.
- **Fast computation** - [Polars](https://pola.rs/) for internal data wrangling, [NumPy](https://numpy.org/) vectorization, [Numba](https://numba.pydata.org/) JIT compilation, and threaded parallel compute.
- **GPU acceleration** - Optional [CuPy](https://cupy.dev/)-accelerated estimation on NVIDIA GPUs, with multi-GPU scaling in distributed environments.
- **Native plots** - Built-in visualizations powered by [plotnine](https://plotnine.org/), returning standard `ggplot` objects you can customize with the full grammar of graphics.
- **Robust inference** - Analytical standard errors, bootstrap (weighted and multiplier), and simultaneous confidence bands.

For detailed documentation, including user guides and API reference, see [moderndid.readthedocs.io](https://moderndid.readthedocs.io/en/latest/).

## Installation

```bash
uv pip install moderndid        # Core estimators (did, drdid, didinter, didtriple)
uv pip install moderndid[all]   # All estimators, plots, numba, spark, dask (excludes gpu)
```

Extras are additive and build on the base install, so you always get the core estimators ([`att_gt`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.att_gt.html), [`drdid`](https://moderndid.readthedocs.io/en/latest/api/generated/drdid/moderndid.drdid.html), [`did_multiplegt`](https://moderndid.readthedocs.io/en/latest/api/generated/didinter/moderndid.did_multiplegt.html), [`ddd`](https://moderndid.readthedocs.io/en/latest/api/generated/didtriple/moderndid.ddd.html)) plus whatever extras you specify:

- **`didcont`** - Continuous treatment DiD ([`cont_did`](https://moderndid.readthedocs.io/en/latest/api/generated/didcont/moderndid.cont_did.html))
- **`didhonest`** - Sensitivity analysis ([`honest_did`](https://moderndid.readthedocs.io/en/latest/api/generated/honestdid/moderndid.honest_did.html))
- **`plots`** - Visualization
- **`numba`** - Faster bootstrap inference
- **`spark`** - Distributed estimation via PySpark
- **`dask`** - Distributed estimation via Dask
- **`gpu`** - GPU-accelerated estimation (requires CUDA)

```bash
uv pip install moderndid[didcont,plots]   # Combine multiple extras
uv pip install moderndid[gpu,spark]       # GPU + distributed
```

## Quick Start

This example uses county-level teen employment data to estimate the effect of minimum wage increases. States adopted higher minimum wages at different times (2004, 2006, or 2007), making this a staggered adoption design.

The [`att_gt()`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.att_gt.html#moderndid.att_gt) function estimates the average treatment effect for each group *g* (defined by when units were first treated) at each time period *t*. We use the doubly robust estimator, which combines outcome regression and propensity score weighting to provide consistent estimates if either model is correctly specified.

```python
import moderndid as did

data = did.load_mpdta()

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

__ModernDiD__ provides "batteries-included" plotting functions ([`plot_event_study`](https://moderndid.readthedocs.io/en/latest/api/generated/plotting/moderndid.plots.plot_event_study.html), [`plot_gt`](https://moderndid.readthedocs.io/en/latest/api/generated/plotting/moderndid.plots.plot_gt.html), [`plot_agg`](https://moderndid.readthedocs.io/en/latest/api/generated/plotting/moderndid.plots.plot_agg.html), and [more](https://moderndid.readthedocs.io/en/latest/api/plotting.html)) as well as data converters for building custom figures with [plotnine](https://plotnine.org/). Since all plot functions return `ggplot` objects, you can restyle them with the full grammar of graphics:

```python
from plotnine import element_text, labs, theme, theme_gray

p = did.plot_gt(attgt_result, ncol=3)
p = (p
    + labs(
        x="Year",
        y="ATT (Log Employment)",
        title="Minimum Wage Effects on Teen Employment",
        subtitle="Group-time average treatment effects by treatment cohort",
    )
    + theme_gray()
    + theme(
        legend_position="bottom",
        strip_text=element_text(size=11, weight="bold"),
    )
)
```

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/att.png" alt="ATT plot">

While group-time effects are useful, they can be difficult to summarize when there are many groups and time periods. The [`aggte`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.aggte.html) function aggregates these into more interpretable summaries. Setting `type="dynamic"` produces an event study that shows how effects evolve relative to treatment timing:

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

Event time 0 is the on-impact effect, negative event times are pre-treatment periods, and positive event times are post-treatment periods. Pre-treatment effects near zero support the parallel trends assumption, while post-treatment effects show how the impact evolves over time.

[Data converters](https://moderndid.readthedocs.io/en/latest/api/plotting.html#data-converters) make it easy to overlay estimates from different estimators. The figure below compares the Callaway and Sant'Anna estimates against a standard TWFE event study estimated with [pyfixest](https://github.com/py-econometrics/pyfixest). See the [Plotting Guide](https://moderndid.readthedocs.io/en/latest/user_guide/plotting.html) for the full code and more examples.

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/event_study.png" alt="CS (2021) vs TWFE event study comparison">

### Consistent API

All estimators share a unified interface:

```python
result = did.att_gt(data, yname="y", tname="t", idname="id", gname="g", ...)
result = did.ddd(data, yname="y", tname="t", idname="id", gname="g", pname="p", ...)
result = did.cont_did(data, yname="y", tname="t", idname="id", gname="g", dname="dose", ...)
result = did.drdid(data, yname="y", tname="t", idname="id", treatname="treat", ...)
result = did.did_multiplegt(data, yname="y", tname="t", idname="id", dname="treat", ...)
```

### Scaling Up

**Distributed** — Pass a Spark or Dask DataFrame and the distributed backend activates automatically. See the [Distributed guide](https://moderndid.readthedocs.io/en/latest/user_guide/distributed.html).

**GPU** — Pass `backend="cupy"` to offload estimation to NVIDIA GPUs. See the [GPU guide](https://moderndid.readthedocs.io/en/latest/user_guide/gpu.html) and [benchmarks](scripts/README.md).

```python
result = did.att_gt(data, yname="lemp", tname="year", idname="countyreal",
                    gname="first.treat", backend="cupy")
```

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
result = did.att_gt(data=spark.read.parquet("panel.parquet"),
                    yname="y", tname="t", idname="id", gname="g")
```

### Example Datasets

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

## Planned Development

- `moderndid.didml` — Machine learning approaches to DiD ([Hatamyar et al., 2023](https://arxiv.org/pdf/2310.11962))
- `moderndid.drdidweak` — Robust to weak overlap ([Ma et al., 2023](https://arxiv.org/pdf/2304.08974))
- `moderndid.didcomp` — Compositional changes in repeated cross-sections ([Sant'Anna & Xu, 2025](https://arxiv.org/pdf/2304.13925))
- `moderndid.didimpute` — Imputation-based estimators ([Borusyak, Jaravel, & Spiess, 2024](https://arxiv.org/pdf/2108.12419))
- `moderndid.didbacon` — Goodman-Bacon decomposition ([Goodman-Bacon, 2019](https://cdn.vanderbilt.edu/vu-my/wp-content/uploads/sites/2318/2019/07/29170757/ddtiming_7_29_2019.pdf))
- `moderndid.didlocal` — Local projections DiD ([Dube et al., 2025](https://www.nber.org/system/files/working_papers/w31184/w31184.pdf))
- `moderndid.did2s` — Two-stage DiD ([Gardner, 2021](https://jrgcmu.github.io/2sdd_current.pdf))
- `moderndid.etwfe` — Extended two-way fixed effects ([Wooldridge, 2021](https://ssrn.com/abstract=3906345); [Wooldridge, 2023](https://doi.org/10.1093/ectj/utad016))
- `moderndid.functional` — Specification tests ([Roth & Sant'Anna, 2023](https://arxiv.org/pdf/2010.04814))

## Acknowledgements

ModernDiD would not be possible without the researchers who developed the underlying econometric methods and implemented them in various R and Stata packages. See our [Acknowledgements](https://moderndid.readthedocs.io/en/latest/acknowledgements.html) page for a full list of the software, packages, and papers that have influenced this project.

## Citation

If you use ModernDiD in your research, please cite it as:

```bibtex
@software{moderndid,
  author  = {{The ModernDiD Authors}},
  title   = {{ModernDiD: Scalable, GPU-Accelerated Difference-in-Differences for Python}},
  year    = {2025},
  url     = {https://github.com/jordandeklerk/moderndid}
}
```
