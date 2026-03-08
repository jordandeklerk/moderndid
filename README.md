<div style="text-align: center;" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png">
  <img alt="moderndid logo" src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png" width="350">
</picture>

<p>
  <em>A scalable, GPU-accelerated difference-in-differences library for Python.</em>
</p>

<p>
  <a href="https://moderndid.readthedocs.io/en/latest/" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://moderndid.readthedocs.io/en/latest/api/index.html" target="_blank"><strong>API Reference</strong></a> ·
  <a href="https://moderndid.readthedocs.io/en/latest/user_guide/index.html" target="_blank"><strong>Tutorials</strong></a> ·
  <a href="https://github.com/jordandeklerk/moderndid/blob/main/CHANGELOG.md" target="_blank"><strong>Changelog</strong></a>
</p>

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
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/moderndid?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/moderndid)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/moderndid?period=monthly&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads/month)](https://pepy.tech/projects/moderndid)
[![Python version](https://img.shields.io/badge/3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Citation](https://img.shields.io/badge/Cite%20as-ModernDiD-blue)](#citation)
<!-- [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) -->
</div>

__ModernDiD__ is a scalable, GPU-accelerated difference-in-differences library for Python. It consolidates modern DiD estimators from leading econometric research and various R and Stata packages into a single framework with a consistent API. Runs on a single machine, NVIDIA GPUs, and distributed Spark and Dask clusters.

## Features

- **DiD Estimators** - [Staggered DiD](https://moderndid.readthedocs.io/en/latest/api/multiperiod.html), [Doubly Robust DiD](https://moderndid.readthedocs.io/en/latest/api/drdid.html), [Continuous DiD](https://moderndid.readthedocs.io/en/latest/api/didcont.html), [Triple DiD](https://moderndid.readthedocs.io/en/latest/api/didtriple.html), [Intertemporal DiD](https://moderndid.readthedocs.io/en/latest/api/didinter.html), [Honest DiD](https://moderndid.readthedocs.io/en/latest/api/honestdid.html).
- **Dataframe agnostic** - Pass any [Arrow-compatible](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) DataFrame such as [polars](https://pola.rs/), [pandas](https://pandas.pydata.org/), [pyarrow](https://arrow.apache.org/docs/python/), [duckdb](https://duckdb.org/), and more powered by [narwhals](https://narwhals-dev.github.io/narwhals/).
- **Distributed computing** - Scale to billions of observations across [Spark](https://spark.apache.org/) and [Dask](https://www.dask.org/) clusters. Pass a distributed DataFrame and the backend activates transparently.
- **Fast computation** - [Polars](https://pola.rs/) for internal data wrangling, [NumPy](https://numpy.org/) vectorization, [Numba](https://numba.pydata.org/) JIT compilation, and threaded parallel compute.
- **GPU acceleration** - Optional [CuPy](https://cupy.dev/)-accelerated estimation on NVIDIA GPUs, with multi-GPU scaling in distributed environments.
- **Native plots** - Built-in visualizations powered by [plotnine](https://plotnine.org/), returning standard `ggplot` objects you can customize with the full grammar of graphics.
- **Publication tables** - Pass any estimator output directly to [maketables](https://py-econometrics.github.io/maketables/) for publication-ready LaTeX, HTML, Word, and Typst tables with no custom extractors.
- **Robust inference** - Analytical standard errors, bootstrap (weighted and multiplier), and simultaneous confidence bands.

For detailed documentation, including user guides and API reference, see [ModernDiD Documentation](https://moderndid.readthedocs.io/en/latest/).

## Installation

```bash
uv pip install moderndid   # Core estimators (did, drdid, didinter, didtriple)
```

Some estimators and features require additional dependencies that are not installed by default. Extras are additive and build on the base install, so you always get the core estimators ([`att_gt()`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.att_gt.html), [`drdid()`](https://moderndid.readthedocs.io/en/latest/api/generated/drdid/moderndid.drdid.html), [`did_multiplegt()`](https://moderndid.readthedocs.io/en/latest/api/generated/didinter/moderndid.did_multiplegt.html), [`ddd()`](https://moderndid.readthedocs.io/en/latest/api/generated/didtriple/moderndid.ddd.html)) plus whatever extras you specify:

- **`didcont`** - Continuous treatment DiD ([`cont_did()`](https://moderndid.readthedocs.io/en/latest/api/generated/didcont/moderndid.cont_did.html))
- **`didhonest`** - Sensitivity analysis ([`honest_did()`](https://moderndid.readthedocs.io/en/latest/api/generated/honestdid/moderndid.honest_did.html))
- **`plots`** - Batteries-included plots
- **`numba`** - Faster bootstrap inference
- **`spark`** - Distributed estimation via PySpark
- **`dask`** - Distributed estimation via Dask
- **`gpu`** - GPU-accelerated estimation (requires CUDA)

```bash
uv pip install "moderndid[all]"             # All extras except gpu
uv pip install "moderndid[didcont,plots]"   # Combine specific extras
uv pip install "moderndid[gpu,spark]"       # GPU + distributed
```

To install the latest development version directly from GitHub:

```bash
uv pip install "moderndid[all] @ git+https://github.com/jordandeklerk/moderndid.git"
```

> [!TIP]
> When a package manager like `uv` or `pip` cannot resolve a dependency required by an extra, it may silently fall back to an older version of __ModernDiD__ where that extra does not exist, rather than raising an error.
>
> The `gpu` extra is the most likely to trigger this, since it depends on `cupy-cuda12x` (Linux and Windows only) and `rmm-cu12` (Linux only), both of which require NVIDIA CUDA. If you see a warning like `The package moderndid==0.0.3 does not have an extra named 'gpu'`, this is what happened. To use the `gpu` extra, install on a machine with NVIDIA CUDA drivers, or pin the version to get a clear error instead of a silent downgrade.
>
> ```bash
> uv pip install "moderndid[gpu]>=0.1.0"
> ```

## Quick Start

```python
import moderndid as did

data = did.load_mpdta()

# Group-time ATTs (Callaway & Sant'Anna, 2021)
result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    xformla="~lpop",
    est_method="dr",
)

# Aggregate into an event study
agg = did.aggte(result, type="dynamic")

# Built-in plots return ggplot objects you can customize
did.plot_gt(result, ncol=3)
```

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/att.png" alt="Group-time ATT estimates">

See the [User Guide](https://moderndid.readthedocs.io/en/latest/user_guide/index.html) for complete tutorials covering all estimators.

### Plotting

Built-in [plotting functions](https://moderndid.readthedocs.io/en/latest/api/plotting.html) return `ggplot` objects you can customize with the full [plotnine](https://plotnine.org/) grammar of graphics. The [`to_df()`](https://moderndid.readthedocs.io/en/latest/api/generated/plotting/moderndid.to_df.html) converter makes it easy to overlay estimates from different estimators.

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/event_study.png" alt="CS (2021) vs TWFE event study comparison">

See the [Plotting Guide](https://moderndid.readthedocs.io/en/latest/user_guide/plotting.html) for event studies, custom overlays, and more examples.

### Publication Tables

__ModernDiD__ result objects implement the [maketables](https://py-econometrics.github.io/maketables/) plug-in interface, so you can pass them directly to `maketables.ETable` for publication-ready LaTeX, HTML, Word, or Typst tables with no custom extractors. You will need to install maketables separately (`uv pip install maketables`).

For more complex layouts, `MTable` gives full control over row grouping, column spanners, and cell formatting. The table below was built from [`att_gt()`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.att_gt.html#moderndid.att_gt) and [`aggte()`](https://moderndid.readthedocs.io/en/latest/api/generated/multiperiod/moderndid.aggte.html) results, showing ATT estimates under unconditional and conditional parallel trends across all four aggregation types.

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/maketables_readme_panel_summary.png" alt="Multi-panel summary table recreating Callaway and Sant'Anna (2021) Table 3">

See the [Publication Tables guide](https://moderndid.readthedocs.io/en/latest/user_guide/publication_tables.html) for full examples, `ETable` customization, custom `MTable` layouts, and output format options.

### Consistent API

All estimators share a unified interface for core arguments. Pass any [Arrow PyCapsule](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)-compatible DataFrame ([polars](https://pola.rs/), [pandas](https://pandas.pydata.org/), [pyarrow](https://arrow.apache.org/docs/python/), [duckdb](https://duckdb.org/), and others) and estimation works the same way:

```python
result = did.att_gt(data, yname="y", tname="t", idname="id", gname="g", ...)
result = did.ddd(data, yname="y", tname="t", idname="id", gname="g", pname="p", ...)
result = did.cont_did(data, yname="y", tname="t", idname="id", gname="g", dname="dose", ...)
result = did.drdid(data, yname="y", tname="t", idname="id", treatname="treat", ...)
result = did.did_multiplegt(data, yname="y", tname="t", idname="id", dname="treat", ...)
```

### Scaling Up

**Distributed Computing.** Pass a Spark or Dask DataFrame and the distributed backend activates automatically. See the [Distributed guide](https://moderndid.readthedocs.io/en/latest/user_guide/distributed.html).

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
result = did.att_gt(data=spark.read.parquet("panel.parquet"), # Spark dataframe
                    yname="y",
                    tname="t",
                    idname="id",
                    gname="g")
```

**GPU Acceleration.** Pass `backend="cupy"` to offload estimation to NVIDIA GPUs. See the [GPU guide](https://moderndid.readthedocs.io/en/latest/user_guide/gpu.html) and [benchmarks](scripts/README.md).

```python
result = did.att_gt(data,
                    yname="lemp",
                    tname="year",
                    idname="countyreal",
                    gname="first.treat",
                    backend="cupy") # GPU backend
```

### Example Datasets

Built-in datasets from published studies are included for testing and reproducing results. All loaders return Arrow-compatible DataFrames that work directly with any estimator.

```python
did.load_mpdta()       # County teen employment
did.load_nsw()         # NSW job training program
did.load_ehec()        # Medicaid expansion
did.load_engel()       # Household expenditure
did.load_favara_imbs() # Bank lending
did.load_cai2016()     # Crop insurance
```

Synthetic data generators are also available for simulations and benchmarking.

```python
did.gen_did_scalable()           # Staggered DiD panel
did.gen_cont_did_data()          # Continuous treatment DiD
did.gen_ddd_2periods()           # Two-period triple DiD
did.gen_ddd_mult_periods()       # Staggered triple DiD
did.gen_ddd_scalable()           # Large-scale triple DiD
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
