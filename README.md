<div style="text-align: center;" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png">
  <img alt="moderndid logo" src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png" width="300">
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
</div>

__ModernDiD__ is a scalable, GPU-accelerated difference-in-differences library for Python. It consolidates modern DiD estimators from leading econometric research and various R and Stata packages into a single framework with a consistent API. Runs on a single machine, NVIDIA GPUs, and distributed Spark and Dask clusters.

## Features

- [Staggered DiD](https://moderndid.readthedocs.io/en/latest/api/multiperiod.html), [Doubly Robust DiD](https://moderndid.readthedocs.io/en/latest/api/drdid.html), [Continuous DiD](https://moderndid.readthedocs.io/en/latest/api/didcont.html), [Triple DiD](https://moderndid.readthedocs.io/en/latest/api/didtriple.html), [Intertemporal DiD](https://moderndid.readthedocs.io/en/latest/api/didinter.html), and [Honest DiD](https://moderndid.readthedocs.io/en/latest/api/honestdid.html).
- Works with any [Arrow-compatible](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) DataFrame ([polars](https://pola.rs/), [pandas](https://pandas.pydata.org/), [pyarrow](https://arrow.apache.org/docs/python/), [duckdb](https://duckdb.org/), and more) via [narwhals](https://narwhals-dev.github.io/narwhals/).
- Distributed computing with [Spark](https://spark.apache.org/) and [Dask](https://www.dask.org/) backends.
- [Polars](https://pola.rs/) for internal data wrangling, [NumPy](https://numpy.org/) vectorization, [Numba](https://numba.pydata.org/) JIT computations, and threaded parallel compute.
- Optional [CuPy](https://cupy.dev/) GPU acceleration with multi-GPU support in distributed mode.
- [plotnine](https://plotnine.org/)-based plots that return `ggplot` objects you can customize.
- Result objects plug directly into [maketables](https://py-econometrics.github.io/maketables/) for LaTeX, HTML, Word, and Typst tables.
- Analytical SEs, weighted and multiplier bootstrap, simultaneous confidence bands.

For detailed documentation, see [ModernDiD Documentation](https://moderndid.readthedocs.io/en/latest/).

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

See the [Installation guide](https://moderndid.readthedocs.io/en/latest/user_guide/installation.html) for troubleshooting and GPU-specific setup.

## Quick Start

Using county-level panel data from [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) to estimate the effect of minimum wage increases on teen employment:

```python
import moderndid as did
from plotnine import element_text, labs, theme, theme_gray

data = did.load_mpdta()

# Group-time ATTs
result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    xformla="~lpop",
    est_method="dr",
)

# Use grammar of graphics to customize plots
p = did.plot_gt(result, ncol=3)
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

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/att.png" alt="Group-time ATT estimates">

The [User Guide](https://moderndid.readthedocs.io/en/latest/user_guide/index.html) has tutorials for every estimator. See also the [Plotting Guide](https://moderndid.readthedocs.io/en/latest/user_guide/plotting.html).

### Consistent API

All estimators use the same naming conventions for core arguments:

```python
result = did.att_gt(data, yname="y", tname="t", idname="id", gname="g", ...)
result = did.ddd(data, yname="y", tname="t", idname="id", gname="g", pname="p", ...)
result = did.cont_did(data, yname="y", tname="t", idname="id", gname="g", dname="dose", ...)
result = did.drdid(data, yname="y", tname="t", idname="id", treatname="treat", ...)
result = did.did_multiplegt(data, yname="y", tname="t", idname="id", dname="treat", ...)
```

### Publication Tables

Result objects plug into [maketables](https://py-econometrics.github.io/maketables/). Pass them to `ETable` and estimates, SEs, CIs, and metadata are extracted automatically:

```python
import maketables as mt

# Aggregate results from earlier into an event study
event_study = did.aggte(result, type="dynamic")

tab = mt.ETable(
    [event_study],
    coef_fmt="b:.3f* \\n (se:.3f)",
    keep=[r"^Event "],
    model_stats=["N", "se_type"],
    caption="Dynamic Treatment Effects",
)
tab.make("tex")  # or "html", "docx", "typst"
```

See the [Publication Tables guide](https://moderndid.readthedocs.io/en/latest/user_guide/publication_tables.html) for `MTable` layouts and more examples.

### Scaling Up

Pass a Spark or Dask DataFrame and estimation distributes automatically. See the [Distributed guide](https://moderndid.readthedocs.io/en/latest/user_guide/distributed.html).

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
result = did.att_gt(data=spark.read.parquet("panel.parquet"),
                    yname="y",
                    tname="t",
                    idname="id",
                    gname="g")
```

For GPUs, pass `backend="cupy"`. See the [GPU guide](https://moderndid.readthedocs.io/en/latest/user_guide/gpu.html) and [benchmarks](scripts/README.md).

```python
result = did.att_gt(data,
                    yname="lemp",
                    tname="year",
                    idname="countyreal",
                    gname="first.treat",
                    backend="cupy")
```

### Example Datasets

Datasets from published studies and synthetic data generators for simulations:

```python
did.load_mpdta()           # County teen employment
did.load_nsw()             # NSW job training program
did.load_ehec()            # Medicaid expansion
did.load_engel()           # Household expenditure
did.load_favara_imbs()     # Bank lending
did.load_cai2016()         # Crop insurance

did.gen_did_scalable()     # Staggered DiD panel
did.gen_cont_did_data()    # Continuous treatment DiD
did.gen_ddd_2periods()     # Two-period triple DiD
did.gen_ddd_mult_periods() # Staggered triple DiD
did.gen_ddd_scalable()     # Large-scale triple DiD
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
