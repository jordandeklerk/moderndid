<div align="center">

<img alt="ModernDiD" src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/logo-wordmark.svg" width="300">

## Modern causal inference in Python

[![License](https://img.shields.io/badge/License-MIT-315bc4.svg)](https://github.com/jordandeklerk/moderndid/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/moderndid.svg?color=315bc4)](https://pypi.org/project/moderndid/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code coverage](https://codecov.io/gh/jordandeklerk/moderndid/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/moderndid)
[![Build status](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/moderndid/badge/?version=latest)](https://moderndid.readthedocs.io/en/latest/)

[What is ModernDiD](#what-is-moderndid) | [Installation](#installation) | [Estimation](#estimation) | [Aggregation and plots](#aggregation-and-plots) | [Scaling](#scaling) | [Documentation](https://moderndid.readthedocs.io/en/latest/)

</div>

## What is ModernDiD?

ModernDiD is an open-source Python library for difference-in-differences (DiD).
It brings estimators from modern econometric research and separate R and Stata
packages into one library, with a consistent API for applied researchers,
economists, and data scientists.

Supported methods include staggered and two-period DiD, triple differences,
continuous and intertemporal treatments, dynamic covariate balancing,
[machine learning DiD](https://moderndid.readthedocs.io/en/latest/api/didml.html),
extended two-way fixed effects, and sensitivity analysis. Nonparametric
instrumental-variable estimation is also included. The
[estimator overview](https://moderndid.readthedocs.io/en/latest/user_guide/estimator_overview.html)
describes the treatment designs and assumptions behind each method.

Pass pandas, Polars, or another Arrow-compatible DataFrame to an estimator, then
aggregate the effects and plot the results. Computation uses NumPy and Polars,
with optional Numba acceleration. Supported estimators also run on NVIDIA GPUs
and distributed Spark or Dask clusters. If you are new to DiD, start with the
[introduction](https://moderndid.readthedocs.io/en/latest/getting_started/causal_inference.html).

## Installation

ModernDiD requires Python 3.11 or later. Install the core estimators from PyPI:

```bash
uv pip install moderndid
```

Use `pip install` in place of `uv pip install` if you prefer pip. Optional
dependencies are selected with extras:

```bash
uv pip install "moderndid[plots]"          # Core estimators and plotting
uv pip install "moderndid[all]"            # All estimator extras, plots, Numba, and Dask
uv pip install "moderndid[didcont,plots]"   # Choose individual extras
uv pip install "moderndid[gpu,spark]"      # CUDA 12 GPU and Spark dependencies
```

The `all` extra includes `didml` and excludes `gpu` and `spark`. The
[installation guide](https://moderndid.readthedocs.io/en/latest/getting_started/installation.html)
covers every extra, GPU setup, and troubleshooting. To install the development
version from GitHub:

```bash
uv pip install "moderndid[all] @ git+https://github.com/jordandeklerk/moderndid.git"
```

## Estimation

This example uses county-level panel data from
[Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001)
to estimate the effect of minimum wage increases on teen employment. `att_gt`
estimates an average treatment effect for each treatment cohort and time period.

```python
import moderndid as did

data = did.load_mpdta()

result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    xformla="~1",
    est_method="dr",
    boot=True,
    random_state=123,
)

print(result)
```

Analytical standard errors and bootstrap inference are available, including
simultaneous confidence bands. Estimators share argument names such as `yname`,
`tname`, and `idname`; each design has its own treatment arguments. See the
[examples](https://moderndid.readthedocs.io/en/latest/examples/index.html)
for complete analyses and the
[dataset reference](https://moderndid.readthedocs.io/en/latest/api/data.html)
for data from published studies and simulation generators.

## Aggregation and plots

Aggregate the group-time effects into an event study, then plot effects by time
relative to treatment. Install the `plots` extra to run the plotting code.

```python
event_study = did.aggte(result, type="dynamic", random_state=123)

plot = did.plot_event_study(event_study)
plot.save("event_study.png", dpi=200, width=8, height=5)
```

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/readme-event-study.png" alt="Event-study treatment effect estimates with confidence intervals">

Plots return standard plotnine `ggplot` objects, so you can add labels, themes,
and other layers. Use `did.plot_gt(result, ncol=3)` to plot each treatment
cohort separately. The
[plotting guide](https://moderndid.readthedocs.io/en/latest/user_guide/plotting.html)
covers customization and plots for other estimators.

## Publication tables

Result objects integrate with
[maketables](https://py-econometrics.github.io/maketables/), installed separately
with `uv pip install maketables`. Pass a result to `ETable` to extract estimates,
standard errors, confidence intervals, and model metadata:

```python
import maketables as mt

tab = mt.ETable(
    [event_study],
    coef_fmt="b:.3f* \\n (se:.3f)",
    keep=[r"^Event "],
    model_stats=["N", "se_type"],
    caption="Dynamic Treatment Effects",
)
tab.make("tex")  # or "html", "docx", "typst"
```

The
[publication tables guide](https://moderndid.readthedocs.io/en/latest/user_guide/publication_tables.html)
covers comparisons across specifications and custom `MTable` layouts.

## Scaling

For staggered or triple DiD on larger panels, pass a Spark or Dask DataFrame
and estimation uses the distributed backend. The
[distributed guide](https://moderndid.readthedocs.io/en/latest/user_guide/distributed.html)
covers cluster setup and supported estimators.

On an NVIDIA GPU, install the `gpu` extra and select the CuPy backend:

```python
result_gpu = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    backend="cupy",
)
```

See the [GPU guide](https://moderndid.readthedocs.io/en/latest/user_guide/gpu.html)
for CUDA requirements and multi-GPU computation, and the
[benchmark scripts](https://github.com/jordandeklerk/moderndid/blob/main/scripts/README.md)
for performance comparisons.

## Documentation

- [Getting Started](https://moderndid.readthedocs.io/en/latest/getting_started/index.html): installation, background, and the quickstart.
- [User Guide](https://moderndid.readthedocs.io/en/latest/user_guide/index.html): estimator selection, panel data, plots, tables, and scaling.
- [Examples](https://moderndid.readthedocs.io/en/latest/examples/index.html): analyses for each treatment design.
- [API Reference](https://moderndid.readthedocs.io/en/latest/api/index.html): function signatures, parameters, and result objects.
- [Development](https://moderndid.readthedocs.io/en/latest/dev/index.html): contributing, architecture, and testing.
- [Release Notes](https://moderndid.readthedocs.io/en/latest/release/index.html): changes between versions.

## Acknowledgements

ModernDiD builds on methods and implementations developed by researchers across
the DiD literature. The
[acknowledgements](https://moderndid.readthedocs.io/en/latest/acknowledgements.html)
list the papers and R and Stata packages behind each estimator.
