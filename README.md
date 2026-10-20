<div align="center">

<img alt="ModernDiD" src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_templates/landing-banner.svg" width="550">

## Modern Difference-in-Differences in Python

[![License](https://img.shields.io/badge/License-MIT-315bc4.svg)](https://github.com/jordandeklerk/moderndid/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/moderndid.svg?color=315bc4)](https://pypi.org/project/moderndid/)
[![Python versions](https://img.shields.io/pypi/pyversions/moderndid.svg?color=315bc4)](https://pypi.org/project/moderndid/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code coverage](https://codecov.io/gh/jordandeklerk/moderndid/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/moderndid)
[![Build status](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/moderndid/badge/?version=latest)](https://moderndid.readthedocs.io/en/latest/)

[**Install guide**](#installation)
| [**Features**](#features)
| [**Changelog**](https://moderndid.readthedocs.io/en/latest/release/index.html)
| [**Documentation**](https://moderndid.readthedocs.io/en/latest/)

</div>

## What is ModernDiD?

ModernDiD is a Python library for difference-in-differences (DiD), designed for
applied researchers, economists, and data scientists who estimate the effects of
policies and treatments.

A two-way fixed effects regression, long the default way to run DiD in applied settings,
[can give misleading answers](https://moderndid.readthedocs.io/en/latest/background/did.html#background-did-twfe)
when treatment effects differ across groups or change over time. ModernDiD
implements the estimators that recent research developed to avoid that problem.

Every estimator reports analytical or bootstrap standard errors for the effects
it estimates. Most can also cluster those errors by a variable such as state.
The staggered, triple-difference, continuous-treatment, and nonparametric IV
estimators can build simultaneous confidence bands too. Estimation runs in
parallel threads on one machine and supported estimators also run on NVIDIA
GPUs for large panels.

ModernDiD is under active development so expect sharp edges. Please help by trying it out,
[reporting bugs](https://github.com/jordandeklerk/moderndid/issues), and telling
us what you think. If you're new to DiD, the
[introduction](https://moderndid.readthedocs.io/en/latest/user_guide/causal_inference.html)
covers the ideas behind the methods before any code.

## Installation

ModernDiD requires Python 3.12 or later. In a project managed by
[uv](https://docs.astral.sh/uv/), add it as a dependency.

```bash
uv add moderndid
```

With pip, install it into your environment instead.

```bash
pip install moderndid
```

Extras add optional dependencies and work the same way with either tool.

```bash
uv add "moderndid[plots]"            # Core estimators and plotting
uv add "moderndid[all]"              # All estimator extras and plots
uv add "moderndid[didcont,plots]"    # Choose individual extras
uv add "moderndid[gpu]"              # CUDA 12 GPU dependencies
uv add "moderndid[spark]"            # PySpark for att_gt on Spark
```

The `all` extra includes `didml` and leaves out `gpu` and `spark`. The
[installation guide](https://moderndid.readthedocs.io/en/latest/user_guide/installation.html)
covers every extra, GPU setup, and troubleshooting. The development version
installs straight from GitHub.

```bash
uv add "moderndid[all] @ git+https://github.com/jordandeklerk/moderndid.git"
```

## Features

ModernDiD covers the main research designs in the modern DiD literature.

- [Staggered DiD](https://moderndid.readthedocs.io/en/latest/user_guide/example_staggered_did.html): `att_gt` and `aggte`, based on Callaway and Sant'Anna (2021)
- [Two-period DiD](https://moderndid.readthedocs.io/en/latest/api/drdid.html): `drdid`, `ipwdid`, and `ordid`, based on Sant'Anna and Zhao (2020)
- [Triple differences](https://moderndid.readthedocs.io/en/latest/user_guide/example_triple_did.html): `ddd` and `agg_ddd`, based on Ortiz-Villavicencio and Sant'Anna (2025)
- [Continuous treatment](https://moderndid.readthedocs.io/en/latest/user_guide/example_cont_did.html): `cont_did`, based on Callaway, Goodman-Bacon, and Sant'Anna (2025)
- [Intertemporal treatment](https://moderndid.readthedocs.io/en/latest/user_guide/example_inter_did.html): `did_multiplegt`, based on de Chaisemartin and D'Haultfoeuille (2024)
- [Dynamic covariate balancing](https://moderndid.readthedocs.io/en/latest/user_guide/example_dyn_balancing.html): `dyn_balancing`, based on Viviano and Bradic (2026)
- [Machine learning DiD](https://moderndid.readthedocs.io/en/latest/api/didml.html): `didml`, based on Hatamyar, Kreif, Rocha, and Huber (2023)
- [Extended TWFE](https://moderndid.readthedocs.io/en/latest/user_guide/example_etwfe.html): `etwfe` and `emfx`, based on Wooldridge (2023, 2025)
- [Sensitivity analysis](https://moderndid.readthedocs.io/en/latest/user_guide/example_honest_did.html): `honest_did`, based on Rambachan and Roth (2023)
- [Nonparametric IV](https://moderndid.readthedocs.io/en/latest/user_guide/example_npiv.html): `npiv`, based on Chen, Christensen, and Kankanala (2024)

Every estimator also comes with the tools the rest of an analysis needs.

- Estimators accept any
  [Arrow-compatible](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
  DataFrame, such as polars, pandas, pyarrow, or a DuckDB result.
- Inference covers analytical and bootstrap standard errors, plus clustering
  and simultaneous confidence bands where the estimator supports them.
- Plots return plotnine `ggplot` objects that take any further plotnine layer or
  theme.
- Results work directly with
  [maketables](https://py-econometrics.github.io/maketables/) to build LaTeX,
  HTML, Word, and Typst tables.
- Parallel threads speed up estimation on one machine.
- Supported estimators also run on NVIDIA GPUs through CuPy for large panels.
- `att_gt`
  [fits a panel held in a Spark DataFrame](https://moderndid.readthedocs.io/en/latest/user_guide/spark.html)
  on the Spark executors without collecting its rows.

## Quickstart

The example below estimates how state minimum wage increases that took effect in
different years affected teen employment. The
[staggered adoption example](https://moderndid.readthedocs.io/en/latest/user_guide/example_staggered_did.html)
works through the same question at a deeper level.

```python
import moderndid as did

# County teen employment and state minimum wage increases from 2003 to 2007
data = did.load_mpdta()

# An average effect for each treatment cohort and year
result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
)

# An event study by time relative to treatment, plotted with the plots extra
event_study = did.aggte(result, type="dynamic")
did.plot_event_study(event_study)
```

## Documentation

- [User Guide](https://moderndid.readthedocs.io/en/latest/user_guide/index.html): installation, a first analysis, data preparation, results, and computation.
- [Examples](https://moderndid.readthedocs.io/en/latest/examples/index.html): analyses for each treatment design.
- [API Reference](https://moderndid.readthedocs.io/en/latest/api/index.html): function signatures, parameters, and result objects.
- [Background](https://moderndid.readthedocs.io/en/latest/background/index.html): the assumptions, identification, and inference behind each estimator.
- [Comparisons with R](https://moderndid.readthedocs.io/en/latest/r_comparisons/index.html): each estimator's results beside those of the R package for the same method.
- [FAQ](https://moderndid.readthedocs.io/en/latest/faq.html): answers to common questions about data, estimators, and results.
- [Development](https://moderndid.readthedocs.io/en/latest/dev/index.html): setup, workflow, testing, releases, architecture, new estimators, debugging, and benchmarking.
- [Changelog](https://moderndid.readthedocs.io/en/latest/release/index.html): changes between versions.

## Acknowledgements

ModernDiD builds on methods and implementations developed by researchers across
the DiD literature. The
[acknowledgements](https://moderndid.readthedocs.io/en/latest/acknowledgements.html)
list the papers and software behind each estimator.
