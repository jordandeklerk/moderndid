<img src="docs/source/_static/didpy-light.png#gh-light-mode-only" width="250" align="left" alt="didpy logo"></img>
<img src="docs/source/_static/didpy-dark.png#gh-dark-mode-only" width="250" align="left" alt="didpy logo"></img>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/jordandeklerk/didpy/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/didpy/actions/workflows/test.yml)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/didpy/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/didpy)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/didpy)](https://github.com/jordandeklerk/didpy/graphs/commit-activity)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/didpy)](https://github.com/jordandeklerk/didpy/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)


__didpy__ is a unified Python implementation of modern difference-in-differences (DiD) methodologies, bringing together the fragmented landscape of DiD estimators into a single, coherent framework. This package consolidates methods from leading econometric research and various R packages into one comprehensive Python library with a consistent API.

<br>

> [!WARNING]
> This package is currently in active development and subject to change.

## Sub-Modules for DiD Estimators

Each subpackage below is designed as a self-contained module with its own estimators, inference procedures, and visualization tools, while sharing common infrastructure for data handling and computation. This modular architecture allows researchers to use exactly the methods they need while benefiting from a unified interface and consistent design principles across all DiD approaches.

**Core DiD Estimators:**

- **[`didpy.did`](https://github.com/jordandeklerk/didpy/tree/main/didpy/did)**: Multiple time periods and variation in treatment timing (Callaway and Sant'Anna, 2021) with group-time effects and flexible aggregation schemes
- **[`didpy.drdid`](https://github.com/jordandeklerk/didpy/tree/main/didpy/drdid)**: Doubly robust difference-in-differences estimators (Sant'Anna and Zhao, 2020) for panel and repeated cross-section data with improved efficiency and robustness

**Advanced Methods:**

- **[`didpy.didcont`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didcont)**: Continuous treatment DiD for dose-response relationships and non-binary treatments
- **[`didpy.didinter`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didinter)**: Intertemporal Did for treatment effects where the treatment may be non-binary, non-absorbing, and the outcome may be affected by treatment lags
- **[`didpy.didml`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didml)**: Machine learning approaches to DiD combining causal inference with ML methods
- **[`didpy.didbacon`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didbacon)**: Goodman-Bacon decomposition to understand two-way fixed effects estimates as weighted averages of all possible 2x2 DiD comparisons
- **[`didpy.drdidweak`](https://github.com/jordandeklerk/didpy/tree/main/didpy/drdidweak)**: New class of doubly robust estimators for treatment effect estimands that is also robust against weak covariate overlap
- **[`didpy.didcomp`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didcomp)**: DiD setups with repeated cross-sectional data and potential compositional changes across time periods
- **[`didpy.didlocal`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didlocal)**: Local projections DiD to address possible biases arising from negative weighting
- **[`didpy.did2s`](https://github.com/jordandeklerk/didpy/tree/main/didpy/did2s)**: Two-stage DiD for estimating TWFE models while avoiding issues with staggered treatment adoption

**Diagnostic and Sensitivity Tools:**

- **[`didpy.didhonest`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didhonest)**: Sensitivity analysis for violations of parallel trends (Rambachan and Roth, 2023) with multiple restriction types
- **[`didpy.functional`](https://github.com/jordandeklerk/didpy/tree/main/didpy/functional)**: Specification tests for functional form assumptions in DiD models
