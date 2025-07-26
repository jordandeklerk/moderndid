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
> This package is currently in active development with core estimators and some sensitivity analysis implemented. The API is subject to change.

## Difference-in-Differences Estimators

Each subpackage below is designed as a self-contained module with its own estimators, inference procedures, and visualization tools, while sharing common infrastructure for data handling and computation. This architecture choice aims to allow researchers to use exactly the methods they need while benefiting from a unified interface and consistent design principles across all DiD approaches.

**Core DiD Estimators**

- **[`didpy.did`](https://github.com/jordandeklerk/didpy/tree/main/didpy/did)**: Multiple time periods and variation in treatment timing with group-time effects and flexible aggregation schemes ([Callaway and Sant'Anna, 2021](https://arxiv.org/pdf/1803.09015))
- **[`didpy.drdid`](https://github.com/jordandeklerk/didpy/tree/main/didpy/drdid)**: Doubly robust difference-in-differences estimators for panel and repeated cross-section data with improved efficiency and robustness ([Sant'Anna and Zhao, 2020](https://arxiv.org/pdf/1812.01723))

**Advanced Methods**

- **[`didpy.didcont`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didcont)**: Continuous treatment DiD for dose-response relationships and non-binary treatments ([Callaway, Goodman-Bacon, and Sant'Anna, 2024](https://arxiv.org/pdf/2107.02637))
- **[`didpy.didinter`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didinter)**: Intertemporal DiD for treatment effects where the treatment may be non-binary, non-absorbing, and the outcome may be affected by treatment lags ([Chaisemartin and D’Haultfœuille, 2024](https://arxiv.org/pdf/2007.04267))
- **[`didpy.didml`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didml)**: Modern machine learning approaches to DiD for estimation of time-varying conditional average treatment effects on the treated (CATT) ([Hatamyar, Kreif, Rocha, and Huber, 2023](https://arxiv.org/pdf/2310.11962))
- **[`didpy.didbacon`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didbacon)**: Goodman-Bacon decomposition to understand two-way fixed effects estimates as weighted averages of all possible 2x2 DiD comparisons ([Goodman-Bacon, 2019](https://cdn.vanderbilt.edu/vu-my/wp-content/uploads/sites/2318/2019/07/29170757/ddtiming_7_29_2019.pdf))
- **[`didpy.drdidweak`](https://github.com/jordandeklerk/didpy/tree/main/didpy/drdidweak)**: New class of doubly robust estimators for treatment effect estimands that is also robust against weak covariate overlap ([Ma, Sant'Anna, Sasaki, and Ura, 2023](https://arxiv.org/pdf/2304.08974))
- **[`didpy.didcomp`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didcomp)**: DiD setups with repeated cross-sectional data and potential compositional changes across time periods ([Sant'Anna and  Xu, 2025](https://arxiv.org/pdf/2304.13925))
- **[`didpy.didlocal`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didlocal)**: Local projections DiD to address possible biases arising from negative weighting ([Dube, Girardi, Jorda, and Taylor, 2025](https://www.nber.org/system/files/working_papers/w31184/w31184.pdf))
- **[`didpy.did2s`](https://github.com/jordandeklerk/didpy/tree/main/didpy/did2s)**: Two-stage DiD for estimating TWFE models while avoiding issues with staggered treatment adoption ([Gardner, 2021](https://jrgcmu.github.io/2sdd_current.pdf))

**Diagnostic and Sensitivity Tools**

- **[`didpy.didhonest`](https://github.com/jordandeklerk/didpy/tree/main/didpy/didhonest)**: Sensitivity analysis for violations of parallel trends with multiple restriction types ([Rambachan and Roth, 2023](https://academic.oup.com/restud/article-abstract/90/5/2555/7039335?redirectedFrom=fulltext))
- **[`didpy.functional`](https://github.com/jordandeklerk/didpy/tree/main/didpy/functional)**: Specification tests for functional form assumptions in DiD models ([Roth and Sant'Anna, 2023](https://arxiv.org/pdf/2010.04814)
)
