<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-light.png#gh-light-mode-only" width="250" align="left" alt="moderndid logo"></img>
<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/moderndid-dark.png#gh-dark-mode-only" width="250" align="left" alt="moderndid logo"></img>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/moderndid/actions/workflows/test.yml)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/moderndid/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/moderndid)
[![Documentation](https://readthedocs.org/projects/moderndid/badge/?version=latest)](https://moderndid.readthedocs.io/en/latest/)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/moderndid)](https://github.com/jordandeklerk/moderndid/graphs/commit-activity)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/moderndid)](https://github.com/jordandeklerk/moderndid/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)


__moderndid__ is a unified Python implementation of modern difference-in-differences (DiD) methodologies, bringing together the fragmented landscape of DiD estimators into a single, coherent framework. This package consolidates methods from leading econometric research and various R packages into one comprehensive Python library with a consistent API.

> [!WARNING]
> This package is currently in active development with core estimators and some sensitivity analysis implemented. The API is subject to change.

## Installation

```bash
uv pip install moderndid
```

Or install from source:

```bash
uv pip install git+https://github.com/jordandeklerk/moderndid.git
```

## Features

- **Multiple DiD estimators** — Staggered adoption, doubly robust, continuous treatment, triple difference, and sensitivity analysis
- **Fast computation** — [Polars](https://pola.rs/) for data wrangling, [NumPy](https://numpy.org/) vectorization, and [Numba](https://numba.pydata.org/) JIT compilation for performance-critical paths
- **Native plots** — Built on [plotnine](https://plotnine.org/) with full customization support
- **Robust inference** — Analytical standard errors, bootstrap (weighted and multiplier), and simultaneous confidence bands
- **Documentation** — [https://moderndid.readthedocs.io/en/latest/index.html](https://moderndid.readthedocs.io/en/latest/index.html)

### Consistent API

All estimators share a unified interface, making it easy to switch between methods:

```python
# Staggered DiD
result = did.att_gt(data, yname="y", tname="t", idname="id", gname="g", ...)

# Triple DiD
result = did.ddd(data, yname="y", tname="t", idname="id", gname="g", pname="p", ...)

# Continuous DiD
result = did.cont_did(data, yname="y", tname="t", idname="id", gname="g", dname="dose", ...)

# Doubly robust 2-period DiD
result = did.drdid(data, yname="y", tname="t", idname="id", treatname="treat", ...)
```

### Example Datasets

Several classic datasets from the DiD literature are included for learning and experimentation:

```python
did.load_mpdta()  # County teen employment (staggered adoption)
did.load_nsw()    # NSW job training program (2-period panel)
did.load_ehec()   # Medicaid expansion (sensitivity analysis)
did.load_engel()  # Household expenditure (continuous treatment)
```

## Quick Start

This example uses county-level teen employment data to estimate the effect of minimum wage increases. States adopted higher minimum wages at different times (2004, 2006, or 2007), making this a staggered adoption design.

```python
import moderndid as did

# County teen employment data
data = did.load_mpdta()

# Estimate group-time average treatment effects
result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    est_method="dr",
)
print(result)
```

The output shows treatment effects for each group (defined by when they were first treated) at each time period:

```
Reference: Callaway and Sant'Anna (2021)

Group-Time Average Treatment Effects:
  Group   Time   ATT(g,t)   Std. Error    [95% Simult.  Conf. Band]
   2004   2004    -0.0105       0.0232    [ -0.0743,   0.0533]
   2004   2005    -0.0704       0.0319    [ -0.1582,   0.0173]
   2004   2006    -0.1373       0.0356    [ -0.2352,  -0.0394] *
   2004   2007    -0.1008       0.0331    [ -0.1918,  -0.0098] *
   2006   2004     0.0065       0.0241    [ -0.0597,   0.0727]
   2006   2005    -0.0028       0.0202    [ -0.0582,   0.0527]
   2006   2006    -0.0046       0.0179    [ -0.0538,   0.0446]
   2006   2007    -0.0412       0.0211    [ -0.0992,   0.0167]
   2007   2004     0.0305       0.0145    [ -0.0095,   0.0705]
   2007   2005    -0.0027       0.0173    [ -0.0502,   0.0448]
   2007   2006    -0.0311       0.0190    [ -0.0833,   0.0211]
   2007   2007    -0.0261       0.0168    [ -0.0721,   0.0200]
---
Signif. codes: '*' confidence band does not cover 0

P-value for pre-test of parallel trends assumption:  0.1681

Control Group:  Never Treated,
Anticipation Periods:  0
Estimation Method:  Doubly Robust
```

These group-time effects can be aggregated into an event study to see how effects evolve relative to treatment:

```python
event_study = did.aggte(result, type="dynamic")
print(event_study)
```

```
==============================================================================
Aggregate Treatment Effects (Event Study)
==============================================================================

Call:
  aggte(MP, type='dynamic')

Overall summary of ATT's based on event-study/dynamic aggregation:

      ATT      Std. Error     [95% Conf. Interval]
  -0.0772          0.0214     [-0.1191, -0.0353] *


Dynamic Effects:

    Event time   Estimate   Std. Error   [95% Simult. Conf. Band]
            -3     0.0305       0.0151   [-0.0084,  0.0694]
            -2    -0.0006       0.0132   [-0.0346,  0.0335]
            -1    -0.0245       0.0139   [-0.0602,  0.0113]
             0    -0.0199       0.0120   [-0.0508,  0.0109]
             1    -0.0510       0.0172   [-0.0951, -0.0068] *
             2    -0.1373       0.0371   [-0.2326, -0.0419] *
             3    -0.1008       0.0352   [-0.1912, -0.0104] *

------------------------------------------------------------------------------
Signif. codes: '*' confidence band does not cover 0

Control Group: Never Treated
Anticipation Periods: 0
Estimation Method: Doubly Robust
==============================================================================
```

We can also use built-in plotting functionality to plot the event study results:

```python
did.plot_event_study(event_study)
```

<img src="https://raw.githubusercontent.com/jordandeklerk/moderndid/main/docs/source/_static/event_study.png" alt="Event study plot">

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

### Planned Development

| Module | Description | Reference |
|--------|-------------|-----------|
| `moderndid.didinter` | Intertemporal DiD with non-absorbing treatment | [Chaisemartin & D'Haultfœuille (2024)](https://arxiv.org/pdf/2007.04267) |
| `moderndid.didml` | Machine learning approaches to DiD | [Hatamyar et al. (2023)](https://arxiv.org/pdf/2310.11962) |
| `moderndid.drdidweak` | Robust to weak overlap | [Ma et al. (2023)](https://arxiv.org/pdf/2304.08974) |
| `moderndid.didcomp` | Compositional changes in repeated cross-sections | [Sant'Anna & Xu (2025)](https://arxiv.org/pdf/2304.13925) |
| `moderndid.didimpute` | Imputation-based estimators | [Borusyak, Jaravel, & Spiess (2024)](https://arxiv.org/pdf/2108.12419) |
| `moderndid.didbacon` | Goodman-Bacon decomposition | [Goodman-Bacon (2019)](https://cdn.vanderbilt.edu/vu-my/wp-content/uploads/sites/2318/2019/07/29170757/ddtiming_7_29_2019.pdf) |
| `moderndid.didlocal` | Local projections DiD | [Dube et al. (2025)](https://www.nber.org/system/files/working_papers/w31184/w31184.pdf) |
| `moderndid.did2s` | Two-stage DiD | [Gardner (2021)](https://jrgcmu.github.io/2sdd_current.pdf) |
| `moderndid.etwfe` | Extended two-way fixed effects | [Wooldridge (2021)](https://ssrn.com/abstract=3906345), [Wooldridge (2023)](https://doi.org/10.1093/ectj/utad016) |
| `moderndid.functional` | Specification tests | [Roth & Sant'Anna (2023)](https://arxiv.org/pdf/2010.04814) |
