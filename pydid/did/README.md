# Difference-in-Differences with Multiple Time Periods

This module provides a comprehensive implementation of difference-in-differences (DiD) estimators for settings with **multiple time periods** and **variation in treatment timing**. Unlike traditional two-period DiD, this framework accommodates staggered treatment adoption, where different units receive treatment at different times, and allows for heterogeneous and dynamic treatment effects across groups and time periods.

The main parameters are **group-time average treatment effects**. These are the average treatment effect for a particular group (group is defined by treatment timing) in a particular time period.

The computational methods here are inspired by the corresponding R package [did](https://github.com/bcallaway11/did) by Callaway and Sant'Anna.

> [!IMPORTANT]
> This module is designed for DiD applications with staggered treatment timing (where units adopt treatment at different times). If you have a simple two-period setting with all units treated at the same time, consider using the `drdid` module instead for more specialized estimators.

## Core Functionality

### 1. **Group-Time Average Treatment Effects** (`att_gt`)

The fundamental building block that estimates average treatment effects for each group (defined by treatment timing) at each time period.

### 2. **Aggregated Treatment Effects** (`aggte`)

Aggregates the numerous group-time ATTs into interpretable summary parameters:

- **Simple ATT**: Overall average treatment effect across all treated groups and post-treatment periods
- **Dynamic effects**: Event-study style estimates showing how effects evolve with treatment exposure
- **Group effects**: Average effects for units treated at the same time
- **Calendar time effects**: Average effects in each calendar period

## Features

### Unified High-Level API

The main entry point provides a pandas-friendly interface with sensible defaults:

```python
from pydid.did import att_gt, aggte

# Estimate group-time ATTs
att_results = att_gt(
    data,
    yname='outcome',               # outcome variable
    tname='time',                  # time variable
    gname='first_treat',           # first treatment period
    idname='id',                   # unit identifier
    xformla='~ age + income',      # covariates formula (optional)
    est_method='dr',               # estimation method
    control_group='nevertreated',  # comparison group
    anticipation=0,                # periods of anticipation
    allow_unbalanced_panel=True
)

# Aggregate to event-study estimates
event_study = aggte(att_results, type='dynamic')

# Aggregate to overall ATT
overall_att = aggte(att_results, type='simple')
```

### Flexible Estimation Methods

Multiple estimation strategies to accomodate different data structures and assumptions:

- **Doubly Robust (`dr`)**: Default method combining outcome regression and propensity score weighting
- **Inverse Propensity Score Weighting (`ipw`)**: Re-weights observations based on treatment propensity
- **Outcome Regression (`reg`)**: Two-way fixed effects regression (not recommended)

### Control Group Options

Flexible comparison group choices for identification:

- **Never Treated**: Units that never receive treatment throughout the sample period
- **Not Yet Treated**: Units that haven't been treated by time t (allows using future-treated units)

### Robust Inference

- **Multiplier Bootstrap**: Default inference method accounting for estimation uncertainty
- **Clustered Standard Errors**: For panel data with within-unit correlation
- **Simultaneous Confidence Bands**: For multiple hypothesis testing in event studies

### Treatment Anticipation

Accounts for potential anticipation effects where units may change behavior before actual treatment:

```python
# Allow for 2 periods of anticipation
att_results = att_gt(data, anticipation=2, ...)
```

## Usage

The dataset used in this example contains 500 observations of county-level teen employment rates from 2003-2007.
Some states are first treated in 2004, some in 2006, and some in 2007. The variable `first.treat`
indicates the first period in which a state is treated.

We can compute group-time average treatment effects for a staggered adoption design. The output is an object of type
`MPResult` which is a container for the results:

```python
import pydid
import pandas as pd
import numpy as np

data = pydid.datasets.load_mpdta()

# Estimate group-time ATTs
attgt_result = att_gt(
     data=df,
     yname="lemp",
     tname="year",
     gname="first.treat",
     idname="countyreal",
     est_method="dr",
     bstrap=False
)
```

The output contains estimates of the group-time average treatment effects and their standard errors
along with other meta information:

```
Reference: Callaway and Sant'Anna (2021)

Group-Time Average Treatment Effects:
  Group   Time   ATT(g,t)   Std. Error      [95% Simult.  Conf. Band]
   2004   2003    -0.0105       0.0241      [ -0.0766,   0.0556]
   2004   2004    -0.0704       0.0320      [ -0.1580,   0.0172]
   2004   2005    -0.1373       0.0389      [ -0.2437,  -0.0308] *
   2004   2006    -0.1008       0.0337      [ -0.1932,  -0.0084] *
   2006   2003     0.0065       0.0221      [ -0.0541,   0.0671]
   2006   2004    -0.0028       0.0188      [ -0.0542,   0.0487]
   2006   2005    -0.0046       0.0172      [ -0.0516,   0.0424]
   2006   2006    -0.0412       0.0194      [ -0.0943,   0.0119]
   2007   2003     0.0305       0.0152      [ -0.0111,   0.0721]
   2007   2004    -0.0027       0.0167      [ -0.0485,   0.0430]
   2007   2005    -0.0311       0.0186      [ -0.0822,   0.0200]
   2007   2006    -0.0261       0.0184      [ -0.0766,   0.0245]
---
Signif. codes: '*' confidence band does not cover 0

P-value for pre-test of parallel trends assumption:  0.0592

Control Group:  Never Treated,
Anticipation Periods:  0
Estimation Method:  Doubly Robust
```

### Event Study

In the example above, it is relatively easy to directly interpret the group-time average treatment effects.
However, there are many cases where it is convenient to aggregate the group-time average treatment effects into
a small number of parameters. One main type of aggregation is into an event study.

We can make an event study by using the `aggte` function:

```python
event_study =  aggte(att_gt_results, type='dynamic')
```

Just like for group-time average treatment effects, these can be summarized in a nice way:

```
==============================================================================
 Aggregate Treatment Effects (Event Study)
==============================================================================

 Call:
   aggte(MP, type='dynamic')

 Overall summary of ATT's based on event-study/dynamic aggregation:

   ATT          Std. Error     [95% Conf. Interval]
      -0.0772       0.0204     [-0.1171, -0.0374] *


 Dynamic Effects:

    Event time   Estimate   Std. Error   [95% Simult. Conf. Band]
            -3     0.0305       0.0151   [-0.0088,  0.0698]
            -2    -0.0006       0.0136   [-0.0361,  0.0349]
            -1    -0.0245       0.0144   [-0.0620,  0.0131]
             0    -0.0199       0.0120   [-0.0513,  0.0114]
             1    -0.0510       0.0166   [-0.0942, -0.0077] *
             2    -0.1373       0.0384   [-0.2375, -0.0370] *
             3    -0.1008       0.0346   [-0.1911, -0.0106] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0

 Control Group: Never Treated
 Anticipation Periods: 0
 Estimation Method: Doubly Robust
==============================================================================
```

### Overall Effect of Participating in the Treatment

The event study above reported an overall effect of participating in the treatment. This was computed by averaging the average effects computed at each length of exposure.

In many cases, a more general purpose overall treatment effect parameter is given by computing the average treatment effect for each group, and then averaging across groups. This sort of procedure provides an average treatment effect parameter with a very similar interpretation to the Average Treatment Effect on the Treated (ATT) in the two period and two group case.

To compute this overall average treatment effect parameter, where we're interested in the estimate for overall ATT, we can switich the type to `group`:

```python
overall_att =  aggte(att_gt_results, type='group')
```

The output shows that we estimate that increasing the minimum wage decreased teen employment by 3.1%,
and the effect is marginally statistically significant.

```
==============================================================================
 Aggregate Treatment Effects (Group/Cohort)
==============================================================================

 Call:
   aggte(MP, type='group')

 Overall summary of ATT's based on group/cohort aggregation:

   ATT          Std. Error     [95% Conf. Interval]
      -0.0310       0.0123     [-0.0551, -0.0069] *


 Group Effects:

         Group   Estimate   Std. Error   [95% Simult. Conf. Band]
          2004    -0.0797       0.0256   [-0.1378, -0.0217] *
          2006    -0.0229       0.0174   [-0.0623,  0.0165]
          2007    -0.0261       0.0166   [-0.0637,  0.0116]

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0

 Control Group: Never Treated
 Anticipation Periods: 0
 Estimation Method: Doubly Robust
==============================================================================
```

## References

Callaway, B., & Sant'Anna, P. H. (2021). *Difference-in-differences with multiple time periods.*
Journal of Econometrics, 225(2), 200-230.

Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
Journal of Econometrics, 219(1), 101-122.
