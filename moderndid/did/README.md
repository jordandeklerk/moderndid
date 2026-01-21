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
from moderndid import att_gt, aggte, plot_att_gt, plot_event_study

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
indicates the first period in which a state is treated:

```bash
      year  countyreal      lpop      lemp  first.treat  treat
0     2003        8001  5.896761  8.461469         2007      1
1     2004        8001  5.896761  8.336870         2007      1
2     2005        8001  5.896761  8.340217         2007      1
3     2006        8001  5.896761  8.378161         2007      1
4     2007        8001  5.896761  8.487352         2007      1
```

We can compute group-time average treatment effects for a staggered adoption design. The output is an object of type
`MPResult` which is a container for the results:

```python
import moderndid as did

data = did.load_mpdta()

# Estimate group-time ATTs using outcome regression
attgt_result = did.att_gt(
     data=data,
     yname="lemp",
     tname="year",
     gname="first.treat",
     idname="countyreal",
     est_method="reg",
     bstrap=False
)
```

The output contains estimates of the group-time average treatment effects and their standard errors
along with other meta information:

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

We can also plot the results using `plot_att_gt()`:

```python
did.plot_att_gt(attgt_result)
```

![Group-Time Average Treatment Effects](/assets/att.png)

### Customizing Plots

All plotting functions in moderndid are built with [plotnine](https://plotnine.org/), a Python implementation of the grammar of graphics. This means you can customize any plot using standard plotnine syntax by adding layers, themes, and scales.

```python
from plotnine import labs, theme, theme_classic, scale_color_manual

# Customize with plotnine
custom_plot = (
    did.plot_att_gt(attgt_result)
    + theme_classic()  # Use a different theme
    + scale_color_manual(values={"Pre": "#2ecc71", "Post": "#9b59b6"})
    + labs(
        title="Effect of Minimum Wage on Teen Employment",
        x="Year",
        y="ATT Estimate"
    )
    + theme(figure_size=(10, 8))  # Adjust figure size
)

custom_plot.save("my_plot.png", dpi=300)
```

![Customized Plot](/assets/att_custom.png)

### Event Study

In the example above, it is relatively easy to directly interpret the group-time average treatment effects.
However, there are many cases where it is convenient to aggregate the group-time average treatment effects into
a small number of parameters. One main type of aggregation is into an event study.

We can make an event study by using the `aggte` function:

```python
event_study = did.aggte(attgt_result, type='dynamic')
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
      -0.0772       0.0214     [-0.1191, -0.0353] *


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

The column event time is for each group relative to when they first participate in the treatment. For example, `event time=0` corresponds
to the on impact effect, and `event time=-1` is the effect in the period before a unit becomes treated (checking that this is equal to 0 is
potentially useful as a pre-test).

We can also plot the event study with `plot_event_study()`:

```python
did.plot_event_study(event_study)
```

![Event Study](/assets/event.png)

### Overall Effect of Participating in the Treatment

The event study above reported an overall effect of participating in the treatment. This was computed by averaging the average effects computed at each length of exposure.

In many cases, a more general purpose overall treatment effect parameter is given by computing the average treatment effect for each group, and then averaging across groups. This sort of procedure provides an average treatment effect parameter with a very similar interpretation to the Average Treatment Effect on the Treated (ATT) in the two period and two group case.

To compute this overall average treatment effect parameter, where we're interested in the estimate for overall ATT, we can switch the type to `group`:

```python
overall_att = did.aggte(attgt_result, type='group')
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
      -0.0313       0.0089     [-0.0487, -0.0140] *


 Group Effects:

         Group   Estimate   Std. Error   [95% Simult. Conf. Band]
          2004    -0.0797       0.0268   [-0.1429, -0.0166] *
          2006    -0.0162       0.0121   [-0.0447,  0.0123]
          2007    -0.0286       0.0109   [-0.0542, -0.0029] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0

 Control Group: Never Treated
 Anticipation Periods: 0
 Estimation Method: Doubly Robust
==============================================================================
```

## References

Callaway, B., & Sant'Anna, P. H. (2021). *Difference-in-differences with multiple time periods.*
Journal of Econometrics, 225(2), 200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
