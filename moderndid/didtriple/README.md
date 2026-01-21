# Triple Difference-in-Differences

This module provides a comprehensive implementation of triple difference-in-differences (DDD) estimators, also known as Difference-in-Differences-in-Differences. DDD designs are widely used in empirical work to relax parallel trends assumptions in Difference-in-Differences settings by leveraging a third dimension of variation.

The main parameters are **group-time average treatment effects** that account for both treatment group membership and eligibility status. The DDD framework allows treatment effects to be identified even when standard DiD parallel trends may fail, as long as such violations are stable across groups.

The computational methods here are inspired by the corresponding R package [triplediff](https://github.com/marcelortizv/triplediff) by Ortiz-Villavicencio and Sant'Anna.

> [!IMPORTANT]
> This module is designed for DDD applications where units must satisfy two criteria to be treated: belonging to a treatment group (e.g., a state that passes a policy) and being in an eligible partition (e.g., individuals eligible for a program). If you have a standard two-group DiD setting without eligibility variation, consider using the [did](https://github.com/jordandeklerk/moderndid/tree/readme/moderndid/did) module instead.

## Core Functionality

### 1. **Triple Difference-in-Differences** (`ddd`)

High-level wrapper that automatically detects whether data has two periods or multiple periods with staggered treatment adoption, calling the appropriate estimator.

### 2. **Aggregated Treatment Effects** (`agg_ddd`)

Aggregates the group-time ATTs into interpretable summary parameters:

- **Simple ATT**: Overall average treatment effect across all treated groups and post-treatment periods
- **Dynamic effects**: Event-study style estimates showing how effects evolve with treatment exposure
- **Group effects**: Average effects for units treated at the same time
- **Calendar time effects**: Average effects in each calendar period

## Features

### Supported Features

- Two-period DDD with single treatment date
- Multiple-period DDD with single treatment date
- DDD with staggered treatment adoption (variation in treatment timing)
- Aggregations (event study, group, calendar time)
- Panel data
- Repeated cross-sectional data
- Unbalanced panel data

### Unified High-Level API

The `ddd` function automatically detects whether data has two periods or multiple periods with staggered treatment adoption and routes to the appropriate estimator.

**Two-Period DDD** (all treated units receive treatment at the same time):

```python
from moderndid.didtriple import ddd, agg_ddd

result = ddd(
    data,
    yname='y',            # outcome variable
    tname='time',         # time period (e.g., 1=pre, 2=post)
    idname='id',          # unit identifier
    gname='state',        # treatment group indicator (0=control, 1=treated)
    pname='partition',    # eligibility indicator (0=ineligible, 1=eligible)
    xformla='~ x1 + x2',  # covariates formula (optional)
    est_method='dr',      # 'dr', 'ipw', or 'reg'
)
```

**Multi-Period DDD with Staggered Adoption** (units adopt treatment at different times):

```python
result = ddd(
    data,
    yname='y',
    tname='time',
    idname='id',
    gname='group',              # first period when treatment begins (0=never-treated)
    pname='partition',
    xformla='~ x1 + x2',
    control_group='nevertreated',  # or 'notyettreated'
    base_period='universal',       # or 'varying'
    est_method='dr',
)

# Aggregate group-time effects to event-study estimates
event_study = agg_ddd(result, aggregation_type='eventstudy')
```

### Flexible Estimation Methods

- **Doubly Robust (`dr`)**: Default method combining outcome regression and propensity score weighting
- **Inverse Propensity Score Weighting (`ipw`)**: Re-weights observations based on treatment propensity
- **Outcome Regression (`reg`)**: Regression-based estimation

### Control Group Options

- **Never Treated**: Units that never receive treatment throughout the sample period
- **Not Yet Treated**: Units that haven't been treated by time t which allows using future-treated units

### Robust Inference

- **Analytical Standard Errors**: Influence function-based standard errors
- **Multiplier Bootstrap**: Bootstrap inference accounting for estimation uncertainty
- **Simultaneous Confidence Bands**: For multiple hypothesis testing in event studies

## Usage

The following are simplified examples of how to use the `ddd()` function for both panel data and repeated cross-section data in the 2 period and multi-period, staggered adoption scenarios.

### Two-Period DDD with Covariates

We can generate synthetic data for a two-period DDD setup using the `gen_dgp_2periods` function. The data contains four subgroups based on treatment (`state`) and eligibility (`partition`) status:

```bash
   id  state  partition  time           y      cov1      cov2      cov3      cov4  cluster
0   1      0          1     1  429.253252  0.052044 -0.202120 -0.012154  0.862487       12
1   1      0          1     2  639.226001  0.052044 -0.202120 -0.012154  0.862487       12
2   2      1          1     1  331.807064 -0.891907 -0.408526 -0.916222 -0.090730        5
3   2      1          1     2  497.952145 -0.891907 -0.408526 -0.916222 -0.090730        5
4   3      0          1     1  425.298174  0.533484 -0.412515 -1.059306  1.064033        1
5   3      0          1     2  636.100873  0.533484 -0.412515 -1.059306  1.064033        1
```

Now we can estimate the average treatment effect on the treated using the `ddd` function:

```python
import moderndid as did

dgp = did.gen_dgp_2periods(n=5000, dgp_type=1, random_state=42)
df = dgp["data"]

result = did.ddd(
    data=df,
    yname="y",
    tname="time",
    idname="id",
    gname="state",
    pname="partition",
    xformla="~ cov1 + cov2 + cov3 + cov4",
    est_method="dr",
)
```

The output contains the DDD point estimate along with standard errors and confidence intervals:

```
==============================================================================
 Triple Difference-in-Differences (DDD) Estimation
==============================================================================

 DR-DDD estimation for the ATT:

       ATT      Std. Error    Pr(>|t|)    [95% Ptwise. Conf. Int.]
       0.0229       0.0828       0.7825    [ -0.1394,   0.1851]

------------------------------------------------------------------------------
 Signif. codes: '*' confidence interval does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Panel data: 2 periods

 No. of units at each subgroup:
   treated-and-eligible: 1235
   treated-but-ineligible: 1246
   eligible-but-untreated: 1291
   untreated-and-ineligible: 1228

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Outcome regression: OLS
 Propensity score: Logistic regression (MLE)

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 See Ortiz-Villavicencio and Sant'Anna (2025) for details.
```

### Multiple Periods DDD with Staggered Treatment Adoption

For settings with staggered treatment adoption, we can generate multi-period data using `gen_dgp_mult_periods`:

```python
dgp_mp = did.gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
data = dgp_mp["data"]
```

The data has treatment cohorts (`group`) that adopt treatment at different times:

```bash
   id  group  partition  time            y      cov1      cov2      cov3      cov4  cluster
0   1      2          1     1  1111.110519  0.052044  1.068661 -0.081955 -0.218837       14
1   1      2          1     2  1407.348064  0.052044  1.068661 -0.081955 -0.218837       14
2   1      2          1     3  1707.612707  0.052044  1.068661 -0.081955 -0.218837       14
3   2      3          1     1  1177.982431 -0.891907  1.221115  0.709174 -1.161969       17
4   2      3          1     2  1450.733689 -0.891907  1.221115  0.709174 -1.161969       17
5   2      3          1     3  1751.295911 -0.891907  1.221115  0.709174 -1.161969       17
```

Now we can estimate the group-time average treatment effects:

```python
result_mp = did.ddd(
    data=data,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    pname="partition",
    xformla="~ cov1 + cov2 + cov3 + cov4",
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
)
```

The output shows ATT estimates for each group-time combination:

```
==============================================================================
 Triple Difference-in-Differences (DDD) Estimation
 Multi-Period / Staggered Treatment Adoption
==============================================================================

 DR-DDD estimation for ATT(g,t):

   Group    Time       ATT(g,t)   Std. Error    [95% Conf. Int.]
       2       1       0.0000          NA              NA
       2       2      11.1769       0.4201    [10.3535, 12.0004] *
       2       3      21.1660       0.4516    [20.2808, 22.0511] *
       3       1      -1.0095       0.5450    [-2.0778,  0.0587]
       3       2       0.0000          NA              NA
       3       3      24.9440       0.4724    [24.0182, 25.8698] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence interval does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Control group: Never Treated
 Base period: universal
 Number of units: 500
 Time periods: 3 (1 to 3)
 Treatment cohorts: 2

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Outcome regression: OLS
 Propensity score: Logistic regression (MLE)

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 See Ortiz-Villavicencio and Sant'Anna (2025) for details.
```

#### Event Study Aggregation

We can aggregate the group-time effects into event-study estimates using `agg_ddd`:

```python
event_study = did.agg_ddd(result_mp, aggregation_type='eventstudy')
```

The output shows effects at each event time (time relative to treatment):

```
==============================================================================
 Aggregate DDD Treatment Effects (Event Study)
==============================================================================

 Overall summary of ATT's based on event-study aggregation:

       ATT      Std. Error     [95% Conf. Interval]
   20.1000          0.3345     [ 19.4444,  20.7557] *


 Dynamic Effects:

    Event time   Estimate   Std. Error   [95% Simult. Conf. Band]
            -2    -1.0095       0.5296   [-2.2638,  0.2448]
            -1     0.0000          nan   [    nan,     nan]
             0    19.0341       0.2502   [18.4415, 19.6266] *
             1    21.1660       0.4611   [20.0740, 22.2580] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0
==============================================================================
```

The column `Event time` shows effects relative to treatment adoption. `Event time=0` is the on-impact effect, and negative event times can be used as a pre-test for parallel trends.

#### Group Aggregation

To compute an overall ATT averaged across treatment cohorts:

```python
group_agg = did.agg_ddd(result_mp, aggregation_type='group')
```

```
==============================================================================
 Aggregate DDD Treatment Effects (Group/Cohort)
==============================================================================

 Overall summary of ATT's based on group/cohort aggregation:

       ATT      Std. Error     [95% Conf. Interval]
   21.1781          0.3637     [ 20.4652,  21.8910] *


 Group Effects:

         Group   Estimate   Std. Error   [95% Simult. Conf. Band]
             2    16.1715       0.3705   [15.3327, 17.0102] *
             3    24.9440       0.4821   [23.8526, 26.0354] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence band does not cover 0
==============================================================================
```

#### Using Not-Yet-Treated as Control Group

We can also use not-yet-treated units as the control group, which can improve efficiency:

```python
result_nyt = did.ddd(
    data=data,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    pname="partition",
    xformla="~ cov1 + cov2 + cov3 + cov4",
    control_group="notyettreated",
    base_period="universal",
    est_method="dr",
)
```

```
==============================================================================
 Triple Difference-in-Differences (DDD) Estimation
 Multi-Period / Staggered Treatment Adoption
==============================================================================

 DR-DDD estimation for ATT(g,t):

   Group    Time       ATT(g,t)   Std. Error    [95% Conf. Int.]
       2       1       0.0000          NA              NA
       2       2      10.6309       0.3189    [10.0058, 11.2559] *
       2       3      21.1660       0.4516    [20.2808, 22.0511] *
       3       1      -1.0095       0.5450    [-2.0778,  0.0587]
       3       2       0.0000          NA              NA
       3       3      24.9440       0.4724    [24.0182, 25.8698] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence interval does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Control group: Not Yet Treated
 Base period: universal
 Number of units: 500
 Time periods: 3 (1 to 3)
 Treatment cohorts: 2

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Outcome regression: OLS
 Propensity score: Logistic regression (MLE)

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 See Ortiz-Villavicencio and Sant'Anna (2025) for details.
```

Note that the standard error for ATT(2,2) is lower when using not-yet-treated controls (0.3189 vs 0.4201) since this leverages additional information from future-treated units.

### Repeated Cross-Section Data

The `ddd()` function also supports repeated cross-section data where different units are sampled in each time period. To use this, set `panel=False`:

```python
dgp = did.gen_dgp_2periods(n=5000, dgp_type=1, panel=False, random_state=42)
df = dgp["data"]
```

Unlike panel data where each unit is observed in both periods, repeated cross-section data has different individuals sampled at each time:

```bash
   id  state  partition  time           y      cov1      cov2      cov3      cov4  cluster
0   1      0          1     1  429.253252  0.052044 -0.202120 -0.012154  0.862487        1
1   2      1          1     1  331.807064 -0.891907 -0.408526 -0.916222 -0.090730       15
2   3      0          1     1  425.298174  0.533484 -0.412515 -1.059306  1.064033       20
3   4      1          1     1  486.619936  0.773799  0.222126 -0.320148  0.973417       26
4   5      1          1     1  292.084954 -1.252112  0.379305 -1.470672 -1.079863       14
5   6      1          1     1  275.877126 -1.012868 -2.308861 -0.146852 -0.740391       22
```

Now we can estimate the DDD ATT for repeated cross-section data by setting `panel=False`:

```python
result = did.ddd(
    data=df,
    yname="y",
    tname="time",
    idname="id",
    gname="state",
    pname="partition",
    xformla="~ cov1 + cov2 + cov3 + cov4",
    est_method="dr",
    panel=False,
)
```

The output is very similar to the panel data methods and shows that repeated cross-section estimation is being used:

```
==============================================================================
 Triple Difference-in-Differences (DDD) Estimation
 Repeated Cross-Section Data
==============================================================================

 DR-DDD estimation for the ATT:

       ATT      Std. Error    Pr(>|t|)    [95% Ptwise. Conf. Int.]
      -0.0494       0.1191       0.6780    [ -0.2829,   0.1840]

------------------------------------------------------------------------------
 Signif. codes: '*' confidence interval does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Repeated cross-section data: 2 periods

 No. of observations at each subgroup:
   treated-and-eligible: 2471
   treated-but-ineligible: 2522
   eligible-but-untreated: 2581
   untreated-and-ineligible: 2426

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Outcome regression: OLS (4 cell-specific models)
 Propensity score: Logistic regression (MLE)

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 See Ortiz-Villavicencio and Sant'Anna (2025) for details.
```

Note that the output now reports "No. of observations" rather than "No. of units" since individuals are not tracked across periods. The estimation method also fits 4 cell-specific outcome regression models (one for each subgroup) rather than a single pooled model used in panel data.

#### Multi-Period Repeated Cross-Section

Repeated cross-section estimation also works with staggered treatment adoption:

```python
dgp_mp = did.gen_dgp_mult_periods(n=500, dgp_type=1, panel=False, random_state=42)
data = dgp_mp["data"]

result_mp = did.ddd(
    data=data,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    pname="partition",
    xformla="~ cov1 + cov2 + cov3 + cov4",
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    panel=False,
)
```

The output again indicates that repeated cros-section data was used in the staggered adoption design:

```
==============================================================================
 Triple Difference-in-Differences (DDD) Estimation
 Multi-Period / Staggered Treatment Adoption (Repeated Cross-Section)
==============================================================================

 DR-DDD estimation for ATT(g,t):

   Group    Time       ATT(g,t)   Std. Error    [95% Conf. Int.]
       2       1       0.0000          NA              NA
       2       2      10.3679       0.5588    [ 9.2726, 11.4632] *
       2       3      19.4277       0.6178    [18.2169, 20.6386] *
       3       1       0.2872       0.5702    [-0.8304,  1.4048]
       3       2       0.0000          NA              NA
       3       3      24.1622       0.5488    [23.0865, 25.2379] *

------------------------------------------------------------------------------
 Signif. codes: '*' confidence interval does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Control group: Never Treated
 Base period: universal
 Number of observations: 1500
 Time periods: 3 (1 to 3)
 Treatment cohorts: 2

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Outcome regression: OLS (4 cell-specific models per comparison)
 Propensity score: Logistic regression (MLE)

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Significance level: 0.05
 Analytical standard errors
==============================================================================
 See Ortiz-Villavicencio and Sant'Anna (2025) for details.
```

All aggregation methods (`agg_ddd`) work the same way for repeated cross-section data as they do for panel data.

## References

Callaway, B., & Sant'Anna, P. H. (2021). *Difference-in-differences with multiple time periods.*
Journal of Econometrics, 225(2), 200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). *Better Understanding Triple Differences Estimators.*
arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942
