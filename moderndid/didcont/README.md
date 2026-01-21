# Difference-in-Differences with Continuous Treatments

The `moderndid.didcont` module extends difference-in-differences estimators to settings where treatment intensity is continuous ("dose") and adoption can be staggered across groups. It implements the estimators in [Callaway, Goodman-Bacon, and Sant'Anna (2024)](https://arxiv.org/abs/2107.02637).

> [!WARNING]
> The core estimators are implemented and tested, but APIs may still evolve and additional features remain under development. Feedback and bug reports are welcome.

## Capabilities

### Currently Supported
- Continuous, time-invariant treatment doses with staggered adoption
- Average treatment effects by dose (`ATT(d)`) and average causal response functions (`ACRT(d)`, derivatives of the dose-response)
- Aggregation across event time (event-study views) or across doses (dose-response views)
- Parametric dose-response estimation via B-splines with user control over degree and knots
- Nonparametric estimation using the Chen–Christensen–Kankanala (CCK) procedure for two-period settings
- Uniform confidence bands via multiplier bootstrap

### Not Yet Supported
- Discrete-only treatments (can be handled by other modules)
- Data-driven dose models with staggered adoption (CCK currently limited to two periods)
- Repeated cross-sections or unbalanced panels
- Time-varying doses or covariate adjustment
- Two-way clustering or user-supplied weights beyond basic diagnostics

## Features

The main entry point is `cont_did()`, which expects a balanced panel in long format. Key requirements:

1. **Dose values are time-invariant.** For units that never receive treatment, set the dose to its never-treated value (typically 0) in every period.
2. **Pre-period convention.** For treated units, set the dose equal to its eventual value even before adoption. This keeps the treatment definition consistent in every period.
3. **Grouping variable.** Provide `gname` with the first period of treatment (0 for never-treated). If omitted, it will be inferred from the dose.

Important arguments:

- `target_parameter`: `'level'` for ATT or `'slope'` for ACRT.
- `aggregation`: `'dose'` for dose-response views or `'eventstudy'` for aggregating across event time.
- `dose_est_method`: `'parametric'` (default B-splines) or `'cck'` for the nonparametric estimator (two-period settings only).
- `control_group`: `'notyettreated'` (default) or `'nevertreated'`.

The return value is a `DoseResult` when `aggregation='dose'` or a `PTEResult` when `aggregation='eventstudy'`. Both objects expose numpy arrays of estimates, standard errors, overall averages, and critical values, making it straightforward to build tables or custom plots.

## Usage

With a continuous treatment, the primitive objects are effects that are local to a timing group `g` in a time period `t` for a specific dose `d`. This collection of group-time-dose effects is high dimensional, so most applications aggregate them. We focus on two aggregation strategies: (i) averaging across timing groups and periods to recover dose-response functions `ATT(d)` or `ACRT(d)`, and (ii) averaging across doses while keeping event-time structure to obtain event-study paths.

For the examples below, we simulate data using `simulate_contdid_data` so that the continuous treatment has no true effect on outcomes; any estimated effects therefore reflect sampling variation.

```python
import moderndid as did
from tests.didcont.dgp import simulate_contdid_data

data = simulate_contdid_data(
    n=5000,
    num_time_periods=4,
    num_groups = 4,
    dose_linear_effect=0,
    dose_quadratic_effect = 0,
    seed=1234,
)
```

### Dose Aggregation

```python
# Dose-response
cd_res = did.cont_did(
    yname='Y',
    tname='time_period',
    idname='id',
    dname='D',
    gname='G',
    data=data,
    target_parameter='slope',
    aggregation='dose',
    treatment_type='continuous',
    control_group='notyettreated',
    biters=100,
    cband=True,
    num_knots=1,
    degree=3,
)

==============================================================================
 Continuous Treatment Dose-Response Results
==============================================================================

Overall ATT:
   ATT          Std. Error   [95% Conf. Interval]
   -0.0265      0.0646       [-0.1530,  0.1001]

Overall ACRT:
   ACRT         Std. Error   [95% Conf. Interval]
   0.1331       0.0596       [ 0.0162,  0.2499] *

---
Signif. codes: '*' confidence band does not cover 0


Control Group: Not Yet Treated
Anticipation Periods: 0
Spline Degree: 3
Number of Knots: 1
==============================================================================
```

```python
did.plot_dose_response(cd_res, effect_type='att')
```

![Dose-Response ATT](/assets/cont_dose_att.png)

```python
did.plot_dose_response(cd_res, effect_type='acrt')
```

![Dose-Response ACRT](/assets/cont_dose_acrt.png)

The first plot reports the estimated dose-response curve `ATT(d)` with simultaneous confidence bands. The second plot shows the derivative `ACRT(d)`, interpreted as the average causal response to a marginal increase in the dose.

### Event Study Aggregations

#### Event Study for ATT

We can also consider event study aggregations. The first is an event study aggregation for `ATT`. The second is an event study aggregation for `ACRT`:

```python
# Event Study
cd_res_es_level = did.cont_did(
  yname = "Y",
  tname = "time_period",
  idname = "id",
  dname = "D",
  data = data,
  gname = "G",
  target_parameter = "level",
  aggregation = "eventstudy",
  treatment_type = "continuous",
  control_group = "notyettreated",
  biters = 100,
  cband = TRUE,
  num_knots = 1,
  degree = 3,
)

==============================================================================
 Aggregate Treatment Effects (Event Study)
==============================================================================

Overall summary of ATT's:

   ATT          Std. Error   [95% Conf. Interval]
   -0.0191      0.0372       [-0.0919,  0.0538]



Event time Effects:

    Event time   Estimate   Std. Error  [95% Simult. Conf. Band]
            -2    -0.0297       0.0438  [ -0.1415,   0.0822]
            -1     0.0210       0.0411  [ -0.0839,   0.1259]
             0    -0.0069       0.0277  [ -0.0778,   0.0639]
             1     0.0009       0.0418  [ -0.1058,   0.1075]
             2    -0.0512       0.0488  [ -0.1757,   0.0733]

---
Signif. codes: '*' confidence band does not cover 0


Control Group: Not Yet Treated
Anticipation Periods: 0
Estimation Method: att
==============================================================================
```

```python
did.plot_event_study(cd_res_es_level)
```

![Event Study ATT](/assets/cont_event_study_att.png)

#### Event Study for ACRT

The event-study view averages across doses to show how effects evolve relative to time of treatment. The ATT plot targets level effects, while the ACRT plot focuses on how the marginal response changes over exposure length.

```python
cd_res_es_slope = did.cont_did(
  yname = "Y",
  tname = "time_period",
  idname = "id",
  dname = "D",
  data = data,
  gname = "G",
  target_parameter = "slope",
  aggregation = "eventstudy",
  treatment_type = "continuous",
  control_group = "notyettreated",
  biters = 100,
  cband = TRUE,
  num_knots = 1,
  degree = 3,
)

==============================================================================
 Aggregate Treatment Effects (Event Study)
==============================================================================

Overall summary of ACRT's:

   ACRT         Std. Error   [95% Conf. Interval]
   -0.1096      0.0550       [-0.2174, -0.0018] *



Event time Effects:

    Event time   Estimate   Std. Error  [95% Simult. Conf. Band]
            -2    -0.0681       0.0827  [ -0.2836,   0.1475]
            -1    -0.2213       0.0690  [ -0.4011,  -0.0416] *
             0     0.1581       0.0493  [  0.0296,   0.2866] *
             1     0.0540       0.0730  [ -0.1361,   0.2441]
             2    -0.5409       0.1058  [ -0.8166,  -0.2652] *

---
Signif. codes: '*' confidence band does not cover 0


Control Group: Not Yet Treated
Anticipation Periods: 0
Estimation Method: dose
==============================================================================
```

```python
did.plot_event_study(cd_res_es_slope)
```

![Event Study ACRT](/assets/cont_event_study_acrt.png)

### Customizing Plots

All plotting functions are built with [plotnine](https://plotnine.org/), a Python implementation of the grammar of graphics. You can customize any plot using standard plotnine syntax:

```python
from plotnine import labs, theme, theme_classic, scale_color_manual

custom_plot = (
    did.plot_dose_response(cd_res, effect_type='att')
    + theme_classic()
    + labs(
        title="Dose-Response Function",
        x="Treatment Dose",
        y="ATT(d)"
    )
    + theme(figure_size=(8, 5))
)

custom_plot.save("dose_response.png", dpi=300)
```

### Nonparametric Dose-Response (CCK)

When the dose-response surface is unknown, the parametric B-spline specification can be too restrictive. The alternative `dose_est_method='cck'` activates the nonparametric IV estimator of Chen, Christensen, and Kankanala (2024). We provide a self-contained implementation in `moderndid/didcont/npiv`, so no external dependencies are required. Consistent with the theory in Callaway, Goodman-Bacon, and Sant'Anna (2024), the current implementation supports two-period settings without staggered adoption; for longer panels, average the pre- and post-treatment periods to reduce to this case before estimation.

```python
data_cck = simulate_contdid_data(
    n=5000,
    num_time_periods=2,
    num_groups = 2,
    dose_linear_effect=0,
    dose_quadratic_effect = 1,
    seed=1234,
)

data_cck.loc[data_cck['G'] == 0, 'D'] = 0
```

```python
cd_res_cck = did.cont_did(
    yname="Y",
    tname="time_period",
    idname="id",
    dname="D",
    data=data_cck,
    gname="G",
    target_parameter="level",
    aggregation="dose",
    treatment_type="continuous",
    dose_est_method="cck",
    control_group="notyettreated",
    biters=100,
    cband=True,
)

==============================================================================
 Continuous Treatment Dose-Response Results
==============================================================================

Overall ATT:
   ATT          Std. Error   [95% Conf. Interval]
   0.3285       0.0443       [ 0.2416,  0.4153] *

Overall ACRT:
   ACRT         Std. Error   [95% Conf. Interval]
   1.0307       0.7691       [-0.4768,  2.5381]

---
Signif. codes: '*' confidence band does not cover 0


Control Group: Not Yet Treated
Anticipation Periods: 0
Spline Degree: 3
Number of Knots: 0
==============================================================================
```

Since plots are built with plotnine, you can overlay the true relationship for comparison:

```python
import numpy as np
from plotnine import geom_line, aes, theme
import polars as pl

# Create truth data (d^2 is the true DGP)
dose_grid = np.linspace(0.05, 1.0, 100)
truth_df = pl.DataFrame({"dose": dose_grid, "truth": dose_grid ** 2}).to_pandas()

# Plot with truth overlay
cck_plot = (
    did.plot_dose_response(cd_res_cck, effect_type='att')
    + geom_line(aes(x="dose", y="truth"), data=truth_df,
                linetype="dashed", color="#c0392b", size=1)
    + theme(figure_size=(8, 5))
)
```

![CCK Dose Response](/assets/cont_cck_truth.png)

## References

- Callaway, Brantly, Andrew Goodman-Bacon, and Pedro H. C. Sant'Anna. 2024. "Difference-in-Differences with a Continuous Treatment." *Journal of Econometrics* (forthcoming). [arXiv:2107.02637](https://arxiv.org/abs/2107.02637)
- Chen, Xiaohong, Timothy Christensen, and Sid Kankanala. 2024. "Adaptive Estimation and Uniform Confidence Bands for Nonparametric Structural Functions and Elasticities." *Review of Economic Studies* 92 (1): 162–96. [arXiv:2107.11869](https://arxiv.org/abs/2107.11869)
