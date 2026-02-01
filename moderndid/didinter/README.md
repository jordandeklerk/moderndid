# Difference-in-Differences for Intertemporal Treatment Effects

This module provides an implementation of difference-in-differences estimators for settings with **non-binary**, **non-absorbing** (time-varying) treatments, following [de Chaisemartin and D'Haultfoeuille (2024)](https://doi.org/10.1162/rest_a_01414). Unlike standard DiD which assumes binary absorbing treatment, this estimator handles complex treatment patterns where units can experience treatment increases, decreases, or multiple changes over time, and where lagged treatments may affect the outcome.

The main parameters are **dynamic treatment effects** that measure the actual-versus-status-quo (AVSQ) effect at each horizon. These effects compare the observed outcome to the counterfactual outcome a unit would have obtained if its treatment had remained at its baseline value.

The computational methods here are inspired by the corresponding Python package [py_did_multiplegt_dyn](https://github.com/Credible-Answers/py_did_multiplegt_dyn) and the Stata package [did_multiplegt_dyn](https://github.com/Credible-Answers/did_multiplegt_dyn) by de Chaisemartin and D'Haultfoeuille.

> [!IMPORTANT]
> This module is designed for settings where treatment can be non-binary (varying intensity) and non-absorbing (time-varying). If you have a standard binary, staggered adoption DiD setting, consider using the [did](https://github.com/jordandeklerk/moderndid/tree/readme/moderndid/did) module instead.

## Core Functionality

### 1. **Intertemporal Treatment Effects** (`did_multiplegt`)

Main function that estimates dynamic treatment effects by comparing the outcome evolution of switchers (units whose treatment changes) to that of non-switchers with the same baseline treatment.

### 2. **Event Study Visualization** (`plot_multiplegt`)

Built-in plotting that creates event study plots showing treatment effects at each horizon, with pre-treatment placebos displayed in a distinct color to assess parallel trends.

## Features

### Supported Features

- Non-binary treatment (varying intensity)
- Non-absorbing treatment (time-varying, can increase or decrease)
- Lagged treatment effects
- Multiple post-treatment horizons (event study effects)
- Pre-treatment placebo tests for parallel trends
- Normalized effects (dividing by cumulative treatment change)
- Test for equality of effects across horizons
- Heterogeneous effects analysis by covariates
- Clustered standard errors
- Sampling weights
- Control variables

### Treatment Effect Parameters

The estimator computes event-study style effects at each horizon:

- **Raw effects**: Average effect of having been exposed to changed treatment for a given number of periods
- **Normalized effects**: Effects divided by the average cumulative treatment change, interpretable as a weighted average of contemporaneous and lagged treatment effects
- **Average total effect**: Sum of effects across horizons

### Sample Selection Options

- **All switchers**: Include both treatment increases and decreases
- **Switchers-in only**: Only treatment increases
- **Switchers-out only**: Only treatment decreases
- **Same switchers**: Use identical set of switchers across all horizons for comparability

### Control Group Options

- **Not-yet-switchers** (default): Units that haven't switched by time t
- **Never-switchers only**: Only units that never switch throughout the sample

### Robust Inference

- **Clustered standard errors**: For panel data with clustering
- **Influence function-based inference**: Analytical standard errors
- **Multiplier bootstrap**: For finite-sample robustness
- **Joint placebo test**: Test that all pre-treatment effects are jointly zero

## Usage

The following examples demonstrate how to use the `did_multiplegt()` function using the Favara and Imbs (2015) banking deregulation dataset, where treatment (interstate branching) is non-binary and potentially non-absorbing.

### Loading the Data

```python
import moderndid as md

df = md.load_favara_imbs()
```

The data contains county-level observations of loan growth and interstate branching deregulation.

```bash
   year  county  state_n  Dl_vloans_b  inter_bra        w1    Dl_hpi
0  1994    1001        1     0.270248          0  0.975312  0.003176
1  1995    1001        1    -0.038427          0  0.975312  0.048912
2  1996    1001        1     0.161633          0  0.975312  0.058203
3  1997    1001        1     0.056523          0  0.975312  0.044366
4  1998    1001        1     0.034236          1  0.975312  0.047092
5  1999    1001        1     0.048719          1  0.975312  0.006280
```

### Event Study with Placebo Tests

Estimate a complete event study with pre-treatment placebo tests, normalized effects, and a test for equality of effects.

```python
result = md.did_multiplegt(
    data=df,
    yname="Dl_vloans_b",
    idname="county",
    tname="year",
    dname="inter_bra",
    effects=8,
    placebo=3,
    cluster="state_n",
    normalized=True,
    same_switchers=True,
    effects_equal=True,
)
```

The output shows the average total effect (sum of effects across horizons), treatment effects at each post-treatment horizon, and placebo effects (pre-treatment effects that should be near zero if parallel trends holds). The joint test for placebos is a chi-squared test that all placebo effects are jointly zero, where p-value = 0.2676 supports parallel trends. The test of equal effects tests whether all treatment effects are equal across horizons (requested via `effects_equal=True`). The `N` and `Switchers` columns show the number of observations and switching units used at each horizon.

```
==============================================================================
 Intertemporal Treatment Effects
==============================================================================

 Average Total Effect:

         ATE   Std. Error   [95% Conf. Interval]          N  Switchers
      0.0562       0.0169   [  0.0231,   0.0893] *    14745       6184


 Treatment Effects by Horizon:

   Horizon   Estimate   Std. Error   [95% Conf. Interval]          N  Switchers
         1     0.0240       0.0183   [ -0.0119,   0.0600]       3368        773
         2     0.0117       0.0106   [ -0.0092,   0.0325]       2604        773
         3     0.0143       0.0083   [ -0.0019,   0.0306]       2031        773
         4     0.0115       0.0082   [ -0.0045,   0.0275]       1541        773
         5     0.0162       0.0068   [  0.0028,   0.0295] *     1412        773
         6     0.0134       0.0062   [  0.0012,   0.0256] *     1293        773
         7     0.0098       0.0049   [  0.0002,   0.0195] *     1248        773
         8     0.0101       0.0043   [  0.0017,   0.0185] *     1248        773


 Placebo Effects (Pre-treatment):

   Horizon   Estimate   Std. Error   [95% Conf. Interval]          N  Switchers
        -1     0.0304       0.0251   [ -0.0188,   0.0796]       2335        764
        -2    -0.0218       0.0230   [ -0.0669,   0.0234]        957        497
        -3    -0.0137       0.0255   [ -0.0636,   0.0362]        511        358

 Joint test (placebos = 0): p-value = 0.2676

 Test of equal effects: p-value = 0.0760

------------------------------------------------------------------------------
 Signif. codes: '*' confidence interval does not cover 0

------------------------------------------------------------------------------
 Data Info
------------------------------------------------------------------------------
 Number of units: 1045
 Switchers: 916
 Never-switchers: 129

------------------------------------------------------------------------------
 Estimation Details
------------------------------------------------------------------------------
 Effects estimated: 8
 Placebos estimated: 3
 Normalized: Yes
 Same switchers across horizons: Yes

------------------------------------------------------------------------------
 Inference
------------------------------------------------------------------------------
 Confidence level: 95%
 Clustered standard errors: state_n
==============================================================================
 See de Chaisemartin and D'Haultfoeuille (2024) for details.
```

The placebo effects test whether switchers and non-switchers had similar outcome trends before switching (parallel trends assumption). The joint test p-value of 0.2676 indicates no significant pre-trends, supporting the identification assumptions.

We can visualize the event study with `plot_multiplegt()`:

```python
md.plot_multiplegt(result)
```

![Intertemporal Treatment Effects Event Study](/assets/didinter_event_study.png)

## References

de Chaisemartin, C., & D'Haultfoeuille, X. (2024). *Difference-in-Differences Estimators of Intertemporal Treatment Effects.*
Review of Economics and Statistics, 106(6), 1723-1736. https://doi.org/10.1162/rest_a_01414

Favara, G., & Imbs, J. (2015). *Credit Supply and the Price of Housing.*
American Economic Review, 105(3), 958-992. https://doi.org/10.1257/aer.20121416
