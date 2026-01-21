# Honest Difference-in-Differences

This module provides tools for **robust inference and sensitivity analysis** for difference-in-differences and event study designs. Rather than relying on the assumption of exactly parallel trends, this framework formalizes the intuition that pre-trends are informative about violations of parallel trends, allowing researchers to conduct **sensitivity analysis** and construct **robust confidence intervals** that remain valid under plausible violations of the parallel trends assumption.

The computational methods here are inspired by the corresponding R package [HonestDiD](https://github.com/asheshrambachan/HonestDiD) by Rambachan and Roth.

> [!IMPORTANT]
> This module is designed for **sensitivity analysis** of existing DiD estimates. You'll need event-study coefficients and their variance-covariance matrix from another estimation method (e.g., `did`, `drdid`, or traditional TWFE) before using `HonestDiD`.

## Background

The robust inference approach in [Rambachan and Roth](https://asheshrambachan.github.io/assets/files/hpt-draft.pdf) formalizes the intuition that pre-trends are informative about violations of parallel trends. They provide several ways to formalize what this means.

### Bounds on Relative Magnitudes

One approach is to impose that violations of parallel trends in the post-treatment period cannot be much larger than those in the pre-treatment period. This is formalized by bounding the post-treatment violation to be no more than $\bar{M}$ times larger than the maximum pre-treatment violation. For example:

- $\bar{M} = 1$: Post-treatment violations are no larger than the worst pre-treatment violation
- $\bar{M} = 2$: Post-treatment violations are at most twice the pre-treatment violations

### Smoothness Restrictions

Another approach is to restrict how much post-treatment violations can deviate from a linear extrapolation of the pre-trend. The paper imposes that the slope of the pre-trend can change by no more than $M$ across consecutive periods. Setting $M = 0$ imposes exactly linear counterfactual trends, while larger $M$ allows more non-linearity.

### Sensitivity Analysis

Given these restrictions, the Honest Did package provides confidence intervals that are guaranteed to have correct coverage when the restrictions are satisfied, accounting for estimation error in both treatment effects and pre-trends.

Researchers can report confidence intervals under different assumptions about the magnitude of post-treatment violations (different values of $\bar{M}$ or $M$) and identify "breakdown values" – the largest restriction for which effects remain significant.

## Features

### Unified High-Level API

The main entry point provides a consistent interface for working with event study objects:

```python

from moderndid import honest_did

# For event study objects (e.g., from moderndid.did.aggte)
result = honest_did(
    event_study,
    event_time=0,
    sensitivity_type="smoothness",  # 'relative_magnitude'
    grid_points=100,
    m_vec=[0, 0.01, 0.02],          # For smoothness
    m_bar_vec=[0.5, 1, 1.5, 2],     # For relative magnitudes
    alpha=0.05
)
```

### Direct API

For users with pre-computed event study coefficients:

```python
from moderndid import (
    create_sensitivity_results_sm,
    create_sensitivity_results_rm,
    construct_original_cs
)

# For smoothness restrictions
sensitivity_results = create_sensitivity_results_sm(
    betahat=event_study_coefs,
    sigma=vcov_matrix,
    num_pre_periods=5,
    num_post_periods=3,
    m_vec=[0, 0.01, 0.02, 0.03],
    l_vec=None,
    method="FLCI",
    alpha=0.05
)

# For relative magnitude restrictions
sensitivity_results_rm = create_sensitivity_results_rm(
    betahat=event_study_coefs,
    sigma=vcov_matrix,
    num_pre_periods=5,
    num_post_periods=3,
    m_bar_vec=[0.5, 1.0, 1.5, 2.0],
    method="C-LF",
    alpha=0.05
)
```

### Advanced Sensitivity Restrictions

Beyond the basic relative magnitudes and smoothness restrictions, the package offers additional restrictions for users to incorporate context-specific knowledge about confounding factors contributing to possible violations of the parallel trends assumption:

- **Relative Magnitudes with Sign Restrictions**: Incorporate knowledge about bias direction (positive/negative) alongside relative magnitude bounds
- **Relative Magnitudes with Monotonicity**: Add monotonicity constraints when treatment effects are expected to evolve smoothly (increasing/decreasing)
- **Second Differences with Sign Restrictions**: Combine smoothness constraints with bias direction
- **Second Differences with Monotonicity**: Enforce both smoothness and monotonic evolution of effects
- **Combined Smoothness and Relative Magnitudes**: Apply both types of restrictions simultaneously
- **Full Constraint Combinations**: Layer all three types of constraints for maximum robustness

All these options are available both at a lower-level API or through the high-level wrapper function `honest_did` with the required `kwargs` relevant to the specific restrictions.

### Multiple Confidence Interval Methods

- **Fixed-Length CI (`FLCI`)**: Default method with optimal length for **smoothness methods**
- **Andrew-Roth-Pakes (`ARP`)**: Data-driven CI construction
- **Conditional CI (`Conditional`)**: Conditions on non-negativity
- **C-LF Method (`C-LF`)**: Computationally efficient for **relative magnitudes**

### Flexible Parameter Analysis

- Analyze any linear combination of post-treatment effects via `l_vec` parameter
- Support for average effects, cumulative effects, or custom weighted combinations
- Fine-grained control over computational parameters (grid resolution, bounds, bootstrap settings)

### Visualizations

Built-in plotting functions using [plotnine](https://plotnine.org/):

```python
from moderndid import plot_sensitivity

# Plot sensitivity analysis results
plot_sensitivity(sensitivity_results)
```

Plots can be customized using standard plotnine syntax:

```python
from plotnine import labs, theme, theme_classic

custom_plot = (
    plot_sensitivity(sensitivity_results)
    + theme_classic()
    + labs(title="Sensitivity Analysis", y="Treatment Effect")
    + theme(figure_size=(8, 5))
)

custom_plot.save("sensitivity.png", dpi=300)
```

## Usage

We will examine the effects of Medicaid expansions on insurance coverage using publicly-available data derived from the ACS. We first load the data and packages relevant for the analysis.

```python
from moderndid import (
    load_ehec,
    create_sensitivity_results_sm,
    create_sensitivity_results_rm,
    construct_original_cs
)

df = load_ehec()
print(df.head())
```

```bash
##    stfips  year      dins     yexp2         W
## 0       1  2008  0.681218       NaN  613156.0
## 1       1  2009  0.658096       NaN  613156.0
## 2       1  2010  0.631473       NaN  613156.0
## 3       1  2011  0.655519       NaN  613156.0
## 4       1  2012  0.671467       NaN  613156.0
```

The data is a state-level panel with information on health insurance coverage and Medicaid expansion. The variable `dins` shows the share of low-income childless adults with health insurance in the state. The variable `yexp2` gives the year that a state expanded Medicaid coverage under the Affordable Care Act, and is missing if the state never expanded.

### Estimate the Baseline Event Study

For simplicity, we will first focus on assessing sensitivity to violations of parallel trends in a non-staggered DiD. We therefore restrict the sample to the years 2015 and earlier, and drop the small number of states who are first treated in 2015. We are now left with a panel dataset where some units are first treated in 2014 and the remaining units are not treated during the sample period.

```python
import pyfixest as pf

if df['year'].dtype.name == 'category':
    df['year'] = df['year'].astype(int)
if 'yexp2' in df.columns and df['yexp2'].dtype.name == 'category':
    df['yexp2'] = df['yexp2'].astype(float)
if df['stfips'].dtype.name == 'category':
    df['stfips'] = df['stfips'].astype(str)

df_nonstaggered = df[
    (df['year'] < 2016) &
    (df['yexp2'].isna() | (df['yexp2'] != 2015))
].copy()

df_nonstaggered['D'] = np.where(df_nonstaggered['yexp2'] == 2014, 1, 0)

years = sorted(df_nonstaggered['year'].unique())
for year in years:
    if year != 2013:
        df_nonstaggered[f'D_year_{year}'] = df_nonstaggered['D'] * (df_nonstaggered['year'] == year)

interaction_terms = [f'D_year_{year}' for year in years if year != 2013]
formula = f"dins ~ {' + '.join(interaction_terms)} | stfips + year"

twfe_results = pf.feols(formula, data=df_nonstaggered, vcov={'CRV1': 'stfips'})

pre_years = [2008, 2009, 2010, 2011, 2012]
post_years = [2014, 2015]

coef_names = [f'D_year_{year}' for year in pre_years + post_years]
betahat = np.array([twfe_results.coef()[name] for name in coef_names])

sigma_full = twfe_results._vcov

all_coef_names = list(twfe_results.coef().index)
coef_indices = [all_coef_names.index(name) for name in coef_names]

sigma = sigma_full[np.ix_(coef_indices, coef_indices)]

print(twfe_results.summary())
```

This gives us event study coefficients for years 2008-2012, 2014, and 2015 (2013 is the reference period):

```
| Coefficient                                 |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
|:--------------------------------------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
| C(year, contr.treatment(base=2013))[2008]:D |     -0.005 |        0.009 |    -0.611 |      0.545 | -0.023 |   0.012 |
| C(year, contr.treatment(base=2013))[2009]:D |     -0.011 |        0.009 |    -1.325 |      0.192 | -0.029 |   0.006 |
| C(year, contr.treatment(base=2013))[2010]:D |     -0.003 |        0.007 |    -0.376 |      0.708 | -0.017 |   0.012 |
| C(year, contr.treatment(base=2013))[2011]:D |     -0.001 |        0.006 |    -0.224 |      0.824 | -0.014 |   0.011 |
| C(year, contr.treatment(base=2013))[2012]:D |      0.000 |        0.007 |     0.046 |      0.964 | -0.015 |   0.015 |
| C(year, contr.treatment(base=2013))[2014]:D |      0.046 |        0.009 |     5.075 |      0.000 |  0.028 |   0.065 |
| C(year, contr.treatment(base=2013))[2015]:D |      0.069 |        0.010 |     6.687 |      0.000 |  0.048 |   0.090 |
```

![Event-Study](/assets/medicaid_event_study.png)

### Sensitivity Analysis Using Relative Magnitudes

We can now apply `HonestDiD` to do sensitivity analysis. Suppose we're interested in assessing the sensitivity of the estimate for 2014, the first year after treatment.

```python
num_pre_periods = 5
num_post_periods = 2

original_ci = construct_original_cs(
    betahat=betahat,
    sigma=sigma,
    num_pre_periods=num_pre_periods,
    num_post_periods=num_post_periods,
    alpha=0.05
)

delta_rm_results = create_sensitivity_results_rm(
    betahat=betahat,
    sigma=sigma,
    num_pre_periods=num_pre_periods,
    num_post_periods=num_post_periods,
    m_bar_vec=[0.5, 1.0, 1.5, 2.0],
    method="C-LF"
)
```

```bash
Original 95% CI for 2014 effect: [0.0285, 0.0644]

         lb        ub method    delta  Mbar
0  0.024130  0.066888   C-LF  DeltaRM   0.5
1  0.017094  0.071963   C-LF  DeltaRM   1.0
2  0.008587  0.079599   C-LF  DeltaRM   1.5
3 -0.000666  0.087946   C-LF  DeltaRM   2.0
```

The output shows a robust confidence interval for different values of $\bar{M}$. We see that the "breakdown value" for a significant effect is $\bar{M} = 2$, meaning that the significant result is robust to allowing for violations of parallel trends up to the same magnitude as the max violation in the pre-treatment period.

```python
from moderndid import plot_sensitivity

plot_sensitivity(delta_rm_results)
```

![Sensitivity-Analysis-Using-Relative-Magnitudes](/assets/medicaid_sensitivity_rm.png)

### Sensitivity Analysis Using Smoothness Restrictions

We can also do a sensitivity analysis based on smoothness restrictions, i.e., imposing that the slope of the difference in trends changes by no more than $M$ between periods.

```python
delta_sd_results = create_sensitivity_results_sm(
    betahat=betahat,
    sigma=sigma,
    num_pre_periods=num_pre_periods,
    num_post_periods=num_post_periods,
    m_vec=np.arange(0, 0.051, 0.01),
    method="FLCI"
)

print(delta_sd_results)
```

> [!CAUTION]
> Minor numerical differences may occur between the confidence interval bounds computed here and those from the R `HonestDiD` package when using smoothness restrictions. These differences arise from variations in the numerical optimization algorithms used for FLCI computation, but are typically negligible in practice.

```bash
         lb        ub method    delta     m
0  0.015038  0.049782   FLCI  DeltaSD  0.00
1  0.013431  0.078979   FLCI  DeltaSD  0.01
2  0.002810  0.090763   FLCI  DeltaSD  0.02
3 -0.007189  0.100762   FLCI  DeltaSD  0.03
4 -0.017189  0.110762   FLCI  DeltaSD  0.04
5 -0.027189  0.120762   FLCI  DeltaSD  0.05
```

We see that the breakdown value for a significant effect is $M \approx 0.03$, meaning that we can reject a null effect unless we are willing to allow for the linear extrapolation across consecutive periods to be off by more than 0.03 percentage points.

```python
plot_sensitivity(delta_sd_results)
```

![Sensitivity-Analysis-Using-Smoothness-Restrictions](/assets/medicaid_sensitivity_sd.png)

## Sensitivity Analysis for Average Effects

So far we have focused on the effect for the first post-treatment period, which is the default in `HonestDiD`. If we are instead interested in the average over the two post-treatment periods, we can use the option `l_vec = [0.5, 0.5]`:

```python
l_vec = np.array([0.5, 0.5])

original_ci_avg = construct_original_cs(
    betahat=betahat,
    sigma=sigma,
    num_pre_periods=num_pre_periods,
    num_post_periods=num_post_periods,
    l_vec=l_vec,
    alpha=0.05
)

delta_rm_results_avg = create_sensitivity_results_rm(
    betahat=betahat,
    sigma=sigma,
    num_pre_periods=num_pre_periods,
    num_post_periods=num_post_periods,
    m_bar_vec=[0, 0.5, 1.0, 1.5, 2.0],
    l_vec=l_vec,
    method="C-LF"
)

print(delta_rm_results_avg)
```

```bash
         lb        ub method    delta  Mbar
0  0.041244  0.074409   C-LF  DeltaRM   0.0
1  0.032511  0.079816   C-LF  DeltaRM   0.5
2  0.019767  0.090149   C-LF  DeltaRM   1.0
3  0.005824  0.103501   C-LF  DeltaRM   1.5
4 -0.008538  0.117249   C-LF  DeltaRM   2.0
```

```python
plot_sensitivity(delta_rm_results_avg)
```

![Sensitivity-Analysis-Average](/assets/medicaid_sensitivity_avg.png)

## Staggered Treatment Timing

The `honest_did` function can be used with any estimator that produces a vector of event study coefficients,
provided you are willing to impose relative magnitudes or smoothness restrictions that relate the bias of the
"post-treatment" estimates to the "pre-treatment" estimates.

Below, we show how the package can be used with modern methods for DiD with staggered treatment timing.

### Using HonestDiD with `att_gt` and `aggte`

We can combine staggered treatment DiD estimators of the Callaway and Sant'Anna type with Honest DiD
sensitivity analysis in a straight-forward way:

```python
from moderndid import att_gt, aggte, honest_did, load_ehec

df = load_ehec()

# Replace missing treatment times with a large number
df['yexp2'] = df['yexp2'].fillna(3000)

cs_results = att_gt(
    yname='dins',
    tname='year',
    idname='stfips',
    gname='yexp2',
    data=df,
    control_group='notyettreated'
)

es = aggte(cs_results, type='dynamic', min_e=-5, max_e=5)
```

This produces event study estimates with pre-treatment deviations below:

```
Dynamic Effects:

Event time   Estimate   Std. Error   [95% Simult. Conf. Band]
        -5    -0.0146       0.0127   [-0.0540,  0.0247]
        -4    -0.0196       0.0140   [-0.0629,  0.0236]
        -3    -0.0039       0.0177   [-0.0585,  0.0508]
        -2    -0.0197       0.0150   [-0.0660,  0.0265]
         0     0.0401       0.0065   [ 0.0201,  0.0601] *
         1     0.0545       0.0127   [ 0.0153,  0.0937] *
         2     0.0492       0.0075   [ 0.0259,  0.0724] *
         3     0.0855       0.0079   [ 0.0610,  0.1101] *
         4     0.0822       0.0102   [ 0.0508,  0.1137] *
         5     0.0803       0.0106   [ 0.0476,  0.1130] *
```

Now we can apply Honest DiD sensitivity analysis via relative magnitudes:

```python
# Immediate treatment effect (event_time=0)
sensitivity_results = honest_did(
    es,
    event_time=0,
    sensitivity_type='relative_magnitude',
    m_bar_vec=[0.5, 1.0, 1.5, 2.0]
)

print(sensitivity_results)
```

```bash
         lb        ub method    delta  Mbar
0  0.001160  0.073665   C-LF  DeltaRM   0.5
1 -0.033459  0.107886   C-LF  DeltaRM   1.0
2 -0.069379  0.143407   C-LF  DeltaRM   1.5
3 -0.102718  0.179631   C-LF  DeltaRM   2.0
```

These results show that the immediate treatment effect has a breakdown value of $\bar{M} = 1.0$. This means the significant positive effect of Medicaid expansion on insurance coverage becomes insignificant if we allow post-treatment violations of parallel trends to be as large as the maximum pre-treatment violation. The effect remains robust only under the more stringent assumption that post-treatment violations are at most half the size of pre-treatment violations.

![CS-Sensitivity-Analysis](/assets/cs_sensitivity_rm.png)

## References

Rambachan, A., & Roth, J. (2023). *A more credible approach to parallel trends.*
American Economic Review, 113(9), 2555-2591.

Sun, L., & Abraham, S. (2021). *Estimating dynamic treatment effects in event studies with heterogeneous treatment effects.*
Journal of Econometrics, 225(2), 175-199.
