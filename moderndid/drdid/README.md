# Doubly-Robust Difference-in-Differences

This module provides a comprehensive suite of modern difference-in-differences estimators for estimating the ATT (average treatment effect on the treated). We go beyond traditional DiD approaches by offering **doubly robust**, **inverse propensity weighted**, and **outcome regression estimators** that address common challenges in observational studies with two time periods (pre-treatment and post-treatment) and two groups (treatment group and comparison group).

The computational methods here are inspired by the corresponding R package [DRDID](https://github.com/pedrohcgs/drdid).

> [!CAUTION]
> The core estimators for this module are the **doubly robust estimators**. We recommend users utilize these estimators in practice as they will give the most robust estimate of the ATT. We include the other estimators mainly for researchers to compare estimates from more traditional DiD estimators.

## Core Functionality

### 1. **Doubly Robust DiD Estimators** (`drdid`)

Doubly robust DiD estimators for the ATT that are consistent when either a working (parametric) model for the propensity score or a working (parametric) model for the outcome evolution for the comparison group is correctly specified. We propose two different classes of DR DID estimators for the ATT that differ from each other depending on whether or not one models the outcome regression for the treated group in both pre and post-treatment periods

### 2. **Inverse Propensity Weighted DiD** (`ipwdid`)

IPW-based DiD estimators that re-weight observations to balance co-variate distributions. We include both Horwitz-Thompson type IPW estimators (weights are not normalized to sum up to 1) and Hajek-type IPW estimators (normalize weights within the treatment and control group).

### 3. **Outcome Regression DiD** (`ordid`)

Regression-based DiD estimators that model outcome evolutions directly.

## Features

### Unified High-Level API

Three main functions provide access to all DR-DiD estimators via the `est_method` with a consistent pandas-friendly interface

```python
from moderndid.drdid import drdid, ipwdid, ordid

# Doubly robust estimation
result = drdid(data, y_col='outcome', time_col='period', treat_col='treated',
               id_col='id', panel=True, covariates_formula='~ age + education + income',
               est_method='imp')

# IPW estimation
result = ipwdid(data, y_col='outcome', time_col='period', treat_col='treated',
               id_col='id', panel=True, covariates_formula='~ age + education + income',
               est_method='std_ipw')

# Outcome regression
result = ordid(data, y_col='outcome', time_col='period', treat_col='treated',
               id_col='id', panel=True, covariates_formula='~ age + education + income')
```

### Flexible Low-Level API

For advanced users, all underlying estimators are directly accessible with NumPy arrays as well

```python
from moderndid.drdid.estimators import drdid_imp_local_rc

# Doubly-Robust locally efficient and improved ATT estimate
result = drdid_imp_local_rc(
    y,
    post,
    d,
    covariates,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=999,
    influence_func=False,
    trim_level=0.995,
)
```

### Robust Inference Options

- **Bootstrap methods**: Weighted and multiplier bootstrap for all estimators
- **Analytical standard errors**: Via influence function calculations
- **Cluster-robust inference**: For panel data with repeated observations

### Advanced Propensity Score Methods

- **Inverse Probability Tilting (IPT)**: Alternative to logistic regression for better finite-sample properties
- **Automatic trimming**: Handles extreme propensity scores to ensure stable estimates
- **AIPW estimators**: Augmented inverse propensity weighted variants

## Usage

The following is a portion of the empirical illustration considered by Sant'Anna and Zhao (2020) that uses the LaLonde sample from the NSW experiment and considers data from the Current Population Survey (CPS) to form a non-experimental comparison group:

```python
import moderndid

# NSW dataset
nsw_data = moderndid.datasets.load_nsw()

# Estimate ATT using doubly robust DiD
att_result = moderndid.drdid(
    data=nsw_data,
    y_col='re',
    time_col='year',
    treat_col='experimental',
    id_col='id',
    panel=True,
    covariates_formula="~ age + educ + black + married + nodegree + hisp + re74",
    est_method='imp',
)
```

The output shows all of the relevant quantities for the estimated ATT and information about the method type, e.g., panel or repeated cross-section data, outcome and propensity models, and inference type:

```bash
=======================================================================
 Doubly Robust DiD Estimator (Improved Method)
=======================================================================
 Computed from 32834 observations and 12 covariates.

       Estimate  Std. Error  t-value  Pr(>|t|)     [95% Conf. Interval]
-----------------------------------------------------------------------
ATT   -901.2703    393.6212  -2.2897    0.0220  [-1672.7679, -129.7727]

-----------------------------------------------------------------------
 Method Details:
   Data structure: Panel data
   Outcome regression: Weighted least squares
   Propensity score: Inverse probability tilting

 Inference:
   Standard errors: Analytical
   Propensity score trimming: 0.995
=======================================================================
 Reference: Sant'Anna and Zhao (2020), Journal of Econometrics
 ```

## References

Abadie, A. (2005). *Semiparametric difference-in-differences estimators.*
The Review of Economic Studies, 72(1), 1-19.

Graham, B. S., Pinto, C. C., & Egel, D. (2012). *Inverse probability tilting for moment condition models with missing data.*
The Review of Economic Studies, 79(3), 1053-1079.

Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
Journal of Econometrics, 219(1), 101-122.
