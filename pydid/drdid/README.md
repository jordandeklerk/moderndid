# Doubly-Robust Difference-in-Differences

The `pydid.drdid` module provides a comprehensive suite of modern difference-in-differences estimators for causal inference, implementing state-of-the-art methods from recent econometric literature. This module goes beyond traditional DiD approaches by offering doubly robust, inverse propensity weighted, and outcome regression estimators that address common challenges in observational studies with two time periods (pre-treatment and post-treatment) and two groups (treatment group and comparison group).

## Core Functionality

### 1. **Doubly Robust DiD Estimators** (`drdid`)

Based on [Sant'Anna and Zhao (2020)](https://doi.org/10.1016/j.jeconom.2020.06.003), these estimators combine outcome regression and propensity score methods to achieve:

- **Double robustness**: Consistent estimates when either the outcome model or the propensity score model is correctly specified
- **Local efficiency**: Optimal estimators for panel and repeated cross-section data that achieve the semi-parametric efficiency bound
- **Flexible nuisance estimation**: Compatible with machine learning methods for high-dimensional co-variates

Available variants:

- `drdid_panel` and `drdid_imp_panel`: For panel data with repeated observations of the same individuals
- `drdid_rc` and `drdid_imp_rc`: For repeated cross-section data
- `drdid_imp_local_rc`: Locally efficient and improved estimator for repeated cross-section data
- `drdid_trad_rc`: Traditional implementation with alternative weighting

### 2. **Inverse Propensity Weighted DiD** (`ipwdid`)

IPW-based estimators that reweight observations to balance covariate distributions:

- `ipw_did_panel` and `ipw_did_rc`: Standard IPW DiD estimators
- `std_ipw_did_panel` and `std_ipw_did_rc`: Stabilized (Hajek-type) IPW estimators with improved finite-sample properties

### 3. **Outcome Regression DiD** (`ordid`)

Regression-based estimators that model outcomes directly:

- `reg_did_panel` and `reg_did_rc`: Flexible outcome regression DiD
- `twfe_did_panel` and `twfe_did_rc`: Two-way fixed effects implementations

> **⚠️ Note:**
> The core estimators for this module are the **doubly robust estimators**. We recommend users to utilize these estimators in practice as they will give the most robust estimate of the ATT. We include the other estimators mainly for researchers to compare estimates from more traditional DiD estimators in their research designs.

## Features

### Unified High-Level API

Three main functions provide access to all estimators with a consistent pandas-friendly interface

```python
from pydid.drdid import drdid, ipwdid, ordid

# Doubly robust estimation
result = drdid(data, y_col='outcome', time_col='period', treat_col='treated',
               id_col='id', panel=True, covariates_formula='~ age + education + income',
               est_method='imp')

# IPW estimation
result = ipwdid(data, y_col='outcome', time_col='period', treat_col='treated',
               id_col='id', panel=True, covariates_formula='~ age + education + income',
               est_method='std_ipw')  # Stabilized weights

# Outcome regression
result = ordid(data, y_col='outcome', time_col='period', treat_col='treated',
               id_col='id', panel=True, covariates_formula='~ age + education + income')
```

### Flexible Low-Level API

For advanced users, all underlying estimators are directly accessible with NumPy arrays

```python
from pydid.drdid.estimators import drdid_imp_local_rc

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

### Usage

The following is a portion of the empirical illustration considered by Sant'Anna and Zhao (2020) that uses the LaLonde sample from the NSW experiment and considers data from the Current Population Survey (CPS) to form a non-experimental comparison group.

```python
import pydid

# Load the NSW example dataset
nsw_data = pydid.datasets.load_nsw()

# Estimate ATT using doubly robust DiD
att_result = pydid.drdid(
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

## Citation

```bibtex
@article{santanna2020doubly,
  title={Doubly Robust Difference-in-Differences Estimators},
  author={Sant'Anna, Pedro H. C. and Zhao, Jun B.},
  journal={arXiv preprint arXiv:1812.01723},
  year={2020}
}
```
