# Difference-in-Differences with Continuous Treatments

This module extends difference-in-differences to settings where treatment intensity is continuous and adoption can be staggered across groups, implementing the estimators in [Callaway, Goodman-Bacon, and Sant'Anna (2024)](https://arxiv.org/abs/2107.02637).

The computational methods here are inspired by the corresponding R package [contdid](https://github.com/bcallaway11/contdid).

## Quick Start

```python
import moderndid as did

data = did.simulate_cont_did_data(
    n=2000,
    num_time_periods=4,
    dose_linear_effect=0.5,
    dose_quadratic_effect=0.3,
    seed=1234,
)

# Estimate dose-response function
result = did.cont_did(
    data=data,
    yname="Y",
    tname="time_period",
    idname="id",
    dname="D",
    gname="G",
    aggregation="dose",
)

# Plot results
did.plot_dose_response(result, effect_type="att")
```

## Key Parameters

**Target parameters** (`target_parameter`)
- `"level"` (default): ATT at different dose levels
- `"slope"`: ACRT, the derivative of the dose-response curve

**Aggregation** (`aggregation`)
- `"dose"`: Dose-response views (ATT(d) and ACRT(d))
- `"eventstudy"`: Event-study views across time

**Estimation method** (`dose_est_method`)
- `"parametric"` (default): B-spline approximation
- `"cck"`: Nonparametric (two-period settings only)

**Control groups** (`control_group`)
- `"notyettreated"` (default): Include not-yet-treated units
- `"nevertreated"`: Never-treated units only

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/didcont.html).
- For a complete worked example with output, see the [Continuous DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_cont_did.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/didcont.html).

## References

Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024). Difference-in-differences with a continuous treatment. *Journal of Econometrics* (forthcoming). [arXiv:2107.02637](https://arxiv.org/abs/2107.02637)
