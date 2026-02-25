# Doubly-Robust Difference-in-Differences

This module provides doubly robust, inverse propensity weighted, and outcome regression difference-in-differences estimators for the ATT with two time periods and two groups, following [Sant'Anna and Zhao (2020)](https://doi.org/10.1016/j.jeconom.2020.09.006).

The computational methods here are inspired by the corresponding R package [DRDID](https://github.com/pedrohcgs/drdid).

## Quick Start

```python
import moderndid

# NSW dataset
nsw_data = moderndid.datasets.load_nsw()

# Estimate ATT using doubly robust DiD
result = moderndid.drdid(
    data=nsw_data,
    yname='re',
    tname='year',
    treatname='experimental',
    idname='id',
    panel=True,
    xformla="~ age + educ + black + married + nodegree + hisp + re74",
    est_method='imp',
)
```

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/drdid.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/drdid.html).

## References

Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics*, 219(1), 101-122.
