# Dynamic Covariate Balancing DiD

This module implements dynamic covariate balancing estimation for panel data with time-varying treatments following the approach of [Viviano and Bradic (2026)](https://doi.org/10.1093/biomet/asag016).

The methods in this module are inspired by the R package [DynBalancing](https://github.com/daviviano/DynBalancing).

## Quick Start

```python
from moderndid.core.data import load_acemoglu
from moderndid.diddynamic import dyn_balancing
import polars as pl

df = load_acemoglu()

# Convert unit identifiers to numeric
units = sorted(df["Unit"].unique().to_list())
unit_map = {u: i for i, u in enumerate(units)}
df = df.with_columns(pl.col("Unit").replace(unit_map).cast(pl.Int64))

# Estimate the ATE of two periods of democracy vs no democracy
result = dyn_balancing(
    data=df,
    yname="Y",
    tname="Time",
    idname="Unit",
    treatment_name="D",
    ds1=[1, 1],
    ds2=[0, 0],
    xformla="~ V1 + V2 + V3 + V4 + V5",
    fixed_effects=["region"],
)

# Trace out effects across history lengths
history = dyn_balancing(
    data=df,
    yname="Y",
    tname="Time",
    idname="Unit",
    treatment_name="D",
    ds1=[1, 1, 1, 1, 1],
    ds2=[0, 0, 0, 0, 0],
    histories_length=[1, 2, 3, 4, 5],
    xformla="~ V1 + V2 + V3 + V4 + V5",
    fixed_effects=["region"],
)
```

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/diddynamic.html).
- For a complete worked example with output, see the [Dynamic Covariate Balancing DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_dyn_balancing.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/diddynamic.html).

## References

Viviano, D. and Bradic, J. (2026). Dynamic covariate balancing: estimating treatment effects over time with potential local projections. *Biometrika*, asag016.
