# Difference-in-Differences with Multiple Time Periods

This module implements difference-in-differences estimation for staggered adoption designs where units receive treatment at different times following the approach of [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001).

The methods in this module are inspired by the excellent R package [did](https://github.com/bcallaway11/did).

## Quick Start

```python
import moderndid as did

data = did.load_mpdta()

# Estimate group-time average treatment effects
result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
)

# Aggregate to event study
event_study = did.aggte(result, type="dynamic")

# Plot results
did.plot_event_study(event_study)
```

## Core Functions

| Function | Description |
|----------|-------------|
| `att_gt` | Estimate group-time average treatment effects |
| `aggte` | Aggregate effects (event study, simple, group, calendar) |
| `plot_gt` | Plot group-time effects |
| `plot_event_study` | Plot event study |

## Key Parameters

**Estimation methods** (`est_method`)
- `"dr"` (default): Doubly robust
- `"ipw"`: Inverse probability weighting
- `"reg"`: Outcome regression

**Control groups** (`control_group`)
- `"nevertreated"` (default): Never-treated units only
- `"notyettreated"`: Include not-yet-treated units

**Aggregation types** (`aggte(..., type=)`)
- `"dynamic"`: Event study
- `"simple"`: Overall ATT
- `"group"`: By treatment cohort
- `"calendar"`: By calendar time

## Documentation

For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/did.html).

For a complete worked example with output, see the [Staggered DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_staggered_did.html).

For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/did.html).

## References

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
