# Triple Difference-in-Differences

This module implements triple difference-in-differences (DDD) estimators for settings where units must satisfy two criteria to be treated: belonging to a treatment group and being in an eligible partition. DDD designs leverage a third dimension of variation to relax parallel trends assumptions.

The computational methods here are inspired by the corresponding R package [triplediff](https://github.com/marcelortizv/triplediff) by Ortiz-Villavicencio and Sant'Anna.

## Quick Start

```python
import moderndid as did

dgp = did.gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
data = dgp["data"]

# Estimate group-time treatment effects
result = did.ddd(
    data=data,
    yname="y",
    tname="time",
    idname="id",
    gname="group",
    pname="partition",
    est_method="dr",
)

# Aggregate to event study
event_study = did.agg_ddd(result, aggregation_type="eventstudy")

# Plot results
did.plot_event_study(event_study)
```

## Key Parameters

**Estimation methods** (`est_method`)
- `"dr"` (default): Doubly robust
- `"ipw"`: Inverse probability weighting
- `"reg"`: Outcome regression

**Control groups** (`control_group`)
- `"nevertreated"` (default): Never-treated units only
- `"notyettreated"`: Include not-yet-treated units

**Aggregation types** (`agg_ddd(..., aggregation_type=)`)
- `"eventstudy"`: Event study
- `"simple"`: Overall ATT
- `"group"`: By treatment cohort
- `"calendar"`: By calendar time

**Data types**
- Panel data (default)
- Repeated cross-sections (`panel=False`)

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/didtriple.html).
- For a complete worked example with output, see the [Triple DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_triple_did.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/tripledid.html).

## References

Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). Better understanding triple differences estimators. *arXiv preprint arXiv:2505.09942*. [arXiv:2505.09942](https://arxiv.org/abs/2505.09942)
