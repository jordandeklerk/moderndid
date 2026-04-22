# Extended Two-Way Fixed Effects

This module implements the Extended Two-Way Fixed Effects (ETWFE) estimator for staggered difference-in-differences following [Wooldridge (2021, 2023)](https://doi.org/10.1093/ectj/utad016). Rather than discarding the TWFE estimator in staggered settings, ETWFE saturates the model with cohort-by-time interaction dummies so that each coefficient recovers a cohort-time-specific average treatment effect on the treated. The approach extends naturally to nonlinear models through the two-way Mundlak regression.

The computational methods here are inspired by the corresponding Python package [etwfe](https://github.com/armandkapllani/etwfe) and the R package [etwfe](https://github.com/grantmcdermott/etwfe) by Grant McDermott.

## Quick Start

```python
import moderndid as did

data = did.load_mpdta()

# Estimate cohort-time ATTs
mod = did.etwfe(
    data=data,
    yname="lemp",
    tname="year",
    gname="first.treat",
    idname="countyreal",
)

# Aggregate to an event study
event = did.emfx(mod, type="event", window=(-3, 3))

# Plot results
did.plot_event_study(event)
```

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/etwfe.html).
- For a complete worked example with output, see the [ETWFE Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_etwfe.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/etwfe.html).

## References

Wooldridge, J. M. (2021). Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators.

Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel data. *The Econometrics Journal*, 26(3), C31-C66.
