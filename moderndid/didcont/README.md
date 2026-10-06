# Difference-in-Differences with Continuous Treatments

This module estimates difference-in-differences effects when treatment intensity varies across units and adoption can be staggered across groups. The [December 2025 paper by Callaway, Goodman-Bacon, and Sant'Anna](https://psantanna.com/files/CGBS_v4.pdf) develops the identification framework, including the additional assumptions needed to interpret fitted dose derivatives as causal responses.

The computational methods here are inspired by the corresponding R package [contdid](https://github.com/bcallaway11/contdid).

## Quick Start

```python
import moderndid as did

data = did.gen_cont_did_data(
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

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/didcont.html).
- For a complete worked example with output, see the [Continuous DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_cont_did.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/didcont.html).

## References

Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025). Difference-in-differences with a continuous treatment. *American Economic Review* (forthcoming). [December 31, 2025 manuscript](https://psantanna.com/files/CGBS_v4.pdf).
