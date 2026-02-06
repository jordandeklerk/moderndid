# Difference-in-Differences with Intertemporal Treatment Effects

This module implements difference-in-differences estimators for settings where treatment can be non-binary, non-absorbing (time-varying), and where lagged treatments may affect current outcomes, following [de Chaisemartin and D'Haultfoeuille (2024)](https://doi.org/10.1162/rest_a_01414).

The computational methods here are inspired by the corresponding Python package [py_did_multiplegt_dyn](https://github.com/Credible-Answers/py_did_multiplegt_dyn) and the Stata package [did_multiplegt_dyn](https://github.com/Credible-Answers/did_multiplegt_dyn) by de Chaisemartin and D'Haultfoeuille.

## Quick Start

```python
import moderndid as md

df = md.load_favara_imbs()

# Estimate intertemporal treatment effects
result = md.did_multiplegt(
    data=df,
    yname="Dl_vloans_b",
    idname="county",
    tname="year",
    dname="inter_bra",
    effects=5,
    placebo=3,
    cluster="state_n",
)

# Plot results
md.plot_multiplegt(result)
```

## Key Parameters

**Treatment effects**
- `effects`: Number of post-treatment horizons to estimate
- `placebo`: Number of pre-treatment placebo tests
- `normalized`: Divide effects by cumulative treatment change

**Sample selection**
- `switchers`: `"both"` (default), `"in"` (increases only), `"out"` (decreases only)
- `same_switchers`: Use identical switchers across all horizons

**Control groups**
- `only_never_switchers`: Use only never-switchers as controls (default: `False`)

**Testing**
- `effects_equal`: Test equality of effects across horizons

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/didinter.html).
- For a complete worked example with output, see the [Intertemporal DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_inter_did.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/didinter.html).

## References

de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-differences estimators of intertemporal treatment effects. *Review of Economics and Statistics*, 106(6), 1723-1736.
