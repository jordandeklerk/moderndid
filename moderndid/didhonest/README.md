# Honest Difference-in-Differences

This module provides sensitivity analysis for difference-in-differences designs, allowing researchers to assess how violations of the parallel trends assumption affect their conclusions. Rather than assuming exactly parallel trends, this framework constructs robust confidence intervals that remain valid under plausible violations.

The computational methods here are inspired by the corresponding R package [HonestDiD](https://github.com/asheshrambachan/HonestDiD) by Rambachan and Roth.

## Quick Start

```python
import moderndid as did

# Load data and estimate event study
data = did.load_mpdta()
result = did.att_gt(
    data=data,
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
)
event_study = did.aggte(result, type="dynamic")

# Sensitivity analysis for event_time=0
sensitivity = did.honest_did(
    event_study,
    event_time=0,
    sensitivity_type="relative_magnitude",
    m_bar_vec=[0.5, 1.0, 1.5, 2.0],
)

# Plot results
did.plot_sensitivity(sensitivity)
```

## Key Parameters

**Sensitivity types** (`sensitivity_type`)
- `"relative_magnitude"`: Bound post-treatment violations relative to pre-treatment
- `"smoothness"`: Restrict deviations from linear extrapolation of pre-trends

**Relative magnitude parameters**
- `m_bar_vec`: Values of Mbar (e.g., `[0.5, 1.0, 1.5, 2.0]`)
- Mbar=1 means post-treatment violations are no larger than worst pre-treatment violation

**Smoothness parameters**
- `m_vec`: Values of M (e.g., `[0, 0.01, 0.02, 0.03]`)
- M=0 imposes exactly linear counterfactual trends

**CI methods** (`method`)
- `"C-LF"` (default for relative magnitude): Computationally efficient
- `"FLCI"` (default for smoothness): Optimal length
- `"ARP"`: Data-driven construction
- `"Conditional"`: Conditions on non-negativity

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/honestdid.html).
- For a complete worked example with output, see the [Honest DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_honest_did.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/didhonest.html).

## References

Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *American Economic Review*, 113(9), 2555-2591.
