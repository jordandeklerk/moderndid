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

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/honestdid.html).
- For a complete worked example with output, see the [Honest DiD Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_honest_did.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/didhonest.html).

## References

Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *American Economic Review*, 113(9), 2555-2591.
