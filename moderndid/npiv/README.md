# Nonparametric Instrumental Variables Estimation

This module implements nonparametric instrumental variables (NPIV) estimation with data-driven sieve dimension selection and honest uniform confidence bands, following [Chen, Christensen, and Kankanala (2024)](https://arxiv.org/abs/2107.11869). The structural function is approximated by a B-spline sieve and estimated by two-stage least squares, with the sieve dimension chosen adaptively through a bootstrap implementation of Lepski's method to achieve the minimax sup-norm convergence rate. Within ModernDiD, the same machinery powers the nonparametric dose-response estimator in `cont_did`.

The computational methods here are inspired by the corresponding R package [npiv](https://github.com/tkitagawa/npiv) by Chen, Christensen, and Kankanala.

## Quick Start

```python
import numpy as np
import moderndid as did

df = did.load_engel()

# Estimate the Engel curve on a uniform grid
logexp_eval = np.linspace(4.5, 6.5, 100).reshape(-1, 1)

result = did.npiv(
    data=df,
    yname="food",
    xname="logexp",
    wname="logwages",
    x_eval=logexp_eval,
    j_x_segments=5,
    biters=200,
    seed=42,
)
```

## Documentation

- For full function signatures and parameters, see the [API Reference](https://moderndid.readthedocs.io/en/latest/api/npiv.html).
- For a complete worked example with output, see the [NPIV Example](https://moderndid.readthedocs.io/en/latest/user_guide/example_npiv.html).
- For theoretical background, see the [Background section](https://moderndid.readthedocs.io/en/latest/background/npiv.html).

## References

Chen, X., Christensen, T., & Kankanala, S. (2024). Adaptive estimation and uniform confidence bands for nonparametric structural functions and elasticities. *Review of Economic Studies* (forthcoming). [arXiv:2107.11869](https://arxiv.org/abs/2107.11869)

Chen, X., & Christensen, T. (2018). Optimal sup-norm rates and uniform inference on nonlinear functionals of nonparametric IV regression. *Quantitative Economics*, 9(1), 39-84.
