"""Synthetic data generation for DDD estimators."""

from __future__ import annotations

import numpy as np

_MEAN_Z1 = np.exp(0.25 / 2)
_SD_Z1 = np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
_MEAN_Z2 = 10.0
_SD_Z2 = 0.54164
_MEAN_Z3 = 0.21887
_SD_Z3 = 0.04453
_MEAN_Z4 = 402.0
_SD_Z4 = 56.63891

_MOVED_NAMES = {
    "gen_ddd_2periods",
    "gen_ddd_mult_periods",
    "gen_ddd_scalable",
    "gen_dgp_2periods",
    "gen_dgp_mult_periods",
    "gen_dgp_scalable",
    "gen_simple_ddd_data",
    "generate_simple_ddd_data",
}


def __getattr__(name):
    if name in _MOVED_NAMES:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _transform_covariates(x: np.ndarray) -> np.ndarray:
    """Transform X to Z via nonlinear functions for doubly robust testing."""
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    z1_tilde = np.exp(x1 / 2)
    z2_tilde = x2 / (1 + np.exp(x1)) + 10
    z3_tilde = (x1 * x3 / 25 + 0.6) ** 3
    z4_tilde = (x1 + x4 + 20) ** 2

    z1 = (z1_tilde - _MEAN_Z1) / _SD_Z1
    z2 = (z2_tilde - _MEAN_Z2) / _SD_Z2
    z3 = (z3_tilde - _MEAN_Z3) / _SD_Z3
    z4 = (z4_tilde - _MEAN_Z4) / _SD_Z4

    return np.column_stack([z1, z2, z3, z4])


def _fps(psi: float, coefs: np.ndarray, xvars: np.ndarray) -> np.ndarray:
    """Compute propensity score index."""
    return psi * (xvars @ coefs)


def _fps2(psi: float, coefs: np.ndarray, xvars: np.ndarray, c: float) -> np.ndarray:
    """Compute propensity score index with constant."""
    return psi * (c + xvars @ coefs)


def _freg(coefs: np.ndarray, xvars: np.ndarray) -> np.ndarray:
    """Compute outcome regression index."""
    return 210 + xvars @ coefs


def _generate_ps_coefficients(coef_rng, n_free):
    """Deterministic weight vectors, psi signs, and constants for multinomial."""
    ws = np.empty((n_free, 4))
    psis = np.empty(n_free)
    cs = np.empty(n_free)
    for i in range(n_free):
        w = coef_rng.standard_normal(4)
        w /= np.linalg.norm(w)
        ws[i] = w
        psis[i] = 1.0 if i % 2 == 0 else -1.0
        cs[i] = coef_rng.uniform(-2.0, 2.0)
    return ws, psis, cs


def _assign_cohort_partition(
    rng,
    n,
    n_free,
    ws,
    psis,
    cs,
    ps_covars,
    cohort_values,
    xsi_ps,
):
    """Vectorized multinomial draw to (cohort, partition) arrays."""
    exp_vals = np.empty((n, n_free))
    for i in range(n_free):
        exp_vals[:, i] = np.exp(_fps2(xsi_ps * psis[i], ws[i], ps_covars, cs[i]))

    sum_exp = 1.0 + exp_vals.sum(axis=1, keepdims=True)
    probs = exp_vals / sum_exp
    prob_ref = 1.0 / sum_exp

    all_probs = np.column_stack([probs, prob_ref])
    cum_probs = np.cumsum(all_probs, axis=1)
    u = rng.uniform(size=n)
    group_types = (u[:, None] >= cum_probs).sum(axis=1)

    treated_cohorts = cohort_values[cohort_values != 0]
    all_cohorts = np.concatenate([treated_cohorts, [0]])
    cohort_idx = group_types // 2
    cohort = all_cohorts[cohort_idx]
    partition = 1 - (group_types % 2)

    return cohort, partition


def _compute_scalable_outcome(
    t,
    baseline,
    index_trend,
    index_pt_violation,
    cohort,
    partition,
    cohort_values,
    att_base,
    n,
    rng,
):
    """Per-period outcome with treatment effects for all active cohorts."""
    baseline_t = baseline + (t - 1) * index_trend + (t - 1) * index_pt_violation
    y = baseline_t + rng.standard_normal(n)

    for g in cohort_values:
        if g == 0 or t < g:
            continue
        k = t - g + 1
        y_g = baseline_t + rng.standard_normal(n) + att_base * g * k * partition
        mask = (cohort == g) & (partition == 1)
        y[mask] = y_g[mask]

    return y


def _build_cov_dict(z_first4, x_raw, n_covariates):
    """Build {"cov1": ..., "covK": ...} dict from transformed + raw arrays."""
    d = {}
    for i in range(4):
        d[f"cov{i + 1}"] = z_first4[:, i]
    if x_raw is not None:
        for i in range(n_covariates - 4):
            d[f"cov{i + 5}"] = x_raw[:, i]
    return d


def _select_covars(dgp_type, x_first4, z_first4):
    """Return (ps_covars_4col, or_covars_4col) tuple per dgp_type."""
    if dgp_type == 1:
        return z_first4, z_first4
    if dgp_type == 2:
        return x_first4, z_first4
    if dgp_type == 3:
        return z_first4, x_first4
    return x_first4, x_first4
