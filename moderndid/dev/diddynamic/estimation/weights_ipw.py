"""IPW, AIPW, and IPW-MSM weight estimation for dynamic treatment regimes."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import statsmodels.api as sm

from moderndid.dev.diddynamic.estimation.coefficients import compute_coefficients


class IPWResult(NamedTuple):
    """Result of IPW-based weight estimation.

    Attributes
    ----------
    mu_hat : float
        Estimated potential outcome under target treatment history.
    variance : float
        Estimated variance of the estimator.
    """

    mu_hat: float
    variance: float


def compute_ipw_estimator(
    n_periods: int,
    outcome: np.ndarray,
    treatment_matrix: np.ndarray,
    covariates_t: dict[int, np.ndarray],
    ds: np.ndarray,
    *,
    method: str = "ipw",
    clip_bounds: tuple[float, float] = (0.01, 0.99),
    regularization: bool = True,
    lags: int | None = None,
    dim_fe: int = 0,
) -> IPWResult:
    r"""Estimate potential outcomes using propensity-score weighting.

    Fits per-period logistic regressions to estimate the joint propensity
    :math:`P(D_{i,1:T} = d_{1:T} \mid X_{i,1:T})` as a product of per-period
    conditional probabilities, then constructs inverse-probability weights for
    causal estimation. The IPW weights for period :math:`t` are

    .. math::

        \hat{\gamma}_{i,t}^{\text{IPW}} = \prod_{s=1}^{t}
        \frac{\mathbf{1}\{D_{i,s} = d_s\}}{P(D_{i,s} = d_s \mid H_{i,s})}.

    Three strategies are available:

    - ``ipw`` uses standard inverse probability weights normalised by the
      joint propensity score.
    - ``aipw`` augments IPW with a LASSO outcome regression for
      double robustness.
    - ``ipw_msm`` uses stabilised marginal structural model weights
      where the numerator is the marginal treatment probability.

    Parameters
    ----------
    n_periods : int
        Number of time periods.
    outcome : ndarray, shape (n,)
        Outcome vector at the final period.
    treatment_matrix : ndarray, shape (n, T)
        Binary treatment assignments per unit and period.
    covariates_t : dict[int, ndarray]
        Per-period covariate matrices keyed by 0-based period index.
    ds : ndarray, shape (T,)
        Target treatment history.
    method : {'ipw', 'aipw', 'ipw_msm'}
        Weighting strategy.
    clip_bounds : tuple[float, float]
        Lower and upper bounds for propensity score clipping.
    regularization : bool
        If True use cross-validated LASSO for outcome model in AIPW.
    lags : int or None
        Treatment lags for the coefficient stage (AIPW only).
    dim_fe : int
        Number of fixed-effect columns (AIPW only).

    Returns
    -------
    IPWResult
        Estimated potential outcome and variance.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    if method == "ipw":
        return _ipw(n_periods, outcome, treatment_matrix, covariates_t, ds, clip_bounds)
    if method == "aipw":
        return _aipw(n_periods, outcome, treatment_matrix, covariates_t, ds, clip_bounds, regularization, lags, dim_fe)
    if method == "ipw_msm":
        return _ipw_msm(n_periods, outcome, treatment_matrix, covariates_t, ds, clip_bounds)
    raise ValueError(f"Unknown method {method!r}. Expected 'ipw', 'aipw', or 'ipw_msm'.")


def _estimate_joint_propensity(n_periods, treatment_matrix, covariates_t, ds, clip_bounds):
    """Estimate joint propensity :math:`P(D=d_s|X)` as product of per-period scores."""
    n = treatment_matrix.shape[0]
    joint_ps = np.ones(n)

    for t in range(n_periods):
        x_t = covariates_t[t]
        d_t = treatment_matrix[:, t]

        valid = ~np.isnan(x_t).any(axis=1)
        x_valid = x_t[valid]
        d_valid = d_t[valid]

        if len(np.unique(d_valid)) < 2:
            ps_t = np.where(d_valid == ds[t], 1.0, 0.0)
        else:
            x_with_const = sm.add_constant(x_valid, has_constant="add")
            model = sm.Logit(d_valid, x_with_const)
            result = model.fit(disp=0, maxiter=100)
            prob_1 = result.predict(x_with_const)
            ps_t = np.where(ds[t] == 1.0, prob_1, 1.0 - prob_1)

        ps_t = np.clip(ps_t, clip_bounds[0], clip_bounds[1])

        ps_full = np.ones(n)
        ps_full[valid] = ps_t
        joint_ps *= ps_full

    return np.clip(joint_ps, clip_bounds[0], clip_bounds[1])


def _match_mask(treatment_matrix, ds):
    """Return boolean mask of units matching the target treatment history."""
    if treatment_matrix.shape[1] == 1:
        return treatment_matrix[:, 0] == ds[0]
    return np.all(treatment_matrix == ds, axis=1)


def _ipw(n_periods, outcome, treatment_matrix, covariates_t, ds, clip_bounds):
    """Compute standard inverse probability weighting estimate."""
    joint_ps = _estimate_joint_propensity(n_periods, treatment_matrix, covariates_t, ds, clip_bounds)
    mask = _match_mask(treatment_matrix, ds)

    w = 1.0 / joint_ps
    w_masked = w * mask

    denom = w_masked.sum()
    mu_hat = float((w_masked * outcome).sum() / denom)
    variance = float(((w**2) * mask * (outcome - mu_hat) ** 2).sum() / denom**2)

    return IPWResult(mu_hat=mu_hat, variance=variance)


def _aipw(n_periods, outcome, treatment_matrix, covariates_t, ds, clip_bounds, regularization, lags, dim_fe):
    """Augmented inverse probability weighting (doubly robust) estimator."""
    joint_ps = _estimate_joint_propensity(n_periods, treatment_matrix, covariates_t, ds, clip_bounds)
    mask = _match_mask(treatment_matrix, ds)

    coefs = compute_coefficients(
        n_periods, outcome, treatment_matrix, covariates_t, ds, "lasso_subsample", regularization, 10, lags, dim_fe
    )

    last = n_periods - 1
    m_x = np.zeros_like(outcome, dtype=float)
    not_nas = coefs.not_nas[last]
    m_x[not_nas] = coefs.pred_t[last]

    w = 1.0 / joint_ps
    w_masked = w * mask
    denom = w_masked.sum()

    n = len(outcome)
    outcome_reg = m_x.mean()
    augmentation = (w_masked * (outcome - m_x)).sum() / denom

    mu_hat = float(outcome_reg + augmentation)

    residuals = m_x + mask * w * (outcome - m_x) / (mask * w).sum() * mask.sum() - mu_hat
    variance = float(residuals.var(ddof=1) / n)

    return IPWResult(mu_hat=mu_hat, variance=variance)


def _ipw_msm(n_periods, outcome, treatment_matrix, covariates_t, ds, clip_bounds):
    """Marginal structural model estimator with stabilised weights."""
    joint_ps = _estimate_joint_propensity(n_periods, treatment_matrix, covariates_t, ds, clip_bounds)
    mask = _match_mask(treatment_matrix, ds)

    marginal_prob = mask.mean()
    sw = marginal_prob / joint_ps

    sw_masked = sw * mask
    denom = sw_masked.sum()
    mu_hat = float((sw_masked * outcome).sum() / denom)
    variance = float(((sw**2) * mask * (outcome - mu_hat) ** 2).sum() / denom**2)

    return IPWResult(mu_hat=mu_hat, variance=variance)
