"""IPW and AIPW estimation for dynamic treatment regimes."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.optimize import linprog

from moderndid.diddynamic.container import IPWResult
from moderndid.diddynamic.estimation.coefficients import compute_coefficients
from moderndid.diddynamic.estimation.inference import compute_variance


def compute_ipw_estimator(
    n_periods,
    outcome,
    treatment_matrix,
    covariates_t,
    ds,
    *,
    method="ipw",
    clip_bounds=(0.01, 0.99),
    regularization=True,
    lags=None,
    dim_fe=0,
):
    r"""Estimate a potential outcome with inverse probability weights.

    The propensity score of each period comes from a logistic regression of
    that period's treatment on its covariates. The regression uses only the
    units that followed the target history in the earlier periods. Like the
    weights of [1]_, the score therefore conditions on past treatments.
    Without that conditioning the weights are biased whenever treatment in
    one period depends on treatment in the period before.

    Columns that are constant among those units or that the earlier columns
    span are dropped first. A full set of fixed-effect dummies therefore
    loses one level. When the covariates perfectly separate treated from
    untreated units, the logistic regression has no maximum likelihood
    estimate and the function raises an error. If all of those units take
    the same treatment, the score is one before clipping.

    A unit enters the weights of a period when it follows the target history
    through that period with observed covariates in every period so far. If
    some period has no such unit with an observed outcome, the function
    raises an error.

    The ``ipw`` strategy averages the final-period outcomes with normalized
    inverse probability weights. The ``aipw`` strategy computes the estimate
    of :func:`compute_dcb_estimator` with the weights of every period in
    place of the balancing weights. Its outcome projections come from the
    ``lasso_subsample`` strategy of :func:`compute_coefficients`. This is the
    augmented estimator of [1]_. It stays consistent when either these
    logistic propensity models or the linear outcome projections are
    correctly specified.

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
    method : {'ipw', 'aipw'}
        Weighting strategy.
    clip_bounds : tuple[float, float]
        Lower and upper bounds for the propensity scores and their products.
    regularization : bool
        If True use cross-validated LASSO for the outcome model in AIPW.
    lags : int or None
        Treatment lags for the coefficient stage (AIPW only).
    dim_fe : int
        Number of fixed-effect columns (AIPW only).

    Returns
    -------
    IPWResult
        Estimated potential outcome, its variance, and the weights behind it.

        - **mu_hat**: Estimated potential outcome under the target treatment history
        - **variance**: Estimated variance of the estimator
        - **gammas**: Normalized weights per unit and period
        - **predictions**: Outcome projections per unit and period
        - **not_nas**: Row indices that enter each period

    Notes
    -----
    The weight of unit :math:`i` in period :math:`t` is proportional to

    .. math::

        \prod_{s=1}^{t} \frac{\mathbf{1}\{D_{i,s} = d_s\}}
        {\hat{P}(D_{i,s} = d_s \mid X_{i,s}, D_{i,1:(s-1)} = d_{1:(s-1)})},

    where each estimated propensity score and their product are clipped to
    ``clip_bounds``. The weights of each period sum to one. The variance
    comes from :func:`compute_variance`. For ``ipw`` it reduces to the
    weighted residual variance of the final-period outcomes. Since it treats
    the estimated propensity scores as known, the ``ipw`` standard error is
    conservative when the logistic models are correctly specified.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    if method not in ("ipw", "aipw"):
        raise ValueError(f"Unknown method {method!r}. Expected 'ipw' or 'aipw'.")

    propensity = _period_propensities(n_periods, treatment_matrix, covariates_t, ds, clip_bounds)
    _check_history_followed(outcome, treatment_matrix, ds, propensity)
    if method == "ipw":
        return _ipw(outcome, treatment_matrix, ds, propensity, clip_bounds)
    return _aipw(
        n_periods, outcome, treatment_matrix, covariates_t, ds, propensity, clip_bounds, regularization, lags, dim_fe
    )


def _period_propensities(n_periods, treatment_matrix, covariates_t, ds, clip_bounds):
    """Return each unit's clipped propensity of the target treatment given the target history before it.

    A unit that left the target history before a period, or whose covariates
    are missing in that period, has no propensity there and gets NaN.
    """
    n = treatment_matrix.shape[0]
    propensity = np.full((n, n_periods), np.nan)

    for t in range(n_periods):
        x_t = covariates_t[t]
        rows = np.flatnonzero(np.all(treatment_matrix[:, :t] == ds[:t], axis=1) & ~np.isnan(x_t).any(axis=1))
        d_rows = treatment_matrix[rows, t]

        if len(np.unique(d_rows)) < 2:
            ps_t = np.where(d_rows == ds[t], 1.0, 0.0)
        else:
            x_rows = x_t[rows]
            design = np.column_stack([np.ones(len(rows)), x_rows[:, _independent_columns(x_rows)]])
            if _is_separated(design, d_rows):
                raise ValueError(
                    f"The covariates perfectly predict the treatment in period {t + 1} of the treatment history "
                    "among the units that followed the history until then. Its propensity score therefore has "
                    "no maximum likelihood estimate. This happens, for example, when every unit at one "
                    "fixed-effect level has the same treatment. Drop or merge such covariates or levels. "
                    "Alternatively, use balancing='dcb'."
                )
            prob_1 = sm.Logit(d_rows, design).fit(disp=0, maxiter=100).predict(design)
            ps_t = np.where(ds[t] == 1.0, prob_1, 1.0 - prob_1)

        propensity[rows, t] = np.clip(ps_t, clip_bounds[0], clip_bounds[1])

    return propensity


def _check_history_followed(outcome, treatment_matrix, ds, propensity):
    """Raise an error when no unit with the data its weights need follows the target history to some period."""
    followed = np.cumprod(treatment_matrix == ds, axis=1).astype(bool)
    followed &= ~np.isnan(np.cumprod(propensity, axis=1)) & ~np.isnan(outcome)[:, None]
    empty = np.flatnonzero(~followed.any(axis=0))
    if len(empty) > 0:
        t = empty[0]
        history = [int(d) for d in ds[: t + 1]]
        raise ValueError(
            f"No unit with observed covariates and outcome follows the treatment history {history} through "
            f"period {t + 1}. The inverse probability weights of that period are therefore undefined."
        )


def _independent_columns(x):
    """Return a mask of the columns that add to the rank of an intercept and the columns before them."""
    n = x.shape[0]
    basis = np.ones((n, 1)) / np.sqrt(n)
    keep = np.zeros(x.shape[1], dtype=bool)
    for j in range(x.shape[1]):
        col = x[:, j]
        resid = col - basis @ (basis.T @ col)
        # A second projection removes what rounding left in the first.
        resid -= basis @ (basis.T @ resid)
        norm = np.linalg.norm(resid)
        # Since the separation check's linear program treats violations below 1e-7 as zero, it could
        # mistake a column this close to the earlier ones for a separating direction.
        if norm > 1e-7 * np.linalg.norm(col):
            keep[j] = True
            basis = np.column_stack([basis, resid / norm])
    return keep


def _is_separated(design, treatment):
    """Return True when a linear index separates treated from untreated units.

    The logistic likelihood has a maximum only if no nonzero coefficient
    vector gives every treated unit a nonnegative index and every untreated
    unit a nonpositive one. A linear program searches the unit box for such a
    vector. It first centers and scales the columns, since that leaves the set
    of separating directions unchanged.
    """
    x = design[:, 1:]
    scaled = np.column_stack([design[:, :1], (x - x.mean(axis=0)) / x.std(axis=0)])
    signed = scaled * np.where(treatment == 1.0, 1.0, -1.0)[:, None]
    result = linprog(
        -signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(len(treatment)),
        bounds=(-1.0, 1.0),
        method="highs",
    )
    return result.status == 0 and -result.fun > 1e-6


def _history_weights(treatment_matrix, ds, propensity, clip_bounds, rows):
    """Return the normalized inverse probability weights of the units on the target history in each period."""
    n, n_periods = propensity.shape
    cumulative = np.cumprod(propensity, axis=1)
    gammas = np.zeros((n, n_periods))
    for t in range(n_periods):
        idx = rows[t]
        # A unit without a propensity in this or an earlier period has no inverse probability weight.
        on_path = np.all(treatment_matrix[idx, : t + 1] == ds[: t + 1], axis=1) & ~np.isnan(cumulative[idx, t])
        weights = np.zeros(len(idx))
        weights[on_path] = 1.0 / np.clip(cumulative[idx[on_path], t], clip_bounds[0], clip_bounds[1])
        gammas[idx, t] = weights / weights.sum()
    return gammas


def _ipw(outcome, treatment_matrix, ds, propensity, clip_bounds):
    """Average the final-period outcomes with normalized inverse probability weights."""
    n, n_periods = propensity.shape
    rows = [np.flatnonzero(~np.isnan(outcome))] * n_periods
    gammas = _history_weights(treatment_matrix, ds, propensity, clip_bounds, rows)
    mu_hat = float(gammas[rows[-1], -1] @ outcome[rows[-1]])
    predictions = np.full((n, n_periods), mu_hat)
    variance = compute_variance(gammas, predictions, rows, outcome)
    return IPWResult(mu_hat=mu_hat, variance=variance, gammas=gammas, predictions=predictions, not_nas=rows)


def _aipw(
    n_periods, outcome, treatment_matrix, covariates_t, ds, propensity, clip_bounds, regularization, lags, dim_fe
):
    """Correct the outcome projections of every period with inverse probability weights."""
    coefs = compute_coefficients(
        n_periods, outcome, treatment_matrix, covariates_t, ds, "lasso_subsample", regularization, 10, lags, dim_fe
    )
    not_nas = list(coefs.not_nas)
    gammas = _history_weights(treatment_matrix, ds, propensity, clip_bounds, not_nas)

    predictions = np.zeros((len(outcome), n_periods))
    previous = np.full(len(not_nas[0]), 1.0 / len(not_nas[0]))
    adjustment = 0.0
    for t in range(n_periods):
        predictions[not_nas[t], t] = coefs.pred_t[t]
        if t > 0:
            previous = gammas[not_nas[t], t - 1]
        adjustment += (gammas[not_nas[t], t] - previous) @ coefs.pred_t[t]

    last = n_periods - 1
    mu_hat = float(gammas[not_nas[last], last] @ outcome[not_nas[last]] - adjustment)
    variance = compute_variance(gammas, predictions, not_nas, outcome)
    return IPWResult(mu_hat=mu_hat, variance=variance, gammas=gammas, predictions=predictions, not_nas=not_nas)
