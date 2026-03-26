"""Per-period LASSO coefficient estimation for dynamic covariate balancing."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LassoCV, Ridge


class CoefficientResult(NamedTuple):
    """Per-period coefficient estimates and predictions.

    Attributes
    ----------
    coef_t : list[ndarray]
        Coefficient vectors per period, each with shape ``(1 + p,)``
        where the first element is the intercept.
    pred_t : list[ndarray]
        Prediction vectors per period on the clean covariate matrix.
    covariates_nonna : list[ndarray]
        Covariate matrices per period with NaN rows removed.
    not_nas : list[ndarray]
        Integer arrays of valid row indices per period.
    model_effect : list[float]
        Last treatment coefficient per period. Empty for ``lasso_subsample``.
    """

    coef_t: list[np.ndarray]
    pred_t: list[np.ndarray]
    covariates_nonna: list[np.ndarray]
    not_nas: list[np.ndarray]
    model_effect: list[float]


def compute_coefficients(
    n_periods: int,
    outcome: np.ndarray,
    treatment_matrix: np.ndarray,
    covariates_t: dict[int, np.ndarray],
    ds: np.ndarray,
    method: str = "lasso_plain",
    regularization: bool = True,
    nfolds: int = 10,
    lags: int | None = None,
    dim_fe: int = 0,
) -> CoefficientResult:
    r"""Estimate coefficients for the potential local projection model.

    Implements the recursive coefficient estimation from Algorithm 2 of [1]_.
    For each period :math:`t = T, \ldots, 1`, estimates
    :math:`\hat{\beta}_{d_{1:T}}^{(t)}` by regressing the predicted outcome
    from period :math:`t+1` onto the history :math:`H_{i,t}`, building the
    chain of projections

    .. math::

        \mathbb{E}[Y_{i,T}(d_{1:T}) \mid H_{i,t}, D_{i,1:(t-1)} = d_{1:(t-1)}]
        = H_{i,t}(d_{1:(t-1)}) \beta_{d_{1:T}}^{(t)}.

    Two estimation strategies are available:

    - ``lasso_subsample`` fits only on units whose observed treatment history
      matches the target sequence ``ds`` up to each period.
    - ``lasso_plain`` fits on all units with treatment columns appended as
      unpenalised regressors, using Frisch-Waugh-Lovell residualisation to
      keep LASSO penalisation on covariates only.

    Parameters
    ----------
    n_periods : int
        Number of time periods.
    outcome : ndarray, shape (n,)
        Outcome vector at the final period.
    treatment_matrix : ndarray, shape (n, T)
        Binary treatment assignments.
    covariates_t : dict[int, ndarray]
        Per-period covariate matrices keyed by 0-based period index.
    ds : ndarray, shape (T,)
        Target treatment history.
    method : {'lasso_plain', 'lasso_subsample'}
        Estimation strategy.
    regularization : bool
        If True use cross-validated LASSO, otherwise ridge with
        :math:`\alpha = e^{-8}`.
    nfolds : int
        Cross-validation folds for LASSO.
    lags : int or None
        Treatment lags to include. Defaults to ``n_periods``.
    dim_fe : int
        Number of fixed-effect dummy columns at the end of each
        covariate matrix, zeroed out in non-final period predictions.

    Returns
    -------
    CoefficientResult
        Per-period coefficients, predictions, clean covariates, and
        valid-row indices.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    if lags is None:
        lags = n_periods

    if method == "lasso_subsample":
        return _lasso_subsample(n_periods, outcome, treatment_matrix, covariates_t, ds, regularization, nfolds, dim_fe)
    if method == "lasso_plain":
        return _lasso_plain(
            n_periods, outcome, treatment_matrix, covariates_t, ds, regularization, nfolds, lags, dim_fe
        )
    raise ValueError(f"Unknown method {method!r}. Expected 'lasso_plain' or 'lasso_subsample'.")


def _lasso_subsample(n_periods, outcome, treatment_matrix, covariates_t, ds, regularization, nfolds, dim_fe):
    """Fit only on units matching the target treatment history."""
    n_units = outcome.shape[0]
    predictions = outcome.copy().astype(float)
    nas_y = np.where(np.isnan(outcome))[0]

    coef_t = [np.array([])] * n_periods
    pred_t = [np.array([])] * n_periods
    covariates_nonna = [np.array([])] * n_periods
    not_nas = [np.array([])] * n_periods

    for t in reversed(range(n_periods)):
        xx_t = covariates_t[t].copy()
        predictions[nas_y] = np.nan

        all_matrix = np.column_stack([xx_t, predictions])
        valid = _valid_rows(all_matrix)
        not_nas[t] = np.where(valid)[0]
        all_clean = all_matrix[valid]
        x_clean = all_clean[:, :-1]
        y_clean = all_clean[:, -1]
        covariates_nonna[t] = x_clean

        subsample_mask = np.all(treatment_matrix[not_nas[t], : t + 1] == ds[: t + 1], axis=1)

        intercept, coefs, model = _fit_model(x_clean[subsample_mask], y_clean[subsample_mask], regularization, nfolds)

        xx_pred = xx_t.copy()
        if dim_fe > 0 and t == n_periods - 1:
            xx_pred[:, -dim_fe:] = 0.0
        valid_pred = _valid_rows(xx_pred)
        predictions = np.full(n_units, np.nan)
        predictions[valid_pred] = model.predict(xx_pred[valid_pred])

        coef_t[t] = np.concatenate([[intercept], coefs])
        pred_t[t] = model.predict(x_clean)

    return CoefficientResult(
        coef_t=coef_t,
        pred_t=pred_t,
        covariates_nonna=covariates_nonna,
        not_nas=not_nas,
        model_effect=[],
    )


def _lasso_plain(n_periods, outcome, treatment_matrix, covariates_t, ds, regularization, nfolds, lags, dim_fe):
    """Fit on all units with unpenalised treatment lags via residualisation."""
    n_units = outcome.shape[0]
    predictions = outcome.copy().astype(float)
    nas_y = np.where(np.isnan(outcome))[0]

    coef_t = [np.array([])] * n_periods
    pred_t = [np.array([])] * n_periods
    covariates_nonna = [np.array([])] * n_periods
    not_nas = [np.array([])] * n_periods
    model_effect = [0.0] * n_periods

    for t in reversed(range(n_periods)):
        xx_t = covariates_t[t].copy()
        predictions[nas_y] = np.nan

        all_matrix = np.column_stack([xx_t, predictions])
        valid = _valid_rows(all_matrix)
        not_nas[t] = np.where(valid)[0]
        all_clean = all_matrix[valid]
        x_cov = all_clean[:, :-1]
        y_clean = all_clean[:, -1]
        covariates_nonna[t] = x_cov

        treat_cols = treatment_matrix[not_nas[t], : t + 1]

        if regularization:
            x_resid, y_resid = _residualize(x_cov, y_clean, treat_cols)
            intercept, coefs_cov, _ = _fit_model(x_resid, y_resid, True, nfolds)

            x_full = np.column_stack([x_cov, treat_cols])
            intercept_full, coefs_full, model_full = _fit_model(x_full, y_clean, False, nfolds)
            treat_coefs = coefs_full[x_cov.shape[1] :]
        else:
            x_full = np.column_stack([x_cov, treat_cols])
            intercept_full, coefs_full, model_full = _fit_model(x_full, y_clean, False, nfolds)
            intercept = intercept_full
            coefs_cov = coefs_full[: x_cov.shape[1]]
            treat_coefs = coefs_full[x_cov.shape[1] :]

        model_effect[t] = float(treat_coefs[-1])
        coef_t[t] = np.concatenate([[intercept], coefs_cov])

        xx_pred = xx_t.copy()
        if dim_fe > 0 and t == n_periods - 1:
            xx_pred[:, -dim_fe:] = 0.0

        if t > 0:
            pred_matrix = np.column_stack([xx_pred, treatment_matrix[:, :t], np.full(n_units, ds[t])])
        else:
            pred_matrix = np.column_stack([xx_pred, np.full(n_units, ds[0])])

        valid_pred = _valid_rows(pred_matrix)
        predictions = np.full(n_units, np.nan)
        predictions[valid_pred] = model_full.predict(pred_matrix[valid_pred])

        ds_tiled = np.tile(ds[: t + 1], (len(not_nas[t]), 1))
        pred_t[t] = model_full.predict(np.column_stack([x_cov, ds_tiled]))

    return CoefficientResult(
        coef_t=coef_t,
        pred_t=pred_t,
        covariates_nonna=covariates_nonna,
        not_nas=not_nas,
        model_effect=model_effect,
    )


def _fit_model(x, y, regularization, nfolds):
    """Fit LASSO or ridge and return intercept, coefficients, and model."""
    if regularization:
        model = LassoCV(cv=min(nfolds, x.shape[0]), max_iter=10_000)
    else:
        model = Ridge(alpha=np.exp(-8), fit_intercept=True)
    model.fit(x, y)
    return model.intercept_, model.coef_, model


def _residualize(x_cov, y, x_treat):
    """Partial out treatment columns via OLS (Frisch-Waugh-Lovell)."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(x_treat.shape[0]), x_treat]))
    y_resid = y - q @ (q.T @ y)
    x_resid = x_cov - q @ (q.T @ x_cov)
    return x_resid, y_resid


def _valid_rows(*arrays):
    """Return boolean mask of rows with no NaN across all arrays."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        if arr.ndim == 1:
            mask &= ~np.isnan(arr)
        else:
            mask &= ~np.isnan(arr).any(axis=1)
    return mask
