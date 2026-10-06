"""Per-period LASSO coefficient estimation for dynamic covariate balancing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV, Ridge, lasso_path
from sklearn.model_selection import KFold

from moderndid.diddynamic.container import CoefficientResult


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

    The ``lasso_subsample`` strategy fits only on units whose observed
    treatment history matches the target sequence ``ds`` up to each period.

    The ``lasso_plain`` strategy fits the linear model on all units with the
    treatment history appended as regressors. The ``lags`` most recent
    treatment indicators stay unpenalized. The covariates and any older
    treatment indicators are penalized after scaling each column to unit
    standard deviation.

    The ``lasso_plain`` penalty comes from ``nfolds``-fold cross-validation
    over contiguous folds. It is the largest penalty whose cross-validated
    error lies within one standard error of the minimum. The projections and
    the stored coefficients, intercept included, come from this penalized fit.

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
        Number of most recent treatment indicators that ``lasso_plain``
        leaves unpenalized. Defaults to ``n_periods``.
    dim_fe : int
        Number of fixed-effect dummy columns at the end of each
        covariate matrix, zeroed out in non-final period predictions.

    Returns
    -------
    CoefficientResult
        Per-period coefficient estimates and predictions.

        - **coef_t**: Coefficient vectors per period, each with shape
          ``(1 + p,)`` where the first element is the intercept
        - **pred_t**: Prediction vectors per period on the clean covariate matrix
        - **covariates_nonna**: Covariate matrices per period with NaN rows removed
        - **not_nas**: Integer arrays of valid row indices per period
        - **model_effect**: Last treatment coefficient per period (empty for
          ``lasso_subsample``)

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
        # Cross-validation must split the units on the history into at least two folds.
        if regularization and subsample_mask.sum() < 2:
            history = [int(d) for d in ds[: t + 1]]
            raise ValueError(
                f"Fewer than two units with observed data follow the treatment history {history} through period "
                f"{t + 1}. The cross-validated LASSO of that period needs at least two. Choose a treatment history "
                "that more units follow."
            )

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
    """Fit on all units with the treatment history as extra regressors."""
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

        x_full = np.column_stack([x_cov, treatment_matrix[not_nas[t], : t + 1]])
        model_full = None
        if regularization:
            penalized = np.arange(x_full.shape[1]) < x_full.shape[1] - min(t + 1, lags)
            intercept, coefs_full = _fit_penalized(x_full, y_clean, penalized, nfolds)
        else:
            intercept, coefs_full, model_full = _fit_model(x_full, y_clean, False, nfolds)

        model_effect[t] = float(coefs_full[-1])
        coef_t[t] = np.concatenate([[intercept], coefs_full[: x_cov.shape[1]]])

        xx_pred = xx_t.copy()
        if dim_fe > 0 and t == n_periods - 1:
            xx_pred[:, -dim_fe:] = 0.0

        if t > 0:
            pred_matrix = np.column_stack([xx_pred, treatment_matrix[:, :t], np.full(n_units, ds[t])])
        else:
            pred_matrix = np.column_stack([xx_pred, np.full(n_units, ds[0])])
        target_matrix = np.column_stack([x_cov, np.tile(ds[: t + 1], (len(not_nas[t]), 1))])

        valid_pred = _valid_rows(pred_matrix)
        predictions = np.full(n_units, np.nan)
        if model_full is None:
            predictions[valid_pred] = intercept + pred_matrix[valid_pred] @ coefs_full
            pred_t[t] = intercept + target_matrix @ coefs_full
        else:
            predictions[valid_pred] = model_full.predict(pred_matrix[valid_pred])
            pred_t[t] = model_full.predict(target_matrix)

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


def _fit_penalized(x, y, penalized, nfolds):
    r"""Fit a cross-validated LASSO that leaves some columns unpenalized.

    Each penalized column is scaled to unit standard deviation. The intercept
    and the unpenalized columns are partialled out exactly. Only the scaled
    columns carry the penalty.

    The penalty grid has 100 geometric steps. It starts at the smallest
    penalty that keeps every penalized slope at zero and ends at
    :math:`10^{-4}` times that value, or :math:`10^{-2}` times it when columns
    outnumber observations. The path stops early once the explained share of
    the outcome variance passes 0.999 or grows by less than a :math:`10^{-5}`
    fraction.

    Each contiguous fold fits its own penalty path on the remaining rows. The
    held-out rows are scored at every full-sample penalty by interpolating
    along that path. The chosen penalty is the largest one whose mean
    held-out error lies within one standard error of the minimum.

    Parameters
    ----------
    x : ndarray, shape (n, k)
        Regressors without an intercept column.
    y : ndarray, shape (n,)
        Outcome.
    penalized : ndarray of bool, shape (k,)
        True for the columns that carry the penalty.
    nfolds : int
        Number of cross-validation folds.

    Returns
    -------
    intercept : float
        Intercept of the penalized fit.
    coefs : ndarray, shape (k,)
        Slopes of all columns on their original scale.
    """
    n = x.shape[0]
    free = np.column_stack([np.ones(n), x[:, ~penalized]])
    x_pen = x[:, penalized]
    slopes = np.zeros(x_pen.shape[1])

    alphas, path = _penalty_path(x_pen, y, free, x.shape[1])
    if alphas is not None:
        errors, folds = _cv_errors(x, y, penalized, alphas, nfolds)
        slopes = path[:, _one_se_index(errors, folds, n)]

    unpenalized = _free_coefficients(free, y - x_pen @ slopes)
    coefs = np.empty(x.shape[1])
    coefs[penalized] = slopes
    coefs[~penalized] = unpenalized[1:]
    return float(unpenalized[0]), coefs


def _free_coefficients(free, target):
    """Return least-squares coefficients on the free columns in their given order."""
    keep = np.zeros(free.shape[1], dtype=bool)
    rank = 0
    for j in range(free.shape[1]):
        keep[j] = True
        new_rank = np.linalg.matrix_rank(free[:, keep])
        if new_rank > rank:
            rank = new_rank
        else:
            keep[j] = False
    # A column that earlier columns span has no separate effect in the sample. Its zero
    # coefficient leaves the joint effect with the earliest column.
    coefs = np.zeros((free.shape[1], *np.shape(target)[1:]))
    coefs[keep] = np.linalg.lstsq(free[:, keep], target, rcond=None)[0]
    return coefs


def _penalty_path(x_pen, y, free, n_columns):
    """Return the penalty grid and slopes of one sample up to where its path stops."""
    n = x_pen.shape[0]
    scale = x_pen.std(axis=0)
    active = scale > 0
    total = np.sum((y - y.mean()) ** 2)
    # A y that is constant up to rounding leaves nothing for the penalized columns to explain.
    if not active.any() or total <= 1e-20 * (y @ y):
        return None, None

    x_std, y_resid = _residualize(x_pen[:, active] / scale[active], y, free)
    alpha_max = np.max(np.abs(x_std.T @ y_resid)) / n
    # The same holds once the free columns fit y up to rounding.
    if alpha_max == 0 or y_resid @ y_resid <= 1e-20 * total:
        return None, None

    ratio = 1e-2 if n < n_columns else 1e-4
    alphas = alpha_max * ratio ** (np.arange(100) / 99)
    path = _lasso_slopes(x_pen, y, free, alphas)

    explained = 1 - np.sum((y_resid[:, None] - x_std @ (path[active] * scale[active, None])) ** 2, axis=0) / total
    for m in range(4, len(alphas)):
        if explained[m] - explained[m - 1] < 1e-5 * explained[m] or explained[m] > 0.999:
            return alphas[: m + 1], path[:, : m + 1]
    return alphas, path


def _lasso_slopes(x_pen, y, free, alphas):
    """Return original-scale LASSO slopes for each penalty after partialling out the free columns."""
    scale = x_pen.std(axis=0)
    active = scale > 0
    slopes = np.zeros((x_pen.shape[1], len(alphas)))
    if active.any():
        x_std, y_resid = _residualize(x_pen[:, active] / scale[active], y, free)
        resid_ss = y_resid @ y_resid
        if resid_ss > 0:
            # The precision target is relative to the total variation of y. A target relative to
            # what the free columns leave over is out of reach when they already fit y exactly.
            tol = 1e-10 * np.sum((y - y.mean()) ** 2) / resid_ss
            _, path, _ = lasso_path(x_std, y_resid, alphas=alphas, max_iter=100_000, tol=tol)
            slopes[active] = path / scale[active, None]
    return slopes


def _cv_errors(x, y, penalized, alphas, nfolds):
    """Return held-out squared errors for every penalty and the fold of each row.

    Each fold fits its own penalty path on the training rows. Its held-out
    predictions at the full-sample penalties come from linear interpolation
    along that path. Outside the path's range they stay at its end values.
    """
    n = x.shape[0]
    errors = np.empty((n, len(alphas)))
    folds = np.empty(n, dtype=int)
    for k, (train, test) in enumerate(KFold(n_splits=min(nfolds, n)).split(x)):
        free_train = np.column_stack([np.ones(len(train)), x[np.ix_(train, ~penalized)]])
        x_train = x[np.ix_(train, penalized)]
        fold_alphas, slopes = _penalty_path(x_train, y[train], free_train, x.shape[1])
        if fold_alphas is None:
            fold_alphas = alphas[:1]
            slopes = np.zeros((x_train.shape[1], 1))
        unpenalized = _free_coefficients(free_train, y[train, None] - x_train @ slopes)
        free_test = np.column_stack([np.ones(len(test)), x[np.ix_(test, ~penalized)]])
        fitted = free_test @ unpenalized + x[np.ix_(test, penalized)] @ slopes
        errors[test] = (y[test, None] - fitted @ _interpolation_weights(fold_alphas, alphas)) ** 2
        folds[test] = k
    return errors, folds


def _interpolation_weights(grid, targets):
    """Return weights that map values on a descending penalty grid to the target penalties."""
    weights = np.zeros((len(grid), len(targets)))
    for j, s in enumerate(targets):
        if s >= grid[0]:
            weights[0, j] = 1.0
        elif s <= grid[-1]:
            weights[-1, j] = 1.0
        else:
            i = np.flatnonzero(grid >= s)[-1]
            frac = (s - grid[i + 1]) / (grid[i] - grid[i + 1])
            weights[i, j] = frac
            weights[i + 1, j] = 1.0 - frac
    return weights


def _one_se_index(errors, folds, n):
    """Return the index of the largest penalty within one standard error of the best."""
    n_folds = folds.max() + 1
    # Folds this small give noisy fold means. The spread then comes from the rows.
    if n < 3 * n_folds:
        cv_mean = errors.mean(axis=0)
        cv_se = np.sqrt(np.mean((errors - cv_mean) ** 2, axis=0) / (n - 1))
    else:
        sizes = np.bincount(folds)
        fold_mean = np.vstack([errors[folds == k].mean(axis=0) for k in range(n_folds)])
        weights = sizes / sizes.sum()
        cv_mean = weights @ fold_mean
        cv_se = np.sqrt(weights @ (fold_mean - cv_mean) ** 2 / (n_folds - 1))
    best = np.flatnonzero(cv_mean <= cv_mean.min())[0]
    return int(np.flatnonzero(cv_mean <= cv_mean[best] + cv_se[best])[0])


def _residualize(x, y, free):
    """Remove the least-squares fit on the free columns from x and y."""
    stacked = np.column_stack([x, y])
    resid = stacked - free @ np.linalg.lstsq(free, stacked, rcond=None)[0]
    return resid[:, :-1], resid[:, -1]


def _valid_rows(*arrays):
    """Return boolean mask of rows with no NaN across all arrays."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        if arr.ndim == 1:
            mask &= ~np.isnan(arr)
        else:
            mask &= ~np.isnan(arr).any(axis=1)
    return mask
