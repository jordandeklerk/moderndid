"""Single-pass cross-validated lasso with out-of-fold predictions."""

import warnings

import numpy as np
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold


def cv_lasso_with_oof(
    X,
    y,
    *,
    k_folds,
    random_state=None,
    n_alphas=100,
    eps=1e-3,
    penalty_factor=None,
    standardize=True,
    lambda_choice="lambda.min",
    max_iter=10_000,
):
    r"""Cross-validated lasso returning out-of-fold predictions and the chosen alpha.

    A single shared alpha grid is built from the full design, every fold
    traces the entire path with warm starts, and the out-of-fold predictions
    stored against that path come from the same fold partition that selects
    the optimal :math:`\lambda`. The final coefficient vector is refit on the
    full data at the selected alpha so it is consistent with the
    out-of-fold predictions.

    Per-column relative penalty weights use a column-scaling identity.
    Dividing column :math:`j` by ``penalty_factor[j]`` before fitting and
    dividing the resulting coefficient by the same factor leaves the loss
    unchanged but rescales the implicit per-coefficient penalty.
    Zero-penalty columns are partialled out via Frisch-Waugh-Lovell, where
    both ``y`` and the penalized columns of ``X`` are residualized against
    ``[1, X_unp]`` so the lasso on the residualized problem reproduces the
    joint KKT solution of the partially-penalized problem.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,)
        Response vector.
    k_folds : int
        Number of CV folds.
    random_state : int, Generator, or None, default=None
        Controls fold-shuffle randomness.
    n_alphas : int, default=100
        Number of alphas in the path.
    eps : float, default=1e-3
        Path resolution as the ratio of smallest to largest alpha.
    penalty_factor : ndarray of shape (p,) or None, default=None
        Per-column relative penalty weights. None means unit weights for
        every column. A penalty of 0 forces a column to be unpenalized.
    standardize : bool, default=True
        If True, columns of ``X`` are standardized to unit variance before
        fitting. Coefficients are returned on the original scale.
    lambda_choice : {"lambda.min", "lambda.1se"}, default="lambda.min"
        Selection rule for the alpha along the path.
    max_iter : int, default=10000
        Iteration cap for the inner Lasso solver.

    Returns
    -------
    dict
        Dictionary containing the cross-validated lasso outputs:

        - **oof_predictions**: Length-:math:`n` array of out-of-fold
          predictions at the selected alpha
        - **selected_alpha**: Alpha chosen by the ``lambda_choice`` rule
        - **coef**: Length-:math:`p` array of coefficients refit on the full
          data at ``selected_alpha``, on the original (unstandardized) scale
        - **intercept**: Intercept refit on the original scale
        - **alphas**: Array holding the full alpha path
        - **cv_errors**: Array of mean squared CV errors, one per alpha in
          the path
    """
    if lambda_choice not in ("lambda.min", "lambda.1se"):
        raise ValueError(f"lambda_choice must be 'lambda.min' or 'lambda.1se', got {lambda_choice!r}.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y length {y.shape} does not match X row count {n}.")
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}.")
    if k_folds > n:
        raise ValueError(f"k_folds={k_folds} exceeds sample size n={n}.")

    if penalty_factor is None:
        penalty_factor = np.ones(p, dtype=float)
    else:
        penalty_factor = np.asarray(penalty_factor, dtype=float)
        if penalty_factor.shape != (p,):
            raise ValueError(f"penalty_factor length {penalty_factor.shape} does not match p={p}.")
        if np.any(penalty_factor < 0):
            raise ValueError("penalty_factor entries must be non-negative.")

    col_std = X.std(axis=0, ddof=0)
    zero_var_mask = col_std == 0

    if zero_var_mask.any():
        warnings.warn(
            f"Detected {int(zero_var_mask.sum())} zero-variance column(s); their coefficients are forced to 0.",
            RuntimeWarning,
            stacklevel=2,
        )

    if standardize:
        col_std_safe = np.where(zero_var_mask, 1.0, col_std)
        X_scaled = X / col_std_safe[None, :]
    else:
        col_std_safe = np.ones(p, dtype=float)
        X_scaled = X.copy()

    # Force zero-variance columns out of the model entirely. Drop them from
    # the unpenalized OLS (where they are collinear with the intercept) and
    # from the lasso path (where they would otherwise be absorbed into the
    # intercept)
    active_mask = ~zero_var_mask
    unpenalized_mask = (penalty_factor == 0) & active_mask
    penalized_mask = (penalty_factor > 0) & active_mask
    pen_idx = np.flatnonzero(penalized_mask)
    unp_idx = np.flatnonzero(unpenalized_mask)

    pf_pen = penalty_factor[pen_idx]
    X_pen_scaled_full = X_scaled[:, pen_idx] / pf_pen[None, :] if pen_idx.size > 0 else X_scaled[:, pen_idx]

    # Residualize both the response and the penalized columns against the
    # intercept and the unpenalized columns so the alpha grid is built on the
    # same residualized problem the lasso will see, matching the joint KKT
    # solution rather than treating the two column blocks as orthogonal
    y_for_grid, X_pen_for_grid = _residualize_for_grid(X_scaled, y, X_pen_scaled_full, unp_idx)

    if pen_idx.size > 0:
        alphas = _build_alpha_grid(X_pen_for_grid, y_for_grid, n_alphas=n_alphas, eps=eps)
    else:
        alphas = np.array([0.0])

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_splits = list(kf.split(X_scaled))

    oof_preval = np.full((n, alphas.size), np.nan)
    fold_mse = np.full((k_folds, alphas.size), np.nan)
    fold_converged = np.ones(k_folds, dtype=bool)

    for k, (train_idx, test_idx) in enumerate(fold_splits):
        X_train_scaled = X_scaled[train_idx]
        y_train = y[train_idx]
        X_test_scaled = X_scaled[test_idx]
        y_test = y[test_idx]

        if pen_idx.size > 0:
            X_pen_train_scaled = X_train_scaled[:, pen_idx] / pf_pen[None, :]
            X_pen_test_scaled = X_test_scaled[:, pen_idx] / pf_pen[None, :]

            # Apply Frisch-Waugh-Lovell to project both the response and the
            # penalized columns out of the unpenalized columns and intercept
            # on the training fold so the lasso sees the partialled-out
            # problem implied by the joint KKT solution. Residualizing only
            # the response is correct when the unpenalized and penalized
            # columns are orthogonal but biases coefficients otherwise
            y_train_res, X_pen_train_res = _residualize_for_grid(X_train_scaled, y_train, X_pen_train_scaled, unp_idx)

            # Pre-center BOTH X and y so lasso_path with no intercept matches
            # what Lasso(fit_intercept=True) would produce internally
            train_mean_X = X_pen_train_res.mean(axis=0)
            train_mean_y = y_train_res.mean()

            X_pen_train_centered = X_pen_train_res - train_mean_X[None, :]
            y_train_centered = y_train_res - train_mean_y

            try:
                _, coefs_path, _ = lasso_path(
                    X_pen_train_centered,
                    y_train_centered,
                    alphas=alphas,
                    max_iter=max_iter,
                )
            except FloatingPointError:
                fold_converged[k] = False
                continue

            # Predictions must be formed on the raw test columns rather than
            # residualized ones, since the residualized coordinates are only
            # defined relative to the training-fold projection
            y_train_post = y_train[:, None] - X_pen_train_scaled @ coefs_path
            beta_unp_path, intercept_unp_path = _recover_unpenalized_path(X_train_scaled, y_train_post, unp_idx)
            unp_predictions_test = _predict_unpenalized_path(X_test_scaled, unp_idx, beta_unp_path, intercept_unp_path)
            pen_predictions_test = X_pen_test_scaled @ coefs_path
            preds_path = unp_predictions_test + pen_predictions_test
        else:
            beta_unp_train, _, intercept_unp_train = _fit_unpenalized(X_train_scaled, y_train, unp_idx)
            path_predictions_test = _predict_unpenalized(X_test_scaled, unp_idx, beta_unp_train, intercept_unp_train)
            preds_path = np.broadcast_to(path_predictions_test[:, None], (test_idx.size, alphas.size)).copy()

        oof_preval[test_idx, :] = preds_path
        fold_mse[k, :] = np.mean((y_test[:, None] - preds_path) ** 2, axis=0)

    converged_cols = ~np.isnan(oof_preval).any(axis=0)
    cv_errors = np.where(converged_cols, np.nanmean(fold_mse, axis=0), np.inf)

    if not np.isfinite(cv_errors).any():
        raise RuntimeError("Lasso path failed to converge for every alpha in every fold.")

    min_idx = int(np.argmin(cv_errors))

    if lambda_choice == "lambda.min":
        chosen_idx = min_idx
    else:
        se = np.full(alphas.size, np.inf)
        finite_cols = np.isfinite(cv_errors)
        n_converged = int(fold_converged.sum())
        se[finite_cols] = np.std(fold_mse[np.ix_(fold_converged, finite_cols)], axis=0, ddof=1) / np.sqrt(n_converged)

        threshold = cv_errors[min_idx] + se[min_idx]

        # Largest alpha with CV error within one SE of the minimum
        candidate_mask = (cv_errors <= threshold) & finite_cols
        chosen_idx = int(np.argmax(np.where(candidate_mask, alphas, -np.inf)))

    selected_alpha = float(alphas[chosen_idx])

    coef_full_scaled, intercept_scaled = _refit_full(
        X_scaled,
        y,
        unp_idx=unp_idx,
        pen_idx=pen_idx,
        pf_pen=pf_pen,
        alpha=selected_alpha,
        max_iter=max_iter,
    )

    # Convert from standardized scale back to raw scale. Standardization is
    # column-wise division by ``col_std`` with no centering, so the
    # intercept passes through unchanged and only the coefficients rescale
    # by ``col_std``
    coef = coef_full_scaled / col_std_safe

    return {
        "oof_predictions": oof_preval[:, chosen_idx],
        "selected_alpha": selected_alpha,
        "coef": coef,
        "intercept": float(intercept_scaled),
        "alphas": alphas,
        "cv_errors": cv_errors,
    }


def _build_alpha_grid(X, y, *, n_alphas, eps):
    """Build a geometric alpha grid from ``alpha_max`` down to ``eps * alpha_max``."""
    n = len(y)
    y_centered = y - y.mean()
    alpha_max = float(np.max(np.abs(X.T @ y_centered))) / n

    # When the response is orthogonal to every column by construction
    # (e.g., a balanced post-period indicator with time-invariant
    # covariates), alpha_max is pure floating-point cancellation noise
    # and a path built from it makes the solver grind on a problem whose
    # exact solution is the null model. Detect that case against the
    # rounding-error scale of the inner products and pin the grid to a
    # single alpha at which the all-zero solution holds on any subsample
    x_extreme = float(np.max(np.abs(X))) if X.size else 0.0
    y_extreme = float(np.max(np.abs(y_centered))) if y_centered.size else 0.0
    noise_floor = np.finfo(float).eps * x_extreme * y_extreme
    if alpha_max <= noise_floor:
        return np.array([max(1.0, 4.0 * x_extreme * y_extreme)])

    return np.geomspace(alpha_max, alpha_max * eps, n_alphas)


def _residualize_for_grid(X_scaled, y, X_pen_scaled, unp_idx):
    """Project response and penalized columns onto the orthogonal complement of the unpenalized columns."""
    if unp_idx.size == 0:
        return y, X_pen_scaled
    Z = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled[:, unp_idx]])
    coef_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    y_res = y - Z @ coef_y
    if X_pen_scaled.shape[1] == 0:
        return y_res, X_pen_scaled
    coef_X, *_ = np.linalg.lstsq(Z, X_pen_scaled, rcond=None)
    X_pen_res = X_pen_scaled - Z @ coef_X
    return y_res, X_pen_res


def _fit_unpenalized(X_scaled, y, unp_idx):
    """Fit OLS on unpenalized columns; return slope coefficients, residual y, and intercept."""
    if unp_idx.size == 0:
        return np.zeros(0), y, 0.0
    Z = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled[:, unp_idx]])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    fitted = Z @ coef
    return coef[1:], y - fitted, float(coef[0])


def _recover_unpenalized_path(X_scaled, y_post_path, unp_idx):
    """Regress each column of the post-lasso residual on the unpenalized columns and intercept."""
    n_alphas = y_post_path.shape[1]
    if unp_idx.size == 0:
        intercept_path = y_post_path.mean(axis=0)
        return np.zeros((0, n_alphas)), intercept_path
    Z = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled[:, unp_idx]])
    coef_path, *_ = np.linalg.lstsq(Z, y_post_path, rcond=None)
    return coef_path[1:, :], coef_path[0, :]


def _predict_unpenalized(X_scaled, unp_idx, beta_unp, intercept):
    """Apply unpenalized coefficients to a held-out block of X."""
    if unp_idx.size == 0:
        return np.full(X_scaled.shape[0], intercept)
    return intercept + X_scaled[:, unp_idx] @ beta_unp


def _predict_unpenalized_path(X_scaled, unp_idx, beta_unp_path, intercept_path):
    """Apply path-shaped unpenalized coefficients to a held-out block of X."""
    n_test = X_scaled.shape[0]
    if unp_idx.size == 0:
        return np.broadcast_to(intercept_path[None, :], (n_test, intercept_path.size)).copy()
    return X_scaled[:, unp_idx] @ beta_unp_path + intercept_path[None, :]


def _refit_full(X_scaled, y, *, unp_idx, pen_idx, pf_pen, alpha, max_iter):
    """Refit the full model at the chosen alpha on standardized X, returning coefs in scaled space."""
    p = X_scaled.shape[1]
    coef = np.zeros(p)

    if pen_idx.size == 0:
        beta_unp, _, intercept_unp = _fit_unpenalized(X_scaled, y, unp_idx)
        if unp_idx.size > 0:
            coef[unp_idx] = beta_unp
        return coef, intercept_unp

    X_pen_scaled = X_scaled[:, pen_idx] / pf_pen[None, :]
    y_res, X_pen_res = _residualize_for_grid(X_scaled, y, X_pen_scaled, unp_idx)
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter).fit(X_pen_res, y_res)
    coef_pen_in_res = np.asarray(model.coef_, dtype=float)
    coef_pen_scaled = coef_pen_in_res / pf_pen
    coef[pen_idx] = coef_pen_scaled

    y_post = y - X_pen_scaled @ coef_pen_in_res
    beta_unp, _, intercept_unp = _fit_unpenalized(X_scaled, y_post, unp_idx)
    if unp_idx.size > 0:
        coef[unp_idx] = beta_unp
    return coef, intercept_unp
