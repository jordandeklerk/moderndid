"""Cross-fitted R-learner of conditional average treatment effects."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold


def fit_rlearner(
    X,
    Y,
    treatment,
    *,
    k_folds=10,
    tune_penalty=False,
    penalty_factor=None,
    lambda_choice="lambda.min",
    random_state=None,
):
    r"""Fit a cross-fitted R-learner of the conditional average treatment effect.

    Estimates :math:`\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]` from
    a binary treatment indicator using the residualization method of [1]_:

    1. Cross-fit nuisances :math:`\hat{m}(x) = \mathbb{E}[Y \mid X = x]` and
       :math:`\hat{e}(x) = \mathbb{E}[D \mid X = x]` via k-fold lasso
       regression. Each fold is fit by an inner lasso with cross-validated
       :math:`\lambda` selection.
    2. Form residuals :math:`\tilde{Y} = Y - \hat{m}(X)` and
       :math:`\tilde{D} = D - \hat{e}(X)`.
    3. Solve the residualized lasso problem

       .. math::

           \hat{\beta} = \operatorname*{argmin}_{\beta}
           \frac{1}{n} \sum_{i=1}^{n}
               \bigl(\tilde{Y}_i - \tilde{D}_i \cdot [1, X_i]^\top \beta\bigr)^2
           + \lambda \sum_{j \ge 1} w_j \, |\beta_j|

       with the intercept-equivalent coefficient :math:`\beta_0` left
       unpenalized and per-coefficient penalty factors :math:`w_j` for
       :math:`j \ge 1`.
    4. Predict :math:`\hat{\tau}(X_i) = [1, X_i]^\top \hat{\beta}` at each
       training row.

    With ``tune_penalty=True`` the function performs an outer k-fold grid
    search over a scalar penalty factor in
    :math:`\{0.01, 0.25, 0.5, 0.75, 0.99, 1\}`, evaluating each candidate
    by the mean squared error of the held-out :math:`\tau` predictions
    against the held-out outcomes, then refits on the full sample with the
    minimizing value.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate design matrix without the intercept column.
    Y : ndarray of shape (n,)
        Outcome variable.
    treatment : ndarray of shape (n,)
        Binary 0/1 treatment indicator.
    k_folds : int, default=10
        Number of folds for outer cross-fitting and inner cross-validation.
    tune_penalty : bool, default=False
        If True, grid-search a single penalty-factor scalar over
        :math:`\{0.01, 0.25, 0.5, 0.75, 0.99, 1\}` by outer k-fold cross
        validation, then refit on the full sample with the best value.
    penalty_factor : ndarray of shape (p,), optional
        Per-coefficient L1 penalty multipliers for the residual lasso.
        Ignored when ``tune_penalty=True``. Defaults to a vector of ones.
    lambda_choice : {'lambda.min', 'lambda.1se'}, default='lambda.min'
        Cross-validation rule for selecting the lasso :math:`\lambda` in the
        final residual fit. ``'lambda.1se'`` returns the largest
        :math:`\lambda` whose mean CV MSE is within one standard error of
        the minimum.
    random_state : int, optional
        Seed controlling the cross-fitting fold splits and inner CV.

    Returns
    -------
    dict
        Dictionary with keys ``tau_hat`` (length-:math:`n` ndarray of
        cross-fitted CATE predictions), ``p_hat`` (length-:math:`n`
        cross-fitted propensity), ``m_hat`` (length-:math:`n` cross-fitted
        outcome mean), ``tau_coef`` (length-:math:`p+1` ndarray with
        intercept first), and ``best_penalty_factor`` (scalar; equal to
        ``penalty_factor[0]`` when tuning is off).

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes, ``k_folds`` is too small, or
        ``lambda_choice`` is invalid.

    References
    ----------

    .. [1] Nie, X., & Wager, S. (2021). "Quasi-oracle estimation of
           heterogeneous treatment effects." Biometrika, 108(2), 299-319.
           https://doi.org/10.1093/biomet/asaa076
    """
    if lambda_choice not in ("lambda.min", "lambda.1se"):
        raise ValueError(f"lambda_choice must be 'lambda.min' or 'lambda.1se', got {lambda_choice!r}.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    n, p = X.shape
    if Y.shape != (n,):
        raise ValueError(f"Y length {Y.shape} does not match X row count {n}.")
    if treatment.shape != (n,):
        raise ValueError(f"treatment length {treatment.shape} does not match X row count {n}.")
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
        if np.any(penalty_factor <= 0):
            raise ValueError("penalty_factor entries must be strictly positive.")

    best_pf_scalar = float(penalty_factor[0]) if not tune_penalty else None

    if tune_penalty:
        candidate_factors = (0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
        best_pf_scalar = _select_penalty_factor(
            X=X,
            Y=Y,
            treatment=treatment,
            candidates=candidate_factors,
            k_folds=k_folds,
            lambda_choice=lambda_choice,
            random_state=random_state,
        )
        penalty_factor = np.full(p, best_pf_scalar, dtype=float)

    return _fit_rlasso(
        X=X,
        Y=Y,
        treatment=treatment,
        penalty_factor=penalty_factor,
        k_folds=k_folds,
        lambda_choice=lambda_choice,
        random_state=random_state,
        best_penalty_factor=best_pf_scalar,
    )


def _fit_rlasso(
    *,
    X,
    Y,
    treatment,
    penalty_factor,
    k_folds,
    lambda_choice,
    random_state,
    best_penalty_factor,
):
    """Single R-learner fit at the given penalty factor."""
    n = X.shape[0]

    m_hat = _cross_fit_lasso(X, Y, k_folds=k_folds, random_state=random_state)
    p_hat = _cross_fit_lasso(X, treatment, k_folds=k_folds, random_state=random_state)

    y_tilde = Y - m_hat
    w_tilde = treatment - p_hat

    w_inner = float(w_tilde @ w_tilde)
    if w_inner < 1e-12:
        raise RuntimeError(
            "Residualized treatment w_tilde is degenerate (w_tilde'w_tilde ≈ 0); "
            "the propensity model perfectly explains the treatment indicator."
        )

    interaction_cols = w_tilde[:, None] * X
    alpha_y = float(w_tilde @ y_tilde) / w_inner
    alpha_z = (w_tilde @ interaction_cols) / w_inner
    y_resid = y_tilde - alpha_y * w_tilde
    z_resid = interaction_cols - w_tilde[:, None] * alpha_z[None, :]

    z_scaled = z_resid / penalty_factor[None, :]
    beta_scaled = _fit_lasso_with_lambda_choice(
        z_scaled,
        y_resid,
        k_folds=k_folds,
        lambda_choice=lambda_choice,
        random_state=random_state,
    )
    beta_penalized = beta_scaled / penalty_factor
    beta_intercept = alpha_y - float(alpha_z @ beta_penalized)

    tau_coef = np.concatenate([[beta_intercept], beta_penalized])
    design = np.column_stack([np.ones(n), X])
    tau_hat = design @ tau_coef

    return {
        "tau_hat": tau_hat,
        "p_hat": p_hat,
        "m_hat": m_hat,
        "tau_coef": tau_coef,
        "best_penalty_factor": best_penalty_factor,
    }


def _select_penalty_factor(
    *,
    X,
    Y,
    treatment,
    candidates,
    k_folds,
    lambda_choice,
    random_state,
):
    """Outer k-fold grid search picking the best penalty-factor scalar."""
    p = X.shape[1]
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_indices = list(kf.split(X))

    cv_errors = np.empty(len(candidates))
    for i, pf_scalar in enumerate(candidates):
        fold_errors = np.empty(k_folds)
        pf_train = np.full(p, float(pf_scalar))
        for k, (train_idx, test_idx) in enumerate(fold_indices):
            fit_kwargs = dict(
                X=X[train_idx],
                Y=Y[train_idx],
                treatment=treatment[train_idx],
                penalty_factor=pf_train,
                k_folds=min(k_folds, len(train_idx) - 1),
                lambda_choice=lambda_choice,
                random_state=random_state,
                best_penalty_factor=float(pf_scalar),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fit = _fit_rlasso(**fit_kwargs)
            design_test = np.column_stack([np.ones(len(test_idx)), X[test_idx]])
            tau_pred = design_test @ fit["tau_coef"]
            fold_errors[k] = float(np.mean((Y[test_idx] - tau_pred) ** 2))
        cv_errors[i] = float(np.mean(fold_errors))

    return float(candidates[int(np.argmin(cv_errors))])


def _cross_fit_lasso(X, y, *, k_folds, random_state):
    """Return out-of-fold lasso predictions for ``y`` on ``X``."""
    n = len(y)
    y_hat = np.empty(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        inner_cv = max(2, min(k_folds, len(train_idx) - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = LassoCV(cv=inner_cv, alphas=100, random_state=random_state).fit(X[train_idx], y[train_idx])
        y_hat[test_idx] = model.predict(X[test_idx])
    return y_hat


def _fit_lasso_with_lambda_choice(features, y, *, k_folds, lambda_choice, random_state):
    r"""Fit LassoCV and return coefficients at the chosen :math:`\lambda`."""
    inner_cv = max(2, min(k_folds, features.shape[0] - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LassoCV(
            cv=inner_cv,
            alphas=100,
            random_state=random_state,
            fit_intercept=False,
        ).fit(features, y)

    if lambda_choice == "lambda.min":
        return np.asarray(model.coef_, dtype=float)

    alphas = np.asarray(model.alphas_)
    mse_path = np.asarray(model.mse_path_)
    mean_mse = mse_path.mean(axis=1)
    se_mse = mse_path.std(axis=1, ddof=1) / np.sqrt(mse_path.shape[1])
    min_idx = int(np.argmin(mean_mse))
    threshold = mean_mse[min_idx] + se_mse[min_idx]
    candidate_idx = np.where(mean_mse <= threshold)[0]
    chosen_alpha = float(alphas[candidate_idx].max())

    return _refit_lasso_at_alpha(features, y, chosen_alpha)


def _refit_lasso_at_alpha(features, y, alpha):
    """Refit a plain Lasso at the supplied ``alpha`` and return its coefficients."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10_000).fit(features, y)
    return np.asarray(model.coef_, dtype=float)
