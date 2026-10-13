"""Cross-fitted R-learner of conditional average treatment effects."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from .cv_lasso import cv_lasso_with_oof


def fit_rlearner(
    X,
    Y,
    treatment,
    *,
    k_folds=10,
    penalty_factor=None,
    lambda_choice="lambda.min",
    random_state=None,
):
    r"""Fit a cross-fitted R-learner of the conditional average treatment effect.

    Estimates the conditional mean contrast

    .. math::

        \tau(x) = \mathbb{E}[Y \mid X = x, D = 1]
                - \mathbb{E}[Y \mid X = x, D = 0],

    where :math:`D` is the binary indicator passed as ``treatment``. The
    doubly robust score passes either the post-period indicator or the
    cohort indicator. The estimator follows the residualization method of
    [1]_.

    Both the outcome and the indicator are first residualized against the
    covariates with cross-fitted lasso nuisances. A penalized regression
    of the outcome residual on the indicator residual interacted with the
    covariates then recovers a linear model for :math:`\tau(x)`.

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
    penalty_factor : ndarray of shape (p,), optional
        Per-coefficient L1 penalty multipliers for the residual lasso.
        Defaults to a vector of ones.
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
        Dictionary containing cross-fitted R-learner outputs:

        - **tau_hat**: Length-:math:`n` array of cross-fitted CATE predictions
          :math:`\hat{\tau}(X_i)` at each training row
        - **p_hat**: Length-:math:`n` array of cross-fitted propensity scores
          :math:`\hat{e}(X_i) = \mathbb{E}[D \mid X_i]`
        - **m_hat**: Length-:math:`n` array of cross-fitted outcome predictions
          :math:`\hat{m}(X_i) = \mathbb{E}[Y \mid X_i]`
        - **tau_coef**: Length-:math:`(p+1)` array of CATE linear-model
          coefficients with intercept first

    Notes
    -----
    Cross-fitted lasso regressions estimate the nuisances
    :math:`\hat{m}(x) = \mathbb{E}[Y \mid X = x]` and
    :math:`\hat{e}(x) = \mathbb{E}[D \mid X = x]`. Their out-of-fold
    predictions come from the same k folds that choose :math:`\lambda`.

    The residuals :math:`\tilde{Y} = Y - \hat{m}(X)` and
    :math:`\tilde{D} = D - \hat{e}(X)` enter the residualized lasso problem

    .. math::

        \hat{\beta} = \operatorname*{argmin}_{\beta}
        \frac{1}{n} \sum_{i=1}^{n}
            \bigl(\tilde{Y}_i - \tilde{D}_i \cdot [1, X_i]^\top \beta\bigr)^2
        + \lambda \sum_{j \ge 1} w_j \, |\beta_j|.

    The intercept-equivalent coefficient :math:`\beta_0` stays unpenalized.
    The weights :math:`w_j` are the per-coefficient penalty factors for
    :math:`j \ge 1`. The prediction at each training row is
    :math:`\hat{\tau}(X_i) = [1, X_i]^\top \hat{\beta}`.

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

    return _fit_rlasso(
        X=X,
        Y=Y,
        treatment=treatment,
        penalty_factor=penalty_factor,
        k_folds=k_folds,
        lambda_choice=lambda_choice,
        random_state=random_state,
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
):
    """Single R-learner fit at the given penalty factors."""
    n = X.shape[0]

    scaler = StandardScaler().fit(X)
    X_scl = scaler.transform(X)

    m_hat = _cross_fit_lasso(X_scl, Y, k_folds=k_folds, random_state=random_state)
    p_hat = _cross_fit_lasso(X_scl, treatment, k_folds=k_folds, random_state=random_state)

    y_tilde = Y - m_hat
    w_tilde = treatment - p_hat

    w_inner = float(w_tilde @ w_tilde)
    if w_inner < 1e-12:
        raise RuntimeError(
            "Residualized treatment w_tilde is degenerate (w_tilde'w_tilde ≈ 0); "
            "the propensity model perfectly explains the treatment indicator."
        )

    interaction_cols = w_tilde[:, None] * X_scl
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

    beta_penalized_scl = beta_scaled / penalty_factor
    beta_intercept_scl = alpha_y - float(alpha_z @ beta_penalized_scl)

    sigma = scaler.scale_
    mu = scaler.mean_
    beta_raw_X = beta_penalized_scl / sigma
    beta_raw_intercept = beta_intercept_scl - float(beta_raw_X @ mu)

    tau_coef = np.concatenate([[beta_raw_intercept], beta_raw_X])
    design = np.column_stack([np.ones(n), X])
    tau_hat = design @ tau_coef

    return {
        "tau_hat": tau_hat,
        "p_hat": p_hat,
        "m_hat": m_hat,
        "tau_coef": tau_coef,
    }


def _cross_fit_lasso(X, y, *, k_folds, random_state):
    r"""Return out-of-fold lasso predictions for ``y`` on ``X`` at the CV-optimal :math:`\lambda`."""
    out = cv_lasso_with_oof(
        X,
        y,
        k_folds=k_folds,
        random_state=random_state,
        lambda_choice="lambda.min",
        standardize=False,
        max_iter=100_000,
    )
    return out["oof_predictions"]


def _fit_lasso_with_lambda_choice(features, y, *, k_folds, lambda_choice, random_state):
    r"""Fit a cross-validated lasso and return the coefficients at the chosen :math:`\lambda`."""
    inner_cv = max(2, min(k_folds, features.shape[0] - 1))
    model = LassoCV(
        cv=inner_cv,
        alphas=100,
        max_iter=10_000,
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
    """Refit a lasso at the supplied ``alpha`` and return its coefficients."""
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10_000).fit(features, y)
    return np.asarray(model.coef_, dtype=float)
