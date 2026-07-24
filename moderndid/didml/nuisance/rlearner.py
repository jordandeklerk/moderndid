"""Cross-fitted R-learner of conditional average treatment effects."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .cv_lasso import cv_lasso_with_oof


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

    Estimates the conditional mean contrast

    .. math::

        \tau(x) = \mathbb{E}[Y \mid X = x, D = 1]
                - \mathbb{E}[Y \mid X = x, D = 0],

    where :math:`D` is whichever binary indicator is passed as
    ``treatment``, either the post-period indicator or the cohort
    indicator, using the residualization method of [1]_.

    Both the outcome and the indicator are first residualized against the
    covariates with cross-fitted penalized nuisances, a squared-error
    lasso for the continuous outcome and an L1-penalized logistic
    regression for the binary indicator. A penalized regression of the
    outcome residual on the indicator residual interacted with the
    covariates then recovers a linear model for :math:`\tau(x)`.

    With ``tune_penalty=True``, an outer k-fold grid search selects the
    scalar penalty factor for the residual lasso before the final fit,
    using the candidate grid documented under ``tune_penalty``.

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
        Seed controlling the cross-fitting fold splits, the inner CV, and the
        penalized logistic fit, whose solver shuffles coordinates internally
        and so needs the same seed for the result to be fully reproducible.

    Returns
    -------
    dict
        Dictionary containing cross-fitted R-learner outputs:

        - **tau_hat**: Length-:math:`n` array of cross-fitted CATE predictions
          :math:`\hat{\tau}(X_i)` at each training row
        - **p_hat**: Length-:math:`n` array of cross-fitted propensity scores
          :math:`\hat{e}(X_i) = \mathbb{P}[D = 1 \mid X_i]`, fitted
          probabilities lying strictly in :math:`(0, 1)`
        - **m_hat**: Length-:math:`n` array of cross-fitted outcome predictions
          :math:`\hat{m}(X_i) = \mathbb{E}[Y \mid X_i]`
        - **tau_coef**: Length-:math:`(p+1)` array of CATE linear-model
          coefficients with intercept first
        - **best_penalty_factor**: Scalar penalty factor used in the residual
          lasso (equals ``penalty_factor[0]`` when ``tune_penalty=False``)

    Notes
    -----
    The estimator proceeds in four steps:

    1. Cross-fit nuisances :math:`\hat{m}(x) = \mathbb{E}[Y \mid X = x]` and
       :math:`\hat{e}(x) = \mathbb{P}[D = 1 \mid X = x]` over k folds, each
       fold predicting its held-out rows from a penalized fit on the
       remaining folds. The outcome nuisance is a squared-error lasso with
       cross-validated :math:`\lambda`; the indicator nuisance is an
       L1-penalized logistic regression whose :math:`\lambda` minimizes the
       cross-validated binomial deviance, so :math:`\hat{e}` comes back
       through the logistic link as a probability.
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

    The ``tune_penalty=True`` grid search evaluates each candidate penalty
    factor by the mean squared error of the held-out :math:`\tau`
    predictions against the held-out outcomes, then refits on the full
    sample with the minimizing value.

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

    scaler = StandardScaler().fit(X)
    X_scl = scaler.transform(X)

    m_hat = _cross_fit_lasso(X_scl, Y, k_folds=k_folds, random_state=random_state)
    p_hat = _cross_fit_logistic_lasso(X_scl, treatment, k_folds=k_folds, random_state=random_state)

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
            fit = _fit_rlasso(**fit_kwargs)

            design_test = np.column_stack([np.ones(len(test_idx)), X[test_idx]])
            tau_pred = design_test @ fit["tau_coef"]
            fold_errors[k] = float(np.mean((Y[test_idx] - tau_pred) ** 2))

        cv_errors[i] = float(np.mean(fold_errors))

    return float(candidates[int(np.argmin(cv_errors))])


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


def _cross_fit_logistic_lasso(X, y, *, k_folds, random_state):
    r"""Return out-of-fold logistic-lasso probabilities for binary ``y`` on standardized columns at min deviance."""
    n = X.shape[0]
    lambdas = _binomial_lambda_grid(X, y)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    oof_linear = np.empty((n, lambdas.size))
    fold_deviance = np.empty((k_folds, lambdas.size))

    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        y_train = y[train_idx]
        y_test = y[test_idx]

        for j, penalty in enumerate(lambdas):
            model = LogisticRegression(
                penalty="l1",
                C=1.0 / (train_idx.size * penalty),
                solver="liblinear",
                # The solver folds the intercept into the design as a constant
                # column that the L1 term would otherwise shrink; scaling that
                # column up drives its share of the penalty to nothing, which
                # is what the binomial lasso asks of an intercept
                intercept_scaling=1e4,
                max_iter=10_000,
                # The solver shuffles coordinates internally, so it needs the
                # seed as well for the fit to be reproducible
                random_state=random_state,
            ).fit(X[train_idx], y_train)
            oof_linear[test_idx, j] = model.decision_function(X[test_idx])

        # Evaluating the deviance on the linear predictor rather than on the
        # probability keeps the logarithms exact when a held-out fit is close
        # to certain about a row
        eta_test = oof_linear[test_idx, :]
        fold_deviance[k, :] = 2.0 * np.mean(np.logaddexp(0.0, eta_test) - y_test[:, None] * eta_test, axis=0)

    best_idx = int(np.argmin(fold_deviance.mean(axis=0)))

    return expit(oof_linear[:, best_idx])


def _binomial_lambda_grid(X, y):
    r"""Build the shared binomial :math:`\lambda` path anchored at the score of the intercept-only model."""
    y_centered = y - y.mean()
    lambda_max = float(np.max(np.abs(X.T @ y_centered))) / len(y)

    # When the indicator is orthogonal to every column by construction (a
    # balanced post-period indicator whose covariate rows repeat across the
    # base and current periods), lambda_max is pure floating-point
    # cancellation noise and an unpenalized path would fit that noise instead
    # of the intercept-only model that solves the problem exactly. Detect that
    # against the rounding-error scale of the score and pin the grid to a
    # single penalty above the largest attainable score, at which every slope
    # is zero on any subsample and the propensity is the observed frequency
    x_extreme = float(np.max(np.abs(X))) if X.size else 0.0
    y_extreme = float(np.max(np.abs(y_centered))) if y_centered.size else 0.0
    noise_floor = np.finfo(float).eps * x_extreme * y_extreme
    if lambda_max <= noise_floor:
        return np.array([max(1.0, 2.0 * x_extreme)])

    return np.geomspace(lambda_max, lambda_max * 1e-3, 100)


def _fit_lasso_with_lambda_choice(features, y, *, k_folds, lambda_choice, random_state):
    r"""Fit LassoCV and return coefficients at the chosen :math:`\lambda`."""
    inner_cv = max(2, min(k_folds, features.shape[0] - 1))
    model = LassoCV(
        cv=inner_cv,
        n_alphas=100,
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
    """Refit a plain Lasso at the supplied ``alpha`` and return its coefficients."""
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10_000).fit(features, y)
    return np.asarray(model.coef_, dtype=float)
