"""Delta nuisance backend for the doubly-robust ML DiD score."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class _NNLSMetaLearner(RegressorMixin, BaseEstimator):
    """Non-negative least-squares meta-learner for the stacking ensemble."""

    _estimator_type = "regressor"

    def fit(self, X, y):
        r"""Solve :math:`\min_{w \ge 0} \|y - Xw\|^2` for the stacking weights."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        coef, _ = nnls(X, y)
        self.coef_ = coef
        self.intercept_ = 0.0
        return self

    def predict(self, X):
        """Return :math:`Xw` using the fitted non-negative weights."""
        return np.asarray(X, dtype=float) @ self.coef_


def fit_delta(
    X,
    response,
    *,
    model="glm",
    k_folds=10,
    tune_penalty=False,
    random_state=None,
):
    r"""Fit the cross-derivative (delta) nuisance model.

    The doubly-robust ML score requires an estimate of the conditional
    covariance of the residualized post-period indicator and the
    residualized cohort indicator,

    .. math::

        \Delta(x) = \mathbb{E}\bigl[(D_i - \hat{e}(X_i))(S_i - \hat{s}(X_i))
        \mid X_i = x\bigr].

    The function takes the pointwise product

    .. math::

        \text{response}_i = (D_i - \hat{e}(X_i))(S_i - \hat{s}(X_i))

    and regresses it on the covariates, returning cross-fitted predictions
    :math:`\hat{\Delta}(X_i)`.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate design matrix without the intercept column.
    response : ndarray of shape (n,)
        Pointwise product of residualized treatment and cohort indicators.
    model : {'glm', 'stack'}, default='glm'
        Backend to use. ``'glm'`` fits an elastic-net regression on
        standardized covariates with optional L1-ratio / penalty-factor
        grid search; cross-fitted predictions come from selecting the
        optimal :math:`\lambda` on full data and refitting per fold.
        ``'stack'`` fits a stacking ensemble with three base learners
        (:class:`~sklearn.linear_model.LassoCV`,
        :class:`~sklearn.ensemble.RandomForestRegressor` with
        ``n_estimators=1000`` and ``max_features=p/3``,
        :class:`xgboost.XGBRegressor` with ``n_estimators=1000``,
        ``max_depth=4``, ``learning_rate=0.1``) combined by a
        non-negative least-squares meta-learner; cross-fitted predictions
        come from :func:`~sklearn.model_selection.cross_val_predict`.
    k_folds : int, default=10
        Number of folds for cross-fitting.
    tune_penalty : bool, default=False
        For the ``'glm'`` backend, when True, grid-search over L1 ratios in
        :math:`\{0.01, 0.1, 0.25, 0.5, 0.75, 1\}` and per-coefficient penalty
        factors in :math:`\{0.01, 0.25, 0.5, 0.75, 0.99, 1\}` to find the
        combination minimizing CV error, then refit at the chosen L1 ratio
        on the full standardized design.
    random_state : int, optional
        Seed controlling fold splits and the inner CV of the base estimators.

    Returns
    -------
    dict
        Dictionary containing delta-fit outputs:

        - **delta_hat**: Length-:math:`n` array of cross-fitted predictions
          :math:`\hat{\Delta}(X_i)`
        - **best_l1_ratio**: Selected elastic-net L1 ratio (``None`` for the
          ``'stack'`` backend)
        - **best_penalty_factor**: Selected penalty factor scalar (``None``
          for the ``'stack'`` backend or when ``tune_penalty=False``)
    """
    if model not in ("glm", "stack"):
        raise ValueError(f"model must be 'glm' or 'stack', got {model!r}.")

    X = np.asarray(X, dtype=float)
    response = np.asarray(response, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    n, _ = X.shape
    if response.shape != (n,):
        raise ValueError(f"response length {response.shape} does not match X row count {n}.")
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}.")
    if k_folds > n:
        raise ValueError(f"k_folds={k_folds} exceeds sample size n={n}.")

    if model == "glm":
        return _fit_delta_glm(X, response, k_folds=k_folds, tune_penalty=tune_penalty, random_state=random_state)
    return _fit_delta_stack(X, response, k_folds=k_folds, random_state=random_state)


def _fit_delta_glm(X, y, *, k_folds, tune_penalty, random_state):
    """Elastic-net fit with optional L1-ratio / penalty-factor grid search."""
    n, p = X.shape
    scaler = StandardScaler().fit(X)
    X_scl = scaler.transform(X)

    if tune_penalty:
        l1_ratio_grid = (0.01, 0.1, 0.25, 0.5, 0.75, 1.0)
        pf_grid = (0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
        best_score = np.inf
        best_l1_ratio = 1.0
        best_pf = 1.0
        for l1_ratio in l1_ratio_grid:
            for pf in pf_grid:
                X_scaled = X_scl / pf
                fit = ElasticNetCV(
                    l1_ratio=l1_ratio,
                    cv=k_folds,
                    n_alphas=100,
                    max_iter=10_000,
                    random_state=random_state,
                ).fit(X_scaled, y)
                cv_score = float(fit.mse_path_.mean(axis=-1).min())
                if cv_score < best_score:
                    best_score = cv_score
                    best_l1_ratio = float(l1_ratio)
                    best_pf = float(pf)
        chosen_pf = np.ones(p)
    else:
        best_l1_ratio = 1.0
        best_pf = 1.0
        chosen_pf = np.ones(p)

    X_chosen = X_scl / chosen_pf[None, :]
    full = ElasticNetCV(
        l1_ratio=best_l1_ratio,
        cv=k_folds,
        n_alphas=100,
        max_iter=10_000,
        random_state=random_state,
    ).fit(X_chosen, y)
    optimal_alpha = float(full.alpha_)

    delta_hat = np.empty(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        fold_model = ElasticNet(
            alpha=optimal_alpha,
            l1_ratio=best_l1_ratio,
            max_iter=10_000,
        ).fit(X_chosen[train_idx], y[train_idx])
        delta_hat[test_idx] = fold_model.predict(X_chosen[test_idx])

    return {
        "delta_hat": delta_hat,
        "best_l1_ratio": best_l1_ratio,
        "best_penalty_factor": best_pf,
    }


def _fit_delta_stack(X, y, *, k_folds, random_state):
    """SuperLearner-style stacking ensemble with NNLS meta-learner."""
    p = X.shape[1]
    base_estimators = [
        ("lasso", LassoCV(cv=10, n_alphas=100, max_iter=10_000, random_state=random_state)),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=1000,
                max_features=max(1.0 / 3.0, 1.0 / max(p, 1)),
                min_samples_leaf=5,
                random_state=random_state,
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.1,
                min_child_weight=10,
                random_state=random_state,
                verbosity=0,
            ),
        ),
    ]

    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=_NNLSMetaLearner(),
        cv=k_folds,
        n_jobs=None,
    )
    delta_hat = cross_val_predict(stack, X, y, cv=k_folds)

    return {
        "delta_hat": np.asarray(delta_hat, dtype=float),
        "best_l1_ratio": None,
        "best_penalty_factor": None,
    }
