"""Delta nuisance backend for the doubly robust ML DiD score."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .cv_lasso import cv_lasso_with_oof


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
    random_state=None,
):
    r"""Fit the conditional covariance (delta) nuisance model.

    The doubly robust ML score requires an estimate of the conditional
    covariance of the post-period indicator :math:`T` and the cohort
    indicator :math:`G`,

    .. math::

        \Delta(x) = \mathbb{E}\bigl[(T - t(X))(G - g(X)) \mid X = x\bigr],

    where :math:`t(x) = \mathbb{E}[T \mid X = x]` and
    :math:`g(x) = \mathbb{E}[G \mid X = x]` are the post-period and cohort
    propensities. The function takes the pointwise product of the
    indicators residualized at their estimated propensities

    .. math::

        \text{response}_i = (T_i - \hat{t}(X_i))(G_i - \hat{g}(X_i))

    and regresses it on the covariates. The result holds the cross-fitted
    predictions :math:`\hat{\Delta}(X_i)`.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate design matrix without the intercept column.
    response : ndarray of shape (n,)
        Pointwise product of residualized post-period and cohort indicators.
    model : {'glm', 'stack'}, default='glm'
        Backend to use. ``'glm'`` fits a cross-validated lasso on
        standardized covariates via :func:`cv_lasso_with_oof`. Out-of-fold
        predictions at the chosen :math:`\lambda` come directly from the
        same fold partition that selects it. ``'stack'`` fits a stacking
        ensemble of lasso, random forest, and gradient boosting base
        learners combined by a non-negative least-squares meta-learner.
    k_folds : int, default=10
        Number of folds for cross-fitting.
    random_state : int, optional
        Seed controlling fold splits and the inner CV of the base estimators.

    Returns
    -------
    dict
        Dictionary containing delta-fit outputs:

        - **delta_hat**: Length-:math:`n` array of cross-fitted predictions
          :math:`\hat{\Delta}(X_i)`
        - **best_l1_ratio**: Always ``1.0`` for the ``'glm'`` backend (lasso),
          ``None`` for the ``'stack'`` backend
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
        return _fit_delta_glm(X, response, k_folds=k_folds, random_state=random_state)
    return _fit_delta_stack(X, response, k_folds=k_folds, random_state=random_state)


def _fit_delta_glm(X, y, *, k_folds, random_state):
    """Fit the delta nuisance by single-pass CV lasso on standardized covariates."""
    scaler = StandardScaler().fit(X)
    fit = cv_lasso_with_oof(
        scaler.transform(X),
        y,
        k_folds=k_folds,
        random_state=random_state,
        standardize=False,
        lambda_choice="lambda.min",
    )

    return {
        "delta_hat": np.asarray(fit["oof_predictions"], dtype=float),
        "best_l1_ratio": 1.0,
    }


def _fit_delta_stack(X, y, *, k_folds, random_state):
    """Fit a super learner stacking ensemble with a non-negative least-squares meta-learner."""
    p = X.shape[1]
    base_estimators = [
        ("lasso", LassoCV(cv=10, alphas=100, max_iter=10_000, random_state=random_state)),
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
    }
