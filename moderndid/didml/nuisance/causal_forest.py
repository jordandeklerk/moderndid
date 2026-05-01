"""Causal-forest nuisance backend for the doubly-robust ML DiD score."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def fit_causal_forest(
    X,
    Y,
    treatment,
    *,
    n_estimators=2000,
    min_samples_leaf=5,
    max_depth=None,
    max_features="sqrt",
    cv=5,
    n_jobs=None,
    random_state=None,
):
    r"""Fit a causal-forest nuisance backend for the doubly-robust score.

    Estimates the conditional treatment effect
    :math:`\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]` using the
    generalized-random-forest DML estimator of [1]_, which internally:

    1. Cross-fits a regression-forest model for the outcome
       :math:`\hat{m}(x) = \mathbb{E}[Y \mid X = x]`.
    2. Cross-fits a regression-forest model for the treatment
       :math:`\hat{e}(x) = \mathbb{E}[D \mid X = x]`.
    3. Forms residuals :math:`\tilde{Y} = Y - \hat{m}(X)` and
       :math:`\tilde{D} = D - \hat{e}(X)`.
    4. Fits an honest causal forest on the residualized data, returning
       :math:`\hat{\tau}(X)` predictions at each training row.

    Returns the same dictionary shape as :func:`fit_rlearner` so the
    nuisance backend is interchangeable in the cross-derivative
    computation. The fields ``tau_coef`` and ``best_penalty_factor`` are
    set to ``None`` because the forest has no linear coefficients and no
    L1 penalty schedule to tune.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate design matrix without the intercept column.
    Y : ndarray of shape (n,)
        Outcome variable.
    treatment : ndarray of shape (n,)
        Binary 0/1 treatment indicator.
    n_estimators : int, default=2000
        Number of trees in each forest (causal forest and the two
        nuisance forests).
    min_samples_leaf : int, default=5
        Minimum number of training samples in each leaf of every forest.
    max_depth : int, optional
        Maximum tree depth. Defaults to unlimited.
    max_features : int, float, or {'sqrt', 'log2'}, default='sqrt'
        Number of features considered at each split.
    cv : int, default=5
        Number of folds used by the DML cross-fitting loop.
    n_jobs : int, optional
        Number of parallel workers for tree fitting.
    random_state : int, optional
        Seed controlling fold splits, bootstrap sampling, and the
        honest-split partitioning.

    Returns
    -------
    dict
        Dictionary containing causal-forest nuisance outputs:

        - **tau_hat**: Length-:math:`n` array of CATE predictions
          :math:`\hat{\tau}(X_i)` from the causal forest
        - **p_hat**: Length-:math:`n` array of cross-fitted propensities
          :math:`\hat{e}(X_i) = \mathbb{E}[D \mid X_i]`
        - **m_hat**: Length-:math:`n` array of cross-fitted outcome
          regressions :math:`\hat{m}(X_i) = \mathbb{E}[Y \mid X_i]`
        - **tau_coef**: ``None`` (forests have no linear coefficients)
        - **best_penalty_factor**: ``None`` (no penalty schedule)

    References
    ----------

    .. [1] Athey, S., Tibshirani, J., & Wager, S. (2019). "Generalized
           random forests." The Annals of Statistics, 47(2), 1148-1178.
           https://doi.org/10.1214/18-AOS1709
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError(
            "The 'cf' nuisance backend requires the econml package. Install it with: uv pip install 'moderndid[didml]'"
        ) from exc

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    n, _ = X.shape
    if Y.shape != (n,):
        raise ValueError(f"Y length {Y.shape} does not match X row count {n}.")
    if treatment.shape != (n,):
        raise ValueError(f"treatment length {treatment.shape} does not match X row count {n}.")
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError(f"n_estimators must be a positive integer, got {n_estimators!r}.")
    if not isinstance(cv, int) or cv < 2:
        raise ValueError(f"cv must be an integer >= 2, got {cv!r}.")
    if cv > n:
        raise ValueError(f"cv={cv} exceeds sample size n={n}.")

    splits = list(KFold(n_splits=cv, shuffle=True, random_state=random_state).split(X))

    nuisance_kwargs = dict(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    estimator = CausalForestDML(
        model_y=RandomForestRegressor(**nuisance_kwargs),
        model_t=RandomForestRegressor(**nuisance_kwargs),
        discrete_treatment=False,
        cv=splits,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
        honest=True,
    )

    estimator.fit(Y=Y, T=treatment, X=X)

    tau_hat = np.asarray(estimator.effect(X), dtype=float).ravel()

    fold_models_y = estimator.models_y[0]
    fold_models_t = estimator.models_t[0]
    m_hat = np.empty(n)
    p_hat = np.empty(n)
    for fold_idx, (_, test_idx) in enumerate(splits):
        m_hat[test_idx] = fold_models_y[fold_idx].predict(X[test_idx])
        p_hat[test_idx] = fold_models_t[fold_idx].predict(X[test_idx])

    return {
        "tau_hat": tau_hat,
        "p_hat": p_hat,
        "m_hat": m_hat,
        "tau_coef": None,
        "best_penalty_factor": None,
    }
