"""Lu-Nie-Wager doubly-robust score for the ML DiD estimator."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from .nuisance import fit_causal_forest, fit_delta, fit_rlearner


def lnw_did(
    X,
    Y,
    post_indicator,
    cohort_indicator,
    *,
    constant_eff="non_constant",
    gamma=None,
    nu_model="rlearner",
    sigma_model="rlearner",
    delta_model="glm",
    t_func=True,
    k_folds=10,
    tune_penalty=False,
    lambda_choice="lambda.min",
    random_state=None,
):
    r"""Compute the doubly-robust ML score for one group-time cell.

    Implements the orthogonal decomposition of [1]_ for difference-in-
    differences with cross-fitted machine-learning nuisance estimates.

    Three nuisance functions are fit, the orthogonal :math:`(A, B, C)`
    coefficients and pseudo-outcome :math:`H` are computed in closed form,
    and the conditional treatment effect :math:`\tau(x)` is recovered by
    one of two routes selected by ``constant_eff``.

    With ``'non_constant'`` (default), :math:`\tau(x)` is fit as a
    penalized linear function of :math:`x` and combined with the
    optionally supplied augmented minimax-linear weights
    :math:`\hat{\gamma}` from [2]_. With ``'constant'``, a single scalar
    :math:`\tau` is fit by no-intercept OLS with HC3 robust standard
    errors.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate design matrix without the intercept column.
    Y : ndarray of shape (n,)
        Observed outcome.
    post_indicator : ndarray of shape (n,)
        Binary 0/1 indicator equal to 1 for the post-treatment period.
    cohort_indicator : ndarray of shape (n,)
        Binary 0/1 indicator equal to 1 for units in the treated cohort.
    constant_eff : {'constant', 'non_constant'}, default='non_constant'
        Whether to fit a constant CATT (single scalar :math:`\tau`) or a
        linear CATT :math:`\tau(x)`.
    gamma : ndarray of shape (n,), optional
        Augmented minimax-linear weights from
        :func:`~moderndid.didml.amle_weights`. Required for
        ``constant_eff='non_constant'`` to produce a non-NaN standard error.
    nu_model : {'rlearner', 'cf'}, default='rlearner'
        Nuisance backend for :math:`\nu`, :math:`m`, and :math:`t`.
    sigma_model : {'rlearner', 'cf'}, default='rlearner'
        Nuisance backend for :math:`\zeta` and :math:`g`.
    delta_model : {'glm', 'stack'}, default='glm'
        Nuisance backend for :math:`\Delta`.
    t_func : bool, default=True
        If False, replace the estimated :math:`\hat{t}(X)` with the constant
        :math:`0.5` to enforce balanced post-period probability.
    k_folds : int, default=10
        Number of folds for cross-fitting.
    tune_penalty : bool, default=False
        Whether to grid-search penalty factors for the nuisance and tau
        fits.
    lambda_choice : {'lambda.min', 'lambda.1se'}, default='lambda.min'
        Cross-validation rule for selecting the penalty in the
        non-constant tau fit.
    random_state : int, optional
        Seed controlling cross-fit fold splits.

    Returns
    -------
    dict
        Dictionary containing the doubly-robust score outputs:

        - **TAU_hat**: Scalar cell-level ATT estimate
        - **std_err**: Standard error for ``TAU_hat`` (HC3 OLS for the
          constant case; ``NaN`` when ``gamma`` is None for the
          non-constant case)
        - **tau_hat**: Length-:math:`n` array of CATT predictions
          :math:`\hat{\tau}(X_i)` (``None`` for ``constant_eff='constant'``)
        - **score**: Length-:math:`n` array of doubly-robust score
          contributions :math:`\hat{\tau}(X_i) + \hat{\gamma}_i (Y_i -
          \hat{y}_i)` (``None`` when ``gamma`` is missing or
          ``constant_eff='constant'``)
        - **y_hat**: Length-:math:`n` array of orthogonal-decomposition
          fitted values :math:`\hat{y}_i`
        - **m_hat**, **t_hat**, **s_hat**, **nu_hat**, **sigma_hat**,
          **delta_hat**: Cross-fitted nuisance components
        - **A_hat**, **B_hat**, **C_hat**: Closed-form orthogonal
          coefficients

    Notes
    -----
    **Outcome decomposition.** The observation-level outcome admits the
    representation

    .. math::

        Y_i = m(X_i)
            + A(X_i, G_i, T_i) \, \nu(X_i)
            + B(X_i, G_i, T_i) \, \zeta(X_i)
            + C(X_i, G_i, T_i) \, \tau(X_i)
            + \varepsilon_i,

    with the conditional response surfaces

    .. math::

        m(x) &= \mathbb{E}[Y \mid X = x],\\
        \nu(x) &= \mathbb{E}[Y \mid X = x, T = 1]
                - \mathbb{E}[Y \mid X = x, T = 0],\\
        \zeta(x) &= \mathbb{E}[Y \mid X = x, G = 1]
                  - \mathbb{E}[Y \mid X = x, G = 0],

    where :math:`\tau(x)` is the conditional treatment effect on the
    treated. Letting :math:`g(x) = \mathbb{E}[G \mid X = x]`,
    :math:`t(x) = \mathbb{E}[T \mid X = x]`, and

    .. math::

        \Delta(x) = \mathbb{E}[(T - t(X))(G - g(X)) \mid X = x],

    the orthogonal coefficients are

    .. math::

        A &= \frac{1}{1 - \Delta^2 / [g(1-g)\,t(1-t)]}
             \left( T - t(X)
             - \frac{\Delta(X)\,(G - g(X))}{g(X)(1 - g(X))} \right),\\
        B &= \frac{1}{1 - \Delta^2 / [g(1-g)\,t(1-t)]}
             \left( G - g(X)
             - \frac{\Delta(X)\,(T - t(X))}{t(X)(1 - t(X))} \right),\\
        C &= G \cdot T
             - \bigl(g(X)\,t(X) + \Delta(X)\bigr)
             - \left(g(X) + \tfrac{\Delta(X)}{t(X)}\right) A
             - \left(t(X) + \tfrac{\Delta(X)}{g(X)}\right) B.

    **Pseudo-outcome and tau regression.** The pseudo-outcome

    .. math::

        H_i = Y_i - \hat{m}(X_i) - A_i \hat{\nu}(X_i) - B_i \hat{\zeta}(X_i)

    isolates the :math:`\tau` signal. With ``constant_eff='constant'``,
    :math:`H = \tau \cdot C + \text{noise}` is fit by no-intercept OLS
    with HC3 robust standard errors. With ``constant_eff='non_constant'``
    a penalized linear CATE :math:`\hat{\tau}(x) = (1, x)^\top \hat{\beta}`
    is fit via the residualized lasso

    .. math::

        \hat{\beta} = \operatorname*{argmin}_{\beta}
        \sum_i \bigl(H_i - C_i \cdot (1, X_i)^\top \beta\bigr)^2
        + \lambda \sum_{j \ge 1} |\beta_j|,

    leaving the intercept-equivalent coefficient unpenalized via
    Frisch-Waugh-Lovell partialling-out.

    **ATT formula.** The cell-level ATT combines :math:`\hat{\tau}(X_i)`
    with the augmented minimax-linear weights :math:`\hat{\gamma}_i` (when
    supplied):

    .. math::

        \widehat{ATT} = \frac{1}{n} \sum_{i=1}^{n}
        \bigl[\hat{\tau}(X_i) + \hat{\gamma}_i (Y_i - \hat{y}_i)\bigr],

    where the orthogonal-decomposition fitted values are

    .. math::

        \hat{y}_i = \hat{m}(X_i) + A_i \hat{\nu}(X_i)
                  + B_i \hat{\zeta}(X_i) + C_i \hat{\tau}(X_i).

    When ``gamma`` is omitted the estimator reduces to
    :math:`\widehat{ATT} = (1/n) \sum_i \hat{\tau}(X_i)` and standard
    errors are not returned.

    References
    ----------

    .. [1] Nie, X., Lu, C., & Wager, S. (2024). "Nonparametric heterogeneous
           treatment effect estimation in repeated cross sectional designs."
           In E. Laber, B. Chakraborty, E. E. M. Moodie, T. Cai, & M. van
           der Laan (Eds.), Handbook of Statistical Methods for Precision
           Medicine (Ch. 9). Chapman and Hall/CRC.
           https://doi.org/10.1201/9781003216223-9

    .. [2] Hirshberg, D. A., & Wager, S. (2021). "Augmented minimax linear
           estimation." The Annals of Statistics, 49(6), 3206-3227.
           https://doi.org/10.1214/21-AOS2080
    """
    if constant_eff not in ("constant", "non_constant"):
        raise ValueError(f"constant_eff must be 'constant' or 'non_constant', got {constant_eff!r}.")
    if nu_model not in ("rlearner", "cf"):
        raise ValueError(f"nu_model must be 'rlearner' or 'cf', got {nu_model!r}.")
    if sigma_model not in ("rlearner", "cf"):
        raise ValueError(f"sigma_model must be 'rlearner' or 'cf', got {sigma_model!r}.")
    if delta_model not in ("glm", "stack"):
        raise ValueError(f"delta_model must be 'glm' or 'stack', got {delta_model!r}.")
    if lambda_choice not in ("lambda.min", "lambda.1se"):
        raise ValueError(f"lambda_choice must be 'lambda.min' or 'lambda.1se', got {lambda_choice!r}.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    post_indicator = np.asarray(post_indicator, dtype=float)
    cohort_indicator = np.asarray(cohort_indicator, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    n, _ = X.shape
    if Y.shape != (n,):
        raise ValueError(f"Y length {Y.shape} does not match X row count {n}.")
    if post_indicator.shape != (n,):
        raise ValueError(f"post_indicator length {post_indicator.shape} does not match X row count {n}.")
    if cohort_indicator.shape != (n,):
        raise ValueError(f"cohort_indicator length {cohort_indicator.shape} does not match X row count {n}.")
    if gamma is not None:
        gamma = np.asarray(gamma, dtype=float)
        if gamma.shape != (n,):
            raise ValueError(f"gamma length {gamma.shape} does not match X row count {n}.")

    nu_kwargs = dict(k_folds=k_folds, tune_penalty=tune_penalty, random_state=random_state)
    if nu_model == "rlearner":
        nu_fit = fit_rlearner(X, Y, post_indicator, lambda_choice=lambda_choice, **nu_kwargs)
    else:
        nu_fit = fit_causal_forest(X, Y, post_indicator, random_state=random_state)
    nu_hat = np.asarray(nu_fit["tau_hat"], dtype=float)
    t_hat = np.asarray(nu_fit["p_hat"], dtype=float)
    m_hat = np.asarray(nu_fit["m_hat"], dtype=float)

    if sigma_model == "rlearner":
        sigma_fit = fit_rlearner(X, Y, cohort_indicator, lambda_choice=lambda_choice, **nu_kwargs)
    else:
        sigma_fit = fit_causal_forest(X, Y, cohort_indicator, random_state=random_state)
    sigma_hat = np.asarray(sigma_fit["tau_hat"], dtype=float)
    s_hat = np.asarray(sigma_fit["p_hat"], dtype=float)

    if not t_func:
        t_hat = np.full(n, 0.5)

    s_safe = np.clip(s_hat, 1e-3, 1.0 - 1e-3)
    t_safe = np.clip(t_hat, 1e-3, 1.0 - 1e-3)

    response = (post_indicator - t_safe) * (cohort_indicator - s_safe)
    delta_fit = fit_delta(
        X,
        response,
        model=delta_model,
        k_folds=k_folds,
        tune_penalty=tune_penalty,
        random_state=random_state,
    )
    delta_hat = np.asarray(delta_fit["delta_hat"], dtype=float)

    var_product = s_safe * (1.0 - s_safe) * t_safe * (1.0 - t_safe)
    inv_scaling = 1.0 / (1.0 - delta_hat**2 / var_product)
    A_hat = inv_scaling * (
        post_indicator - t_safe - delta_hat * (cohort_indicator - s_safe) / (s_safe * (1.0 - s_safe))
    )
    B_hat = inv_scaling * (
        cohort_indicator - s_safe - delta_hat * (post_indicator - t_safe) / (t_safe * (1.0 - t_safe))
    )
    iota = s_safe * t_safe + delta_hat
    C_hat = (
        cohort_indicator * post_indicator
        - iota
        - (s_safe + delta_hat / t_safe) * A_hat
        - (t_safe + delta_hat / s_safe) * B_hat
    )

    H_hat = Y - m_hat - A_hat * nu_hat - B_hat * sigma_hat

    if constant_eff == "constant":
        ols = sm.OLS(H_hat, C_hat[:, None]).fit()
        tau_scalar = float(ols.params[0])
        robust = ols.get_robustcov_results(cov_type="HC3")
        std_err = float(np.sqrt(robust.cov_params()[0, 0]))
        return {
            "TAU_hat": tau_scalar,
            "std_err": std_err,
            "tau_hat": None,
            "score": None,
            "y_hat": None,
            "m_hat": m_hat,
            "t_hat": t_hat,
            "s_hat": s_hat,
            "nu_hat": nu_hat,
            "sigma_hat": sigma_hat,
            "delta_hat": delta_hat,
            "A_hat": A_hat,
            "B_hat": B_hat,
            "C_hat": C_hat,
        }

    tau_coef = _fit_tau_coef(
        X=X,
        H_hat=H_hat,
        C_hat=C_hat,
        k_folds=k_folds,
        lambda_choice=lambda_choice,
        random_state=random_state,
    )
    design = np.column_stack([np.ones(n), X])
    tau_hat = design @ tau_coef
    y_hat = m_hat + A_hat * nu_hat + B_hat * sigma_hat + C_hat * tau_hat

    if gamma is None:
        TAU_hat = float(np.mean(tau_hat))
        std_err = float("nan")
        score = None
    else:
        score = tau_hat + gamma * (Y - y_hat)
        TAU_hat = float(np.mean(score))
        residual_var = np.mean((tau_hat - TAU_hat) ** 2 + gamma**2 * (Y - y_hat) ** 2)
        std_err = float(np.sqrt(residual_var / n))

    return {
        "TAU_hat": TAU_hat,
        "std_err": std_err,
        "tau_hat": tau_hat,
        "score": score,
        "y_hat": y_hat,
        "tau_coef": tau_coef,
        "m_hat": m_hat,
        "t_hat": t_hat,
        "s_hat": s_hat,
        "nu_hat": nu_hat,
        "sigma_hat": sigma_hat,
        "delta_hat": delta_hat,
        "A_hat": A_hat,
        "B_hat": B_hat,
        "C_hat": C_hat,
    }


def _fit_tau_coef(*, X, H_hat, C_hat, k_folds, lambda_choice, random_state):
    """Fit the linear CATT coefficients via residualized penalized lasso."""
    n = X.shape[0]

    scaler = StandardScaler().fit(X)
    X_scl = scaler.transform(X)

    interaction = C_hat[:, None] * X_scl
    c_inner = float(C_hat @ C_hat)
    if c_inner < 1e-12:
        raise RuntimeError("C_hat is degenerate (C_hat'C_hat ≈ 0); check propensity overlap and delta estimate.")

    alpha_y = float(C_hat @ H_hat) / c_inner
    alpha_z = (C_hat @ interaction) / c_inner
    H_resid = H_hat - alpha_y * C_hat
    Z_resid = interaction - C_hat[:, None] * alpha_z[None, :]

    inner_cv = max(2, min(k_folds, n - 1))
    cv_fit = LassoCV(
        cv=inner_cv,
        n_alphas=100,
        max_iter=10_000,
        random_state=random_state,
        fit_intercept=False,
    ).fit(Z_resid, H_resid)

    if lambda_choice == "lambda.min":
        beta_scl = np.asarray(cv_fit.coef_, dtype=float)
    else:
        alphas = np.asarray(cv_fit.alphas_)
        mse_path = np.asarray(cv_fit.mse_path_)
        mean_mse = mse_path.mean(axis=1)
        se_mse = mse_path.std(axis=1, ddof=1) / np.sqrt(mse_path.shape[1])
        min_idx = int(np.argmin(mean_mse))
        threshold = mean_mse[min_idx] + se_mse[min_idx]
        chosen_alpha = float(alphas[np.where(mean_mse <= threshold)[0]].max())
        beta_scl = np.asarray(
            Lasso(alpha=chosen_alpha, fit_intercept=False, max_iter=10_000).fit(Z_resid, H_resid).coef_,
            dtype=float,
        )

    sigma = scaler.scale_
    mu = scaler.mean_
    beta_raw_X = beta_scl / sigma
    beta_raw_intercept = alpha_y - float(alpha_z @ beta_scl) - float(beta_raw_X @ mu)
    return np.concatenate([[beta_raw_intercept], beta_raw_X])
