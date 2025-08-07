"""Uniform confidence band construction for nonparametric instrumental variables estimation."""

import warnings

import numpy as np

from ..utils import _quantile_basis, avoid_zero_division
from .cck_ucb import compute_cck_ucb
from .estimators import npiv_est
from .results import NPIVResult


def compute_ucb(
    y,
    x,
    w,
    x_eval=None,
    alpha=0.05,
    boot_num=99,
    basis="tensor",
    j_x_degree=3,
    j_x_segments=None,
    k_w_degree=4,
    k_w_segments=None,
    knots="uniform",
    ucb_h=True,
    ucb_deriv=True,
    deriv_index=1,
    deriv_order=1,
    w_min=None,
    w_max=None,
    x_min=None,
    x_max=None,
    seed=None,
    selection_result=None,
):
    r"""Compute uniform confidence bands for nonparametric instrumental variables.

    Constructs simultaneous confidence bands for the structural function :math:`h_0` and its
    derivatives :math:`\partial^a h_0`. For a fixed sieve dimension :math:`J`, the bands are constructed
    by under-smoothing, which requires :math:`J` to be larger than the optimal dimension for estimation.
    The confidence bands are given by

    .. math::

        C_{n,J}(x) = \left[\hat{h}_J(x) \pm z_{1-\alpha,J}^* \hat{\sigma}_J(x)\right],

    where :math:`\hat{\sigma}_J(x)` is the estimated standard error and :math:`z_{1-\alpha,J}^*` is the
    :math:`(1-\alpha)` quantile of the supremum of a multiplier bootstrap process given by

    .. math::

        \sup_{x \in \mathcal{X}} \left| \frac{D_J^*(x)}{\hat{\sigma}_J(x)} \right| =
        \sup_{x \in \mathcal{X}} \left| \frac{(\psi^J(x))' \mathbf{M}_J
        \hat{\mathbf{u}}_J^*}{\hat{\sigma}_J(x)} \right|,

    where :math:`\hat{\mathbf{u}}_J^*` contains residuals multiplied by IID :math:`N(0,1)` draws.
    This approach ensures that the bias of the estimator is asymptotically negligible relative
    to the sampling uncertainty, but at the cost of slower convergence rates for the confidence bands.

    When a data-driven dimension :math:`\tilde{J}` is selected, this function routes to the
    procedure from [2]_ to construct honest and adaptive uniform confidence bands, which contract
    at the minimax optimal rate.

    Parameters
    ----------
    y : ndarray
        Dependent variable vector of length :math:`n`.
    x : ndarray
        Endogenous regressor matrix of shape :math:`(n, p_x)`.
    w : ndarray
        Instrument matrix of shape :math:`(n, p_w)`.
    x_eval : ndarray, optional
        Evaluation points for :math:`X`. If None, uses :math:`x`.
    alpha : float, default=0.05
        Significance level (1-alpha confidence level).
    boot_num : int, default=99
        Number of bootstrap replications.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate X.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`.
    j_x_segments : int, optional
        Number of segments for :math:`X` basis. If None, chosen automatically.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`.
    k_w_segments : int, optional
        Number of segments for :math:`W` basis. If None, chosen automatically.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    ucb_h : bool, default=True
        Whether to compute confidence bands for function estimates.
    ucb_deriv : bool, default=True
        Whether to compute confidence bands for derivative estimates.
    deriv_index : int, default=1
        Index (1-based) of :math:`X` variable for derivative computation.
    deriv_order : int, default=1
        Order of derivative to compute.
    w_min : float, optional
        Minimum value for :math:`W` range.
    w_max : float, optional
        Maximum value for :math:`W` range.
    x_min : float, optional
        Minimum value for :math:`X` range.
    x_max : float, optional
        Maximum value for :math:`X` range.
    seed : int, optional
        Random seed for bootstrap.
    selection_result : dict, optional
        Result from data-driven selection.

    Returns
    -------
    NPIVResult
        NPIV results with uniform confidence bands included.

    See Also
    --------
    compute_cck_ucb : Compute honest and adaptive UCBs

    References
    ----------

    .. [1] Chen, X., & Christensen, T. M. (2018). Optimal sup-norm rates and uniform
        inference on nonlinear functionals of nonparametric IV regression.
        Quantitative Economics, 9(1), 39-84. https://arxiv.org/abs/1508.03365.

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2024).
        Adaptive Estimation and Uniform Confidence Bands for Nonparametric
        Structural Functions and Elasticities. https://arxiv.org/abs/2107.11869.
    """
    main_result = npiv_est(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        basis=basis,
        j_x_degree=j_x_degree,
        j_x_segments=j_x_segments,
        k_w_degree=k_w_degree,
        k_w_segments=k_w_segments,
        knots=knots,
        deriv_index=deriv_index,
        deriv_order=deriv_order,
        w_min=w_min,
        w_max=w_max,
        x_min=x_min,
        x_max=x_max,
        data_driven=selection_result is not None,
    )

    n = len(y)
    n_eval = len(main_result.h)

    if ucb_h:
        boot_h_stats = np.zeros(boot_num)
    else:
        boot_h_stats = None

    if ucb_deriv:
        boot_deriv_stats = np.zeros(boot_num)
    else:
        boot_deriv_stats = None

    if selection_result is not None:
        return compute_cck_ucb(
            y=y,
            x=x,
            w=w,
            x_eval=x_eval,
            alpha=alpha,
            boot_num=boot_num,
            basis=basis,
            j_x_degree=main_result.j_x_degree,
            k_w_degree=main_result.k_w_degree,
            knots=knots,
            ucb_h=ucb_h,
            ucb_deriv=ucb_deriv,
            deriv_index=deriv_index,
            deriv_order=deriv_order,
            w_min=w_min,
            w_max=w_max,
            x_min=x_min,
            x_max=x_max,
            seed=seed,
            selection_result=selection_result,
        )

    rng = np.random.default_rng(seed)

    tmp = main_result.args["tmp"]
    psi_x_eval = main_result.args["psi_x_eval"]
    psi_x_deriv_eval = main_result.args["psi_x_deriv_eval"]

    for b in range(boot_num):
        try:
            boot_draws = rng.normal(0, 1, n)

            if ucb_h:
                boot_h_diff = psi_x_eval @ (tmp @ (main_result.residuals * boot_draws))
                studentized_h = np.abs(boot_h_diff) / avoid_zero_division(main_result.asy_se)
                boot_h_stats[b] = np.max(studentized_h)

            if ucb_deriv:
                boot_deriv_diff = psi_x_deriv_eval @ (tmp @ (main_result.residuals * boot_draws))
                studentized_deriv = np.abs(boot_deriv_diff) / avoid_zero_division(main_result.deriv_asy_se)
                boot_deriv_stats[b] = np.max(studentized_deriv)

        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Bootstrap replication {b + 1} failed: {e}", UserWarning)
            if ucb_h:
                boot_h_stats[b] = np.nan
            if ucb_deriv:
                boot_deriv_stats[b] = np.nan

    cv_h = None
    cv_deriv = None

    if ucb_h and boot_h_stats is not None:
        finite_stats = boot_h_stats[np.isfinite(boot_h_stats)]
        if len(finite_stats) > 0:
            cv_h = _quantile_basis(finite_stats, 1 - alpha)
        else:
            warnings.warn("All bootstrap statistics are non-finite for function estimates", UserWarning)

    if ucb_deriv and boot_deriv_stats is not None:
        finite_stats = boot_deriv_stats[np.isfinite(boot_deriv_stats)]
        if len(finite_stats) > 0:
            cv_deriv = _quantile_basis(finite_stats, 1 - alpha)
        else:
            warnings.warn("All bootstrap statistics are non-finite for derivative estimates", UserWarning)

    h_lower = None
    h_upper = None
    h_lower_deriv = None
    h_upper_deriv = None

    if ucb_h and cv_h is not None:
        if np.all(np.isfinite(main_result.asy_se)) and np.all(main_result.asy_se > 0):
            margin_h = cv_h * main_result.asy_se
        else:
            warnings.warn("Using unstudentized confidence bands for function estimates", UserWarning)
            margin_h = cv_h * np.ones(n_eval)

        h_lower = main_result.h - margin_h
        h_upper = main_result.h + margin_h

    if ucb_deriv and cv_deriv is not None:
        if np.all(np.isfinite(main_result.deriv_asy_se)) and np.all(main_result.deriv_asy_se > 0):
            margin_deriv = cv_deriv * main_result.deriv_asy_se
        else:
            warnings.warn("Using unstudentized confidence bands for derivative estimates", UserWarning)
            margin_deriv = cv_deriv * np.ones(n_eval)

        h_lower_deriv = main_result.deriv - margin_deriv
        h_upper_deriv = main_result.deriv + margin_deriv

    updated_args = main_result.args.copy()
    updated_args.update(
        {
            "boot_num": boot_num,
            "alpha": alpha,
            "ucb_h": ucb_h,
            "ucb_deriv": ucb_deriv,
            "boot_success_rate_h": np.mean(np.isfinite(boot_h_stats)) if boot_h_stats is not None else None,
            "boot_success_rate_deriv": np.mean(np.isfinite(boot_deriv_stats)) if boot_deriv_stats is not None else None,
        }
    )

    return NPIVResult(
        h=main_result.h,
        h_lower=h_lower,
        h_upper=h_upper,
        deriv=main_result.deriv,
        h_lower_deriv=h_lower_deriv,
        h_upper_deriv=h_upper_deriv,
        beta=main_result.beta,
        asy_se=main_result.asy_se,
        deriv_asy_se=main_result.deriv_asy_se,
        cv=cv_h,
        cv_deriv=cv_deriv,
        residuals=main_result.residuals,
        j_x_degree=main_result.j_x_degree,
        j_x_segments=main_result.j_x_segments,
        k_w_degree=main_result.k_w_degree,
        k_w_segments=main_result.k_w_segments,
        args=updated_args,
    )
