"""Honest uniform confidence bands."""

import numpy as np

from ..utils import _quantile_basis, avoid_zero_division
from .estimators import _ginv, npiv_est
from .prodspline import prodspline
from .results import NPIVResult


def compute_cck_ucb(
    y,
    x,
    w,
    x_eval=None,
    alpha=0.05,
    boot_num=99,
    basis="tensor",
    j_x_degree=3,
    k_w_degree=4,
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
    r"""Compute honest and adaptive UCBs.

    Implements the data-driven uniform confidence band (UCB) construction from [1]_.
    These UCBs are honest, guaranteeing uniform coverage over a class of data-generating
    processes, and adaptive, contracting at or near the minimax optimal rate.

    The UCB for :math:`h_0` is constructed as

    .. math::

        C_n(x) = \left[\hat{h}_{\tilde{J}}(x) \pm \text{cv}^*(x) \hat{\sigma}_{\tilde{J}}(x)\right],

    where :math:`\tilde{J}` is the data-driven dimension choice. The critical value :math:`\text{cv}^*(x)`
    is a combination of a bootstrap quantile :math:`z_{1-\alpha}^*` and a penalty term involving
    the critical value from the dimension selection step, :math:`\theta_{1-\hat{\alpha}}^*`.

    For models in the "mildly ill-posed" regime (including nonparametric regression),
    the critical value is

    .. math::

        \text{cv}^*(x) = z_{1-\alpha}^* + (\log\log\tilde{J}) \theta_{1-\hat{\alpha}}^*.

    The quantile :math:`z_{1-\alpha}^*` is the :math:`(1-\alpha)` quantile of

    .. math::

        \sup_{(x,J) \in \mathcal{X} \times \hat{\mathcal{J}}_-} |D_J^*(x) / \hat{\sigma}_J(x)|

    ensuring robustness to the choice of :math:`J`. A similar construction applies to the
    derivatives of :math:`h_0`. This method provides efficiency improvements over traditional
    undersmoothing by adapting to the data.

    Parameters
    ----------
    y : ndarray
        Dependent variable vector.
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    x_eval : ndarray, optional
        Evaluation points for X. If None, uses x.
    alpha : float, default=0.05
        Significance level (1-alpha confidence level).
    boot_num : int, default=99
        Number of bootstrap replications.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate X.
    j_x_degree : int, default=3
        Degree of B-spline basis for X.
    k_w_degree : int, default=4
        Degree of B-spline basis for W.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    ucb_h : bool, default=True
        Whether to compute confidence bands for function estimates.
    ucb_deriv : bool, default=True
        Whether to compute confidence bands for derivative estimates.
    deriv_index : int, default=1
        Index (1-based) of X variable for derivative computation.
    deriv_order : int, default=1
        Order of derivative to compute.
    w_min : float, optional
        Minimum value for W range.
    w_max : float, optional
        Maximum value for W range.
    x_min : float, optional
        Minimum value for X range.
    x_max : float, optional
        Maximum value for X range.
    seed : int, optional
        Random seed for reproducibility.
    selection_result : dict, optional
        Result from data-driven selection.

    Returns
    -------
    NPIVResult
        NPIV results with CCK uniform confidence bands.

    See Also
    --------
    compute_ucb : Compute UCBs using under-smoothing

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024).
        Adaptive Estimation and Uniform Confidence Bands for Nonparametric
        Structural Functions and Elasticities. https://arxiv.org/abs/2107.11869.
    """
    y = np.asarray(y).ravel()
    x = np.atleast_2d(x)
    w = np.atleast_2d(w)

    n = len(y)
    p_x = x.shape[1]
    p_w = w.shape[1]

    if x_eval is None:
        x_eval = x.copy()
    else:
        x_eval = np.atleast_2d(x_eval)

    j_x_segments = selection_result["j_x_seg"]
    k_w_segments = selection_result["k_w_seg"]
    j_tilde = selection_result["j_tilde"]
    theta_star = selection_result["theta_star"]
    j_x_segments_set = selection_result["j_x_segments_set"]
    k_w_segments_set = selection_result["k_w_segments_set"]

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
        data_driven=True,
    )

    if ucb_h or ucb_deriv:
        if len(j_x_segments_set) > 2:
            j_segments_boot = j_x_segments_set[j_x_segments_set <= max(j_x_segments, j_x_segments_set[-3])]
        else:
            j_segments_boot = j_x_segments_set

        n_j_boot = len(j_segments_boot)
        if ucb_h:
            z_sup_boot_h = np.zeros((boot_num, n_j_boot))
        if ucb_deriv:
            z_sup_boot_deriv = np.zeros((boot_num, n_j_boot))

        rng = np.random.default_rng(seed)
        boot_draws_all = rng.normal(0, 1, (boot_num, n))

        for j_idx, j_seg in enumerate(j_segments_boot):
            k_seg = k_w_segments_set[np.where(j_x_segments_set == j_seg)[0][0]]

            K_x = np.column_stack([np.full(p_x, j_x_degree), np.full(p_x, j_seg)])
            K_w = np.column_stack([np.full(p_w, k_w_degree), np.full(p_w, k_seg)])

            psi_x = prodspline(
                x=x,
                K=K_x,
                knots=knots,
                basis=basis,
                x_min=np.full(p_x, x_min) if x_min else None,
                x_max=np.full(p_x, x_max) if x_max else None,
            ).basis
            b_w = prodspline(
                x=w,
                K=K_w,
                knots=knots,
                basis=basis,
                x_min=np.full(p_w, w_min) if w_min else None,
                x_max=np.full(p_w, w_max) if w_max else None,
            ).basis

            psi_x_eval = prodspline(
                x=x,
                xeval=x_eval,
                K=K_x,
                knots=knots,
                basis=basis,
                x_min=np.full(p_x, x_min) if x_min else None,
                x_max=np.full(p_x, x_max) if x_max else None,
            ).basis

            if ucb_deriv:
                psi_x_deriv_eval = prodspline(
                    x=x,
                    xeval=x_eval,
                    K=K_x,
                    knots=knots,
                    basis=basis,
                    deriv_index=deriv_index,
                    deriv=deriv_order,
                    x_min=np.full(p_x, x_min) if x_min else None,
                    x_max=np.full(p_x, x_max) if x_max else None,
                ).basis

            if basis in ("additive", "glp"):
                psi_x = np.c_[np.ones(n), psi_x]
                psi_x_eval = np.c_[np.ones(x_eval.shape[0]), psi_x_eval]
                if ucb_deriv:
                    psi_x_deriv_eval = np.c_[np.zeros(x_eval.shape[0]), psi_x_deriv_eval]
                b_w = np.c_[np.ones(n), b_w]

            btb_inv = _ginv(b_w.T @ b_w)
            design_matrix = psi_x.T @ b_w @ btb_inv @ b_w.T
            gram_inv = _ginv(design_matrix @ psi_x)
            tmp = gram_inv @ design_matrix

            beta = tmp @ y
            residuals = y - psi_x @ beta

            weighted_tmp = tmp.T * residuals[:, np.newaxis]
            D_inv_rho_D_inv = weighted_tmp.T @ weighted_tmp

            if ucb_h:
                var_h = np.diag(psi_x_eval @ D_inv_rho_D_inv @ psi_x_eval.T)
                asy_se_h = np.sqrt(np.maximum(var_h, 0))

            if ucb_deriv:
                var_deriv = np.diag(psi_x_deriv_eval @ D_inv_rho_D_inv @ psi_x_deriv_eval.T)
                asy_se_deriv = np.sqrt(np.maximum(var_deriv, 0))

            for b in range(boot_num):
                boot_draws = boot_draws_all[b]

                if ucb_h:
                    boot_h = psi_x_eval @ (tmp @ (residuals * boot_draws))
                    z_sup_boot_h[b, j_idx] = np.max(np.abs(boot_h) / avoid_zero_division(asy_se_h))

                if ucb_deriv:
                    boot_deriv = psi_x_deriv_eval @ (tmp @ (residuals * boot_draws))
                    z_sup_boot_deriv[b, j_idx] = np.max(np.abs(boot_deriv) / avoid_zero_division(asy_se_deriv))

        cv_h = None
        cv_deriv = None

        if ucb_h:
            z_boot_h = np.max(z_sup_boot_h, axis=1)
            z_star_h = _quantile_basis(z_boot_h, 1 - alpha)

            if j_tilde is not None:
                cv_h = z_star_h + max(0, np.log(np.log(j_tilde))) * theta_star
            else:
                cv_h = z_star_h

        if ucb_deriv:
            z_boot_deriv = np.max(z_sup_boot_deriv, axis=1)
            z_star_deriv = _quantile_basis(z_boot_deriv, 1 - alpha)

            if j_tilde is not None:
                cv_deriv = z_star_deriv + max(0, np.log(np.log(j_tilde))) * theta_star
            else:
                cv_deriv = z_star_deriv

        if ucb_h and cv_h is not None:
            h_lower = main_result.h - cv_h * main_result.asy_se
            h_upper = main_result.h + cv_h * main_result.asy_se
        else:
            h_lower = h_upper = None

        if ucb_deriv and cv_deriv is not None:
            h_lower_deriv = main_result.deriv - cv_deriv * main_result.deriv_asy_se
            h_upper_deriv = main_result.deriv + cv_deriv * main_result.deriv_asy_se
        else:
            h_lower_deriv = h_upper_deriv = None
    else:
        cv_h = cv_deriv = None
        h_lower = h_upper = None
        h_lower_deriv = h_upper_deriv = None

    args = main_result.args.copy()
    args.update(
        {
            "cck_method": True,
            "j_tilde": j_tilde,
            "theta_star": theta_star,
            "n_j_boot": len(j_segments_boot) if ucb_h or ucb_deriv else 0,
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
        args=args,
    )
