"""Uniform confidence band construction for nonparametric instrumental variables estimation."""

import warnings

import numpy as np

from ..cupy.backend import get_backend, to_device
from .cck_ucb import compute_cck_ucb
from .container import NPIVResult
from .estimators import _fit_sieve_tsls
from .utils import _quantile_basis, avoid_zero_division


def compute_ucb(
    y,
    x,
    w,
    x_eval=None,
    alpha=0.05,
    biters=99,
    basis="tensor",
    j_x_degree=3,
    j_x_segments=None,
    k_w_degree=4,
    k_w_segments=None,
    k_w_smooth=2,
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

    Builds simultaneous confidence bands for the structural function :math:`h_0` and its
    derivative :math:`\partial^a h_0` at a fixed sieve dimension :math:`J`. These are the
    undersmoothed bands of [1]_. Their validity rests on a dimension :math:`J` larger than
    the optimal one for estimation. At such a dimension the approximation bias is negligible
    next to the sampling uncertainty. The bands are then wider than bands at the optimal
    dimension.

    When ``selection_result`` holds a data-driven selection, the function returns the
    data-driven bands of [2]_ from :func:`compute_cck_ucb` instead.

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
    biters : int, default=99
        Number of bootstrap replications.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate X.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`.
    j_x_segments : int, optional
        Number of segments for :math:`X` basis. If None, chosen from the
        sample size.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`.
    k_w_segments : int, optional
        Number of segments for :math:`W` basis. If None, set to
        ``j_x_segments * 2**k_w_smooth``.
    k_w_smooth : int, default=2
        Number of dyadic refinements of the :math:`W` basis relative to the
        :math:`X` basis, used when ``k_w_segments`` is None.
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
        Output of :func:`npiv_choose_j`. When given, the segments come from it.
        The function then ignores ``j_x_segments``, ``k_w_segments``, and
        ``k_w_smooth``.

    Returns
    -------
    NPIVResult
        Named tuple with the following fields. When ``selection_result`` is given, they
        come from :func:`compute_cck_ucb` at the selected dimension.

        - **h**: Estimate :math:`\hat{h}_J(x)` at the evaluation points
        - **h_lower**: Lower uniform confidence band for :math:`h_0`, or None when ``ucb_h`` is False
        - **h_upper**: Upper uniform confidence band for :math:`h_0`, or None when ``ucb_h`` is False
        - **deriv**: Estimate :math:`\partial^a \hat{h}_J(x)` at the evaluation points
        - **h_lower_deriv**: Lower uniform confidence band for :math:`\partial^a h_0`, or None when ``ucb_deriv``
          is False
        - **h_upper_deriv**: Upper uniform confidence band for :math:`\partial^a h_0`, or None when ``ucb_deriv``
          is False
        - **beta**: Sieve coefficient vector :math:`\hat{c}_J`
        - **asy_se**: Pointwise standard errors :math:`\hat{\sigma}_J(x)`
        - **deriv_asy_se**: Pointwise standard errors :math:`\hat{\sigma}^a_J(x)` of the derivative
        - **cv**: Critical value :math:`z_{1-\alpha,J}^*` of the function band, or :math:`\text{cv}^*` of
          :func:`compute_cck_ucb` after a data-driven selection
        - **cv_deriv**: Critical value of the derivative band, built the same way as ``cv``
        - **residuals**: Residuals :math:`Y_i - \hat{h}_J(X_i)`
        - **j_x_degree**: Degree of the basis for :math:`X`
        - **j_x_segments**: Segments of the basis for :math:`X`
        - **k_w_degree**: Degree of the basis for :math:`W`
        - **k_w_segments**: Segments of the basis for :math:`W`
        - **args**: Diagnostic dictionary that also holds ``biters``, ``alpha``, and the share of bootstrap draws
          that succeeded. After a data-driven selection it holds ``j_tilde``, ``theta_star``, and ``n_j_boot``
          instead.

    Examples
    --------
    Compute 95 percent uniform confidence bands for the Engel curve by
    undersmoothing at a fixed sieve dimension:

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import load_engel
           ...: from moderndid.npiv import compute_ucb
           ...:
           ...: df = load_engel()
           ...: y = df["food"].to_numpy()
           ...: x = df["logexp"].to_numpy().reshape(-1, 1)
           ...: w = df["logwages"].to_numpy().reshape(-1, 1)
           ...: result = compute_ucb(
           ...:     y=y, x=x, w=w, j_x_segments=5, biters=500, seed=42,
           ...: )
           ...: print(f"Function UCB critical value: {result.cv:.3f}")
           ...: print(f"Derivative UCB critical value: {result.cv_deriv:.3f}")
           ...: print(f"Bands at {len(result.h)} evaluation points")

    See Also
    --------
    compute_cck_ucb : Compute data-driven uniform confidence bands

    Notes
    -----
    The band is

    .. math::

        C_{n,J}(x) = \left[\hat{h}_J(x) \pm z_{1-\alpha,J}^* \hat{\sigma}_J(x)\right],

    where :math:`\hat{\sigma}_J(x)` is the pointwise standard error and :math:`z_{1-\alpha,J}^*` is
    the :math:`(1-\alpha)` quantile across bootstrap draws of

    .. math::

        \sup_{x} \left| \frac{D_J^*(x)}{\hat{\sigma}_J(x)} \right|,
        \qquad
        D_J^*(x) = (\psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J^*.

    Here :math:`\mathbf{M}_J` is the matrix that maps :math:`\mathbf{Y}` to the coefficients
    :math:`\hat{c}_J` of :func:`npiv`. The vector :math:`\hat{\mathbf{u}}_J^*` multiplies each residual
    by an independent standard normal draw. The supremum runs over the evaluation points. The
    derivative band uses :math:`\partial^a \psi^J` in place of :math:`\psi^J`.

    References
    ----------

    .. [1] Chen, X., & Christensen, T. M. (2018). Optimal sup-norm rates and uniform
        inference on nonlinear functionals of nonparametric IV regression.
        Quantitative Economics, 9(1), 39-84. https://arxiv.org/abs/1508.03365.

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive estimation and
        uniform confidence bands for nonparametric structural functions and elasticities.
        *Review of Economic Studies*, 92(1), 162-196. https://doi.org/10.1093/restud/rdae025.
    """
    if selection_result is not None:
        return compute_cck_ucb(
            y=y,
            x=x,
            w=w,
            x_eval=x_eval,
            alpha=alpha,
            biters=biters,
            basis=basis,
            j_x_degree=j_x_degree,
            k_w_degree=k_w_degree,
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

    main_result, (tmp, psi_x_eval, psi_x_deriv_eval) = _fit_sieve_tsls(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        basis=basis,
        j_x_degree=j_x_degree,
        j_x_segments=j_x_segments,
        k_w_degree=k_w_degree,
        k_w_segments=k_w_segments,
        k_w_smooth=k_w_smooth,
        knots=knots,
        deriv_index=deriv_index,
        deriv_order=deriv_order,
        check_is_fullrank=False,
        w_min=w_min,
        w_max=w_max,
        x_min=x_min,
        x_max=x_max,
        data_driven=False,
    )

    n = len(y)
    n_eval = len(main_result.h)

    boot_h_stats = np.zeros(biters) if ucb_h else None

    boot_deriv_stats = np.zeros(biters) if ucb_deriv else None

    xp = get_backend()
    rng = np.random.default_rng(seed)

    for b in range(biters):
        try:
            boot_draws = to_device(rng.normal(0, 1, n))

            if ucb_h:
                boot_h_diff = psi_x_eval @ (tmp @ (main_result.residuals * boot_draws))
                studentized_h = xp.abs(boot_h_diff) / avoid_zero_division(main_result.asy_se)
                boot_h_stats[b] = xp.max(studentized_h)

            if ucb_deriv:
                boot_deriv_diff = psi_x_deriv_eval @ (tmp @ (main_result.residuals * boot_draws))
                studentized_deriv = xp.abs(boot_deriv_diff) / avoid_zero_division(main_result.deriv_asy_se)
                boot_deriv_stats[b] = xp.max(studentized_deriv)

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
            "biters": biters,
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
