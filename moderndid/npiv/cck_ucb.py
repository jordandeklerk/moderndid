"""Honest uniform confidence bands."""

import numpy as np

from ..cupy.backend import get_backend, to_device, to_numpy
from .container import NPIVResult
from .estimators import _ginv, npiv_est
from .prodspline import prodspline
from .utils import _as_eval_points, _as_matrix, _quantile_basis, avoid_zero_division


def compute_cck_ucb(
    y,
    x,
    w,
    x_eval=None,
    alpha=0.05,
    biters=99,
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
    r"""Compute data-driven uniform confidence bands.

    Implements the data-driven uniform confidence bands (UCBs) of [1]_ for :math:`h_0`
    and its derivative at the dimension :math:`\tilde{J}` that :func:`npiv_choose_j`
    selects. The critical value adds a margin for that data-driven choice to a bootstrap
    quantile taken across several candidate dimensions. The band therefore needs no
    undersmoothed dimension.

    In mildly ill-posed models, and so in nonparametric regression, [1]_ shows that these
    bands cover :math:`h_0` uniformly over a class of functions. They contract within a
    :math:`\log\log n` factor of the minimax sup-norm rate.

    The critical value takes the same form whether the Lepski test or the truncation at
    the second-largest grid dimension sets :math:`\tilde{J}`. When truncation sets it,
    the procedure in [1]_ adds a bias envelope that needs a lower bound on the smoothness
    of :math:`h_0`. Since that bound is unknown in practice and the envelope depends on
    the units of :math:`Y`, this function leaves it out. The coverage guarantee of [1]_
    for severely ill-posed models assumes the envelope.

    When ``w`` equals ``x``, the instrument basis takes the degree and segments of the
    basis for :math:`X` at every dimension. The function then ignores ``k_w_degree`` and
    the instrument segments in ``selection_result``. Unless the range limits on ``w``
    differ from those on ``x``, the fits are least squares regressions on the basis for
    :math:`X`.

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
    biters : int, default=99
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
    selection_result : dict
        Output of :func:`npiv_choose_j`.

    Returns
    -------
    NPIVResult
        Named tuple with the following fields:

        - **h**: Estimate :math:`\hat{h}_{\tilde{J}}(x)` at the evaluation points
        - **h_lower**: Lower uniform confidence band for :math:`h_0`, or None when ``ucb_h`` is False
        - **h_upper**: Upper uniform confidence band for :math:`h_0`, or None when ``ucb_h`` is False
        - **deriv**: Estimate :math:`\partial^a \hat{h}_{\tilde{J}}(x)` at the evaluation points
        - **h_lower_deriv**: Lower uniform confidence band for :math:`\partial^a h_0`, or None when ``ucb_deriv``
          is False
        - **h_upper_deriv**: Upper uniform confidence band for :math:`\partial^a h_0`, or None when ``ucb_deriv``
          is False
        - **beta**: Sieve coefficient vector :math:`\hat{c}_{\tilde{J}}`
        - **asy_se**: Pointwise standard errors :math:`\hat{\sigma}_{\tilde{J}}(x)`
        - **deriv_asy_se**: Pointwise standard errors :math:`\hat{\sigma}^a_{\tilde{J}}(x)` of the derivative
        - **cv**: Critical value :math:`\text{cv}^*` of the function band
        - **cv_deriv**: Critical value :math:`\text{cv}^{a*}` of the derivative band
        - **residuals**: Residuals :math:`Y_i - \hat{h}_{\tilde{J}}(X_i)`
        - **j_x_degree**: Degree of the basis for :math:`X`
        - **j_x_segments**: Segments of the basis for :math:`X` at :math:`\tilde{J}`
        - **k_w_degree**: Degree of the basis for :math:`W`
        - **k_w_segments**: Segments of the basis for :math:`W`
        - **args**: Diagnostic dictionary that also holds ``j_tilde``, ``theta_star``, and ``n_j_boot``, the
          number of grid dimensions in the bootstrap

    Examples
    --------
    :func:`npiv` calls this function when ``j_x_segments`` is None. The example
    runs the selection and the band for the Engel curve as two separate steps:

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import load_engel
           ...: from moderndid.npiv import npiv_choose_j, compute_cck_ucb
           ...:
           ...: df = load_engel()
           ...: y = df["food"].to_numpy()
           ...: x = df["logexp"].to_numpy().reshape(-1, 1)
           ...: w = df["logwages"].to_numpy().reshape(-1, 1)
           ...: sel = npiv_choose_j(y=y, x=x, w=w, biters=50, seed=42)
           ...: result = compute_cck_ucb(
           ...:     y=y, x=x, w=w, biters=500, seed=42, selection_result=sel,
           ...: )
           ...: print(f"Adaptive CV: {result.cv:.3f}")
           ...: print(f"Selected dimension: {result.args['j_tilde']}")

    See Also
    --------
    compute_ucb : Compute UCBs using under-smoothing

    Notes
    -----
    The band for :math:`h_0` is

    .. math::

        C_n(x) = \left[\hat{h}_{\tilde{J}}(x) \pm \text{cv}^* \hat{\sigma}_{\tilde{J}}(x)\right],
        \qquad
        \text{cv}^* = z_{1-\alpha}^* + \max\{0, \log\log\tilde{J}\}\, \theta_{1-\hat{\alpha}}^*,

    where :math:`\theta_{1-\hat{\alpha}}^*` is the critical value of the Lepski test in
    :func:`npiv_j`. The quantile :math:`z_{1-\alpha}^*` is the :math:`(1-\alpha)` quantile across
    bootstrap draws of

    .. math::

        \sup_{x} \max_{J \in \mathcal{J}_b} \left| \frac{D_J^*(x)}{\hat{\sigma}_J(x)} \right|.

    Here :math:`D_J^*` is the multiplier bootstrap process of :func:`compute_ucb`. The supremum runs
    over the evaluation points. The set :math:`\mathcal{J}_b` holds the grid dimensions up to the
    larger of :math:`\tilde{J}` and the third-largest grid dimension, or the whole grid when it has
    two entries or fewer. When :math:`\tilde{J}` lies below the second-largest grid dimension
    :math:`\hat{J}_n`, this is the set :math:`\hat{\mathcal{J}}_-` of [1]_. When
    :math:`\tilde{J} = \hat{J}_n`, it holds every grid dimension but the largest. [1]_ then takes
    the whole grid if truncation set :math:`\tilde{J}` and the dimensions below :math:`\hat{J}_n` if
    the Lepski test did. Both cases apply when the test selects :math:`\hat{J}_n` itself.

    For the derivative band, :math:`D_J^*` uses :math:`\partial^a \psi^J` in place of :math:`\psi^J`
    and :math:`\hat{\sigma}^a_J` replaces :math:`\hat{\sigma}_J`. The resulting quantile
    :math:`z^{a*}_{1-\alpha}` gets the same margin as the function band,

    .. math::

        \text{cv}^{a*} = z^{a*}_{1-\alpha} + \max\{0, \log\log\tilde{J}\}\, \theta_{1-\hat{\alpha}}^*.

    When truncation sets :math:`\tilde{J}`, the procedure in [1]_ replaces
    :math:`\theta_{1-\hat{\alpha}}^*` in the margin by

    .. math::

        \max\left\{\theta_{1-\hat{\alpha}}^*, \frac{\tilde{J}^{-p/d}}{\hat{\sigma}_{\tilde{J}}(x)}\right\}
        \quad \text{and} \quad
        \max\left\{\theta_{1-\hat{\alpha}}^*, \frac{\tilde{J}^{(|a|-p)/d}}{\hat{\sigma}^a_{\tilde{J}}(x)}\right\}

    for the function and the derivative, where :math:`p` is a lower bound on the smoothness of
    :math:`h_0` and :math:`d` is the number of regressors. These envelopes are left out here.

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive estimation and
        uniform confidence bands for nonparametric structural functions and elasticities.
        *Review of Economic Studies*, 92(1), 162-196. https://doi.org/10.1093/restud/rdae025.
    """
    xp = get_backend()
    y = xp.asarray(y).ravel()
    x = _as_matrix(x)
    w = _as_matrix(w)

    n = len(y)
    p_x = x.shape[1]
    p_w = w.shape[1]

    x_eval = x.copy() if x_eval is None else _as_eval_points(x_eval, p_x)

    if selection_result is None:
        raise ValueError(
            "compute_cck_ucb needs selection_result, the output of npiv_choose_j. Run npiv_choose_j first, or use "
            "compute_ucb with a fixed j_x_segments."
        )

    j_x_segments = selection_result["j_x_seg"]
    k_w_segments = selection_result["k_w_seg"]
    j_tilde = selection_result["j_tilde"]
    theta_star = selection_result["theta_star"]
    j_x_segments_set = selection_result["j_x_segments_set"]
    k_w_segments_set = selection_result["k_w_segments_set"]

    if np.array_equal(to_numpy(x), to_numpy(w)):
        # The instrument basis takes the degree and segments of the X basis, as in npiv_est and the selection step.
        k_w_degree = j_x_degree
        k_w_segments = j_x_segments
        k_w_segments_set = j_x_segments_set

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
            z_sup_boot_h = np.zeros((biters, n_j_boot))
        if ucb_deriv:
            z_sup_boot_deriv = np.zeros((biters, n_j_boot))

        rng = np.random.default_rng(seed)
        boot_draws_all = to_device(rng.normal(0, 1, (biters, n)))

        for j_idx, j_seg in enumerate(j_segments_boot):
            k_seg = k_w_segments_set[np.where(j_x_segments_set == j_seg)[0][0]]

            # prodspline takes the number of segments minus one, as in npiv_est and the selection step.
            K_x = np.column_stack([np.full(p_x, j_x_degree), np.full(p_x, j_seg - 1)])
            K_w = np.column_stack([np.full(p_w, k_w_degree), np.full(p_w, k_seg - 1)])

            psi_x = prodspline(
                x=x,
                K=K_x,
                knots=knots,
                basis=basis,
                x_min=np.full(p_x, x_min) if x_min is not None else None,
                x_max=np.full(p_x, x_max) if x_max is not None else None,
            ).basis
            b_w = prodspline(
                x=w,
                K=K_w,
                knots=knots,
                basis=basis,
                x_min=np.full(p_w, w_min) if w_min is not None else None,
                x_max=np.full(p_w, w_max) if w_max is not None else None,
            ).basis

            psi_x_eval = prodspline(
                x=x,
                xeval=x_eval,
                K=K_x,
                knots=knots,
                basis=basis,
                x_min=np.full(p_x, x_min) if x_min is not None else None,
                x_max=np.full(p_x, x_max) if x_max is not None else None,
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
                    x_min=np.full(p_x, x_min) if x_min is not None else None,
                    x_max=np.full(p_x, x_max) if x_max is not None else None,
                ).basis

            if basis in ("additive", "glp"):
                psi_x = xp.concatenate([xp.ones((n, 1)), psi_x], axis=1)
                psi_x_eval = xp.concatenate([xp.ones((x_eval.shape[0], 1)), psi_x_eval], axis=1)
                if ucb_deriv:
                    psi_x_deriv_eval = xp.concatenate([xp.zeros((x_eval.shape[0], 1)), psi_x_deriv_eval], axis=1)
                b_w = xp.concatenate([xp.ones((n, 1)), b_w], axis=1)

            btb_inv = _ginv(b_w.T @ b_w)
            design_matrix = psi_x.T @ b_w @ btb_inv @ b_w.T
            gram_inv = _ginv(design_matrix @ psi_x)
            tmp = gram_inv @ design_matrix

            beta = tmp @ y
            residuals = y - psi_x @ beta

            weighted_tmp = tmp.T * residuals[:, None]
            D_inv_rho_D_inv = weighted_tmp.T @ weighted_tmp

            # Row sums give the diagonal of psi D psi' without forming the m x m matrix.
            if ucb_h:
                var_h = xp.sum((psi_x_eval @ D_inv_rho_D_inv) * psi_x_eval, axis=1)
                asy_se_h = xp.sqrt(xp.maximum(var_h, 0))

            if ucb_deriv:
                var_deriv = xp.sum((psi_x_deriv_eval @ D_inv_rho_D_inv) * psi_x_deriv_eval, axis=1)
                asy_se_deriv = xp.sqrt(xp.maximum(var_deriv, 0))

            for b in range(biters):
                boot_draws = boot_draws_all[b]

                if ucb_h:
                    boot_h = psi_x_eval @ (tmp @ (residuals * boot_draws))
                    z_sup_boot_h[b, j_idx] = float(xp.max(xp.abs(boot_h) / avoid_zero_division(asy_se_h)))

                if ucb_deriv:
                    boot_deriv = psi_x_deriv_eval @ (tmp @ (residuals * boot_draws))
                    z_sup_boot_deriv[b, j_idx] = float(xp.max(xp.abs(boot_deriv) / avoid_zero_division(asy_se_deriv)))

        cv_h = None
        cv_deriv = None

        # The margin max{0, log log J} theta is 0 for J up to e. Checking J first avoids log(0) at J of 0 or 1.
        margin = np.log(np.log(j_tilde)) * theta_star if j_tilde is not None and j_tilde > np.e else 0.0

        if ucb_h:
            z_boot_h = np.max(z_sup_boot_h, axis=1)
            cv_h = _quantile_basis(z_boot_h, 1 - alpha) + margin

        if ucb_deriv:
            z_boot_deriv = np.max(z_sup_boot_deriv, axis=1)
            cv_deriv = _quantile_basis(z_boot_deriv, 1 - alpha) + margin

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
