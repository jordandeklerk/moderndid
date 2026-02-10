"""Nonparametric instrumental variables estimation."""

import warnings

import numpy as np

from .confidence_bands import compute_ucb
from .estimators import npiv_est
from .selection import npiv_choose_j


def npiv(
    y,
    x,
    w,
    x_eval=None,
    x_grid=None,
    alpha=0.05,
    basis="tensor",
    boot_num=99,
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
    check_is_fullrank=False,
    w_min=None,
    w_max=None,
    x_min=None,
    x_max=None,
    seed=None,
):
    r"""Estimate nonparametric instrumental variables model with uniform confidence bands.

    Estimates the structural function :math:`h_0` and its derivatives in the
    nonparametric IV model

    .. math::
        \mathbb{E}[Y - h_0(X) \mid W] = 0 \quad \text{(a.s.)}

    where :math:`Y` is a scalar outcome, :math:`X` is a (possibly endogenous)
    regressor vector, and :math:`W` is a vector of instrumental variables. The
    function is approximated by a B-spline sieve :math:`h_0(x) \approx (\psi^J(x))' c_J`
    and coefficients are estimated by two-stage least squares using :math:`K`
    B-spline basis functions of :math:`W` as instruments

    .. math::
        \hat{c}_J = (\boldsymbol{\Psi}_J' \mathbf{P}_K \boldsymbol{\Psi}_J)^{-}
        \boldsymbol{\Psi}_J' \mathbf{P}_K \mathbf{Y},

    where :math:`\mathbf{P}_K = \mathbf{B}_K (\mathbf{B}_K' \mathbf{B}_K)^{-} \mathbf{B}_K'`
    projects onto the instrument space. Function and derivative estimates are then given by

    .. math::
        \hat{h}_J(x) = (\psi^J(x))' \hat{c}_J, \quad
        \partial^a \hat{h}_J(x) = (\partial^a \psi^J(x))' \hat{c}_J.

    When ``j_x_segments`` is None, a bootstrap implementation of Lepski's
    method selects the sieve dimension :math:`\tilde{J}` that adapts to the
    unknown smoothness of :math:`h_0` and instrument strength, achieving the
    minimax sup-norm convergence rate for both :math:`h_0` and its derivatives.

    The adaptive CCK procedure then constructs honest uniform confidence bands
    that guarantee coverage uniformly over a class of data-generating processes.
    When a fixed ``j_x_segments`` is supplied, the standard undersmoothing approach
    of [1]_ is used instead.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Outcome variable.
    x : ndarray of shape (n,) or (n, p_x)
        Endogenous regressors. Automatically promoted to 2-d if needed.
    w : ndarray of shape (n,) or (n, p_w)
        Instrumental variables. Requires :math:`K \geq J`.
    x_eval : ndarray of shape (m, p_x), optional
        Points at which to evaluate :math:`\hat{h}` and its derivatives. If
        None, evaluates at the sample points ``x``.
    x_grid : ndarray, optional
        Alias for ``x_eval``. Ignored when ``x_eval`` is provided.
    alpha : float, default=0.05
        Significance level for :math:`100(1-\alpha)\%` confidence bands.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Multivariate basis construction for :math:`X`:

        - ``"tensor"``: Full tensor product of univariate B-splines.
        - ``"additive"``: Sum of univariate B-splines (additive model).
        - ``"glp"``: Generalized linear product (hierarchical interactions).
    boot_num : int, default=99
        Number of multiplier bootstrap draws for critical value computation.
        Each draw generates i.i.d. :math:`N(0,1)` weights
        :math:`(\varpi_i)_{i=1}^n` to form bootstrap sup-:math:`t` statistics.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X` (order
        :math:`r = \text{degree} + 1`). For UCBs of first derivatives, degree
        :math:`\geq 2` is required; for second derivatives, :math:`\geq 3`.
    j_x_segments : int, optional
        Number of segments for the :math:`X` basis, determining sieve dimension
        :math:`J`. When None, the data-driven Lepski procedure selects
        :math:`\tilde{J}` adaptively. Supplying a fixed value triggers the
        undersmoothing UCB approach.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`. Defaults to
        ``j_x_degree + 1`` because the reduced form
        :math:`\mathbb{E}[h_0(X) \mid W]` is smoother than :math:`h_0`.
    k_w_segments : int, optional
        Number of segments for the instrument basis. When None, chosen
        proportionally to ``j_x_segments`` via the resolution-level mapping
        :math:`l_w = \lceil (l + q) \, d / d_w \rceil`, where :math:`q` is controlled by ``k_w_smooth``.
    k_w_smooth : int, default=2
        Controls the resolution gap :math:`q` between the :math:`X` and
        :math:`W` bases in the data-driven procedure. Larger values yield more
        instrument basis functions relative to the :math:`X` basis.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement strategy:

        - ``"uniform"``: Equally spaced knots on the support.
        - ``"quantiles"``: Knots at empirical quantiles of the data.
    ucb_h : bool, default=True
        Compute uniform confidence bands for :math:`\hat{h}`.
    ucb_deriv : bool, default=True
        Compute uniform confidence bands for :math:`\partial^a \hat{h}`.
    deriv_index : int, default=1
        Which component of :math:`X` to differentiate with respect to
        (1-based indexing).
    deriv_order : int, default=1
        Order :math:`|a|` of the derivative (1 = first, 2 = second, etc.).
    check_is_fullrank : bool, default=False
        Verify that the basis matrices :math:`\boldsymbol{\Psi}_J` and
        :math:`\mathbf{B}_K` have full column rank before estimation.
    w_min, w_max : float, optional
        Override the support bounds for :math:`W`. Defaults to data range.
    x_min, x_max : float, optional
        Override the support bounds for :math:`X`. Defaults to data range.
    seed : int, optional
        Random seed for bootstrap reproducibility.

    Returns
    -------
    NPIVResult
        Named tuple with the following fields:

        - **h** -- Estimated :math:`\hat{h}_J(x)` at evaluation points.
        - **deriv** -- Estimated :math:`\partial^a \hat{h}_J(x)`.
        - **h_lower**, **h_upper** -- Lower/upper UCB for :math:`h_0`.
        - **h_lower_deriv**, **h_upper_deriv** -- Lower/upper UCB for
          :math:`\partial^a h_0`.
        - **beta** -- Sieve coefficient vector :math:`\hat{c}_J`.
        - **asy_se** -- Pointwise asymptotic standard errors
          :math:`\hat{\sigma}_J(x)`.
        - **deriv_asy_se** -- Pointwise asymptotic standard errors
          :math:`\hat{\sigma}_J^a(x)` for derivatives.
        - **cv**, **cv_deriv** -- Bootstrap critical values
          :math:`z_{1-\alpha}^*` used for band construction.
        - **residuals** -- TSLS residuals
          :math:`\hat{u}_{i,J} = Y_i - \hat{h}_J(X_i)`.
        - **j_x_degree**, **j_x_segments** -- Basis parameters for :math:`X`
          (segments may differ from input when data-driven).
        - **k_w_degree**, **k_w_segments** -- Basis parameters for :math:`W`.
        - **args** -- Diagnostic dictionary. When data-driven selection is
          used, includes ``j_x_seg``, ``k_w_seg``, ``j_hat_max``,
          ``theta_star``, and other selection diagnostics.

    See Also
    --------
    npiv_est : Core sieve TSLS estimation (no confidence bands).
    compute_ucb : Multiplier bootstrap confidence band construction.
    npiv_choose_j : Data-driven sieve dimension selection.

    References
    ----------

    .. [1] Chen, X., & Christensen, T. M. (2018). Optimal sup-norm rates and
        uniform inference on nonlinear functionals of nonparametric IV
        regression. *Quantitative Economics*, 9(1), 39-84.

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive
        estimation and uniform confidence bands for nonparametric structural
        functions and elasticities. *Review of Economic Studies*.
        https://arxiv.org/abs/2107.11869.

    .. [3] Newey, W. K., & Powell, J. L. (2003). Instrumental variable
        estimation of nonparametric models. *Econometrica*, 71(5), 1565-1578.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    w = np.asarray(w)

    if y.ndim > 1:
        y = y.ravel()
        if len(y) != y.size:
            raise ValueError("y must be a 1-dimensional array")

    x = np.atleast_2d(x)
    w = np.atleast_2d(w)

    n = len(y)
    if x.shape[0] != n or w.shape[0] != n:
        raise ValueError("All input arrays must have the same number of observations")

    p_x = x.shape[1]

    if x_eval is None and x_grid is not None:
        warnings.warn("Using x_grid as x_eval", UserWarning)
        x_eval = x_grid

    if x_eval is not None:
        x_eval = np.atleast_2d(x_eval)
        if x_eval.shape[1] != p_x:
            raise ValueError("x_eval must have same number of columns as x")

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

    if boot_num < 1:
        raise ValueError("boot_num must be positive")

    if j_x_degree < 0:
        raise ValueError("j_x_degree must be non-negative")

    if k_w_degree < 0:
        raise ValueError("k_w_degree must be non-negative")

    if deriv_order < 0:
        raise ValueError("deriv_order must be non-negative")

    if deriv_index < 1 or deriv_index > p_x:
        raise ValueError(f"deriv_index must be between 1 and {p_x}")

    if basis not in ("tensor", "additive", "glp"):
        raise ValueError("basis must be one of: 'tensor', 'additive', 'glp'")

    if knots not in ("uniform", "quantiles"):
        raise ValueError("knots must be 'uniform' or 'quantiles'")

    if n < 50:
        warnings.warn(f"Small sample size (n={n}) may lead to unreliable results", UserWarning)

    if 0 < j_x_degree < deriv_order:
        warnings.warn(
            f"deriv_order ({deriv_order}) > j_x_degree ({j_x_degree}), derivative will be zero everywhere",
            UserWarning,
        )

    data_driven = j_x_segments is None
    selection_result = None
    if data_driven:
        try:
            selection_result = npiv_choose_j(
                y=y,
                x=x,
                w=w,
                x_grid=x_grid,
                j_x_degree=j_x_degree,
                k_w_degree=k_w_degree,
                k_w_smooth=k_w_smooth,
                knots=knots,
                basis=basis,
                x_min=x_min,
                x_max=x_max,
                w_min=w_min,
                w_max=w_max,
                grid_num=50,
                boot_num=boot_num if boot_num > 0 else 99,
                check_is_fullrank=check_is_fullrank,
                seed=seed,
            )
            j_x_segments = selection_result["j_x_seg"]
            k_w_segments = selection_result["k_w_seg"]

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Data-driven selection failed: {e}. Using default values.",
                UserWarning,
            )
            j_x_segments = max(3, min(int(np.ceil(n ** (1 / (2 * j_x_degree + p_x)))), 10))
            k_w_segments = None

    args = {"data_driven": data_driven}
    if selection_result:
        args.update(selection_result)

    if ucb_h or ucb_deriv:
        result = compute_ucb(
            y=y,
            x=x,
            w=w,
            x_eval=x_eval,
            alpha=alpha,
            boot_num=boot_num,
            basis=basis,
            j_x_degree=j_x_degree,
            j_x_segments=j_x_segments,
            k_w_degree=k_w_degree,
            k_w_segments=k_w_segments,
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
    else:
        result = npiv_est(
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
            check_is_fullrank=check_is_fullrank,
            w_min=w_min,
            w_max=w_max,
            x_min=x_min,
            x_max=x_max,
        )

    if selection_result:
        result.args.update(selection_result)
        result.args["data_driven"] = True

    return result
