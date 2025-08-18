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

    Estimates the structural function :math:`h_0(x)` in the nonparametric IV model

    .. math::
        \mathbb{E}[Y - h_0(X) \mid W] = 0

    using B-spline sieves. Provides data-driven dimension selection when sieve dimensions
    are not specified, and constructs uniform confidence bands via multiplier bootstrap.

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
    x_grid : ndarray, optional
        Alternative name for x_eval (for R compatibility). Ignored if x_eval provided.
    alpha : float, default=0.05
        Significance level for confidence bands.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate :math:`X`:

        - "tensor": Full tensor product of univariate bases
        - "additive": Sum of univariate bases
        - "glp": Generalized linear product (hierarchical)
    boot_num : int, default=99
        Number of bootstrap replications for confidence bands.
    j_x_degree : int, default=3
        Degree of B-spline basis for endogenous variable :math:`X`.
    j_x_segments : int, optional
        Number of segments for :math:`X` basis. If None, chosen via data-driven selection.
    k_w_degree : int, default=4
        Degree of B-spline basis for instruments :math:`W`.
    k_w_segments : int, optional
        Number of segments for the instrument basis. If None, chosen proportionally to the number
        of segments for the :math:`X` basis.
    k_w_smooth : int, default=2
        Smoothness parameter for automatic K selection.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method:

        - "uniform": Uniformly spaced knots
        - "quantiles": Knots at empirical quantiles
    ucb_h : bool, default=True
        Whether to compute uniform confidence bands for function estimates.
    ucb_deriv : bool, default=True
        Whether to compute uniform confidence bands for derivative estimates.
    deriv_index : int, default=1
        Index (1-based) of :math:`X` variable for derivative computation.
    deriv_order : int, default=1
        Order of derivative to compute (1=first derivative, 2=second, etc.).
    check_is_fullrank : bool, default=False
        Whether to check if basis matrices have full rank.
    w_min : float, optional
        Minimum value for :math:`W` range. If None, uses data minimum.
    w_max : float, optional
        Maximum value for :math:`W` range. If None, uses data maximum.
    x_min : float, optional
        Minimum value for :math:`X` range. If None, uses data minimum.
    x_max : float, optional
        Maximum value for :math:`X` range. If None, uses data maximum.

    Returns
    -------
    NPIVResult
        NPIVResult object containing:

        - **h**: Function estimates
        - **h_lower**, **h_upper**: Uniform confidence bands for function estimates
        - **deriv**: Derivative estimates
        - **h_lower_deriv**, **h_upper_deriv**: Uniform confidence bands for derivative estimates
        - **beta**: Coefficient vector
        - **asy_se**, **deriv_asy_se**: Asymptotic standard errors
        - **cv**, **cv_deriv**: Critical values for confidence bands
        - **residuals**: Model residuals
        - **j_x_degree**, **j_x_segments**: :math:`X` basis parameters
        - **k_w_degree**, **k_w_segments**: :math:`W` basis parameters
        - **args**: Additional diagnostic information

    Notes
    -----
    The NPIV estimator solves the population moment condition

    .. math::
        \mathbb{E}[W \epsilon] = 0

    by projecting onto the space spanned by the instrument basis functions. For the
    choice of basis dimensions, when `j_x_segments` is not provided, the function uses
    Lepski's method for data-driven selection following [1]_.

    The uniform confidence bands are constructed using the supremum of
    studentized bootstrap statistics, providing simultaneous coverage
    over the entire evaluation domain.

    References
    ----------

    .. [1] Chen, X., & Christensen, T. M. (2018). Optimal sup-norm rates and
        uniform inference on nonlinear functionals of nonparametric IV regression.
        Quantitative Economics, 9(1), 39-84. https://arxiv.org/abs/1508.03365.

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2024).
        Adaptive Estimation and Uniform Confidence Bands for Nonparametric
        Structural Functions and Elasticities. https://arxiv.org/abs/2107.11869.

    .. [3] Newey, W. K., & Powell, J. L. (2003). Instrumental variable estimation of
        nonparametric models. Econometrica, 71(5), 1565-1578.
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
