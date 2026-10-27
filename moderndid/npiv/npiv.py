"""Nonparametric instrumental variables estimation."""

import warnings

import numpy as np

from ..core.dataframe import to_polars
from .confidence_bands import compute_ucb
from .estimators import npiv_est
from .selection import npiv_choose_j


def npiv(
    data=None,
    yname=None,
    xname=None,
    wname=None,
    y=None,
    x=None,
    w=None,
    x_eval=None,
    x_grid=None,
    alpha=0.05,
    basis="tensor",
    biters=99,
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
    r"""Estimate a nonparametric instrumental variables model with uniform confidence bands.

    Estimates a structural function :math:`h_0` and its derivatives for a scalar outcome :math:`Y`
    when the regressors :math:`X` may be endogenous. With instruments :math:`W`, the function
    satisfies the conditional moment restriction

    .. math::

        \mathbb{E}[Y - h_0(X) \mid W] = 0 \quad \text{(a.s.)}.

    As in the sieve approach of [1]_, two-stage least squares on B-splines in :math:`W` estimates
    the coefficients of B-splines in :math:`X` that approximate :math:`h_0`.

    When ``j_x_segments`` is None, :func:`~moderndid.npiv_choose_j` picks the number of segments
    with the bootstrap Lepski procedure of [2]_. The bands from :func:`~moderndid.compute_cck_ucb`
    then add a margin for that choice to their critical value. A fixed ``j_x_segments`` gives the
    undersmoothed bands of [3]_ from :func:`~moderndid.compute_ucb` instead. They are valid only
    when the number of segments is large enough for the approximation bias to be negligible.

    See the :ref:`nonparametric IV example <example_npiv>` for a full analysis of the Engel curve
    data.

    Parameters
    ----------
    data : DataFrame, optional
        Data holding the columns ``yname``, ``xname``, and ``wname``, in place of ``y``, ``x``,
        and ``w``. Accepts any object implementing the Arrow PyCapsule Interface
        (``__arrow_c_stream__``), including polars, pandas, pyarrow Table, and cudf DataFrames.
    yname : str, optional
        The name of the outcome column in ``data``.
    xname, wname : str or list of str, optional
        The names of the regressor and instrument columns in ``data``.
    y : ndarray of shape (n,), optional
        The outcome, required without ``data``.
    x : ndarray of shape (n,) or (n, p_x), optional
        The endogenous regressors, required without ``data``. A 1-d array holds one regressor.
    w : ndarray of shape (n,) or (n, p_w), optional
        The instruments, required without ``data``. A 1-d array holds one instrument.
    x_eval : ndarray of shape (m, p_x), optional
        The points at which to evaluate :math:`\hat{h}` and its derivatives. If None, ``x_grid``
        when it is given and the sample points otherwise. With one regressor, a 1-d array holds
        :math:`m` points.
    x_grid : ndarray of shape (m, p_x), optional
        The points at which the data-driven selection compares sieve dimensions. If None, 50
        equally spaced values between the smallest and largest value of each regressor.
    alpha : float, default=0.05
        The significance level for :math:`100(1-\alpha)\%` confidence bands.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        How the univariate B-splines combine for several regressors, as their full tensor
        product, their sum, or the generalized polynomial basis of :func:`~moderndid.prodspline`.
    biters : int, default=99
        The number of multiplier bootstrap draws for the critical values. Each draw multiplies the
        residuals by independent standard normal weights :math:`(\varpi_i)_{i=1}^n` to form a
        bootstrap sup-:math:`t` statistic.
    j_x_degree : int, default=3
        The degree of the B-spline basis for :math:`X`, one less than its order :math:`r`. The
        data-driven procedures of [2]_ take degree 2 or more, or 3 or more for bands on second
        derivatives.
    j_x_segments : int, optional
        The number of segments of the basis for :math:`X`. It sets the sieve dimension :math:`J`.
        If None, the Lepski procedure selects :math:`\tilde{J}` from the data.
    k_w_degree : int, default=4
        The degree of the B-spline basis for :math:`W`. Since the reduced form
        :math:`\mathbb{E}[h_0(X) \mid W]` is smoother than :math:`h_0`, the default is one above
        the default ``j_x_degree``.
    k_w_segments : int, optional
        The number of segments of the basis for :math:`W` when ``j_x_segments`` is fixed. If
        None, ``j_x_segments * 2**k_w_smooth``.
    k_w_smooth : int, default=2
        The number :math:`q` of dyadic refinements of the basis for :math:`W`. That basis has
        :math:`2^q` times as many segments as the basis for :math:`X`, unless a fixed
        ``j_x_segments`` comes with ``k_w_segments``.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement, equally spaced over the support or at the empirical quantiles of the data.
    ucb_h, ucb_deriv : bool, default=True
        Whether to compute the uniform confidence bands for :math:`\hat{h}` and for
        :math:`\partial^a \hat{h}`.
    deriv_index : int, default=1
        The 1-based position of the regressor to differentiate with respect to.
    deriv_order : int, default=1
        The order :math:`|a|` of the derivative, such as 1 for the first and 2 for the second.
    check_is_fullrank : bool, default=False
        Whether to check that :math:`\boldsymbol{\Psi}_J` and :math:`\mathbf{B}_K` have full
        column rank before estimation.
    w_min, w_max, x_min, x_max : float, optional
        The support bounds of the bases for :math:`W` and :math:`X`. If None, the range of the data.
    seed : int, optional
        The seed of the bootstrap draws, for reproducible selection and bands.

    Returns
    -------
    NPIVResult
        Named tuple with the following fields:

        - **h**: Estimated :math:`\hat{h}_J(x)` at the evaluation points
        - **h_lower**: Lower uniform confidence band for :math:`h_0`
        - **h_upper**: Upper uniform confidence band for :math:`h_0`
        - **deriv**: Estimated :math:`\partial^a \hat{h}_J(x)`
        - **h_lower_deriv**: Lower uniform confidence band for :math:`\partial^a h_0`
        - **h_upper_deriv**: Upper uniform confidence band for :math:`\partial^a h_0`
        - **beta**: Sieve coefficient vector :math:`\hat{c}_J`
        - **asy_se**: Pointwise asymptotic standard errors :math:`\hat{\sigma}_J(x)`
        - **deriv_asy_se**: Pointwise asymptotic standard errors :math:`\hat{\sigma}_J^a(x)` of
          the derivatives
        - **cv**: Critical value of the function bands, the bootstrap quantile
          :math:`z_{1-\alpha}^*` for a fixed dimension and that quantile plus the selection
          margin of :func:`~moderndid.compute_cck_ucb` otherwise
        - **cv_deriv**: Critical value of the derivative bands, built the same way as ``cv``
        - **residuals**: Two-stage least squares residuals
          :math:`\hat{u}_{i,J} = Y_i - \hat{h}_J(X_i)`
        - **j_x_degree**: Degree of the basis for :math:`X`
        - **j_x_segments**: Segments of the basis for :math:`X`, the selected value when data-driven
        - **k_w_degree**: Degree of the basis for :math:`W`
        - **k_w_segments**: Segments of the basis for :math:`W`
        - **args**: Diagnostic dictionary that also holds the selection diagnostics when data-driven

    See Also
    --------
    npiv_est : Core sieve TSLS estimation (no confidence bands).
    compute_ucb : Multiplier bootstrap confidence band construction.
    npiv_choose_j : Data-driven sieve dimension selection.

    Notes
    -----
    The sieve approximates :math:`h_0(x) \approx (\psi^J(x))' c_J` with :math:`J` B-spline
    functions of :math:`X`. With :math:`K` B-spline functions of :math:`W` as instruments, the
    two-stage least squares coefficients are

    .. math::

        \hat{c}_J = (\boldsymbol{\Psi}_J' \mathbf{P}_K \boldsymbol{\Psi}_J)^{-}
        \boldsymbol{\Psi}_J' \mathbf{P}_K \mathbf{Y},

    where :math:`\mathbf{P}_K = \mathbf{B}_K (\mathbf{B}_K' \mathbf{B}_K)^{-} \mathbf{B}_K'`
    projects onto the instrument space. The estimates of the function and its derivatives are

    .. math::

        \hat{h}_J(x) = (\psi^J(x))' \hat{c}_J, \quad
        \partial^a \hat{h}_J(x) = (\partial^a \psi^J(x))' \hat{c}_J.

    Since the estimate is otherwise not identified, :math:`K < J` raises an error. When ``w``
    equals ``x``, the basis for :math:`W` takes the degree and segments of the basis for
    :math:`X`. As long as ``w_min`` and ``w_max`` equal ``x_min`` and ``x_max``, the fit is then a
    least squares regression. The :ref:`nonparametric IV background <background-npiv>` derives
    the selection of :math:`J` and the bands.

    References
    ----------

    .. [1] Newey, W. K., & Powell, J. L. (2003). Instrumental variable
        estimation of nonparametric models. *Econometrica*, 71(5), 1565-1578.

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive
        estimation and uniform confidence bands for nonparametric structural
        functions and elasticities. *Review of Economic Studies*, 92(1),
        162-196. https://doi.org/10.1093/restud/rdae025.

    .. [3] Chen, X., & Christensen, T. M. (2018). Optimal sup-norm rates and
        uniform inference on nonlinear functionals of nonparametric IV
        regression. *Quantitative Economics*, 9(1), 39-84.
    """
    if data is not None:
        if y is not None or x is not None or w is not None:
            raise ValueError("Cannot specify both 'data' and array arguments (y, x, w)")
        if yname is None or xname is None or wname is None:
            raise ValueError("When 'data' is provided, 'yname', 'xname', and 'wname' are required")

        df = to_polars(data)
        y = df[yname].to_numpy()
        if isinstance(xname, str):
            xname = [xname]
        x = df.select(xname).to_numpy()
        if isinstance(wname, str):
            wname = [wname]
        w = df.select(wname).to_numpy()
    elif y is None or x is None or w is None:
        raise ValueError("Must provide either 'data' with column names, or array arguments (y, x, w)")

    y = np.asarray(y)
    x = np.asarray(x)
    w = np.asarray(w)

    if y.ndim > 1:
        y = y.ravel()
        if len(y) != y.size:
            raise ValueError("y must be a 1-dimensional array")

    # A 1-d array holds the n observations of a single regressor or instrument.
    x = x.reshape(-1, 1) if x.ndim == 1 else np.atleast_2d(x)
    w = w.reshape(-1, 1) if w.ndim == 1 else np.atleast_2d(w)

    n = len(y)
    if x.shape[0] != n or w.shape[0] != n:
        raise ValueError("All input arrays must have the same number of observations")

    p_x = x.shape[1]

    if x_eval is None and x_grid is not None:
        warnings.warn("Using x_grid as x_eval", UserWarning)
        x_eval = x_grid

    if x_eval is not None:
        x_eval = np.asarray(x_eval)
        if x_eval.ndim == 1:
            # A 1-d array lists evaluation points with one regressor and holds a single point with several.
            x_eval = x_eval.reshape(-1, 1) if p_x == 1 else x_eval.reshape(1, -1)
        x_eval = np.atleast_2d(x_eval)
        if x_eval.shape[1] != p_x:
            raise ValueError("x_eval must have same number of columns as x")

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

    if biters < 1:
        raise ValueError("biters must be positive")

    if j_x_degree < 0:
        raise ValueError("j_x_degree must be non-negative")

    if k_w_degree < 0:
        raise ValueError("k_w_degree must be non-negative")

    if k_w_smooth < 0:
        raise ValueError("k_w_smooth must be non-negative")

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

    if np.array_equal(x, w):
        # Setting w to x asks for a regression. The instrument basis then takes the degree and segments of the X basis.
        k_w_degree = j_x_degree
        k_w_smooth = 0

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
                biters=biters if biters > 0 else 99,
                check_is_fullrank=check_is_fullrank,
                seed=seed,
            )
            j_x_segments = selection_result["j_x_seg"]
            k_w_segments = selection_result["k_w_seg"]

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            j_x_segments = max(3, min(int(np.ceil(n ** (1 / (2 * j_x_degree + p_x)))), 10))
            k_w_segments = None
            # Exception texts that end in a period would otherwise print two before the next sentence.
            warnings.warn(
                f"Data-driven selection failed: {str(e).rstrip('.')}. Using {j_x_segments} segments for the basis of x "
                "instead.",
                UserWarning,
            )

    if ucb_h or ucb_deriv:
        result = compute_ucb(
            y=y,
            x=x,
            w=w,
            x_eval=x_eval,
            alpha=alpha,
            biters=biters,
            basis=basis,
            j_x_degree=j_x_degree,
            j_x_segments=j_x_segments,
            k_w_degree=k_w_degree,
            k_w_segments=k_w_segments,
            k_w_smooth=k_w_smooth,
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
            k_w_smooth=k_w_smooth,
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
