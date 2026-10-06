"""Core nonparametric instrumental variables estimation functions."""

import warnings

import numpy as np

from ..cupy.backend import get_backend, to_numpy
from .container import NPIVResult
from .prodspline import prodspline
from .utils import _as_eval_points, _as_matrix, is_full_rank


def npiv_est(
    y,
    x,
    w,
    x_eval=None,
    basis="tensor",
    j_x_degree=3,
    j_x_segments=None,
    k_w_degree=4,
    k_w_segments=None,
    k_w_smooth=2,
    knots="uniform",
    deriv_index=1,
    deriv_order=1,
    check_is_fullrank=False,
    w_min=None,
    w_max=None,
    x_min=None,
    x_max=None,
    data_driven=False,
):
    r"""Core sieve TSLS estimation for the nonparametric IV model.

    Computational backend for :func:`npiv` that constructs B-spline bases
    for :math:`X` and :math:`W`, estimates the sieve coefficient vector
    :math:`\hat{c}_J` by two-stage least squares, and evaluates the function
    estimate :math:`\hat{h}_J` and its derivatives at the requested points.

    The instrument basis needs at least as many columns as the basis for
    :math:`X`. Since the TSLS problem has no unique solution with fewer
    columns, the function raises an error instead of returning one of many fits.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Outcome variable.
    x : ndarray of shape (n,) or (n, p_x)
        Endogenous regressors. A 1-d array is treated as a single regressor.
    w : ndarray of shape (n,) or (n, p_w)
        Instrumental variables. A 1-d array is treated as a single instrument.
    x_eval : ndarray of shape (m, p_x), optional
        Evaluation points for :math:`\hat{h}`. If None, uses ``x``. A 1-d
        array holds m points when there is one regressor.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Multivariate basis construction for :math:`X`.
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
    deriv_index : int, default=1
        Which component of :math:`X` to differentiate (1-based).
    deriv_order : int, default=1
        Order of derivative to compute.
    check_is_fullrank : bool, default=False
        Verify full column rank of basis matrices before estimation.
    w_min, w_max : float, optional
        Override support bounds for :math:`W`.
    x_min, x_max : float, optional
        Override support bounds for :math:`X`.
    data_driven : bool, default=False
        Whether the segments come from data-driven selection. When ``w``
        equals ``x``, the instrument segments are then kept as given instead
        of being set to ``j_x_segments``.

    Returns
    -------
    NPIVResult
        Named tuple containing estimation results.

    Examples
    --------
    Estimate the Engel curve without confidence bands using the core TSLS
    estimator. This is the computational backend that :func:`npiv` calls
    internally:

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import load_engel
           ...: from moderndid.npiv import npiv_est
           ...:
           ...: df = load_engel()
           ...: y = df["food"].to_numpy()
           ...: x = df["logexp"].to_numpy().reshape(-1, 1)
           ...: w = df["logwages"].to_numpy().reshape(-1, 1)
           ...: result = npiv_est(y=y, x=x, w=w, j_x_segments=5)
           ...: print(f"Estimates at {len(result.h)} points")
           ...: print(f"h[:5] = {result.h[:5]}")
           ...: print(f"Pointwise SE[:5] = {result.asy_se[:5]}")

    Evaluate the estimate on a uniform grid of 50 points:

    .. ipython::
        :okwarning:

        In [2]: x_grid = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
           ...: result = npiv_est(y=y, x=x, w=w, x_eval=x_grid, j_x_segments=5)
           ...: print(f"Grid estimates: {len(result.h)} points")
           ...: print(f"h[:5] = {result.h[:5]}")

    See Also
    --------
    npiv : Public API with data-driven selection and confidence bands.
    """
    result, _ = _fit_sieve_tsls(
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
        data_driven=data_driven,
    )
    return result


def _fit_sieve_tsls(
    y,
    x,
    w,
    x_eval,
    basis,
    j_x_degree,
    j_x_segments,
    k_w_degree,
    k_w_segments,
    k_w_smooth,
    knots,
    deriv_index,
    deriv_order,
    check_is_fullrank,
    w_min,
    w_max,
    x_min,
    x_max,
    data_driven,
):
    """Fit the sieve TSLS estimator and return the influence and evaluation matrices next to the result."""
    xp = get_backend()
    y = xp.asarray(y).ravel()
    x = _as_matrix(x)
    w = _as_matrix(w)

    n = len(y)
    if x.shape[0] != n or w.shape[0] != n:
        raise ValueError("All input arrays must have the same number of observations")

    p_x = x.shape[1]
    p_w = w.shape[1]

    train_is_eval = False
    if x_eval is None:
        x_eval = x.copy()
        train_is_eval = True
    else:
        x_eval = _as_eval_points(x_eval, p_x)
        if x_eval.shape[1] != p_x:
            raise ValueError("x_eval must have same number of columns as x")
        if np.array_equal(to_numpy(x), to_numpy(x_eval)):
            train_is_eval = True

    n_eval = x_eval.shape[0]

    is_regression_case = np.array_equal(to_numpy(x), to_numpy(w))

    j_x_segments, k_w_segments, k_w_degree = _determine_segments(
        j_x_segments, k_w_segments, j_x_degree, k_w_degree, k_w_smooth, n, p_x, is_regression_case, data_driven
    )

    K_x = np.column_stack([np.full(p_x, j_x_degree), np.full(p_x, j_x_segments - 1)])
    K_w = np.column_stack([np.full(p_w, k_w_degree), np.full(p_w, k_w_segments - 1)])

    try:
        basis_matrices = _construct_basis_matrices(
            x,
            w,
            x_eval,
            K_x,
            K_w,
            knots,
            basis,
            x_min,
            x_max,
            w_min,
            w_max,
            deriv_index,
            deriv_order,
            n,
            n_eval,
            p_x,
            p_w,
            train_is_eval,
        )
        psi_x = basis_matrices["psi_x"]
        psi_x_eval = basis_matrices["psi_x_eval"]
        psi_x_deriv_eval = basis_matrices["psi_x_deriv_eval"]
        b_w = basis_matrices["b_w"]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Numerical error constructing B-spline bases: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid parameters for B-spline construction: {e}") from e

    if b_w.shape[1] < psi_x.shape[1]:
        raise ValueError(
            f"The instrument basis has {b_w.shape[1]} columns but the basis for x has {psi_x.shape[1]}, so the "
            "sieve TSLS estimate is not identified. Increase k_w_segments or k_w_degree, or lower j_x_segments."
        )

    if check_is_fullrank:
        rank_check_psi = is_full_rank(psi_x)
        rank_check_bw = is_full_rank(b_w)

        if not rank_check_psi.is_full_rank:
            warnings.warn(
                f"Psi_x matrix may not have full rank (condition number: {rank_check_psi.condition_number:.2e})",
                UserWarning,
            )

        if not rank_check_bw.is_full_rank:
            warnings.warn(
                f"B_w matrix may not have full rank (condition number: {rank_check_bw.condition_number:.2e})",
                UserWarning,
            )

    try:
        beta, tmp = _perform_tsls_estimation(psi_x, b_w, y)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Numerical error in NPIV estimation: {e}") from e

    h_estimates = psi_x_eval @ beta
    deriv_estimates = psi_x_deriv_eval @ beta
    residuals = y - psi_x @ beta

    asy_se, deriv_asy_se = _compute_asymptotic_standard_errors(psi_x_eval, psi_x_deriv_eval, tmp, residuals, n_eval)

    args = {
        "n_obs": n,
        "n_eval": n_eval,
        "p_x": p_x,
        "p_w": p_w,
        "basis_type": basis,
        "knots_type": knots,
        "psi_x_dim": psi_x.shape[1],
        "b_w_dim": b_w.shape[1],
        "residual_mse": xp.mean(residuals**2),
        "train_is_eval": train_is_eval,
    }

    result = NPIVResult(
        h=h_estimates,
        h_lower=None,
        h_upper=None,
        deriv=deriv_estimates,
        h_lower_deriv=None,
        h_upper_deriv=None,
        beta=beta,
        asy_se=asy_se,
        deriv_asy_se=deriv_asy_se,
        cv=None,
        cv_deriv=None,
        residuals=residuals,
        j_x_degree=j_x_degree,
        j_x_segments=j_x_segments,
        k_w_degree=k_w_degree,
        k_w_segments=k_w_segments,
        args=args,
    )
    return result, (tmp, psi_x_eval, psi_x_deriv_eval)


def _construct_basis_matrices(
    x,
    w,
    x_eval,
    K_x,
    K_w,
    knots,
    basis,
    x_min,
    x_max,
    w_min,
    w_max,
    deriv_index,
    deriv_order,
    n,
    n_eval,
    p_x,
    p_w,
    train_is_eval,
):
    """Construct all B-spline basis matrices needed for NPIV estimation."""
    psi_x_result = prodspline(
        x=x,
        K=K_x,
        knots=knots,
        basis=basis,
        x_min=np.full(p_x, x_min) if x_min is not None else None,
        x_max=np.full(p_x, x_max) if x_max is not None else None,
    )
    psi_x = psi_x_result.basis

    if train_is_eval:
        psi_x_eval = psi_x.copy()
    else:
        psi_x_eval_result = prodspline(
            x=x,
            K=K_x,
            xeval=x_eval,
            knots=knots,
            basis=basis,
            x_min=np.full(p_x, x_min) if x_min is not None else None,
            x_max=np.full(p_x, x_max) if x_max is not None else None,
        )
        psi_x_eval = psi_x_eval_result.basis

    psi_x_deriv_eval = prodspline(
        x=x,
        K=K_x,
        xeval=x if train_is_eval else x_eval,
        knots=knots,
        basis=basis,
        deriv_index=deriv_index,
        deriv=deriv_order,
        x_min=np.full(p_x, x_min) if x_min is not None else None,
        x_max=np.full(p_x, x_max) if x_max is not None else None,
    ).basis

    b_w_result = prodspline(
        x=w,
        K=K_w,
        knots=knots,
        basis=basis,
        x_min=np.full(p_w, w_min) if w_min is not None else None,
        x_max=np.full(p_w, w_max) if w_max is not None else None,
    )
    b_w = b_w_result.basis

    if basis in ("additive", "glp"):
        xp = get_backend()
        psi_x = xp.concatenate([xp.ones((n, 1)), psi_x], axis=1)
        psi_x_eval = xp.concatenate([xp.ones((n_eval, 1)), psi_x_eval], axis=1)
        psi_x_deriv_eval = xp.concatenate([xp.zeros((n_eval, 1)), psi_x_deriv_eval], axis=1)
        b_w = xp.concatenate([xp.ones((n, 1)), b_w], axis=1)

    return {
        "psi_x": psi_x,
        "psi_x_eval": psi_x_eval,
        "psi_x_deriv_eval": psi_x_deriv_eval,
        "b_w": b_w,
    }


def _perform_tsls_estimation(psi_x, b_w, y):
    """Perform two-stage least squares estimation."""
    btb_inv = _ginv(b_w.T @ b_w)

    # Multiplying from the left keeps every product at J x n or smaller and never forms the n x n projection.
    design_matrix = ((psi_x.T @ b_w) @ btb_inv) @ b_w.T

    gram_inv = _ginv(design_matrix @ psi_x)
    tmp = gram_inv @ design_matrix
    beta = tmp @ y

    return beta, tmp


def _compute_asymptotic_standard_errors(psi_x_eval, psi_x_deriv_eval, tmp, residuals, n_eval):
    """Compute asymptotic standard errors for estimates and derivatives."""
    xp = get_backend()
    try:
        weighted_tmp = tmp.T * residuals[:, None]
        D_inv_rho_D_inv = weighted_tmp.T @ weighted_tmp

        # Row sums give the diagonal of psi D psi' without forming the m x m matrix.
        asy_se = xp.sqrt(xp.abs(xp.sum((psi_x_eval @ D_inv_rho_D_inv) * psi_x_eval, axis=1)))
        deriv_asy_se = xp.sqrt(xp.abs(xp.sum((psi_x_deriv_eval @ D_inv_rho_D_inv) * psi_x_deriv_eval, axis=1)))

        return asy_se, deriv_asy_se

    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Error computing asymptotic standard errors: {e}", UserWarning)
        return xp.full(n_eval, np.nan), xp.full(n_eval, np.nan)


def _determine_segments(
    j_x_segments, k_w_segments, j_x_degree, k_w_degree, k_w_smooth, n, p_x, is_regression_case, data_driven
):
    """Determine the number of segments for basis functions."""
    if not data_driven:
        if j_x_segments is None:
            j_x_segments = max(3, min(int(np.ceil(n ** (1 / (2 * j_x_degree + p_x)))), 10))
        if k_w_segments is None:
            # The instrument basis refines the X basis by k_w_smooth dyadic levels, as on the selection grid.
            k_w_segments = int(j_x_segments * 2**k_w_smooth)

    if is_regression_case:
        k_w_degree = j_x_degree
        if not data_driven:
            k_w_segments = j_x_segments

    return j_x_segments, k_w_segments, k_w_degree


def _ginv(X, tol=None):
    """Generalized matrix inverse."""
    xp = get_backend()
    if tol is None:
        tol = float(np.sqrt(np.finfo(float).eps))

    U, s, Vt = xp.linalg.svd(X, full_matrices=False)
    V = Vt.T

    positive = s > max(float(tol * s[0]) if len(s) > 0 else 0, 0)

    if xp.all(positive):
        return V @ xp.diag(1 / s) @ U.T
    if not xp.any(positive):
        return xp.zeros((X.shape[1], X.shape[0]))
    return V[:, positive] @ xp.diag(1 / s[positive]) @ U[:, positive].T
