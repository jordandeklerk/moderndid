"""Core nonparametric instrumental variables estimation functions."""

import warnings

import numpy as np

from ..utils import is_full_rank
from .prodspline import prodspline
from .results import NPIVResult


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
    r"""Nonparametric instrumental variables estimation with B-splines.

    Estimates the structural function :math:`h_0(x)` in the nonparametric IV model,
    identified by the conditional moment restriction

    .. math::

        \mathbb{E}[Y - h_0(X) \mid W] = 0 \quad (\text{almost surely}),

    where :math:`X` is a vector of endogenous regressors and :math:`W` is a vector of
    instrumental variables. The function :math:`h_0` is approximated by a linear
    combination of :math:`J` basis functions (a sieve), such that
    :math:`h_0(x) \approx (\psi^J(x))' c_J`.

    The coefficients :math:`c_J` are estimated using a two-stage least squares (TSLS)
    regression of :math:`Y` on the basis functions of :math:`X`, using :math:`K` basis
    functions of :math:`W` as instruments, where :math:`K \geq J`.

    The TSLS estimator for the coefficients is given by

    .. math::

        \hat{c}_J = (\boldsymbol{\Psi}_{J}' \mathbf{P}_{K} \boldsymbol{\Psi}_{J})^{-}
        \boldsymbol{\Psi}_{J}' \mathbf{P}_{K} \mathbf{Y},

    where :math:`\boldsymbol{\Psi}_{J}` is the :math:`n \times J` matrix of basis functions for :math:`X`,
    :math:`\mathbf{B}_{K}` is the :math:`n \times K` matrix of basis functions for :math:`W`,
    :math:`\mathbf{P}_{K} = \mathbf{B}_{K}(\mathbf{B}_{K}' \mathbf{B}_{K})^{-} \mathbf{B}_{K}'` is the projection matrix
    onto the instrument space, and :math:`\mathbf{Y}` is the :math:`n \times 1` vector of outcomes.

    The estimator for :math:`h_0(x)` and its derivative :math:`\partial^a h_0(x)` are then

    .. math::

        \hat{h}_J(x) = (\psi^J(x))' \hat{c}_J, \quad \partial^a \hat{h}_J(x) = (\partial^a \psi^J(x))' \hat{c}_J.

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
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate X:

        - "tensor": Full tensor product
        - "additive": Additive components
        - "glp": Generalized linear product
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
    deriv_index : int, default=1
        Index (1-based) of :math:`X` variable for derivative computation.
    deriv_order : int, default=1
        Order of derivative to compute.
    check_is_fullrank : bool, default=False
        Whether to check if design matrices have full rank.
    w_min : float, optional
        Minimum value for :math:`W` range.
    w_max : float, optional
        Maximum value for :math:`W` range.
    x_min : float, optional
        Minimum value for :math:`X` range.
    x_max : float, optional
        Maximum value for :math:`X` range.

    Returns
    -------
    NPIVResult
        NamedTuple containing estimation results.

    References
    ----------

    .. [1] Chen, X., & Christensen, T. M. (2018). Optimal sup-norm rates and
        uniform inference on nonlinear functionals of nonparametric IV regression.
        Quantitative Economics, 9(1), 39-84. https://arxiv.org/abs/1508.03365.

    .. [2] Chen, X., Christensen, T. M., & Tamer, E. (2018). Monte Carlo confidence
        sets for identified sets. Econometrica, 86(6), 1965-2018.
    """
    y = np.asarray(y).ravel()
    x = np.atleast_2d(x)
    w = np.atleast_2d(w)

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
        x_eval = np.atleast_2d(x_eval)
        if x_eval.shape[1] != p_x:
            raise ValueError("x_eval must have same number of columns as x")
        if np.array_equal(x, x_eval):
            train_is_eval = True

    n_eval = x_eval.shape[0]

    is_regression_case = np.array_equal(x, w)

    j_x_segments, k_w_segments, k_w_degree = _determine_segments(
        j_x_segments, k_w_segments, j_x_degree, k_w_degree, n, p_x, p_w, is_regression_case, data_driven
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
    except ValueError as e:
        raise ValueError(f"Invalid parameters for B-spline construction: {e}") from e
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Numerical error constructing B-spline bases: {e}") from e

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

    # NPIV estimation
    try:
        beta, gram_inv, design_matrix = _perform_tsls_estimation(psi_x, b_w, y)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Numerical error in NPIV estimation: {e}") from e

    h_estimates = psi_x_eval @ beta
    deriv_estimates = psi_x_deriv_eval @ beta
    residuals = y - psi_x @ beta

    asy_se, deriv_asy_se, tmp = _compute_asymptotic_standard_errors(
        psi_x_eval, psi_x_deriv_eval, gram_inv, design_matrix, residuals, n_eval
    )

    args = {
        "n_obs": n,
        "n_eval": n_eval,
        "p_x": p_x,
        "p_w": p_w,
        "basis_type": basis,
        "knots_type": knots,
        "psi_x_dim": psi_x.shape[1],
        "b_w_dim": b_w.shape[1],
        "residual_mse": np.mean(residuals**2),
        "train_is_eval": train_is_eval,
        "tmp": tmp,
        "psi_x_eval": psi_x_eval,
        "psi_x_deriv_eval": psi_x_deriv_eval,
        "b_w": b_w,
        "b_w_deriv": basis_matrices["b_w_deriv"],
    }

    return NPIVResult(
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

    psi_x_deriv_result = prodspline(
        x=x,
        K=K_x,
        xeval=x,
        knots=knots,
        basis=basis,
        deriv_index=deriv_index,
        deriv=deriv_order,
        x_min=np.full(p_x, x_min) if x_min is not None else None,
        x_max=np.full(p_x, x_max) if x_max is not None else None,
    )
    psi_x_deriv = psi_x_deriv_result.basis

    if train_is_eval:
        psi_x_deriv_eval = psi_x_deriv.copy()
    else:
        psi_x_deriv_eval_result = prodspline(
            x=x,
            K=K_x,
            xeval=x_eval,
            knots=knots,
            basis=basis,
            deriv_index=deriv_index,
            deriv=deriv_order,
            x_min=np.full(p_x, x_min) if x_min is not None else None,
            x_max=np.full(p_x, x_max) if x_max is not None else None,
        )
        psi_x_deriv_eval = psi_x_deriv_eval_result.basis

    b_w_result = prodspline(
        x=w,
        K=K_w,
        knots=knots,
        basis=basis,
        x_min=np.full(p_w, w_min) if w_min is not None else None,
        x_max=np.full(p_w, w_max) if w_max is not None else None,
    )
    b_w = b_w_result.basis

    b_w_deriv_result = prodspline(
        x=w,
        K=K_w,
        xeval=w,
        knots=knots,
        basis=basis,
        deriv_index=1,
        deriv=1,
        x_min=np.full(p_w, w_min) if w_min is not None else None,
        x_max=np.full(p_w, w_max) if w_max is not None else None,
    )
    b_w_deriv = b_w_deriv_result.basis

    if basis in ("additive", "glp"):
        psi_x = np.c_[np.ones(n), psi_x]
        psi_x_eval = np.c_[np.ones(n_eval), psi_x_eval]
        psi_x_deriv = np.c_[np.zeros(n), psi_x_deriv]
        psi_x_deriv_eval = np.c_[np.zeros(n_eval), psi_x_deriv_eval]
        b_w = np.c_[np.ones(n), b_w]
        b_w_deriv = np.c_[np.zeros(n), b_w_deriv]

    return {
        "psi_x": psi_x,
        "psi_x_eval": psi_x_eval,
        "psi_x_deriv": psi_x_deriv,
        "psi_x_deriv_eval": psi_x_deriv_eval,
        "b_w": b_w,
        "b_w_deriv": b_w_deriv,
    }


def _perform_tsls_estimation(psi_x, b_w, y):
    """Perform two-stage least squares estimation."""
    btb = b_w.T @ b_w
    btb_inv = _ginv(btb)

    # projection matrix onto instrument space
    p_w = b_w @ btb_inv @ b_w.T

    design_matrix = psi_x.T @ p_w

    gram_matrix = design_matrix @ psi_x
    gram_inv = _ginv(gram_matrix)

    beta = gram_inv @ design_matrix @ y

    return beta, gram_inv, design_matrix


def _compute_asymptotic_standard_errors(psi_x_eval, psi_x_deriv_eval, gram_inv, design_matrix, residuals, n_eval):
    """Compute asymptotic standard errors for estimates and derivatives."""
    try:
        tmp = gram_inv @ design_matrix

        weighted_tmp = tmp.T * residuals[:, np.newaxis]
        D_inv_rho_D_inv = weighted_tmp.T @ weighted_tmp

        var_matrix = psi_x_eval @ D_inv_rho_D_inv @ psi_x_eval.T
        asy_se = np.sqrt(np.abs(np.diag(var_matrix)))

        var_matrix_deriv = psi_x_deriv_eval @ D_inv_rho_D_inv @ psi_x_deriv_eval.T
        deriv_asy_se = np.sqrt(np.abs(np.diag(var_matrix_deriv)))

        return asy_se, deriv_asy_se, tmp

    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Error computing asymptotic standard errors: {e}", UserWarning)
        return np.full(n_eval, np.nan), np.full(n_eval, np.nan), None


def _determine_segments(
    j_x_segments, k_w_segments, j_x_degree, k_w_degree, n, p_x, p_w, is_regression_case, data_driven
):
    """Determine the number of segments for basis functions."""
    if not data_driven:
        if j_x_segments is None:
            j_x_segments = max(3, min(int(np.ceil(n ** (1 / (2 * j_x_degree + p_x)))), 10))
        if k_w_segments is None:
            if is_regression_case:
                k_w_segments = j_x_segments
            else:
                k_w_segments = max(3, min(int(np.ceil(n ** (1 / (2 * k_w_degree + p_w)))), 10))

    if is_regression_case:
        k_w_degree = j_x_degree
        if not data_driven:
            k_w_segments = j_x_segments

    return j_x_segments, k_w_segments, k_w_degree


def _ginv(X, tol=None):
    """Generalized matrix inverse."""
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T

    positive = s > max(tol * s[0] if len(s) > 0 else 0, 0)

    if np.all(positive):
        return V @ np.diag(1 / s) @ U.T
    if not np.any(positive):
        return np.zeros((X.shape[1], X.shape[0]))
    return V[:, positive] @ np.diag(1 / s[positive]) @ U[:, positive].T
