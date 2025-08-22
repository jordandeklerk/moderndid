"""Lepski method for optimal dimension selection in nonparametric instrumental variables."""

import numpy as np

from ..utils import _quantile_basis, avoid_zero_division, basis_dimension, matrix_sqrt
from .estimators import _ginv, npiv_est
from .prodspline import prodspline


def npiv_j(
    y,
    x,
    w,
    x_grid=None,
    j_x_degree=3,
    k_w_degree=4,
    j_x_segments_set=None,
    k_w_segments_set=None,
    knots="uniform",
    basis="tensor",
    x_min=None,
    x_max=None,
    w_min=None,
    w_max=None,
    grid_num=50,
    boot_num=99,
    alpha=0.5,
    check_is_fullrank=False,
    seed=None,
):
    r"""Implement Lepski's method for optimal sieve dimension selection.

    Implements the bootstrap-based test from [1]_ for selecting the optimal number of
    B-spline basis functions in nonparametric instrumental variables (NPIV) estimation.
    The method compares estimates across a grid of sieve dimensions :math:`\hat{\mathcal{J}}`.
    For each pair :math:`(J, J_2)` with :math:`J_2 > J`, it computes a sup-t-statistic for the
    difference in estimates

    .. math::

        \sup_{x \in \mathcal{X}} \left| \frac{\hat{h}_J(x) - \hat{h}_{J_2}(x)}{\hat{\sigma}_{J, J_2}(x)} \right|.

    The optimal dimension :math:`\hat{J}` is the smallest :math:`J \in \hat{\mathcal{J}}` for which this statistic
    is below a bootstrap critical value :math:`\theta_{1-\hat{\alpha}}^*` for all :math:`J_2 > J`.

    The bootstrap critical value is the :math:`(1-\hat{\alpha})` quantile of the multiplier bootstrap process

    .. math::

        \sup_{\left\{\left(x, J, J_{2}\right) \in \mathcal{X} \times \hat{\mathcal{J}}
        \times \hat{\mathcal{J}}: J_{2}>J\right\}}
        \left|\frac{D_{J}^{*}(x)-D_{J_{2}}^{*}(x)}{\hat{\sigma}_{J, J_{2}}(x)}\right|,

    where :math:`D_J^*(x) = (\psi^J(x))' \mathbf{M}_J \hat{\mathbf{u}}_J^*` is a multiplier bootstrap
    version of the estimation error,
    and :math:`\hat{\sigma}_{J, J_2}^2(x)` is the estimated variance of the difference in estimators.
    This procedure avoids the need to select tuning parameters for the test itself and performs well in practice.

    Parameters
    ----------
    y : ndarray
        Dependent variable vector.
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    x_grid : ndarray, optional
        Grid points for evaluation. If None, created automatically.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`.
    j_x_segments_set : ndarray, optional
        Set of :math:`J` values to test. If None, uses [1, 3, 7, 15, 31, 63].
    k_w_segments_set : ndarray, optional
        Set of :math:`K` values to test. If None, computed from :math:`J` values.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis.
    x_min, x_max, w_min, w_max : float, optional
        Range limits for basis construction.
    grid_num : int, default=50
        Number of grid points for evaluation.
    boot_num : int, default=99
        Number of bootstrap replications.
    alpha : float, default=0.5
        Significance level for test.
    check_is_fullrank : bool, default=False
        Whether to check for full rank.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - **j_tilde**: Selected J value
        - **j_hat**: Unadjusted Lepski choice
        - **j_hat_n**: Truncated value
        - **j_x_seg**: Final selected J segments
        - **k_w_seg**: Corresponding K segments
        - **theta_star**: Bootstrap critical value

    See Also
    --------
    npiv_choose_j : Full data-driven selection procedure
    npiv_jhat_max : Compute maximum feasible dimension

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

    if j_x_segments_set is None:
        j_x_segments_set = np.array([2, 4, 8, 16, 32, 64])
    if k_w_segments_set is None:
        k_w_segments_set = np.array([2, 4, 8, 16, 32, 64])

    if x_grid is None:
        x_grid = np.zeros((grid_num, p_x))
        for j in range(p_x):
            x_grid[:, j] = np.linspace(x[:, j].min(), x[:, j].max(), grid_num)

    j1_j2_pairs = []
    for i, j1 in enumerate(j_x_segments_set):
        for j, j2 in enumerate(j_x_segments_set[i + 1 :], i + 1):
            j1_j2_pairs.append((i, j, j1, j2))

    n_pairs = len(j1_j2_pairs)
    z_sup = np.zeros(n_pairs)
    z_sup_boot = np.zeros((boot_num, n_pairs))

    rng = np.random.default_rng(seed)
    boot_draws_all = rng.normal(0, 1, (boot_num, n))

    for pair_idx, (i, j, j1, j2) in enumerate(j1_j2_pairs):
        k1 = k_w_segments_set[i]
        k2 = k_w_segments_set[j]

        try:
            result_j1 = npiv_est(
                y=y,
                x=x,
                w=w,
                x_eval=x_grid,
                basis=basis,
                j_x_degree=j_x_degree,
                j_x_segments=j1,
                k_w_degree=k_w_degree,
                k_w_segments=k1,
                knots=knots,
                check_is_fullrank=check_is_fullrank,
                w_min=w_min,
                w_max=w_max,
                x_min=x_min,
                x_max=x_max,
            )

            result_j2 = npiv_est(
                y=y,
                x=x,
                w=w,
                x_eval=x_grid,
                basis=basis,
                j_x_degree=j_x_degree,
                j_x_segments=j2,
                k_w_degree=k_w_degree,
                k_w_segments=k2,
                knots=knots,
                check_is_fullrank=check_is_fullrank,
                w_min=w_min,
                w_max=w_max,
                x_min=x_min,
                x_max=x_max,
            )

            # basis and influence matrices for j1
            psi_x_j1_eval, tmp_j1 = _compute_basis_and_influence(
                x, w, x_grid, j1, k1, j_x_degree, k_w_degree, p_x, p_w, knots, basis, x_min, x_max, w_min, w_max
            )

            # basis and influence matrices for j2
            psi_x_j2_eval, tmp_j2 = _compute_basis_and_influence(
                x, w, x_grid, j2, k2, j_x_degree, k_w_degree, p_x, p_w, knots, basis, x_min, x_max, w_min, w_max
            )

            u_j1 = result_j1.residuals
            u_j2 = result_j2.residuals

            z_sup[pair_idx], asy_se, (tmp_j1, tmp_j2, u_j1, u_j2) = _compute_test_statistic(
                result_j1, result_j2, psi_x_j1_eval, psi_x_j2_eval, tmp_j1, tmp_j2, u_j1, u_j2
            )

            z_sup_boot[:, pair_idx] = _bootstrap_comparison(
                psi_x_j1_eval, psi_x_j2_eval, tmp_j1, tmp_j2, u_j1, u_j2, asy_se, boot_draws_all, boot_num
            )

        except (ValueError, np.linalg.LinAlgError):
            z_sup[pair_idx] = np.inf
            z_sup_boot[:, pair_idx] = np.inf

    z_boot_max = np.max(z_sup_boot, axis=1)
    theta_star = _quantile_basis(z_boot_max, 1 - alpha)

    # Lepski selection
    j_seg, test_mat = _select_optimal_dimension(z_sup, j1_j2_pairs, j_x_segments_set, theta_star)

    j_hat = basis_dimension(
        basis=basis,
        degree=np.full(p_x, j_x_degree),
        segments=np.full(p_x, j_seg),
    )

    # truncated value (second-largest j)
    if len(j_x_segments_set) > 1:
        j_seg_n = j_x_segments_set[-2]
    else:
        j_seg_n = j_seg

    j_hat_n = basis_dimension(
        basis=basis,
        degree=np.full(p_x, j_x_degree),
        segments=np.full(p_x, j_seg_n),
    )

    j_x_seg = min(j_seg, j_seg_n)
    j_tilde = min(j_hat, j_hat_n)
    k_w_seg = k_w_segments_set[np.where(j_x_segments_set == j_x_seg)[0][0]]

    return {
        "j_tilde": j_tilde,
        "j_hat": j_hat,
        "j_hat_n": j_hat_n,
        "j_x_seg": j_x_seg,
        "k_w_seg": k_w_seg,
        "theta_star": theta_star,
        "test_matrix": test_mat,
        "z_sup": z_sup,
    }


def npiv_jhat_max(
    x,
    w,
    j_x_degree=3,
    k_w_degree=4,
    k_w_smooth=2,
    knots="uniform",
    basis="tensor",
    x_min=None,
    x_max=None,
    w_min=None,
    w_max=None,
):
    r"""Determine the upper limit of the sieve dimension grid.

    Computes the maximum feasible number of B-spline basis functions, :math:`\hat{J}_{\max}`,
    based on the sample size and an estimate of the sieve measure of ill-posedness, :math:`\hat{s}_J`.
    This serves as the upper bound for the grid of dimensions searched over in the Lepski-style
    selection procedure.

    :math:`\hat{J}_{\max}` is defined as the largest :math:`J` in a dyadic grid :math:`\mathcal{T}`
    that satisfies

    .. math::

        J \sqrt{\log J} \hat{s}_{J}^{-1} \leq c \sqrt{n}

    for a constant :math:`c` (here, 10). The term :math:`\hat{s}_J` is the smallest singular value of
    a matrix related to the instrumented basis functions, which captures the degree of ill-posedness.

    Parameters
    ----------
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    j_x_degree : int, default=3
        Degree of B-spline basis for X.
    k_w_degree : int, default=4
        Degree of B-spline basis for W.
    k_w_smooth : int, default=2
        Smoothness parameter for K selection.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis.
    x_min, x_max, w_min, w_max : float, optional
        Range limits for basis construction.
    check_is_fullrank : bool, default=False
        Whether to check for full rank.

    Returns
    -------
    dict
        Dictionary containing:

        - **j_x_segments_set**: Array of J values to test
        - **k_w_segments_set**: Corresponding K values
        - **j_hat_max**: Maximum feasible dimension
        - **alpha_hat**: Recommended alpha for testing

    See Also
    --------
    npiv_choose_j : Full data-driven selection procedure
    npiv_j : Lepski-style selection procedure

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024).
        Adaptive Estimation and Uniform Confidence Bands for Nonparametric
        Structural Functions and Elasticities. https://arxiv.org/abs/2107.11869.
    """
    x = np.atleast_2d(x)
    w = np.atleast_2d(w)
    n = x.shape[0]
    p_x = x.shape[1]
    p_w = w.shape[1]

    is_regression = np.array_equal(x, w)

    L_max = max(int(np.floor(np.log(n) / np.log(2 * p_x))), 3)
    j_x_segments_set = 2 ** np.arange(L_max + 1)
    k_w_segments_set = 2 ** (np.arange(L_max + 1) + k_w_smooth)

    test_val = np.full(L_max + 1, np.nan)

    for i in range(L_max + 1):
        if i <= 1 or (i > 1 and test_val[i - 2] <= 10 * np.sqrt(n)):
            j_x_segments = j_x_segments_set[i]
            k_w_segments = k_w_segments_set[i]

            if is_regression:
                s_hat_j = max(1, (0.1 * np.log(n)) ** 4)
            else:
                s_hat_j = _compute_sieve_measure(
                    x,
                    w,
                    j_x_segments,
                    k_w_segments,
                    j_x_degree,
                    k_w_degree,
                    p_x,
                    p_w,
                    knots,
                    basis,
                    x_min,
                    x_max,
                    w_min,
                    w_max,
                    n,
                )

            j_x_dim = basis_dimension(
                basis=basis,
                degree=np.full(p_x, j_x_degree),
                segments=np.full(p_x, j_x_segments),
            )

            test_val[i] = j_x_dim * np.sqrt(np.log(j_x_dim)) * max((0.1 * np.log(n)) ** 4, 1 / s_hat_j)
        elif i > 1:
            test_val[i] = test_val[i - 1]

    valid_idx = np.where((test_val[:-1] <= 10 * np.sqrt(n)) & (10 * np.sqrt(n) < test_val[1:]))[0]

    if len(valid_idx) > 0:
        L_hat_max = valid_idx[0] + 1
    else:
        L_hat_max = L_max

    j_x_segments_set = j_x_segments_set[:L_hat_max]
    k_w_segments_set = k_w_segments_set[:L_hat_max]

    j_hat_max = basis_dimension(
        basis=basis,
        degree=np.full(p_x, j_x_degree),
        segments=np.full(p_x, j_x_segments_set[-1]),
    )

    alpha_hat = min(0.5, np.sqrt(np.log(j_hat_max) / j_hat_max))

    return {
        "j_x_segments_set": j_x_segments_set,
        "k_w_segments_set": k_w_segments_set,
        "j_hat_max": j_hat_max,
        "alpha_hat": alpha_hat,
    }


def _compute_sieve_measure(
    x, w, j_x_segments, k_w_segments, j_x_degree, k_w_degree, p_x, p_w, knots, basis, x_min, x_max, w_min, w_max, n
):
    """Compute sieve measure of ill-posedness."""
    try:
        K_x = np.column_stack([np.full(p_x, j_x_degree), np.full(p_x, j_x_segments - 1)])
        K_w = np.column_stack([np.full(p_w, k_w_degree), np.full(p_w, k_w_segments - 1)])

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

        psi_x_gram_sqrt = matrix_sqrt(_ginv(psi_x.T @ psi_x))
        b_w_gram_sqrt = matrix_sqrt(_ginv(b_w.T @ b_w))

        svd_matrix = psi_x_gram_sqrt @ (psi_x.T @ b_w) @ b_w_gram_sqrt
        s_hat_j = np.min(np.linalg.svd(svd_matrix, compute_uv=False))

    except (ValueError, np.linalg.LinAlgError):
        s_hat_j = max(1, (0.1 * np.log(n)) ** 4)

    return s_hat_j


def _compute_basis_and_influence(
    x, w, x_grid, j_seg, k_seg, j_x_degree, k_w_degree, p_x, p_w, knots, basis, x_min, x_max, w_min, w_max
):
    """Compute basis matrices and influence components for a given dimension pair."""
    K_x = np.column_stack([np.full(p_x, j_x_degree), np.full(p_x, j_seg - 1)])
    K_w = np.column_stack([np.full(p_w, k_w_degree), np.full(p_w, k_seg - 1)])

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
        xeval=x_grid,
        K=K_x,
        knots=knots,
        basis=basis,
        x_min=np.full(p_x, x_min) if x_min else None,
        x_max=np.full(p_x, x_max) if x_max else None,
    ).basis

    # influence matrices
    btb_inv = _ginv(b_w.T @ b_w)
    design_matrix = psi_x.T @ b_w @ btb_inv @ b_w.T
    gram_inv = _ginv(design_matrix @ psi_x)
    tmp_matrix = gram_inv @ design_matrix

    return psi_x_eval, tmp_matrix


def _compute_test_statistic(result_j1, result_j2, psi_x_eval_j1, psi_x_eval_j2, tmp_j1, tmp_j2, u_j1, u_j2):
    """Compute sup-t test statistic for comparing two estimators."""
    # variance components
    D_j1_inv_rho = tmp_j1.T * u_j1[:, np.newaxis]
    D_j1_var = D_j1_inv_rho.T @ D_j1_inv_rho
    var_j1 = np.diag(psi_x_eval_j1 @ D_j1_var @ psi_x_eval_j1.T)

    D_j2_inv_rho = tmp_j2.T * u_j2[:, np.newaxis]
    D_j2_var = D_j2_inv_rho.T @ D_j2_inv_rho
    var_j2 = np.diag(psi_x_eval_j2 @ D_j2_var @ psi_x_eval_j2.T)

    # cross-covariance
    cov_j1_j2 = np.diag(psi_x_eval_j1 @ (D_j1_inv_rho.T @ D_j2_inv_rho) @ psi_x_eval_j2.T)

    asy_var_diff = var_j1 + var_j2 - 2 * cov_j1_j2
    asy_se = np.sqrt(np.maximum(asy_var_diff, 0))

    # sup t-statistic
    diff = result_j1.h - result_j2.h
    z_sup = np.max(np.abs(diff) / avoid_zero_division(asy_se))

    return z_sup, asy_se, (tmp_j1, tmp_j2, u_j1, u_j2)


def _bootstrap_comparison(psi_x_eval_j1, psi_x_eval_j2, tmp_j1, tmp_j2, u_j1, u_j2, asy_se, boot_draws_all, boot_num):
    """Perform bootstrap test for dimension pair comparison."""
    z_boot = np.zeros(boot_num)

    for b in range(boot_num):
        boot_draws = boot_draws_all[b]
        boot_diff_j1 = psi_x_eval_j1 @ (tmp_j1 @ (u_j1 * boot_draws))
        boot_diff_j2 = psi_x_eval_j2 @ (tmp_j2 @ (u_j2 * boot_draws))
        boot_diff = boot_diff_j1 - boot_diff_j2

        z_boot[b] = np.max(np.abs(boot_diff) / avoid_zero_division(asy_se))

    return z_boot


def _select_optimal_dimension(z_sup, j1_j2_pairs, j_x_segments_set, theta_star):
    """Apply Lepski selection criterion to choose optimal dimension."""
    num_j = len(j_x_segments_set)
    test_mat = np.full((num_j, num_j), np.nan)

    for pair_idx, (i, j, _, _) in enumerate(j1_j2_pairs):
        test_mat[i, j] = z_sup[pair_idx] <= 1.1 * theta_star

    # check which j values pass the test
    test_vec = np.zeros(num_j - 1)
    for i in range(num_j - 1):
        test_vec[i] = np.all(test_mat[i, (i + 1) : num_j] == 1)

    if np.any(test_vec == 1):
        j_seg = j_x_segments_set[np.where(test_vec == 1)[0][0]]
    elif np.all(test_vec == 0) or np.all(np.isnan(test_vec)):
        j_seg = j_x_segments_set[-1]
    else:
        j_seg = j_x_segments_set[0]

    return j_seg, test_mat
