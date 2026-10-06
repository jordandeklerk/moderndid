"""Lepski method for optimal dimension selection in nonparametric instrumental variables."""

import numpy as np

from ..cupy.backend import get_backend, to_device, to_numpy
from .estimators import _ginv, npiv_est
from .prodspline import prodspline
from .utils import _as_eval_points, _as_matrix, _quantile_basis, avoid_zero_division, basis_dimension, matrix_sqrt


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
    biters=99,
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
    is below :math:`1.1\, \theta_{1-\hat{\alpha}}^*` for all :math:`J_2 > J`, where
    :math:`\theta_{1-\hat{\alpha}}^*` is a bootstrap critical value.

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
        Numbers of segments for the :math:`X` basis to compare. If None, uses
        [2, 4, 8, 16, 32, 64].
    k_w_segments_set : ndarray, optional
        Numbers of segments for the :math:`W` basis, one for each entry of
        ``j_x_segments_set``. If None, uses [2, 4, 8, 16, 32, 64]. Each
        instrument basis needs at least as many columns as its :math:`X` basis.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis.
    x_min, x_max, w_min, w_max : float, optional
        Range limits for basis construction.
    grid_num : int, default=50
        Number of grid points for evaluation.
    biters : int, default=99
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
        - **test_matrix**: Matrix whose entry (i, j) with j > i is 1 when the sup-t-statistic for grid
          dimensions i and j is at most :math:`1.1\, \theta_{1-\hat{\alpha}}^*` and 0 otherwise
        - **z_sup**: Sup-t-statistic for each pair of grid dimensions, in the row order of the entries
          of ``test_matrix`` above the diagonal

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
    xp = get_backend()
    y = xp.asarray(y).ravel()
    x = _as_matrix(x)
    w = _as_matrix(w)

    n = len(y)
    p_x = x.shape[1]
    p_w = w.shape[1]

    if j_x_segments_set is None:
        j_x_segments_set = np.array([2, 4, 8, 16, 32, 64])
    if k_w_segments_set is None:
        k_w_segments_set = np.array([2, 4, 8, 16, 32, 64])

    # Since npiv_est sets the instrument basis to the X basis when w equals x, only the IV case can be unidentified.
    if not np.array_equal(to_numpy(x), to_numpy(w)):
        for j_seg, k_seg in zip(j_x_segments_set, k_w_segments_set, strict=False):
            j_dim = basis_dimension(basis=basis, degree=np.full(p_x, j_x_degree), segments=np.full(p_x, j_seg))
            k_dim = basis_dimension(basis=basis, degree=np.full(p_w, k_w_degree), segments=np.full(p_w, k_seg))
            if k_dim < j_dim:
                raise ValueError(
                    f"With {k_seg} instrument segments the instrument basis has {k_dim} functions, fewer than the "
                    f"{j_dim} functions of the X basis with {j_seg} segments, so that dimension is not identified."
                )

    if x_grid is None:
        x_np = to_numpy(x)
        x_grid = np.zeros((grid_num, p_x))
        for j in range(p_x):
            x_grid[:, j] = np.linspace(x_np[:, j].min(), x_np[:, j].max(), grid_num)
        x_grid = to_device(x_grid)
    else:
        x_grid = _as_eval_points(x_grid, p_x)

    j1_j2_pairs = []
    for i, j1 in enumerate(j_x_segments_set):
        for j, j2 in enumerate(j_x_segments_set[i + 1 :], i + 1):
            j1_j2_pairs.append((i, j, j1, j2))

    n_pairs = len(j1_j2_pairs)
    z_sup = np.zeros(n_pairs)
    z_sup_boot = np.zeros((biters, n_pairs))

    rng = np.random.default_rng(seed)
    boot_draws_all_np = rng.normal(0, 1, (biters, n))
    boot_draws_all = to_device(boot_draws_all_np)

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
                psi_x_j1_eval, psi_x_j2_eval, tmp_j1, tmp_j2, u_j1, u_j2, asy_se, boot_draws_all, biters
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
    j_seg_n = j_x_segments_set[-2] if len(j_x_segments_set) > 1 else j_seg

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

    Following [1]_, :math:`\hat{J}_{\max}` is the first :math:`J` in a dyadic grid :math:`\mathcal{T}`
    that satisfies

    .. math::

        J \sqrt{\log J} \hat{s}_{J}^{-1} \leq c \sqrt{n}

    while the next grid point does not, for a constant :math:`c` (here, 10). The term :math:`\hat{s}_J`, the
    smallest singular value of a matrix related to the instrumented basis functions, measures the degree of
    ill-posedness. Since a zero :math:`\hat{s}_J` means the instruments cannot identify that dimension, the
    grid stops below it.

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
        Number of dyadic refinements of the :math:`W` basis relative to the
        :math:`X` basis at each grid point.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis.
    x_min, x_max, w_min, w_max : float, optional
        Range limits for basis construction.

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
    x = _as_matrix(x)
    w = _as_matrix(w)
    n = x.shape[0]
    p_x = x.shape[1]
    p_w = w.shape[1]

    is_regression = np.array_equal(to_numpy(x), to_numpy(w))

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

            if s_hat_j > 0:
                test_val[i] = j_x_dim * np.sqrt(np.log(j_x_dim)) * max((0.1 * np.log(n)) ** 4, 1 / s_hat_j)
            else:
                # An infinite inverse sieve measure fails the bound and ends the grid here.
                test_val[i] = np.inf
        elif i > 1:
            test_val[i] = test_val[i - 1]

    valid_idx = np.where((test_val[:-1] <= 10 * np.sqrt(n)) & (10 * np.sqrt(n) < test_val[1:]))[0]

    L_hat_max = valid_idx[0] + 1 if len(valid_idx) > 0 else L_max

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
    """Compute the sieve measure of ill-posedness.

    It returns 0 when the measure cannot be computed.
    """
    xp = get_backend()
    try:
        K_x = np.column_stack([np.full(p_x, j_x_degree), np.full(p_x, j_x_segments - 1)])
        K_w = np.column_stack([np.full(p_w, k_w_degree), np.full(p_w, k_w_segments - 1)])

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

        if basis in ("additive", "glp"):
            psi_x = xp.concatenate([xp.ones((n, 1)), psi_x], axis=1)
            b_w = xp.concatenate([xp.ones((n, 1)), b_w], axis=1)

        psi_x_gram_sqrt = matrix_sqrt(_ginv(psi_x.T @ psi_x))
        b_w_gram_sqrt = matrix_sqrt(_ginv(b_w.T @ b_w))

        svd_matrix = psi_x_gram_sqrt @ (psi_x.T @ b_w) @ b_w_gram_sqrt
        s_hat_j = float(xp.min(xp.linalg.svd(svd_matrix, compute_uv=False)))

    except (ValueError, np.linalg.LinAlgError):
        s_hat_j = 0.0

    return s_hat_j if np.isfinite(s_hat_j) else 0.0


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
        xeval=x_grid,
        K=K_x,
        knots=knots,
        basis=basis,
        x_min=np.full(p_x, x_min) if x_min is not None else None,
        x_max=np.full(p_x, x_max) if x_max is not None else None,
    ).basis

    if basis in ("additive", "glp"):
        xp = get_backend()
        psi_x = xp.concatenate([xp.ones((psi_x.shape[0], 1)), psi_x], axis=1)
        psi_x_eval = xp.concatenate([xp.ones((psi_x_eval.shape[0], 1)), psi_x_eval], axis=1)
        b_w = xp.concatenate([xp.ones((b_w.shape[0], 1)), b_w], axis=1)

    # influence matrices
    btb_inv = _ginv(b_w.T @ b_w)
    design_matrix = psi_x.T @ b_w @ btb_inv @ b_w.T
    gram_inv = _ginv(design_matrix @ psi_x)
    tmp_matrix = gram_inv @ design_matrix

    return psi_x_eval, tmp_matrix


def _compute_test_statistic(result_j1, result_j2, psi_x_eval_j1, psi_x_eval_j2, tmp_j1, tmp_j2, u_j1, u_j2):
    """Compute sup-t test statistic for comparing two estimators."""
    xp = get_backend()
    # Row sums give the diagonals of the variance and covariance matrices without forming them.
    D_j1_inv_rho = tmp_j1.T * u_j1[:, None]
    D_j1_var = D_j1_inv_rho.T @ D_j1_inv_rho
    var_j1 = xp.sum((psi_x_eval_j1 @ D_j1_var) * psi_x_eval_j1, axis=1)

    D_j2_inv_rho = tmp_j2.T * u_j2[:, None]
    D_j2_var = D_j2_inv_rho.T @ D_j2_inv_rho
    var_j2 = xp.sum((psi_x_eval_j2 @ D_j2_var) * psi_x_eval_j2, axis=1)

    cov_j1_j2 = xp.sum((psi_x_eval_j1 @ (D_j1_inv_rho.T @ D_j2_inv_rho)) * psi_x_eval_j2, axis=1)

    asy_var_diff = var_j1 + var_j2 - 2 * cov_j1_j2
    asy_se = xp.sqrt(xp.maximum(asy_var_diff, 0))

    # sup t-statistic
    diff = result_j1.h - result_j2.h
    z_sup = float(xp.max(xp.abs(diff) / avoid_zero_division(asy_se)))

    return z_sup, asy_se, (tmp_j1, tmp_j2, u_j1, u_j2)


def _bootstrap_comparison(psi_x_eval_j1, psi_x_eval_j2, tmp_j1, tmp_j2, u_j1, u_j2, asy_se, boot_draws_all, biters):
    """Perform bootstrap test for dimension pair comparison."""
    xp = get_backend()
    z_boot = np.zeros(biters)

    for b in range(biters):
        boot_draws = boot_draws_all[b]
        boot_diff_j1 = psi_x_eval_j1 @ (tmp_j1 @ (u_j1 * boot_draws))
        boot_diff_j2 = psi_x_eval_j2 @ (tmp_j2 @ (u_j2 * boot_draws))
        boot_diff = boot_diff_j1 - boot_diff_j2

        z_boot[b] = float(xp.max(xp.abs(boot_diff) / avoid_zero_division(asy_se)))

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
