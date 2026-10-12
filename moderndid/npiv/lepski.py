"""Lepski method for optimal dimension selection in nonparametric instrumental variables."""

import warnings

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

    Implements Steps 2 and 3 of the data-driven choice of sieve dimension in [1]_. The
    function fits the sieve estimator at every dimension of the grid and compares each
    pair of fits by a sup-t statistic. It keeps the smallest dimension whose fit no larger
    dimension rejects. A multiplier bootstrap of the same statistic sets the rejection
    threshold.

    The returned ``j_tilde`` is the smaller of that Lepski choice and the second-largest
    grid dimension. This truncation also applies when ``w`` equals ``x``. The regression
    version of the procedure in [1]_ keeps the Lepski choice instead. A grid with a single
    entry leaves nothing to compare. Its dimension is returned as ``j_tilde`` with
    ``theta_star`` equal to 0.

    When ``w`` equals ``x``, the instrument basis takes the degree and segments of the
    basis for :math:`X` at every dimension. The function then ignores ``k_w_degree`` and
    ``k_w_segments_set``. Unless the range limits on ``w`` differ from those on ``x``, the
    fits are least squares regressions on the basis for :math:`X`.

    Parameters
    ----------
    y : ndarray
        Dependent variable vector.
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    x_grid : ndarray, optional
        Points over which the fits are compared. If None, ``grid_num`` equally
        spaced values between the smallest and largest value of each regressor.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`, at least 1.
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
        Significance level :math:`\hat{\alpha}` of the bootstrap critical value.
    check_is_fullrank : bool, default=False
        Whether to check for full rank.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - **j_tilde**: Selected dimension, the smaller of ``j_hat`` and ``j_hat_n``
        - **j_hat**: Dimension chosen by the Lepski test before truncation
        - **j_hat_n**: Truncation point, the dimension of the second-largest grid entry
        - **j_x_seg**: Number of segments of the :math:`X` basis at ``j_tilde``
        - **k_w_seg**: Number of segments of the :math:`W` basis paired with ``j_x_seg``
        - **theta_star**: Bootstrap critical value :math:`\theta_{1-\hat{\alpha}}^*` of the Lepski test
        - **test_matrix**: Matrix whose entry (i, j) with j > i is 1 when the sup-t-statistic for grid
          dimensions i and j is at most :math:`1.1\, \theta_{1-\hat{\alpha}}^*` and 0 otherwise
        - **z_sup**: Sup-t-statistic for each pair of grid dimensions, in the row order of the entries
          of ``test_matrix`` above the diagonal

    See Also
    --------
    npiv_choose_j : Full data-driven selection procedure
    npiv_jhat_max : Compute maximum feasible dimension

    Notes
    -----
    For grid dimensions :math:`J < J_2`, the sup-t statistic compares the two fits over the
    points of ``x_grid``,

    .. math::

        T_{J, J_2} = \sup_{x} \left| \frac{\hat{h}_J(x) - \hat{h}_{J_2}(x)}{\hat{\sigma}_{J, J_2}(x)} \right|,

    where :math:`\hat{\sigma}_{J, J_2}^2(x)` estimates the variance of the difference. The critical
    value :math:`\theta_{1-\hat{\alpha}}^*` is the :math:`(1-\hat{\alpha})` quantile across bootstrap
    draws of

    .. math::

        \sup_{(x, J, J_2) : J_2 > J} \left| \frac{D_J^*(x) - D_{J_2}^*(x)}{\hat{\sigma}_{J, J_2}(x)} \right|.

    Here :math:`D_J^*` is the multiplier bootstrap process of :func:`compute_ucb`. Every pair uses the
    same draws. With :math:`\hat{J}_n` the second-largest grid dimension, the selection is

    .. math::

        \hat{J} = \min \left\{ J : T_{J, J_2} \leq 1.1\, \theta_{1-\hat{\alpha}}^*
        \text{ for all } J_2 > J \right\},
        \qquad
        \tilde{J} = \min\{\hat{J}, \hat{J}_n\}.

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive estimation and
        uniform confidence bands for nonparametric structural functions and elasticities.
        *Review of Economic Studies*, 92(1), 162-196. https://doi.org/10.1093/restud/rdae025.
    """
    _check_selection_degree(j_x_degree)

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

    if np.array_equal(to_numpy(x), to_numpy(w)):
        # Since npiv_est gives the instrument basis the X degree and segments here, the influence matrices must match.
        k_w_degree = j_x_degree
        k_w_segments_set = j_x_segments_set
    else:
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

        psi_x_j1_eval, tmp_j1 = _compute_basis_and_influence(
            x, w, x_grid, j1, k1, j_x_degree, k_w_degree, p_x, p_w, knots, basis, x_min, x_max, w_min, w_max
        )

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

    # With no pair to compare in a one-entry grid, the critical value is 0 and the band adds no selection margin.
    z_boot_max = np.max(z_sup_boot, axis=1) if n_pairs > 0 else np.zeros(0)
    theta_star = _quantile_basis(z_boot_max, 1 - alpha)

    j_seg, test_mat = _select_optimal_dimension(z_sup, j1_j2_pairs, j_x_segments_set, theta_star)

    j_hat = basis_dimension(
        basis=basis,
        degree=np.full(p_x, j_x_degree),
        segments=np.full(p_x, j_seg),
    )

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

    Implements Step 1 of the data-driven choice of sieve dimension in [1]_. The grid holds
    the dyadic dimensions from one segment up to :math:`\hat{J}_{\max}`, the largest
    dimension that the sample size and the estimated strength of the instruments support.
    At each grid point the instrument basis has ``2**k_w_smooth`` times as many segments
    as the basis for :math:`X`.

    The grid has no lower cutoff. [1]_ starts it at :math:`0.1 (\log \hat{J}_{\max})^2`.
    With the default cubic tensor basis, that cutoff stays below the smallest dimension
    until the sample size reaches about one million. With linear splines, or with an
    additive or glp basis on one regressor, it removes the smallest dimensions at moderate
    sample sizes.

    When the sample size and the instruments support no dimension beyond the smallest, the
    grid holds that dimension alone. A warning then says that the Lepski test in
    :func:`npiv_j` has nothing to select.

    When ``w`` equals ``x``, the rule uses the constant of the regression version of the
    procedure in place of the estimated instrument strength. The function then ignores
    ``k_w_degree`` and ``k_w_smooth`` and pairs each grid point with as many instrument
    segments as the basis for :math:`X` has.

    Parameters
    ----------
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`, at least 1.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`.
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

        - **j_x_segments_set**: Numbers of segments of the :math:`X` basis on the grid
        - **k_w_segments_set**: Numbers of segments of the :math:`W` basis paired with ``j_x_segments_set``
        - **j_hat_max**: Largest grid dimension :math:`\hat{J}_{\max}`
        - **alpha_hat**: Significance level :math:`\hat{\alpha}` for the Lepski test in :func:`npiv_j`

    See Also
    --------
    npiv_choose_j : Full data-driven selection procedure
    npiv_j : Lepski-style selection procedure

    Notes
    -----
    With :math:`\mathcal{T}` the dyadic grid of dimensions and :math:`J^+` the grid dimension
    after :math:`J`, the upper limit is

    .. math::

        \hat{J}_{\max} = \min \left\{ J \in \mathcal{T} : J \sqrt{\log J}\, \hat{v}_J \leq 10 \sqrt{n}
        < J^{+} \sqrt{\log J^{+}}\, \hat{v}_{J^{+}} \right\},
        \qquad
        \hat{v}_J = \max\left\{(0.1 \log n)^4, \hat{s}_J^{-1}\right\},

    where :math:`\hat{s}_J` is the smallest singular value of
    :math:`(\boldsymbol{\Psi}_J' \boldsymbol{\Psi}_J)^{-1/2} \boldsymbol{\Psi}_J' \mathbf{B}_K
    (\mathbf{B}_K' \mathbf{B}_K)^{-1/2}`. A small :math:`\hat{s}_J` signals weak instruments. A zero
    :math:`\hat{s}_J` fails the bound. [1]_ uses :math:`\hat{s}_J^{-1}` alone. Since
    :math:`\hat{s}_J \leq 1`, the floor :math:`(0.1 \log n)^4` can bind only when :math:`n` exceeds
    :math:`e^{10}`, about 22,026. When ``w`` equals ``x``, :math:`\hat{v}_J` is
    :math:`\max\{1, (0.1 \log n)^4\}`, the constant of the regression version in [1]_.

    The grid runs through :math:`2^0, 2^1, \ldots` segments. When no grid point satisfies both
    inequalities, it stops at :math:`2^{L-1}` segments with
    :math:`L = \max\{\lfloor \log n / \log(2d) \rfloor, 3\}` for :math:`d` regressors. The
    significance level of the Lepski test is

    .. math::

        \hat{\alpha} = \min\left\{0.5, \left(\log \hat{J}_{\max} / \hat{J}_{\max}\right)^{1/2}\right\}.

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive estimation and
        uniform confidence bands for nonparametric structural functions and elasticities.
        *Review of Economic Studies*, 92(1), 162-196. https://doi.org/10.1093/restud/rdae025.
    """
    _check_selection_degree(j_x_degree)

    x = _as_matrix(x)
    w = _as_matrix(w)
    n = x.shape[0]
    p_x = x.shape[1]
    p_w = w.shape[1]

    is_regression = np.array_equal(to_numpy(x), to_numpy(w))
    if is_regression:
        # The regression version uses the X basis as its own instrument basis at every dimension.
        k_w_smooth = 0

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

    if L_hat_max == 1:
        warnings.warn(
            "The sample size and the estimated strength of the instruments support only the smallest grid "
            f"dimension, J = {j_hat_max}. With a single candidate there is nothing to select. A data-driven band at "
            f"J = {j_hat_max} then adds no margin for the choice of dimension and is valid only if the approximation "
            "bias there is negligible.",
            UserWarning,
        )

    alpha_hat = min(0.5, np.sqrt(np.log(j_hat_max) / j_hat_max))

    return {
        "j_x_segments_set": j_x_segments_set,
        "k_w_segments_set": k_w_segments_set,
        "j_hat_max": j_hat_max,
        "alpha_hat": alpha_hat,
    }


def _check_selection_degree(j_x_degree):
    """Raise when a degree-0 basis for X leaves nothing to select."""
    if j_x_degree < 1:
        raise ValueError(
            f"The data-driven selection needs j_x_degree of at least 1, got {j_x_degree}. Since a degree-0 basis is "
            "constant for every number of segments, there is nothing to select. Pass j_x_degree >= 1, or pass "
            "j_x_segments to npiv for a fixed basis."
        )


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
