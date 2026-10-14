"""Utility functions for sensitivity analysis."""

import re
import warnings

import numpy as np
import scipy.optimize as opt


def basis_vector(index=1, size=1):
    """Create a standard basis vector.

    Parameters
    ----------
    index : int, default=1
        Position for the 1 value.
    size : int, default=1
        Length of the vector.

    Returns
    -------
    ndarray
        Column vector with shape (size, 1).
    """
    if index < 1 or index > size:
        raise ValueError(f"index must be between 1 and {size}, got {index}")

    v = np.zeros((size, 1))
    v[index - 1] = 1
    return v


def validate_symmetric_psd(sigma):
    """Check if a matrix is symmetric and positive semi-definite.

    Issues a UserWarning if the matrix is not exactly symmetric or not numerically
    positive semi-definite. It never raises.

    Parameters
    ----------
    sigma : ndarray
        Matrix to validate.
    """
    sigma = np.asarray(sigma)

    asymmetry = np.max(np.abs(sigma - sigma.T))
    if asymmetry > 1e-10:
        warnings.warn(
            f"Matrix sigma not exactly symmetric (largest asymmetry was {asymmetry:.6g})",
            UserWarning,
        )

    eigenvalues = np.linalg.eigvals(sigma)
    min_eigenvalue = np.min(eigenvalues.real)

    if min_eigenvalue < -1e-10:
        warnings.warn(
            f"Matrix sigma not numerically positive semi-definite (smallest eigenvalue was {min_eigenvalue:.6g})",
            UserWarning,
        )


def validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec):
    """Validate dimensions of inputs for sensitivity analysis.

    Parameters
    ----------
    betahat : ndarray
        Estimated coefficients vector.
    sigma : ndarray
        Covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : array-like
        Weight vector for post-treatment periods.
    """
    betahat = np.asarray(betahat)
    sigma = np.asarray(sigma)
    l_vec = np.asarray(l_vec)

    if betahat.ndim > 2:
        raise ValueError(f"Expected a vector but betahat has shape {betahat.shape}")

    betahat_flat = betahat.flatten()
    beta_length = len(betahat_flat)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Expected a square matrix but sigma was {sigma.shape[0]} by {sigma.shape[1]}")

    if sigma.shape[0] != beta_length:
        raise ValueError(f"betahat ({betahat.shape}) and sigma ({sigma.shape}) were non-conformable")

    num_periods = num_pre_periods + num_post_periods
    if num_periods != beta_length:
        raise ValueError(
            f"betahat ({betahat.shape}) and pre + post periods "
            f"({num_pre_periods} + {num_post_periods}) were non-conformable"
        )

    if len(l_vec) != num_post_periods:
        raise ValueError(f"l_vec (length {len(l_vec)}) and post periods ({num_post_periods}) were non-conformable")


def relative_magnitudes_grid_bounds(betahat, sigma, num_pre_periods, num_post_periods, l_vec, m_bar):
    r"""Compute default grid bounds for a confidence set under relative magnitudes.

    The grid centers on :math:`l'\hat{\beta}_{post}`. Under :math:`\Delta^{RM}(\bar{M})` each first
    difference of :math:`\delta` after the reference period is at most :math:`\bar{M}` times the largest
    pre-treatment one. The identified set at :math:`\hat{\beta}` therefore lies within
    :math:`\bar{M} \sum_t t |l_t| \max_s |\hat{\beta}_{s+1} - \hat{\beta}_s|` of the center. The differences
    run over the pre-treatment coefficients and the reference period.

    A margin for sampling error extends both ends. It is the larger of 20 standard deviations of
    :math:`l'\hat{\beta}_{post}` and six standard deviations of the noisiest pre-treatment first difference
    times :math:`\bar{M} \sum_t t |l_t|`. The second term takes over at long horizons. The
    uncertainty about the largest pre-treatment change is multiplied the most there.

    Parameters
    ----------
    betahat : ndarray
        Estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of betahat.
    num_pre_periods : int
        Number of pre-treatment periods. Must be at least 1.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Weights that define the parameter of interest :math:`\theta = l'\tau_{post}`.
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`.

    Returns
    -------
    tuple
        - **grid_lb**: Lower end of the grid
        - **grid_ub**: Upper end of the grid
    """
    betahat = np.asarray(betahat, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).flatten()
    post = slice(num_pre_periods, num_pre_periods + num_post_periods)

    # Each row takes one first difference over the pre-treatment coefficients and the reference period's zero.
    differences = np.diff(np.eye(num_pre_periods + 1)[:, :num_pre_periods], axis=0)
    pre_sigma = sigma[:num_pre_periods, :num_pre_periods]
    largest = np.max(np.abs(differences @ betahat[:num_pre_periods]))
    largest_sd = np.max(np.sqrt(np.diag(differences @ pre_sigma @ differences.T)))

    sd_theta = np.sqrt(l_vec @ sigma[post, post] @ l_vec)
    reach = m_bar * (np.arange(1, num_post_periods + 1) @ np.abs(l_vec))
    half_width = reach * largest + max(20 * sd_theta, 6 * reach * largest_sd)
    center = l_vec @ betahat[post]

    return center - half_width, center + half_width


def linear_trend_grid_bounds(betahat, sigma, num_pre_periods, num_post_periods, l_vec, bound, relative=False):
    r"""Compute default grid bounds for a confidence set under a bound on second differences.

    The grid centers on the extension of the last pre-treatment slope,
    :math:`l'\hat{\beta}_{post} + \hat{\beta}_{-1} \sum_t t \, l_t`. If every second difference of
    :math:`\delta` from the reference period on is at most :math:`B` in absolute value, :math:`\delta_t`
    departs from that line by at most :math:`B \, t(t+1)/2`. The identified set at :math:`\hat{\beta}`
    therefore lies within :math:`B \sum_t |l_t| \, t(t+1)/2` of the center. Under :math:`\Delta^{SD}(M)`
    the bound :math:`B` is :math:`M`. Under :math:`\Delta^{SDRM}(\bar{M})` it is :math:`\bar{M}` times the
    largest pre-treatment second difference in absolute value.

    A margin for sampling error extends both ends. It is 20 standard deviations of the center or of
    :math:`l'\hat{\beta}_{post}`, whichever is larger. Under :math:`\Delta^{SDRM}(\bar{M})` the margin is
    at least six standard deviations of the noisiest pre-treatment second difference times
    :math:`\bar{M} \sum_t |l_t| \, t(t+1)/2`. That term takes over at long horizons. The uncertainty
    about the largest pre-treatment second difference is multiplied the most there.

    Parameters
    ----------
    betahat : ndarray
        Estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of betahat.
    num_pre_periods : int
        Number of pre-treatment periods. Must be at least 2 when ``relative`` is True.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Weights that define the parameter of interest :math:`\theta = l'\tau_{post}`.
    bound : float
        Smoothness bound :math:`M`, or the relative magnitude parameter :math:`\bar{M}` when
        ``relative`` is True.
    relative : bool, default=False
        Whether ``bound`` scales the largest pre-treatment second difference, as in
        :math:`\Delta^{SDRM}(\bar{M})`.

    Returns
    -------
    tuple
        - **grid_lb**: Lower end of the grid
        - **grid_ub**: Upper end of the grid
    """
    betahat = np.asarray(betahat, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).flatten()
    post = slice(num_pre_periods, num_pre_periods + num_post_periods)
    periods = np.arange(1, num_post_periods + 1)

    center_weights = np.zeros(num_pre_periods + num_post_periods)
    center_weights[num_pre_periods - 1] = periods @ l_vec
    center_weights[post] = l_vec
    center = center_weights @ betahat
    sd_center = np.sqrt(center_weights @ sigma @ center_weights)
    sd_theta = np.sqrt(l_vec @ sigma[post, post] @ l_vec)

    reach = np.abs(l_vec) @ (periods * (periods + 1) / 2)
    margin = 20 * max(sd_center, sd_theta)

    if relative:
        # Each row takes one second difference over the pre-treatment coefficients and the reference period's zero.
        second_differences = np.diff(np.eye(num_pre_periods + 1)[:, :num_pre_periods], n=2, axis=0)
        pre_sigma = sigma[:num_pre_periods, :num_pre_periods]
        largest = np.max(np.abs(second_differences @ betahat[:num_pre_periods]))
        largest_sd = np.max(np.sqrt(np.diag(second_differences @ pre_sigma @ second_differences.T)))
        half_width = bound * reach * largest + max(margin, 6 * bound * reach * largest_sd)
    else:
        half_width = bound * reach + margin

    return center - half_width, center + half_width


def identified_set_bounds(true_beta, l_vec, a_matrix, d_vec, num_pre_periods, num_post_periods):
    r"""Compute the identified set for one polyhedral restriction on the differential trends.

    The bounds are :math:`l'\beta_{post}` minus the largest and the smallest value of
    :math:`l'\delta_{post}` over the :math:`\delta` with :math:`A\delta \le d` and
    :math:`\delta_{pre} = \beta_{pre}`. When no :math:`\delta` meets both conditions, the restriction
    contradicts the pre-treatment coefficients. The identified set is then empty and both bounds are NaN.

    Parameters
    ----------
    true_beta : ndarray
        Coefficient vector :math:`\beta = (\beta_{pre}', \beta_{post}')'`.
    l_vec : ndarray
        Weights that define the parameter of interest :math:`\theta = l'\tau_{post}`.
    a_matrix : ndarray
        Constraint matrix :math:`A` over the pre- and post-treatment periods.
    d_vec : ndarray
        Constraint bounds :math:`d`.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    tuple
        - **id_lb**: Lower bound of the identified set, or NaN when the set is empty
        - **id_ub**: Upper bound of the identified set, or NaN when the set is empty
    """
    true_beta = np.asarray(true_beta, dtype=float).flatten()
    l_vec = np.asarray(l_vec, dtype=float).flatten()
    objective = np.concatenate([np.zeros(num_pre_periods), l_vec])
    a_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]
    d_vec = np.asarray(d_vec, dtype=float).flatten()

    # Since the solver's feasibility tolerances are absolute, a program whose coefficients are all small is solved at
    # unit scale. Shrinking a larger program would instead push a small d toward the tolerances.
    scale = min(1.0, max(np.max(np.abs(b_eq), initial=0.0), np.max(np.abs(d_vec), initial=0.0)))
    scale = scale if scale > 0 else 1.0
    largest = _solve_trend_program(-objective, a_matrix, d_vec / scale, a_eq, b_eq / scale)
    smallest = _solve_trend_program(objective, a_matrix, d_vec / scale, a_eq, b_eq / scale)

    if largest.status == 2 or smallest.status == 2:
        return np.nan, np.nan

    l_beta_post = l_vec @ true_beta[num_pre_periods:]
    id_lb = -np.inf if largest.status == 3 else l_beta_post + scale * largest.fun
    id_ub = np.inf if smallest.status == 3 else l_beta_post - scale * smallest.fun

    return id_lb, id_ub


def union_of_identified_sets(bounds):
    """Combine the identified sets of the polyhedra that make up a restriction.

    The result runs from the smallest lower bound to the largest upper bound. Empty sets, whose
    bounds are NaN, drop out. When every set is empty, so is the union.

    Parameters
    ----------
    bounds : list of tuple
        Lower and upper bound of each polyhedron's identified set.

    Returns
    -------
    tuple
        - **id_lb**: Lower bound of the union, or NaN when every set is empty
        - **id_ub**: Upper bound of the union, or NaN when every set is empty
    """
    nonempty = [(lb, ub) for lb, ub in bounds if not np.isnan(lb)]
    if not nonempty:
        return np.nan, np.nan

    return min(lb for lb, _ in nonempty), max(ub for _, ub in nonempty)


def _solve_trend_program(objective, a_ub, b_ub, a_eq, b_eq):
    """Solve one linear program over the differential trends with every variable free.

    Parameters
    ----------
    objective : ndarray
        Coefficients of the objective to minimize.
    a_ub : ndarray
        Inequality constraint matrix.
    b_ub : ndarray
        Inequality constraint bounds.
    a_eq : ndarray
        Equality constraint matrix.
    b_eq : ndarray
        Equality constraint values.

    Returns
    -------
    OptimizeResult
        Solver result with status 0 (solved), 2 (infeasible), or 3 (unbounded).
    """
    bounds = [(None, None)] * len(objective)
    result = opt.linprog(c=objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.status == 4:
        # Presolve can stop at "unbounded or infeasible". Solving without it tells the two apart.
        result = opt.linprog(
            c=objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": False},
        )

    if result.status not in (0, 2, 3):
        raise RuntimeError(
            f"The linear program for the identified set stopped without a solution ({result.message}). "
            "Check that true_beta holds finite values on a reasonable scale."
        )

    return result


def lee_coefficient(eta, sigma):
    r"""Compute the coefficient that separates a linear combination from the rest of the data.

    For data :math:`y` with covariance :math:`\Sigma`, the coefficient of the linear combination
    :math:`\eta' y` is

    .. math::

        c = \frac{\Sigma \eta}{\eta' \Sigma \eta}.

    The remainder :math:`(I - c \eta') y` is uncorrelated with :math:`\eta' y`. The function raises an
    error when :math:`\eta' \Sigma \eta` is effectively zero.

    Parameters
    ----------
    eta : array-like
        Weights of the linear combination.
    sigma : ndarray
        Covariance matrix of the data. Must have as many rows as ``eta`` has entries.

    Returns
    -------
    ndarray
        Coefficient vector with the same length as ``eta``.
    """
    eta = np.asarray(eta, dtype=np.float64).flatten()
    # Since a product with a strided or transposed matrix can round differently, the covariance is made contiguous.
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)
    sigma_eta = sigma @ eta
    eta_sigma_eta = eta.T @ sigma_eta
    if np.abs(eta_sigma_eta) < 1e-10:
        raise ValueError("Estimated coefficient is effectively zero, cannot compute coefficient.")
    return sigma_eta / eta_sigma_eta


def selection_matrix(selection, size, select="columns"):
    """Create a selection matrix for extracting specific rows or columns.

    With ``select="rows"`` the product of the result and a matrix holds the selected rows of that matrix.
    With ``select="columns"`` the product of a matrix and the result holds the selected columns.

    Parameters
    ----------
    selection : array-like
        One-based positions to select.
    size : int
        Number of rows or columns of the matrix to select from, according to ``select``.
    select : {"rows", "columns"}, default="columns"
        Whether the result selects rows or columns.

    Returns
    -------
    ndarray
        Matrix of zeros and ones with shape ``(len(selection), size)`` for rows or ``(size, len(selection))``
        for columns.
    """
    selection = np.asarray(selection)
    selection_0idx = selection - 1
    n_selections = len(selection)
    if select == "rows":
        m = np.zeros((n_selections, size))
        for i, idx in enumerate(selection_0idx):
            m[i, idx] = 1
    else:
        m = np.zeros((size, n_selections))
        for i, idx in enumerate(selection_0idx):
            m[idx, i] = 1
    return m


def compute_bounds(eta, sigma, A, b, z):
    r"""Compute the truncation bounds of a linear combination under linear inequalities.

    Write the data as :math:`y = c \, \eta' y + z` with :math:`c` from :func:`~moderndid.lee_coefficient`. The
    inequalities :math:`A y \le b` hold exactly when :math:`\eta' y` lies between the lower and the upper bound
    that this function returns. Since an inequality whose entry of :math:`A c` is at most :math:`10^{-10}` in
    absolute value places no usable bound on :math:`\eta' y`, the function skips it.

    Parameters
    ----------
    eta : array-like
        Weights of the linear combination.
    sigma : ndarray
        Covariance matrix of the data.
    A : ndarray
        Coefficient matrix of the inequalities.
    b : array-like
        Right-hand side of the inequalities.
    z : array-like
        Remainder :math:`(I - c \eta') y` of the data.

    Returns
    -------
    tuple
        Truncation bounds of the linear combination:

        - **lower_bound**: Largest lower bound, or ``-inf`` when no inequality bounds it from below
        - **upper_bound**: Smallest upper bound, or ``inf`` when no inequality bounds it from above
    """
    c = lee_coefficient(eta, sigma)
    # Since a product with a strided or transposed matrix can round differently, the matrix is made contiguous.
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    z = np.asarray(z, dtype=np.float64).flatten()

    Az = A @ z
    Ac = A @ c
    nonzero_mask = np.abs(Ac) > 1e-10
    objective = np.full_like(Ac, np.nan)
    objective[nonzero_mask] = (b[nonzero_mask] - Az[nonzero_mask]) / Ac[nonzero_mask]
    ac_negative_idx = nonzero_mask & (Ac < 0)
    ac_positive_idx = nonzero_mask & (Ac > 0)
    lower_bound = np.max(objective[ac_negative_idx]) if np.any(ac_negative_idx) else -np.inf
    upper_bound = np.min(objective[ac_positive_idx]) if np.any(ac_positive_idx) else np.inf
    return lower_bound, upper_bound


def find_rows_with_post_period_values(A, post_period_indices):
    """Find the rows of a matrix with nonzero entries in the post-period columns.

    Parameters
    ----------
    A : ndarray
        Constraint matrix.
    post_period_indices : list[int], ndarray
        Zero-based positions of the post-treatment columns.

    Returns
    -------
    ndarray or None
        Indices of the rows with a nonzero entry in any of the columns. None if no row has one.
    """
    has_post_period_values = np.any(A[:, post_period_indices] != 0, axis=1)
    rows_for_arp = np.where(has_post_period_values)[0]
    return rows_for_arp if len(rows_for_arp) > 0 else None


def create_first_differences_matrix(num_pre_periods, num_post_periods):
    """Create the matrix of first differences of the event study coefficients.

    Row ``r`` holds -1 in column ``r`` and 1 in column ``r + 1``. The columns cover the pre-treatment periods,
    the reference period, and the post-treatment periods in order.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    ndarray
        Matrix with ``num_pre_periods + num_post_periods`` rows and one more column.
    """
    total_periods = num_pre_periods + num_post_periods + 1
    a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))
    for r in range(num_pre_periods + num_post_periods):
        a_tilde[r, r : (r + 2)] = [-1, 1]
    return a_tilde


def create_second_differences_matrix(num_constraints, total_periods):
    """Create the matrix of second differences of the event study coefficients.

    Row ``i`` holds 1, -2, and 1 in columns ``i`` to ``i + 2``. A row that would run past the last column
    stays zero.

    Parameters
    ----------
    num_constraints : int
        Number of rows.
    total_periods : int
        Number of columns.

    Returns
    -------
    ndarray
        Matrix of shape ``(num_constraints, total_periods)``.
    """
    A_positive = np.zeros((num_constraints, total_periods))
    for i in range(num_constraints):
        if i + 3 <= total_periods:
            A_positive[i, i : i + 3] = [1, -2, 1]
    return A_positive


def create_bounds_second_difference_matrix(num_pre_periods, num_post_periods):
    """Create the matrix of second differences within the pre-treatment and post-treatment coefficients.

    The rows for the pre-treatment coefficients come first and the rows for the post-treatment coefficients
    follow. No row spans both groups. This is the matrix that :func:`~moderndid.create_second_difference_matrix`
    returns.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    ndarray
        Matrix with ``num_pre_periods + num_post_periods`` columns. It has no rows unless one group has at least
        three periods.
    """
    total_periods = num_pre_periods + num_post_periods
    n_pre_diffs = num_pre_periods - 2 if num_pre_periods >= 3 else 0
    n_post_diffs = num_post_periods - 2 if num_post_periods >= 3 else 0
    n_diffs = n_pre_diffs + n_post_diffs

    if n_diffs == 0:
        return np.zeros((0, total_periods))

    a_sd = np.zeros((n_diffs, total_periods))

    if n_pre_diffs > 0:
        for r in range(n_pre_diffs):
            a_sd[r, r : (r + 3)] = [1, -2, 1]

    if n_post_diffs > 0:
        for r in range(n_post_diffs):
            post_idx = n_pre_diffs + r
            coef_idx = num_pre_periods + r
            a_sd[post_idx, coef_idx : (coef_idx + 3)] = [1, -2, 1]
    return a_sd


def create_monotonicity_matrix(num_pre_periods, num_post_periods):
    r"""Create the base matrix of the monotonicity constraints.

    The inequalities :math:`A \delta \le 0` hold when :math:`\delta` is nondecreasing over the pre-treatment
    periods up to the reference period at zero and nondecreasing from zero over the post-treatment periods.
    :func:`~moderndid.create_monotonicity_constraint_matrix` drops the zero rows and flips the signs for a
    decreasing sequence.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    ndarray
        Square matrix with ``num_pre_periods + num_post_periods`` rows.
    """
    total_periods = num_pre_periods + num_post_periods
    a_m = np.zeros((total_periods, total_periods))

    for r in range(num_pre_periods - 1):
        a_m[r, r : (r + 2)] = [1, -1]

    a_m[num_pre_periods - 1, num_pre_periods - 1] = 1

    if num_post_periods > 0:
        a_m[num_pre_periods, num_pre_periods] = -1
        if num_post_periods > 1:
            for r in range(num_pre_periods + 1, num_pre_periods + num_post_periods):
                a_m[r, (r - 1) : r + 1] = [1, -1]
    return a_m


def create_sdrm_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive=True, drop_zero=True):
    r"""Create the constraint matrix for second differences under relative magnitudes.

    The rows bound every second difference in absolute value by the second difference at period ``s`` before
    treatment and by ``m_bar`` times that second difference after treatment. Rows that the construction leaves
    at zero are dropped. See :func:`~moderndid.compute_identified_set_sdrm` for the restriction
    :math:`\Delta^{SDRM}_{s,\text{sign}}(\bar{M})` that the matrix defines through :math:`A \delta \le 0`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`.
    s : int
        Period of the largest pre-treatment second difference. Must run from ``-(num_pre_periods - 2)`` to 0.
    max_positive : bool, default=True
        Whether the second difference at period ``s`` is positive rather than negative.
    drop_zero : bool, default=True
        Whether to drop the column of the reference period.

    Returns
    -------
    ndarray
        Matrix with ``num_pre_periods + num_post_periods + 1`` columns, or one fewer if ``drop_zero`` is True.
    """
    total_periods = num_pre_periods + num_post_periods + 1

    num_second_diffs = num_pre_periods + num_post_periods - 1
    a_tilde = np.zeros((num_second_diffs, total_periods))

    for r in range(num_second_diffs):
        a_tilde[r, r : r + 3] = [1, -2, 1]

    v_max_diff = np.zeros((1, total_periods))
    idx = num_pre_periods + s - 2
    v_max_diff[0, idx : idx + 3] = [1, -2, 1]

    if not max_positive:
        v_max_diff = -v_max_diff

    a_ub_pre = np.tile(v_max_diff, (num_pre_periods - 1, 1))
    a_ub_post = np.tile(m_bar * v_max_diff, (num_post_periods, 1))
    a_ub = np.vstack([a_ub_pre, a_ub_post])

    a_upper = a_tilde - a_ub
    a_lower = -a_tilde - a_ub
    a_matrix = np.vstack([a_upper, a_lower])

    row_norms = np.sum(a_matrix**2, axis=1)
    non_zero_rows = row_norms > 1e-10
    a_matrix = a_matrix[non_zero_rows]

    if drop_zero:
        a_matrix = np.delete(a_matrix, num_pre_periods, axis=1)

    return a_matrix


def prepare_theta_grid_y_values(beta_hat_or_y, period_vec_or_a_inv, theta_grid):
    """Compute the data adjusted for each candidate value of the parameter of interest.

    Row ``i`` of the result is ``beta_hat_or_y - theta_grid[i] * period_vec_or_a_inv``.

    Parameters
    ----------
    beta_hat_or_y : ndarray
        Vector of estimated coefficients or of constraint moments.
    period_vec_or_a_inv : ndarray
        Vector of the same length that gives the direction in which the parameter enters the data.
    theta_grid : ndarray
        Candidate values of the parameter.

    Returns
    -------
    ndarray
        Matrix of shape ``(len(theta_grid), len(beta_hat_or_y))``.
    """
    return beta_hat_or_y - np.outer(theta_grid, period_vec_or_a_inv)


def compute_hybrid_dbar(flci_halflength, vbar, d_vec, a_gamma_inv_one, theta):
    r"""Compute the bounds of the fixed-length interval constraint in the hybrid test.

    The first stage of the hybrid test keeps a candidate value :math:`\theta` when
    :math:`\bar{v}' y_\theta \le \bar{d}_1` and :math:`-\bar{v}' y_\theta \le \bar{d}_2`. Here :math:`y_\theta`
    holds the constraint moments at :math:`\theta`. The bounds are

    .. math::

        \bar{d}_1 = h - \bar{v}' d + (1 - \bar{v}' a) \theta, \qquad
        \bar{d}_2 = h + \bar{v}' d - (1 - \bar{v}' a) \theta.

    Parameters
    ----------
    flci_halflength : float
        Half-length :math:`h` of the fixed-length confidence interval.
    vbar : ndarray
        Weights :math:`\bar{v}` of the interval's estimator on the rows of the constraint matrix.
    d_vec : ndarray
        Right-hand side :math:`d` of the restriction :math:`A \delta \le d`.
    a_gamma_inv_one : ndarray
        Coefficients :math:`a` of the parameter of interest in the transformed constraint matrix.
    theta : float
        Candidate value :math:`\theta` of the parameter of interest.

    Returns
    -------
    ndarray
        Vector of length 2 that holds :math:`\bar{d}_1` and :math:`\bar{d}_2`.
    """
    vbar_d = np.dot(vbar, d_vec)
    vbar_a = np.dot(vbar, a_gamma_inv_one)
    return np.array([flci_halflength - vbar_d + (1 - vbar_a) * theta, flci_halflength + vbar_d - (1 - vbar_a) * theta])


def bin_factor(bin_spec, values, name=""):
    """Apply binning transformation to array values.

    Groups values according to various binning specifications including
    consecutive grouping, pattern matching, and custom mappings.

    Parameters
    ----------
    bin_spec : str, list, or dict
        Binning specification:

        - ``"bin::n"``: Group every n consecutive values
        - ``"!bin::n"``: Group n values starting from first
        - ``"!!bin::n"``: Group n values starting from last
        - ``"@pattern"``: Regex pattern for matching values
        - list: Values to group together (all map to first)
        - dict: Custom mapping of new names to old values

    values : array-like
        Values to bin.
    name : str, optional
        Variable name for error messages.

    Returns
    -------
    ndarray
        Array with binned values.
    """
    values = np.asarray(values)
    unique_values = np.unique(values)

    if isinstance(bin_spec, str):
        return _handle_string_bin_spec(bin_spec, values, unique_values)

    if isinstance(bin_spec, list):
        return _handle_list_bin_spec(bin_spec, values, unique_values)

    if isinstance(bin_spec, dict):
        return _handle_dict_bin_spec(bin_spec, values, unique_values)

    return bin_factor(list(bin_spec), values, name)


def create_interactions(
    factor_var, var=None, ref=None, keep=None, ref2=None, keep2=None, bin=None, bin2=None, name=None, return_dict=False
):
    """Create dummy variables from factors with optional interactions.

    Parameters
    ----------
    factor_var : array-like
        Primary factor variable to create dummies from.
    var : array-like, optional
        Variable to interact with factor_var. Can be numeric or categorical.
    ref : scalar, list, or True, optional
        Reference levels to exclude. If True, drops first level.
    keep : list, optional
        Levels to keep (all others dropped).
    ref2 : scalar or list, optional
        Reference levels for var (if categorical).
    keep2 : list, optional
        Levels to keep from var (if categorical).
    bin : str, list, or dict, optional
        Binning specification for factor_var.
    bin2 : str, list, or dict, optional
        Binning specification for var.
    name : str, optional
        Name prefix for generated variables.
    return_dict : bool, optional
        If True, return dictionary with matrix and metadata.

    Returns
    -------
    ndarray or dict
        Matrix of dummy variables. If return_dict=True, returns dict with
        'matrix', 'names', and 'reference_info' keys.
    """
    factor_var = np.asarray(factor_var)
    n = len(factor_var)

    if bin is not None:
        factor_var = bin_factor(bin, factor_var, name or "factor")

    is_interaction = var is not None
    is_numeric_interaction = False
    is_factor_interaction = False

    if is_interaction:
        var = np.asarray(var)
        if len(var) != len(factor_var):
            raise ValueError("factor_var and var must have the same length")

        if bin2 is not None:
            var = bin_factor(bin2, var, "var")

        if np.issubdtype(var.dtype, np.number):
            is_numeric_interaction = True
        else:
            is_factor_interaction = True

    factor_unique = np.unique(factor_var)

    if ref is not None:
        if ref is True:
            ref = [factor_unique[0]]
        elif not isinstance(ref, list):
            ref = [ref]
        factor_unique = factor_unique[~np.isin(factor_unique, ref)]

    if keep is not None:
        if not isinstance(keep, list):
            keep = [keep]
        factor_unique = factor_unique[np.isin(factor_unique, keep)]

    if is_factor_interaction:
        var_unique = np.unique(var)

        if ref2 is not None:
            if not isinstance(ref2, list):
                ref2 = [ref2]
            var_unique = var_unique[~np.isin(var_unique, ref2)]

        if keep2 is not None:
            if not isinstance(keep2, list):
                keep2 = [keep2]
            var_unique = var_unique[np.isin(var_unique, keep2)]

        interactions = []
        col_names = []

        for f_val in factor_unique:
            for v_val in var_unique:
                mask = (factor_var == f_val) & (var == v_val)
                if np.any(mask):
                    col = np.zeros(n)
                    col[mask] = 1
                    interactions.append(col)

                    if name is not None:
                        col_names.append(f"{name}::{f_val}:var::{v_val}")
                    else:
                        col_names.append(f"{f_val}:{v_val}")

        matrix = np.zeros((n, 0)) if not interactions else np.column_stack(interactions)

    elif is_numeric_interaction:
        k = len(factor_unique)
        matrix = np.zeros((n, k))
        col_names = []

        for j, f_val in enumerate(factor_unique):
            mask = factor_var == f_val
            matrix[mask, j] = var[mask]

            if name is not None:
                col_names.append(f"{name}::{f_val}:var")
            else:
                col_names.append(f"{f_val}:var")

    else:
        k = len(factor_unique)
        matrix = np.zeros((n, k))
        col_names = []

        for j, f_val in enumerate(factor_unique):
            matrix[factor_var == f_val, j] = 1

            if name is not None:
                col_names.append(f"{name}::{f_val}")
            else:
                col_names.append(str(f_val))

    if return_dict:
        return {
            "matrix": matrix,
            "names": col_names,
            "reference_info": {
                "ref": ref,
                "ref2": ref2 if is_factor_interaction else None,
                "is_interaction": is_interaction,
                "is_numeric": is_numeric_interaction,
                "is_factor": is_factor_interaction,
            },
        }

    return matrix


def _handle_string_bin_spec(bin_spec, values, unique_values):
    """Handle string-based binning specifications."""
    if bin_spec.startswith("@"):
        return _apply_regex_binning(bin_spec[1:], values, unique_values)

    if "bin::" in bin_spec:
        return _apply_consecutive_binning(bin_spec, values, unique_values)

    raise ValueError(f"Unknown string bin specification: {bin_spec}")


def _apply_regex_binning(pattern, values, unique_values):
    """Apply regex pattern matching to group values."""
    regex = re.compile(pattern)

    matches = [val for val in unique_values if regex.search(str(val))]

    if not matches:
        raise ValueError(f"No values matched regex pattern '{pattern}'")

    if len(matches) > 1:
        mapping = {val: matches[0] for val in matches[1:]}
        result = values.copy()
        for old_val, new_val in mapping.items():
            result[values == old_val] = new_val
        return result

    return values


def _apply_consecutive_binning(bin_spec, values, unique_values):
    """Apply consecutive binning based on specification."""
    match = re.match(r"^(!!?)?bin::(\d+)$", bin_spec)
    if not match:
        raise ValueError(f"Invalid bin specification: {bin_spec}")

    prefix, bin_size = match.groups()
    bin_size = int(bin_size)

    if prefix is None and np.issubdtype(values.dtype, np.number):
        return (values // bin_size) * bin_size

    groups = _create_consecutive_groups(unique_values, bin_size, prefix)

    return _apply_grouping(values, groups)


def _create_consecutive_groups(unique_values, bin_size, prefix):
    """Create groups of consecutive values based on prefix."""
    n_values = len(unique_values)
    groups = []

    if prefix == "!!":
        for i in range(n_values - 1, -1, -bin_size):
            group_start = max(0, i - bin_size + 1)
            groups.append(unique_values[group_start : i + 1])
    elif prefix == "!":
        for i in range(0, n_values, bin_size):
            groups.append(unique_values[i : i + bin_size])
    else:
        for i in range(0, n_values, bin_size):
            groups.append(unique_values[i : i + bin_size])

    return groups


def _apply_grouping(values, groups):
    """Apply grouping by mapping values to group representatives."""
    result = values.copy()

    for group in groups:
        if len(group) > 1:
            group_representative = group[0]
            for val in group[1:]:
                result[values == val] = group_representative

    return result


def _handle_list_bin_spec(bin_spec, values, unique_values):
    """Handle list-based binning specifications."""
    if not bin_spec:
        return values

    existing_values = [val for val in bin_spec if val in unique_values]

    if not existing_values:
        return values

    result = values.copy()
    first_value = existing_values[0]

    for val in existing_values[1:]:
        result[values == val] = first_value

    return result


def _handle_dict_bin_spec(bin_spec, values, unique_values):
    """Handle dictionary-based binning specifications."""
    result = values.copy()

    for new_name, old_values in bin_spec.items():
        if isinstance(new_name, str) and new_name.startswith("@"):
            pattern = re.compile(new_name[1:])
            old_values = [val for val in unique_values if pattern.search(str(val))]
        else:
            if not isinstance(old_values, list):
                old_values = [old_values]

        for val in old_values:
            if val in unique_values:
                result[values == val] = new_name

    return result
