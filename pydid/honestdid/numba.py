"""Numba operations."""

import numpy as np

try:
    import numba as nb
    from numba import float64, guvectorize

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


__all__ = [
    "HAS_NUMBA",
    "lee_coefficient",
    "selection_matrix",
    "compute_bounds",
    "find_rows_with_post_period_values",
    "create_first_differences_matrix",
    "create_second_differences_matrix",
    "check_matrix_sparsity",
    "quadratic_form",
    "safe_divide",
    "clip_values",
    "prepare_theta_grid_y_values",
    "compute_hybrid_dbar",
]


def _find_rows_with_post_period_values_impl(A, post_period_indices):
    has_post_period_values = np.any(A[:, post_period_indices] != 0, axis=1)
    rows_for_arp = np.where(has_post_period_values)[0]
    return rows_for_arp if len(rows_for_arp) > 0 else None


def _create_first_differences_matrix_impl(num_pre_periods, num_post_periods):
    total_periods = num_pre_periods + num_post_periods + 1
    a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))
    for r in range(num_pre_periods + num_post_periods):
        a_tilde[r, r : (r + 2)] = [-1, 1]
    return a_tilde


def _create_second_differences_matrix_impl(num_constraints, total_periods):
    A_positive = np.zeros((num_constraints, total_periods))
    for i in range(num_constraints):
        if i + 3 <= total_periods:
            A_positive[i, i : i + 3] = [1, -2, 1]
    return A_positive


def _check_matrix_sparsity_pattern_impl(A, threshold=1e-10):
    nnz = np.sum(np.abs(A) > threshold)
    total_elements = A.size
    sparsity_ratio = 1.0 - nnz / total_elements
    return nnz, sparsity_ratio


def _safe_divide_impl(x, y, out=None):
    if out is None:
        out = np.zeros_like(x, dtype=float)
    mask = np.abs(y) >= 1e-10
    np.divide(x, y, out=out, where=mask)
    out[~mask] = 0.0
    return out


def _clip_values_impl(x, lower, upper, out=None):
    return np.clip(x, lower, upper, out=out)


def _quadratic_form_impl(x, A):
    return x @ A @ x


def _prepare_theta_grid_y_values_impl(beta_hat_or_y, period_vec_or_a_inv, theta_grid):
    return beta_hat_or_y - np.outer(theta_grid, period_vec_or_a_inv)


def _compute_hybrid_dbar_impl(flci_halflength, vbar, d_vec, a_gamma_inv_one, theta):
    vbar_d = np.dot(vbar, d_vec)
    vbar_a = np.dot(vbar, a_gamma_inv_one)
    return np.array([flci_halflength - vbar_d + (1 - vbar_a) * theta, flci_halflength + vbar_d - (1 - vbar_a) * theta])


def _lee_coefficient_impl(eta, sigma):
    sigma_eta = sigma @ eta
    eta_sigma_eta = eta.T @ sigma_eta
    if np.abs(eta_sigma_eta) < 1e-10:
        raise ValueError("Estimated coefficient is effectively zero, cannot compute coefficient.")
    return sigma_eta / eta_sigma_eta


def _selection_matrix_impl(selection_0idx, size, n_selections, select_rows):
    if select_rows:
        m = np.zeros((n_selections, size))
        for i, idx in enumerate(selection_0idx):
            m[i, idx] = 1
    else:
        m = np.zeros((size, n_selections))
        for i, idx in enumerate(selection_0idx):
            m[idx, i] = 1
    return m


def _compute_bounds_impl(eta, sigma, A, b, z):
    c = _lee_coefficient_impl(eta, sigma)
    Az = A @ z
    Ac = A @ c
    nonzero_mask = np.abs(Ac) > 1e-10
    objective = np.full_like(Ac, np.nan)
    objective[nonzero_mask] = (b[nonzero_mask] - Az[nonzero_mask]) / Ac[nonzero_mask]
    ac_negative_idx = Ac < 0
    ac_positive_idx = Ac > 0
    lower_bound = np.max(objective[ac_negative_idx]) if np.any(ac_negative_idx) else -np.inf
    upper_bound = np.min(objective[ac_positive_idx]) if np.any(ac_positive_idx) else np.inf
    return lower_bound, upper_bound


if HAS_NUMBA:

    @nb.jit(nopython=True, cache=True)
    def _find_rows_with_post_period_values_impl(A, post_period_indices):
        rows = []
        for i in range(A.shape[0]):
            for j in post_period_indices:
                if A[i, j] != 0:
                    rows.append(i)
                    break
        return np.array(rows) if rows else None

    @nb.jit(nopython=True, cache=True)
    def _create_first_differences_matrix_impl(num_pre_periods, num_post_periods):
        total_periods = num_pre_periods + num_post_periods + 1
        a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))
        for r in range(num_pre_periods + num_post_periods):
            a_tilde[r, r] = -1.0
            a_tilde[r, r + 1] = 1.0
        return a_tilde

    @nb.jit(nopython=True, cache=True)
    def _create_second_differences_matrix_impl(num_constraints, total_periods):
        A_positive = np.zeros((num_constraints, total_periods))
        for i in range(num_constraints):
            if i + 3 <= total_periods:
                A_positive[i, i] = 1.0
                A_positive[i, i + 1] = -2.0
                A_positive[i, i + 2] = 1.0
        return A_positive

    @nb.jit(nopython=True, cache=True)
    def _check_matrix_sparsity_pattern_impl(A, threshold=1e-10):
        nnz = 0
        total_elements = A.shape[0] * A.shape[1]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.abs(A[i, j]) > threshold:
                    nnz += 1
        sparsity_ratio = 1.0 - nnz / total_elements
        return nnz, sparsity_ratio

    @guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->(n)", nopython=True, cache=True)
    def _safe_divide_impl(x, y, result):
        for i in range(x.shape[0]):
            if np.abs(y[i]) < 1e-10:
                result[i] = 0.0
            else:
                result[i] = x[i] / y[i]

    @guvectorize([(float64[:], float64, float64, float64[:])], "(n),(),()->(n)", nopython=True, cache=True)
    def _clip_values_impl(x, lower, upper, result):
        for i in range(x.shape[0]):
            if x[i] < lower:
                result[i] = lower
            elif x[i] > upper:
                result[i] = upper
            else:
                result[i] = x[i]

    @nb.jit(nopython=True, cache=True, parallel=True)
    def _quadratic_form_impl(x, A):
        n = x.shape[0]
        result = 0.0
        for i in nb.prange(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += A[i, j] * x[j]
            result += x[i] * row_sum
        return result

    @nb.jit(nopython=True, parallel=True, cache=True)
    def _prepare_theta_grid_y_values_impl(beta_hat_or_y, period_vec_or_a_inv, theta_grid):
        n_grid = len(theta_grid)
        n_params = len(beta_hat_or_y)
        y_matrix = np.empty((n_grid, n_params))
        for i in nb.prange(n_grid):
            y_matrix[i] = beta_hat_or_y - period_vec_or_a_inv * theta_grid[i]
        return y_matrix

    @nb.jit(nopython=True, cache=True)
    def _compute_hybrid_dbar_impl(flci_halflength, vbar, d_vec, a_gamma_inv_one, theta):
        vbar_d = np.dot(vbar, d_vec)
        vbar_a = np.dot(vbar, a_gamma_inv_one)
        return np.array(
            [flci_halflength - vbar_d + (1 - vbar_a) * theta, flci_halflength + vbar_d - (1 - vbar_a) * theta]
        )

    @nb.jit(nopython=True, cache=True)
    def _lee_coefficient_impl(eta, sigma):
        sigma_eta = np.dot(sigma, eta)
        eta_sigma_eta = np.dot(eta, sigma_eta)
        if np.abs(eta_sigma_eta) < 1e-10:
            raise ValueError("Estimated coefficient is effectively zero, cannot compute coefficient.")
        return sigma_eta / eta_sigma_eta

    @nb.jit(nopython=True, cache=True)
    def _selection_matrix_impl(selection_0idx, size, n_selections, select_rows):
        if select_rows:
            m = np.zeros((n_selections, size))
            for i in range(n_selections):
                m[i, selection_0idx[i]] = 1.0
        else:
            m = np.zeros((size, n_selections))
            for i in range(n_selections):
                m[selection_0idx[i], i] = 1.0
        return m

    @nb.jit(nopython=True, cache=True)
    def _compute_bounds_impl(eta, sigma, A, b, z):
        sigma_eta = np.dot(sigma, eta)
        eta_sigma_eta = np.dot(eta, sigma_eta)
        c = sigma_eta / eta_sigma_eta
        Az = np.dot(A, z)
        Ac = np.dot(A, c)
        lower_bound = -np.inf
        upper_bound = np.inf
        for i, ac_val in enumerate(Ac):
            if abs(ac_val) > 1e-10:
                obj_val = (b[i] - Az[i]) / ac_val
                if ac_val < 0:
                    lower_bound = max(lower_bound, obj_val)
                elif obj_val < upper_bound:
                    upper_bound = obj_val
        return lower_bound, upper_bound


def lee_coefficient(eta, sigma):
    """Compute coefficient for constructing confidence intervals."""
    eta = np.asarray(eta, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64)
    return _lee_coefficient_impl(eta, sigma)


def selection_matrix(selection, size, select="columns"):
    """Create a selection matrix for extracting specific rows or columns."""
    selection = np.asarray(selection)
    selection_0idx = selection - 1
    n_selections = len(selection)
    select_rows = select == "rows"
    return _selection_matrix_impl(selection_0idx, size, n_selections, select_rows)


def compute_bounds(eta, sigma, A, b, z):
    """Compute lower and upper bounds for confidence intervals."""
    eta = np.asarray(eta, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    z = np.asarray(z, dtype=np.float64).flatten()
    return _compute_bounds_impl(eta, sigma, A, b, z)


def find_rows_with_post_period_values(A, post_period_indices):
    """Find rows with non-zero values in post-period columns."""
    if isinstance(post_period_indices, list):
        post_period_indices = np.array(post_period_indices)
    return _find_rows_with_post_period_values_impl(A, post_period_indices)


def create_first_differences_matrix(num_pre_periods, num_post_periods):
    """Create first differences matrix."""
    return _create_first_differences_matrix_impl(num_pre_periods, num_post_periods)


def create_second_differences_matrix(num_constraints, total_periods):
    """Create second differences constraint matrix."""
    return _create_second_differences_matrix_impl(num_constraints, total_periods)


def check_matrix_sparsity(A, threshold=1e-10):
    """Check sparsity pattern of a matrix."""
    nnz, sparsity_ratio = _check_matrix_sparsity_pattern_impl(A, threshold)
    return {"nnz": int(nnz), "sparsity_ratio": float(sparsity_ratio), "is_sparse": sparsity_ratio > 0.5}


def quadratic_form(x, A):
    """Compute quadratic form x'Ax."""
    return _quadratic_form_impl(x, A)


def safe_divide(x, y, out=None):
    """Element-wise safe division avoiding division by zero."""
    return _safe_divide_impl(x, y, out)


def clip_values(x, lower, upper, out=None):
    """Element-wise clipping to bounds."""
    return _clip_values_impl(x, lower, upper, out)


def prepare_theta_grid_y_values(beta_hat_or_y, period_vec_or_a_inv, theta_grid):
    """Prepare y values for all theta values in grid."""
    return _prepare_theta_grid_y_values_impl(beta_hat_or_y, period_vec_or_a_inv, theta_grid)


def compute_hybrid_dbar(flci_halflength, vbar, d_vec, a_gamma_inv_one, theta):
    """Compute hybrid dbar for FLCI case."""
    return _compute_hybrid_dbar_impl(flci_halflength, vbar, d_vec, a_gamma_inv_one, theta)
