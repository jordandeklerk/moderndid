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
    "find_rows_with_post_period_values",
    "create_first_differences_matrix",
    "create_second_differences_matrix",
    "check_matrix_sparsity",
    "quadratic_form",
    "safe_divide",
    "clip_values",
]


if HAS_NUMBA:

    @nb.jit(nopython=True, cache=True)
    def find_rows_with_post_period_values_jit(A, post_period_indices):
        """Find rows with non-zero values in post-period columns.

        Parameters
        ----------
        A : ndarray
            The constraint matrix to search.
        post_period_indices : ndarray or list
            Column indices corresponding to post-periods.

        Returns
        -------
        ndarray or None
            Array of row indices with non-zero post-period values,
            or None if no such rows exist.
        """
        rows = []
        for i in range(A.shape[0]):
            has_nonzero = False
            for j in post_period_indices:
                if A[i, j] != 0:
                    has_nonzero = True
                    break
            if has_nonzero:
                rows.append(i)

        if len(rows) > 0:
            return np.array(rows)
        return None

    @nb.jit(nopython=True, cache=True)
    def create_first_differences_matrix_jit(num_pre_periods, num_post_periods):
        """Create first differences matrix.

        Creates a matrix where each row represents the first difference
        between consecutive periods: delta_t - delta_{t-1}.

        Parameters
        ----------
        num_pre_periods : int
            Number of pre-treatment periods.
        num_post_periods : int
            Number of post-treatment periods.

        Returns
        -------
        ndarray
            First differences matrix of shape (T-1, T+1) where T = num_pre + num_post.
        """
        total_periods = num_pre_periods + num_post_periods + 1
        a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))
        for r in range(num_pre_periods + num_post_periods):
            a_tilde[r, r] = -1.0
            a_tilde[r, r + 1] = 1.0
        return a_tilde

    @nb.jit(nopython=True, cache=True)
    def create_second_differences_matrix_jit(num_constraints, total_periods):
        """Create second differences constraint matrix.

        Creates a matrix where each row represents the second difference:
        delta_{t-1} - 2*delta_t + delta_{t+1}.

        Parameters
        ----------
        num_constraints : int
            Number of second difference constraints.
        total_periods : int
            Total number of periods.

        Returns
        -------
        ndarray
            Second differences matrix.
        """
        A_positive = np.zeros((num_constraints, total_periods))
        for i in range(num_constraints):
            if i + 3 <= total_periods:
                A_positive[i, i] = 1.0
                A_positive[i, i + 1] = -2.0
                A_positive[i, i + 2] = 1.0
        return A_positive

    @nb.jit(nopython=True, cache=True)
    def check_matrix_sparsity_pattern_jit(A, threshold=1e-10):
        """Check sparsity pattern of a matrix.

        Parameters
        ----------
        A : ndarray
            Matrix to check.
        threshold : float, default=1e-10
            Values below this threshold are considered zero.

        Returns
        -------
        tuple
            (nnz, sparsity_ratio) where nnz is number of non-zeros
            and sparsity_ratio is fraction of zeros.
        """
        nnz = 0
        total_elements = A.shape[0] * A.shape[1]

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.abs(A[i, j]) > threshold:
                    nnz += 1

        sparsity_ratio = 1.0 - nnz / total_elements
        return nnz, sparsity_ratio

    @guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->(n)", nopython=True, cache=True)
    def safe_divide_ufunc(x, y, result):
        """Element-wise safe division avoiding division by zero."""
        for i in range(x.shape[0]):
            if np.abs(y[i]) < 1e-10:
                result[i] = 0.0
            else:
                result[i] = x[i] / y[i]

    @guvectorize([(float64[:], float64, float64, float64[:])], "(n),(),()->(n)", nopython=True, cache=True)
    def clip_values_ufunc(x, lower, upper, result):
        """Element-wise clipping to bounds."""
        for i in range(x.shape[0]):
            if x[i] < lower:
                result[i] = lower
            elif x[i] > upper:
                result[i] = upper
            else:
                result[i] = x[i]

    @nb.jit(nopython=True, cache=True, parallel=True)
    def quadratic_form_jit(x, A):
        """Compute quadratic form x'Ax.

        Parameters
        ----------
        x : ndarray
            Vector of shape (n,).
        A : ndarray
            Matrix of shape (n, n).

        Returns
        -------
        float
            The quadratic form x'Ax.
        """
        n = x.shape[0]
        result = 0.0

        for i in nb.prange(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += A[i, j] * x[j]
            result += x[i] * row_sum

        return result


else:

    def find_rows_with_post_period_values_jit(A, post_period_indices):
        """Find rows with non-zero values in post-period columns."""
        has_post_period_values = np.any(A[:, post_period_indices] != 0, axis=1)
        rows_for_arp = np.where(has_post_period_values)[0]
        return rows_for_arp if len(rows_for_arp) > 0 else None

    def create_first_differences_matrix_jit(num_pre_periods, num_post_periods):
        """Create first differences matrix."""
        total_periods = num_pre_periods + num_post_periods + 1
        a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))
        for r in range(num_pre_periods + num_post_periods):
            a_tilde[r, r : (r + 2)] = [-1, 1]
        return a_tilde

    def create_second_differences_matrix_jit(num_constraints, total_periods):
        """Create second differences constraint matrix."""
        A_positive = np.zeros((num_constraints, total_periods))
        for i in range(num_constraints):
            if i + 3 <= total_periods:
                A_positive[i, i : i + 3] = [1, -2, 1]
        return A_positive

    def check_matrix_sparsity_pattern_jit(A, threshold=1e-10):
        """Check sparsity pattern of a matrix."""
        nnz = np.sum(np.abs(A) > threshold)
        total_elements = A.size
        sparsity_ratio = 1.0 - nnz / total_elements
        return nnz, sparsity_ratio

    def safe_divide_ufunc(x, y, out=None):
        """Element-wise safe division avoiding division by zero."""
        if out is None:
            out = np.zeros_like(x)
        mask = np.abs(y) >= 1e-10
        out[mask] = x[mask] / y[mask]
        out[~mask] = 0.0
        return out

    def clip_values_ufunc(x, lower, upper, out=None):
        """Element-wise clipping to bounds."""
        return np.clip(x, lower, upper, out=out)

    def quadratic_form_jit(x, A):
        """Compute quadratic form x'Ax."""
        return x @ A @ x


def find_rows_with_post_period_values(A, post_period_indices):
    """Find rows with non-zero values in post-period columns.

    Parameters
    ----------
    A : ndarray
        The constraint matrix to search.
    post_period_indices : array-like
        Column indices corresponding to post-periods.

    Returns
    -------
    ndarray or None
        Array of row indices with non-zero post-period values,
        or None if no such rows exist.
    """
    if isinstance(post_period_indices, list):
        post_period_indices = np.array(post_period_indices)
    return find_rows_with_post_period_values_jit(A, post_period_indices)


def create_first_differences_matrix(num_pre_periods, num_post_periods):
    """Create first differences matrix.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    ndarray
        First differences matrix of shape (T-1, T+1) where T = num_pre + num_post.
    """
    return create_first_differences_matrix_jit(num_pre_periods, num_post_periods)


def create_second_differences_matrix(num_constraints, total_periods):
    """Create second differences constraint matrix.

    Parameters
    ----------
    num_constraints : int
        Number of second difference constraints.
    total_periods : int
        Total number of periods.

    Returns
    -------
    ndarray
        Second differences matrix.
    """
    return create_second_differences_matrix_jit(num_constraints, total_periods)


def check_matrix_sparsity(A, threshold=1e-10):
    """Check sparsity pattern of a matrix.

    Parameters
    ----------
    A : ndarray
        Matrix to check.
    threshold : float, default=1e-10
        Values below this threshold are considered zero.

    Returns
    -------
    dict
        Dictionary with 'nnz' (number of non-zeros), 'sparsity_ratio'
        (fraction of zeros), and 'is_sparse' (True if >50% zeros).
    """
    nnz, sparsity_ratio = check_matrix_sparsity_pattern_jit(A, threshold)
    return {"nnz": int(nnz), "sparsity_ratio": float(sparsity_ratio), "is_sparse": sparsity_ratio > 0.5}


def quadratic_form(x, A):
    """Compute quadratic form x'Ax.

    Parameters
    ----------
    x : ndarray
        Vector of shape (n,).
    A : ndarray
        Matrix of shape (n, n).

    Returns
    -------
    float
        The quadratic form x'Ax.
    """
    return quadratic_form_jit(x, A)


def safe_divide(x, y, out=None):
    """Element-wise safe division avoiding division by zero.

    Parameters
    ----------
    x : ndarray
        Numerator array.
    y : ndarray
        Denominator array.
    out : ndarray, optional
        Output array.

    Returns
    -------
    ndarray
        Result of x/y with zeros where y is near zero.
    """
    return safe_divide_ufunc(x, y, out)


def clip_values(x, lower, upper, out=None):
    """Element-wise clipping to bounds.

    Parameters
    ----------
    x : ndarray
        Array to clip.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    out : ndarray, optional
        Output array.

    Returns
    -------
    ndarray
        Clipped array.
    """
    return clip_values_ufunc(x, lower, upper, out)
