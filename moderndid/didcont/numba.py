"""Numba operations for continuous treatment DiD."""

import numpy as np

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


__all__ = [
    "HAS_NUMBA",
    "check_full_rank_crossprod",
    "compute_rsquared",
    "matrix_sqrt_eigendecomp",
    "create_nonzero_divisor",
    "compute_basis_dimension",
]


def _check_full_rank_crossprod_impl(x, tol=None):
    """Check if :math:`X'X` has full rank using eigenvalue decomposition."""
    xtx = x.T @ x

    eigenvalues = np.linalg.eigvalsh(xtx)

    n, p = x.shape
    max_dim = max(n, p)

    min_eig = eigenvalues[0]
    max_eig = eigenvalues[-1]

    if tol is None:
        max_sqrt_eig = np.sqrt(np.max(np.abs(eigenvalues)))
        tol = max_dim * max_sqrt_eig * np.finfo(float).eps

    is_full_rank = max_eig > 0 and np.abs(min_eig / max_eig) > tol
    condition_number = np.abs(max_eig / min_eig) if min_eig != 0 else np.inf

    return is_full_rank, condition_number, min_eig, max_eig


def _compute_rsquared_impl(y, y_pred):
    """Compute R-squared between observed and predicted values."""
    y_mean = np.mean(y)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r_squared = 1.0 - (ss_res / ss_tot)

    return np.clip(r_squared, 0.0, 1.0)


def _matrix_sqrt_eigendecomp_impl(x):
    """Compute matrix square root using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(x)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))

    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T


def _create_nonzero_divisor_impl(a, eps):
    """Ensure values are bounded away from zero."""
    a = np.asarray(a)
    result = np.empty_like(a, dtype=np.float64)

    mask_negative = a < 0
    mask_positive = a >= 0

    result[mask_negative] = np.minimum(a[mask_negative], -eps)
    result[mask_positive] = np.maximum(a[mask_positive], eps)

    return result


def _compute_additive_dimension(degree, segments):
    """Compute dimension of additive basis."""
    mask = degree > 0
    if not np.any(mask):
        return 0

    return np.sum(degree[mask] + segments[mask] - 1)


def _compute_tensor_dimension(degree, segments):
    """Compute dimension of tensor product basis."""
    mask = degree > 0
    if not np.any(mask):
        return 0

    return np.prod(degree[mask] + segments[mask])


def _compute_glp_dimension(degree, segments):
    """Compute dimension of generalized linear product basis."""
    mask = degree > 0
    if not np.any(mask):
        return 0

    dims = degree[mask] + segments[mask] - 1
    dims = dims[dims > 0]

    if len(dims) == 0:
        return 0

    dims = np.sort(dims)[::-1]
    k = len(dims)

    if k == 1:
        return dims[0]

    nd1 = np.ones(dims[0], dtype=np.int32)
    nd1[dims[0] - 1] = 0
    ncol_bs = dims[0]

    for i in range(1, k):
        ncol_bs, nd1 = _two_dimension_update(dims[0], dims[i], nd1, ncol_bs)

    return ncol_bs + k - 1


def _two_dimension_update(d1, d2, nd1, pd12):
    """Update dimension calculation for GLP basis."""
    if d2 == 1:
        return pd12, nd1

    d12 = d2

    for i in range(d1 - d2):
        d12 += d2 * nd1[i]

    if d2 > 1:
        for i in range(1, d2):
            d12 += (i + 1) * nd1[d1 - i - 1]

    d12 += nd1[d1 - 1]

    nd2 = np.zeros_like(nd1)

    for j in range(d1 - 1):
        for i in range(j, max(-1, j - d2), -1):
            if i >= 0:
                nd2[j] += nd1[i]
            else:
                nd2[j] += 1

    if d2 > 1:
        nd2[d1 - 1] = nd1[d1 - 1]
        for i in range(d1 - d2, d1 - 1):
            nd2[d1 - 1] += nd1[i]
    else:
        nd2[d1 - 1] = nd1[d1 - 1]

    return d12, nd2


if HAS_NUMBA:

    @nb.njit(cache=True)
    def _check_full_rank_crossprod_impl(x, tol=None):
        """Check if :math:`X'X` has full rank using eigenvalue decomposition."""
        xtx = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[1]):
                    xtx[j, k] += x[i, j] * x[i, k]

        eigenvalues = np.linalg.eigvals(xtx).real
        eigenvalues = np.sort(eigenvalues)

        n, p = x.shape
        max_dim = max(n, p)

        min_eig = eigenvalues[0]
        max_eig = eigenvalues[-1]

        if tol is None:
            max_sqrt_eig = np.sqrt(np.max(np.abs(eigenvalues)))
            tol = max_dim * max_sqrt_eig * np.finfo(np.float64).eps

        is_full_rank = max_eig > 0 and np.abs(min_eig / max_eig) > tol
        condition_number = np.abs(max_eig / min_eig) if min_eig != 0 else np.inf

        return is_full_rank, condition_number, min_eig, max_eig

    @nb.njit(cache=True)
    def _compute_rsquared_impl(y, y_pred):
        """Compute R-squared between observed and predicted values."""
        y_mean = 0.0
        n = len(y)
        for i in range(n):
            y_mean += y[i]
        y_mean /= n

        ss_res = 0.0
        ss_tot = 0.0

        for i in range(n):
            res = y[i] - y_pred[i]
            tot = y[i] - y_mean
            ss_res += res * res
            ss_tot += tot * tot

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        r_squared = 1.0 - (ss_res / ss_tot)

        if r_squared < 0.0:
            return 0.0
        if r_squared > 1.0:
            return 1.0
        return r_squared

    @nb.njit(cache=True)
    def _matrix_sqrt_eigendecomp_impl(x):
        """Compute matrix square root using eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eig(x)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        sqrt_eigenvalues = np.zeros_like(eigenvalues)
        for i, eigenvalue in enumerate(eigenvalues):
            if eigenvalue > 0:
                sqrt_eigenvalues[i] = np.sqrt(eigenvalue)

        n = x.shape[0]
        temp = np.zeros((n, n))
        result = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                temp[i, j] = eigenvectors[i, j] * sqrt_eigenvalues[j]

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i, j] += temp[i, k] * eigenvectors[j, k]

        return result

    @nb.njit(cache=True)
    def _create_nonzero_divisor_impl(a, eps):
        """Ensure values are bounded away from zero."""
        if a.ndim == 0:
            a_val = a.item()
            if a_val < 0:
                return a_val if a_val < -eps else -eps
            return a_val if a_val > eps else eps

        result = np.empty_like(a)

        for i in range(a.shape[0]):
            if a[i] < 0:
                result[i] = a[i] if a[i] < -eps else -eps
            else:
                result[i] = a[i] if a[i] > eps else eps

        return result

    @nb.njit(cache=True)
    def _compute_additive_dimension(degree, segments):
        """Compute dimension of additive basis."""
        dim = 0
        for i, deg in enumerate(degree):
            if deg > 0:
                dim += deg + segments[i] - 1
        return dim

    @nb.njit(cache=True)
    def _compute_tensor_dimension(degree, segments):
        """Compute dimension of tensor product basis."""
        dim = 1
        has_nonzero = False
        for i, deg in enumerate(degree):
            if deg > 0:
                dim *= deg + segments[i]
                has_nonzero = True
        return dim if has_nonzero else 0

    @nb.njit(cache=True)
    def _compute_glp_dimension(degree, segments):
        """Compute dimension of generalized linear product basis."""
        count = 0
        for deg in degree:
            if deg > 0:
                count += 1

        if count == 0:
            return 0

        dims = np.zeros(count, dtype=np.int32)
        idx = 0
        for i, deg in enumerate(degree):
            if deg > 0:
                dim_val = deg + segments[i] - 1
                if dim_val > 0:
                    dims[idx] = dim_val
                    idx += 1

        if idx == 0:
            return 0
        dims = dims[:idx]

        dims = np.sort(dims)[::-1]
        k = len(dims)

        if k == 1:
            return dims[0]

        nd1 = np.ones(dims[0], dtype=np.int32)
        nd1[dims[0] - 1] = 0
        ncol_bs = dims[0]

        for i in range(1, k):
            ncol_bs, nd1 = _two_dimension_update(dims[0], dims[i], nd1, ncol_bs)

        return ncol_bs + k - 1

    @nb.njit(cache=True)
    def _two_dimension_update(d1, d2, nd1, pd12):
        """Update dimension calculation for GLP basis."""
        if d2 == 1:
            return pd12, nd1

        d12 = d2

        for i in range(d1 - d2):
            d12 += d2 * nd1[i]

        if d2 > 1:
            for i in range(1, d2):
                d12 += (i + 1) * nd1[d1 - i - 1]

        d12 += nd1[d1 - 1]

        nd2 = np.zeros_like(nd1)

        for j in range(d1 - 1):
            for i in range(j, max(-1, j - d2), -1):
                if i >= 0:
                    nd2[j] += nd1[i]
                else:
                    nd2[j] += 1

        if d2 > 1:
            nd2[d1 - 1] = nd1[d1 - 1]
            for i in range(d1 - d2, d1 - 1):
                nd2[d1 - 1] += nd1[i]
        else:
            nd2[d1 - 1] = nd1[d1 - 1]

        return d12, nd2


def check_full_rank_crossprod(x, tol=None):
    """Check if :math:`X'X` has full rank using eigenvalue decomposition."""
    x = np.asarray(x, dtype=np.float64)
    return _check_full_rank_crossprod_impl(x, tol)


def compute_rsquared(y, y_pred):
    """Compute R-squared between observed and predicted values."""
    y = np.asarray(y, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _compute_rsquared_impl(y, y_pred)


def matrix_sqrt_eigendecomp(x):
    """Compute matrix square root using eigendecomposition."""
    x = np.asarray(x, dtype=np.float64)
    return _matrix_sqrt_eigendecomp_impl(x)


def create_nonzero_divisor(a, eps):
    """Ensure values are bounded away from zero."""
    a = np.asarray(a, dtype=np.float64)
    return _create_nonzero_divisor_impl(a, eps)


def compute_basis_dimension(basis_type, degree, segments):
    """Compute basis dimension with string dispatch."""
    degree = np.asarray(degree, dtype=np.int32)
    segments = np.asarray(segments, dtype=np.int32)

    if basis_type == "additive":
        return _compute_additive_dimension(degree, segments)
    if basis_type == "tensor":
        return _compute_tensor_dimension(degree, segments)
    if basis_type == "glp":
        return _compute_glp_dimension(degree, segments)
    return 0
