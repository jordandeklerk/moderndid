# pylint: disable=too-many-nested-blocks, function-redefined
"""Numba operations for continuous treatment DiD."""

from itertools import combinations, product

import numpy as np

try:
    import numba as nb
    from numba.typed import List

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None
    List = None


__all__ = [
    "HAS_NUMBA",
    "check_full_rank_crossprod",
    "compute_rsquared",
    "matrix_sqrt_eigendecomp",
    "create_nonzero_divisor",
    "compute_basis_dimension",
    "tensor_prod_model_matrix",
    "glp_model_matrix",
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


def _tensor_prod_model_matrix_impl(bases_flat, n_obs, total_cols):
    """Tensor product model matrix computation."""
    result = np.empty((n_obs, total_cols), dtype=np.float64)
    for row in range(n_obs):
        row_vectors = [basis[row, :] for basis in bases_flat]
        tensor_row = row_vectors[0].copy()
        for vec in row_vectors[1:]:
            tensor_row = np.kron(tensor_row, vec)
        result[row, :] = tensor_row
    return result


def _glp_model_matrix_impl(bases_flat, n_obs, dims):
    """GLP model matrix computation."""
    num_bases = len(dims)
    total_cols = sum(dims)

    for order in range(2, num_bases + 1):
        for indices in combinations(range(num_bases), order):
            interaction_dim = 1
            for idx in indices:
                interaction_dim *= dims[idx]
            total_cols += interaction_dim

    result = np.empty((n_obs, total_cols), dtype=np.float64)
    col_idx = 0

    for basis in bases_flat:
        n_cols = basis.shape[1]
        result[:, col_idx : col_idx + n_cols] = basis
        col_idx += n_cols

    for order in range(2, num_bases + 1):
        for indices in combinations(range(num_bases), order):
            selected_bases = [bases_flat[idx] for idx in indices]
            selected_dims = [dims[idx] for idx in indices]

            interaction_cols = int(np.prod(selected_dims))
            interaction_result = np.empty((n_obs, interaction_cols))

            interaction_col_idx = 0
            for func_indices in product(*[range(dim) for dim in selected_dims]):
                interaction_col = np.ones(n_obs)
                for basis_idx, func_idx in enumerate(func_indices):
                    interaction_col *= selected_bases[basis_idx][:, func_idx]

                interaction_result[:, interaction_col_idx] = interaction_col
                interaction_col_idx += 1

            result[:, col_idx : col_idx + interaction_cols] = interaction_result
            col_idx += interaction_cols

    return result


if HAS_NUMBA:

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

    @nb.vectorize([nb.float64(nb.float64, nb.float64)], cache=True)
    def _create_nonzero_divisor_impl(a, eps):
        """Ensure values are bounded away from zero."""
        if a < 0:
            return min(a, -eps)
        return max(a, eps)

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

    @nb.njit(cache=True)
    def _tensor_prod_model_matrix_impl(bases_flat, n_obs, dims, total_cols):
        """Tensor product model matrix computation."""
        result = np.empty((n_obs, total_cols), dtype=np.float64)

        for row in range(n_obs):
            tensor_row = bases_flat[0][row, :].copy()

            for i in range(1, len(dims)):
                vec = bases_flat[i][row, :]
                new_tensor_row = np.empty(len(tensor_row) * len(vec), dtype=np.float64)

                idx = 0
                for _, tensor_val in enumerate(tensor_row):
                    for vec_val in vec:
                        new_tensor_row[idx] = tensor_val * vec_val
                        idx += 1

                tensor_row = new_tensor_row

            result[row, :] = tensor_row

        return result

    @nb.njit(cache=True)
    def _glp_model_matrix_numba_impl(bases_flat, n_obs, dims):
        """GLP model matrix computation."""
        num_bases = len(dims)

        total_cols = 0
        for dim in dims:
            total_cols += dim

        for order in range(2, num_bases + 1):
            n_combinations = 1
            for i in range(order):
                n_combinations = n_combinations * (num_bases - i) // (i + 1)

            if order == 2:
                for i in range(num_bases):
                    for j in range(i + 1, num_bases):
                        total_cols += dims[i] * dims[j]
            elif order == 3:
                for i in range(num_bases):
                    for j in range(i + 1, num_bases):
                        for k in range(j + 1, num_bases):
                            total_cols += dims[i] * dims[j] * dims[k]

        result = np.empty((n_obs, total_cols), dtype=np.float64)
        col_idx = 0

        for basis_idx in range(num_bases):
            n_cols = dims[basis_idx]
            for col in range(n_cols):
                for row in range(n_obs):
                    result[row, col_idx] = bases_flat[basis_idx][row, col]
                col_idx += 1

        for i in range(num_bases):
            for j in range(i + 1, num_bases):
                for col_i in range(dims[i]):
                    for col_j in range(dims[j]):
                        for row in range(n_obs):
                            result[row, col_idx] = bases_flat[i][row, col_i] * bases_flat[j][row, col_j]
                        col_idx += 1

        for i in range(num_bases):
            for j in range(i + 1, num_bases):
                for k in range(j + 1, num_bases):
                    for col_i in range(dims[i]):
                        for col_j in range(dims[j]):
                            for col_k in range(dims[k]):
                                for row in range(n_obs):
                                    result[row, col_idx] = (
                                        bases_flat[i][row, col_i]
                                        * bases_flat[j][row, col_j]
                                        * bases_flat[k][row, col_k]
                                    )
                                col_idx += 1

        return result

    _glp_model_matrix_impl = _glp_model_matrix_numba_impl


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


def tensor_prod_model_matrix(bases):
    """Tensor product model matrix computation."""
    if not bases:
        raise ValueError("bases cannot be empty")

    for i, basis in enumerate(bases):
        if not isinstance(basis, np.ndarray):
            raise TypeError(f"bases[{i}] must be a NumPy array")
        if basis.ndim != 2:
            raise ValueError(f"bases[{i}] must be 2-dimensional")

    n_obs = bases[0].shape[0]
    for i, basis in enumerate(bases[1:], 1):
        if basis.shape[0] != n_obs:
            raise ValueError(
                f"All matrices must have same number of rows. bases[0] has {n_obs}, bases[{i}] has {basis.shape[0]}"
            )

    if HAS_NUMBA:
        bases_typed = List()
        for basis in bases:
            bases_typed.append(np.asarray(basis, dtype=np.float64))
    else:
        bases_typed = []
        for basis in bases:
            bases_typed.append(np.asarray(basis, dtype=np.float64))

    dims = np.array([basis.shape[1] for basis in bases], dtype=np.int32)
    total_cols = int(np.prod(dims))

    return _tensor_prod_model_matrix_impl(bases_typed, n_obs, dims, total_cols)


def glp_model_matrix(bases):
    """GLP model matrix computation."""
    if not bases:
        raise ValueError("bases cannot be empty")

    for i, basis in enumerate(bases):
        if not isinstance(basis, np.ndarray):
            raise TypeError(f"bases[{i}] must be a NumPy array")
        if basis.ndim != 2:
            raise ValueError(f"bases[{i}] must be 2-dimensional")

    n_obs = bases[0].shape[0]
    for i, basis in enumerate(bases[1:], 1):
        if basis.shape[0] != n_obs:
            raise ValueError(
                f"All matrices must have same number of rows. bases[0] has {n_obs}, bases[{i}] has {basis.shape[0]}"
            )

    if n_obs == 0:
        return np.empty((0, 0))

    dims = np.array([basis.shape[1] for basis in bases], dtype=np.int32)

    if HAS_NUMBA and len(bases) <= 3:
        bases_typed = List()
        for basis in bases:
            bases_typed.append(np.asarray(basis, dtype=np.float64))
        return _glp_model_matrix_numba_impl(bases_typed, n_obs, dims)

    bases_typed = []
    for basis in bases:
        bases_typed.append(np.asarray(basis, dtype=np.float64))

    return _glp_model_matrix_impl(bases_typed, n_obs, dims)
