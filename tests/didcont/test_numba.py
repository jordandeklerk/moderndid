"""Test Numba functions."""

import timeit

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didcont import numba as nb_module


def test_check_full_rank_crossprod_correctness(matrices_small):
    x, _, _, _ = matrices_small

    is_full_rank, cond_num, min_eig, max_eig = nb_module.check_full_rank_crossprod(x)

    assert isinstance(is_full_rank, bool | np.bool_)
    assert cond_num > 0
    assert min_eig > 0
    assert max_eig > 0
    assert max_eig >= min_eig

    x_rank_def = np.column_stack([x[:, 0], x[:, 0] * 2])
    is_full_rank_def, _, _, _ = nb_module.check_full_rank_crossprod(x_rank_def)
    assert not is_full_rank_def


def test_compute_rsquared_correctness(matrices_small):
    _, y, y_pred, _ = matrices_small

    r2 = nb_module.compute_rsquared(y, y_pred)

    assert 0 <= r2 <= 1

    r2_perfect = nb_module.compute_rsquared(y, y)
    assert np.isclose(r2_perfect, 1.0)

    y_const = np.ones_like(y)
    r2_const = nb_module.compute_rsquared(y_const, y_const)
    assert np.isclose(r2_const, 1.0)


def test_matrix_sqrt_eigendecomp_correctness(matrices_small):
    _, _, _, sym_matrix = matrices_small

    sym_matrix = sym_matrix @ sym_matrix.T + 0.1 * np.eye(len(sym_matrix))

    sqrt_matrix = nb_module.matrix_sqrt_eigendecomp(sym_matrix)

    reconstructed = sqrt_matrix @ sqrt_matrix
    assert np.allclose(reconstructed, sym_matrix, rtol=1e-10, atol=1e-10)


def test_create_nonzero_divisor_correctness():
    eps = 1e-10

    assert nb_module.create_nonzero_divisor(2.0, eps) == 2.0
    assert nb_module.create_nonzero_divisor(-2.0, eps) == -2.0
    assert nb_module.create_nonzero_divisor(0.0, eps) == eps
    assert nb_module.create_nonzero_divisor(eps / 2, eps) == eps
    assert nb_module.create_nonzero_divisor(-eps / 2, eps) == -eps

    a = np.array([2.0, -2.0, 0.0, eps / 2, -eps / 2])
    result = nb_module.create_nonzero_divisor(a, eps)
    expected = np.array([2.0, -2.0, eps, eps, -eps])
    assert np.allclose(result, expected)


def test_compute_basis_dimension_correctness():
    degree = np.array([2, 3, 1])
    segments = np.array([3, 2, 4])

    dim_add = nb_module.compute_basis_dimension("additive", degree, segments)
    assert dim_add == 12

    dim_tensor = nb_module.compute_basis_dimension("tensor", degree, segments)
    assert dim_tensor == 125

    degree_zero = np.array([2, 0, 1])
    segments_zero = np.array([3, 2, 4])

    dim_add_zero = nb_module.compute_basis_dimension("additive", degree_zero, segments_zero)
    assert dim_add_zero == 8

    dim_tensor_zero = nb_module.compute_basis_dimension("tensor", degree_zero, segments_zero)
    assert dim_tensor_zero == 25


def test_tensor_prod_model_matrix_correctness(basis_matrices):
    bases = basis_matrices
    n_obs = bases[0].shape[0]
    expected_cols = np.prod([b.shape[1] for b in bases])

    result = nb_module.tensor_prod_model_matrix(bases)

    assert result.shape == (n_obs, expected_cols)

    row0_expected = np.kron(np.kron(bases[0][0, :], bases[1][0, :]), bases[2][0, :])
    assert np.allclose(result[0, :], row0_expected)

    single_result = nb_module.tensor_prod_model_matrix([bases[0]])
    assert np.allclose(single_result, bases[0])


def test_glp_model_matrix_correctness(basis_matrices):
    bases = basis_matrices[:2]
    n_obs = bases[0].shape[0]

    result = nb_module.glp_model_matrix(bases)

    expected_cols = 3 + 4 + (3 * 4)
    assert result.shape == (n_obs, expected_cols)

    assert np.allclose(result[:, :3], bases[0])
    assert np.allclose(result[:, 3:7], bases[1])

    assert np.allclose(result[:, 7], bases[0][:, 0] * bases[1][:, 0])
    assert np.allclose(result[:, 8], bases[0][:, 0] * bases[1][:, 1])


def test_numpy_performance_check_full_rank(matrices_large):
    """Test that NumPy implementation is efficient for full rank check."""
    x, _, _, _ = matrices_large
    x = x[:1000, :20]

    time_taken = timeit.timeit(lambda: nb_module.check_full_rank_crossprod(x), number=10)

    assert time_taken < 1.0, f"Function took {time_taken:.2f}s, should be < 1s"


def test_numpy_performance_matrix_sqrt(matrices_large):
    """Test that NumPy implementation is efficient for matrix sqrt."""
    _, _, _, sym_matrix = matrices_large
    sym_matrix = sym_matrix[:20, :20]

    time_taken = timeit.timeit(lambda: nb_module.matrix_sqrt_eigendecomp(sym_matrix), number=50)

    assert time_taken < 0.5, f"Function took {time_taken:.2f}s, should be < 0.5s"


@pytest.mark.skipif(not nb_module.HAS_NUMBA, reason="Numba not installed")
def test_numba_performance_tensor_product():
    from numba.typed import List

    np.random.seed(42)
    n_obs = 1000
    bases = [
        np.random.randn(n_obs, 5),
        np.random.randn(n_obs, 4),
        np.random.randn(n_obs, 3),
    ]
    dims = np.array([b.shape[1] for b in bases], dtype=np.int32)
    total_cols = int(np.prod(dims))

    # Convert to numba typed list to avoid reflection warning
    bases_typed = List()
    for basis in bases:
        bases_typed.append(np.asarray(basis, dtype=np.float64))

    _ = nb_module._tensor_prod_model_matrix_impl(bases_typed, n_obs, dims, total_cols)

    def pure_python_tensor():
        result = np.empty((n_obs, total_cols), dtype=np.float64)
        for row in range(n_obs):
            row_vectors = [basis[row, :] for basis in bases]
            tensor_row = row_vectors[0].copy()
            for vec in row_vectors[1:]:
                tensor_row = np.kron(tensor_row, vec)
            result[row, :] = tensor_row
        return result

    time_pure = timeit.timeit(pure_python_tensor, number=5)

    time_numba = timeit.timeit(
        lambda: nb_module._tensor_prod_model_matrix_impl(bases_typed, n_obs, dims, total_cols), number=5
    )

    speedup = time_pure / time_numba
    assert speedup > 10.0, f"Numba speedup only {speedup:.2f}x, expected > 10x"


def test_error_handling():
    with pytest.raises(ValueError, match="bases cannot be empty"):
        nb_module.tensor_prod_model_matrix([])

    with pytest.raises(ValueError, match="bases cannot be empty"):
        nb_module.glp_model_matrix([])

    with pytest.raises(TypeError, match="must be a NumPy array"):
        nb_module.tensor_prod_model_matrix([[[1, 2], [3, 4]]])

    bases_mismatch = [
        np.random.randn(10, 3),
        np.random.randn(15, 4),
    ]
    with pytest.raises(ValueError, match="All matrices must have same number of rows"):
        nb_module.tensor_prod_model_matrix(bases_mismatch)


@pytest.mark.skipif(nb_module.HAS_NUMBA, reason="Testing without Numba")
def test_without_numba():
    x = np.random.randn(10, 3)
    is_full_rank, _, _, _ = nb_module.check_full_rank_crossprod(x)
    assert isinstance(is_full_rank, bool | np.bool_)

    y = np.random.randn(10)
    y_pred = y + 0.1 * np.random.randn(10)
    r2 = nb_module.compute_rsquared(y, y_pred)
    assert 0 <= r2 <= 1
