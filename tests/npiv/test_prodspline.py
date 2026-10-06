"""Test the spline module."""

import numpy as np
import pytest

from moderndid.npiv.prodspline import (
    glp_model_matrix,
    prodspline,
    tensor_prod_model_matrix,
)
from moderndid.npiv.utils import basis_dimension


@pytest.mark.parametrize("basis_type", ["additive", "tensor", "glp"])
def test_basic_spline_types(continuous_data, degree_matrix, basis_type):
    result = prodspline(continuous_data, degree_matrix, basis=basis_type)

    assert result.basis.shape[0] == continuous_data.shape[0]
    assert result.basis_type == basis_type
    assert result.dim_no_tensor > 0
    assert np.array_equal(result.degree_matrix, degree_matrix)
    assert np.array_equal(result.n_segments, degree_matrix[:, 1] + 1)


def test_with_discrete_variables(continuous_data, discrete_data, degree_matrix, indicator_vector):
    result = prodspline(continuous_data, degree_matrix, z=discrete_data, indicator=indicator_vector, basis="additive")

    assert result.basis.shape[0] == continuous_data.shape[0]
    assert result.basis.shape[1] > degree_matrix.shape[0] * 5


@pytest.mark.filterwarnings("ignore:Some 'x' values beyond boundary knots:UserWarning")
def test_evaluation_data(degree_matrix, indicator_vector):
    np.random.seed(42)
    x = np.random.normal(0, 1, (100, 3))
    z = np.column_stack(
        [
            np.random.choice([0, 1, 2], 100),
            np.random.choice([0, 1], 100),
        ]
    )
    xeval = np.random.normal(0, 1, (50, 3))
    zeval = np.column_stack(
        [
            np.random.choice([0, 1, 2], 50),
            np.random.choice([0, 1], 50),
        ]
    )

    result = prodspline(x, degree_matrix, z=z, indicator=indicator_vector, xeval=xeval, zeval=zeval)

    assert result.basis.shape[0] == 50


@pytest.mark.parametrize(
    "deriv,deriv_index",
    [
        (0, 1),
        (1, 1),
        (2, 1),
        (1, 2),
    ],
)
def test_derivative_computation(simple_setup, deriv, deriv_index):
    x, K = simple_setup
    result = prodspline(x, K, deriv=deriv, deriv_index=deriv_index)

    assert result.basis.shape[0] == x.shape[0]
    assert result.basis.shape[1] > 0


@pytest.mark.parametrize("knots_type", ["quantiles", "uniform"])
def test_knot_types(knots_type):
    np.random.seed(42)
    x = np.random.exponential(1, (200, 2))
    K = np.array([[3, 3], [3, 3]])

    result = prodspline(x, K, knots=knots_type)

    assert result.basis.shape[0] == 200
    assert result.basis.shape[1] == 12


@pytest.mark.filterwarnings("ignore:Some 'x' values beyond boundary knots:UserWarning")
def test_min_max_bounds():
    x = np.random.uniform(-2, 2, (100, 2))
    K = np.array([[3, 3], [2, 4]])
    x_min = np.array([-1, -1])
    x_max = np.array([1, 1])

    result = prodspline(x, K, x_min=x_min, x_max=x_max)

    assert result.basis.shape[0] == 100


@pytest.mark.parametrize(
    "n_vars,K_shape",
    [
        (1, (1, 2)),
        (2, (2, 2)),
        (3, (3, 2)),
        (5, (5, 2)),
    ],
)
def test_multiple_variables(n_vars, K_shape):
    np.random.seed(42)
    x = np.random.normal(0, 1, (100, n_vars))
    K = np.tile([[3, 4]], (n_vars, 1))

    result = prodspline(x, K)

    assert result.basis.shape[0] == 100
    assert result.degree_matrix.shape == K_shape


def test_no_continuous_variables():
    n = 100
    x = np.zeros((n, 2))
    K = np.array([[0, 0], [0, 0]])
    z = np.random.choice([0, 1, 2], (n, 1))
    indicator = np.array([1])

    result = prodspline(x, K, z=z, indicator=indicator)

    assert result.basis.shape[0] == n
    assert result.dim_no_tensor >= 0


@pytest.mark.parametrize(
    "error_case,error_match",
    [
        ({"x": None}, "Must provide x and K"),
        ({"K": np.array([1, 2, 3])}, "K must be a two-column matrix"),
        ({"K": np.array([[3, 3]])}, "Dimension of x and K incompatible"),
        ({"deriv": -1}, "deriv is invalid"),
        ({"deriv_index": 3}, "deriv_index is invalid"),
    ],
)
def test_input_validation(simple_setup, error_case, error_match):
    x, K = simple_setup
    base_args = {"x": x, "K": K}
    base_args.update(error_case)

    with pytest.raises(ValueError, match=error_match):
        prodspline(**base_args)


@pytest.mark.filterwarnings("ignore:deriv order too large:UserWarning")
def test_derivative_warning(simple_setup):
    x, _ = simple_setup
    K = np.array([[2, 3], [3, 4]])

    with pytest.raises(ValueError, match="deriv must be smaller than degree plus 2"):
        prodspline(x, K, deriv=4, deriv_index=1)


def test_tensor_prod_model_matrix(basis_list):
    result = tensor_prod_model_matrix(basis_list)

    assert result.shape == (50, 3 * 2 * 4)


def test_tensor_prod_empty():
    with pytest.raises(ValueError, match="bases cannot be empty"):
        tensor_prod_model_matrix([])


def test_glp_model_matrix_basic():
    np.random.seed(42)
    bases = [
        np.random.normal(0, 1, (50, 3)),
        np.random.normal(0, 1, (50, 2)),
    ]

    result = glp_model_matrix(bases)

    assert result.shape == (50, 7)


def test_discrete_only():
    n = 100
    x = np.zeros((n, 1))
    K = np.array([[0, 0]])
    z = np.column_stack(
        [
            np.random.choice([0, 1, 2], n),
            np.random.choice([0, 1], n),
        ]
    )
    indicator = np.array([1, 1])

    result = prodspline(x, K, z=z, indicator=indicator)

    assert result.basis.shape[0] == n


@pytest.mark.parametrize(
    "basis_type1,basis_type2",
    [
        ("additive", "tensor"),
        ("additive", "glp"),
        ("tensor", "glp"),
    ],
)
def test_interaction_basis_dimensions(continuous_data, degree_matrix, basis_type1, basis_type2):
    result1 = prodspline(continuous_data, degree_matrix, basis=basis_type1)
    result2 = prodspline(continuous_data, degree_matrix, basis=basis_type2)

    assert result1.basis.shape[0] == result2.basis.shape[0]
    if basis_type1 == "additive" and basis_type2 == "tensor":
        assert result2.basis.shape[1] > result1.basis.shape[1]


@pytest.mark.parametrize("basis_type", ["additive", "tensor", "glp"])
def test_basis_properties(basis_type):
    np.random.seed(42)
    x = np.random.uniform(0, 1, (200, 2))
    K = np.array([[3, 3], [3, 3]])

    result = prodspline(x, K, basis=basis_type)

    assert np.all(np.isfinite(result.basis))
    assert np.min(result.basis) >= -10
    assert np.max(result.basis) <= 10


@pytest.mark.parametrize("segments", [1, 2, 3, 4])
def test_glp_dimension_matches_basis_dimension(segments):
    np.random.seed(1)
    x = np.random.uniform(0, 1, (400, 2))
    K = np.array([[3, segments - 1], [3, segments - 1]])

    glp = prodspline(x, K, knots="uniform", basis="glp").basis
    tensor = prodspline(x, K, knots="uniform", basis="tensor").basis

    assert glp.shape[1] == basis_dimension("glp", degree=[3, 3], segments=[segments, segments])
    assert glp.shape[1] < tensor.shape[1]
    assert np.linalg.matrix_rank(np.column_stack([np.ones(400), glp])) == glp.shape[1] + 1


def test_glp_derivative_zeroes_columns_without_the_variable():
    np.random.seed(2)
    x = np.random.uniform(0, 1, (100, 2))
    K = np.array([[3, 1], [3, 1]])

    level = prodspline(x, K, knots="uniform", basis="glp").basis
    deriv = prodspline(x, K, knots="uniform", basis="glp", deriv=1, deriv_index=2).basis
    x1_basis = prodspline(x[:, :1], K[:1], knots="uniform", basis="glp").basis

    assert level.shape == (100, 14)
    np.testing.assert_allclose(level[:, :4], x1_basis, rtol=1e-12)
    np.testing.assert_array_equal(deriv[:, :4], 0.0)
    assert np.all(np.any(deriv[:, 4:] != 0, axis=0))


def test_glp_small_marginal_bases_raise():
    np.random.seed(3)
    x = np.random.uniform(0, 1, (100, 2))

    with pytest.raises(ValueError, match="glp"):
        prodspline(x, np.array([[1, 0], [3, 1]]), knots="uniform", basis="glp")


def test_additive_derivative_ignores_other_regressors():
    np.random.seed(4)
    x = np.random.uniform(0, 1, (300, 2))
    K = np.array([[3, 3], [3, 3]])
    xeval = np.array([[0.2, 0.5], [0.8, 0.5]])

    deriv = prodspline(x, K, xeval=xeval, knots="uniform", basis="additive", deriv=1, deriv_index=2).basis

    np.testing.assert_array_equal(deriv[:, :6], 0.0)
    np.testing.assert_allclose(deriv[0], deriv[1], rtol=1e-12)


@pytest.mark.filterwarnings("ignore:deriv order too large:UserWarning")
@pytest.mark.parametrize("basis_type", ["additive", "tensor", "glp"])
def test_derivative_in_degree_zero_variable_is_zero(basis_type):
    np.random.seed(5)
    x = np.random.uniform(0, 1, (50, 2))
    K = np.array([[0, 0], [3, 2]])

    result = prodspline(x, K, knots="uniform", basis=basis_type, deriv=1, deriv_index=1)

    assert result.basis.shape[0] == 50
    np.testing.assert_array_equal(result.basis, 0.0)


@pytest.mark.filterwarnings("ignore:deriv order too large:UserWarning")
def test_derivative_without_splines_is_zero():
    x = np.random.uniform(0, 1, (20, 1))

    level = prodspline(x, np.array([[0, 0]]), knots="uniform")
    deriv = prodspline(x, np.array([[0, 0]]), knots="uniform", deriv=1)

    np.testing.assert_array_equal(level.basis, 1.0)
    np.testing.assert_array_equal(deriv.basis, 0.0)
