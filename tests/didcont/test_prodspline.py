# pylint: disable=redefined-outer-name
"""Test the spline module."""

import numpy as np
import pytest

from moderndid.didcont.npiv.spline import (
    glp_model_matrix,
    prodspline,
    tensor_prod_model_matrix,
)


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

    assert result.shape == (50, 11)


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
