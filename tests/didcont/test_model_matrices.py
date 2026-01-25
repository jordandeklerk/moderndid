# pylint: disable=redefined-outer-name
"""Test model matrices for continuous treatment DiD."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didcont.npiv.prodspline import glp_model_matrix, tensor_prod_model_matrix


def test_tensor_prod_basic_functionality(simple_bases):
    result = tensor_prod_model_matrix(simple_bases)

    assert result.shape == (3, 4)

    expected_row_0 = np.kron([1, 2], [0.5, 1.5])
    assert np.allclose(result[0, :], expected_row_0)


def test_tensor_prod_single_basis():
    bases = [np.array([[1, 2, 3], [4, 5, 6]])]
    result = tensor_prod_model_matrix(bases)

    assert np.array_equal(result, bases[0])


def test_tensor_prod_three_bases(three_bases):
    result = tensor_prod_model_matrix(three_bases)

    assert result.shape == (2, 6)

    expected_row_0 = np.kron(np.kron([1, 2], [1]), [0.5, 1.5, 2.5])
    assert np.allclose(result[0, :], expected_row_0)


def test_tensor_prod_dimension_consistency():
    bases = [
        np.random.normal(0, 1, (100, 5)),
        np.random.normal(0, 1, (100, 3)),
        np.random.normal(0, 1, (100, 2)),
    ]

    result = tensor_prod_model_matrix(bases)
    assert result.shape == (100, 5 * 3 * 2)


def test_tensor_prod_error_cases():
    with pytest.raises(ValueError, match="bases cannot be empty"):
        tensor_prod_model_matrix([])

    with pytest.raises(ValueError, match="must be 2-dimensional"):
        tensor_prod_model_matrix([np.array([1, 2])])

    with pytest.raises(TypeError):
        tensor_prod_model_matrix([np.array([[1, 2]]), "not_array"])

    with pytest.raises(ValueError, match="same number of rows"):
        tensor_prod_model_matrix([np.array([[1, 2], [3, 4]]), np.array([[5, 6]])])


def test_glp_basic_functionality(simple_bases):
    result = glp_model_matrix(simple_bases)

    assert result.shape[0] == 3
    assert result.shape[1] > 4

    assert np.allclose(result[:, :2], simple_bases[0])
    assert np.allclose(result[:, 2:4], simple_bases[1])


def test_glp_single_basis():
    bases = [np.array([[1, 2, 3], [4, 5, 6]])]
    result = glp_model_matrix(bases)

    assert np.array_equal(result, bases[0])


def test_glp_interaction_structure(simple_bases):
    result = glp_model_matrix(simple_bases)

    interaction_part = result[:, 4:]
    expected_interactions = np.array(
        [
            [1 * 0.5, 1 * 1.5, 2 * 0.5, 2 * 1.5],
            [3 * 2.5, 3 * 3.5, 4 * 2.5, 4 * 3.5],
            [5 * 4.5, 5 * 5.5, 6 * 4.5, 6 * 5.5],
        ]
    )

    assert np.allclose(interaction_part, expected_interactions)


def test_glp_three_way_interactions():
    bases = [
        np.array([[1, 2], [3, 4]]),
        np.array([[1], [2]]),
        np.array([[0.5], [1.5]]),
    ]

    result = glp_model_matrix(bases)

    marginal_cols = 2 + 1 + 1
    pairwise_cols = 2 * 1 + 2 * 1 + 1 * 1
    threeway_cols = 2 * 1 * 1
    expected_cols = marginal_cols + pairwise_cols + threeway_cols

    assert result.shape == (2, expected_cols)


def test_glp_empty_bases():
    result = glp_model_matrix([np.empty((0, 0))])
    assert result.shape == (0, 0)


def test_glp_error_cases():
    with pytest.raises(ValueError, match="bases cannot be empty"):
        glp_model_matrix([])

    with pytest.raises(ValueError, match="must be 2-dimensional"):
        glp_model_matrix([np.array([1, 2])])

    with pytest.raises(TypeError):
        glp_model_matrix([np.array([[1, 2]]), "not_array"])

    with pytest.raises(ValueError, match="same number of rows"):
        glp_model_matrix([np.array([[1, 2], [3, 4]]), np.array([[5, 6]])])


def test_numerical_stability():
    bases = [
        np.random.normal(0, 1, (1000, 10)),
        np.random.normal(0, 1, (1000, 5)),
    ]

    tensor_result = tensor_prod_model_matrix(bases)
    glp_result = glp_model_matrix(bases)

    assert np.all(np.isfinite(tensor_result))
    assert np.all(np.isfinite(glp_result))

    assert tensor_result.shape == (1000, 50)
    assert glp_result.shape[0] == 1000
    assert glp_result.shape[1] == 10 + 5 + 50


@pytest.mark.parametrize("n_bases", [2, 3, 4, 5])
def test_scaling_behavior(n_bases):
    bases = [np.random.normal(0, 1, (50, 3)) for _ in range(n_bases)]

    tensor_result = tensor_prod_model_matrix(bases)
    glp_result = glp_model_matrix(bases)

    assert tensor_result.shape == (50, 3**n_bases)

    expected_glp_cols = 0
    for order in range(1, n_bases + 1):
        from math import comb

        expected_glp_cols += comb(n_bases, order) * (3**order)

    assert glp_result.shape[1] == expected_glp_cols


def test_orthogonality_properties():
    np.random.seed(42)
    n = 3
    q1, _ = np.linalg.qr(np.random.normal(0, 1, (n, n)))
    q2, _ = np.linalg.qr(np.random.normal(0, 1, (n, n)))

    bases = [
        q1[:, :2],
        q2[:, :2],
    ]

    result = glp_model_matrix(bases)

    assert result.shape == (3, 8)
    assert np.allclose(result[:, :2], bases[0])
    assert np.allclose(result[:, 2:4], bases[1])
