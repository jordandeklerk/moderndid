"""Tests for utility functions."""

import warnings

import numpy as np
import pytest

from moderndid import (
    basis_vector,
    bin_factor,
    compute_bounds,
    create_interactions,
    lee_coefficient,
    selection_matrix,
    validate_conformable,
    validate_symmetric_psd,
)


def test_selection_matrix_columns():
    m = selection_matrix([1, 3], 4, select="columns")
    assert m.shape == (4, 2)
    expected = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(m, expected)


def test_selection_matrix_rows():
    m = selection_matrix([2, 4], 5, select="rows")
    assert m.shape == (2, 5)
    expected = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(m, expected)


def test_selection_matrix_single_element():
    m = selection_matrix([2], 3, select="columns")
    assert m.shape == (3, 1)
    expected = np.array([[0], [1], [0]])
    np.testing.assert_array_equal(m, expected)


def test_lee_coefficient():
    eta = np.array([1, 2])
    sigma = np.array([[2, 1], [1, 3]])
    c = lee_coefficient(eta, sigma)

    expected = np.array([4 / 18, 7 / 18])
    np.testing.assert_allclose(c, expected, rtol=1e-10)


def test_lee_coefficient_zero_denominator():
    eta = np.array([1, -1])
    sigma = np.array([[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="Estimated coefficient is effectively zero"):
        lee_coefficient(eta, sigma)


def test_compute_bounds():
    eta = np.array([1, 0])
    sigma = np.array([[1, 0], [0, 1]])
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([2, 0, 3, -1])
    z = np.array([1, 1])

    VLo, VUp = compute_bounds(eta, sigma, A, b, z)

    assert VLo == -1.0
    assert VUp == 1.0


def test_compute_bounds_no_constraints():
    eta = np.array([0, 1])
    sigma = np.array([[1, 0], [0, 1]])
    A = np.array([[1, 0], [1, 0]])
    b = np.array([2, 3])
    z = np.array([0, 0])

    VLo, VUp = compute_bounds(eta, sigma, A, b, z)

    assert VLo == -np.inf
    assert VUp == np.inf


def test_basis_vector():
    e2 = basis_vector(2, 4)
    assert e2.shape == (4, 1)
    expected = np.array([[0], [1], [0], [0]])
    np.testing.assert_array_equal(e2, expected)


def test_basis_vector_first():
    e1 = basis_vector(1, 3)
    assert e1.shape == (3, 1)
    expected = np.array([[1], [0], [0]])
    np.testing.assert_array_equal(e1, expected)


def test_basis_vector_invalid_index():
    with pytest.raises(ValueError, match="index must be between 1 and 3"):
        basis_vector(4, 3)

    with pytest.raises(ValueError, match="index must be between 1 and 3"):
        basis_vector(0, 3)


def test_validate_symmetric_psd_symmetric():
    sigma = np.array([[1, 0.5], [0.5, 1]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_symmetric_psd(sigma)
        assert len(w) == 0


def test_validate_symmetric_psd_asymmetric():
    sigma = np.array([[1, 0.5], [0.6, 1]])

    with pytest.warns(UserWarning, match="Matrix sigma not exactly symmetric"):
        validate_symmetric_psd(sigma)


def test_validate_symmetric_psd_not_psd():
    sigma = np.array([[1, 2], [2, 1]])

    with pytest.warns(UserWarning, match="Matrix sigma not numerically positive semi-definite"):
        validate_symmetric_psd(sigma)


def test_validate_conformable_valid():
    betahat = np.array([1, 2, 3, 4])
    sigma = np.eye(4)
    num_pre_periods = 2
    num_post_periods = 2
    l_vec = np.array([0.5, 0.5])

    validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec)


def test_validate_conformable_betahat_not_vector():
    betahat = np.ones((2, 2, 2))
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="Expected a vector"):
        validate_conformable(betahat, sigma, 2, 2, [0.5, 0.5])


def test_validate_conformable_sigma_not_square():
    betahat = np.array([1, 2, 3])
    sigma = np.ones((3, 4))

    with pytest.raises(ValueError, match="Expected a square matrix"):
        validate_conformable(betahat, sigma, 2, 1, [1])


def test_validate_conformable_betahat_sigma_mismatch():
    betahat = np.array([1, 2, 3])
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="betahat .* and sigma .* were non-conformable"):
        validate_conformable(betahat, sigma, 2, 1, [1])


def test_validate_conformable_periods_mismatch():
    betahat = np.array([1, 2, 3, 4])
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="betahat .* and pre \\+ post periods .* were non-conformable"):
        validate_conformable(betahat, sigma, 2, 3, [1, 1, 1])


def test_validate_conformable_l_vec_mismatch():
    betahat = np.array([1, 2, 3, 4])
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="l_vec .* and post periods .* were non-conformable"):
        validate_conformable(betahat, sigma, 2, 2, [1, 1, 1])


@pytest.fixture
def numeric_values():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def string_values():
    return np.array(["a", "b", "c", "d", "e"])


@pytest.fixture
def factor_data():
    return np.array(["A", "B", "A", "C", "B"])


@pytest.fixture
def numeric_var():
    return np.array([1.5, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def categorical_var():
    return np.array(["X", "Y", "Y", "X", "Z"])


@pytest.mark.parametrize(
    "bin_spec,expected",
    [
        ("bin::3", np.array([0, 0, 3, 3, 3, 6, 6, 6, 9, 9])),
        ("bin::2", np.array([0, 2, 2, 4, 4, 6, 6, 8, 8, 10])),
    ],
)
def test_bin_factor_consecutive_numeric(numeric_values, bin_spec, expected):
    result = bin_factor(bin_spec, numeric_values)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "bin_spec,expected",
    [
        ("!bin::2", np.array(["a", "a", "c", "c", "e"])),
    ],
)
def test_bin_factor_consecutive_forward(string_values, bin_spec, expected):
    result = bin_factor(bin_spec, string_values)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "bin_spec,expected",
    [
        ("!!bin::2", np.array(["a", "b", "b", "d", "d"])),
    ],
)
def test_bin_factor_consecutive_backward(string_values, bin_spec, expected):
    result = bin_factor(bin_spec, string_values)
    np.testing.assert_array_equal(result, expected)


def test_bin_factor_list_spec():
    values = np.array(["cat", "dog", "bird", "cat", "fish", "dog"])
    result = bin_factor(["cat", "dog"], values)
    expected = np.array(["cat", "cat", "bird", "cat", "fish", "cat"])
    np.testing.assert_array_equal(result, expected)


def test_bin_factor_list_spec_empty(numeric_values):
    result = bin_factor([], numeric_values)
    np.testing.assert_array_equal(result, numeric_values)


@pytest.mark.parametrize(
    "bin_spec,expected",
    [
        ({10: [1, 2], 20: [3, 4]}, np.array([10, 10, 20, 20, 5])),
        ({0: [2, 4], 1: [1, 3, 5]}, np.array([1, 0, 1, 0, 1])),
    ],
)
def test_bin_factor_dict_spec(bin_spec, expected):
    values = np.array([1, 2, 3, 4, 5])
    result = bin_factor(bin_spec, values)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("@2020-", np.array(["2020-01", "2020-01", "2020-01", "2021-01", "2021-02"])),
    ],
)
def test_bin_factor_regex_pattern(pattern, expected):
    values = np.array(["2020-01", "2020-02", "2020-03", "2021-01", "2021-02"])
    result = bin_factor(pattern, values)
    np.testing.assert_array_equal(result, expected)


def test_bin_factor_dict_with_strings():
    values = np.array(["cat", "dog", "bird", "fish", "lizard"])
    result = bin_factor({"mammal": ["cat", "dog"], "others": ["bird", "fish"]}, values)
    expected = np.array(["mammal", "mammal", "others", "others", "lizard"])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "bin_spec,error_match",
    [
        ("invalid::spec", "Unknown string bin specification"),
        ("@xyz", "No values matched regex pattern"),
    ],
)
def test_bin_factor_invalid_specs(string_values, bin_spec, error_match):
    with pytest.raises(ValueError, match=error_match):
        bin_factor(bin_spec, string_values)


def test_bin_factor_numeric_consecutive_float():
    values = np.array([5.5, 6.7, 8.1, 9.3, 10.5])
    result = bin_factor("bin::2", values)
    expected = np.array([4.0, 6.0, 8.0, 8.0, 10.0])
    np.testing.assert_array_equal(result, expected)


def test_create_interactions_basic(factor_data):
    result = create_interactions(factor_data)
    assert result.shape == (5, 3)
    assert np.sum(result[0, :]) == 1
    assert np.sum(result[1, :]) == 1
    assert np.sum(result, axis=0).tolist() == [2, 2, 1]


@pytest.mark.parametrize(
    "ref,expected_shape,zero_rows",
    [
        ("A", (5, 2), [0, 2]),
        (True, (5, 2), None),
    ],
)
def test_create_interactions_with_ref(factor_data, ref, expected_shape, zero_rows):
    result = create_interactions(factor_data, ref=ref)
    assert result.shape == expected_shape
    if zero_rows:
        for row_idx in zero_rows:
            assert np.all(result[row_idx, :] == 0)


@pytest.mark.parametrize(
    "keep,expected_shape,zero_rows",
    [
        (["A", "C"], (5, 2), [1, 4]),
        (["B"], (5, 1), [0, 2, 3]),
    ],
)
def test_create_interactions_with_keep(factor_data, keep, expected_shape, zero_rows):
    result = create_interactions(factor_data, keep=keep)
    assert result.shape == expected_shape
    for row_idx in zero_rows:
        assert np.sum(result[row_idx, :]) == 0


def test_create_interactions_numeric_interaction(factor_data, numeric_var):
    result = create_interactions(factor_data, var=numeric_var)
    assert result.shape == (5, 3)
    assert result[0, 0] == 1.5
    assert result[2, 0] == 3.0
    assert result[1, 1] == 2.0
    assert result[4, 1] == 5.0
    assert result[3, 2] == 4.0


def test_create_interactions_factor_interaction():
    factor = np.array(["A", "B", "A", "B"])
    var = np.array(["X", "Y", "Y", "X"])
    result = create_interactions(factor, var=var)
    assert result.shape == (4, 4)
    assert np.sum(result) == 4
    assert np.all(np.sum(result, axis=1) == 1)


def test_create_interactions_factor_with_refs(factor_data, categorical_var):
    result = create_interactions(factor_data, var=categorical_var, ref="A", ref2="X")
    assert result.shape[0] == 5
    assert np.sum(result[0, :]) == 0
    assert np.sum(result[2, :]) == 0


@pytest.mark.parametrize(
    "bin_spec,expected_shape",
    [
        ("bin::2", (6, 4)),
        ([1, 2], (6, 5)),
    ],
)
def test_create_interactions_with_binning(bin_spec, expected_shape):
    factor = np.array([1, 2, 3, 4, 5, 6])
    result = create_interactions(factor, bin=bin_spec)
    assert result.shape == expected_shape


def test_create_interactions_return_dict(factor_data):
    result = create_interactions(factor_data, return_dict=True)
    assert isinstance(result, dict)
    assert "matrix" in result
    assert "names" in result
    assert "reference_info" in result
    assert result["matrix"].shape == (5, 3)
    assert len(result["names"]) == 3
    assert result["names"] == ["A", "B", "C"]


@pytest.mark.parametrize(
    "name,var_type,expected_names",
    [
        ("cohort", None, ["cohort::A", "cohort::B"]),
        ("period", "factor", ["period::A:var::X", "period::B:var::Y"]),
    ],
)
def test_create_interactions_with_name(name, var_type, expected_names):
    factor = np.array(["A", "B"])
    if var_type == "numeric":
        var = np.array([1.0, 2.0])
    elif var_type == "factor":
        var = np.array(["X", "Y"])
    else:
        var = None

    result = create_interactions(factor, var=var, name=name, return_dict=True)
    if var_type == "factor":
        for expected in expected_names:
            assert expected in result["names"]
    else:
        assert result["names"] == expected_names


def test_create_interactions_mismatched_lengths(factor_data):
    var = np.array([1, 2])
    with pytest.raises(ValueError, match="must have the same length"):
        create_interactions(factor_data, var=var)


def test_create_interactions_empty_after_filter(factor_data):
    result = create_interactions(factor_data, ref=["A", "B", "C"])
    assert result.shape == (len(factor_data), 0)


def test_create_interactions_keep2():
    factor = np.array(["A", "B", "A", "B"])
    var = np.array(["X", "Y", "Z", "X"])
    result = create_interactions(factor, var=var, keep2=["X", "Y"])
    assert result.shape[1] == 3
    for row in result:
        if np.sum(row) > 0:
            assert np.sum(row) == 1


@pytest.mark.parametrize(
    "bin2,expected_cols",
    [
        ("bin::2", 2),
    ],
)
def test_create_interactions_bin2(factor_data, categorical_var, bin2, expected_cols):
    result = create_interactions(factor_data[:4], var=categorical_var[:4], bin2=bin2)
    assert result.shape[1] >= expected_cols
