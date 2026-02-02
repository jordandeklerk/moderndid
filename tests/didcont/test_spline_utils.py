"""Tests for spline utility functions."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didcont.spline.utils import (
    append_zero_columns,
    arrays_almost_equal,
    compute_quantiles,
    create_string_sequence,
    drop_first_column,
    filter_within_bounds,
    has_duplicates,
    is_close,
    linspace_interior,
    reverse_cumsum,
    to_1d,
    to_2d,
)


class TestIsClose:
    def test_basic_functionality(self):
        assert is_close(1.0, 1.0) is True
        assert is_close(1.0, 2.0) is False

    def test_with_tolerance(self):
        assert is_close(1.0, 1.0000001, rtol=1e-6) is True
        assert is_close(1.0, 1.0000001, atol=1e-6) is True
        assert is_close(1.0, 1.001, rtol=1e-6) is False

    def test_arrays(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        c = np.array([1.1, 2.1])
        assert is_close(a, b) is True
        assert is_close(a, c) is False

    def test_edge_cases(self):
        assert is_close(0.0, 0.0) is True
        assert is_close(np.inf, np.inf) is True
        assert is_close(np.nan, np.nan) is False


class TestArraysAlmostEqual:
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert arrays_almost_equal(a, b) is True

    def test_close_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0000001, 2.0000001, 3.0000001])
        assert arrays_almost_equal(a, b, rtol=1e-6) is True

    def test_different_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        assert arrays_almost_equal(a, b) is False

    def test_different_shapes(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        assert arrays_almost_equal(a, b) is False


class TestHasDuplicates:
    def test_no_duplicates(self):
        assert has_duplicates([1, 2, 3]) is False
        assert has_duplicates(np.array([1.0, 2.0, 3.0])) is False

    def test_with_duplicates(self):
        assert has_duplicates([1, 2, 2]) is True
        assert has_duplicates(np.array([1.0, 2.0, 2.0])) is True

    def test_empty_array(self):
        assert has_duplicates([]) is False
        assert has_duplicates(np.array([])) is False

    def test_single_element(self):
        assert has_duplicates([1]) is False
        assert has_duplicates(np.array([1.0])) is False


class TestReverseCumSum:
    def test_basic_functionality(self):
        x = np.array([1, 2, 3, 4])
        expected = np.array([10, 9, 7, 4])
        result = reverse_cumsum(x)
        assert np.array_equal(result, expected)

    def test_single_element(self):
        x = np.array([5])
        expected = np.array([5])
        result = reverse_cumsum(x)
        assert np.array_equal(result, expected)

    def test_empty_array(self):
        x = np.array([])
        result = reverse_cumsum(x)
        assert len(result) == 0

    def test_with_negatives(self):
        x = np.array([-1, 2, -3])
        expected = np.array([-2, -1, -3])
        result = reverse_cumsum(x)
        assert np.array_equal(result, expected)


class TestFilterWithinBounds:
    def test_basic_functionality(self):
        x = np.array([1, 2, 3, 4, 5])
        result = filter_within_bounds(x, 2, 4)
        expected = np.array([2, 3, 4])
        assert np.array_equal(result, expected)

    def test_exclusive_bounds(self):
        x = np.array([1, 2, 3, 4, 5])
        result = filter_within_bounds(x, 2, 4, include_bounds=False)
        expected = np.array([3])
        assert np.array_equal(result, expected)

    def test_no_matches(self):
        x = np.array([1, 2, 3])
        result = filter_within_bounds(x, 10, 20)
        assert len(result) == 0

    def test_all_matches(self):
        x = np.array([2, 3, 4])
        result = filter_within_bounds(x, 1, 5)
        assert np.array_equal(result, x)


class TestComputeQuantiles:
    def test_basic_functionality(self):
        x = np.array([1, 2, 3, 4, 5])
        probs = [0.0, 0.5, 1.0]
        result = compute_quantiles(x, probs, method=7)
        expected = [1.0, 3.0, 5.0]
        assert np.allclose(result, expected)

    def test_different_methods(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        probs = [0.25, 0.5, 0.75]

        result_7 = compute_quantiles(x, probs, method=7)
        result_1 = compute_quantiles(x, probs, method=1)

        assert not np.array_equal(result_7, result_1)

    def test_invalid_method_warning(self):
        x = np.array([1, 2, 3, 4, 5])
        probs = [0.5]

        with pytest.warns(UserWarning, match="Unknown quantile method"):
            result = compute_quantiles(x, probs, method=999)
            expected = compute_quantiles(x, probs, method=7)
            assert np.array_equal(result, expected)

    def test_single_value(self):
        x = np.array([5])
        probs = [0.0, 0.5, 1.0]
        result = compute_quantiles(x, probs)
        expected = [5.0, 5.0, 5.0]
        assert np.array_equal(result, expected)


class TestLinspaceInterior:
    def test_basic_functionality(self):
        result = linspace_interior(0, 10, 3)
        expected = np.array([2.5, 5.0, 7.5])
        assert np.allclose(result, expected)

    def test_zero_points(self):
        result = linspace_interior(0, 10, 0)
        assert len(result) == 0

    def test_negative_points(self):
        result = linspace_interior(0, 10, -1)
        assert len(result) == 0

    def test_single_point(self):
        result = linspace_interior(0, 10, 1)
        expected = np.array([5.0])
        assert np.allclose(result, expected)

    def test_different_bounds(self):
        result = linspace_interior(-5, 5, 2)
        expected = np.array([-5 / 3, 5 / 3])
        assert np.allclose(result, expected)


class TestTo1D:
    def test_1d_array(self):
        x = np.array([1, 2, 3])
        result = to_1d(x)
        assert np.array_equal(result, x)
        assert result.ndim == 1

    def test_2d_array(self):
        x = np.array([[1, 2], [3, 4]])
        result = to_1d(x)
        expected = np.array([1, 2, 3, 4])
        assert np.array_equal(result, expected)
        assert result.ndim == 1

    def test_scalar(self):
        x = 5
        result = to_1d(x)
        expected = np.array([5])
        assert np.array_equal(result, expected)
        assert result.ndim == 1


class TestTo2D:
    def test_1d_to_column(self):
        x = np.array([1, 2, 3])
        result = to_2d(x, axis=0)
        expected = np.array([[1], [2], [3]])
        assert np.array_equal(result, expected)
        assert result.shape == (3, 1)

    def test_1d_to_row(self):
        x = np.array([1, 2, 3])
        result = to_2d(x, axis=1)
        expected = np.array([[1, 2, 3]])
        assert np.array_equal(result, expected)
        assert result.shape == (1, 3)

    def test_2d_unchanged(self):
        x = np.array([[1, 2], [3, 4]])
        result = to_2d(x)
        assert np.array_equal(result, x)
        assert result.shape == (2, 2)

    def test_scalar(self):
        x = 5
        result = to_2d(x)
        expected = np.array([[5]])
        assert np.array_equal(result, expected)
        assert result.shape == (1, 1)


class TestDropFirstColumn:
    def test_basic_functionality(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = drop_first_column(x)
        expected = np.array([[2, 3], [5, 6]])
        assert np.array_equal(result, expected)

    def test_single_column(self):
        x = np.array([[1], [2], [3]])
        result = drop_first_column(x)
        assert result.shape == (3, 0)

    def test_empty_columns(self):
        x = np.array([]).reshape(3, 0)
        result = drop_first_column(x)
        assert result.shape == (3, 0)

    def test_invalid_input(self):
        x = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Input must be a 2-dimensional array"):
            drop_first_column(x)


class TestAppendZeroColumns:
    def test_basic_functionality(self):
        x = np.array([[1, 2], [3, 4]])
        result = append_zero_columns(x, 2)
        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0]])
        assert np.array_equal(result, expected)

    def test_1d_input(self):
        x = np.array([1, 2, 3])
        result = append_zero_columns(x, 2)
        expected = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        assert np.array_equal(result, expected)

    def test_zero_columns(self):
        x = np.array([[1, 2], [3, 4]])
        result = append_zero_columns(x, 0)
        assert np.array_equal(result, x)

    def test_negative_columns(self):
        x = np.array([[1, 2], [3, 4]])
        result = append_zero_columns(x, -1)
        assert np.array_equal(result, x)


class TestCreateStringSequence:
    def test_basic_functionality(self):
        result = create_string_sequence("var", 3)
        expected = ["var1", "var2", "var3"]
        assert result == expected

    def test_zero_count(self):
        result = create_string_sequence("test", 0)
        assert result == []

    def test_negative_count(self):
        result = create_string_sequence("test", -1)
        assert result == []

    def test_single_count(self):
        result = create_string_sequence("x", 1)
        expected = ["x1"]
        assert result == expected

    def test_different_prefix(self):
        result = create_string_sequence("beta", 2)
        expected = ["beta1", "beta2"]
        assert result == expected


class TestIntegration:
    def test_array_processing_pipeline(self):
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        x_dropped = drop_first_column(x)
        assert x_dropped.shape == (2, 3)

        x_flat = to_1d(x_dropped)
        assert len(x_flat) == 6

        assert has_duplicates(x_flat) is False

        quantiles = compute_quantiles(x_flat, [0.25, 0.75])
        assert len(quantiles) == 2

    def test_numerical_precision(self):
        x = np.array([1e-10, 1e-9, 1e-8])

        rev_sum = reverse_cumsum(x)
        expected_total = np.sum(x)
        assert is_close(rev_sum[0], expected_total, rtol=1e-15)

        quantiles = compute_quantiles(x, [0.5])
        assert quantiles[0] > 0

    def test_edge_case_robustness(self):
        empty_array = np.array([])
        single_element = np.array([42])

        assert len(reverse_cumsum(empty_array)) == 0
        assert len(filter_within_bounds(empty_array, 0, 1)) == 0
        assert has_duplicates(empty_array) is False

        assert reverse_cumsum(single_element)[0] == 42
        assert has_duplicates(single_element) is False
