# pylint: disable=redefined-outer-name
"""Tests for nonparametric instrumental variables estimation."""

import numpy as np
import pytest

from moderndid.didcont.npiv.npiv import npiv
from moderndid.didcont.npiv.results import NPIVResult


def test_basic_npiv(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
    )

    assert isinstance(result, NPIVResult)
    assert result.h is not None
    assert len(result.h) == len(y)
    assert result.h_lower is not None
    assert result.h_upper is not None


def test_npiv_with_evaluation_points(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)

    result = npiv(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        j_x_segments=3,
        k_w_segments=4,
    )

    assert len(result.h) == len(x_eval)


def test_npiv_with_x_grid_compatibility(simple_data):
    y, x, w = simple_data
    x_grid = np.linspace(0, 1, 50).reshape(-1, 1)

    with pytest.warns(UserWarning, match="Using x_grid as x_eval"):
        result = npiv(
            y=y,
            x=x,
            w=w,
            x_grid=x_grid,
            j_x_segments=3,
            k_w_segments=4,
        )

    assert len(result.h) == len(x_grid)


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        basis=basis,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
    )

    assert result.h is not None


def test_derivative_estimation(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_index=1,
        deriv_order=1,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
    )

    assert result.h is not None


def test_no_confidence_bands(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=False,
        j_x_segments=3,
        k_w_segments=4,
    )

    assert result.h is not None
    assert result.h_lower is None
    assert result.h_upper is None


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_confidence_levels(simple_data, alpha):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        alpha=alpha,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
    )

    assert result.cv > 0


def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
    )

    assert result.h is not None


def test_reproducibility_with_seed(simple_data):
    y, x, w = simple_data

    result1 = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
        seed=123,
    )

    result2 = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
        seed=123,
    )

    assert np.allclose(result1.h_lower, result2.h_lower)
    assert np.allclose(result1.h_upper, result2.h_upper)


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        knots=knots,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=30,
    )

    assert result.h is not None


def test_input_validation():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 2))
    w = np.random.normal(0, 1, (n - 10, 2))

    with pytest.raises(ValueError, match="same number of observations"):
        npiv(y=y, x=x, w=w)


def test_invalid_alpha():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        npiv(y=y, x=x, w=w, alpha=1.5)


def test_invalid_basis():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.raises(ValueError, match="basis must be one of"):
        npiv(y=y, x=x, w=w, basis="invalid")


def test_invalid_deriv_index(multivariate_data):
    y, x, w = multivariate_data

    with pytest.raises(ValueError, match="deriv_index must be between"):
        npiv(y=y, x=x, w=w, deriv_index=3)


def test_small_sample_warning():
    n = 30
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.warns(UserWarning, match="Small sample size"):
        npiv(y=y, x=x, w=w, j_x_segments=2, k_w_segments=3)


def test_invalid_boot_num():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.raises(ValueError, match="boot_num must be positive"):
        npiv(y=y, x=x, w=w, boot_num=0)


def test_multidimensional_y():
    n = 100
    y = np.random.normal(0, 1, (n, 1))
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    result = npiv(y=y, x=x, w=w, j_x_segments=3, k_w_segments=4)

    assert result.h is not None
