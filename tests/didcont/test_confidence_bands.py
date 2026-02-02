"""Tests for uniform confidence band construction."""

import numpy as np
import pytest

from moderndid.didcont.npiv.confidence_bands import compute_ucb
from moderndid.didcont.npiv.results import NPIVResult


def test_basic_confidence_bands(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        alpha=0.05,
        boot_num=50,
    )

    assert isinstance(result, NPIVResult)
    assert result.h is not None
    assert result.h_lower is not None
    assert result.h_upper is not None
    assert np.all(result.h_lower <= result.h)
    assert np.all(result.h <= result.h_upper)
    assert result.cv > 0


def test_confidence_bands_with_evaluation_points(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        boot_num=50,
    )

    assert len(result.h) == len(x_eval)
    assert len(result.h_lower) == len(x_eval)
    assert len(result.h_upper) == len(x_eval)


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        basis=basis,
        boot_num=30,
    )

    assert result.h is not None
    assert result.h_lower is not None
    assert result.h_upper is not None


def test_derivative_confidence_bands(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_index=1,
        deriv_order=1,
        boot_num=50,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None
    assert np.all(result.h_lower_deriv <= result.deriv)
    assert np.all(result.deriv <= result.h_upper_deriv)
    assert result.cv_deriv > 0


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
    )

    assert result.h is not None
    assert len(result.h) == len(y)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_confidence_levels(simple_data, alpha):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        alpha=alpha,
        boot_num=50,
        seed=123,
    )

    coverage = np.mean(result.h_upper - result.h_lower)
    assert coverage > 0
    assert result.cv > 0


def test_no_confidence_bands(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=False,
    )

    assert result.h is not None
    assert result.h_lower is None
    assert result.h_upper is None
    assert result.h_lower_deriv is None
    assert result.h_upper_deriv is None
    assert result.cv is None
    assert result.cv_deriv is None


def test_function_bands_only(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=True,
        ucb_deriv=False,
        boot_num=50,
    )

    assert result.h_lower is not None
    assert result.h_upper is not None
    assert result.h_lower_deriv is None
    assert result.h_upper_deriv is None


def test_derivative_bands_only(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=True,
        boot_num=50,
    )

    assert result.h_lower is None
    assert result.h_upper is None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None


def test_with_fixed_dimensions(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=50,
    )

    assert result.j_x_segments == 3
    assert result.k_w_segments == 4


def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        boot_num=30,
    )

    assert result.h is not None


def test_reproducibility_with_seed(simple_data):
    y, x, w = simple_data

    result1 = compute_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
        seed=123,
    )

    result2 = compute_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
        seed=123,
    )

    assert np.allclose(result1.h_lower, result2.h_lower)
    assert np.allclose(result1.h_upper, result2.h_upper)


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        knots=knots,
        boot_num=30,
    )

    assert result.h is not None


def test_with_selection_result(simple_data):
    y, x, w = simple_data

    selection_result = {
        "j_x_seg": 3,
        "k_w_seg": 4,
        "j_tilde": 5,
        "theta_star": 1.2,
        "j_x_segments_set": np.array([2, 3, 4, 5]),
        "k_w_segments_set": np.array([3, 4, 5, 6]),
    }

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        boot_num=50,
        selection_result=selection_result,
    )

    assert result.h is not None
    assert result.args["cck_method"] is True


def test_bootstrap_statistics_tracking(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=100,
    )

    assert "boot_success_rate_h" in result.args
    assert "boot_success_rate_deriv" in result.args
    assert result.args["boot_success_rate_h"] >= 0.9


@pytest.mark.parametrize("j_x_degree,k_w_degree", [(2, 3), (3, 4), (4, 5)])
def test_different_spline_degrees(simple_data, j_x_degree, k_w_degree):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        j_x_degree=j_x_degree,
        k_w_degree=k_w_degree,
        boot_num=30,
    )

    assert result.j_x_degree == j_x_degree
    assert result.k_w_degree == k_w_degree


def test_higher_order_derivatives(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_order=2,
        boot_num=30,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None
