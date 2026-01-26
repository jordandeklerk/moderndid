# pylint: disable=redefined-outer-name
"""Tests for CCK uniform confidence bands."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.didcont.npiv.cck_ucb import compute_cck_ucb
from moderndid.didcont.npiv.results import NPIVResult


def test_basic_cck_ucb(simple_data, selection_result):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        alpha=0.05,
        boot_num=50,
        selection_result=selection_result,
    )

    assert isinstance(result, NPIVResult)
    assert result.h is not None
    assert len(result.h) == len(y)
    assert result.h_lower is not None
    assert result.h_upper is not None
    assert np.all(result.h_lower <= result.h)
    assert np.all(result.h <= result.h_upper)


def test_cck_ucb_with_evaluation_points(simple_data, selection_result):
    y, x, w = simple_data
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        boot_num=50,
        selection_result=selection_result,
    )

    assert len(result.h) == len(x_eval)
    assert len(result.h_lower) == len(x_eval)
    assert len(result.h_upper) == len(x_eval)


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, selection_result, basis):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        basis=basis,
        boot_num=30,
        selection_result=selection_result,
    )

    assert result.h is not None
    assert result.args["cck_method"] is True


def test_derivative_confidence_bands(simple_data, selection_result):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_index=1,
        deriv_order=1,
        boot_num=50,
        selection_result=selection_result,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None
    assert np.all(result.h_lower_deriv <= result.deriv)
    assert np.all(result.deriv <= result.h_upper_deriv)


def test_multivariate_case(multivariate_data, selection_result):
    y, x, w = multivariate_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
        selection_result=selection_result,
    )

    assert result.h is not None
    assert len(result.h) == len(y)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_confidence_levels(simple_data, selection_result, alpha):
    y, x, w = simple_data

    np.random.seed(42)

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        alpha=alpha,
        boot_num=200,
        selection_result=selection_result,
    )

    coverage = np.mean(result.h_upper - result.h_lower)
    assert coverage > 0

    if alpha == 0.01:
        np.random.seed(42)
        result_10 = compute_cck_ucb(y=y, x=x, w=w, alpha=0.10, boot_num=200, selection_result=selection_result)
        coverage_10 = np.mean(result_10.h_upper - result_10.h_lower)
        assert coverage >= coverage_10 * 0.95


def test_no_ucb_computation(simple_data, selection_result):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=False,
        selection_result=selection_result,
    )

    assert result.h is not None
    assert result.h_lower is None
    assert result.h_upper is None
    assert result.h_lower_deriv is None
    assert result.h_upper_deriv is None


def test_with_range_constraints(simple_data, selection_result):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        boot_num=30,
        selection_result=selection_result,
    )

    assert result.h is not None


def test_reproducibility_with_seed(simple_data, selection_result):
    y, x, w = simple_data

    result1 = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
        seed=123,
        selection_result=selection_result,
    )

    result2 = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
        seed=123,
        selection_result=selection_result,
    )

    assert np.allclose(result1.h_lower, result2.h_lower)
    assert np.allclose(result1.h_upper, result2.h_upper)


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, selection_result, knots):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        knots=knots,
        boot_num=30,
        selection_result=selection_result,
    )

    assert result.h is not None


def test_small_j_segments_set(simple_data):
    small_selection_result = {
        "j_x_seg": 2,
        "k_w_seg": 3,
        "j_tilde": 3,
        "theta_star": 1.0,
        "j_x_segments_set": np.array([2, 3]),
        "k_w_segments_set": np.array([3, 4]),
    }

    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        boot_num=30,
        selection_result=small_selection_result,
    )

    assert result.h is not None
    assert result.args["n_j_boot"] == 2


def test_higher_order_derivatives(simple_data, selection_result):
    y, x, w = simple_data

    result = compute_cck_ucb(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_order=2,
        boot_num=30,
        selection_result=selection_result,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None
