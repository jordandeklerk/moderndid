# pylint: disable=redefined-outer-name
"""Tests for data-driven dimension selection."""

import numpy as np
import pytest

from moderndid.didcont.npiv.selection import npiv_choose_j


def test_basic_dimension_selection(simple_data):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
    )

    assert isinstance(result, dict)
    assert "j_tilde" in result
    assert "j_x_seg" in result
    assert "k_w_seg" in result
    assert "theta_star" in result
    assert "j_hat_max" in result

    assert result["j_tilde"] > 0
    assert result["j_x_seg"] > 0
    assert result["k_w_seg"] > 0
    assert result["theta_star"] > 0


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        basis=basis,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_regression_case(regression_data):
    y, x, w = regression_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_with_custom_grid(simple_data):
    y, x, w = simple_data

    x_grid = np.linspace(0, 1, 30).reshape(-1, 1)

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        x_grid=x_grid,
        grid_num=30,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_reproducibility_with_seed(simple_data):
    y, x, w = simple_data

    result1 = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
        seed=123,
    )

    result2 = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
        seed=123,
    )

    assert result1["j_tilde"] == result2["j_tilde"]
    assert result1["j_x_seg"] == result2["j_x_seg"]
    assert result1["k_w_seg"] == result2["k_w_seg"]


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        knots=knots,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_with_fullrank_check(simple_data):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        check_is_fullrank=True,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


@pytest.mark.parametrize("j_x_degree,k_w_degree", [(2, 3), (3, 4), (4, 5)])
def test_different_spline_degrees(simple_data, j_x_degree, k_w_degree):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        j_x_degree=j_x_degree,
        k_w_degree=k_w_degree,
        boot_num=20,
    )

    assert result["j_tilde"] > 0


def test_small_sample():
    n = 50
    np.random.seed(42)
    w = np.random.uniform(0, 1, (n, 1))
    x = w + 0.2 * np.random.normal(0, 1, (n, 1))
    y = np.sin(2 * np.pi * x).ravel() + 0.1 * np.random.normal(0, 1, n)

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
    )

    assert result["j_tilde"] > 0
    assert result["j_hat_max"] > 0


def test_consistency_between_components(simple_data):
    y, x, w = simple_data

    result = npiv_choose_j(
        y=y,
        x=x,
        w=w,
        boot_num=20,
    )

    assert result["j_tilde"] <= result["j_hat"]
    assert result["j_tilde"] <= result["j_hat_n"]
    assert result["j_x_seg"] in result["j_x_segments_set"]
    assert result["k_w_seg"] in result["k_w_segments_set"]
