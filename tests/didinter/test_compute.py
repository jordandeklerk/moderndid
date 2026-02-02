"""Tests for compute_did_multiplegt functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.didinter.compute_did_multiplegt import (
    _compute_ate,
    _compute_delta_d,
    _test_effects_equality,
    compute_same_switchers_mask,
    get_group_vars,
)


@pytest.mark.parametrize(
    "trends_nonparam,expected_len,expected_vars",
    [
        (None, 2, ["time", "d_sq"]),
        ([], 2, ["time", "d_sq"]),
        (["region"], 3, ["time", "d_sq", "region"]),
        (["region", "industry"], 4, ["time", "d_sq", "region", "industry"]),
    ],
)
def test_get_group_vars(trends_nonparam, expected_len, expected_vars):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        trends_nonparam=trends_nonparam,
    )

    group_vars = get_group_vars(config)

    assert len(group_vars) == expected_len
    for var in expected_vars:
        assert var in group_vars


@pytest.mark.parametrize(
    "horizon_type,n_horizons,has_l_g",
    [
        ("effect", 2, True),
        ("placebo", 1, False),
        ("effect", 3, True),
    ],
)
def test_compute_same_switchers_mask_basic(horizon_type, n_horizons, has_l_g):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )

    if has_l_g:
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "time": [1, 2, 1, 2, 1, 2],
                "L_g": [3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "F_g": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0, float("inf"), float("inf"), float("inf")],
            }
        )

    result = compute_same_switchers_mask(df, config, n_horizons=n_horizons, _t_max=3, horizon_type=horizon_type)

    assert "same_switcher_valid" in result.columns


def test_compute_same_switchers_mask_effect_validation():
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )

    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "L_g": [3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
        }
    )

    result = compute_same_switchers_mask(df, config, n_horizons=2, _t_max=2, horizon_type="effect")
    result_sorted = result.sort(["id", "time"])
    valid_values = result_sorted["same_switcher_valid"].to_list()

    assert valid_values[0] is True
    assert valid_values[2] is True
    assert valid_values[4] is False


def test_compute_same_switchers_mask_no_l_g():
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )

    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
        }
    )

    result = compute_same_switchers_mask(df, config, n_horizons=2, _t_max=2, horizon_type="effect")

    assert "same_switcher_valid" in result.columns
    assert result["same_switcher_valid"].all()


def test_compute_same_switchers_mask_large_n_horizons():
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )

    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [1, 2, 3, 4, 1, 2, 3, 4],
            "L_g": [4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0],
        }
    )

    result = compute_same_switchers_mask(df, config, n_horizons=3, _t_max=4, horizon_type="effect")

    assert "same_switcher_valid" in result.columns

    id1_valid = result.filter(pl.col("id") == 1)["same_switcher_valid"][0]
    id2_valid = result.filter(pl.col("id") == 2)["same_switcher_valid"][0]

    assert id1_valid is True
    assert id2_valid is False


def test_compute_delta_d_positive_for_increasing_treatment(switcher_data, basic_config):
    result = _compute_delta_d(switcher_data, basic_config, horizon=1, horizon_type="effect")

    assert result is not None
    assert result > 0


def test_compute_delta_d_returns_none_for_no_switchers(basic_config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "d": [0, 0, 0, 0],
            "d_sq": [0.0, 0.0, 0.0, 0.0],
            "F_g": [float("inf"), float("inf"), float("inf"), float("inf")],
            "S_g": [0, 0, 0, 0],
            "weight_gt": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = _compute_delta_d(df, basic_config, horizon=1, horizon_type="effect")

    assert result is None


def test_compute_delta_d_handles_decreasing_treatment(basic_config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 1],
            "time": [1, 2, 3],
            "d": [1, 1, 0],
            "d_sq": [1.0, 1.0, 1.0],
            "F_g": [3.0, 3.0, 3.0],
            "S_g": [-1, -1, -1],
            "weight_gt": [1.0, 1.0, 1.0],
            "dist_to_switch_1": [0.0, 0.0, 1.0],
        }
    )

    result = _compute_delta_d(df, basic_config, horizon=1, horizon_type="effect")

    assert result is not None


def test_compute_ate_weighted_average(effects_results_basic):
    z_crit = 1.96
    n_groups = 100

    result = _compute_ate(effects_results_basic, z_crit, n_groups)

    assert result is not None
    total_sw = 100 + 90 + 80
    expected = (100 / total_sw) * 0.5 + (90 / total_sw) * 0.6 + (80 / total_sw) * 0.7
    np.testing.assert_almost_equal(result.estimate, expected, decimal=5)


def test_compute_ate_confidence_interval(effects_results_basic):
    z_crit = 1.96
    n_groups = 100

    result = _compute_ate(effects_results_basic, z_crit, n_groups)

    assert result.ci_lower < result.estimate
    assert result.ci_upper > result.estimate


def test_compute_ate_handles_nan_estimates(effects_results_basic):
    effects_results_basic["estimates"][1] = np.nan
    effects_results_basic["estimates_unnorm"][1] = np.nan
    z_crit = 1.96
    n_groups = 100

    result = _compute_ate(effects_results_basic, z_crit, n_groups)

    assert result is not None
    assert not np.isnan(result.estimate)


def test_compute_ate_returns_none_for_all_nan():
    effects_results = {
        "estimates": np.array([np.nan, np.nan]),
        "estimates_unnorm": np.array([np.nan, np.nan]),
        "n_switchers": np.array([0.0, 0.0]),
        "n_switchers_weighted": np.array([0.0, 0.0]),
        "delta_d_arr": np.array([1.0, 1.0]),
    }
    z_crit = 1.96
    n_groups = 100

    result = _compute_ate(effects_results, z_crit, n_groups)

    assert result is None


@pytest.mark.parametrize(
    "n_switchers,expected_weight_ratio",
    [
        (np.array([100.0, 100.0]), 0.5),
        (np.array([100.0, 200.0]), 1 / 3),
        (np.array([200.0, 100.0]), 2 / 3),
    ],
)
def test_compute_ate_weighting_by_switchers(n_switchers, expected_weight_ratio):
    effects_results = {
        "estimates": np.array([1.0, 0.0]),
        "estimates_unnorm": np.array([1.0, 0.0]),
        "std_errors": np.array([0.1, 0.1]),
        "n_switchers": n_switchers,
        "n_switchers_weighted": n_switchers,
        "delta_d_arr": np.array([1.0, 1.0]),
        "n_observations": np.array([500.0, 500.0]),
        "vcov": np.diag([0.01, 0.01]),
    }
    z_crit = 1.96
    n_groups = 100

    result = _compute_ate(effects_results, z_crit, n_groups)

    np.testing.assert_almost_equal(result.estimate, expected_weight_ratio, decimal=5)


@pytest.mark.parametrize(
    "key",
    ["chi2_stat", "df", "p_value"],
)
def test_test_effects_equality_result_keys(key):
    effects_results = {
        "estimates": np.array([0.5, 0.6]),
        "vcov": np.array([[0.01, 0.002], [0.002, 0.015]]),
    }

    result = _test_effects_equality(effects_results)

    assert key in result


def test_test_effects_equality_df_is_n_minus_1():
    effects_results = {
        "estimates": np.array([0.5, 0.6, 0.55, 0.58]),
        "vcov": np.eye(4) * 0.01,
    }

    result = _test_effects_equality(effects_results)

    assert result["df"] == 3


def test_test_effects_equality_p_value_in_range():
    effects_results = {
        "estimates": np.array([0.5, 0.6, 0.55]),
        "vcov": np.eye(3) * 0.01,
    }

    result = _test_effects_equality(effects_results)

    assert 0 <= result["p_value"] <= 1


@pytest.mark.parametrize(
    "estimates,vcov,expected",
    [
        (np.array([0.5]), np.array([[0.01]]), None),
        (np.array([0.5, 0.6]), None, None),
    ],
)
def test_test_effects_equality_returns_none(estimates, vcov, expected):
    effects_results = {"estimates": estimates, "vcov": vcov}

    result = _test_effects_equality(effects_results)

    assert result == expected


def test_test_effects_equality_equal_effects_high_p_value():
    effects_results = {
        "estimates": np.array([0.5, 0.5, 0.5]),
        "vcov": np.eye(3) * 0.01,
    }

    result = _test_effects_equality(effects_results)

    assert result["p_value"] > 0.99
    assert result["chi2_stat"] < 0.01


def test_test_effects_equality_unequal_effects_low_p_value():
    effects_results = {
        "estimates": np.array([0.1, 0.5, 0.9]),
        "vcov": np.eye(3) * 0.001,
    }

    result = _test_effects_equality(effects_results)

    assert result["p_value"] < 0.01
    assert result["chi2_stat"] > 10


def test_test_effects_equality_handles_nan_estimates():
    effects_results = {
        "estimates": np.array([0.5, np.nan, 0.55]),
        "vcov": np.eye(3) * 0.01,
    }

    result = _test_effects_equality(effects_results)

    assert result is not None
    assert result["df"] == 1
