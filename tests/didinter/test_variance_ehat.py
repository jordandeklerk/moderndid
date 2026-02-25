"""Tests for conditional expectation and DOF adjustments in variance estimation."""

import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.didinter.variance_ehat import (
    _get_group_cols,
    build_treatment_paths_full,
    compute_cohort_means,
    compute_dof,
    compute_e_hat,
    compute_variance_influence,
)


@pytest.fixture
def config():
    return DIDInterConfig(yname="y", tname="time", gname="id", dname="d")


@pytest.fixture
def panel_with_paths():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            "d": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "F_g": [3, 3, 3, 3, 2, 2, 2, 2, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999],
            "d_sq": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )


@pytest.mark.parametrize(
    "path_col, trends, expected",
    [
        ("path_0", None, ["path_0"]),
        ("path_0", ["region"], ["path_0", "region"]),
        ("path_1", ["region", "sector"], ["path_1", "region", "sector"]),
    ],
)
def test_get_group_cols(path_col, trends, expected):
    assert _get_group_cols(path_col, trends) == expected


@pytest.mark.parametrize(
    "horizon, expected_cols, unexpected_cols",
    [
        (0, ["path_0"], ["path_1"]),
        (1, ["path_0", "path_1", "num_g_paths_1", "cohort_fullpath_1"], []),
        (2, ["path_0", "path_1", "path_2", "num_g_paths_2"], []),
    ],
)
def test_build_treatment_paths_full_horizon_columns(panel_with_paths, config, horizon, expected_cols, unexpected_cols):
    result = build_treatment_paths_full(panel_with_paths, horizon=horizon, config=config)
    for col in expected_cols:
        assert col in result.columns
    for col in unexpected_cols:
        assert col not in result.columns


def test_build_treatment_paths_full_uses_dname_fallback(config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 1.5, 2.5],
            "d": [0, 1, 0, 0],
            "F_g": [2, 2, 9999, 9999],
        }
    )
    result = build_treatment_paths_full(df, horizon=0, config=config)
    assert "path_0" in result.columns


def test_compute_cohort_means_basic(config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "diff_y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.0],
            "path_0": [1, 1, 1, 1, 2, 2],
            "first_obs_by_gp": [1, 0, 1, 0, 1, 0],
        }
    )
    result = compute_cohort_means(df, horizon=0, diff_col="diff_y")
    assert "mean_cohort_0_s_t" in result.columns
    assert "dof_cohort_0_s_t" in result.columns


def test_compute_cohort_means_with_weights(config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "diff_y": [1.0, 2.0, 1.5, 2.5],
            "path_0": [1, 1, 1, 1],
            "first_obs_by_gp": [1, 0, 1, 0],
            "weight_gt": [1.0, 1.0, 2.0, 2.0],
        }
    )
    result = compute_cohort_means(df, horizon=0, diff_col="diff_y")
    assert "diff_y_0_N_gt" in result.columns


@pytest.mark.parametrize(
    "cluster, expected_col",
    [
        ("id", "cluster_dof_0_s"),
        (None, "dof_cohort_0_s_t"),
    ],
)
def test_compute_cohort_means_cluster_variants(cluster, expected_col):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "diff_y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.0],
            "path_0": [1, 1, 1, 1, 2, 2],
            "first_obs_by_gp": [1, 0, 1, 0, 1, 0],
        }
    )
    result = compute_cohort_means(df, horizon=0, diff_col="diff_y", cluster=cluster)
    assert expected_col in result.columns


def test_compute_cohort_means_horizon_1(config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "diff_y": [1.0, 2.0, 1.5, 2.5],
            "path_0": [1, 1, 1, 1],
            "path_1": [1, 1, 1, 1],
            "first_obs_by_gp": [1, 0, 1, 0],
        }
    )
    result = compute_cohort_means(df, horizon=1, diff_col="diff_y")
    assert "count_cohort_1_s1_t" in result.columns
    assert "total_cohort_1_s1_t" in result.columns


def test_compute_cohort_means_with_trends_nonparam(config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "diff_y": [1.0, 2.0, 1.5, 2.5],
            "path_0": [1, 1, 1, 1],
            "first_obs_by_gp": [1, 0, 1, 0],
            "region": ["A", "A", "B", "B"],
        }
    )
    result = compute_cohort_means(df, horizon=0, diff_col="diff_y", trends_nonparam=["region"])
    assert "mean_cohort_0_s_t" in result.columns


def test_compute_e_hat_basic(config):
    df = pl.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "F_g": [3, 3, 3, 9999, 9999, 9999],
            "mean_cohort_0_s_t": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "dof_cohort_0_s_t": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    result = compute_e_hat(df, horizon=0, config=config)
    assert "E_hat_0" in result.columns
    assert result["E_hat_0"].null_count() < len(result)


def test_compute_e_hat_early_pre_treatment_is_zero(config):
    df = pl.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "F_g": [4, 4, 4, 4],
            "mean_cohort_0_s_t": [0.5, 0.5, 0.5, 0.5],
            "dof_cohort_0_s_t": [5.0, 5.0, 5.0, 5.0],
        }
    )
    result = compute_e_hat(df, horizon=0, config=config)
    early_pre = result.filter(pl.col("time") < pl.col("F_g") - 1)["E_hat_0"].to_list()
    assert all(v == 0.0 for v in early_pre)


def test_compute_e_hat_null_mean_propagates(config):
    df = pl.DataFrame(
        {
            "time": [1, 2],
            "F_g": [2, 2],
            "mean_cohort_0_s_t": [None, None],
            "dof_cohort_0_s_t": [5.0, 5.0],
        }
    )
    result = compute_e_hat(df, horizon=0, config=config)
    assert result["E_hat_0"].null_count() == 2


def test_compute_dof_basic(config):
    df = pl.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "F_g": [3, 3, 3, 9999, 9999, 9999],
            "dof_cohort_0_s_t": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    result = compute_dof(df, horizon=0, config=config)
    assert "DOF_0" in result.columns


def test_compute_dof_early_pre_treatment_is_one(config):
    df = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "F_g": [4, 4, 4],
            "dof_cohort_0_s_t": [5.0, 5.0, 5.0],
        }
    )
    result = compute_dof(df, horizon=0, config=config)
    early_pre = result.filter(pl.col("time") < pl.col("F_g") - 1)["DOF_0"].to_list()
    assert all(v == 1.0 for v in early_pre)


def test_compute_dof_scaling_when_dof_greater_than_one(config):
    df = pl.DataFrame(
        {
            "time": [3],
            "F_g": [3],
            "dof_cohort_1_s_t": [10.0],
        }
    )
    result = compute_dof(df, horizon=1, config=config)
    val = result["DOF_1"][0]
    assert val is not None
    assert val > 1.0


def test_compute_dof_null_dof_propagates(config):
    df = pl.DataFrame(
        {
            "time": [2],
            "F_g": [2],
            "dof_cohort_0_s_t": [None],
        }
    )
    result = compute_dof(df, horizon=0, config=config)
    assert result["DOF_0"].null_count() == 1


def test_compute_variance_influence_basic(config):
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "diff_y": [1.0, 2.0, 1.5, 2.5],
            "dist_to_switch": [0.5, 0.5, 0.3, 0.3],
            "never_switch": [0.0, 0.0, 1.0, 1.0],
            "n_treated": [1.0, 1.0, 1.0, 1.0],
            "n_control": [1.0, 1.0, 1.0, 1.0],
            "weight_gt": [1.0, 1.0, 1.0, 1.0],
            "first_obs_by_gp": [1, 0, 1, 0],
            "E_hat_0": [0.0, 0.5, 0.0, 0.5],
            "DOF_0": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = compute_variance_influence(
        df,
        horizon=0,
        config=config,
        diff_col="diff_y",
        dist_col="dist_to_switch",
        never_col="never_switch",
        n_treated_col="n_treated",
        n_control_col="n_control",
        n_groups=2,
        n_switchers=1.0,
    )
    assert "inf_var_0" in result.columns


def test_compute_variance_influence_missing_columns_returns_unchanged(config):
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "time": [1, 2],
        }
    )
    result = compute_variance_influence(
        df,
        horizon=0,
        config=config,
        diff_col="diff_y",
        dist_col="dist_to_switch",
        never_col="never_switch",
        n_treated_col="n_treated",
        n_control_col="n_control",
        n_groups=2,
        n_switchers=1.0,
    )
    assert "inf_var_0" not in result.columns
    assert result.shape == df.shape


def test_compute_variance_influence_zero_control_handled(config):
    df = pl.DataFrame(
        {
            "id": [1, 1],
            "time": [1, 2],
            "diff_y": [1.0, 2.0],
            "dist_to_switch": [0.5, 0.5],
            "never_switch": [0.0, 0.0],
            "n_treated": [1.0, 1.0],
            "n_control": [0.0, 0.0],
            "weight_gt": [1.0, 1.0],
            "first_obs_by_gp": [1, 0],
            "E_hat_0": [0.0, 0.5],
            "DOF_0": [1.0, 1.0],
        }
    )
    result = compute_variance_influence(
        df,
        horizon=0,
        config=config,
        diff_col="diff_y",
        dist_col="dist_to_switch",
        never_col="never_switch",
        n_treated_col="n_treated",
        n_control_col="n_control",
        n_groups=1,
        n_switchers=1.0,
    )
    assert "inf_var_0" in result.columns
    assert result["inf_var_0"].null_count() == 0
