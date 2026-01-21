"""Tests for the DDD repeated cross-section estimator."""

import numpy as np
import pytest

from moderndid.didtriple.estimators.ddd_rc import ddd_rc


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_rc_basic(two_period_rcs_data, est_method):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method=est_method,
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_ddd_rc_no_covariates(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.ones((len(data), 1))

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)
    assert result.se > 0


def test_ddd_rc_with_weights(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])
    rng = np.random.default_rng(42)
    i_weights = rng.uniform(0.5, 1.5, len(data))

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method="dr",
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)
    assert result.se > 0


def test_ddd_rc_influence_function(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        influence_func=True,
    )

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == len(data)
    assert np.abs(np.mean(result.att_inf_func)) < 0.5


def test_ddd_rc_bootstrap(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        boot_type="multiplier",
        nboot=50,
    )

    assert result.boots is not None
    assert len(result.boots) == 50
    assert result.se > 0
    assert result.lci < result.uci


def test_ddd_rc_did_atts(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert "att_4v3" in result.did_atts
    assert "att_4v2" in result.did_atts
    assert "att_4v1" in result.did_atts
    combined = result.did_atts["att_4v3"] + result.did_atts["att_4v2"] - result.did_atts["att_4v1"]
    assert np.isclose(result.att, combined)


def test_ddd_rc_subgroup_counts(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert "subgroup_1" in result.subgroup_counts
    assert "subgroup_2" in result.subgroup_counts
    assert "subgroup_3" in result.subgroup_counts
    assert "subgroup_4" in result.subgroup_counts
    total = sum(result.subgroup_counts.values())
    assert total == len(data)


def test_ddd_rc_args_stored(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=25,
        alpha=0.10,
    )

    assert result.args["est_method"] == "dr"
    assert result.args["boot"] is True
    assert result.args["nboot"] == 25
    assert result.args["alpha"] == 0.10


def test_ddd_rc_reproducibility(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result1 = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=20,
        random_state=123,
    )

    result2 = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=20,
        random_state=123,
    )

    assert np.allclose(result1.boots, result2.boots)
    assert result1.se == result2.se


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_rc_print(two_period_rcs_data, est_method):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method=est_method,
    )

    output = str(result)
    assert "Triple Difference-in-Differences" in output
    assert f"{est_method.upper()}-DDD" in output
    assert "ATT" in output
    assert "Std. Error" in output


def test_ddd_rc_print_bootstrap(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=50,
    )

    output = str(result)
    assert "Bootstrap standard errors" in output
    assert "50 reps" in output


def test_ddd_rc_trim_level(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        trim_level=0.99,
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)


def _create_subgroup(state, partition):
    subgroup = np.zeros(len(state), dtype=int)
    subgroup[(state == 0) & (partition == 0)] = 1
    subgroup[(state == 0) & (partition == 1)] = 2
    subgroup[(state == 1) & (partition == 0)] = 3
    subgroup[(state == 1) & (partition == 1)] = 4
    return subgroup
