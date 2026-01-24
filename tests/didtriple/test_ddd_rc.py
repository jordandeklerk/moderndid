"""Tests for the DDD repeated cross-section estimator."""

import numpy as np
import polars as pl
import pytest

from moderndid.didtriple.estimators.ddd_rc import _ddd_rc_2period, _validate_inputs_rc, ddd_rc


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


def test_ddd_rc_weighted_bootstrap(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2"]).to_numpy()])

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        boot_type="weighted",
        nboot=20,
        random_state=42,
    )

    assert len(result.boots) == 20
    assert result.se > 0
    boot_std = np.nanstd(result.boots)
    assert 0.5 * boot_std < result.se < 2.0 * boot_std


def test_validate_inputs_rc_length_mismatch():
    y = np.array([1.0, 2.0, 3.0])
    post = np.array([0, 1])
    subgroup = np.array([1, 2, 3])
    covariates = np.ones((3, 1))

    with pytest.raises(ValueError, match="same length"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_invalid_post():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 2, 0])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((4, 1))

    with pytest.raises(ValueError, match="only 0 and 1"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_invalid_subgroup():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 3, 5])
    covariates = np.ones((4, 1))

    with pytest.raises(ValueError, match="only values 1, 2, 3, 4"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_no_treated():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 3, 1])
    covariates = np.ones((4, 1))

    with pytest.raises(ValueError, match="subgroup 4"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_negative_weights():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((4, 1))
    weights = np.array([1.0, -1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        _validate_inputs_rc(y, post, subgroup, covariates, weights)


def test_validate_inputs_rc_weights_length_mismatch():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((4, 1))
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="same length as y"):
        _validate_inputs_rc(y, post, subgroup, covariates, weights)


def test_validate_inputs_rc_covariates_rows_mismatch():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((3, 1))

    with pytest.raises(ValueError, match="same number of rows"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_no_post_obs():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 0, 0, 0])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((4, 1))

    with pytest.raises(ValueError, match="No post-treatment"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_no_pre_obs():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([1, 1, 1, 1])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((4, 1))

    with pytest.raises(ValueError, match="No pre-treatment"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_validate_inputs_rc_missing_subgroup_warns():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 4, 4])
    covariates = np.ones((4, 1))

    with pytest.warns(UserWarning, match="subgroup 3"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


def test_ddd_rc_2period_wrapper(two_period_rcs_data):
    data = two_period_rcs_data

    result = _ddd_rc_2period(
        data=data,
        yname="y",
        tname="time",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2",
        weightsname=None,
        est_method="dr",
        boot=False,
        boot_type="multiplier",
        nboot=50,
        alpha=0.05,
        trim_level=0.995,
        random_state=None,
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)


def test_ddd_rc_2period_no_covariates(two_period_rcs_data):
    data = two_period_rcs_data

    result = _ddd_rc_2period(
        data=data,
        yname="y",
        tname="time",
        gname="state",
        pname="partition",
        xformla="~1",
        weightsname=None,
        est_method="dr",
        boot=False,
        boot_type="multiplier",
        nboot=50,
        alpha=0.05,
        trim_level=0.995,
        random_state=None,
    )

    assert isinstance(result.att, float)


def test_ddd_rc_2period_invalid_periods():
    data = pl.DataFrame({"y": [1, 2, 3], "time": [1, 2, 3], "state": [0, 1, 0], "partition": [1, 0, 1]})

    with pytest.raises(ValueError, match="exactly 2 time periods"):
        _ddd_rc_2period(
            data=data,
            yname="y",
            tname="time",
            gname="state",
            pname="partition",
            xformla=None,
            weightsname=None,
            est_method="dr",
            boot=False,
            boot_type="multiplier",
            nboot=50,
            alpha=0.05,
            trim_level=0.995,
            random_state=None,
        )


def test_ddd_rc_1d_covariates(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.ones(len(data))

    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert np.isfinite(result.att)
    assert 0.5 < result.att < 4.0


def test_validate_inputs_rc_weight_normalization():
    y = np.ones(8)
    post = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    subgroup = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    weights = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    _, _, _, _, normalized_weights, _ = _validate_inputs_rc(y, post, subgroup, None, weights)
    np.testing.assert_allclose(np.mean(normalized_weights), 1.0)
