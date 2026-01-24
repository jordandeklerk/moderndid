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

    assert 0.5 < result.att < 4.0
    assert 0.01 < result.se < 2.0
    assert result.lci < result.att < result.uci
    ci_width = result.uci - result.lci
    assert 1.5 * result.se < ci_width < 5.0 * result.se


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

    assert 0.5 < result.att < 4.0
    assert 0.01 < result.se < 2.0


def test_ddd_rc_with_weights(two_period_rcs_data):
    data = two_period_rcs_data
    y = data["y"].to_numpy()
    post = data["time"].to_numpy()
    subgroup = _create_subgroup(data["state"].to_numpy(), data["partition"].to_numpy())
    covariates = np.column_stack([np.ones(len(data)), data.select(["cov1", "cov2", "cov3", "cov4"]).to_numpy()])
    rng = np.random.default_rng(42)
    i_weights = rng.uniform(0.5, 1.5, len(data))

    result_unweighted = ddd_rc(y=y, post=post, subgroup=subgroup, covariates=covariates, est_method="dr")
    result_weighted = ddd_rc(
        y=y, post=post, subgroup=subgroup, covariates=covariates, i_weights=i_weights, est_method="dr"
    )

    assert 0.5 < result_weighted.att < 4.0
    assert result_weighted.att != result_unweighted.att


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

    assert len(result.att_inf_func) == len(data)
    inf_se = np.std(result.att_inf_func) / np.sqrt(len(data))
    np.testing.assert_allclose(result.se, inf_se, rtol=0.1)


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

    assert len(result.boots) == 50
    assert result.se > 0
    ci_width = result.uci - result.lci
    assert ci_width > 2 * result.se


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


@pytest.mark.parametrize(
    "y,post,subgroup,covariates,weights,match",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([0, 1]), np.array([1, 2, 3]), np.ones((3, 1)), None, "same length"),
        (np.ones(4), np.array([0, 1, 2, 0]), np.array([1, 2, 3, 4]), np.ones((4, 1)), None, "only 0 and 1"),
        (np.ones(4), np.array([0, 1, 0, 1]), np.array([1, 2, 3, 5]), np.ones((4, 1)), None, "only values 1, 2, 3, 4"),
        (np.ones(4), np.array([0, 1, 0, 1]), np.array([1, 2, 3, 1]), np.ones((4, 1)), None, "subgroup 4"),
        (
            np.ones(4),
            np.array([0, 1, 0, 1]),
            np.array([1, 2, 3, 4]),
            np.ones((4, 1)),
            np.array([1.0, -1.0, 1.0, 1.0]),
            "non-negative",
        ),
        (
            np.ones(4),
            np.array([0, 1, 0, 1]),
            np.array([1, 2, 3, 4]),
            np.ones((4, 1)),
            np.array([1.0, 1.0]),
            "same length as y",
        ),
        (np.ones(4), np.array([0, 1, 0, 1]), np.array([1, 2, 3, 4]), np.ones((3, 1)), None, "same number of rows"),
        (np.ones(4), np.array([0, 0, 0, 0]), np.array([1, 2, 3, 4]), np.ones((4, 1)), None, "No post-treatment"),
        (np.ones(4), np.array([1, 1, 1, 1]), np.array([1, 2, 3, 4]), np.ones((4, 1)), None, "No pre-treatment"),
    ],
)
def test_validate_inputs_rc_errors(y, post, subgroup, covariates, weights, match):
    with pytest.raises(ValueError, match=match):
        _validate_inputs_rc(y, post, subgroup, covariates, weights)


def test_validate_inputs_rc_missing_subgroup_warns():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([0, 1, 0, 1])
    subgroup = np.array([1, 2, 4, 4])
    covariates = np.ones((4, 1))

    with pytest.warns(UserWarning, match="subgroup 3"):
        _validate_inputs_rc(y, post, subgroup, covariates, None)


@pytest.mark.parametrize("xformla", ["~ cov1 + cov2", "~1"])
def test_ddd_rc_2period(two_period_rcs_data, xformla):
    result = _ddd_rc_2period(
        data=two_period_rcs_data,
        yname="y",
        tname="time",
        gname="state",
        pname="partition",
        xformla=xformla,
        weightsname=None,
        est_method="dr",
        boot=False,
        boot_type="multiplier",
        nboot=50,
        alpha=0.05,
        trim_level=0.995,
        random_state=None,
    )

    assert 0.5 < result.att < 4.0


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
