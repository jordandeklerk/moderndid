"""Tests for the DDD panel estimator."""

import numpy as np
import pytest

from moderndid import ddd_panel
from moderndid.didtriple.estimators.ddd_panel import _validate_inputs


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_panel_basic(ddd_data_with_covariates, est_method):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
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


def test_ddd_panel_no_covariates(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)
    assert result.se > 0


def test_ddd_panel_with_weights(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates
    rng = np.random.default_rng(42)
    i_weights = rng.uniform(0.5, 1.5, len(ddd_data.y1))

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method="dr",
    )

    assert isinstance(result.att, float)
    assert np.isfinite(result.att)
    assert result.se > 0


def test_ddd_panel_influence_function(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
        influence_func=True,
    )

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == len(ddd_data.y1)
    assert np.abs(np.mean(result.att_inf_func)) < 0.5


@pytest.mark.parametrize("boot_type", ["multiplier", "weighted"])
def test_ddd_panel_bootstrap(ddd_data_with_covariates, boot_type):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        boot_type=boot_type,
        nboot=50,
    )

    assert result.boots is not None
    assert len(result.boots) == 50
    assert result.se > 0
    assert result.lci < result.uci


def test_ddd_panel_did_atts(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert "att_4v3" in result.did_atts
    assert "att_4v2" in result.did_atts
    assert "att_4v1" in result.did_atts
    combined = result.did_atts["att_4v3"] + result.did_atts["att_4v2"] - result.did_atts["att_4v1"]
    assert np.isclose(result.att, combined)


def test_ddd_panel_subgroup_counts(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert "subgroup_1" in result.subgroup_counts
    assert "subgroup_2" in result.subgroup_counts
    assert "subgroup_3" in result.subgroup_counts
    assert "subgroup_4" in result.subgroup_counts
    total = sum(result.subgroup_counts.values())
    assert total == len(ddd_data.y1)


def test_ddd_panel_args_stored(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
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


def test_ddd_panel_reproducibility(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    result1 = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=20,
        random_state=123,
    )

    result2 = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=20,
        random_state=123,
    )

    assert np.allclose(result1.boots, result2.boots)
    assert result1.se == result2.se


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_panel_print(ddd_data_with_covariates, est_method):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method=est_method,
    )

    output = str(result)
    assert "Triple Difference-in-Differences" in output
    assert f"{est_method.upper()}-DDD" in output
    assert "ATT" in output
    assert "Std. Error" in output
    assert "treated-and-eligible" in output
    assert "treated-but-ineligible" in output
    assert "eligible-but-untreated" in output
    assert "untreated-and-ineligible" in output


def test_ddd_panel_print_bootstrap(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
        boot=True,
        nboot=50,
    )

    output = str(result)
    assert "Bootstrap standard errors" in output
    assert "50 reps" in output


def test_validate_inputs_length_mismatch():
    y1 = np.array([1.0, 2.0, 3.0])
    y0 = np.array([0.5, 1.5])
    subgroup = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="same length"):
        _validate_inputs(y1, y0, subgroup, None, None)


def test_validate_inputs_invalid_subgroup():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 5])

    with pytest.raises(ValueError, match="only values 1, 2, 3, 4"):
        _validate_inputs(y1, y0, subgroup, None, None)


def test_validate_inputs_no_treated():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 1])

    with pytest.raises(ValueError, match="subgroup 4"):
        _validate_inputs(y1, y0, subgroup, None, None)


def test_validate_inputs_negative_weights():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    weights = np.array([1.0, -1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        _validate_inputs(y1, y0, subgroup, None, weights)


def test_validate_inputs_weights_length():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="same length"):
        _validate_inputs(y1, y0, subgroup, None, weights)


def test_validate_inputs_covariates_rows():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((3, 1))

    with pytest.raises(ValueError, match="same number of rows"):
        _validate_inputs(y1, y0, subgroup, covariates, None)


def test_validate_inputs_missing_subgroup_warns():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 4, 4])

    with pytest.warns(UserWarning, match="subgroup 3"):
        _validate_inputs(y1, y0, subgroup, None, None)


def test_validate_inputs_1d_covariates():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones(4)

    result = _validate_inputs(y1, y0, subgroup, covariates, None)
    assert result[3].ndim == 2
    assert result[3].shape == (4, 1)


def test_validate_inputs_weight_normalization():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    weights = np.array([2.0, 2.0, 2.0, 2.0])

    _, _, _, _, normalized_weights, _ = _validate_inputs(y1, y0, subgroup, None, weights)
    np.testing.assert_allclose(np.mean(normalized_weights), 1.0)
