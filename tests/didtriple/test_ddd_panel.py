"""Tests for the DDD panel estimator."""

import numpy as np
import pytest

from moderndid import ddd_panel


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
