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

    assert 0.1 < result.att < 4.0
    assert 0.01 < result.se < 2.0
    assert result.lci < result.att < result.uci
    ci_width = result.uci - result.lci
    assert 1.5 * result.se < ci_width < 5.0 * result.se


def test_ddd_panel_no_covariates(ddd_data_no_covariates):
    ddd_data, covariates = ddd_data_no_covariates

    result = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        est_method="dr",
    )

    assert np.isfinite(result.att)
    assert result.se > 0
    assert result.lci < result.att < result.uci


def test_ddd_panel_with_weights(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates
    rng = np.random.default_rng(42)
    i_weights = rng.uniform(0.5, 1.5, len(ddd_data.y1))

    result_unweighted = ddd_panel(
        y1=ddd_data.y1, y0=ddd_data.y0, subgroup=ddd_data.subgroup, covariates=covariates, est_method="dr"
    )
    result_weighted = ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method="dr",
    )

    assert 0.1 < result_weighted.att < 4.0
    assert result_weighted.att != result_unweighted.att


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

    assert len(result.att_inf_func) == len(ddd_data.y1)
    inf_se = np.std(result.att_inf_func) / np.sqrt(len(ddd_data.y1))
    np.testing.assert_allclose(result.se, inf_se, rtol=0.1)


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

    assert len(result.boots) == 50
    assert result.se > 0
    ci_width = result.uci - result.lci
    assert ci_width > 2 * result.se


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


@pytest.mark.parametrize(
    "y1,y0,subgroup,covariates,weights,match",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5]), np.array([1, 2, 3]), None, None, "same length"),
        (np.ones(4), np.ones(4), np.array([1, 2, 3, 5]), None, None, "only values 1, 2, 3, 4"),
        (np.ones(4), np.ones(4), np.array([1, 2, 3, 1]), None, None, "subgroup 4"),
        (np.ones(4), np.ones(4), np.array([1, 2, 3, 4]), None, np.array([1.0, -1.0, 1.0, 1.0]), "non-negative"),
        (np.ones(4), np.ones(4), np.array([1, 2, 3, 4]), None, np.array([1.0, 1.0]), "same length"),
        (np.ones(4), np.ones(4), np.array([1, 2, 3, 4]), np.ones((3, 1)), None, "same number of rows"),
    ],
)
def test_validate_inputs_errors(y1, y0, subgroup, covariates, weights, match):
    with pytest.raises(ValueError, match=match):
        _validate_inputs(y1, y0, subgroup, covariates, weights)


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
