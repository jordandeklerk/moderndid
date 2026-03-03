"""Tests for effects_equal range, HC2/HC2-BM, more_granular_demeaning, and vcov warnings."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

import moderndid as md
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.didinter.compute_did_multiplegt import _run_het_regression, _test_effects_equality
from moderndid.didinter.format import format_didinter_result
from moderndid.didinter.results import DIDInterResult
from moderndid.didinter.variance import compute_joint_test


@pytest.mark.parametrize(
    "lb,ub,expected_df",
    [
        (None, None, 4),
        (2, 4, 2),
        (1, 2, 1),
        (1, 5, 4),
    ],
)
def test_effects_equality_range_df(effects_results_5, lb, ub, expected_df):
    config = None
    if lb is not None:
        config = DIDInterConfig(
            yname="y",
            tname="t",
            gname="id",
            dname="d",
            effects_equal_lb=lb,
            effects_equal_ub=ub,
        )
    result = _test_effects_equality(effects_results_5, config=config)
    assert result is not None
    assert result["df"] == expected_df


@pytest.mark.parametrize(
    "estimates,lb,ub,p_above",
    [
        (np.array([0.1, 0.5, 0.5, 0.5, 0.9]), 2, 4, 0.99),
        (np.array([0.5, 0.1, 0.5, 0.9, 0.5]), 2, 4, None),
    ],
)
def test_effects_equality_range_p_value(estimates, lb, ub, p_above):
    vcov = np.eye(5) * (0.01 if p_above else 0.001)
    effects_results = {"estimates": estimates, "vcov": vcov}
    config = DIDInterConfig(
        yname="y",
        tname="t",
        gname="id",
        dname="d",
        effects_equal_lb=lb,
        effects_equal_ub=ub,
    )
    result = _test_effects_equality(effects_results, config=config)
    assert result is not None
    if p_above:
        assert result["p_value"] > p_above
    else:
        assert result["p_value"] < 0.01


def test_effects_equality_warnings_key_present(effects_results_5):
    result = _test_effects_equality(effects_results_5)
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


def test_effects_equality_no_config_backward_compat(effects_results_5):
    result_none = _test_effects_equality(effects_results_5, config=None)
    result_default = _test_effects_equality(effects_results_5)
    assert result_none["df"] == result_default["df"]
    np.testing.assert_allclose(result_none["chi2_stat"], result_default["chi2_stat"])


@pytest.mark.parametrize(
    "effects_equal,match",
    [
        ("badstring", "not valid"),
        ((1, 2, 3), "exactly 2 elements"),
        ((0, 3), "lower bound must be >= 1"),
        ((3, 2), "upper bound.*must be greater"),
    ],
)
def test_effects_equal_parsing_raises(simple_panel_data, effects_equal, match):
    with pytest.raises(ValueError, match=match):
        md.did_multiplegt(
            simple_panel_data,
            yname="y",
            idname="id",
            tname="time",
            dname="d",
            effects=5,
            effects_equal=effects_equal,
        )


@pytest.mark.parametrize("effects_equal", ["1, 3", (1, 3)])
def test_effects_equal_parsing_string_and_tuple(simple_panel_data, effects_equal):
    result = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=effects_equal,
    )
    assert result.effects_equal_test is not None
    assert result.effects_equal_test["df"] == 2


def test_effects_equal_all_same_as_true(simple_panel_data):
    result_true = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
    )
    result_all = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal="all",
    )
    np.testing.assert_allclose(
        result_true.effects_equal_test["chi2_stat"],
        result_all.effects_equal_test["chi2_stat"],
    )


def test_hc2bm_requires_predict_het(simple_panel_data):
    with pytest.raises(ValueError, match="predict_het_hc2bm.*requires predict_het"):
        md.did_multiplegt(
            simple_panel_data,
            yname="y",
            idname="id",
            tname="time",
            dname="d",
            effects=2,
            predict_het_hc2bm=True,
        )


def test_het_regression_default_hc2(het_sample, hc2_config):
    result = _run_het_regression(het_sample, ["x1", "x2"], 1, hc2_config)
    assert result is not None
    assert len(result.std_errors) == 2
    assert all(se > 0 for se in result.std_errors)


def test_het_regression_hc2bm_differs_from_hc2(het_sample, hc2_config, hc2bm_config):
    result_hc2 = _run_het_regression(het_sample, ["x1", "x2"], 1, hc2_config)
    result_hc2bm = _run_het_regression(het_sample, ["x1", "x2"], 1, hc2bm_config)
    assert result_hc2 is not None
    assert result_hc2bm is not None
    assert not np.allclose(result_hc2.std_errors, result_hc2bm.std_errors)


def test_het_regression_hc2bm_warns_without_cluster(het_sample):
    sample = het_sample.rename({"cluster_id": "unit_id"})
    config = SimpleNamespace(
        trends_nonparam=None,
        predict_het_hc2bm=True,
        cluster=None,
        gname="unit_id",
    )
    with pytest.warns(UserWarning, match="predict_het_hc2bm has no effect"):
        result = _run_het_regression(sample, ["x1", "x2"], 1, config)
    assert result is not None
    assert all(se > 0 for se in result.std_errors)


def test_het_regression_hc2bm_result_fields(het_sample, hc2bm_config):
    result = _run_het_regression(het_sample, ["x1"], 1, hc2bm_config)
    assert result is not None
    assert result.horizon == 1
    assert result.covariates == ["x1"]
    assert len(result.estimates) == 1
    assert len(result.std_errors) == 1
    assert len(result.t_stats) == 1
    assert result.n_obs > 0
    assert 0 <= result.f_pvalue <= 1


def test_more_granular_demeaning_matches_less_conservative_se(simple_panel_data):
    result_granular = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        more_granular_demeaning=True,
    )
    result_less_cons = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        less_conservative_se=True,
    )
    np.testing.assert_array_equal(
        result_granular.effects.estimates,
        result_less_cons.effects.estimates,
    )
    np.testing.assert_array_equal(
        result_granular.effects.std_errors,
        result_less_cons.effects.std_errors,
    )


def test_more_granular_demeaning_differs_from_default(simple_panel_data):
    result_default = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )
    result_granular = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        more_granular_demeaning=True,
    )
    np.testing.assert_array_equal(
        result_default.effects.estimates,
        result_granular.effects.estimates,
    )
    assert not np.allclose(
        result_default.effects.std_errors,
        result_granular.effects.std_errors,
    )


@pytest.mark.parametrize(
    "vcov,expect_nan,warning_substr",
    [
        (np.array([[1.0, 1.0], [1.0, 1.0]]), True, "not invertible"),
        (np.diag([1e-8, 0.01, 0.01]), False, "close to singular"),
        (np.eye(3) * 0.01, False, None),
    ],
    ids=["singular", "near-singular", "well-conditioned"],
)
def test_joint_test_vcov_warnings(vcov, expect_nan, warning_substr):
    n = vcov.shape[0]
    estimates = np.linspace(0.1, 0.3, n)
    result = compute_joint_test(estimates, vcov)
    assert result is not None
    assert "warnings" in result

    if expect_nan:
        assert np.isnan(result["chi2_stat"])
        assert np.isnan(result["p_value"])
    else:
        assert np.isfinite(result["chi2_stat"])
        assert np.isfinite(result["p_value"])

    if warning_substr:
        assert len(result["warnings"]) >= 1
        assert any(warning_substr in w for w in result["warnings"])
    else:
        assert len(result["warnings"]) == 0


@pytest.mark.parametrize(
    "vcov_diag,expect_nan,has_warning",
    [
        (None, True, True),
        ([0.01, 0.01, 0.01], False, False),
    ],
    ids=["singular-contrast", "well-conditioned"],
)
def test_effects_equality_vcov_warnings(vcov_diag, expect_nan, has_warning):
    estimates = np.array([0.1, 0.2, 0.3]) if expect_nan else np.array([0.5, 0.5, 0.5])
    if vcov_diag is None:
        vcov = np.ones((3, 3)) * 0.01
    else:
        vcov = np.diag(vcov_diag)
    result = _test_effects_equality({"estimates": estimates, "vcov": vcov})
    assert result is not None
    if expect_nan:
        assert np.isnan(result["chi2_stat"])
    if has_warning:
        assert len(result["warnings"]) >= 1
    else:
        assert len(result["warnings"]) == 0


def test_pinv_robustness_near_singular():
    vcov = np.eye(4) * 0.01
    vcov[0, 1] = vcov[1, 0] = 0.0099999
    estimates = np.array([0.1, 0.2, 0.3, 0.4])
    result = compute_joint_test(estimates, vcov)
    assert result is not None
    assert np.isfinite(result["chi2_stat"])
    assert np.isfinite(result["p_value"])


@pytest.mark.parametrize(
    "warnings_list,expect_in_output",
    [
        ([], False),
        (["Matrix is near-singular"], True),
        (["Warning 1", "Warning 2"], True),
    ],
    ids=["empty", "single-warning", "multiple-warnings"],
)
def test_vcov_warnings_in_result_and_format(minimal_effects, warnings_list, expect_in_output):
    result = DIDInterResult(
        effects=minimal_effects,
        estimation_params={"effects": 2, "placebo": 0},
        vcov_warnings=warnings_list,
    )
    assert result.vcov_warnings == warnings_list
    formatted = format_didinter_result(result)
    if expect_in_output:
        assert "Warnings" in formatted
        for w in warnings_list:
            assert w in formatted
    else:
        assert "Warnings" not in formatted


def test_vcov_warnings_default_empty(minimal_effects):
    result = DIDInterResult(effects=minimal_effects)
    assert result.vcov_warnings == []


@pytest.mark.parametrize(
    "predict_het_hc2bm,more_granular_demeaning,effects_equal_lb,effects_equal_ub",
    [
        (False, False, None, None),
        (True, True, 2, 5),
    ],
    ids=["defaults", "all-set"],
)
def test_config_fields(predict_het_hc2bm, more_granular_demeaning, effects_equal_lb, effects_equal_ub):
    config = DIDInterConfig(
        yname="y",
        tname="t",
        gname="id",
        dname="d",
        predict_het_hc2bm=predict_het_hc2bm,
        more_granular_demeaning=more_granular_demeaning,
        effects_equal_lb=effects_equal_lb,
        effects_equal_ub=effects_equal_ub,
    )
    assert config.predict_het_hc2bm is predict_het_hc2bm
    assert config.more_granular_demeaning is more_granular_demeaning
    assert config.effects_equal_lb == effects_equal_lb
    assert config.effects_equal_ub == effects_equal_ub


def test_effects_equal_range_integration(simple_panel_data):
    result = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=(1, 3),
    )
    assert result.effects_equal_test is not None
    assert "chi2_stat" in result.effects_equal_test
    assert "p_value" in result.effects_equal_test
    assert "warnings" in result.effects_equal_test


def test_vcov_warnings_propagated_integration(simple_panel_data):
    result = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
        placebo=2,
    )
    assert isinstance(result.vcov_warnings, list)


def test_more_granular_demeaning_integration(simple_panel_data):
    result = md.did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        more_granular_demeaning=True,
    )
    assert isinstance(result, DIDInterResult)
    assert len(result.effects.estimates) == 2
