import numpy as np
import pytest

from moderndid.didtriple.format import (
    format_ddd_mp_rc_result,
    format_ddd_mp_result,
    format_ddd_panel_result,
    format_ddd_rc_result,
)


@pytest.mark.parametrize(
    "expected_text",
    [
        "Triple Difference-in-Differences",
        "ATT",
        "Std. Error",
        "treated-and-eligible",
        "treated-but-ineligible",
        "eligible-but-untreated",
        "untreated-and-ineligible",
        "Analytical standard errors",
    ],
)
def test_format_ddd_panel_contains_expected_text(ddd_panel_result, expected_text):
    output = format_ddd_panel_result(ddd_panel_result)
    assert expected_text in output


def test_format_ddd_panel_contains_estimate(ddd_panel_result):
    output = format_ddd_panel_result(ddd_panel_result)
    assert "2.5" in output or "2.50" in output


@pytest.mark.parametrize(
    "est_method,expected_label,expected_detail",
    [
        ("dr", "DR-DDD", "Propensity score: Logistic regression"),
        ("reg", "REG-DDD", "Propensity score: N/A"),
        ("ipw", "IPW-DDD", "Outcome regression: N/A"),
    ],
)
def test_format_ddd_panel_method_labels(ddd_panel_result, est_method, expected_label, expected_detail):
    result = ddd_panel_result._replace(args={**ddd_panel_result.args, "est_method": est_method})
    output = format_ddd_panel_result(result)
    assert expected_label in output
    assert expected_detail in output


def test_format_ddd_panel_bootstrap_inference(ddd_panel_result):
    result = ddd_panel_result._replace(
        args={**ddd_panel_result.args, "boot": True, "boot_type": "multiplier", "nboot": 999}
    )
    output = format_ddd_panel_result(result)
    assert "Bootstrap standard errors" in output
    assert "multiplier" in output
    assert "999" in output


def test_format_ddd_panel_significance_marker(ddd_panel_result):
    output = format_ddd_panel_result(ddd_panel_result)
    assert "*" in output


@pytest.mark.parametrize("method", [repr, str])
def test_format_ddd_panel_repr_str(ddd_panel_result, method):
    output = method(ddd_panel_result)
    assert "Triple Difference-in-Differences" in output


@pytest.mark.parametrize(
    "expected_text",
    [
        "Triple Difference-in-Differences",
        "Multi-Period",
        "ATT(g,t)",
        "Group",
        "Time",
        "Never Treated",
        "Number of units: 500",
        "Time periods: 5",
        "Treatment cohorts: 2",
    ],
)
def test_format_ddd_mp_contains_expected_text(ddd_mp_result_fixture, expected_text):
    output = format_ddd_mp_result(ddd_mp_result_fixture)
    assert expected_text in output


def test_format_ddd_mp_control_group_not_yet_treated(ddd_mp_result_fixture):
    result = ddd_mp_result_fixture._replace(args={**ddd_mp_result_fixture.args, "control_group": "notyettreated"})
    output = format_ddd_mp_result(result)
    assert "Not Yet Treated" in output


def test_format_ddd_mp_nan_standard_error(ddd_mp_result_fixture):
    se_with_nan = np.array([0.3, np.nan, 0.4, 0.45])
    result = ddd_mp_result_fixture._replace(se=se_with_nan)
    output = format_ddd_mp_result(result)
    assert "NA" in output


@pytest.mark.parametrize("method", [repr, str])
def test_format_ddd_mp_repr_str(ddd_mp_result_fixture, method):
    output = method(ddd_mp_result_fixture)
    assert "Triple Difference-in-Differences" in output


@pytest.mark.parametrize(
    "expected_text",
    [
        "Triple Difference-in-Differences",
        "Repeated Cross-Section",
        "Repeated cross-section data: 2 periods",
        "No. of observations",
        "4 cell-specific models",
    ],
)
def test_format_ddd_rc_contains_expected_text(ddd_rc_result, expected_text):
    output = format_ddd_rc_result(ddd_rc_result)
    assert expected_text in output


def test_format_ddd_rc_contains_estimate(ddd_rc_result):
    output = format_ddd_rc_result(ddd_rc_result)
    assert "2.0" in output or "2.00" in output


@pytest.mark.parametrize("method", [repr, str])
def test_format_ddd_rc_repr_str(ddd_rc_result, method):
    output = method(ddd_rc_result)
    assert "Repeated Cross-Section" in output


@pytest.mark.parametrize(
    "expected_text",
    [
        "Triple Difference-in-Differences",
        "Repeated Cross-Section",
        "Multi-Period",
        "Group",
        "Time",
        "Number of observations: 1500",
        "4 cell-specific models per comparison",
    ],
)
def test_format_ddd_mp_rc_contains_expected_text(ddd_mp_rc_result, expected_text):
    output = format_ddd_mp_rc_result(ddd_mp_rc_result)
    assert expected_text in output


@pytest.mark.parametrize("method", [repr, str])
def test_format_ddd_mp_rc_repr_str(ddd_mp_rc_result, method):
    output = method(ddd_mp_rc_result)
    assert "Repeated Cross-Section" in output
