"""Tests for DR-DiD result formatting."""

from collections import namedtuple

import numpy as np
import pytest

from moderndid.drdid.format import (
    _get_estimator_title,
    _get_method_description,
    _infer_estimator_type,
    _infer_panel_type,
    format_did_result,
    print_did_result,
)

_DRPanel = namedtuple("DRDIDPanelResult", ["att", "se", "lci", "uci", "args"])


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DRDIDPanelResult", "dr"),
        ("drdid_rc", "dr"),
        ("IPWDIDPanelResult", "ipw"),
        ("StdIPWDIDRC", "ipw"),
        ("RegDIDPanelResult", "or"),
        ("ORDIDResult", "or"),
        ("TWFEDIDPanel", "twfe"),
        ("SomeOtherResult", "unknown"),
    ],
)
def test_infer_estimator_type(name, expected):
    assert _infer_estimator_type(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DRDIDPanelResult", True),
        ("DRDIDRcResult", False),
        ("SomeOtherResult", None),
    ],
)
def test_infer_panel_type(name, expected):
    assert _infer_panel_type(name) is expected


@pytest.mark.parametrize(
    "etype, method, expected_substr",
    [
        ("dr", "imp", "Improved"),
        ("dr", "trad", "Traditional"),
        ("dr", "imp_local", "Locally Efficient"),
        ("dr", "trad_local", "Local"),
        ("dr", "default", "Doubly Robust"),
        ("dr", "nonexistent", "Doubly Robust"),
        ("ipw", "ipw", "Inverse Probability"),
        ("ipw", "std_ipw", "Hajek"),
        ("ipw", "default", "Inverse Probability"),
        ("or", "default", "Outcome Regression"),
        ("twfe", "default", "Two-Way Fixed Effects"),
        ("unknown_type", "any", "Difference-in-Differences"),
    ],
)
def test_get_estimator_title(etype, method, expected_substr):
    assert expected_substr in _get_estimator_title(etype, method)


@pytest.mark.parametrize(
    "etype, method, args, expected_substr",
    [
        ("dr", "imp", {}, "Weighted least squares"),
        ("dr", "trad", {}, "Ordinary least squares"),
        ("dr", "imp_local", {}, "Locally weighted"),
        ("dr", "trad_local", {}, "Local linear"),
        ("dr", "other_method", {}, "propensity score"),
        ("ipw", "std_ipw", {"normalized": True}, "Normalized"),
        ("ipw", "ipw", {"normalized": False}, "Unnormalized"),
        ("or", "default", {}, "Ordinary least squares"),
        ("twfe", "default", {}, "Unit and time"),
    ],
)
def test_get_method_description(etype, method, args, expected_substr):
    details = _get_method_description(etype, method, args)
    assert any(expected_substr.lower() in d.lower() for d in details)


def test_method_description_unknown_returns_empty():
    assert _get_method_description("nonexistent", "default", {}) == []


def test_format_basic_dr_panel(dr_panel_result):
    formatted = format_did_result(dr_panel_result)
    assert isinstance(formatted, str)
    assert "ATT" in formatted
    assert "1.5" in formatted


def test_format_explicit_args():
    args = {
        "type": "dr",
        "est_method": "trad",
        "panel": True,
        "boot": True,
        "nboot": 500,
        "boot_type": "weighted",
    }
    result = _DRPanel(att=2.0, se=0.5, lci=1.0, uci=3.0, args=args)
    formatted = format_did_result(result)
    assert "Traditional" in formatted
    assert "Panel data" in formatted
    assert "Bootstrapped" in formatted
    assert "500" in formatted


def test_format_analytical_se(dr_panel_result):
    result = dr_panel_result._replace(args={"boot": False})
    formatted = format_did_result(result)
    assert "Analytical" in formatted


def test_format_rc_data(dr_rc_result):
    formatted = format_did_result(dr_rc_result)
    assert "Repeated cross-sections" in formatted


@pytest.mark.parametrize(
    "etype, trim_level, should_show_trim",
    [
        ("ipw", 0.01, True),
        ("dr", 0.05, True),
        ("or", 0.01, False),
    ],
)
def test_format_trim_level(etype, trim_level, should_show_trim):
    result = _DRPanel(att=1.0, se=0.3, lci=0.4, uci=1.6, args={"type": etype, "trim_level": trim_level})
    formatted = format_did_result(result)
    if should_show_trim:
        assert "trimming" in formatted.lower()
    else:
        assert "trimming" not in formatted.lower()


@pytest.mark.parametrize(
    "att, se, has_warning",
    [
        (np.nan, 0.3, True),
        (1.0, np.nan, True),
        (1.0, 0.3, False),
    ],
)
def test_format_nan_warning(att, se, has_warning):
    result = _DRPanel(att=att, se=se, lci=np.nan, uci=np.nan, args={})
    formatted = format_did_result(result)
    assert ("Warning" in formatted) == has_warning


def test_format_nan_with_custom_warning_msg():
    result = _DRPanel(
        att=np.nan,
        se=np.nan,
        lci=np.nan,
        uci=np.nan,
        args={"warning_msg": "Custom error detail"},
    )
    formatted = format_did_result(result)
    assert "Custom error detail" in formatted


def test_format_nan_default_suggestions():
    result = _DRPanel(att=np.nan, se=0.3, lci=np.nan, uci=np.nan, args={})
    formatted = format_did_result(result)
    for expected in ("Insufficient variation", "Perfect separation", "Collinearity"):
        assert expected in formatted


def test_format_zero_se():
    result = _DRPanel(att=1.0, se=0.0, lci=0.5, uci=1.5, args={})
    assert isinstance(format_did_result(result), str)


def test_format_significance_marker_when_zero_excluded(dr_panel_result):
    assert "*" in format_did_result(dr_panel_result)


def test_format_no_significance_when_interval_includes_zero():
    result = _DRPanel(att=0.1, se=0.5, lci=-0.9, uci=1.1, args={})
    formatted = format_did_result(result)
    lines_with_att = [line for line in formatted.split("\n") if "ATT" in line and "0.1" in line]
    for line in lines_with_att:
        assert "***" not in line


def test_format_call_params_data_shape(result_with_call_params):
    formatted = format_did_result(result_with_call_params)
    assert "500" in formatted
    assert "4 covariates" in formatted


@pytest.mark.parametrize(
    "fixture_name, expected_text",
    [
        ("twfe_result", "Two-Way Fixed Effects"),
        ("unknown_result", "Difference-in-Differences"),
    ],
)
def test_format_estimator_type_in_output(fixture_name, expected_text, request):
    result = request.getfixturevalue(fixture_name)
    assert expected_text in format_did_result(result)


@pytest.mark.parametrize(
    "etype, expected_ref",
    [
        ("dr", "Sant'Anna"),
        ("ipw", "Abadie"),
        ("or", "Heckman"),
    ],
)
def test_format_reference_by_type(etype, expected_ref):
    result = _DRPanel(att=1.0, se=0.2, lci=0.6, uci=1.4, args={"type": etype})
    assert expected_ref in format_did_result(result)


def test_format_panel_none_omitted(unknown_result):
    result = unknown_result._replace(args={"panel": None})
    formatted = format_did_result(result)
    assert "Panel data" not in formatted
    assert "Repeated cross-sections" not in formatted


def test_format_boot_type_capitalized():
    args = {"boot": True, "nboot": 200, "boot_type": "multiplier"}
    result = _DRPanel(att=1.0, se=0.2, lci=0.6, uci=1.4, args=args)
    assert "Multiplier" in format_did_result(result)


def test_format_estmethod_from_camel_case_key():
    result = _DRPanel(att=1.0, se=0.2, lci=0.6, uci=1.4, args={"type": "dr", "estMethod": "trad"})
    assert "Traditional" in format_did_result(result)


@pytest.mark.parametrize("dunder", ["__str__", "__repr__"])
def test_print_did_result_attaches_dunder(dunder):
    MyResult = namedtuple("DRDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    print_did_result(MyResult)
    r = MyResult(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})
    assert "ATT" in getattr(r, dunder)()
