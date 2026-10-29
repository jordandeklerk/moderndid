"""Tests for DR-DiD result formatting."""

import math
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
    "etype, method, title",
    [
        ("dr", "imp", "Doubly Robust DiD Estimator (Improved Method)"),
        ("dr", "trad", "Doubly Robust DiD Estimator (Traditional Method)"),
        ("dr", "default", "Doubly Robust DiD Estimator"),
        ("dr", "nonexistent", "Doubly Robust DiD Estimator"),
        ("ipw", "ipw", "Inverse Probability Weighted DiD Estimator"),
        ("ipw", "std_ipw", "Standardized IPW DiD Estimator (Hajek-type)"),
        ("ipw", "default", "Inverse Probability Weighted DiD Estimator"),
        ("or", "default", "Outcome Regression DiD Estimator"),
        ("twfe", "default", "Two-Way Fixed Effects DiD Estimator"),
        ("unknown_type", "any", "Difference-in-Differences Estimator"),
    ],
)
def test_get_estimator_title(etype, method, title):
    assert _get_estimator_title(etype, method) == title


@pytest.mark.parametrize(
    "method, title",
    [
        ("imp", "Doubly Robust DiD Estimator (Improved Method)"),
        ("trad", "Doubly Robust DiD Estimator (Traditional Method)"),
        ("other", "Doubly Robust DiD Estimator"),
    ],
)
def test_get_estimator_title_names_only_the_dr_methods_drdid_records(method, title):
    assert _get_estimator_title("dr", method) == title


@pytest.mark.parametrize(
    "etype, method, details",
    [
        ("dr", "other_method", ["Outcome regression and propensity score models"]),
        ("or", "default", ["Estimation method: Ordinary least squares"]),
        ("twfe", "default", ["Fixed effects: Unit and time", "Estimation method: Within transformation"]),
    ],
)
def test_get_method_description(etype, method, details):
    assert _get_method_description(etype, method, {}) == details


@pytest.mark.parametrize(
    "method, panel, details",
    [
        ("imp", True, ["Outcome regression: Weighted least squares", "Propensity score: Inverse probability tilting"]),
        ("imp", None, ["Outcome regression: Weighted least squares", "Propensity score: Inverse probability tilting"]),
        (
            "imp",
            False,
            [
                "Outcome regression (comparison group): Weighted least squares",
                "Outcome regression (treated group): Ordinary least squares",
                "Propensity score: Inverse probability tilting",
            ],
        ),
        ("trad", True, ["Outcome regression: Ordinary least squares", "Propensity score: Logistic regression"]),
        ("trad", False, ["Outcome regression: Ordinary least squares", "Propensity score: Logistic regression"]),
    ],
)
def test_get_method_description_names_the_fits_of_each_dr_method(method, panel, details):
    assert _get_method_description("dr", method, {}, panel) == details


@pytest.mark.parametrize(
    "method, args, weight_type",
    [
        ("std_ipw", {"normalized": True}, "Weight type: Normalized (Hajek-type estimator)"),
        ("ipw", {"normalized": False}, "Weight type: Unnormalized (Horvitz-Thompson-type)"),
        ("std_ipw", {}, "Weight type: Normalized (Hajek-type estimator)"),
        ("ipw", {}, "Weight type: Unnormalized (Horvitz-Thompson-type)"),
        ("ipw", {"normalized": True}, "Weight type: Normalized (Hajek-type estimator)"),
        ("std_ipw", {"normalized": False}, "Weight type: Unnormalized (Horvitz-Thompson-type)"),
    ],
)
def test_get_method_description_weight_type_follows_the_flag_or_the_method(method, args, weight_type):
    assert _get_method_description("ipw", method, args) == [weight_type, "Propensity score: Logistic regression"]


def test_method_description_unknown_returns_empty():
    assert _get_method_description("nonexistent", "default", {}) == []


def test_format_basic_dr_panel(dr_panel_result):
    lines = format_did_result(dr_panel_result).splitlines()
    table = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert lines[1] == " Doubly Robust DiD Estimator"
    assert table == [
        ["ATT", "Std. Error", "Pr(>|t|)", "[95% Conf. Interval]"],
        ["1.5000", "0.3000", "<0.001", "[  0.9000,   2.1000] *"],
    ]
    assert " Data structure: Panel data" in lines


def test_format_reports_the_two_sided_normal_p_value(dr_panel_result):
    result = dr_panel_result._replace(att=-0.5, se=0.25, lci=-0.99, uci=-0.01)

    lines = format_did_result(result).splitlines()
    table = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert table[1] == ["-0.5000", "0.2500", f"{math.erfc(2 / math.sqrt(2)):.4f}", "[ -0.9900,  -0.0100] *"]


def test_format_explicit_args(dr_panel_result):
    args = {"type": "dr", "est_method": "trad", "panel": True, "boot": True, "biters": 500, "boot_type": "weighted"}
    result = dr_panel_result._replace(att=2.0, se=0.5, lci=1.0, uci=3.0, args=args)

    lines = format_did_result(result).splitlines()

    assert {
        " Doubly Robust DiD Estimator (Traditional Method)",
        " Data structure: Panel data",
        " Outcome regression: Ordinary least squares",
        " Propensity score: Logistic regression",
        " Standard errors: Bootstrapped (500 replications)",
        " Bootstrap type: Weighted",
    } <= set(lines)


def test_format_analytical_se(dr_panel_result):
    result = dr_panel_result._replace(args={"boot": False})

    lines = format_did_result(result).splitlines()

    assert " Standard errors: Analytical" in lines
    assert [line for line in lines if "Bootstrap" in line] == []


def test_format_rc_data(dr_rc_result):
    assert " Data structure: Repeated cross-sections" in format_did_result(dr_rc_result).splitlines()


@pytest.mark.parametrize(
    "etype, trim_level, trim_lines",
    [
        ("ipw", 0.01, [" Propensity score trimming: 0.01"]),
        ("dr", 0.05, [" Propensity score trimming: 0.05"]),
        ("or", 0.01, []),
        ("twfe", 0.01, []),
    ],
)
def test_format_trim_level(dr_panel_result, etype, trim_level, trim_lines):
    result = dr_panel_result._replace(att=1.0, se=0.3, lci=0.4, uci=1.6, args={"type": etype, "trim_level": trim_level})

    lines = format_did_result(result).splitlines()

    assert [line for line in lines if "trimming" in line.lower()] == trim_lines


@pytest.mark.parametrize(
    "att, se, has_warning",
    [
        (np.nan, 0.3, True),
        (1.0, np.nan, True),
        (1.0, 0.3, False),
    ],
)
def test_format_nan_warning(dr_panel_result, att, se, has_warning):
    result = dr_panel_result._replace(att=att, se=se, lci=np.nan, uci=np.nan, args={})
    formatted = format_did_result(result)
    assert ("Warning" in formatted) == has_warning


def test_format_nan_with_custom_warning_msg(dr_panel_result):
    args = {"warning_msg": "Custom error detail"}
    result = dr_panel_result._replace(att=np.nan, se=np.nan, lci=np.nan, uci=np.nan, args=args)

    lines = format_did_result(result).splitlines()
    start = lines.index("Warning: Estimation failed. Check for data issues:")

    assert lines[start + 1 : lines.index("=" * 78, start)] == ["     Custom error detail"]


def test_format_nan_default_suggestions(dr_panel_result):
    result = dr_panel_result._replace(att=np.nan, se=0.3, lci=np.nan, uci=np.nan, args={})

    lines = format_did_result(result).splitlines()
    start = lines.index("Warning: Estimation failed. Check for data issues:")

    assert lines[start + 1 : lines.index("=" * 78, start)] == [
        "     - Insufficient variation in treatment/control groups",
        "     - Perfect separation in propensity score",
        "     - Collinearity in covariates",
    ]


@pytest.mark.parametrize("se", [0.0, np.float64(0.0)], ids=["python_float", "numpy_float"])
def test_format_zero_se(dr_panel_result, se):
    result = dr_panel_result._replace(att=1.0, se=se, lci=0.5, uci=1.5, args={})

    lines = format_did_result(result).splitlines()
    table = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert table[1] == ["1.0000", "0.0000", "NaN", "[  0.5000,   1.5000] *"]


@pytest.mark.parametrize(
    "lci, uci, interval",
    [(0.9, 2.1, "[  0.9000,   2.1000] *"), (-2.1, -0.9, "[ -2.1000,  -0.9000] *")],
)
def test_format_significance_marker_when_zero_excluded(dr_panel_result, lci, uci, interval):
    result = dr_panel_result._replace(att=(lci + uci) / 2, lci=lci, uci=uci)

    lines = format_did_result(result).splitlines()
    table = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert table[1][-1] == interval


def test_format_no_significance_when_interval_includes_zero(dr_panel_result):
    result = dr_panel_result._replace(att=0.1, se=0.5, lci=-0.9, uci=1.1, args={})

    lines = format_did_result(result).splitlines()
    table = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert table[1][-1] == "[ -0.9000,   1.1000]"


@pytest.mark.parametrize(
    "n_covariates, header",
    [
        (3, " Computed from 500 observations and 3 covariates."),
        (1, " Computed from 500 observations and 1 covariate."),
        (0, " Computed from 500 observations and 0 covariates."),
    ],
)
def test_format_header_counts_the_rows_and_covariates_used(dr_panel_result, n_covariates, header):
    args = {"n_obs": 500, "n_covariates": n_covariates}
    result = dr_panel_result._replace(att=1.0, se=0.2, lci=0.6, uci=1.4, args=args)
    assert header in format_did_result(result).splitlines()


def test_format_header_needs_the_recorded_counts(result_with_call_params):
    assert "Computed from" not in format_did_result(result_with_call_params)


@pytest.mark.parametrize(
    "fixture_name, title",
    [
        ("twfe_result", " Two-Way Fixed Effects DiD Estimator"),
        ("unknown_result", " Difference-in-Differences Estimator"),
    ],
)
def test_format_estimator_type_in_output(fixture_name, title, request):
    result = request.getfixturevalue(fixture_name)
    assert format_did_result(result).splitlines()[1] == title


@pytest.mark.parametrize(
    "etype, last_line",
    [
        ("dr", " Reference: Sant'Anna and Zhao (2020), Journal of Econometrics"),
        ("ipw", " Reference: Abadie (2005), Review of Economic Studies"),
        ("or", " Reference: Heckman et al. (1997), Review of Economic Studies"),
        ("twfe", "=" * 78),
        ("unknown", "=" * 78),
    ],
)
def test_format_reference_by_type(dr_panel_result, etype, last_line):
    result = dr_panel_result._replace(att=1.0, se=0.2, lci=0.6, uci=1.4, args={"type": etype})
    assert format_did_result(result).splitlines()[-1] == last_line


def test_format_panel_none_omitted(unknown_result):
    result = unknown_result._replace(args={"panel": None})
    formatted = format_did_result(result)
    assert "Panel data" not in formatted
    assert "Repeated cross-sections" not in formatted


def test_format_falls_back_to_the_documented_bootstrap_defaults(dr_panel_result):
    result = dr_panel_result._replace(args={"boot": True})

    lines = format_did_result(result).splitlines()

    assert " Standard errors: Bootstrapped (999 replications)" in lines
    assert " Bootstrap type: Weighted" in lines


def test_format_boot_type_capitalized(dr_panel_result):
    args = {"boot": True, "biters": 200, "boot_type": "multiplier"}
    result = dr_panel_result._replace(att=1.0, se=0.2, lci=0.6, uci=1.4, args=args)

    lines = format_did_result(result).splitlines()

    assert " Standard errors: Bootstrapped (200 replications)" in lines
    assert " Bootstrap type: Multiplier" in lines


def test_format_estmethod_from_camel_case_key(dr_panel_result):
    result = dr_panel_result._replace(att=1.0, se=0.2, lci=0.6, uci=1.4, args={"type": "dr", "estMethod": "trad"})
    assert format_did_result(result).splitlines()[1] == " Doubly Robust DiD Estimator (Traditional Method)"


@pytest.mark.parametrize("dunder", ["__str__", "__repr__"])
def test_print_did_result_attaches_dunder(dunder):
    MyResult = namedtuple("DRDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    print_did_result(MyResult)
    r = MyResult(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})
    assert "ATT" in getattr(r, dunder)()
