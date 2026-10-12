"""Tests for shared formatting utilities."""

import warnings
from collections import namedtuple

import numpy as np
import pytest

from moderndid.core.format import (
    THICK_SEP,
    THIN_SEP,
    WIDTH,
    attach_format,
    compute_significance,
    format_conf_interval,
    format_event_table,
    format_footer,
    format_group_time_table,
    format_horizon_table,
    format_kv_line,
    format_p_value,
    format_section_header,
    format_significance_note,
    format_single_result_table,
    format_standard_errors,
    format_title,
    format_value,
    warn_few_clusters,
)


class TestConstants:
    def test_width(self):
        assert WIDTH == 78

    def test_thick_sep_length(self):
        assert len(THICK_SEP) == WIDTH
        assert THICK_SEP == "=" * 78

    def test_thin_sep_length(self):
        assert len(THIN_SEP) == WIDTH
        assert THIN_SEP == "-" * 78


class TestFormatTitle:
    def test_title_only(self):
        lines = format_title("My Title")
        assert lines[0] == THICK_SEP
        assert lines[1] == " My Title"
        assert lines[2] == THICK_SEP
        assert len(lines) == 3

    def test_title_with_subtitle(self):
        lines = format_title("Main Title", "Sub Title")
        assert lines[0] == THICK_SEP
        assert lines[1] == " Main Title"
        assert lines[2] == " Sub Title"
        assert lines[3] == THICK_SEP
        assert len(lines) == 4

    def test_subtitle_none(self):
        lines = format_title("Title", None)
        assert len(lines) == 3


class TestFormatSectionHeader:
    def test_basic(self):
        lines = format_section_header("Data Info")
        assert lines == ["", THIN_SEP, " Data Info", THIN_SEP]

    def test_different_label(self):
        lines = format_section_header("Inference")
        assert " Inference" in lines


class TestFormatFooter:
    def test_no_reference(self):
        lines = format_footer()
        assert lines == [THICK_SEP]

    def test_with_reference(self):
        lines = format_footer("Reference: Author (2024)")
        assert lines == [THICK_SEP, " Reference: Author (2024)"]


class TestFormatSignificanceNote:
    def test_interval_wording(self):
        lines = format_significance_note(band=False)
        assert "confidence interval does not cover 0" in lines[-1]

    def test_band_wording(self):
        lines = format_significance_note(band=True)
        assert "confidence band does not cover 0" in lines[-1]

    def test_structure(self):
        lines = format_significance_note()
        assert lines[0] == ""
        assert lines[1] == THIN_SEP
        assert lines[2].startswith(" Signif.")


class TestFormatValue:
    def test_normal_float(self):
        assert format_value(1.2345) == "1.2345"

    def test_nan(self):
        assert format_value(float("nan")) == "NA"

    def test_none(self):
        assert format_value(None) == "NA"

    def test_custom_format(self):
        assert format_value(1.5, fmt=".2f") == "1.50"

    def test_custom_na_str(self):
        assert format_value(None, na_str="---") == "---"

    def test_integer_value(self):
        assert format_value(3, fmt=".0f") == "3"

    def test_zero(self):
        assert format_value(0.0) == "0.0000"


class TestFormatConfInterval:
    def test_basic(self):
        result = format_conf_interval(1.0, 2.0)
        assert result == "[1.0000, 2.0000]"

    def test_negative_bounds(self):
        result = format_conf_interval(-1.5, -0.5)
        assert result == "[-1.5000, -0.5000]"

    def test_nan_bound(self):
        result = format_conf_interval(float("nan"), 1.0)
        assert "NA" in result

    def test_custom_format(self):
        result = format_conf_interval(1.0, 2.0, fmt=".2f")
        assert result == "[1.00, 2.00]"


class TestComputeSignificance:
    def test_significant_positive(self):
        assert compute_significance(0.5, 1.5) == "*"

    def test_significant_negative(self):
        assert compute_significance(-1.5, -0.5) == "*"

    def test_not_significant(self):
        assert compute_significance(-0.5, 0.5) == " "

    def test_boundary_zero_in_interval(self):
        assert compute_significance(-1.0, 0.0) == " "
        assert compute_significance(0.0, 1.0) == " "

    def test_nan_lower(self):
        assert compute_significance(float("nan"), 1.0) == " "

    def test_nan_upper(self):
        assert compute_significance(1.0, float("nan")) == " "

    def test_both_nan(self):
        assert compute_significance(float("nan"), float("nan")) == " "


class TestFormatPValue:
    def test_normal(self):
        assert format_p_value(0.05) == "0.0500"

    def test_very_small(self):
        assert format_p_value(0.0001) == "<0.001"

    def test_nan(self):
        assert format_p_value(float("nan")) == "NaN"

    def test_none(self):
        assert format_p_value(None) == "NaN"

    def test_zero(self):
        assert format_p_value(0.0) == "<0.001"

    def test_one(self):
        assert format_p_value(1.0) == "1.0000"


class TestFormatKvLine:
    def test_basic(self):
        assert format_kv_line("Key", "Value") == " Key: Value"

    def test_custom_indent(self):
        assert format_kv_line("Key", "Value", indent=3) == "   Key: Value"

    def test_numeric_value(self):
        assert format_kv_line("Count", 42) == " Count: 42"


class TestFormatSingleResultTable:
    def test_without_pvalue(self):
        lines = format_single_result_table("ATT", 1.5, 0.3, 95, 0.9, 2.1)
        text = "\n".join(lines)
        assert "ATT" in text
        assert "Std. Error" in text
        assert "95% Conf. Interval" in text
        assert "*" in text

    def test_with_pvalue(self):
        lines = format_single_result_table("ATT", 1.5, 0.3, 95, 0.9, 2.1, p_value=0.01)
        text = "\n".join(lines)
        assert "Pr(>|t|)" in text

    def test_not_significant(self):
        lines = format_single_result_table("ATT", 0.1, 0.5, 95, -0.9, 1.1)
        text = "\n".join(lines)
        assert "*" not in text.replace("Pr(>|t|)", "")

    def test_custom_sig_marker(self):
        lines = format_single_result_table("ATT", 1.5, 0.3, 95, 0.9, 2.1, sig_marker="**")
        text = "\n".join(lines)
        assert "**" in text


class TestFormatGroupTimeTable:
    def test_basic(self):
        groups = np.array([2000, 2001])
        times = np.array([2001, 2002])
        att = np.array([1.0, 2.0])
        se = np.array([0.3, 0.4])
        lci = np.array([0.4, 1.2])
        uci = np.array([1.6, 2.8])

        lines = format_group_time_table(groups, times, att, se, lci, uci, 95, "Pointwise")
        text = "\n".join(lines)
        assert "Group" in text
        assert "Time" in text
        assert "ATT(g,t)" in text
        assert "Pointwise" in text
        assert "2000" in text
        assert "2001" in text

    def test_nan_se_row(self):
        groups = np.array([2000])
        times = np.array([2001])
        att = np.array([1.0])
        se = np.array([float("nan")])
        lci = np.array([float("nan")])
        uci = np.array([float("nan")])

        lines = format_group_time_table(groups, times, att, se, lci, uci, 95, "Pointwise")
        text = "\n".join(lines)
        assert "NA" in text


class TestFormatEventTable:
    def test_basic(self):
        events = np.array([-1, 0, 1])
        att = np.array([0.1, 1.5, 2.0])
        se = np.array([0.2, 0.3, 0.4])
        lower = np.array([-0.3, 0.9, 1.2])
        upper = np.array([0.5, 2.1, 2.8])

        lines = format_event_table("Event time", events, att, se, lower, upper, 95, "Pointwise Conf. Band")
        text = "\n".join(lines)
        assert "Event time" in text
        assert "Estimate" in text
        assert "Std. Error" in text
        assert "Pointwise Conf. Band" in text

    def test_nan_se_row(self):
        events = np.array([0])
        att = np.array([1.0])
        se = np.array([float("nan")])
        lower = np.array([float("nan")])
        upper = np.array([float("nan")])

        lines = format_event_table("Event time", events, att, se, lower, upper, 95, "Conf. Band")
        text = "\n".join(lines)
        assert "NA" in text

    def test_significance_markers(self):
        events = np.array([1, 2])
        att = np.array([2.0, 0.1])
        se = np.array([0.3, 0.5])
        lower = np.array([1.4, -0.9])
        upper = np.array([2.6, 1.1])

        lines = format_event_table("Event time", events, att, se, lower, upper, 95, "Conf. Band")
        text = "\n".join(lines)
        assert "*" in text


class TestFormatHorizonTable:
    def test_without_counts(self):
        horizons = np.array([0, 1, 2])
        estimates = np.array([1.0, 1.5, 2.0])
        std_errors = np.array([0.2, 0.3, 0.4])
        ci_lower = np.array([0.6, 0.9, 1.2])
        ci_upper = np.array([1.4, 2.1, 2.8])

        lines = format_horizon_table(horizons, estimates, std_errors, ci_lower, ci_upper, 95)
        text = "\n".join(lines)
        assert "Horizon" in text
        assert "Estimate" in text
        assert "Switchers" not in text

    def test_with_counts(self):
        horizons = np.array([0, 1])
        estimates = np.array([1.0, 1.5])
        std_errors = np.array([0.2, 0.3])
        ci_lower = np.array([0.6, 0.9])
        ci_upper = np.array([1.4, 2.1])
        n_obs = np.array([100, 90])
        n_switchers = np.array([50, 45])

        lines = format_horizon_table(
            horizons,
            estimates,
            std_errors,
            ci_lower,
            ci_upper,
            95,
            n_obs=n_obs,
            n_switchers=n_switchers,
        )
        text = "\n".join(lines)
        assert "Switchers" in text
        assert "100" in text
        assert "50" in text


class TestAttachFormat:
    def test_attaches_repr_and_str(self):
        MyResult = namedtuple("MyResult", ["value"])
        attach_format(MyResult, lambda self: f"Result({self.value})")

        r = MyResult(value=42)
        assert repr(r) == "Result(42)"
        assert str(r) == "Result(42)"

    def test_different_format_functions(self):
        A = namedtuple("A", ["x"])
        B = namedtuple("B", ["x"])

        attach_format(A, lambda self: f"A={self.x}")
        attach_format(B, lambda self: f"B={self.x}")

        assert str(A(x=1)) == "A=1"
        assert str(B(x=2)) == "B=2"


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_format_value_prints_numpy_nan_of_any_float_type_as_na(dtype):
    assert format_value(dtype("nan")) == "NA"
    assert format_value(dtype(1.5)) == "1.5000"


def test_format_event_table_prints_every_missing_value_as_na():
    lines = format_event_table(
        "Group",
        np.array([2, 3, 4, 5]),
        np.array([1.25, np.nan, 0.0, 2.0]),
        np.array([0.3, np.nan, np.nan, 0.5]),
        np.array([0.66, np.nan, np.nan, np.nan]),
        np.array([1.84, np.nan, np.nan, np.nan]),
        95,
        "Pointwise Conf. Band",
    )
    rows = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert rows[1:] == [
        ["2", "1.2500", "0.3000", "[ 0.6600,  1.8400] *"],
        ["3", "NA", "NA", "NA"],
        ["4", "0.0000", "NA", "NA"],
        ["5", "2.0000", "0.5000", "[     NA,      NA]"],
    ]


def test_format_group_time_table_prints_every_missing_value_as_na():
    lines = format_group_time_table(
        np.array([2004, 2004, 2006]),
        np.array([2004, 2005, 2006]),
        np.array([-0.0105, np.nan, 0.03]),
        np.array([0.0233, np.nan, 0.01]),
        np.array([-0.0561, np.nan, np.nan]),
        np.array([0.0351, np.nan, np.nan]),
        95,
        "Pointwise Conf. Band",
    )
    rows = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert rows[1:] == [
        ["2004", "2004", "-0.0105", "0.0233", "[-0.0561,  0.0351]"],
        ["2004", "2005", "NA", "NA", "NA"],
        ["2006", "2006", "0.0300", "0.0100", "[     NA,      NA]"],
    ]


def test_format_horizon_table_prints_every_missing_value_as_na():
    lines = format_horizon_table(
        np.array([1.0, 2.0]),
        np.array([0.5, np.nan]),
        np.array([0.1, np.nan]),
        np.array([0.3, np.nan]),
        np.array([0.7, np.nan]),
        95,
        n_obs=np.array([100.0, 0.0]),
        n_switchers=np.array([50.0, 0.0]),
    )
    rows = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert rows[1:] == [
        ["1", "0.5000", "0.1000", "[  0.3000,   0.7000] *", "100", "50"],
        ["2", "NA", "NA", "NA", "0", "0"],
    ]


@pytest.mark.parametrize(
    ("bootstrap", "clustervar", "detail", "expected"),
    [
        (False, None, None, " Analytical standard errors"),
        (True, None, None, " Bootstrap standard errors"),
        (True, "state", None, " Bootstrap standard errors clustered by state"),
        (False, "state", None, " Analytical standard errors clustered by state"),
        (True, None, "multiplier, 999 reps", " Bootstrap standard errors (multiplier, 999 reps)"),
        (True, "cl", "multiplier, 999 reps", " Bootstrap standard errors clustered by cl (multiplier, 999 reps)"),
        (True, True, "multiplier, 99 reps", " Bootstrap standard errors clustered (multiplier, 99 reps)"),
        (False, True, None, " Analytical standard errors clustered"),
    ],
)
def test_format_standard_errors_names_the_cluster_variable(bootstrap, clustervar, detail, expected):
    assert format_standard_errors(bootstrap, clustervar, detail) == expected


@pytest.mark.parametrize(
    ("cluster", "cohort_clusters", "message"),
    [
        (
            np.repeat(np.arange(29), 2),
            {2004.0: np.array([0, 0]), 2006.0: np.array([1, 2, 3, 1]), 2007.0: np.arange(4, 13)},
            "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
            "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give confidence "
            "intervals and bands that are too narrow.",
        ),
        (
            np.repeat(np.arange(40), 2),
            {3: np.array([0, 1]), 4: np.arange(5, 15)},
            "The treated units of cohort 3 fall in 2 clusters. Cluster-robust inference for this cohort rests on very "
            "few clusters and can give confidence intervals and bands that are too narrow.",
        ),
        (
            np.repeat(np.arange(40), 2),
            {4: np.array([3, 4, 5]), 2: np.array([0]), 3.5: np.array([1, 2])},
            "The treated units of cohort 2 fall in 1 cluster, those of cohort 3.5 in 2 clusters, and those of cohort "
            "4 in 3 clusters. Cluster-robust inference for these cohorts rests on very few clusters and can give "
            "confidence intervals and bands that are too narrow.",
        ),
        (
            np.repeat(np.arange(25), 2),
            {2: np.arange(10), 3: np.arange(10, 25)},
            "The units fall in only 25 clusters. Cluster-robust inference rests on very few clusters and can give "
            "confidence intervals and bands that are too narrow.",
        ),
        (
            np.repeat(np.arange(50), 2),
            {None: np.array([0, 1, 2, 2])},
            "The treated units fall in 3 clusters. Cluster-robust inference for their effect rests on very few "
            "clusters and can give a confidence interval that is too narrow.",
        ),
        (
            np.repeat(np.arange(1), 4),
            {None: np.array([0, 0])},
            "The units fall in only 1 cluster. The treated units fall in 1 cluster. Cluster-robust inference rests on "
            "very few clusters and can give a confidence interval that is too narrow.",
        ),
    ],
    ids=["both", "one_cohort", "three_cohorts", "total", "one_group", "one_cluster"],
)
def test_warn_few_clusters_names_the_cohorts_and_their_clusters(cluster, cohort_clusters, message):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_few_clusters(cluster, cohort_clusters)

    assert [(w.category, str(w.message), w.filename) for w in caught] == [(UserWarning, message, __file__)]


@pytest.mark.parametrize(
    ("cluster", "cohort_clusters"),
    [
        (np.repeat(np.arange(30), 2), {2: np.arange(5), 3: np.arange(5, 30)}),
        (np.arange(10), {2: np.array([0]), 3: np.array([1, 2])}),
        (np.repeat(np.arange(40), 2), {2: np.array([]), 3: np.arange(10)}),
        (np.repeat(np.arange(30), 2), {None: np.arange(5)}),
    ],
    ids=["thresholds", "unit_level", "cohort_without_units", "one_group"],
)
def test_warn_few_clusters_stays_silent_with_plentiful_or_unit_level_clusters(cluster, cohort_clusters):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_few_clusters(cluster, cohort_clusters)

    assert caught == []
