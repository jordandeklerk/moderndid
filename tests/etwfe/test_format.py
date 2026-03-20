"""Tests for ETWFE and EMFX result formatting."""

import pytest

from tests.helpers import importorskip

pl = importorskip("polars")
importorskip("pyfixest")

from moderndid import emfx, etwfe
from moderndid.etwfe.format import format_emfx_result, format_etwfe_result


@pytest.mark.parametrize(
    "expected",
    [
        "Extended Two-Way Fixed Effects (ETWFE)",
        "Group",
        "Time",
        "ATT(g,t)",
        "Std. Error",
        "Not Yet Treated",
        "Observations:",
        "2500",
        "Units:",
        "500",
        "Fixed Effects:",
        "countyreal + year",
        "Extended TWFE (OLS)",
        "R-squared:",
        "Significance level:",
        "hetero",
        "Wooldridge (2021, 2023)",
    ],
)
def test_format_etwfe_result_contains(etwfe_baseline, expected):
    output = format_etwfe_result(etwfe_baseline)
    assert expected in output


def test_format_etwfe_never_control(etwfe_never):
    output = format_etwfe_result(etwfe_never)
    assert "Never Treated" in output


def test_format_etwfe_poisson_family(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", family="poisson")
    output = format_etwfe_result(mod)
    assert "Extended TWFE (poisson)" in output


@pytest.mark.parametrize("method", [str, repr])
def test_format_etwfe_str_repr(etwfe_baseline, method):
    output = method(etwfe_baseline)
    assert "Extended Two-Way Fixed Effects" in output


def test_format_etwfe_str_equals_repr(etwfe_baseline):
    assert str(etwfe_baseline) == repr(etwfe_baseline)


@pytest.mark.parametrize(
    "agg_type,expected_strings",
    [
        ("simple", ["Simple Average", "Overall ATT:", "ATT", "Std. Error"]),
        ("event", ["Event Study", "Dynamic Effects:", "Event time"]),
        ("group", ["Group/Cohort", "Group Effects:"]),
        ("calendar", ["Calendar Time", "Time Effects:"]),
    ],
)
def test_format_emfx_by_type(etwfe_baseline, agg_type, expected_strings):
    result = emfx(etwfe_baseline, type=agg_type)
    output = format_emfx_result(result)
    for s in expected_strings:
        assert s in output


@pytest.mark.parametrize(
    "expected",
    [
        "Control Group:",
        "Not Yet Treated",
        "Observations:",
        "Delta method standard errors",
        "Extended TWFE (OLS)",
    ],
)
def test_format_emfx_data_info(etwfe_baseline, expected):
    result = emfx(etwfe_baseline, type="simple")
    output = format_emfx_result(result)
    assert expected in output


@pytest.mark.parametrize("method", [str, repr])
def test_format_emfx_str_repr(etwfe_baseline, method):
    result = emfx(etwfe_baseline, type="event")
    output = method(result)
    assert "Aggregate Treatment Effects" in output


def test_format_emfx_str_equals_repr(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    assert str(result) == repr(result)


def test_etwfe_maketables_coef_table(etwfe_baseline):
    table = etwfe_baseline.__maketables_coef_table__
    assert table is not None
    assert len(table) == len(etwfe_baseline.gt_pairs)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("N", 2500),
        ("n_units", 500),
        ("se_type", "hetero"),
    ],
)
def test_etwfe_maketables_stat(etwfe_baseline, key, expected):
    assert etwfe_baseline.__maketables_stat__(key) == expected


def test_etwfe_maketables_stat_R2(etwfe_baseline):
    r2 = etwfe_baseline.__maketables_stat__("R2")
    assert r2 is not None
    assert 0 < r2 <= 1


def test_etwfe_maketables_stat_unknown_returns_none(etwfe_baseline):
    assert etwfe_baseline.__maketables_stat__("nonexistent") is None


def test_etwfe_maketables_depvar(etwfe_baseline):
    assert etwfe_baseline.__maketables_depvar__ == "lemp"


def test_etwfe_maketables_fixef(etwfe_baseline):
    assert etwfe_baseline.__maketables_fixef_string__ == "countyreal + year"


def test_etwfe_maketables_vcov_info(etwfe_baseline):
    info = etwfe_baseline.__maketables_vcov_info__
    assert info["vcov_type"] == "hetero"
    assert info["clustervar"] is None


def test_emfx_maketables_coef_table(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    table = result.__maketables_coef_table__
    assert table is not None
    assert len(table) == 1 + len(result.event_times)


def test_emfx_maketables_stat_aggregation(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    assert result.__maketables_stat__("aggregation") == "event"


def test_emfx_maketables_stat_unknown_returns_none(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert result.__maketables_stat__("nonexistent") is None


def test_emfx_maketables_depvar(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert result.__maketables_depvar__ == "lemp"


def test_emfx_maketables_fixef_is_none(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert result.__maketables_fixef_string__ is None


def test_emfx_maketables_vcov_info(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    info = result.__maketables_vcov_info__
    assert info["vcov_type"] == "Delta method"


def test_etwfe_maketables_stat_labels(etwfe_baseline):
    labels = etwfe_baseline.__maketables_stat_labels__
    assert "n_units" in labels
    assert "R2" in labels


def test_etwfe_maketables_default_stat_keys(etwfe_baseline):
    keys = etwfe_baseline.__maketables_default_stat_keys__
    assert "N" in keys
    assert "n_units" in keys
    assert "se_type" in keys


def test_etwfe_maketables_default_stat_keys_includes_r2(etwfe_baseline):
    keys = etwfe_baseline.__maketables_default_stat_keys__
    assert "R2" in keys


def test_emfx_maketables_stat_labels(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    labels = result.__maketables_stat_labels__
    assert "aggregation" in labels


def test_emfx_maketables_default_stat_keys(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    keys = result.__maketables_default_stat_keys__
    assert "N" in keys
    assert "aggregation" in keys


def test_emfx_maketables_coef_table_simple(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    table = result.__maketables_coef_table__
    assert len(table) == 1


def test_format_etwfe_fe_none_no_fe_line(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", fe="none")
    output = format_etwfe_result(mod)
    assert "Fixed Effects:" not in output


def test_format_emfx_poisson_family(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", family="poisson")
    result = emfx(mod, type="event")
    output = format_emfx_result(result)
    assert "Extended TWFE (poisson)" in output


@pytest.mark.parametrize(
    "key,expected",
    [
        ("N", 2500),
        ("se_type", "hetero"),
    ],
)
def test_emfx_maketables_stat_n_and_se_type(etwfe_baseline, key, expected):
    result = emfx(etwfe_baseline, type="event")
    assert result.__maketables_stat__(key) == expected


def test_etwfe_maketables_stat_r2_adj(etwfe_baseline):
    val = etwfe_baseline.__maketables_stat__("R2_adj")
    assert val is None or isinstance(val, float)
