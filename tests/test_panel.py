"""Tests for the public panel utility module."""

import pytest

from tests.helpers import importorskip

pl = importorskip("polars")
pd = importorskip("pandas")
pa = importorskip("pyarrow")

from moderndid.core.panel import (
    PanelDiagnostics,
    are_varying,
    assign_rc_ids,
    complete_data,
    deduplicate_panel,
    diagnose_panel,
    fill_panel_gaps,
    get_first_difference,
    get_group,
    has_gaps,
    is_balanced_panel,
    make_balanced_panel,
    panel_to_wide,
    scan_gaps,
    wide_to_panel,
)


def test_make_balanced_panel_drops_incomplete_units(unbalanced_panel):
    result = make_balanced_panel(unbalanced_panel, "id", "time")
    assert result["id"].n_unique() == 3
    assert 3 not in result["id"].to_list()


def test_make_balanced_panel_preserves_balanced(balanced_panel):
    result = make_balanced_panel(balanced_panel, "id", "time")
    assert len(result) == len(balanced_panel)


def test_make_balanced_panel_empty():
    empty = pl.DataFrame({"id": [], "time": [], "y": []})
    result = make_balanced_panel(empty, "id", "time")
    assert len(result) == 0


def test_is_balanced_panel_true(balanced_panel):
    assert is_balanced_panel(balanced_panel, "id", "time") is True


def test_is_balanced_panel_false(unbalanced_panel):
    assert is_balanced_panel(unbalanced_panel, "id", "time") is False


def test_get_first_difference_adds_dy(balanced_panel):
    result = get_first_difference(balanced_panel, "id", "y", "time")
    assert "dy" in result.columns


def test_get_first_difference_correct_values(balanced_panel):
    result = get_first_difference(balanced_panel, "id", "y", "time")
    unit1 = result.filter(pl.col("id") == 1).sort("time")
    assert unit1["dy"][0] is None
    assert unit1["dy"][1] == 2.0
    assert unit1["dy"][2] == 3.0


def test_get_group_adds_G(staggered_panel):
    result = get_group(staggered_panel, "id", "time", "treat")
    assert "G" in result.columns


def test_get_group_correct_groups(staggered_panel):
    result = get_group(staggered_panel, "id", "time", "treat")
    g_by_id = result.group_by("id").agg(pl.col("G").first()).sort("id")
    assert g_by_id["G"].to_list() == [3, 2, 0]


def test_get_group_all_untreated():
    df = pl.DataFrame({"id": [1, 1, 2, 2], "time": [1, 2, 1, 2], "treat": [0, 0, 0, 0]})
    result = get_group(df, "id", "time", "treat")
    assert (result["G"] == 0).all()


def test_get_group_static_indicator():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treat": [1, 1, 1, 1, 1, 1, 0, 0, 0],
        }
    )
    result = get_group(df, "id", "time", "treat", treat_period=2)
    g_by_id = result.group_by("id").agg(pl.col("G").first()).sort("id")
    assert g_by_id["G"].to_list() == [2, 2, 0]


def test_get_group_treat_period_all_untreated():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "treat": [0, 0, 0, 0],
        }
    )
    result = get_group(df, "id", "time", "treat", treat_period=5)
    assert (result["G"] == 0).all()


def test_are_varying_time_invariant():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "static": [10, 10, 20, 20],
            "dynamic": [1, 2, 3, 4],
        }
    )
    result = are_varying(df, "id")
    assert result["static"] is False
    assert result["dynamic"] is True
    assert result["time"] is True


def test_are_varying_subset_cols():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "x": [5, 5, 5, 5],
        }
    )
    result = are_varying(df, "id", cols=["x"])
    assert "x" in result
    assert "time" not in result
    assert result["x"] is False


def test_are_varying_single_obs_per_unit():
    df = pl.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    result = are_varying(df, "id")
    assert result["x"] is False


def test_scan_gaps_no_gaps(balanced_panel):
    gaps = scan_gaps(balanced_panel, "id", "time")
    assert len(gaps) == 0


def test_scan_gaps_finds_gaps(unbalanced_panel):
    gaps = scan_gaps(unbalanced_panel, "id", "time")
    assert len(gaps) == 1
    row = gaps.row(0, named=True)
    assert row["id"] == 3
    assert row["time"] == 1


def test_scan_gaps_multiple():
    df = pl.DataFrame({"id": [1, 1, 2], "time": [1, 3, 2], "y": [10, 15, 22]})
    gaps = scan_gaps(df, "id", "time")
    assert len(gaps) == 3


def test_fill_panel_gaps_fills_missing(unbalanced_panel):
    result = fill_panel_gaps(unbalanced_panel, "id", "time")
    assert len(result) == 12


def test_fill_panel_gaps_null_in_filled_rows(unbalanced_panel):
    result = fill_panel_gaps(unbalanced_panel, "id", "time")
    unit3_t1 = result.filter((pl.col("id") == 3) & (pl.col("time") == 1))
    assert len(unit3_t1) == 1
    assert unit3_t1["y"][0] is None


def test_fill_panel_gaps_balanced_unchanged(balanced_panel):
    result = fill_panel_gaps(balanced_panel, "id", "time")
    assert len(result) == len(balanced_panel)
    assert result["y"].null_count() == 0


def test_complete_data_default_all_periods(unbalanced_panel):
    result = complete_data(unbalanced_panel, "id", "time")
    assert result["id"].n_unique() == 3
    assert 3 not in result["id"].to_list()


def test_complete_data_min_periods_2(unbalanced_panel):
    result = complete_data(unbalanced_panel, "id", "time", min_periods=2)
    assert result["id"].n_unique() == 4


def test_complete_data_min_periods_1(unbalanced_panel):
    result = complete_data(unbalanced_panel, "id", "time", min_periods=1)
    assert len(result) == len(unbalanced_panel)


def test_complete_data_empty():
    empty = pl.DataFrame({"id": [], "time": [], "y": []})
    result = complete_data(empty, "id", "time")
    assert len(result) == 0


def test_deduplicate_panel_keep_last(panel_with_duplicates):
    result = deduplicate_panel(panel_with_duplicates, "id", "time", strategy="last")
    assert len(result) == 5


def test_deduplicate_panel_keep_first(panel_with_duplicates):
    result = deduplicate_panel(panel_with_duplicates, "id", "time", strategy="first")
    assert len(result) == 5


def test_deduplicate_panel_mean(panel_with_duplicates):
    result = deduplicate_panel(panel_with_duplicates, "id", "time", strategy="mean")
    assert len(result) == 5
    row = result.filter((pl.col("id") == 1) & (pl.col("time") == 1))
    assert row["y"][0] == pytest.approx(10.5)


def test_deduplicate_panel_invalid_strategy(balanced_panel):
    with pytest.raises(ValueError, match="strategy"):
        deduplicate_panel(balanced_panel, "id", "time", strategy="invalid")


def test_deduplicate_panel_no_duplicates(balanced_panel):
    result = deduplicate_panel(balanced_panel, "id", "time")
    assert len(result) == len(balanced_panel)


def test_assign_rc_ids_adds_column(balanced_panel):
    result = assign_rc_ids(balanced_panel)
    assert "rowid" in result.columns
    assert len(result) == len(balanced_panel)


def test_assign_rc_ids_unique(balanced_panel):
    result = assign_rc_ids(balanced_panel)
    assert result["rowid"].n_unique() == len(result)


def test_assign_rc_ids_sequential(balanced_panel):
    result = assign_rc_ids(balanced_panel)
    assert result["rowid"].to_list() == list(range(len(balanced_panel)))


def test_diagnose_balanced(balanced_panel):
    diag = diagnose_panel(balanced_panel, "id", "time")
    assert isinstance(diag, PanelDiagnostics)
    assert diag.n_units == 3
    assert diag.n_periods == 3
    assert diag.n_observations == 9
    assert diag.is_balanced is True
    assert diag.n_duplicate_unit_time == 0
    assert diag.n_gaps == 0
    assert diag.n_unbalanced_units == 0
    assert diag.n_missing_rows == 0
    assert diag.n_single_period_units == 0
    assert diag.n_early_treated is None
    assert diag.treatment_time_varying is None
    assert len(diag.suggestions) == 0


def test_diagnose_unbalanced(unbalanced_panel):
    diag = diagnose_panel(unbalanced_panel, "id", "time")
    assert diag.is_balanced is False
    assert diag.n_unbalanced_units == 1
    assert diag.n_gaps == 1
    assert diag.n_missing_rows == 0
    assert diag.n_single_period_units == 0
    assert any("fill_panel_gaps" in s for s in diag.suggestions)


def test_diagnose_with_duplicates(panel_with_duplicates):
    diag = diagnose_panel(panel_with_duplicates, "id", "time")
    assert diag.n_duplicate_unit_time == 2
    assert any("deduplicate_panel" in s for s in diag.suggestions)


def test_diagnose_treatment_varying(staggered_panel):
    diag = diagnose_panel(staggered_panel, "id", "time", treatname="treat")
    assert diag.treatment_time_varying is True
    assert any("Treatment varies" in s for s in diag.suggestions)


def test_diagnose_treatment_invariant():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "treat": [1, 1, 0, 0],
        }
    )
    diag = diagnose_panel(df, "id", "time", treatname="treat")
    assert diag.treatment_time_varying is False


def test_diagnose_missing_treatname(balanced_panel):
    diag = diagnose_panel(balanced_panel, "id", "time", treatname="nonexistent")
    assert diag.treatment_time_varying is None


def test_has_gaps_true(unbalanced_panel):
    assert has_gaps(unbalanced_panel, "id", "time") is True


def test_has_gaps_false(balanced_panel):
    assert has_gaps(balanced_panel, "id", "time") is False


def test_has_gaps_with_duplicates(panel_with_duplicates):
    assert has_gaps(panel_with_duplicates, "id", "time") is True


def test_panel_to_wide_basic(balanced_panel):
    wide = panel_to_wide(balanced_panel, "id", "time")
    assert len(wide) == 3
    assert "y_1" in wide.columns
    assert "y_2" in wide.columns
    assert "y_3" in wide.columns


def test_panel_to_wide_constants():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "y": [10, 20, 30, 40],
            "group": ["a", "a", "b", "b"],
        }
    )
    wide = panel_to_wide(df, "id", "time")
    assert "group" in wide.columns
    assert "group_1" not in wide.columns
    assert "y_1" in wide.columns
    assert "y_2" in wide.columns


def test_panel_to_wide_custom_separator(balanced_panel):
    wide = panel_to_wide(balanced_panel, "id", "time", separator=".")
    assert "y.1" in wide.columns


def test_panel_to_wide_no_value_cols():
    df = pl.DataFrame({"id": [1, 1, 2, 2], "time": [1, 2, 1, 2]})
    wide = panel_to_wide(df, "id", "time")
    assert len(wide) == 2
    assert wide.columns == ["id"]


def test_wide_to_panel_basic():
    wide = pl.DataFrame({"id": [1, 2], "y_1": [10, 30], "y_2": [20, 40], "y_3": [15, 35]})
    long = wide_to_panel(wide, "id", ["y"])
    assert len(long) == 6
    assert "time" in long.columns
    assert "y" in long.columns
    assert long.filter(pl.col("id") == 1).sort("time")["y"].to_list() == [10, 20, 15]


def test_wide_to_panel_multiple_stubs():
    wide = pl.DataFrame(
        {
            "id": [1, 2],
            "y_1": [10, 30],
            "y_2": [20, 40],
            "x_1": [1.0, 3.0],
            "x_2": [2.0, 4.0],
        }
    )
    long = wide_to_panel(wide, "id", ["y", "x"])
    assert len(long) == 4
    assert "y" in long.columns
    assert "x" in long.columns


def test_wide_to_panel_constants():
    wide = pl.DataFrame({"id": [1, 2], "y_1": [10, 30], "y_2": [20, 40], "group": ["a", "b"]})
    long = wide_to_panel(wide, "id", ["y"])
    assert "group" in long.columns
    assert long.filter(pl.col("id") == 1)["group"].to_list() == ["a", "a"]


def test_wide_to_panel_custom_separator():
    wide = pl.DataFrame({"id": [1, 2], "y.1": [10, 30], "y.2": [20, 40]})
    long = wide_to_panel(wide, "id", ["y"], separator=".")
    assert len(long) == 4
    assert "y" in long.columns


def test_wide_to_panel_custom_tname():
    wide = pl.DataFrame({"id": [1, 2], "y_1": [10, 30], "y_2": [20, 40]})
    long = wide_to_panel(wide, "id", ["y"], tname="wave")
    assert "wave" in long.columns


def test_panel_to_wide_roundtrip(balanced_panel):
    wide = panel_to_wide(balanced_panel, "id", "time")
    long = wide_to_panel(wide, "id", ["y", "x"], tname="time")
    long = long.sort(["id", "time"])
    orig = balanced_panel.sort(["id", "time"])
    assert long["y"].to_list() == orig["y"].to_list()


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_make_balanced_panel_roundtrip(converter, balanced_panel):
    data = converter(balanced_panel)
    result = make_balanced_panel(data, "id", "time")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_is_balanced_panel_roundtrip(converter, balanced_panel):
    data = converter(balanced_panel)
    assert is_balanced_panel(data, "id", "time") is True


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_get_first_difference_roundtrip(converter, balanced_panel):
    data = converter(balanced_panel)
    result = get_first_difference(data, "id", "y", "time")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_get_group_roundtrip(converter):
    df = pl.DataFrame({"id": [1, 1, 2, 2], "time": [1, 2, 1, 2], "treat": [0, 1, 0, 0]})
    data = converter(df)
    result = get_group(data, "id", "time", "treat")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_scan_gaps_roundtrip(converter, unbalanced_panel):
    data = converter(unbalanced_panel)
    result = scan_gaps(data, "id", "time")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_fill_panel_gaps_roundtrip(converter, unbalanced_panel):
    data = converter(unbalanced_panel)
    result = fill_panel_gaps(data, "id", "time")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_complete_data_roundtrip(converter, unbalanced_panel):
    data = converter(unbalanced_panel)
    result = complete_data(data, "id", "time", min_periods=2)
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_deduplicate_panel_roundtrip(converter, balanced_panel):
    data = converter(balanced_panel)
    result = deduplicate_panel(data, "id", "time")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_assign_rc_ids_roundtrip(converter, balanced_panel):
    data = converter(balanced_panel)
    result = assign_rc_ids(data)
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_panel_to_wide_roundtrip_types(converter, balanced_panel):
    data = converter(balanced_panel)
    result = panel_to_wide(data, "id", "time")
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])


def test_diagnostics_repr_unbalanced(unbalanced_panel):
    diag = diagnose_panel(unbalanced_panel, "id", "time", treatname="treat")
    text = str(diag)
    assert "Panel Diagnostics" in text
    assert "Units" in text
    assert "Suggestions" in text
    assert "fill_panel_gaps" in text
    assert "Treatment varies" in text


def test_diagnostics_repr_balanced(balanced_panel):
    diag = diagnose_panel(balanced_panel, "id", "time")
    text = str(diag)
    assert "Panel Diagnostics" in text
    assert "Units" in text
    assert "Suggestions" not in text


def test_diagnose_missing_values():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "y": [10.0, None, 20.0, 25.0],
        }
    )
    diag = diagnose_panel(df, "id", "time")
    assert diag.n_missing_rows == 1
    assert any("missing values" in s for s in diag.suggestions)


def test_diagnose_no_missing_values(balanced_panel):
    diag = diagnose_panel(balanced_panel, "id", "time")
    assert diag.n_missing_rows == 0


def test_diagnose_single_period_units():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "time": [1, 2, 1, 2, 1],
            "y": [10, 12, 20, 22, 30],
        }
    )
    diag = diagnose_panel(df, "id", "time")
    assert diag.n_single_period_units == 1
    assert any("observed in only one period" in s for s in diag.suggestions)


def test_diagnose_no_single_period(balanced_panel):
    diag = diagnose_panel(balanced_panel, "id", "time")
    assert diag.n_single_period_units == 0


def test_diagnose_early_treated():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "y": [10, 12, 20, 22, 30, 32],
            "treat": [1, 1, 0, 1, 0, 0],
        }
    )
    diag = diagnose_panel(df, "id", "time", treatname="treat")
    assert diag.n_early_treated == 1
    assert any("already treated in the first period" in s for s in diag.suggestions)


def test_diagnose_no_early_treated(staggered_panel):
    diag = diagnose_panel(staggered_panel, "id", "time", treatname="treat")
    assert diag.n_early_treated == 0


def test_diagnose_early_treated_no_treatname(balanced_panel):
    diag = diagnose_panel(balanced_panel, "id", "time")
    assert diag.n_early_treated is None


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_wide_to_panel_roundtrip_types(converter):
    wide = pl.DataFrame({"id": [1, 2], "y_1": [10, 30], "y_2": [20, 40]})
    data = converter(wide)
    result = wide_to_panel(data, "id", ["y"])
    assert type(result).__module__.startswith(type(data).__module__.split(".")[0])
