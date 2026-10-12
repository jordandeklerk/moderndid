"""Tests for the public panel utility module."""

import pytest

from tests.helpers import importorskip

pl = importorskip("polars")
pd = importorskip("pandas")
pa = importorskip("pyarrow")

from moderndid import att_gt
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


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas(), lambda df: df.to_arrow()],
    ids=["polars", "pandas", "pyarrow"],
)
def test_get_group_keeps_user_group_column(converter, staggered_panel):
    data = converter(staggered_panel.with_columns(pl.lit(9).alias("_group")))
    result = pl.DataFrame(get_group(data, "id", "time", "treat"))
    assert result.columns == ["id", "time", "y", "treat", "_group", "G"]
    assert result["_group"].to_list() == [9] * 12
    assert result["G"].to_list() == [3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0]


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


def test_are_varying_ignores_missing_values(static_treatment_panel):
    incomplete = static_treatment_panel.filter(~((pl.col("id") == 1) & (pl.col("time") == 2)))
    filled = fill_panel_gaps(incomplete, "id", "time")
    with_nan = static_treatment_panel.with_columns(
        pl.when((pl.col("id") == 2) & (pl.col("time") == 2)).then(float("nan")).otherwise(pl.col("x")).alias("x")
    )
    assert are_varying(filled, "id", cols=["treat", "x"]) == {"treat": False, "x": False}
    assert with_nan["x"].is_nan().sum() == 1
    assert are_varying(with_nan, "id", cols=["x"]) == {"x": False}


def test_are_varying_names_a_missing_column(balanced_panel):
    with pytest.raises(ValueError, match=r"^'z' in cols is not a column in the data\.$"):
        are_varying(balanced_panel, "id", cols=["x", "z"])


def test_are_varying_names_a_missing_unit_column(balanced_panel):
    with pytest.raises(ValueError, match=r"^idname='unit' is not a column in the data\.$"):
        are_varying(balanced_panel, "unit", cols=["x"])


@pytest.mark.parametrize("func", [diagnose_panel, fill_panel_gaps, complete_data, scan_gaps, has_gaps])
def test_panel_utilities_name_a_missing_period_column(func, balanced_panel):
    with pytest.raises(ValueError, match=r"^tname='year' is not a column in the data\.$"):
        func(balanced_panel, "id", "year")


def test_diagnose_suggests_the_closest_unit_column(balanced_panel):
    with pytest.raises(ValueError, match=r"^idname='ids' is not a column in the data\. Did you mean 'id'\?$"):
        diagnose_panel(balanced_panel, "ids", "time")


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


def test_scan_gaps_ignores_missing_period(panel_with_null_period):
    gaps = scan_gaps(panel_with_null_period, "id", "time")
    assert gaps.to_dicts() == [{"id": 2, "time": 2}]


def test_scan_gaps_ignores_nan_period(panel_with_null_period):
    data = panel_with_null_period.with_columns(pl.col("time").cast(pl.Float64).fill_null(float("nan")))
    gaps = scan_gaps(data, "id", "time")
    assert gaps.to_dicts() == [{"id": 2, "time": 2.0}]


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


def test_fill_panel_gaps_keeps_row_with_missing_period(panel_with_null_period):
    result = fill_panel_gaps(panel_with_null_period, "id", "time")
    assert len(result) == 10
    assert result.columns == ["id", "time", "y"]
    assert result.filter(pl.col("time").is_null()).to_dicts() == [{"id": 2, "time": None, "y": 22.0}]
    assert result.filter((pl.col("id") == 2) & (pl.col("time") == 2))["y"].to_list() == [None]


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


def test_complete_data_ignores_missing_period(panel_with_null_period):
    result = complete_data(panel_with_null_period, "id", "time")
    assert sorted(result["id"].unique().to_list()) == [1, 3]
    assert len(result) == 6


def test_complete_data_keeps_unit_with_repeated_period():
    df = pl.DataFrame({"id": [1, 1, 1, 2, 2], "time": [1, 2, 2, 1, 2], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert sorted(complete_data(df, "id", "time")["id"].unique().to_list()) == [1, 2]
    assert make_balanced_panel(df, "id", "time")["id"].unique().to_list() == [2]


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
    assert diag.suggestions == [
        "Call fill_panel_gaps() to fill 1 missing unit-time pair",
        "Call make_balanced_panel() to drop 1 unit not observed in all periods",
    ]


def test_diagnose_with_duplicates(panel_with_duplicates):
    diag = diagnose_panel(panel_with_duplicates, "id", "time")
    assert diag.n_duplicate_unit_time == 2
    assert diag.suggestions == [
        "Call deduplicate_panel() to remove 2 duplicate rows",
        "Call fill_panel_gaps() to fill 1 missing unit-time pair",
    ]


def test_diagnose_counts_rows_beyond_the_first_of_a_repeated_pair(panel_with_repeated_pair):
    diag = diagnose_panel(panel_with_repeated_pair, "id", "time")
    assert diag.n_duplicate_unit_time == 2
    assert len(deduplicate_panel(panel_with_repeated_pair, "id", "time")) == 3
    assert diag.suggestions == [
        "Call deduplicate_panel() to remove 2 duplicate rows",
        "Call fill_panel_gaps() to fill 1 missing unit-time pair",
        "Call complete_data(min_periods=2) to drop 1 unit observed in only one period",
    ]
    assert any(line.split() == ["│", "Duplicate", "rows", "│", "2", "│"] for line in str(diag).splitlines())


def test_diagnose_treatment_varying(staggered_panel):
    diag = diagnose_panel(staggered_panel, "id", "time", treatname="treat")
    assert diag.treatment_time_varying is True
    assert diag.suggestions == [
        "Treatment varies within units. If your estimator takes gname, call get_group() to build it"
    ]


def test_diagnose_static_treatment_after_fill_panel_gaps(static_treatment_panel):
    incomplete = static_treatment_panel.filter(~((pl.col("id") == 1) & (pl.col("time") == 2)))
    filled = fill_panel_gaps(incomplete, "id", "time")
    diag = diagnose_panel(filled, "id", "time", treatname="treat")
    assert filled["treat"].null_count() == 1
    assert diag.treatment_time_varying is False
    assert diag.suggestions == ["Missing values appear in 1 row. Estimators check only the columns they use"]


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


def test_has_gaps_ignores_row_with_missing_period(balanced_panel):
    extra = pl.DataFrame({"id": [1], "time": [None], "y": [11], "x": [1.05]}, schema=balanced_panel.schema)
    data = pl.concat([balanced_panel, extra])
    assert has_gaps(data, "id", "time") is False


def test_diagnose_ignores_missing_period_in_panel_structure(panel_with_null_period):
    diag = diagnose_panel(panel_with_null_period, "id", "time")
    assert diag.n_units == 3
    assert diag.n_periods == 3
    assert diag.n_gaps == 1
    assert diag.n_unbalanced_units == 1
    assert diag.n_missing_rows == 1
    assert diag.is_balanced is False
    assert diag.suggestions == [
        "Call fill_panel_gaps() to fill 1 missing unit-time pair",
        "Call make_balanced_panel() to drop 1 unit not observed in all periods",
        "Missing values appear in 1 row. Estimators check only the columns they use",
    ]


def test_diagnose_counts_infinite_period_as_missing(panel_with_null_period):
    data = panel_with_null_period.with_columns(pl.col("time").cast(pl.Float64).fill_null(float("inf")))
    diag = diagnose_panel(data, "id", "time")
    assert diag.n_periods == 3
    assert diag.n_gaps == 1
    assert diag.n_missing_rows == 1


def test_balancing_utilities_follow_diagnose_panel_on_missing_period(panel_with_null_period):
    diag = diagnose_panel(panel_with_null_period, "id", "time")
    balanced = make_balanced_panel(panel_with_null_period, "id", "time")
    assert diag.n_unbalanced_units == 1
    assert is_balanced_panel(panel_with_null_period, "id", "time") is False
    assert balanced.equals(panel_with_null_period.filter(pl.col("id") != 2))


def test_balancing_utilities_follow_diagnose_panel_on_row_without_period(balanced_panel):
    extra = pl.DataFrame({"id": [1], "time": [None], "y": [11], "x": [1.05]}, schema=balanced_panel.schema)
    data = pl.concat([balanced_panel, extra])
    assert diagnose_panel(data, "id", "time").is_balanced is True
    assert is_balanced_panel(data, "id", "time") is True
    assert make_balanced_panel(data, "id", "time").equals(balanced_panel)


@pytest.mark.filterwarnings("error::UserWarning")
def test_diagnose_without_a_usable_period_counts_no_early_treated_units():
    df = pl.DataFrame({"id": [1, 2], "time": pl.Series([None, None], dtype=pl.Int64), "treat": [0, 1]})
    diag = diagnose_panel(df, "id", "time", treatname="treat")
    assert diag.n_early_treated == 0
    assert diag.treatment_time_varying is False
    assert diag.suggestions == ["Missing values appear in 2 rows. Estimators check only the columns they use"]


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
    assert diag.suggestions == ["Missing values appear in 1 row. Estimators check only the columns they use"]


@pytest.mark.parametrize(
    "converter",
    [lambda df: df, lambda df: df.to_pandas()],
    ids=["polars", "pandas"],
)
def test_diagnose_counts_nan_as_missing(converter, mpdta_with_nan):
    diag = diagnose_panel(converter(mpdta_with_nan), "countyreal", "year")
    assert diag.n_missing_rows == 2
    assert diag.suggestions == ["Missing values appear in 2 rows. Estimators check only the columns they use"]


def test_diagnose_does_not_count_infinite_cohort_as_missing(two_by_two_panel):
    diag = diagnose_panel(two_by_two_panel, "id", "period")
    assert diag.n_missing_rows == 0
    assert diag.suggestions == []


@pytest.mark.filterwarnings("error:Dropped")
def test_diagnose_missing_value_in_unused_column_leaves_att_gt_sample_unchanged(mpdta):
    data = mpdta.with_columns(pl.Series("unused", [None] + [1.0] * (mpdta.height - 1)))
    diag = diagnose_panel(data, "countyreal", "year")
    result = att_gt(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", boot=False)
    assert diag.n_missing_rows == 1
    assert result.n_units == 500


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
    assert diag.suggestions == [
        "Call fill_panel_gaps() to fill 1 missing unit-time pair",
        "Call make_balanced_panel() to drop 1 unit not observed in all periods",
        "Call complete_data(min_periods=2) to drop 1 unit observed in only one period",
    ]


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
    assert diag.suggestions == [
        "Treatment varies within units. If your estimator takes gname, call get_group() to build it",
        "Estimators that take gname drop the 1 unit already treated in the first period",
    ]


def test_diagnose_static_treatment_has_no_early_treated_suggestion(static_treatment_panel):
    diag = diagnose_panel(static_treatment_panel, "id", "time", treatname="treat")
    rows = [line.split() for line in str(diag).splitlines()]
    assert diag.n_early_treated == 2
    assert diag.treatment_time_varying is False
    assert diag.suggestions == []
    assert ["│", "Treated", "in", "first", "period", "│", "2", "│"] in rows
    assert "Early-treated" not in str(diag)


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
