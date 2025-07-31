"""Tests for panel data and repeated cross-section utilities."""

import numpy as np
import pytest

from doublediff.utils import (
    are_varying,
    complete_data,
    convert_panel_time_to_int,
    create_relative_time_indicators,
    datetime_to_int,
    fill_panel_gaps,
    is_panel_balanced,
    is_repeated_cross_section,
    long_panel,
    make_panel_balanced,
    panel_has_gaps,
    panel_to_cross_section_diff,
    prepare_data_for_did,
    unpanel,
    validate_treatment_timing,
    widen_panel,
)

from .helpers import importorskip

pd = importorskip("pandas")


def test_is_panel_balanced():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df_balanced = pd.DataFrame({"y": np.random.randn(6)}, index=idx)
    assert is_panel_balanced(df_balanced)

    entities = [1, 1, 1, 2, 2]
    times = [2000, 2001, 2002, 2000, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df_unbalanced = pd.DataFrame({"y": np.random.randn(5)}, index=idx)
    assert not is_panel_balanced(df_unbalanced)


def test_is_panel_balanced_invalid_index():
    df = pd.DataFrame({"y": [1, 2, 3]}, index=[1, 2, 3])
    with pytest.raises(ValueError, match="Data must have a 2-level MultiIndex"):
        is_panel_balanced(df)

    idx = pd.MultiIndex.from_tuples([(1, 2000, "A"), (1, 2001, "A")])
    df = pd.DataFrame({"y": [1, 2]}, index=idx)
    with pytest.raises(ValueError, match="Data must have a 2-level MultiIndex"):
        is_panel_balanced(df)


def test_panel_has_gaps():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2002, 2003, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": np.random.randn(6)}, index=idx)

    gaps = panel_has_gaps(df)
    assert 1 in gaps
    assert 2001 in gaps[1]
    assert 2 not in gaps

    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df_no_gaps = pd.DataFrame({"y": np.random.randn(6)}, index=idx)

    gaps = panel_has_gaps(df_no_gaps)
    assert len(gaps) == 0


def test_is_repeated_cross_section():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df_panel = pd.DataFrame({"y": np.random.randn(4)}, index=idx)
    assert not is_repeated_cross_section(df_panel)

    entities = [1, 2, 3, 4]
    times = [2000, 2000, 2001, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df_rcs = pd.DataFrame({"y": np.random.randn(4)}, index=idx)
    assert is_repeated_cross_section(df_rcs)


def test_datetime_to_int_yearly():
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="YS")
    mapping = datetime_to_int(dates, freq="YS")

    assert mapping[pd.Timestamp("2020-01-01")] == 2020
    assert mapping[pd.Timestamp("2021-01-01")] == 2021
    assert mapping[pd.Timestamp("2022-01-01")] == 2022


def test_datetime_to_int_quarterly():
    dates = pd.date_range("2020-01-01", "2021-01-01", freq="QS")
    mapping = datetime_to_int(dates, freq="QS", start_value=1000)

    assert mapping[pd.Timestamp("2020-01-01")] == 1000
    assert mapping[pd.Timestamp("2020-04-01")] == 1001
    assert mapping[pd.Timestamp("2020-07-01")] == 1002
    assert mapping[pd.Timestamp("2020-10-01")] == 1003
    assert mapping[pd.Timestamp("2021-01-01")] == 1004


def test_datetime_to_int_invalid_input():
    with pytest.raises(TypeError, match="dates must be pd.Series or pd.DatetimeIndex"):
        datetime_to_int([1, 2, 3])


def test_convert_panel_time_to_int():
    entities = [1, 1, 2, 2]
    times = pd.to_datetime(["2020-01-01", "2021-01-01", "2020-01-01", "2021-01-01"])
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": np.random.randn(4)}, index=idx)

    df_int, reverse_map = convert_panel_time_to_int(df, freq="YS")

    assert df_int.index.get_level_values("time").unique().tolist() == [2020, 2021]
    assert reverse_map[2020] == pd.Timestamp("2020-01-01")
    assert reverse_map[2021] == pd.Timestamp("2021-01-01")


def test_convert_panel_time_to_int_non_datetime():
    entities = [1, 1, 2, 2]
    times = [2020, 2021, 2020, 2021]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": np.random.randn(4)}, index=idx)

    with pytest.raises(TypeError, match="Time index 'time' must be datetime type"):
        convert_panel_time_to_int(df)


def test_panel_to_cross_section_diff():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"y": [1.0, 1.5, 2.0, 3.0], "x1": [0.5, 0.5, 0.8, 0.8], "x2": [1.0, 1.2, 2.0, 2.5]},
        index=idx,
    )

    cs = panel_to_cross_section_diff(df, y_col="y", x_base_cols=["x1"], x_delta_cols=["x2"])

    assert np.isclose(cs.loc[1, "y"], 0.5)
    assert np.isclose(cs.loc[2, "y"], 1.0)

    assert np.isclose(cs.loc[1, "x1"], 0.5)
    assert np.isclose(cs.loc[2, "x1"], 0.8)

    assert np.isclose(cs.loc[1, "x2"], 0.2)
    assert np.isclose(cs.loc[2, "x2"], 0.5)


def test_panel_to_cross_section_diff_not_two_periods():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": np.random.randn(6)}, index=idx)

    with pytest.raises(ValueError, match="Panel must have exactly 2 time periods"):
        panel_to_cross_section_diff(df, y_col="y")


def test_panel_to_cross_section_diff_delta_name_conflict():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 1.5, 2.0, 3.0], "x1": [0.5, 0.6, 0.8, 0.9]}, index=idx)

    cs = panel_to_cross_section_diff(df, y_col="y", x_base_cols=["x1"], x_delta_cols=["x1"])

    assert "x1" in cs.columns
    assert "delta_x1" in cs.columns
    assert np.isclose(cs.loc[1, "delta_x1"], 0.1)


def test_fill_panel_gaps():
    entities = [1, 1, 2, 2, 2]
    times = [2000, 2002, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 1.5, 2.0, 2.2, 2.5]}, index=idx)

    filled = fill_panel_gaps(df)
    assert (1, 2001) in filled.index
    assert np.isnan(filled.loc[(1, 2001), "y"])

    filled_zero = fill_panel_gaps(df, fill_value=0.0)
    assert filled_zero.loc[(1, 2001), "y"] == 0.0

    filled_ffill = fill_panel_gaps(df, method="ffill")
    assert filled_ffill.loc[(1, 2001), "y"] == 1.0

    filled_bfill = fill_panel_gaps(df, method="bfill")
    assert filled_bfill.loc[(1, 2001), "y"] == 1.5


def test_fill_panel_gaps_invalid_method():
    entities = [1, 1]
    times = [2000, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 2.0]}, index=idx)

    with pytest.raises(ValueError, match="Unknown method 'invalid'"):
        fill_panel_gaps(df, method="invalid")


def test_make_panel_balanced_drop():
    entities = [1, 1, 1, 2, 2, 3]
    times = [2000, 2001, 2002, 2000, 2002, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": np.random.randn(6)}, index=idx)

    balanced = make_panel_balanced(df, method="drop")

    assert balanced.index.get_level_values("entity").unique().tolist() == [1]
    assert len(balanced) == 3


def test_make_panel_balanced_fill():
    entities = [1, 1, 2, 2, 3]
    times = [2000, 2001, 2000, 2002, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 1.5, 2.0, 2.5, 3.0]}, index=idx)

    balanced = make_panel_balanced(df, min_periods=2, method="fill", fill_value=0.0)

    entities_kept = balanced.index.get_level_values("entity").unique()
    assert 1 in entities_kept
    assert 2 in entities_kept
    assert 3 not in entities_kept

    assert balanced.loc[(2, 2001), "y"] == 0.0


def test_make_panel_balanced_no_entities_warning():
    entities = [1, 2, 3]
    times = [2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=idx)

    with pytest.warns(UserWarning, match="No entities have 5 periods"):
        balanced = make_panel_balanced(df, min_periods=5, method="drop")

    assert len(balanced) == 0


def test_create_relative_time_indicators():
    entities = [1, 1, 1, 1, 2, 2, 2, 2]
    times = [2000, 2001, 2002, 2003] * 2
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"cohort": [2002, 2002, 2002, 2002, 2001, 2001, 2001, 2001]}, index=idx)

    indicators = create_relative_time_indicators(df, "cohort")

    expected_cols = ["rel_time_-2", "rel_time_+0", "rel_time_+1", "rel_time_+2"]
    assert all(col in indicators.columns for col in expected_cols)

    assert "rel_time_-1" not in indicators.columns

    assert indicators.loc[(1, 2000), "rel_time_-2"] == 1
    assert indicators.loc[(1, 2002), "rel_time_+0"] == 1
    assert indicators.loc[(1, 2003), "rel_time_+1"] == 1


def test_create_relative_time_indicators_never_treated():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"cohort": [2001, 2001, np.nan, np.nan]}, index=idx)

    indicators = create_relative_time_indicators(df, "cohort")

    assert indicators.loc[(2, 2000)].sum() == 0
    assert indicators.loc[(2, 2001)].sum() == 0


def test_validate_treatment_timing():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002] * 2
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"treated": [0, 1, 1, 0, 0, 1], "cohort": [2001, 2001, 2001, 2002, 2002, 2002]},
        index=idx,
    )

    validation = validate_treatment_timing(df, "treated", "cohort")

    assert not validation["has_reversals"]
    assert validation["timing_consistent"]
    assert len(validation["always_treated"]) == 0
    assert len(validation["never_treated"]) == 0


def test_validate_treatment_timing_with_reversals():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002] * 2
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"treated": [0, 1, 0, 0, 0, 1], "cohort": [2001, 2001, 2001, 2002, 2002, 2002]},
        index=idx,
    )

    validation = validate_treatment_timing(df, "treated", "cohort")

    assert validation["has_reversals"]
    assert 1 in validation["entities_with_reversals"]


def test_validate_treatment_timing_always_never_treated():
    entities = [1, 1, 2, 2, 3, 3]
    times = [2000, 2001] * 3
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {
            "treated": [1, 1, 0, 0, 0, 1],
        },
        index=idx,
    )

    validation = validate_treatment_timing(df, "treated")

    assert 1 in validation["always_treated"]
    assert 2 in validation["never_treated"]
    assert 3 not in validation["always_treated"]
    assert 3 not in validation["never_treated"]


def test_validate_treatment_timing_inconsistent():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002] * 2
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"treated": [0, 0, 1, 0, 1, 1], "cohort": [2001, 2001, 2001, 2001, 2001, 2001]},
        index=idx,
    )

    validation = validate_treatment_timing(df, "treated", "cohort")

    assert not validation["timing_consistent"]
    assert 1 in validation["inconsistent_entities"]


def test_prepare_data_for_did():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "y": [1.0, 1.5, 2.0, 2.5],
            "treated": [0, 1, 0, 0],
            "x1": [0.5, 0.6, 0.8, 0.9],
        }
    )

    prepared = prepare_data_for_did(
        df, y_col="y", entity_col="unit", time_col="year", treat_col="treated", covariates=["x1"]
    )

    assert isinstance(prepared.index, pd.MultiIndex)
    assert prepared.index.names == ["unit", "year"]

    assert "y" in prepared.columns
    assert "treated" in prepared.columns
    assert "x1" in prepared.columns

    assert prepared.index.is_monotonic_increasing


def test_prepare_data_for_did_missing_columns():
    df = pd.DataFrame({"unit": [1, 1, 2, 2], "year": [2000, 2001, 2000, 2001], "y": [1.0, 1.5, 2.0, 2.5]})

    with pytest.raises(ValueError, match="Columns not found in data: \\['missing_col'\\]"):
        prepare_data_for_did(df, y_col="y", entity_col="unit", time_col="year", covariates=["missing_col"])


def test_prepare_data_for_did_missing_values_warning():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "y": [1.0, np.nan, 2.0, 2.5],
            "treated": [0, 1, 0, 0],
        }
    )

    with pytest.warns(UserWarning, match="Missing values found"):
        prepare_data_for_did(df, y_col="y", entity_col="unit", time_col="year", treat_col="treated")


def test_prepare_data_for_did_non_binary_treatment_warning():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "y": [1.0, 1.5, 2.0, 2.5],
            "treated": [0, 2, 0, 1],
        }
    )

    with pytest.warns(UserWarning, match="Treatment column 'treated' contains non-binary values"):
        prepare_data_for_did(df, y_col="y", entity_col="unit", time_col="year", treat_col="treated")


def test_widen_panel_basic():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"y": [1.0, 1.5, 2.0, 2.5], "x_varying": [0.5, 0.6, 0.8, 0.9], "x_constant": [1, 1, 2, 2]},
        index=idx,
    )

    wide = widen_panel(df)

    assert len(wide) == 2
    assert list(wide.index) == [1, 2]

    assert "y_2000" in wide.columns
    assert "y_2001" in wide.columns
    assert "x_varying_2000" in wide.columns
    assert "x_varying_2001" in wide.columns
    assert "x_constant" in wide.columns

    assert wide.loc[1, "y_2000"] == 1.0
    assert wide.loc[1, "y_2001"] == 1.5
    assert wide.loc[2, "y_2000"] == 2.0
    assert wide.loc[2, "y_2001"] == 2.5

    assert wide.loc[1, "x_constant"] == 1
    assert wide.loc[2, "x_constant"] == 2


def test_widen_panel_custom_separator():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {
            "y": [1.0, 1.5, 2.0, 2.5],
        },
        index=idx,
    )

    wide = widen_panel(df, separator=".")

    assert "y.2000" in wide.columns
    assert "y.2001" in wide.columns


def test_widen_panel_explicit_varying():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"y": [1.0, 1.5, 2.0, 2.5], "x1": [0.5, 0.5, 0.8, 0.8], "x2": [1, 2, 3, 4]},
        index=idx,
    )

    wide = widen_panel(df, varying=["y"])

    assert "y_2000" in wide.columns
    assert "y_2001" in wide.columns
    assert "x1" in wide.columns
    assert "x2" in wide.columns

    assert "x1_2000" not in wide.columns
    assert "x2_2000" not in wide.columns


def test_widen_panel_no_varying_vars():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"x1": [1, 1, 2, 2], "x2": [3, 3, 4, 4]}, index=idx)

    with pytest.warns(UserWarning, match="No time-varying variables found"):
        wide = widen_panel(df)

    assert len(wide) == 2
    assert wide.loc[1, "x1"] == 1
    assert wide.loc[2, "x2"] == 4


def test_widen_panel_invalid_index():
    df = pd.DataFrame({"y": [1, 2, 3]}, index=[1, 2, 3])
    with pytest.raises(ValueError, match="Data must have a 2-level MultiIndex"):
        widen_panel(df)


def test_long_panel_basic():
    wide_df = pd.DataFrame(
        {
            "entity": [1, 2, 3],
            "y_2000": [1.0, 2.0, 3.0],
            "y_2001": [1.5, 2.5, 3.5],
            "x_2000": [0.5, 0.7, 0.9],
            "x_2001": [0.6, 0.8, 1.0],
            "gender": ["M", "F", "M"],
        }
    )

    long = long_panel(wide_df, entity_col="entity")

    assert isinstance(long.index, pd.MultiIndex)
    assert long.index.names == ["entity", "time"]

    assert len(long) == 6
    assert set(long.columns) == {"gender", "y", "x"}

    assert long.loc[(1, "2000"), "y"] == 1.0
    assert long.loc[(1, "2001"), "y"] == 1.5
    assert long.loc[(2, "2000"), "y"] == 2.0

    assert long.loc[(1, "2000"), "gender"] == "M"
    assert long.loc[(1, "2001"), "gender"] == "M"


def test_long_panel_custom_separator():
    wide_df = pd.DataFrame(
        {
            "entity": [1, 2],
            "y.0": [1.0, 2.0],
            "y.1": [1.5, 2.5],
        }
    )

    long = long_panel(wide_df, entity_col="entity", separator=".")

    assert len(long) == 4
    assert long.loc[(1, "0"), "y"] == 1.0
    assert long.loc[(1, "1"), "y"] == 1.5


def test_long_panel_explicit_stubs():
    wide_df = pd.DataFrame(
        {
            "entity": [1, 2],
            "outcome_2000": [1.0, 2.0],
            "outcome_2001": [1.5, 2.5],
            "treatment_2000": [0, 0],
            "treatment_2001": [1, 0],
            "constant": [10, 20],
        }
    )

    long = long_panel(wide_df, entity_col="entity", stub_names=["outcome", "treatment"])

    assert "outcome" in long.columns
    assert "treatment" in long.columns
    assert "constant" in long.columns


def test_long_panel_custom_time_values():
    wide_df = pd.DataFrame(
        {
            "entity": [1, 2],
            "y_pre": [1.0, 2.0],
            "y_post": [1.5, 2.5],
        }
    )

    long = long_panel(wide_df, entity_col="entity", suffix="pre|post", time_values=["pre", "post"])

    assert set(long.index.get_level_values("time").unique()) == {"pre", "post"}
    assert long.loc[(1, "pre"), "y"] == 1.0
    assert long.loc[(1, "post"), "y"] == 1.5


def test_long_panel_unbalanced():
    wide_df = pd.DataFrame(
        {
            "entity": [1, 2, 3],
            "y_2000": [1.0, 2.0, np.nan],
            "y_2001": [1.5, 2.5, 3.5],
            "y_2002": [1.7, np.nan, 3.7],
        }
    )

    long = long_panel(wide_df, entity_col="entity")

    assert len(long) == 7
    assert (2, "2002") not in long.index
    assert (3, "2000") not in long.index
    assert (2, "2001") in long.index
    assert long.loc[(2, "2001"), "y"] == 2.5


def test_long_panel_no_entity_column():
    wide_df = pd.DataFrame({"y_2000": [1.0, 2.0], "y_2001": [1.5, 2.5]})

    with pytest.raises(ValueError, match="Entity column 'entity' not found"):
        long_panel(wide_df, entity_col="entity")


def test_long_panel_no_time_varying():
    wide_df = pd.DataFrame({"entity": [1, 2, 3], "constant1": [1, 2, 3], "constant2": ["A", "B", "C"]})

    with pytest.raises(ValueError, match="No time-varying variables detected"):
        long_panel(wide_df, entity_col="entity")


def test_widen_long_panel_roundtrip():
    entities = np.repeat([1, 2, 3], 3)
    times = np.tile([2000, 2001, 2002], 3)
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])

    original = pd.DataFrame(
        {
            "y": np.random.randn(9),
            "x_varying": np.random.randn(9),
            "x_constant": np.repeat([10, 20, 30], 3),
            "category": np.repeat(["A", "B", "C"], 3),
        },
        index=idx,
    )

    wide = widen_panel(original)
    back_to_long = long_panel(wide, entity_col="entity")

    assert len(back_to_long) == len(original)
    assert set(back_to_long.columns) == set(original.columns)

    for entity in [1, 2, 3]:
        for time in [2000, 2001, 2002]:
            orig_val = original.loc[(entity, time), "y"]
            round_val = back_to_long.loc[(entity, str(time)), "y"]
            assert np.isclose(orig_val, round_val)


def test_complete_data_basic():
    entities = [1, 1, 1, 2, 2, 2, 3, 3]
    times = [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {
            "y": [1.0, 1.5, 1.7, 2.0, np.nan, 2.5, 3.0, 3.5],
            "x": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        },
        index=idx,
    )

    result = complete_data(df, "y", min_periods=2)

    assert 1 in result.index.get_level_values("entity").unique()
    assert 2 in result.index.get_level_values("entity").unique()
    assert 3 in result.index.get_level_values("entity").unique()

    # Test with min_periods=3
    result_strict = complete_data(df, "y", min_periods=3)
    assert 1 in result_strict.index.get_level_values("entity").unique()
    assert 2 not in result_strict.index.get_level_values("entity").unique()
    assert 3 not in result_strict.index.get_level_values("entity").unique()


def test_complete_data_all_waves():
    entities = [1, 1, 1, 2, 2, 3, 3, 3]
    times = [2000, 2001, 2002, 2000, 2002, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": np.random.randn(8), "x": np.random.randn(8)}, index=idx)

    result = complete_data(df, min_periods="all")

    assert len(result.index.get_level_values("entity").unique()) == 2
    assert 1 in result.index.get_level_values("entity").unique()
    assert 3 in result.index.get_level_values("entity").unique()
    assert 2 not in result.index.get_level_values("entity").unique()


def test_complete_data_formula():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"y": [1.0, np.nan, 2.0, 2.5], "x1": [0.5, 0.6, 0.8, 0.9], "x2": [1.0, 1.1, 1.2, 1.3]},
        index=idx,
    )

    result = complete_data(df, formula="y ~ x1", min_periods=2)

    assert 1 not in result.index.get_level_values("entity").unique()
    assert 2 in result.index.get_level_values("entity").unique()


def test_complete_data_vars_list():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 1.5, 2.0, 2.5], "x": [0.5, np.nan, 0.8, 0.9]}, index=idx)

    result = complete_data(df, variables=["y", "x"], min_periods=2)

    assert 1 not in result.index.get_level_values("entity").unique()
    assert 2 in result.index.get_level_values("entity").unique()


def test_complete_data_no_complete_obs_warning():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [np.nan, np.nan, np.nan, np.nan]}, index=idx)

    with pytest.warns(UserWarning, match="No complete observations"):
        result = complete_data(df, min_periods="all")

    assert len(result) == 0


def test_complete_data_no_entities_warning():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, np.nan, 2.0, np.nan]}, index=idx)

    with pytest.warns(UserWarning, match="No entities have 2 complete observations"):
        result = complete_data(df, "y", min_periods=2)

    assert len(result) == 0


def test_complete_data_invalid_column():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 1.5, 2.0, 2.5]}, index=idx)

    with pytest.raises(ValueError, match="Column 'missing' not found"):
        complete_data(df, "missing")


def test_are_varying_basic():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"constant": [1, 1, 1, 1], "time_var": [1, 2, 3, 4], "entity_fe": [1, 1, 2, 2]},
        index=idx,
    )

    result = are_varying(df, ["constant", "time_var", "entity_fe"], return_names=False)

    assert not result["constant"]
    assert result["time_var"]
    assert not result["entity_fe"]


def test_are_varying_individual():
    entities = [1, 1, 1, 2, 2, 2]
    times = [2000, 2001, 2002, 2000, 2001, 2002]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {"age": [20, 21, 22, 30, 31, 32], "varying_pattern": [1, 2, 3, 1, 3, 5]},
        index=idx,
    )

    result = are_varying(df, ["age", "varying_pattern"], variation_type="individual", return_names=False)

    assert not result["age"]
    assert result["varying_pattern"]


def test_are_varying_both():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"x1": [1, 1, 2, 2], "x2": [1, 2, 3, 4]}, index=idx)

    result = are_varying(df, ["x1", "x2"], variation_type="both", return_names=False)

    assert isinstance(result, pd.DataFrame)
    assert not result.loc["time", "x1"]
    assert result.loc["time", "x2"]
    assert not result.loc["individual", "x1"]
    assert not result.loc["individual", "x2"]


def test_are_varying_with_tolerance():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"x": [1.0, 1.0000001, 2.0, 2.0000001]}, index=idx)

    result_default = are_varying(df, ["x"], return_names=False)
    assert not result_default["x"]

    result_tight = are_varying(df, ["x"], rtol=1e-10, atol=1e-10, return_names=False)
    assert result_tight["x"]


def test_are_varying_enhanced():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"x1": [1, 1, 2, 2], "x2": [1, 2, 3, 4]}, index=idx)

    result_names = are_varying(df, ["x1", "x2"])
    assert result_names == ["x2"]

    result_dict = are_varying(df, ["x1", "x2"], return_names=False)
    assert not result_dict["x1"]
    assert result_dict["x2"]

    result_both = are_varying(df, variation_type="both", return_names=False)
    assert isinstance(result_both, pd.DataFrame)
    assert result_both.shape == (2, 2)


def test_unpanel_basic():
    entities = [1, 1, 2, 2]
    times = [2000, 2001, 2000, 2001]
    idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    df = pd.DataFrame({"y": [1.0, 1.5, 2.0, 2.5], "x": [0.5, 0.6, 0.8, 0.9]}, index=idx)

    result = unpanel(df)

    assert isinstance(result.index, pd.RangeIndex)
    assert "entity" in result.columns
    assert "time" in result.columns
    assert "y" in result.columns
    assert "x" in result.columns
    assert len(result) == 4


def test_unpanel_non_multiindex():
    df = pd.DataFrame({"y": [1, 2, 3]}, index=[0, 1, 2])
    result = unpanel(df)

    assert result.equals(df)
