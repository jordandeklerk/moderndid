"""Tests for DIDInter preprocessing."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.core.preprocess.models import DIDInterData
from moderndid.core.preprocess.transformers import ControlsTimeFilter, SwitcherIdentifier


@pytest.fixture
def simple_panel():
    n_units = 30
    n_periods = 5

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 10:
            treatment[unit_mask & (periods >= 3)] = 1
        elif unit < 15:
            treatment[unit_mask & (periods >= 4)] = 1

    y = np.random.default_rng(42).standard_normal(len(units)) + 2.0 * treatment

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


@pytest.fixture
def basic_config():
    return DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )


def test_preprocess_creates_didinter_data(simple_panel, basic_config):
    basic_config.effects = 2

    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    assert isinstance(result, DIDInterData)


@pytest.mark.parametrize(
    "expected_column",
    ["F_g", "d_sq", "S_g", "L_g"],
)
def test_preprocess_computes_switcher_columns(simple_panel, basic_config, expected_column):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    assert expected_column in result.data.columns


@pytest.mark.parametrize(
    "unit_id,expected_f_g",
    [
        (0, 3),
        (10, 4),
        (20, float("inf")),
    ],
)
def test_preprocess_f_g_values(simple_panel, basic_config, unit_id, expected_f_g):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    unit_f_g = result.data.filter(pl.col("id") == unit_id)["F_g"][0]
    assert unit_f_g == expected_f_g


@pytest.mark.parametrize(
    "unit_id,expected_s_g",
    [
        (0, 1),
        (20, 0),
    ],
)
def test_preprocess_s_g_values(simple_panel, basic_config, unit_id, expected_s_g):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    unit_s_g = result.data.filter(pl.col("id") == unit_id)["S_g"][0]
    assert unit_s_g == expected_s_g


@pytest.mark.parametrize(
    "unit_id,expected_l_g",
    [
        (0, 3),
        (10, 2),
    ],
)
def test_preprocess_l_g_values(simple_panel, basic_config, unit_id, expected_l_g):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    unit_l_g = result.data.filter(pl.col("id") == unit_id)["L_g"][0]
    assert unit_l_g == expected_l_g


def test_preprocess_d_sq_value(simple_panel, basic_config):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    unit_0_d_sq = result.data.filter(pl.col("id") == 0)["d_sq"][0]
    assert unit_0_d_sq == 0.0


@pytest.mark.parametrize(
    "property_name,expected_value",
    [
        ("n_switchers", 15),
        ("n_never_switchers", 15),
    ],
)
def test_switcher_count_properties(simple_panel, basic_config, property_name, expected_value):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    assert getattr(result, property_name) == expected_value


def test_has_never_switchers_property(simple_panel, basic_config):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    assert result.has_never_switchers is True


def test_preprocess_with_cluster(simple_panel, basic_config):
    df = simple_panel.with_columns((pl.col("id") // 5).alias("cluster"))
    basic_config.cluster = "cluster"

    result = PreprocessDataBuilder().with_data(df).with_config(basic_config).validate().transform().build()

    assert result.cluster is not None


def test_preprocess_with_weights(simple_panel, basic_config):
    df = simple_panel.with_columns(pl.lit(1.0).alias("w"))
    basic_config.weightsname = "w"

    result = PreprocessDataBuilder().with_data(df).with_config(basic_config).validate().transform().build()

    assert result.weights is not None


def test_preprocess_with_controls(simple_panel, basic_config):
    rng = np.random.default_rng(42)
    df = simple_panel.with_columns(
        [
            pl.Series("x1", rng.standard_normal(len(simple_panel))),
            pl.Series("x2", rng.standard_normal(len(simple_panel))),
        ]
    )
    basic_config.controls = ["x1", "x2"]

    result = PreprocessDataBuilder().with_data(df).with_config(basic_config).validate().transform().build()

    assert result.has_controls is True


def test_validation_missing_column(simple_panel):
    config = DIDInterConfig(
        yname="missing_y",
        tname="time",
        gname="id",
        dname="d",
    )

    with pytest.raises(ValueError, match="missing_y"):
        PreprocessDataBuilder().with_data(simple_panel).with_config(config).validate()


def test_validation_no_switchers():
    n_units = 20
    n_periods = 4

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(len(units))
    y = np.random.default_rng(42).standard_normal(len(units))

    df = pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )

    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )

    with pytest.raises(ValueError, match="No units change treatment"):
        PreprocessDataBuilder().with_data(df).with_config(config).validate()


@pytest.fixture
def bidirectional_panel():
    n_units = 20
    n_periods = 6

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 5:
            treatment[unit_mask & (periods >= 3) & (periods <= 4)] = 1
        elif unit < 10:
            treatment[unit_mask & (periods >= 3)] = 1

    y = np.random.default_rng(42).standard_normal(len(units))

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


@pytest.mark.parametrize("keep_bidirectional", [True, False])
def test_bidirectional_switchers_handling(bidirectional_panel, keep_bidirectional):
    config = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        keep_bidirectional_switchers=keep_bidirectional,
    )

    result = PreprocessDataBuilder().with_data(bidirectional_panel).with_config(config).validate().transform().build()

    assert isinstance(result, DIDInterData)


def test_drop_bidirectional_reduces_units(bidirectional_panel):
    config_keep = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        keep_bidirectional_switchers=True,
    )

    config_drop = DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
        keep_bidirectional_switchers=False,
    )

    result_keep = (
        PreprocessDataBuilder().with_data(bidirectional_panel).with_config(config_keep).validate().transform().build()
    )
    result_drop = (
        PreprocessDataBuilder().with_data(bidirectional_panel).with_config(config_drop).validate().transform().build()
    )

    n_units_keep = result_keep.data["id"].n_unique()
    n_units_drop = result_drop.data["id"].n_unique()

    assert n_units_drop <= n_units_keep


@pytest.mark.parametrize(
    "expected_column",
    ["F_g", "d_sq", "S_g"],
)
def test_time_invariant_data_columns(simple_panel, basic_config, expected_column):
    result = PreprocessDataBuilder().with_data(simple_panel).with_config(basic_config).validate().transform().build()

    assert result.time_invariant_data is not None
    assert expected_column in result.time_invariant_data.columns


def test_time_ranker_numbers_periods_by_rank_and_keeps_their_values(simple_panel, basic_config):
    years = {1: 1990, 2: 1992, 3: 1993, 4: 1999, 5: 2004}
    df = simple_panel.with_columns(pl.col("time").replace_strict(years))
    basic_config.trends_lin = True

    result = PreprocessDataBuilder().with_data(df).with_config(basic_config).validate().transform().build()

    assert sorted(result.data["time"].unique().to_list()) == [2, 3, 4, 5]
    np.testing.assert_array_equal(result.config.time_periods, [1992, 1993, 1999, 2004])
    assert result.data.filter(pl.col("id") == 10)["F_g"][0] == 4


def test_controls_time_filter_keeps_periods_with_not_yet_switched_groups():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 4] * 3,
            "d_sq": [0.0] * 12,
            "F_g": [2.0] * 4 + [3.0] * 4 + [4.0] * 4,
        }
    )
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d")

    result = ControlsTimeFilter().transform(df, config)

    assert sorted(result["time"].unique().to_list()) == [1, 2, 3]
    assert result.height == 9


@pytest.mark.filterwarnings("ignore:Requested effects=4:UserWarning")
def test_switchers_out_keep_groups_whose_treatment_rises(two_way_panel_data):
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d", switchers="out", effects=4)

    result = PreprocessDataBuilder().with_data(two_way_panel_data).with_config(config).validate().transform().build()

    assert result.data.filter(pl.col("S_g") == 1)["id"].n_unique() == 28
    assert config.effects == 3


def test_continuous_pools_distinct_baselines_and_marks_each_switch(continuous_panel_data):
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d", effects=3, continuous=2)

    data = (
        PreprocessDataBuilder().with_data(continuous_panel_data).with_config(config).validate().transform().build().data
    )
    switched = data.filter(pl.col("F_g") != float("inf"))
    trends = [name for name in data.columns if name.startswith("_baseline_trend_")]
    row = data.filter((pl.col("id") == 5) & (pl.col("time") == 4))

    assert data["id"].n_unique() == 120
    assert (data["d_sq"] == 0).all()
    assert (data["d_sq_int"] == 1).all()
    assert (data.filter(pl.col("F_g") == float("inf"))["weight_gt"] == 1).all()
    np.testing.assert_array_equal(switched["d"], switched["S_g"] * (switched["time"] >= switched["F_g"]))
    np.testing.assert_array_equal(switched["d_fg"], switched["S_g"])
    assert trends == [f"_baseline_trend_{t}_{k}" for t in range(2, 7) for k in (1, 2)]
    np.testing.assert_allclose(row["_baseline_trend_3_2"], row["d_sq_orig"] ** 2)
    np.testing.assert_allclose(row["_baseline_trend_5_1"], 0.0)


def test_trends_lin_keeps_outcome_levels_and_baseline_trends_after_the_first_period(continuous_panel_data):
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d", effects=3, continuous=1, trends_lin=True)

    data = (
        PreprocessDataBuilder().with_data(continuous_panel_data).with_config(config).validate().transform().build().data
    )
    levels = data.join(continuous_panel_data.select("id", "time", pl.col("y").alias("raw")), on=["id", "time"])

    assert [name for name in data.columns if name.startswith("_baseline_trend_")] == [
        f"_baseline_trend_{t}_1" for t in range(3, 7)
    ]
    np.testing.assert_allclose(levels["_outcome_levels"], levels["raw"])


def test_balanced_panel_keeps_the_observed_baseline_in_every_row(baseline_shift_panels):
    tenth, _ = baseline_shift_panels
    config = DIDInterConfig(yname="y", tname="t", gname="g", dname="d", effects=3)

    data = PreprocessDataBuilder().with_data(tenth).with_config(config).validate().transform().build().data

    assert data.height == data["g"].n_unique() * data["t"].n_unique()
    assert data["d_sq"].unique().to_list() == [0.1]
    assert data["d_sq_int"].unique().to_list() == [1]


def test_switcher_identifier_ranks_the_baseline_treatments(order_sensitive_panel):
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d")

    result = SwitcherIdentifier().transform(order_sensitive_panel, config)

    ranks = dict(result.group_by("id").agg(pl.col("d_sq_int").first()).iter_rows())
    assert ranks == {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 2, 7: 3, 8: 3}


def test_switcher_identifier_reads_each_group_in_time_order_whatever_order_joins_return(
    order_sensitive_panel, shuffling_joins
):
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d")

    result = SwitcherIdentifier().transform(order_sensitive_panel, config)

    groups = result.group_by("id").agg(pl.col("time").max(), pl.col("S_g").first()).sort("id")
    assert groups["time"].to_list() == [6, 6, 3, 4, 6, 6, 3, 4]
    assert groups["S_g"].to_list() == [1, 1, -1, 1, 0, 0, 1, -1]
    assert result.select("id", "time").equals(result.select("id", "time").sort("id", "time"))
