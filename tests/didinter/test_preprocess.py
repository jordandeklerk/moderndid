"""Tests for DIDInter preprocessing."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.core.preprocess.models import DIDInterData


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
