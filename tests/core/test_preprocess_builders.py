"""Tests for builder paths."""

import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import gen_cont_did_data
from moderndid.core.preprocess.builders import PreprocessDataBuilder
from moderndid.core.preprocess.config import (
    ContDIDConfig,
    DDDConfig,
    DIDConfig,
    DIDInterConfig,
    TwoPeriodDIDConfig,
)
from moderndid.core.preprocessing import preprocess_cont_did, preprocess_did


@pytest.fixture
def builder():
    return PreprocessDataBuilder()


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(42)
    n_units = 60
    n_periods = 4
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    treat_time = np.where(np.arange(n_units) < 20, 3, np.where(np.arange(n_units) < 40, 4, 0))
    groups = np.repeat(treat_time, n_periods)
    y = rng.standard_normal(n_units * n_periods) + (groups > 0).astype(float) * 2.0
    x1 = rng.standard_normal(n_units * n_periods)
    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "group": groups,
            "x1": x1,
        }
    )


@pytest.mark.parametrize(
    "config_type, config_kwargs, expected_cls",
    [
        ("did", {"yname": "y", "tname": "time", "idname": "id", "gname": "group"}, DIDConfig),
        ("cont_did", {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "dname": "dose"}, ContDIDConfig),
    ],
)
def test_with_config_dict(builder, panel_data, config_type, config_kwargs, expected_cls):
    if config_type == "cont_did":
        panel_data = panel_data.with_columns(pl.Series("dose", np.random.default_rng(0).uniform(0, 1, len(panel_data))))
    b = builder.with_data(panel_data).with_config_dict(config_type=config_type, **config_kwargs)
    assert b._config is not None
    assert isinstance(b._config, expected_cls)


@pytest.mark.parametrize(
    "config",
    [
        TwoPeriodDIDConfig(yname="y", tname="t", idname="id", treat_col="D"),
        DIDInterConfig(yname="y", tname="t", gname="id", dname="d"),
        DDDConfig(yname="y", tname="t", idname="id", gname="group", pname="p"),
        DIDConfig(yname="y", tname="t", idname="id", gname="g"),
    ],
)
def test_with_config_dispatches_correctly(builder, config):
    b = builder.with_config(config)
    assert b._validator is not None


@pytest.mark.parametrize(
    "setup, method, match",
    [
        ("config_only", "validate", "Data not set"),
        ("data_only", "validate", "Configuration not set"),
        ("empty", "transform", "Must set data and config"),
        ("empty", "build", "Must set data and config"),
    ],
)
def test_builder_raises_on_missing_prerequisites(builder, panel_data, setup, method, match):
    if setup == "config_only":
        builder.with_config(DIDConfig(yname="y", tname="t", idname="id", gname="g"))
    elif setup == "data_only":
        builder.with_data(panel_data)

    with pytest.raises(ValueError, match=match):
        getattr(builder, method)()


def test_transform_raises_without_transformer(builder, panel_data):
    builder.with_data(panel_data)
    builder._config = DIDConfig(yname="y", tname="t", idname="id", gname="g")
    with pytest.raises(ValueError, match="Transformer not initialized"):
        builder.transform()


@pytest.mark.parametrize(
    "panel, n, expected_attr",
    [
        (True, 40, "y1"),
        (False, 80, "y"),
    ],
)
def test_build_two_period(panel, n, expected_attr):
    rng = np.random.default_rng(42)
    if panel:
        ids = np.repeat(np.arange(n), 2)
        times = np.tile([1, 2], n)
        d = np.repeat(rng.binomial(1, 0.5, n), 2)
        y = rng.standard_normal(n * 2)
    else:
        ids = np.arange(n)
        times = np.concatenate([np.ones(n // 2), np.full(n // 2, 2)]).astype(int)
        d = rng.binomial(1, 0.5, n)
        y = rng.standard_normal(n)

    df = pl.DataFrame({"id": ids, "t": times, "y": y, "D": d})
    config = TwoPeriodDIDConfig(yname="y", tname="t", idname="id", treat_col="D", panel=panel)
    result = PreprocessDataBuilder().with_data(df).with_config(config).validate().transform().build()
    assert getattr(result, expected_attr) is not None


@pytest.mark.filterwarnings("ignore:Be aware that there are some small groups:UserWarning")
def test_validate_transformed_data_small_groups():
    rng = np.random.default_rng(42)
    n_units = 20
    n_periods = 4
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    groups = np.repeat(np.where(np.arange(n_units) < 2, 3, 0), n_periods)
    y = rng.standard_normal(n_units * n_periods)
    covs = {f"x{i}": rng.standard_normal(n_units * n_periods) for i in range(1, 7)}

    df = pl.DataFrame({"id": units, "time": periods, "y": y, "group": groups, **covs})

    config = DIDConfig(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ x1 + x2 + x3 + x4 + x5 + x6",
    )

    with pytest.warns(UserWarning, match="small groups"):
        PreprocessDataBuilder().with_data(df).with_config(config).validate().transform()


@pytest.mark.parametrize(
    "expected_substr",
    ["DiD Preprocessing Summary", "Data Format", "Control Group"],
)
def test_get_did_summary(builder, panel_data, expected_substr):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="group")
    builder.with_data(panel_data).with_config(config)
    builder.validate().transform()

    tensor_data = {
        "cohort_counts": pl.DataFrame({"cohort": [0.0, 3.0, 4.0], "cohort_size": [20, 20, 20]}),
    }
    summary = builder._get_did_summary(tensor_data)
    assert summary is not None
    assert expected_substr in summary


def test_get_did_summary_with_warnings(builder, panel_data):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="group")
    builder.with_data(panel_data).with_config(config)
    builder._warnings = ["warning 1", "warning 2", "warning 3", "warning 4"]

    tensor_data = {
        "cohort_counts": pl.DataFrame({"cohort": [0.0, 3.0], "cohort_size": [30, 30]}),
    }
    summary = builder._get_did_summary(tensor_data)
    assert "Warnings (4)" in summary
    assert "warning 1" in summary
    assert "... and 1 more" in summary


@pytest.mark.parametrize(
    "summary_method, config_val",
    [
        ("_get_did_summary", None),
        ("_get_cont_did_summary", DIDConfig(yname="y", tname="time", idname="id", gname="g")),
    ],
)
def test_summary_returns_none_for_wrong_config(builder, panel_data, summary_method, config_val):
    builder.with_data(panel_data)
    builder._config = config_val
    assert getattr(builder, summary_method)({}) is None


def test_get_did_summary_never_treated_cohort(builder, panel_data):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="group")
    builder.with_data(panel_data).with_config(config)
    builder.validate().transform()

    tensor_data = {
        "cohort_counts": pl.DataFrame({"cohort": [float("inf"), 3.0], "cohort_size": [30, 30]}),
    }
    summary = builder._get_did_summary(tensor_data)
    assert "Never Treated" in summary


def test_get_cont_did_summary(builder):
    config = ContDIDConfig(yname="y", tname="time", idname="id", gname="group", dname="dose")
    config.has_dose = False
    builder._config = config

    summary_tables = {
        "cohort_counts": pl.DataFrame({"cohort": [0.0, 3.0], "cohort_size": [25, 15]}),
    }
    summary = builder._get_cont_did_summary(summary_tables)
    assert summary is not None
    assert "Continuous Treatment" in summary


@pytest.mark.parametrize(
    "expected_substr",
    ["Dose Variable", "Spline Degree"],
)
def test_get_cont_did_summary_with_dose(builder, expected_substr):
    config = ContDIDConfig(
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        dname="dose",
        degree=3,
        num_knots=5,
    )
    config.has_dose = True
    builder._config = config

    summary_tables = {
        "cohort_counts": pl.DataFrame({"cohort": [0.0, 3.0], "cohort_size": [25, 15]}),
    }
    summary = builder._get_cont_did_summary(summary_tables)
    assert expected_substr in summary


def test_get_cont_did_summary_with_warnings(builder):
    config = ContDIDConfig(yname="y", tname="time", idname="id", gname="group", dname="dose")
    config.has_dose = False
    builder._config = config
    builder._warnings = ["w1", "w2", "w3", "w4"]

    summary_tables = {
        "cohort_counts": pl.DataFrame({"cohort": [0.0, 3.0], "cohort_size": [25, 15]}),
    }
    summary = builder._get_cont_did_summary(summary_tables)
    assert "Warnings (4)" in summary
    assert "... and 1 more" in summary


def test_get_cont_did_summary_many_cohorts(builder):
    config = ContDIDConfig(yname="y", tname="time", idname="id", gname="group", dname="dose")
    config.has_dose = False
    config.treated_groups_count = 12
    builder._config = config

    cohorts = list(range(12))
    sizes = [10] * 12
    summary_tables = {
        "cohort_counts": pl.DataFrame({"cohort": [float(c) for c in cohorts], "cohort_size": sizes}),
    }
    summary = builder._get_cont_did_summary(summary_tables)
    assert "... and 2 more cohorts" in summary


def test_two_period_band_rule_spares_continuous_treatment():
    rng = np.random.default_rng(3)
    n = 200
    group = np.repeat(rng.choice([0, 2], n), 2)
    df = pl.DataFrame(
        {
            "id": np.repeat(np.arange(n), 2),
            "time": np.tile([1, 2], n),
            "y": rng.standard_normal(2 * n),
            "group": group,
            "dose": np.repeat(rng.uniform(0.1, 1.0, n), 2) * (group > 0),
        }
    )

    did_data = preprocess_did(df, yname="y", tname="time", gname="group", idname="id", cband=True)
    dose_data = preprocess_cont_did(df, yname="y", tname="time", gname="group", dname="dose", idname="id", cband=True)
    event_data = preprocess_cont_did(
        df, yname="y", tname="time", gname="group", dname="dose", idname="id", cband=True, aggregation="eventstudy"
    )

    assert did_data.config.cband is False
    assert dose_data.config.cband is True
    assert event_data.config.cband is False


def test_cont_did_preprocessing_recodes_groups_to_period_positions():
    df = gen_cont_did_data(n=200, seed=5).with_columns(
        (pl.col("time_period") + 2000).alias("time_period"),
        pl.when(pl.col("G") > 0).then(pl.col("G") + 2000).otherwise(0).alias("G"),
    )
    data = preprocess_cont_did(df, yname="Y", tname="time_period", gname="G", dname="D", idname="id")

    assert data.time_map == {2001: 1, 2002: 2, 2003: 3, 2004: 4}
    assert sorted(data.data["time_period"].unique().to_list()) == [1, 2, 3, 4]
    np.testing.assert_array_equal(data.config.treated_groups, [2.0, 3.0, 4.0])


def test_cont_did_preprocessing_without_never_treated_drops_late_periods():
    df = gen_cont_did_data(n=200, p_untreated=0.0, seed=5)
    kwargs = {"yname": "Y", "tname": "time_period", "gname": "G", "dname": "D", "idname": "id"}

    with pytest.warns(UserWarning, match="no unit is untreated from period 4 on"):
        data = preprocess_cont_did(df, **kwargs)
    with pytest.raises(ValueError, match="needs never-treated units"):
        preprocess_cont_did(df, control_group="nevertreated", **kwargs)

    np.testing.assert_array_equal(data.config.time_periods, [1, 2, 3])


def test_cont_did_preprocessing_without_never_treated_needs_a_treated_period():
    df = gen_cont_did_data(n=200, seed=5).filter(pl.col("G") == 3)

    with pytest.raises(ValueError, match="no treated period has untreated units to compare with"):
        preprocess_cont_did(df, yname="Y", tname="time_period", gname="G", dname="D", idname="id")


def test_cont_did_preprocessing_rejects_dose_that_changes_over_time():
    df = gen_cont_did_data(n=200, seed=5).with_columns(
        pl.when(pl.col("time_period") == 4).then(pl.col("D") + 0.1).otherwise(pl.col("D")).alias("D")
    )

    with pytest.raises(ValueError, match="must stay the same over time"):
        preprocess_cont_did(df, yname="Y", tname="time_period", gname="G", dname="D", idname="id")


def test_cont_did_preprocessing_fills_dose_recorded_as_zero_before_treatment():
    df = gen_cont_did_data(n=200, seed=5)
    zero_before = df.with_columns(
        pl.when(pl.col("time_period") < pl.col("G")).then(0.0).otherwise(pl.col("D")).alias("D")
    )
    kwargs = {"yname": "Y", "tname": "time_period", "gname": "G", "dname": "D", "idname": "id"}

    filled = preprocess_cont_did(zero_before, **kwargs).data
    constant = preprocess_cont_did(df, **kwargs).data

    np.testing.assert_array_equal(filled["id"].to_numpy(), constant["id"].to_numpy())
    np.testing.assert_array_equal(filled["D"].to_numpy(), constant["D"].to_numpy())


def test_cont_did_preprocessing_rejects_group_between_observed_periods():
    df = gen_cont_did_data(n=200, seed=5).filter(pl.col("time_period") != 3)

    with pytest.raises(ValueError, match="Treatment starts between observed periods for group 3\\."):
        preprocess_cont_did(df, yname="Y", tname="time_period", gname="G", dname="D", idname="id")


def test_cont_did_preprocessing_names_each_group_between_observed_periods():
    df = gen_cont_did_data(n=200, seed=5).filter(pl.col("time_period").is_in([1, 4]))

    with pytest.raises(ValueError, match="Treatment starts between observed periods for groups 2, 3\\."):
        preprocess_cont_did(df, yname="Y", tname="time_period", gname="G", dname="D", idname="id")


def test_small_group_guard_unbalanced_panel_counts_rows_per_period(small_never_treated_panel):
    sparse_controls = small_never_treated_panel.filter(~((pl.col("g") == 0) & pl.col("t").is_in([1, 3])))
    config = DIDConfig(yname="y", tname="t", idname="id", gname="g", allow_unbalanced_panel=True)
    builder = PreprocessDataBuilder().with_data(sparse_controls).with_config(config).validate()

    with pytest.warns(UserWarning, match="Check groups: inf$"), pytest.raises(ValueError, match="too small"):
        builder.transform()


def test_small_group_guard_keeps_balanced_panel_with_enough_controls(small_never_treated_panel):
    config = DIDConfig(yname="y", tname="t", idname="id", gname="g", allow_unbalanced_panel=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PreprocessDataBuilder().with_data(small_never_treated_panel).with_config(config).validate().transform()

    assert not [w for w in caught if "small groups" in str(w.message)]


def test_small_group_guard_cross_sections_count_rows_per_period(small_never_treated_cross_sections):
    config = DIDConfig(yname="y", tname="t", gname="g", panel=False)
    builder = PreprocessDataBuilder().with_data(small_never_treated_cross_sections).with_config(config).validate()

    with pytest.warns(UserWarning, match="Check groups: inf$"), pytest.raises(ValueError, match="too small"):
        builder.transform()


@pytest.mark.parametrize(
    ("config_class", "config_kwargs", "data_name"),
    [
        (
            DIDConfig,
            {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~ lpop"},
            "mpdta_with_nan",
        ),
        (
            ContDIDConfig,
            {"yname": "Y", "tname": "time_period", "idname": "id", "gname": "G", "dname": "D"},
            "cont_did_panel_with_nan",
        ),
        (
            TwoPeriodDIDConfig,
            {"yname": "y", "tname": "year", "treat_col": "treat", "idname": "id", "xformla": "~ x", "panel": False},
            "drdid_panel_with_nan",
        ),
        (
            DDDConfig,
            {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition"},
            "ddd_panel_with_nan",
        ),
        (
            DIDInterConfig,
            {"yname": "y", "tname": "t", "gname": "id", "dname": "d", "xformla": "~ x"},
            "didinter_panel_with_nan",
        ),
    ],
)
def test_builder_treats_nan_as_missing_for_pandas_and_polars(request, config_class, config_kwargs, data_name):
    data = request.getfixturevalue(data_name)
    built = [
        PreprocessDataBuilder().with_data(frame).with_config(config_class(**config_kwargs)).validate().transform()._data
        for frame in (data, data.to_pandas())
    ]

    assert built[0].equals(built[1])
    assert sum(int(column.is_nan().sum()) for column in built[0].iter_columns() if column.dtype.is_float()) == 0


@pytest.mark.parametrize(
    ("config_class", "config_kwargs", "data_name"),
    [
        (
            DIDConfig,
            {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~ lpop"},
            "mpdta_with_nan",
        ),
        (
            ContDIDConfig,
            {"yname": "Y", "tname": "time_period", "idname": "id", "gname": "G", "dname": "D"},
            "cont_did_panel_with_nan",
        ),
        (
            TwoPeriodDIDConfig,
            {"yname": "y", "tname": "year", "treat_col": "treat", "idname": "id", "xformla": "~ x", "panel": False},
            "drdid_panel_with_nan",
        ),
        (
            DDDConfig,
            {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition"},
            "ddd_panel_with_nan",
        ),
        (
            DIDInterConfig,
            {"yname": "y", "tname": "t", "gname": "id", "dname": "d", "xformla": "~ x"},
            "didinter_panel_with_nan",
        ),
    ],
)
def test_builder_treats_infinity_like_nan(request, config_class, config_kwargs, data_name):
    with_nan = request.getfixturevalue(data_name)
    with_infinity = with_nan.with_columns(pl.col(pl.Float64).fill_nan(float("inf")))
    built = [
        PreprocessDataBuilder().with_data(frame).with_config(config_class(**config_kwargs)).validate().transform()._data
        for frame in (with_nan, with_infinity)
    ]

    assert built[0].equals(built[1])


@pytest.mark.parametrize(
    ("config_class", "config_kwargs", "data_name"),
    [
        (DIDConfig, {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}, "mpdta"),
        (
            ContDIDConfig,
            {"yname": "Y", "tname": "time_period", "idname": "id", "gname": "G", "dname": "D"},
            "cont_did_panel_with_nan",
        ),
        (
            TwoPeriodDIDConfig,
            {"yname": "y", "tname": "year", "treat_col": "treat", "idname": "id"},
            "drdid_panel_data",
        ),
        (
            DDDConfig,
            {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition"},
            "ddd_panel_with_nan",
        ),
        (DIDInterConfig, {"yname": "y", "tname": "t", "gname": "id", "dname": "d"}, "didinter_panel_with_nan"),
    ],
)
def test_builder_rejects_weights_without_positive_mean(request, config_class, config_kwargs, data_name):
    data = request.getfixturevalue(data_name).with_columns(pl.lit(0.0).alias("w"))
    config = config_class(**config_kwargs, weightsname="w")
    builder = PreprocessDataBuilder().with_data(data).with_config(config).validate()

    with pytest.raises(ValueError, match="^The weights variable 'w' must be non-negative with a positive mean\\.$"):
        builder.transform()


@pytest.mark.parametrize(
    ("config_class", "config_kwargs"),
    [
        (DIDConfig, {"yname": "y", "tname": "time", "idname": "id", "gname": "group"}),
        (ContDIDConfig, {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "dname": "x1"}),
    ],
)
def test_builder_rejects_negative_cohort(panel_data, config_class, config_kwargs):
    data = panel_data.with_columns(pl.col("group").replace(0, -2), pl.col("x1").abs())
    builder = PreprocessDataBuilder().with_data(data).with_config(config_class(**config_kwargs)).validate()

    with pytest.raises(ValueError, match="^gname = 'group' holds negative values such as -2\\."):
        builder.transform()


def test_builder_names_the_columns_when_no_row_is_complete(mpdta):
    data = mpdta.with_columns(pl.lit(float("nan")).alias("lpop"))
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat", xformla="~ lpop")
    builder = PreprocessDataBuilder().with_data(data).with_config(config).validate()

    with pytest.raises(ValueError, match="^Every row has a missing value in 'lpop'\\. No data is left"):
        builder.transform()
