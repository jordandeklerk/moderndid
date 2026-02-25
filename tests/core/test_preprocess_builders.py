"""Tests for builder paths."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess.builders import PreprocessDataBuilder
from moderndid.core.preprocess.config import (
    ContDIDConfig,
    DDDConfig,
    DIDConfig,
    DIDInterConfig,
    TwoPeriodDIDConfig,
)


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
