"""Shared fixtures for drdid tests."""

from collections import namedtuple

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import load_nsw


@pytest.fixture(scope="module")
def nsw_data():
    return load_nsw()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dr_panel_result():
    DRDIDPanel = namedtuple("DRDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return DRDIDPanel(att=1.5, se=0.3, lci=0.9, uci=2.1, args={})


@pytest.fixture
def dr_rc_result():
    DRDIDRc = namedtuple("DRDIDRcResult", ["att", "se", "lci", "uci", "args"])
    return DRDIDRc(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})


@pytest.fixture
def ipw_result():
    IPW = namedtuple("IPWDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return IPW(att=1.0, se=0.3, lci=0.4, uci=1.6, args={})


@pytest.fixture
def reg_result():
    Reg = namedtuple("RegDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return Reg(att=1.0, se=0.3, lci=0.4, uci=1.6, args={})


@pytest.fixture
def twfe_result():
    TWFE = namedtuple("TWFEDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return TWFE(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})


@pytest.fixture
def unknown_result():
    Unknown = namedtuple("FooBarResult", ["att", "se", "lci", "uci", "args"])
    return Unknown(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})


@pytest.fixture
def result_with_call_params():
    WithCP = namedtuple("DRDIDPanelCallResult", ["att", "se", "lci", "uci", "args", "call_params"])
    return WithCP(att=1.0, se=0.2, lci=0.6, uci=1.4, args={}, call_params={"data_shape": (500, 8)})


@pytest.fixture(params=["repeated_row", "hidden_gap", "relabeled_row"])
def nsw_duplicated(request, nsw_data):
    """NSW panel in which unit 15995 has two rows in 1975."""
    unit = pl.col("id") == 15995
    row = nsw_data.filter(unit & (pl.col("year") == 1975))
    if request.param == "repeated_row":
        return pl.concat([nsw_data, row])
    if request.param == "hidden_gap":
        return pl.concat([nsw_data.filter(~(unit & (pl.col("year") == 1978))), row])
    return pl.concat([nsw_data, row.with_columns(pl.col("re") + 1000)])


@pytest.fixture
def nsw_one_infinite(request, nsw_data):
    """NSW panel with weights in w and an infinite value in one 1978 row of unit 15995."""
    column, value = request.param
    row = (pl.col("id") == 15995) & (pl.col("year") == 1978)
    data = nsw_data.with_columns(pl.lit(1.5).alias("w"))
    return data.with_columns(pl.when(row).then(value).otherwise(pl.col(column).cast(pl.Float64)).alias(column))
