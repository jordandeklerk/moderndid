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
