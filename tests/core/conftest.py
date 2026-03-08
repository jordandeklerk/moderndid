"""Shared fixtures for core tests."""

import numpy as np
import polars as pl
import pytest

from moderndid import (
    agg_ddd,
    aggte,
    att_gt,
    cont_did,
    ddd,
    did_multiplegt,
    gen_cont_did_data,
    gen_ddd_mult_periods,
    honest_did,
    load_favara_imbs,
    load_mpdta,
)
from moderndid.drdid.drdid import drdid
from moderndid.drdid.ipwdid import ipwdid
from moderndid.drdid.ordid import ordid


@pytest.fixture(scope="session")
def mpdta():
    return load_mpdta()


@pytest.fixture(scope="session")
def att_gt_analytical(mpdta):
    return att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
        cband=False,
    )


@pytest.fixture(scope="session")
def att_gt_bootstrap(mpdta):
    return att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        control_group="nevertreated",
        boot=True,
        cband=True,
        biters=100,
        random_state=42,
    )


@pytest.fixture(
    scope="session",
    params=["simple", "dynamic", "group", "calendar"],
)
def aggte_result(request, att_gt_analytical):
    return aggte(att_gt_analytical, type=request.param)


@pytest.fixture(scope="session")
def drdid_panel_data():
    rng = np.random.default_rng(42)
    n = 200
    d = rng.binomial(1, 0.5, n)
    x = rng.normal(0, 1, n)
    y0 = 1.0 + 0.5 * x + rng.normal(0, 0.5, n)
    y1 = y0 + 0.3 * d + rng.normal(0, 0.5, n)
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n), 2),
            "year": np.tile([2000, 2001], n),
            "y": np.concatenate([y0, y1]),
            "treat": np.repeat(d, 2),
            "x": np.repeat(x, 2),
        }
    )


@pytest.fixture(scope="session", params=[drdid, ipwdid, ordid], ids=["drdid", "ipwdid", "ordid"])
def drdid_result(request, drdid_panel_data):
    return request.param(
        data=drdid_panel_data,
        yname="y",
        tname="year",
        idname="id",
        treatname="treat",
    )


@pytest.fixture(scope="session")
def ddd_mp_data():
    return gen_ddd_mult_periods(n=300, random_state=42)["data"]


@pytest.fixture(scope="session")
def ddd_mp_result(ddd_mp_data):
    return ddd(
        data=ddd_mp_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        boot=False,
    )


@pytest.fixture(scope="session")
def ddd_agg_result(ddd_mp_result):
    return agg_ddd(ddd_mp_result, type="eventstudy")


@pytest.fixture(scope="session")
def didinter_result():
    data = load_favara_imbs()
    return did_multiplegt(
        data=data,
        yname="Dl_vloans_b",
        tname="year",
        idname="county",
        dname="inter_bra",
        effects=2,
        placebo=1,
    )


@pytest.fixture(scope="session")
def cont_did_result():
    data = gen_cont_did_data(n=100, num_time_periods=4, seed=42)
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        boot=False,
    )


@pytest.fixture
def balanced_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [10, 12, 15, 20, 22, 25, 30, 32, 35],
            "x": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
        }
    )


@pytest.fixture
def unbalanced_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            "time": [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3],
            "y": [10, 12, 15, 20, 22, 25, 32, 35, 40, 42, 45],
            "treat": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def panel_with_duplicates():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2],
            "time": [1, 1, 2, 3, 1, 2, 2],
            "y": [10.0, 11.0, 12.0, 15.0, 20.0, 22.0, 24.0],
            "cat": ["a", "b", "a", "a", "c", "c", "d"],
        }
    )


@pytest.fixture
def staggered_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "y": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
            "treat": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        }
    )


@pytest.fixture(scope="session")
def aggte_dynamic(att_gt_analytical):
    return aggte(att_gt_analytical, type="dynamic")


@pytest.fixture(scope="session")
def aggte_group(att_gt_analytical):
    return aggte(att_gt_analytical, type="group")


@pytest.fixture(scope="session")
def aggte_calendar(att_gt_analytical):
    return aggte(att_gt_analytical, type="calendar")


@pytest.fixture(scope="session")
def cont_did_event():
    data = gen_cont_did_data(n=100, num_time_periods=4, seed=42)
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="eventstudy",
        boot=False,
    )


@pytest.fixture(scope="session")
def honest_did_result(aggte_dynamic):
    return honest_did(aggte_dynamic, event_time=0, sensitivity_type="relative_magnitude")
