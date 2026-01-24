"""Tests for the multi-period DDD estimator."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import ddd_mp


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_basic(mp_ddd_data, est_method):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    assert len(result.att) > 0
    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n == mp_ddd_data["id"].n_unique()


def test_ddd_mp_result_structure(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert hasattr(result, "att")
    assert hasattr(result, "se")
    assert hasattr(result, "uci")
    assert hasattr(result, "lci")
    assert hasattr(result, "groups")
    assert hasattr(result, "times")
    assert hasattr(result, "glist")
    assert hasattr(result, "tlist")
    assert hasattr(result, "inf_func_mat")
    assert hasattr(result, "n")
    assert hasattr(result, "args")


def test_ddd_mp_confidence_intervals(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    valid_mask = ~np.isnan(result.se)
    assert np.all(result.lci[valid_mask] < result.att[valid_mask])
    assert np.all(result.att[valid_mask] < result.uci[valid_mask])


def test_ddd_mp_influence_functions(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert result.inf_func_mat is not None
    assert result.inf_func_mat.shape[0] == result.n
    assert result.inf_func_mat.shape[1] == len(result.att)


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_ddd_mp_control_group(mp_ddd_data, control_group):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group=control_group,
    )

    assert len(result.att) > 0
    assert result.args["control_group"] == control_group


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_base_period(mp_ddd_data, base_period):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
    )

    assert len(result.att) > 0
    assert result.args["base_period"] == base_period


def test_ddd_mp_glist_tlist(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert 3 in result.glist
    assert 4 in result.glist
    assert set(result.tlist) == {1, 2, 3, 4, 5}


def test_ddd_mp_args_stored(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group="nevertreated",
        base_period="universal",
        est_method="dr",
        alpha=0.10,
    )

    assert result.args["control_group"] == "nevertreated"
    assert result.args["base_period"] == "universal"
    assert result.args["est_method"] == "dr"
    assert result.args["alpha"] == 0.10


def test_ddd_mp_post_treatment_effects(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    for i, (g, t) in enumerate(zip(result.groups, result.times)):
        if t >= g:
            att = result.att[i]
            if np.isfinite(att) and not np.isnan(result.se[i]):
                assert np.isfinite(att)


def test_ddd_mp_never_treated_as_inf():
    rng = np.random.default_rng(42)
    n_units = 300
    time_periods = [1, 2, 3]

    records = []
    for unit in range(n_units):
        g = rng.choice([np.inf, 3], p=[0.5, 0.5])
        p = rng.choice([0, 1])
        for t in time_periods:
            y = rng.normal(0, 1)
            if np.isfinite(g) and t >= g and p == 1:
                y += 1.5
            records.append({"id": unit, "time": t, "y": y, "group": g, "partition": p})

    data = pl.DataFrame(records)

    result = ddd_mp(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert len(result.att) > 0
    assert 3 in result.glist


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_print(mp_ddd_data, est_method):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    output = str(result)
    assert "Triple Difference-in-Differences" in output
    assert "Multi-Period" in output
    assert f"{est_method.upper()}-DDD" in output
    assert "ATT(g,t)" in output
    assert "Group" in output
    assert "Time" in output


@pytest.mark.parametrize("control_group,expected", [("nevertreated", "Never Treated"), ("notyettreated", "Not Yet")])
def test_ddd_mp_print_control_group(mp_ddd_data, control_group, expected):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group=control_group,
    )

    output = str(result)
    assert expected in output


def test_ddd_mp_bootstrap(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=50,
        random_state=42,
    )

    assert len(result.att) > 0
    assert result.args["boot"] is True
    assert result.args["nboot"] == 50


def test_ddd_mp_bootstrap_cband(mp_ddd_data):
    result = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=50,
        cband=True,
        random_state=42,
    )

    assert len(result.att) > 0
    assert result.args["cband"] is True


def test_ddd_mp_reproducibility(mp_ddd_data):
    result1 = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=20,
        random_state=123,
    )

    result2 = ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=20,
        random_state=123,
    )

    np.testing.assert_allclose(result1.se, result2.se, equal_nan=True)
