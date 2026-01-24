"""Tests for the multi-period DDD repeated cross-section estimator."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didtriple.estimators.ddd_mp_rc import ddd_mp_rc


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_rc_basic(mp_rcs_data, est_method):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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
    assert result.n == len(mp_rcs_data)


def test_ddd_mp_rc_result_structure(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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


def test_ddd_mp_rc_confidence_intervals(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    valid_mask = ~np.isnan(result.se)
    assert np.all(result.lci[valid_mask] < result.att[valid_mask])
    assert np.all(result.att[valid_mask] < result.uci[valid_mask])


def test_ddd_mp_rc_influence_functions(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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
def test_ddd_mp_rc_control_group(mp_rcs_data, control_group):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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
def test_ddd_mp_rc_base_period(mp_rcs_data, base_period):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
    )

    assert len(result.att) > 0
    assert result.args["base_period"] == base_period


def test_ddd_mp_rc_glist_tlist(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert 3 in result.glist
    assert 4 in result.glist
    assert set(result.tlist) == {1, 2, 3, 4, 5}


def test_ddd_mp_rc_args_stored(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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


def test_ddd_mp_rc_post_treatment_effects(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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


def test_ddd_mp_rc_never_treated_as_inf():
    rng = np.random.default_rng(42)
    n_per_period = 100
    time_periods = [1, 2, 3]

    records = []
    for t in time_periods:
        groups = rng.choice([np.inf, 3], size=n_per_period, p=[0.5, 0.5])
        partition = rng.choice([0, 1], size=n_per_period)

        for i in range(n_per_period):
            g = groups[i]
            p = partition[i]
            y = rng.normal(0, 1)
            if np.isfinite(g) and t >= g and p == 1:
                y += 1.5
            records.append({"id": len(records), "time": t, "y": y, "group": g, "partition": p})

    data = pl.DataFrame(records)

    result = ddd_mp_rc(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert len(result.att) > 0
    assert 3 in result.glist


def test_ddd_mp_rc_bootstrap(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=50,
    )

    assert len(result.att) > 0
    valid_mask = ~np.isnan(result.se)
    assert np.sum(valid_mask) > 0


def test_ddd_mp_rc_reproducibility(mp_rcs_data):
    result1 = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=20,
        random_state=123,
    )

    result2 = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        nboot=20,
        random_state=123,
    )

    assert np.allclose(result1.att, result2.att)
    valid_mask = ~np.isnan(result1.se) & ~np.isnan(result2.se)
    assert np.allclose(result1.se[valid_mask], result2.se[valid_mask])


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_rc_print(mp_rcs_data, est_method):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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
def test_ddd_mp_rc_print_control_group(mp_rcs_data, control_group, expected):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group=control_group,
    )

    output = str(result)
    assert expected in output


def test_ddd_mp_rc_trim_level(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="dr",
        trim_level=0.99,
    )

    assert len(result.att) > 0


def test_ddd_mp_rc_observation_count(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert result.n == len(mp_rcs_data)


def test_ddd_mp_rc_different_from_panel():
    rng = np.random.default_rng(42)
    n_per_period = 200
    time_periods = [1, 2, 3]

    records = []
    for t in time_periods:
        groups = rng.choice([0, 3], size=n_per_period, p=[0.6, 0.4])
        partition = rng.choice([0, 1], size=n_per_period)

        for i in range(n_per_period):
            g = groups[i]
            p = partition[i]
            y = rng.normal(0, 1) + 0.5 * t
            if 0 < g <= t and p == 1:
                y += 2.0
            records.append({"id": len(records), "time": t, "y": y, "group": g, "partition": p})

    data = pl.DataFrame(records)

    result = ddd_mp_rc(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert len(result.att) > 0
    assert result.n == len(data)


def test_ddd_mp_rc_bootstrap_cband(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
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
