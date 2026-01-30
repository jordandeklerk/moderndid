"""Tests for the main DDD wrapper function."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import ddd
from moderndid.didtriple.estimators.ddd_mp import DDDMultiPeriodResult
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_2period_basic(two_period_df, est_method):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2",
        est_method=est_method,
    )

    assert isinstance(result, DDDPanelResult)
    assert isinstance(result.att, float)
    assert np.isfinite(result.att)
    assert result.se > 0
    assert result.lci < result.att < result.uci


def test_ddd_2period_no_covariates(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
    )

    assert isinstance(result, DDDPanelResult)
    assert np.isfinite(result.att)
    assert result.se > 0


def test_ddd_2period_with_covariates(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
    )

    assert isinstance(result, DDDPanelResult)
    assert np.isfinite(result.att)


def test_ddd_2period_bootstrap(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2",
        est_method="dr",
        boot=True,
        nboot=50,
        random_state=42,
    )

    assert result.boots is not None
    assert len(result.boots) == 50
    assert result.se > 0


@pytest.mark.parametrize("boot_type", ["multiplier", "weighted"])
def test_ddd_2period_boot_types(two_period_df, boot_type):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1",
        est_method="dr",
        boot=True,
        boot_type=boot_type,
        nboot=30,
        random_state=42,
    )

    assert result.boots is not None
    assert len(result.boots) == 30


def test_ddd_2period_influence_func(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
    )

    assert result.att_inf_func is not None
    n_units = two_period_df["id"].n_unique()
    assert len(result.att_inf_func) == n_units


def test_ddd_2period_subgroup_counts(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
    )

    assert "subgroup_1" in result.subgroup_counts
    assert "subgroup_2" in result.subgroup_counts
    assert "subgroup_3" in result.subgroup_counts
    assert "subgroup_4" in result.subgroup_counts
    assert all(c > 0 for c in result.subgroup_counts.values())


def test_ddd_2period_reproducibility(two_period_df):
    result1 = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
        boot=True,
        nboot=20,
        random_state=123,
    )

    result2 = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
        boot=True,
        nboot=20,
        random_state=123,
    )

    assert np.allclose(result1.boots, result2.boots)
    assert result1.se == result2.se


def test_ddd_2period_args_stored(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="reg",
        boot=True,
        nboot=25,
        alpha=0.10,
    )

    assert result.args["est_method"] == "reg"
    assert result.args["boot"] is True
    assert result.args["nboot"] == 25
    assert result.args["alpha"] == 0.10


def test_ddd_2period_print(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
    )

    output = str(result)
    assert "Triple Difference-in-Differences" in output
    assert "DR-DDD" in output
    assert "ATT" in output
    assert "Std. Error" in output


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_basic(multi_period_df, est_method):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method=est_method,
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert len(result.att) > 0
    assert len(result.se) == len(result.att)
    assert len(result.groups) == len(result.att)
    assert len(result.times) == len(result.att)


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_ddd_mp_control_group(multi_period_df, control_group):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group=control_group,
        est_method="dr",
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert result.args["control_group"] == control_group


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_base_period(multi_period_df, base_period):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        base_period=base_period,
        est_method="dr",
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert result.args["base_period"] == base_period


def test_ddd_mp_bootstrap(multi_period_df):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
        boot=True,
        nboot=50,
        random_state=42,
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert result.args["boot"] is True
    assert result.args["nboot"] == 50
    assert all(np.isfinite(se) or np.isnan(se) for se in result.se)


def test_ddd_mp_clustered(multi_period_df):
    np.random.seed(42)
    unique_ids = multi_period_df["id"].unique().to_list()
    cluster_map = {uid: np.random.randint(1, 51) for uid in unique_ids}
    df = multi_period_df.with_columns(pl.col("id").replace_strict(cluster_map, default=1).alias("cluster"))

    result = ddd(
        data=df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
        boot=True,
        nboot=50,
        cluster="cluster",
        random_state=42,
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert result.args["cluster"] == "cluster"


def test_ddd_mp_influence_func(multi_period_df):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
    )

    assert result.inf_func_mat is not None
    assert result.inf_func_mat.shape[0] == result.n
    assert result.inf_func_mat.shape[1] == len(result.att)


def test_ddd_mp_glist_tlist(multi_period_df):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
    )

    assert len(result.glist) > 0
    assert len(result.tlist) > 0
    assert all(g in result.glist for g in np.unique(result.groups))


def test_ddd_mp_reproducibility(multi_period_df):
    result1 = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
        boot=True,
        nboot=30,
        random_state=456,
    )

    result2 = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
        boot=True,
        nboot=30,
        random_state=456,
    )

    assert np.allclose(result1.att, result2.att)
    assert np.allclose(result1.se, result2.se, equal_nan=True)


def test_ddd_mp_args_stored(multi_period_df):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        control_group="notyettreated",
        base_period="varying",
        est_method="ipw",
        boot=True,
        nboot=30,
        alpha=0.01,
    )

    assert result.args["control_group"] == "notyettreated"
    assert result.args["base_period"] == "varying"
    assert result.args["est_method"] == "ipw"
    assert result.args["boot"] is True
    assert result.args["nboot"] == 30
    assert result.args["alpha"] == 0.01


def test_ddd_mp_print(multi_period_df):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
    )

    output = str(result)
    assert "Triple Difference-in-Differences" in output
    assert "Multi-Period" in output
    assert "ATT(g,t)" in output
    assert "Group" in output
    assert "Time" in output


def test_ddd_detects_2period(two_period_df):
    result = ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
    )

    assert isinstance(result, DDDPanelResult)


def test_ddd_detects_multiperiod(multi_period_df):
    result = ddd(
        data=multi_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
    )

    assert isinstance(result, DDDMultiPeriodResult)


def test_ddd_missing_covariate_error(multi_period_df):
    with pytest.raises(ValueError, match="Covariates not found"):
        ddd(
            data=multi_period_df,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            xformla="~ nonexistent_var",
            est_method="dr",
        )


@pytest.mark.parametrize("ddd_converted", ["pandas", "pyarrow", "duckdb"], indirect=True)
def test_ddd_dataframe_interoperability(ddd_converted, ddd_baseline_result):
    result = ddd(
        data=ddd_converted,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2",
        est_method="dr",
    )

    assert np.isclose(result.att, ddd_baseline_result.att)
    assert np.isclose(result.se, ddd_baseline_result.se)
    assert np.isclose(result.lci, ddd_baseline_result.lci)
    assert np.isclose(result.uci, ddd_baseline_result.uci)
