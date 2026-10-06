"""Tests for the main DDD wrapper function."""

import re

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import ddd, mboot_ddd
from moderndid.didtriple.container import DDDMultiPeriodResult, DDDPanelResult


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


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
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
        biters=50,
        random_state=42,
    )

    assert result.boots is not None
    assert len(result.boots) == 50
    assert result.se > 0


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
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
        biters=30,
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


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
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
        biters=20,
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
        biters=20,
        random_state=123,
    )

    assert np.allclose(result1.boots, result2.boots)
    assert result1.se == result2.se


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
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
        biters=25,
        alpha=0.10,
    )

    assert result.args["est_method"] == "reg"
    assert result.args["boot"] is True
    assert result.args["biters"] == 25
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
        biters=50,
        random_state=42,
    )

    assert isinstance(result, DDDMultiPeriodResult)
    assert result.args["boot"] is True
    assert result.args["biters"] == 50
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
        biters=50,
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
        biters=30,
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
        biters=30,
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
        biters=30,
        alpha=0.01,
    )

    assert result.args["control_group"] == "notyettreated"
    assert result.args["base_period"] == "varying"
    assert result.args["est_method"] == "ipw"
    assert result.args["boot"] is True
    assert result.args["biters"] == 30
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


@pytest.mark.parametrize("panel", [True, False])
def test_ddd_dotted_covariate_names(two_period_df, panel):
    renamed = two_period_df.with_columns(pl.col("cov1").alias("cov.1"), pl.col("cov2").alias("cov 2"))
    idname = "id" if panel else None
    spec = {"yname": "y", "tname": "time", "idname": idname, "gname": "state", "pname": "partition", "panel": panel}
    base = ddd(data=two_period_df, xformla="~ cov1 + cov2", **spec)
    dotted = ddd(data=renamed, xformla="~ cov.1 + `cov 2`", **spec)

    assert dotted.att == base.att
    assert dotted.se == base.se


def test_ddd_mp_dotted_covariate_names(multi_period_df):
    renamed = multi_period_df.with_columns(pl.col("cov1").alias("cov.1"))
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition"}
    base = ddd(data=multi_period_df, xformla="~ cov1 + cov2", **spec)
    dotted = ddd(data=renamed, xformla="~ cov.1 + cov2", **spec)

    np.testing.assert_array_equal(dotted.att, base.att)
    np.testing.assert_array_equal(dotted.se, base.se)


@pytest.mark.parametrize(
    "data_fixture, gname, panel",
    [("two_period_df", "state", True), ("two_period_df", "state", False), ("multi_period_df", "group", True)],
)
def test_ddd_rejects_transformed_covariates(request, data_fixture, gname, panel):
    data = request.getfixturevalue(data_fixture)

    with pytest.raises(ValueError, match=re.escape("xformla term 'I(cov1**2)' is not a column name")):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id" if panel else None,
            gname=gname,
            pname="partition",
            xformla="~ cov1 + I(cov1**2)",
            panel=panel,
        )


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
@pytest.mark.parametrize("data_fixture, panel", [("two_period_df", True), ("two_period_rcs_data", False)])
def test_ddd_2period_cluster_bootstrap(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("id") % 40).cast(pl.String).alias("cl"))
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition", "panel": panel}
    clustered = ddd(data=data, boot=True, biters=199, cluster="cl", random_state=3, **spec)
    plain = ddd(data=data, boot=True, biters=199, random_state=3, **spec)
    cluster = data.filter(pl.col("time") == data["time"].min()).sort("id")["cl"] if panel else data["cl"]
    expected = mboot_ddd(clustered.att_inf_func, 199, 0.05, cluster=cluster.to_numpy(), random_state=3)

    assert clustered.att == plain.att
    assert clustered.se == expected.se[0]
    assert clustered.se != plain.se
    assert clustered.uci == clustered.att + expected.crit_val * expected.se[0]


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
@pytest.mark.parametrize("data_fixture, panel", [("two_period_df", True), ("two_period_rcs_data", False)])
def test_ddd_2period_cluster_sets_boot(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("id") % 40).alias("cl"))
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition", "panel": panel}

    with pytest.warns(UserWarning, match="Clustered SEs require bootstrap"):
        forced = ddd(data=data, biters=199, cluster="cl", random_state=3, **spec)
    booted = ddd(data=data, boot=True, biters=199, cluster="cl", random_state=3, **spec)

    assert forced.args["boot"] is True
    assert forced.se == booted.se


@pytest.mark.parametrize("data_fixture, panel", [("two_period_df", True), ("two_period_rcs_data", False)])
def test_ddd_2period_cluster_rejects_weighted_bootstrap(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("id") % 40).alias("cl"))

    with pytest.raises(ValueError, match="cluster requires boot_type='multiplier'"):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            panel=panel,
            boot=True,
            boot_type="weighted",
            cluster="cl",
        )


@pytest.mark.parametrize(
    "cluster, match",
    [("county", "cluster='county' not found in data."), ("cl", "cluster='cl' has missing values.")],
)
def test_ddd_2period_rcs_cluster_errors(two_period_rcs_data, cluster, match):
    data = two_period_rcs_data.with_columns(
        pl.when(pl.col("id") % 7 == 0).then(None).otherwise(pl.col("id") % 40).alias("cl")
    )

    with pytest.raises(ValueError, match=re.escape(match)):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            panel=False,
            boot=True,
            cluster=cluster,
        )


@pytest.mark.filterwarnings("ignore:Setting cband=True for bootstrap:UserWarning")
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_ddd_2period_time_varying_cluster_raises(two_period_df, allow_unbalanced_panel):
    data = two_period_df.with_columns((pl.col("id") + 1000 * pl.col("time")).alias("cl")).filter(
        ~((pl.col("id") % 9 == 0) & (pl.col("time") == 2))
    )

    with pytest.raises(ValueError, match="Cluster variable must be time-invariant within units"):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            allow_unbalanced_panel=allow_unbalanced_panel,
            boot=True,
            cluster="cl",
        )
