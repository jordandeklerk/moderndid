"""Tests for the main DDD wrapper function."""

import re
import warnings

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


@pytest.mark.parametrize("data_fixture, panel", [("two_period_df", True), ("two_period_rcs_data", False)])
def test_ddd_2period_cluster_sets_boot(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("id") % 40).alias("cl"))
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition", "panel": panel}

    with pytest.warns(UserWarning, match=r"^Clustered SEs require bootstrap\. Setting boot=True\.$"):
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


@pytest.mark.parametrize(
    "data_fixture, gname, panel",
    [
        ("two_period_df", "state", True),
        ("two_period_rcs_data", "state", False),
        ("multi_period_df", "group", True),
        ("mp_rcs_data", "group", False),
    ],
)
def test_ddd_alpha_above_tenth_uses_005_on_every_route(request, data_fixture, gname, panel):
    data = request.getfixturevalue(data_fixture)
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id" if panel else None,
        "gname": gname,
        "pname": "partition",
        "panel": panel,
        "est_method": "reg",
    }

    with pytest.warns(UserWarning, match=re.escape("alpha=0.2 is above 0.10. Using alpha=0.05.")):
        high = ddd(data=data, alpha=0.2, **spec)
    default = ddd(data=data, **spec)

    assert high.args["alpha"] == 0.05
    np.testing.assert_array_equal(high.lci, default.lci)
    np.testing.assert_array_equal(high.uci, default.uci)


@pytest.mark.parametrize("data_fixture, panel", [("multi_period_df", True), ("mp_rcs_data", False)])
def test_ddd_mp_cluster_without_boot_warns(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("id") % 30).alias("cl"))
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id" if panel else None,
        "gname": "group",
        "pname": "partition",
        "panel": panel,
        "est_method": "reg",
    }

    with pytest.warns(UserWarning, match="cluster has no effect with several periods unless boot=True"):
        clustered = ddd(data=data, cluster="cl", **spec)
    plain = ddd(data=data, **spec)

    assert clustered.args["boot"] is False
    np.testing.assert_array_equal(clustered.se, plain.se)


def test_ddd_2period_bootstrap_does_not_warn_about_cband(two_period_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ddd(
            data=two_period_df,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            boot=True,
            biters=49,
            random_state=1,
        )

    assert all("cband" not in str(w.message) for w in caught)
    assert result.args["boot"] is True


@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_ddd_2period_duplicated_unit_periods_raise(two_period_duplicated_df, allow_unbalanced_panel):
    with pytest.raises(ValueError, match=re.escape("The value of idname must be unique (by tname).")):
        ddd(
            data=two_period_duplicated_df,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla="~ cov1 + cov2 + cov3 + cov4",
            est_method="reg",
            allow_unbalanced_panel=allow_unbalanced_panel,
        )


@pytest.mark.parametrize("data_fixture, panel", [("two_period_df", True), ("two_period_rcs_data", False)])
def test_ddd_2period_partition_must_be_binary(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("partition") + 1).alias("partition"))
    message = (
        "pname='partition' must be 1 for eligible units and 0 for ineligible units, but it also takes the values [2]."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id" if panel else None,
            gname="state",
            pname="partition",
            panel=panel,
        )


def test_ddd_2period_boolean_partition_matches_integer(two_period_df):
    flagged = two_period_df.with_columns(pl.col("partition") == 1)
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition"}

    assert ddd(data=flagged, **spec).att == ddd(data=two_period_df, **spec).att


@pytest.mark.parametrize("data_fixture, panel", [("two_period_df", True), ("two_period_rcs_data", False)])
def test_ddd_2period_requires_all_four_subgroups(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).filter(~((pl.col("state") == 0) & (pl.col("partition") == 1)))

    with pytest.raises(ValueError, match=re.escape("No units fall in subgroup 2 (untreated and eligible).")):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id" if panel else None,
            gname="state",
            pname="partition",
            panel=panel,
        )


def test_ddd_2period_balancing_warns_and_keeps_complete_units(two_period_df):
    unbalanced = two_period_df.filter(~((pl.col("id") % 25 == 0) & (pl.col("time") == 2)))
    complete = unbalanced.filter(pl.len().over("id") == 2)
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition", "est_method": "reg"}

    with pytest.warns(UserWarning, match="Dropped 40 units while converting to balanced panel"):
        result = ddd(data=unbalanced, **spec)
    expected = ddd(data=complete, **spec)

    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize(
    "two_period_df_one_infinite",
    [("y", float("inf")), ("cov1", float("-inf")), ("w", float("inf"))],
    indirect=True,
)
def test_ddd_2period_drops_infinite_rows_like_missing_ones(two_period_df_one_infinite):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "state",
        "pname": "partition",
        "xformla": "~ cov1 + cov2",
        "weightsname": "w",
        "est_method": "reg",
    }
    expected = ddd(data=two_period_df_one_infinite.filter(pl.col("id") != 11), **spec)

    with (
        pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"),
        pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"),
    ):
        result = ddd(data=two_period_df_one_infinite, **spec)

    assert result.att == expected.att
    assert result.se == expected.se


def test_ddd_2period_rejects_weights_without_positive_mean(two_period_df):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        ddd(
            data=two_period_df.with_columns(pl.lit(0.0).alias("w")),
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            weightsname="w",
            est_method="reg",
        )


def test_ddd_2period_outcome_named_weights(two_period_df):
    spec = {
        "tname": "time",
        "idname": "id",
        "gname": "state",
        "pname": "partition",
        "xformla": "~ cov1 + cov2",
        "est_method": "reg",
    }
    renamed = ddd(data=two_period_df.rename({"y": "weights"}), yname="weights", **spec)
    expected = ddd(data=two_period_df, yname="y", **spec)

    assert renamed.att == expected.att
    assert renamed.se == expected.se


@pytest.mark.parametrize(("column", "argument"), [("y", "yname"), ("cov2", "xformla")])
@pytest.mark.parametrize("reserved", [".w", "_post", "_subgroup"])
def test_ddd_2period_rejects_reserved_column_names(two_period_df, column, argument, reserved):
    with pytest.raises(ValueError, match=re.escape(f"{argument} names the column '{reserved}'")):
        ddd(
            data=two_period_df.rename({column: reserved}),
            yname=reserved if column == "y" else "y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla=f"~ cov1 + {reserved}" if column == "cov2" else "~ cov1 + cov2",
        )


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_drops_units_treated_in_first_period(mp_first_period_cohort_df, base_period):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "pname": "partition",
        "base_period": base_period,
        "est_method": "reg",
    }
    n_early = mp_first_period_cohort_df.filter(pl.col("group") == 1)["id"].n_unique()

    with pytest.warns(UserWarning, match=f"^Dropped {n_early} units that were already treated in the first period$"):
        result = ddd(data=mp_first_period_cohort_df, **spec)
    expected = ddd(data=mp_first_period_cohort_df.filter(pl.col("group") != 1), **spec)

    assert result.n == expected.n
    np.testing.assert_array_equal(result.glist, expected.glist)
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_no_never_treated_cells_match_trimmed_panel(mp_no_never_treated_df, base_period):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "pname": "partition",
        "control_group": "notyettreated",
        "base_period": base_period,
        "est_method": "reg",
    }
    result = ddd(data=mp_no_never_treated_df, **spec)
    trimmed = ddd(data=mp_no_never_treated_df.filter(pl.col("time") < 4), **spec)

    cells = {(g, t): i for i, (g, t) in enumerate(zip(result.groups, result.times))}
    rows = [cells[(g, t)] for g, t in zip(trimmed.groups, trimmed.times)]
    np.testing.assert_allclose(result.att[rows], trimmed.att, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.se[rows], trimmed.se, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_failed_comparison_warns_and_leaves_other_cells(multi_period_df, base_period):
    all_eligible = multi_period_df.with_columns(
        pl.when(pl.col("group") == 2).then(1).otherwise(pl.col("partition")).alias("partition")
    )
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "pname": "partition",
        "base_period": base_period,
        "est_method": "reg",
    }

    with pytest.warns(UserWarning, match=re.escape("Skipping comparison group 0 for ATT(2, 3) because its estimation")):
        result = ddd(data=all_eligible, **spec)
    expected = ddd(data=multi_period_df, **spec)

    assert not np.any((result.groups == 2) & (result.times >= 2))
    np.testing.assert_array_equal(result.att[result.groups == 3], expected.att[expected.groups == 3])
    np.testing.assert_array_equal(result.se[result.groups == 3], expected.se[expected.groups == 3])


@pytest.mark.parametrize("mp_recoded_never_treated_df", [99.0, float("inf")], indirect=True)
def test_ddd_mp_late_or_infinite_cohort_counts_as_never_treated(mp_recoded_never_treated_df, multi_period_df):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition"}
    result = ddd(data=mp_recoded_never_treated_df, **spec)
    expected = ddd(data=multi_period_df, **spec)

    np.testing.assert_array_equal(result.glist, expected.glist)
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    assert f"Units never enabling treatment: {np.sum(expected.unit_groups == 0)}" in str(result)


@pytest.mark.parametrize("mp_missing_outcome_df", [None, float("nan")], indirect=True)
def test_ddd_mp_drops_rows_with_missing_values(mp_missing_outcome_df, multi_period_df):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition"}
    complete = multi_period_df.with_row_index("row").filter(~pl.col("row").is_in([1, 5, 9])).drop("row")

    with pytest.warns(UserWarning, match="^Dropped 3 rows from original data due to missing values$"):
        result = ddd(data=mp_missing_outcome_df, **spec)
    expected = ddd(data=complete, **spec)

    assert np.isfinite(result.att).all()
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)


@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_ddd_mp_duplicated_unit_periods_raise(multi_period_df, allow_unbalanced_panel):
    repeated = pl.concat([multi_period_df, multi_period_df.filter((pl.col("id") == 3) & (pl.col("time") == 1))])
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (id, time) pair (3, 1)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        ddd(
            data=repeated,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            allow_unbalanced_panel=allow_unbalanced_panel,
        )


@pytest.mark.parametrize(
    ("column", "change", "message"),
    [
        ("partition", 1 - pl.col("partition"), "The value of partition must be the same across all periods"),
        ("group", pl.col("group") + 1, "The value of group must be the same across all periods"),
        ("cl", pl.col("cl") + 1, "Cluster variable must be time-invariant within units."),
        ("w", pl.col("w") + 1, "Weights must be the same across all periods for each unit."),
    ],
)
def test_ddd_mp_rejects_unit_attributes_that_change(multi_period_df, column, change, message):
    data = multi_period_df.with_columns((pl.col("id") % 7).alias("cl"), pl.lit(1.0).alias("w"))
    changed = (pl.col("id") == 3) & (pl.col("time") == 3)
    data = data.with_columns(pl.when(changed).then(change).otherwise(pl.col(column)).alias(column))

    with pytest.raises(ValueError, match=re.escape(message)):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            weightsname="w",
            cluster="cl",
            boot=True,
        )


@pytest.mark.parametrize("argument", ["pname", "cluster", "weightsname"])
def test_ddd_mp_reports_missing_columns(multi_period_df, argument):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "pname": "partition", "boot": True}

    with pytest.raises(ValueError, match=re.escape(f"{argument}='nope' not found in data.")):
        ddd(data=multi_period_df, **(spec | {argument: "nope"}))


@pytest.mark.parametrize("data_fixture, panel", [("multi_period_df", True), ("mp_rcs_data", False)])
def test_ddd_mp_partition_must_be_binary(request, data_fixture, panel):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("partition") + 1).alias("partition"))
    message = (
        "pname='partition' must be 1 for eligible units and 0 for ineligible units, but it also takes the values [2]."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id" if panel else None,
            gname="group",
            pname="partition",
            panel=panel,
        )


@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_ddd_mp_integer_weights_match_replicated_units(mp_weighted_df, mp_weighted_replicated_df, est_method):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "pname": "partition",
        "xformla": "~ cov1 + cov2",
        "est_method": est_method,
    }
    weighted = ddd(data=mp_weighted_df, weightsname="w", **spec)
    replicated = ddd(data=mp_weighted_replicated_df, **spec)
    unweighted = ddd(data=mp_weighted_df, **spec)

    np.testing.assert_allclose(weighted.att, replicated.att, rtol=1e-8, atol=1e-8)
    assert np.max(np.abs(weighted.att - unweighted.att)) > 1e-3


def test_ddd_mp_rcs_integer_weights_match_replicated_rows(mp_rcs_weighted_df, mp_rcs_weighted_replicated_df):
    spec = {"yname": "y", "tname": "time", "gname": "group", "pname": "partition", "panel": False}
    weighted = ddd(data=mp_rcs_weighted_df, weightsname="w", **spec)
    replicated = ddd(data=mp_rcs_weighted_replicated_df, **spec)

    np.testing.assert_allclose(weighted.att, replicated.att, rtol=1e-8, atol=1e-8)


def test_ddd_mp_rejects_negative_weights(mp_weighted_df):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        ddd(
            data=mp_weighted_df.with_columns(-pl.col("w")),
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            weightsname="w",
        )


@pytest.mark.parametrize(
    "data_fixture, gname, panel",
    [
        ("two_period_df", "state", True),
        ("two_period_rcs_data", "state", False),
        ("multi_period_df", "group", True),
        ("mp_rcs_data", "group", False),
    ],
)
def test_ddd_rejects_left_hand_side_in_xformla(request, data_fixture, gname, panel):
    data = request.getfixturevalue(data_fixture)

    with pytest.raises(ValueError, match=re.escape("xformla='y ~ 1' has a left-hand side.")):
        ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id" if panel else None,
            gname=gname,
            pname="partition",
            xformla="y ~ 1",
            panel=panel,
        )
