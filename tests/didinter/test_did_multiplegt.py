"""Tests for the did_multiplegt main entry point."""

import re

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

import moderndid.didinter.compute_did_multiplegt as compute_module
from moderndid import did_multiplegt
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.didinter import ATEResult, DIDInterResult, EffectsResult, PlacebosResult
from moderndid.didinter.bootstrap import cluster_bootstrap
from moderndid.didinter.container import BootstrapResult


def test_basic_estimation(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=1,
    )

    assert isinstance(result, DIDInterResult)
    assert isinstance(result.effects, EffectsResult)
    assert len(result.effects.horizons) == 1
    assert len(result.effects.estimates) == 1
    assert result.effects.std_errors[0] > 0
    assert result.n_switchers > 0
    assert result.n_units > 0


def test_multiple_effects_horizons(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
    )

    assert len(result.effects.horizons) == 3
    np.testing.assert_array_equal(result.effects.horizons, [1, 2, 3])
    assert all(se > 0 or np.isnan(se) for se in result.effects.std_errors)


def test_with_placebos(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    assert result.placebos is not None
    assert isinstance(result.placebos, PlacebosResult)
    assert len(result.placebos.horizons) == 2
    np.testing.assert_array_equal(result.placebos.horizons, [-1, -2])


def test_ate_computation(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert result.ate is not None
    assert isinstance(result.ate, ATEResult)
    assert isinstance(result.ate.estimate, float)
    assert result.ate.std_error > 0
    assert result.ate.ci_lower < result.ate.ci_upper


def test_normalized_effects(simple_panel_data):
    result_unnorm = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        normalized=False,
    )

    result_norm = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        normalized=True,
    )

    assert not np.allclose(result_norm.effects.estimates, result_unnorm.effects.estimates, rtol=0.01)


@pytest.mark.parametrize("switchers", ["", "in"])
def test_switcher_types(simple_panel_data, switchers):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        switchers=switchers,
    )

    assert result.n_switchers > 0


def test_same_switchers_option(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        same_switchers=True,
    )

    n_switchers = result.effects.n_switchers
    unique_counts = np.unique(n_switchers[~np.isnan(n_switchers)])
    if len(unique_counts) > 0:
        assert len(unique_counts) == 1


def test_only_never_switchers_control(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        only_never_switchers=True,
    )

    assert isinstance(result, DIDInterResult)
    assert result.n_never_switchers > 0


def test_with_weights(weighted_panel_data):
    result = did_multiplegt(
        weighted_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        weightsname="w",
    )

    assert isinstance(result, DIDInterResult)
    assert result.estimation_params.get("weightsname") == "w"


def test_with_clustering(clustered_panel_data):
    result = did_multiplegt(
        clustered_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        cluster="cluster",
    )

    assert isinstance(result, DIDInterResult)
    assert result.estimation_params.get("cluster") == "cluster"
    assert result.effects.std_errors[0] > 0


def test_with_controls(panel_with_controls):
    result = did_multiplegt(
        panel_with_controls,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        xformla="~ x1 + x2",
    )

    assert isinstance(result, DIDInterResult)
    assert result.estimation_params.get("xformla") == "~ x1 + x2"


@pytest.mark.parametrize(
    "test_key",
    ["chi2_stat", "p_value"],
)
def test_effects_equal_test_keys(simple_panel_data, test_key):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
    )

    assert result.effects_equal_test is not None
    assert test_key in result.effects_equal_test


def test_effects_equal_test_p_value_range(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
    )

    assert 0 <= result.effects_equal_test["p_value"] <= 1


@pytest.mark.parametrize(
    "test_key",
    ["chi2_stat", "p_value"],
)
def test_placebo_joint_test_keys(simple_panel_data, test_key):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    if result.placebo_joint_test is not None:
        assert test_key in result.placebo_joint_test


def test_placebo_joint_test_p_value_range(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    if result.placebo_joint_test is not None:
        assert 0 <= result.placebo_joint_test["p_value"] <= 1


@pytest.mark.parametrize("ci_level", [90.0, 95.0, 99.0])
def test_confidence_interval_levels(simple_panel_data, ci_level):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        ci_level=ci_level,
    )

    assert result.ci_level == ci_level
    assert np.all(result.effects.ci_lower <= result.effects.estimates)
    assert np.all(result.effects.ci_upper >= result.effects.estimates)


def test_influence_functions_returned(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=1,
    )

    assert result.influence_effects is not None
    assert result.influence_effects.shape[1] == 2
    if result.influence_placebos is not None:
        assert result.influence_placebos.shape[1] == 1


@pytest.mark.parametrize(
    "param_key,expected_value",
    [
        ("effects", 3),
        ("placebo", 2),
        ("normalized", True),
        ("same_switchers", True),
        ("boot", False),
    ],
)
def test_estimation_params_stored(simple_panel_data, param_key, expected_value):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        placebo=2,
        normalized=True,
        same_switchers=True,
    )

    assert result.estimation_params[param_key] == expected_value


@pytest.mark.filterwarnings("ignore:Keeping bidirectional switchers:UserWarning")
@pytest.mark.parametrize("keep_bidirectional", [True, False])
def test_bidirectional_switchers_handling(bidirectional_panel_data, keep_bidirectional):
    result = did_multiplegt(
        bidirectional_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        keep_bidirectional_switchers=keep_bidirectional,
    )

    assert isinstance(result, DIDInterResult)


def test_unbalanced_panel(unbalanced_panel_data):
    result = did_multiplegt(
        unbalanced_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert isinstance(result, DIDInterResult)


def test_result_counts(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert result.n_units == result.n_switchers + result.n_never_switchers
    assert result.n_switchers >= 0
    assert result.n_never_switchers >= 0


def test_n_switchers_per_horizon(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
    )

    assert len(result.effects.n_switchers) == 3
    assert len(result.effects.n_observations) == 3
    assert np.all(result.effects.n_switchers >= 0)
    assert np.all(result.effects.n_observations >= 0)


def test_real_data_basic(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
    )

    assert isinstance(result, DIDInterResult)
    assert result.n_switchers > 0
    assert result.n_never_switchers > 0
    assert len(result.effects.estimates) == 2


def test_real_data_with_cluster(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        cluster="state_n",
    )

    assert isinstance(result, DIDInterResult)
    assert all(se > 0 for se in result.effects.std_errors)


def test_real_data_normalized(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        normalized=True,
    )

    assert isinstance(result, DIDInterResult)
    assert len(result.effects.estimates) == 3


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
@pytest.mark.parametrize("placebo", [0, 1])
def test_bootstrap_standard_errors(simple_panel_data, placebo):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=placebo,
        boot=True,
        biters=50,
        random_state=42,
    )

    assert isinstance(result, DIDInterResult)
    assert all(se > 0 or np.isnan(se) for se in result.effects.std_errors)
    if placebo > 0 and result.placebos is not None:
        assert all(se > 0 or np.isnan(se) for se in result.placebos.std_errors)


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_bootstrap_with_cluster(favara_imbs_data):
    result = did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        cluster="state_n",
        boot=True,
        biters=50,
        random_state=42,
    )

    assert isinstance(result, DIDInterResult)
    assert all(se > 0 for se in result.effects.std_errors)


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_bootstrap_reproducibility(simple_panel_data):
    kwargs = {
        "yname": "y",
        "idname": "id",
        "tname": "time",
        "dname": "d",
        "effects": 2,
        "boot": True,
        "biters": 50,
        "random_state": 123,
    }
    result1 = did_multiplegt(simple_panel_data, **kwargs)
    result2 = did_multiplegt(simple_panel_data, **kwargs)

    np.testing.assert_array_almost_equal(
        result1.effects.std_errors,
        result2.effects.std_errors,
    )


@pytest.mark.parametrize(
    "kwargs,expected_warning",
    [
        ({"continuous": 1}, "continuous.*Bootstrap inference"),
        ({"trends_lin": True}, "trends_lin.*ATE.*not computed"),
        ({"keep_bidirectional_switchers": True}, "bidirectional switchers"),
    ],
)
def test_warnings_for_options(simple_panel_data, kwargs, expected_warning):
    with pytest.warns(UserWarning, match=expected_warning):
        did_multiplegt(
            simple_panel_data,
            yname="y",
            idname="id",
            tname="time",
            dname="d",
            effects=1,
            **kwargs,
        )


def test_clustered_ate_standard_error_sums_influence_within_clusters(clustered_panel_data):
    result = did_multiplegt(
        clustered_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        cluster="cluster",
    )

    weights = result.effects.n_switchers / result.effects.n_switchers.sum()
    denominator = weights @ result.effects.estimates / result.ate.estimate
    ate_influence = result.influence_effects @ weights / denominator
    cluster_sums = np.bincount(np.arange(50) // 10, weights=ate_influence)

    np.testing.assert_allclose(result.ate.std_error, np.sqrt(np.sum(cluster_sums**2)) / 50, rtol=1e-12)


def test_clustered_placebo_joint_test_uses_uncentered_cluster_sums(clustered_panel_data):
    result = did_multiplegt(
        clustered_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
        cluster="cluster",
    )

    clusters = np.arange(50) // 10
    sums = np.column_stack([np.bincount(clusters, weights=col) for col in result.influence_placebos.T])
    vcov = sums.T @ sums / 50**2
    estimates = result.placebos.estimates

    np.testing.assert_allclose(np.sqrt(np.diag(vcov)), result.placebos.std_errors, rtol=1e-12)
    np.testing.assert_allclose(
        result.placebo_joint_test["chi2_stat"], estimates @ np.linalg.solve(vcov, estimates), rtol=1e-10
    )


def test_placebo_joint_test_uses_uncentered_covariance(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        placebo=2,
    )

    vcov = result.influence_placebos.T @ result.influence_placebos / 50**2
    estimates = result.placebos.estimates

    np.testing.assert_allclose(
        result.placebo_joint_test["chi2_stat"], estimates @ np.linalg.solve(vcov, estimates), rtol=1e-10
    )


def test_clustered_effects_equal_test_uses_cluster_sums(clustered_panel_data):
    result = did_multiplegt(
        clustered_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=3,
        effects_equal=True,
        cluster="cluster",
    )

    sums = np.column_stack([np.bincount(np.arange(50) // 10, weights=col) for col in result.influence_effects.T])
    vcov = sums.T @ sums / 50**2
    contrast = np.eye(2, 3) - np.ones((2, 3)) / 3
    diff = contrast @ result.effects.estimates

    np.testing.assert_allclose(
        result.effects_equal_test["chi2_stat"],
        diff @ np.linalg.solve(contrast @ vcov @ contrast.T, diff),
        rtol=1e-10,
    )


def test_clustered_se_keeps_groups_missing_their_first_period(clustered_panel_data):
    kwargs = {
        "yname": "y",
        "idname": "id",
        "tname": "time",
        "dname": "d",
        "effects": 2,
        "placebo": 1,
        "cluster": "cluster",
    }
    first_row = (pl.col("id") == 25) & (pl.col("time") == 1)
    dropped = clustered_panel_data.filter(~first_row)
    missing = clustered_panel_data.with_columns(pl.when(first_row).then(None).otherwise(pl.col("y")).alias("y"))

    result_dropped = did_multiplegt(dropped, **kwargs)
    result_missing = did_multiplegt(missing, **kwargs)

    np.testing.assert_allclose(result_dropped.effects.std_errors, result_missing.effects.std_errors, rtol=1e-12)
    np.testing.assert_allclose(result_dropped.placebos.std_errors, result_missing.placebos.std_errors, rtol=1e-12)
    np.testing.assert_allclose(result_dropped.ate.std_error, result_missing.ate.std_error, rtol=1e-12)


def test_clustering_leaves_heterogeneity_covariates_as_observed(favara_imbs_data):
    kwargs = {
        "yname": "Dl_vloans_b",
        "idname": "county",
        "tname": "year",
        "dname": "inter_bra",
        "effects": 2,
        "predict_het": (["state_n"], [-1]),
    }

    clustered = did_multiplegt(favara_imbs_data, **kwargs, cluster="state_n")
    unclustered = did_multiplegt(favara_imbs_data, **kwargs)

    for het_clustered, het_unclustered in zip(clustered.heterogeneity, unclustered.heterogeneity, strict=True):
        assert het_clustered.n_obs == het_unclustered.n_obs
        np.testing.assert_allclose(het_clustered.estimates, het_unclustered.estimates, rtol=1e-12)


def test_placebo_switchers_need_the_outcome_of_the_matching_effect(simple_panel_data):
    late_outcome = (pl.col("id") >= 20) & (pl.col("id") < 30) & (pl.col("time") == 4)
    data = simple_panel_data.with_columns(pl.when(late_outcome).then(None).otherwise(pl.col("y")).alias("y"))

    result = did_multiplegt(
        data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=1,
        placebo=1,
    )

    assert result.effects.n_switchers[0] == 20
    assert result.placebos.n_switchers[0] == 20
    assert np.all(np.isfinite(result.placebos.std_errors))


def test_ate_counts_each_cell_once(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
    )

    assert result.effects.n_observations.tolist() == [80.0, 70.0]
    assert result.ate.n_observations == 130
    assert result.ate.n_switchers == 60


def test_ate_switcher_count_is_unweighted(weighted_panel_data):
    result = did_multiplegt(
        weighted_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        weightsname="w",
        effects=2,
    )

    assert result.ate.n_switchers == result.effects.n_switchers.sum()


@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
def test_trends_lin_reports_no_ate(simple_panel_data):
    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        trends_lin=True,
    )

    assert result.ate is None


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_bootstrap_reports_bootstrap_ate_standard_error(simple_panel_data, fake_bootstrap, monkeypatch):
    monkeypatch.setattr(compute_module, "cluster_bootstrap", lambda **kwargs: fake_bootstrap)

    result = did_multiplegt(
        simple_panel_data,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        effects=2,
        boot=True,
    )

    assert result.ate.std_error == 0.25
    np.testing.assert_allclose(result.ate.estimate - result.ate.ci_lower, 1.959963984540054 * 0.25, rtol=1e-12)
    np.testing.assert_allclose(result.ate.ci_upper - result.ate.estimate, 1.959963984540054 * 0.25, rtol=1e-12)
    np.testing.assert_array_equal(result.effects.std_errors, [0.5, 0.6])


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_bootstrap_results_label_their_standard_errors(simple_panel_data, fake_bootstrap, monkeypatch):
    monkeypatch.setattr(compute_module, "cluster_bootstrap", lambda **kwargs: fake_bootstrap)

    result = did_multiplegt(simple_panel_data, yname="y", idname="id", tname="time", dname="d", effects=2, boot=True)

    assert result.estimation_params["boot"] is True
    assert " Standard errors: Bootstrap" in str(result)
    assert "Analytical" not in str(result)
    assert result.__maketables_stat__("se_type") == "Bootstrap"
    assert result.__maketables_vcov_info__ == {"vcov_type": "bootstrap", "clustervar": None}


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_bootstrap_matches_fits_on_relabelled_cluster_copies(clustered_panel_data, fixed_draws, relabel_cluster_copies):
    draws = [[0, 0, 2, 3, 3], [1, 2, 2, 4, 4], [0, 1, 1, 3, 4]]
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "effects": 2, "placebo": 1}

    result = did_multiplegt(
        clustered_panel_data, **kwargs, cluster="cluster", boot=True, biters=3, random_state=fixed_draws(draws)
    )
    fits = [
        did_multiplegt(relabel_cluster_copies(clustered_panel_data, draw, "cluster", "id"), **kwargs) for draw in draws
    ]

    np.testing.assert_allclose(
        result.effects.std_errors, np.std([fit.effects.estimates for fit in fits], axis=0, ddof=1), rtol=1e-10
    )
    np.testing.assert_allclose(
        result.placebos.std_errors, np.std([fit.placebos.estimates for fit in fits], axis=0, ddof=1), rtol=1e-10
    )
    np.testing.assert_allclose(result.ate.std_error, np.std([fit.ate.estimate for fit in fits], ddof=1), rtol=1e-10)


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
@pytest.mark.filterwarnings("ignore:Requested effects=4:UserWarning")
def test_bootstrap_draw_estimates_the_horizons_its_switchers_reach(
    clustered_panel_data, fixed_draws, relabel_cluster_copies
):
    draws = [[2, 2, 3, 4, 4], [2, 3, 3, 3, 4]]
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "effects": 4, "trends_lin": True}

    result = did_multiplegt(
        clustered_panel_data, **kwargs, cluster="cluster", boot=True, biters=2, random_state=fixed_draws(draws)
    )
    fits = [
        did_multiplegt(relabel_cluster_copies(clustered_panel_data, draw, "cluster", "id"), **kwargs) for draw in draws
    ]

    np.testing.assert_allclose(
        result.effects.std_errors[:3], np.std([fit.effects.estimates for fit in fits], axis=0, ddof=1), rtol=1e-10
    )
    assert np.isnan(result.effects.std_errors[3])


def test_cluster_bootstrap_gives_each_cluster_copy_new_group_ids(clustered_panel_data, fixed_draws, draw_recorder):
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d", cluster="cluster")

    result = cluster_bootstrap(
        clustered_panel_data, config, draw_recorder, biters=1, random_state=fixed_draws([[0, 0, 2, 3, 3]])
    )
    drawn = draw_recorder.frames[0]
    first_cluster = clustered_panel_data.filter(pl.col("cluster") == 0)["y"].to_numpy()

    assert isinstance(result, BootstrapResult)
    assert drawn.height == 300
    assert drawn["id"].n_unique() == 50
    assert drawn.group_by("cluster").agg(pl.col("id").n_unique()).sort("cluster")["id"].to_list() == [20, 10, 20]
    np.testing.assert_array_equal(
        np.sort(drawn.filter(pl.col("cluster") == 0)["y"].to_numpy()), np.sort(np.tile(first_cluster, 2))
    )


def test_rows_with_a_missing_cluster_are_dropped_with_a_warning(clustered_panel_data):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "effects": 2, "placebo": 1}
    missing = ((pl.col("id") < 3) & (pl.col("time") == 2)) | (pl.col("id") == 40)
    blanked = clustered_panel_data.with_columns(
        pl.when(missing).then(None).otherwise(pl.col("cluster")).alias("cluster")
    )

    with pytest.warns(UserWarning, match="Dropped 9 rows from original data due to a missing cluster in 'cluster'"):
        result = did_multiplegt(blanked, **kwargs, cluster="cluster")
    expected = did_multiplegt(clustered_panel_data.filter(~missing), **kwargs, cluster="cluster")

    assert result.n_units == expected.n_units == 49
    np.testing.assert_array_equal(result.effects.estimates, expected.effects.estimates)
    np.testing.assert_array_equal(result.effects.std_errors, expected.effects.std_errors)
    np.testing.assert_array_equal(result.placebos.estimates, expected.placebos.estimates)
    np.testing.assert_array_equal(result.ate.std_error, expected.ate.std_error)


def test_groups_in_more_than_one_cluster_raise(clustered_panel_data):
    moved = clustered_panel_data.with_columns(
        pl.when((pl.col("id") == 5) & (pl.col("time") >= 4)).then(3).otherwise(pl.col("cluster")).alias("cluster")
    )

    with pytest.raises(ValueError, match="Some groups belong to more than one cluster in 'cluster'"):
        did_multiplegt(moved, yname="y", idname="id", tname="time", dname="d", cluster="cluster")


def test_clustering_by_the_group_column_matches_no_clustering(clustered_panel_data):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "effects": 2, "placebo": 1}

    by_group = did_multiplegt(clustered_panel_data, **kwargs, cluster="id")
    unclustered = did_multiplegt(clustered_panel_data, **kwargs)

    np.testing.assert_allclose(by_group.effects.std_errors, unclustered.effects.std_errors, rtol=1e-12)
    np.testing.assert_allclose(by_group.placebos.std_errors, unclustered.placebos.std_errors, rtol=1e-12)
    np.testing.assert_allclose(by_group.ate.std_error, unclustered.ate.std_error, rtol=1e-12)


def test_cluster_bootstrap_leaves_out_rows_without_a_cluster(clustered_panel_data, fixed_draws, draw_recorder):
    missing = ((pl.col("id") < 3) & (pl.col("time") == 2)) | (pl.col("id") == 40)
    data = clustered_panel_data.with_columns(pl.when(missing).then(None).otherwise(pl.col("cluster")).alias("cluster"))
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d", cluster="cluster")

    cluster_bootstrap(data, config, draw_recorder, biters=1, random_state=fixed_draws([[0, 1, 2, 3, 4]]))
    drawn = draw_recorder.frames[0]

    assert drawn.height == 291
    assert drawn["cluster"].null_count() == 0
    assert drawn["id"].n_unique() == 49


def test_cluster_bootstrap_keeps_a_cluster_without_usable_rows_in_the_pool(
    clustered_panel_data, fixed_draws, draw_recorder
):
    data = clustered_panel_data.with_columns(
        pl.when(pl.col("cluster") == 4).then(None).otherwise(pl.col("y")).alias("y")
    )
    config = DIDInterConfig(yname="y", tname="time", gname="id", dname="d", cluster="cluster")
    draws = [[4, 4, 4, 4, 4], [0, 1, 2, 3, 4]]

    cluster_bootstrap(data, config, draw_recorder, biters=2, random_state=fixed_draws(draws))

    assert [frame.height for frame in draw_recorder.frames] == [0, 240]


def test_bootstrap_warns_once_about_the_rows_it_drops(panel_with_controls):
    data = panel_with_controls.with_columns(
        pl.when((pl.col("time") == 2) & (pl.col("id") < 10)).then(None).otherwise(pl.col("x1")).alias("x1"),
        pl.when(pl.col("id") == 45).then(None).otherwise(pl.col("id") // 10).alias("cluster"),
    )

    with pytest.warns(UserWarning) as caught:
        did_multiplegt(
            data,
            yname="y",
            idname="id",
            tname="time",
            dname="d",
            xformla="~ x1",
            cluster="cluster",
            boot=True,
            biters=5,
            random_state=0,
        )
    dropped = [str(w.message) for w in caught if str(w.message).startswith("Dropped")]

    assert dropped == [
        "Dropped 10 rows from original data due to missing covariates",
        "Dropped 6 rows from original data due to a missing cluster in 'cluster'",
    ]


def test_placebos_cannot_outnumber_the_effects(simple_panel_data):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "effects": 1}

    with pytest.warns(UserWarning, match="Requested placebo=2 but the number of placebos cannot exceed the number"):
        result = did_multiplegt(simple_panel_data, **kwargs, placebo=2, same_switchers=True, same_switchers_pl=True)
    expected = did_multiplegt(simple_panel_data, **kwargs, placebo=1, same_switchers=True, same_switchers_pl=True)

    assert result.estimation_params["placebo"] == 1
    np.testing.assert_array_equal(result.placebos.estimates, expected.placebos.estimates)
    np.testing.assert_array_equal(result.placebos.n_switchers, expected.placebos.n_switchers)


@pytest.mark.parametrize(("switchers", "directions"), [("", [1, -1]), ("in", [1]), ("out", [-1])])
def test_first_effect_matches_hand_computation_with_not_yet_switched_controls(
    two_way_panel_data, first_effect_by_hand, switchers, directions
):
    result = did_multiplegt(
        two_way_panel_data, yname="y", idname="id", tname="time", dname="d", effects=1, switchers=switchers
    )

    np.testing.assert_allclose(
        result.effects.estimates[0], first_effect_by_hand(two_way_panel_data, directions), rtol=1e-12
    )


def test_switchers_out_measure_the_effect_with_its_sign(two_way_panel_data):
    result = did_multiplegt(
        two_way_panel_data, yname="y", idname="id", tname="time", dname="d", effects=2, switchers="out"
    )

    assert np.all(result.effects.estimates > 0.5)
    np.testing.assert_array_equal(result.effects.n_switchers, [12, 12])


@pytest.mark.parametrize("cluster", [None, "cluster"])
def test_pooled_estimates_combine_both_directions_by_switcher_counts(two_way_panel_data, cluster):
    kwargs = dict(yname="y", idname="id", tname="time", dname="d", effects=2, placebo=1, cluster=cluster)
    pooled = did_multiplegt(two_way_panel_data, **kwargs)
    rises = did_multiplegt(two_way_panel_data, **kwargs, switchers="in")
    falls = did_multiplegt(two_way_panel_data, **kwargs, switchers="out")
    n_rise, n_fall = rises.effects.n_switchers, falls.effects.n_switchers
    influence = (n_rise * rises.influence_effects + n_fall * falls.influence_effects) / (n_rise + n_fall)
    pl_rise, pl_fall = rises.placebos.n_switchers, falls.placebos.n_switchers

    np.testing.assert_allclose(
        pooled.effects.estimates,
        (n_rise * rises.effects.estimates + n_fall * falls.effects.estimates) / (n_rise + n_fall),
        rtol=1e-12,
    )
    np.testing.assert_allclose(pooled.influence_effects, influence, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        pooled.placebos.estimates,
        (pl_rise * rises.placebos.estimates + pl_fall * falls.placebos.estimates) / (pl_rise + pl_fall),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(pooled.effects.n_switchers, n_rise + n_fall)
    np.testing.assert_array_less(
        pooled.effects.n_observations, rises.effects.n_observations + falls.effects.n_observations
    )
    assert (pooled.n_switchers, rises.n_switchers, falls.n_switchers) == (40, 28, 12)
    assert pooled.n_units == rises.n_units == falls.n_units == 60


@pytest.mark.parametrize(("switchers", "directions"), [("", [1, -1]), ("in", [1]), ("out", [-1])])
def test_sample_sizes_count_each_cell_once(two_way_panel_data, cells_used_by_hand, switchers, directions):
    result = did_multiplegt(
        two_way_panel_data, yname="y", idname="id", tname="time", dname="d", effects=2, placebo=2, switchers=switchers
    )
    effect_cells = [cells_used_by_hand(two_way_panel_data, h, directions) for h in (1, 2)]
    placebo_cells = [cells_used_by_hand(two_way_panel_data, h, directions, placebo=True) for h in (1, 2)]

    np.testing.assert_array_equal(result.effects.n_observations, effect_cells)
    np.testing.assert_array_equal(result.placebos.n_observations, placebo_cells)


def test_switchers_out_without_decreases_raises(simple_panel_data):
    with pytest.raises(ValueError, match="no switching group whose treatment decreases"):
        did_multiplegt(simple_panel_data, yname="y", idname="id", tname="time", dname="d", switchers="out")


@pytest.mark.filterwarnings("error")
def test_groups_that_switch_later_are_controls_when_no_group_never_switches(
    all_switch_panel_data, first_effect_by_hand
):
    result = did_multiplegt(all_switch_panel_data, yname="y", idname="id", tname="time", dname="d", effects=2)

    np.testing.assert_allclose(
        result.effects.estimates[0], first_effect_by_hand(all_switch_panel_data, [1, -1]), rtol=1e-12
    )
    np.testing.assert_array_equal(result.effects.n_switchers, [45, 15])
    assert result.n_never_switchers == 0


@pytest.mark.parametrize(("switchers", "n_effects"), [("", 2), ("in", 2), ("out", 1)])
def test_requested_horizons_stop_where_the_controls_run_out(all_switch_panel_data, switchers, n_effects):
    kwargs = dict(yname="y", idname="id", tname="time", dname="d", effects=3, placebo=3, switchers=switchers)

    with (
        pytest.warns(UserWarning, match=f"effects can only be estimated up to horizon {n_effects}"),
        pytest.warns(UserWarning, match="placebos can only be estimated up to horizon 1"),
    ):
        result = did_multiplegt(all_switch_panel_data, **kwargs)

    np.testing.assert_array_equal(result.effects.horizons, np.arange(1, n_effects + 1))
    np.testing.assert_array_equal(result.placebos.horizons, [-1])
    assert np.all(np.isfinite(result.effects.std_errors))
    assert np.isfinite(result.placebos.std_errors[0])


@pytest.mark.parametrize(
    "coding",
    [
        pl.col("time") - 1,
        2000 + 2 * pl.col("time"),
        pl.col("time").replace_strict({1: 3, 2: 4, 3: 9, 4: 10, 5: 17, 6: 30}),
    ],
    ids=["zero-based", "spacing-two", "irregular-gaps"],
)
@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
@pytest.mark.parametrize("extra", [{}, {"trends_lin": True}, {"cluster": "cluster", "normalized": True}])
def test_estimates_do_not_depend_on_how_periods_are_coded(two_way_panel_data, coding, extra):
    kwargs = dict(yname="y", idname="id", tname="time", dname="d", effects=2, placebo=1, **extra)
    base = did_multiplegt(two_way_panel_data, **kwargs)
    recoded = did_multiplegt(two_way_panel_data.with_columns(coding.alias("time")), **kwargs)

    np.testing.assert_allclose(recoded.effects.estimates, base.effects.estimates, rtol=1e-12)
    np.testing.assert_allclose(recoded.effects.std_errors, base.effects.std_errors, rtol=1e-12)
    np.testing.assert_allclose(recoded.placebos.estimates, base.placebos.estimates, rtol=1e-12)
    np.testing.assert_array_equal(recoded.effects.n_switchers, base.effects.n_switchers)


def test_ate_divides_by_the_average_treatment_change_at_each_horizon(dose_panel_data):
    result = did_multiplegt(dose_panel_data, yname="y", idname="id", tname="time", dname="d", effects=2)

    weights = np.array([2.0, 1.0]) / 3.0
    np.testing.assert_array_equal(result.effects.n_switchers, [6, 3])
    np.testing.assert_allclose(
        result.ate.estimate, weights @ result.effects.estimates / (weights @ [1.5, 1.0]), rtol=1e-12
    )


@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_trends_lin_sums_effects_over_the_switchers_that_reach_the_horizon(
    trends_panel_data, first_differences, trend_switchers_by_hand, horizon
):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d"}
    result = did_multiplegt(trends_panel_data, **kwargs, effects=3, placebo=2, trends_lin=True)
    same = did_multiplegt(first_differences(trends_panel_data), **kwargs, effects=horizon, same_switchers=True)
    influence = same.influence_effects.sum(axis=1)

    np.testing.assert_allclose(result.effects.estimates[horizon - 1], same.effects.estimates.sum(), rtol=1e-12)
    np.testing.assert_allclose(result.effects.std_errors[horizon - 1], np.sqrt(influence @ influence) / influence.size)
    assert result.effects.n_switchers[horizon - 1] == trend_switchers_by_hand(trends_panel_data, horizon)


@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
@pytest.mark.parametrize("horizon", [1, 2])
def test_trends_lin_sums_placebos_over_the_switchers_that_reach_the_placebo(
    trends_panel_data, first_differences, horizon
):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d"}
    result = did_multiplegt(trends_panel_data, **kwargs, effects=3, placebo=2, trends_lin=True)
    same = did_multiplegt(
        first_differences(trends_panel_data),
        **kwargs,
        effects=horizon,
        placebo=horizon,
        same_switchers=True,
        same_switchers_pl=True,
    )
    influence = same.influence_placebos.sum(axis=1)

    np.testing.assert_allclose(result.placebos.estimates[horizon - 1], same.placebos.estimates.sum(), rtol=1e-12)
    np.testing.assert_allclose(result.placebos.std_errors[horizon - 1], np.sqrt(influence @ influence) / influence.size)


@pytest.mark.filterwarnings("ignore:When trends_lin=True:UserWarning")
def test_trends_lin_leaves_differences_across_absent_periods_missing(trends_panel_data):
    gaps = (pl.col("id") < 24) & (
        ((pl.col("id") % 4 == 3) & (pl.col("time") == 4))
        | ((pl.col("id") % 4 == 0) & (pl.col("time") == 5))
        | ((pl.col("id") % 4 == 1) & (pl.col("time") == 6))
    )
    kwargs = {
        "yname": "y",
        "idname": "id",
        "tname": "time",
        "dname": "d",
        "effects": 3,
        "placebo": 2,
        "trends_lin": True,
    }
    absent = did_multiplegt(trends_panel_data.filter(~gaps), **kwargs)
    blank = did_multiplegt(
        trends_panel_data.with_columns(pl.when(gaps).then(None).otherwise(pl.col("y")).alias("y")), **kwargs
    )

    np.testing.assert_allclose(absent.effects.estimates, blank.effects.estimates, rtol=1e-12)
    np.testing.assert_allclose(absent.effects.std_errors, blank.effects.std_errors, rtol=1e-12)
    np.testing.assert_allclose(absent.placebos.estimates, blank.placebos.estimates, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:When continuous > 0:UserWarning")
@pytest.mark.parametrize("degree", [1, 2])
def test_continuous_compares_all_groups_with_period_by_baseline_controls(
    continuous_panel_data, baseline_trend_controls, degree
):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "effects": 3, "placebo": 2}
    result = did_multiplegt(continuous_panel_data, dname="d", continuous=degree, **kwargs)
    data, formula = baseline_trend_controls(continuous_panel_data, degree)
    expected = did_multiplegt(data, dname="d_change", xformla=formula, **kwargs)

    np.testing.assert_allclose(result.effects.estimates, expected.effects.estimates, rtol=1e-12)
    np.testing.assert_allclose(result.effects.std_errors, expected.effects.std_errors, rtol=1e-12)
    np.testing.assert_allclose(result.placebos.estimates, expected.placebos.estimates, rtol=1e-12)
    np.testing.assert_allclose(result.placebos.std_errors, expected.placebos.std_errors, rtol=1e-12)
    np.testing.assert_allclose(result.ate.estimate, expected.ate.estimate, rtol=1e-12)
    np.testing.assert_allclose(result.ate.std_error, expected.ate.std_error, rtol=1e-12)
    np.testing.assert_array_equal(result.effects.n_switchers, [90, 90, 60])


def test_same_switchers_pl_restricts_only_the_placebos(simple_panel_data):
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "effects": 2, "placebo": 2}
    same = did_multiplegt(simple_panel_data, **kwargs, same_switchers=True)
    same_pl = did_multiplegt(simple_panel_data, **kwargs, same_switchers=True, same_switchers_pl=True)

    np.testing.assert_array_equal(same_pl.effects.estimates, same.effects.estimates)
    np.testing.assert_array_equal(same_pl.effects.n_switchers, [30, 30])
    np.testing.assert_array_equal(same.placebos.n_switchers, [30, 10])
    np.testing.assert_array_equal(same_pl.placebos.n_switchers, [10, 10])


@pytest.mark.parametrize("options", [{}, {"xformla": "~ x"}, {"normalized": True}])
def test_shifting_every_treatment_by_a_constant_leaves_the_estimates_unchanged(baseline_shift_panels, options):
    kwargs = {"yname": "y", "idname": "g", "tname": "t", "dname": "d", "effects": 3, "placebo": 1, "cluster": "cl"}
    tenth = did_multiplegt(baseline_shift_panels[0], **kwargs, **options)
    eighth = did_multiplegt(baseline_shift_panels[1], **kwargs, **options)

    np.testing.assert_array_equal(tenth.effects.n_switchers, [105, 91, 72])
    np.testing.assert_array_equal(tenth.effects.n_switchers, eighth.effects.n_switchers)
    np.testing.assert_array_equal(tenth.effects.n_observations, eighth.effects.n_observations)
    np.testing.assert_allclose(tenth.effects.estimates, eighth.effects.estimates, rtol=1e-12)
    np.testing.assert_allclose(tenth.effects.std_errors, eighth.effects.std_errors, rtol=1e-12)
    np.testing.assert_allclose(tenth.placebos.estimates, eighth.placebos.estimates, rtol=1e-12)
    np.testing.assert_allclose(tenth.placebos.std_errors, eighth.placebos.std_errors, rtol=1e-12)
    np.testing.assert_allclose(tenth.ate.estimate, eighth.ate.estimate, rtol=1e-12)
    np.testing.assert_allclose(tenth.ate.std_error, eighth.ate.std_error, rtol=1e-12)


def test_outcome_named_weights_matches_baseline(simple_panel_data):
    kwargs = {"idname": "id", "tname": "time", "dname": "d", "effects": 2, "placebo": 1}
    renamed = did_multiplegt(simple_panel_data.rename({"y": "weights"}), yname="weights", **kwargs)
    expected = did_multiplegt(simple_panel_data, yname="y", **kwargs)

    np.testing.assert_array_equal(renamed.effects.estimates, expected.effects.estimates)
    np.testing.assert_array_equal(renamed.effects.std_errors, expected.effects.std_errors)
    np.testing.assert_array_equal(renamed.placebos.estimates, expected.placebos.estimates)


@pytest.mark.parametrize("column", ["y", "d", "w"])
def test_infinite_outcome_treatment_or_weight_counts_as_missing(weighted_clustered_panel, column):
    row = (pl.col("id") == 35) & (pl.col("time") == 4)
    kwargs = {
        "yname": "y",
        "idname": "id",
        "tname": "time",
        "dname": "d",
        "weightsname": "w",
        "effects": 2,
        "placebo": 1,
    }
    missing = weighted_clustered_panel.with_columns(pl.when(row).then(None).otherwise(pl.col(column)).alias(column))
    infinite = weighted_clustered_panel.with_columns(
        pl.when(row).then(float("inf")).otherwise(pl.col(column)).alias(column)
    )

    expected = did_multiplegt(missing, **kwargs)
    result = did_multiplegt(infinite, **kwargs)

    np.testing.assert_array_equal(result.effects.estimates, expected.effects.estimates)
    np.testing.assert_array_equal(result.effects.std_errors, expected.effects.std_errors)
    np.testing.assert_array_equal(result.placebos.estimates, expected.placebos.estimates)


def test_rows_with_an_infinite_control_are_dropped(weighted_clustered_panel):
    row = (pl.col("id") == 35) & (pl.col("time") == 4)
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "xformla": "~ x1 + x2", "effects": 2}
    infinite = weighted_clustered_panel.with_columns(
        pl.when(row).then(float("-inf")).otherwise(pl.col("x1")).alias("x1")
    )

    expected = did_multiplegt(weighted_clustered_panel.filter(~row), **kwargs)
    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing covariates$"):
        result = did_multiplegt(infinite, **kwargs)

    np.testing.assert_array_equal(result.effects.estimates, expected.effects.estimates)
    np.testing.assert_array_equal(result.effects.std_errors, expected.effects.std_errors)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("boot", [False, True])
def test_non_finite_cluster_counts_as_missing(weighted_clustered_panel, value, boot):
    group = pl.col("id") == 1
    kwargs = {"yname": "y", "idname": "id", "tname": "time", "dname": "d", "cluster": "cl", "effects": 2}
    if boot:
        kwargs |= {"boot": True, "biters": 30, "random_state": 7}
    missing = weighted_clustered_panel.with_columns(pl.when(group).then(None).otherwise(pl.col("cl")).alias("cl"))
    non_finite = weighted_clustered_panel.with_columns(pl.when(group).then(value).otherwise(pl.col("cl")).alias("cl"))

    expected = did_multiplegt(missing, **kwargs)
    result = did_multiplegt(non_finite, **kwargs)

    np.testing.assert_array_equal(result.effects.estimates, expected.effects.estimates)
    np.testing.assert_array_equal(result.effects.std_errors, expected.effects.std_errors)


@pytest.mark.parametrize("weights", ["zero", "one negative"])
def test_weights_without_positive_mean_raise(weighted_clustered_panel, weights):
    row = (pl.col("id") == 1) & (pl.col("time") == 5)
    values = {"zero": pl.lit(0.0), "one negative": pl.when(row).then(-1.0).otherwise(pl.col("w"))}
    data = weighted_clustered_panel.with_columns(values[weights].alias("w"))
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        did_multiplegt(data, yname="y", idname="id", tname="time", dname="d", weightsname="w")


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_bootstrap_leaves_out_draws_without_positive_weight(
    zero_weight_cluster_panel, fixed_draws, relabel_cluster_copies
):
    draws = [
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [5, 5, 6, 7, 8, 9, 0, 1, 2, 2],
        [9, 8, 7, 6, 5, 4, 3, 3, 1, 0],
    ]
    kwargs = {
        "yname": "y",
        "idname": "id",
        "tname": "time",
        "dname": "d",
        "weightsname": "w",
        "effects": 2,
        "placebo": 1,
    }

    result = did_multiplegt(
        zero_weight_cluster_panel, **kwargs, cluster="cl", boot=True, biters=4, random_state=fixed_draws(draws)
    )
    fits = [
        did_multiplegt(relabel_cluster_copies(zero_weight_cluster_panel, draw, "cl", "id"), **kwargs)
        for draw in draws[1:]
    ]

    np.testing.assert_allclose(
        result.effects.std_errors, np.std([fit.effects.estimates for fit in fits], axis=0, ddof=1), rtol=1e-10
    )
    np.testing.assert_allclose(
        result.placebos.std_errors, np.std([fit.placebos.estimates for fit in fits], axis=0, ddof=1), rtol=1e-10
    )
    np.testing.assert_allclose(result.ate.std_error, np.std([fit.ate.estimate for fit in fits], ddof=1), rtol=1e-10)


@pytest.mark.filterwarnings("ignore:did_multiplegt computes analytical standard errors:UserWarning")
def test_seeded_bootstrap_with_a_zero_weight_draw_gives_known_standard_errors(zero_weight_cluster_panel):
    result = did_multiplegt(
        zero_weight_cluster_panel,
        yname="y",
        idname="id",
        tname="time",
        dname="d",
        weightsname="w",
        effects=2,
        placebo=1,
        cluster="cl",
        boot=True,
        biters=20,
        random_state=46,
    )

    np.testing.assert_allclose(result.effects.std_errors, [0.4562282558172498, 0.3962547382974173], rtol=1e-10)
    np.testing.assert_allclose(result.placebos.std_errors, [0.6528766325260316], rtol=1e-10)
    np.testing.assert_allclose(result.ate.std_error, 0.4042719986527167, rtol=1e-10)


@pytest.mark.parametrize(
    "reserved",
    [".w", "F_g", "d_sq", "d_sq_int", "d_fg", "S_g", "L_g", "T_g", "weight_gt", "first_obs_by_gp", "t_max_by_group"],
)
@pytest.mark.parametrize(("column", "argument"), [("y", "yname"), ("x1", "xformla")])
def test_reserved_column_names_raise(panel_with_controls, reserved, column, argument):
    with pytest.raises(ValueError, match=re.escape(f"{argument} names the column '{reserved}'")):
        did_multiplegt(
            panel_with_controls.rename({column: reserved}),
            yname=reserved if column == "y" else "y",
            idname="id",
            tname="time",
            dname="d",
            xformla=f"~ {reserved} + x2" if column == "x1" else "~ x1 + x2",
        )


def test_did_multiplegt_rejects_repeated_unit_periods(simple_panel_duplicated):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (id, time) pair (5, 3)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        did_multiplegt(simple_panel_duplicated, yname="y", tname="time", idname="id", dname="d")
