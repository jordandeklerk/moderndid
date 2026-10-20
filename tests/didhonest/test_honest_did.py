"""Tests for sensitivity analysis using the approach of Rambachan and Roth (2023)."""

import importlib

import numpy as np
import pytest
from scipy import stats

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import AGGTEResult, HonestDiDResult, aggte, att_gt, honest_did, load_mpdta
from moderndid.didhonest import (
    compute_identified_set_rm,
    compute_identified_set_sdrm,
    create_sensitivity_results_rm,
    create_sensitivity_results_sm,
)

honest_did_module = importlib.import_module("moderndid.didhonest.honest_did")


@pytest.fixture
def sample_aggte_result():
    np.random.seed(42)

    event_times = np.array([-4, -3, -2, -1, 1, 2, 3])
    att_by_event = np.array([0.05, 0.1, -0.05, 0.02, 0.5, 0.3, 0.4])
    se_by_event = np.array([0.09, 0.1, 0.122, 0.141, 0.158, 0.173, 0.187])

    n_units = 120
    n_events = len(event_times)
    influence_func = np.random.normal(0, 0.1, (n_units, n_events))

    for i in range(1, n_events):
        influence_func[:, i] = 0.5 * influence_func[:, i - 1] + 0.5 * influence_func[:, i]

    return AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        influence_func=influence_func,
        estimation_params={"alpha": 0.05, "control_group": "nevertreated"},
    )


@pytest.fixture
def sample_aggte_result_with_ref():
    np.random.seed(42)

    event_times = np.array([-4, -3, -2, -1, 0, 1, 2, 3])
    att_by_event = np.array([0.05, 0.1, -0.05, 0.02, 0.0, 0.5, 0.3, 0.4])
    se_by_event = np.array([0.09, 0.1, 0.122, 0.141, 0.0, 0.158, 0.173, 0.187])

    n_units = 120
    n_events = len(event_times)
    influence_func = np.random.normal(0, 0.1, (n_units, n_events))

    influence_func[:, 4] = 0

    return AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        influence_func=influence_func,
        estimation_params={"alpha": 0.05},
    )


def test_honest_did_basic(sample_aggte_result):
    result = honest_did(sample_aggte_result, event_time=1)

    assert isinstance(result, HonestDiDResult)
    assert isinstance(result.robust_ci, pl.DataFrame)
    assert result.sensitivity_type == "smoothness"

    assert "lb" in result.robust_ci.columns
    assert "ub" in result.robust_ci.columns
    assert "method" in result.robust_ci.columns
    assert "delta" in result.robust_ci.columns
    assert "m" in result.robust_ci.columns

    assert result.original_ci.lb < result.original_ci.ub
    assert result.original_ci.method == "Original"


def test_honest_did_with_reference_period(sample_aggte_result_with_ref):
    result = honest_did(sample_aggte_result_with_ref, event_time=1, m_vec=np.array([0.0, 0.1]))

    assert isinstance(result, HonestDiDResult)
    assert isinstance(result.robust_ci, pl.DataFrame)
    assert len(result.robust_ci) > 0


def test_honest_did_relative_magnitude(sample_aggte_result):
    result = honest_did(
        sample_aggte_result,
        event_time=2,
        sensitivity_type="relative_magnitude",
        method="Conditional",
        m_bar_vec=np.array([0.0, 1.0]),
        grid_points=20,
    )

    assert result.sensitivity_type == "relative_magnitude"
    assert len(result.robust_ci) == 2
    assert "Mbar" in result.robust_ci.columns
    assert result.robust_ci["Mbar"].to_list() == [0.0, 1.0]


@pytest.mark.parametrize(
    "event_time,method",
    [
        (1, "FLCI"),
        (1, "Conditional"),
    ],
)
def test_honest_did_methods(sample_aggte_result, event_time, method):
    result = honest_did(
        sample_aggte_result,
        event_time=event_time,
        method=method,
        m_vec=np.array([0, 0.1]),
        grid_points=20,
    )

    assert (result.robust_ci["method"] == method).all()


def test_honest_did_custom_alpha(sample_aggte_result):
    result = honest_did(
        sample_aggte_result,
        event_time=1,
        alpha=0.10,
        m_vec=np.array([0.0]),
    )

    result_05 = honest_did(
        sample_aggte_result,
        event_time=1,
        alpha=0.05,
        m_vec=np.array([0.0]),
    )

    ci_width_10 = result.original_ci.ub - result.original_ci.lb
    ci_width_05 = result_05.original_ci.ub - result_05.original_ci.lb
    assert ci_width_10 < ci_width_05


def test_honest_did_grid_points(sample_aggte_result):
    result = honest_did(
        sample_aggte_result,
        event_time=1,
        sensitivity_type="smoothness",
        method="Conditional",
        grid_points=12,
        m_vec=np.array([0.1]),
    )

    assert isinstance(result.robust_ci, pl.DataFrame)


def test_honest_did_errors():
    static_result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="simple",
    )

    with pytest.raises(ValueError, match="event study"):
        honest_did(static_result)

    no_inf_result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([-1, 1, 2]),
        att_by_event=np.array([0.1, 0.5, 0.3]),
        influence_func=None,
    )

    with pytest.raises(ValueError, match="influence functions"):
        honest_did(no_inf_result)

    non_consec_result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([-4, -2, -1, 1, 3]),
        att_by_event=np.array([0.05, -0.05, 0.02, 0.5, 0.4]),
        influence_func=np.random.normal(0, 0.1, (500, 5)),
    )

    with pytest.raises(ValueError, match="consecutive"):
        honest_did(non_consec_result)


def test_honest_did_invalid_event_time(sample_aggte_result):
    with pytest.raises(ValueError, match="Event time .* not found"):
        honest_did(sample_aggte_result, event_time=5)


def test_honest_did_invalid_sensitivity_type(sample_aggte_result):
    with pytest.raises(ValueError, match="sensitivity_type must be"):
        honest_did(sample_aggte_result, event_time=1, sensitivity_type="invalid")


def test_honest_did_invalid_object_type():
    with pytest.raises(TypeError, match="not implemented for object of type"):
        honest_did("not an event study object")


def test_honest_did_no_pre_periods():
    result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([1, 2, 3]),
        att_by_event=np.array([0.5, 0.3, 0.4]),
        influence_func=np.random.normal(0, 0.1, (500, 3)),
    )

    with pytest.raises(ValueError, match="pre-treatment periods"):
        honest_did(result)


def test_honest_did_no_post_periods():
    result = AGGTEResult(
        overall_att=0.25,
        overall_se=0.08,
        aggregation_type="dynamic",
        event_times=np.array([-3, -2, -1]),
        att_by_event=np.array([0.1, -0.05, 0.02]),
        influence_func=np.random.normal(0, 0.1, (500, 3)),
    )

    with pytest.raises(ValueError, match="post-treatment periods"):
        honest_did(result)


def test_honest_did_rejects_varying_base_period(sample_aggte_result):
    params = {**sample_aggte_result.estimation_params, "base_period": "varying"}
    varying = sample_aggte_result._replace(estimation_params=params)

    with pytest.raises(ValueError, match="universal base period"):
        honest_did(varying, event_time=1)


def test_honest_did_accepts_universal_base_period(sample_aggte_result):
    params = {**sample_aggte_result.estimation_params, "base_period": "universal"}
    universal = sample_aggte_result._replace(estimation_params=params)

    result = honest_did(universal, event_time=1, m_vec=np.array([0.0]))

    assert len(result.robust_ci) == 1


def test_honest_did_anticipation_moves_reference_period():
    rng = np.random.default_rng(3)
    event_times = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
    att_by_event = np.array([0.02, -0.01, 0.03, 0.0, 0.04, 0.3, 0.35, 0.4])
    influence_func = rng.normal(0, 0.5, (200, 8))
    influence_func[:, 3] = 0.0
    event_study = AGGTEResult(
        overall_att=0.3,
        overall_se=0.05,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=np.sqrt(np.sum(influence_func**2, axis=0)) / 200,
        influence_func=influence_func,
        estimation_params={"base_period": "universal", "anticipation_periods": 1},
    )

    result = honest_did(event_study, event_time=0, m_vec=np.array([0.0]))
    anticipation = honest_did(event_study, event_time=-1, m_vec=np.array([0.0]))

    se = np.sqrt(np.sum(influence_func[:, 5] ** 2)) / 200
    assert result.original_ci.lb == pytest.approx(0.3 - 1.959963984540054 * se)
    assert result.original_ci.ub == pytest.approx(0.3 + 1.959963984540054 * se)
    assert anticipation.original_ci.lb < 0.04 < anticipation.original_ci.ub


def test_honest_did_grid_points_defaults(sample_aggte_result, monkeypatch):
    calls = []

    def record(**kwargs):
        calls.append(kwargs["grid_points"])
        return pl.DataFrame({"lb": [0.0], "ub": [1.0]})

    monkeypatch.setattr(honest_did_module, "create_sensitivity_results_sm", record)
    monkeypatch.setattr(honest_did_module, "create_sensitivity_results_rm", record)

    honest_did(sample_aggte_result, event_time=1, bias_direction="positive")
    honest_did(sample_aggte_result, event_time=1, sensitivity_type="relative_magnitude")
    honest_did(sample_aggte_result, event_time=1, grid_points=12)
    honest_did(sample_aggte_result, event_time=1, sensitivity_type="relative_magnitude", grid_points=12)

    assert calls == [1000, 100, 12, 12]


def test_honest_did_relative_magnitude_one_pre_period(make_event_study):
    event_study = make_event_study(np.array([-2, -1, 0, 1]), np.array([-0.03, 0.0, 0.10, 0.12]))
    m_bar_vec = np.array([0.0, 1.0])

    result = honest_did(
        event_study, event_time=0, sensitivity_type="relative_magnitude", method="Conditional", m_bar_vec=m_bar_vec
    )

    influence_func = np.delete(event_study.influence_func, 1, axis=1)
    sigma = influence_func.T @ influence_func / 200**2
    betahat = np.array([-0.03, 0.10, 0.12])
    expected = create_sensitivity_results_rm(
        betahat, sigma, 1, 2, method="Conditional", l_vec=np.array([1.0, 0.0]), m_bar_vec=m_bar_vec, grid_points=100
    )
    assert result.robust_ci["lb"].to_list() == pytest.approx(expected["lb"].to_list(), abs=1e-12)
    assert result.robust_ci["ub"].to_list() == pytest.approx(expected["ub"].to_list(), abs=1e-12)

    for row in result.robust_ci.iter_rows(named=True):
        id_set = compute_identified_set_rm(row["Mbar"], betahat, np.array([1.0, 0.0]), 1, 2)
        assert id_set.id_lb == pytest.approx(0.10 - 0.03 * row["Mbar"])
        assert id_set.id_ub == pytest.approx(0.10 + 0.03 * row["Mbar"])
        assert row["lb"] < id_set.id_lb
        assert row["ub"] > id_set.id_ub


def test_honest_did_linear_trend_two_pre_periods(make_event_study):
    event_study = make_event_study(np.array([-3, -2, -1, 0, 1]), np.array([0.05, -0.02, 0.0, 0.30, 0.40]), seed=1)
    m_bar_vec = np.array([0.0, 1.0])

    result = honest_did(
        event_study,
        event_time=1,
        sensitivity_type="relative_magnitude",
        bound="deviation from linear trend",
        method="Conditional",
        m_bar_vec=m_bar_vec,
    )

    betahat = np.array([0.05, -0.02, 0.30, 0.40])
    assert result.robust_ci["delta"].to_list() == ["DeltaSDRM", "DeltaSDRM"]
    for row in result.robust_ci.iter_rows(named=True):
        id_set = compute_identified_set_sdrm(row["Mbar"], betahat, np.array([0.0, 1.0]), 2, 2)
        assert id_set.id_lb == pytest.approx(0.36 - 3 * 0.09 * row["Mbar"])
        assert id_set.id_ub == pytest.approx(0.36 + 3 * 0.09 * row["Mbar"])
        assert row["lb"] < id_set.id_lb
        assert row["ub"] > id_set.id_ub


def test_honest_did_smoothness_one_pre_period_uses_single_estimator(make_event_study):
    event_study = make_event_study(np.array([-2, -1, 0]), np.array([-0.03, 0.0, 0.20]), seed=2)
    m_vec = np.array([0.0, 0.02, 0.2])

    result = honest_did(event_study, event_time=0, m_vec=m_vec)

    influence_func = np.delete(event_study.influence_func, 1, axis=1)
    sigma = influence_func.T @ influence_func / 200**2
    h = np.sqrt(sigma[0, 0] + 2 * sigma[0, 1] + sigma[1, 1])
    half_lengths = [stats.foldnorm.ppf(0.95, c=m / h) * h for m in m_vec]
    assert result.robust_ci["lb"].to_list() == pytest.approx([0.17 - half for half in half_lengths], rel=1e-6)
    assert result.robust_ci["ub"].to_list() == pytest.approx([0.17 + half for half in half_lengths], rel=1e-6)


def test_honest_did_clusters_the_covariance(clustered_event_study):
    result = honest_did(clustered_event_study, event_time=0, m_vec=np.array([0.0]))

    cluster_sums = np.array([2.0, -1.7, 1.3, -1.1, -0.5])
    se = np.sqrt(np.sum(cluster_sums**2)) / 10
    z = stats.norm.ppf(0.975)
    assert result.original_ci.lb == pytest.approx(0.30 - z * se)
    assert result.original_ci.ub == pytest.approx(0.30 + z * se)


def test_honest_did_robust_ci_uses_the_clustered_covariance(clustered_event_study):
    m_vec = np.array([0.0, 0.1])

    result = honest_did(clustered_event_study, event_time=1, m_vec=m_vec)

    influence_func = np.delete(clustered_event_study.influence_func, 1, axis=1)
    cluster_sums = influence_func.reshape(5, 2, 3).sum(axis=1)
    sigma = cluster_sums.T @ cluster_sums / 100
    expected = create_sensitivity_results_sm(
        np.array([0.02, 0.30, 0.40]), sigma, 1, 2, l_vec=np.array([0.0, 1.0]), m_vec=m_vec
    )
    assert result.robust_ci["lb"].to_list() == pytest.approx(expected["lb"].to_list(), rel=1e-10)
    assert result.robust_ci["ub"].to_list() == pytest.approx(expected["ub"].to_list(), rel=1e-10)


def test_honest_did_without_clustervars_ignores_the_cluster_vector(clustered_event_study):
    params = {**clustered_event_study.estimation_params, "clustervars": []}
    unclustered = clustered_event_study._replace(estimation_params=params)

    result = honest_did(unclustered, event_time=0, m_vec=np.array([0.0]))

    se = np.sqrt(np.sum(clustered_event_study.influence_func[:, 2] ** 2)) / 10
    assert result.original_ci.ub - result.original_ci.lb == pytest.approx(2 * stats.norm.ppf(0.975) * se)


@pytest.mark.parametrize(
    "cluster,error_match",
    [
        (None, "no cluster for the units"),
        (np.array([1, 1, 2, 2, 3, 3, 4, 4, 5]), "has 9 entries but its influence function has 10 rows"),
        (np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), "2 clusters and 3 coefficients"),
    ],
)
def test_honest_did_rejects_unusable_clusters(clustered_event_study, cluster, error_match):
    params = {**clustered_event_study.estimation_params, "cluster": cluster}
    event_study = clustered_event_study._replace(estimation_params=params)

    with pytest.raises(ValueError, match=error_match):
        honest_did(event_study, event_time=0, m_vec=np.array([0.0]))


def test_honest_did_clusters_an_att_gt_event_study():
    data = load_mpdta().with_columns((pl.col("countyreal") // 1000).alias("state"))
    group_time = att_gt(
        data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        base_period="universal",
        boot=True,
        cband=False,
        biters=100,
        clustervars=["state"],
        random_state=7,
    )
    event_study = aggte(group_time, type="dynamic")

    result = honest_did(event_study, event_time=0, m_vec=np.array([0.0]))

    cluster = np.asarray(event_study.estimation_params["cluster"])
    influence_func = event_study.influence_func[:, list(event_study.event_times).index(0)]
    _, cluster_idx = np.unique(cluster, return_inverse=True)
    cluster_se = np.sqrt(np.sum(np.bincount(cluster_idx, weights=influence_func) ** 2)) / len(influence_func)
    unit_se = np.sqrt(np.sum(influence_func**2)) / len(influence_func)
    assert len(np.unique(cluster)) == 29
    assert (result.original_ci.ub - result.original_ci.lb) / 2 == pytest.approx(stats.norm.ppf(0.975) * cluster_se)
    assert cluster_se != pytest.approx(unit_se, rel=0.01)


def test_honest_did_covariance_matches_the_clustered_standard_errors_of_an_event_study():
    data = load_mpdta().with_columns((pl.col("countyreal") // 1000).alias("state"))
    fit = att_gt(
        data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        base_period="universal",
        clustervars=["state"],
    )
    event_study = aggte(fit, type="dynamic", cband=False)

    estimated = event_study.event_times != -1
    covariance = honest_did_module._event_study_covariance(
        event_study.influence_func[:, estimated], event_study.estimation_params
    )
    sensitivity = honest_did(event_study, event_time=0, m_vec=np.array([0.0]))
    on_impact = list(event_study.event_times).index(0)

    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), event_study.se_by_event[estimated], rtol=1e-12)
    np.testing.assert_allclose(
        (sensitivity.original_ci.ub - sensitivity.original_ci.lb) / 2,
        stats.norm.ppf(0.975) * event_study.se_by_event[on_impact],
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "sensitivity_type, options",
    [
        ("smoothness", {"m_vec": np.array([0.0, 0.01])}),
        ("relative_magnitude", {"m_bar_vec": np.array([0.0, 1.0]), "grid_points": 20}),
    ],
    ids=["smoothness", "relative_magnitude"],
)
def test_honest_did_on_an_event_study_of_a_shard_fit_matches_the_in_memory_fit(
    att_gt_on_shards, sensitivity_type, options
):
    data = load_mpdta()
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": "universal",
        "cband": False,
    }
    memory = aggte(att_gt(data, **spec), type="dynamic")
    sharded = aggte(att_gt_on_shards(data, 3, **spec), type="dynamic")

    from_memory = honest_did(memory, event_time=0, sensitivity_type=sensitivity_type, **options)
    from_shards = honest_did(sharded, event_time=0, sensitivity_type=sensitivity_type, **options)

    assert sharded.influence_func is None
    np.testing.assert_allclose(from_shards.robust_ci["lb"], from_memory.robust_ci["lb"], rtol=1e-10)
    np.testing.assert_allclose(from_shards.robust_ci["ub"], from_memory.robust_ci["ub"], rtol=1e-10)
    np.testing.assert_allclose(
        [from_shards.original_ci.lb, from_shards.original_ci.ub],
        [from_memory.original_ci.lb, from_memory.original_ci.ub],
        rtol=1e-12,
    )


def test_honest_did_on_an_event_study_of_a_clustered_shard_fit_matches_the_in_memory_fit(att_gt_on_shards):
    data = load_mpdta().with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": "universal",
        "clustervars": ["state"],
        "cband": False,
    }
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory = aggte(att_gt(data, **spec), type="dynamic")
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded = aggte(att_gt_on_shards(data, 3, **spec), type="dynamic")
    unclustered = aggte(att_gt_on_shards(data, 3, **{**spec, "clustervars": None}), type="dynamic")

    from_memory = honest_did(memory, event_time=0, m_vec=np.array([0.0, 0.01]))
    from_shards = honest_did(sharded, event_time=0, m_vec=np.array([0.0, 0.01]))
    from_unclustered = honest_did(unclustered, event_time=0, m_vec=np.array([0.0, 0.01]))

    assert sharded.estimation_params["n_clusters"] == 29
    np.testing.assert_allclose(from_shards.robust_ci["lb"], from_memory.robust_ci["lb"], rtol=1e-10)
    np.testing.assert_allclose(from_shards.robust_ci["ub"], from_memory.robust_ci["ub"], rtol=1e-10)
    np.testing.assert_allclose(
        [from_shards.original_ci.lb, from_shards.original_ci.ub],
        [from_memory.original_ci.lb, from_memory.original_ci.ub],
        rtol=1e-12,
    )
    assert np.max(np.abs(from_unclustered.robust_ci["lb"].to_numpy() - from_shards.robust_ci["lb"].to_numpy())) > 1e-3
