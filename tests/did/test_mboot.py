"""Tests for the multiplier bootstrap function."""

import inspect
import itertools
from importlib import import_module

import numpy as np
import pytest

from moderndid import aggte, att_gt, mboot
from moderndid.did.mboot import _mboot, _mboot_from_draws

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "influence_vector",
    [
        "constant",
        "linear_trend",
        "high_variance",
        "low_variance",
        "normal",
        "uniform",
        "t3",
        "t10",
        "chi2",
        "exponential",
    ],
    indirect=True,
)
def test_influence_vector_gets_the_documented_draws_variance_standard_error_and_critical_value(
    monkeypatch, draws_from_weights, rademacher_weights, influence_vector
):
    weights = rademacher_weights(999, 300)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = mboot(influence_vector, n_units=600, biters=999)
    perturbations = weights @ influence_vector / 300
    scale = np.subtract(*np.percentile(perturbations, [75, 25], method="inverted_cdf")) / 1.3489795
    draws = np.sqrt(600) * perturbations

    assert result["V"].shape == (1, 1)
    np.testing.assert_allclose(result["bres"], draws[:, None], rtol=1e-12)
    np.testing.assert_allclose(result["V"], [[np.var(draws, ddof=1)]], rtol=1e-12)
    np.testing.assert_allclose(result["se"], [scale], rtol=1e-12)
    np.testing.assert_allclose(
        result["crit_val"], np.percentile(np.abs(perturbations) / scale, 95, method="inverted_cdf"), rtol=1e-12
    )


@pytest.mark.parametrize("n_rows, n_params", [(100, 1), (100, 5), (100, 10), (1000, 1), (1000, 20)])
def test_multivariate_influence_function_gets_the_documented_draws_covariance_errors_and_band(
    monkeypatch, draws_from_weights, rademacher_weights, n_rows, n_params
):
    rng = np.random.default_rng(3)
    inf_func = rng.standard_normal((n_rows, n_params)) @ rng.random((n_params, n_params))
    weights = rademacher_weights(99, n_rows)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = mboot(inf_func, n_units=2 * n_rows, biters=99)
    perturbations = weights @ inf_func / n_rows
    scale = np.subtract(*np.percentile(perturbations, [75, 25], axis=0, method="inverted_cdf")) / 1.3489795
    draws = np.sqrt(2 * n_rows) * perturbations
    centered = draws - draws.mean(axis=0)
    largest = np.max(np.abs(perturbations) / scale, axis=1)

    np.testing.assert_allclose(result["bres"], draws, rtol=1e-12)
    np.testing.assert_allclose(result["V"], centered.T @ centered / 98, rtol=1e-12)
    np.testing.assert_allclose(result["se"], scale, rtol=1e-12)
    np.testing.assert_allclose(result["crit_val"], np.percentile(largest, 95, method="inverted_cdf"), rtol=1e-12)


def test_clustering_gives_each_cluster_sum_one_sign_and_scales_by_the_rows_rather_than_n_units():
    inf_func = np.zeros(30)
    inf_func[9:11] = [1.5, 2.5]
    cluster = np.repeat([4, 9, 2, 7, 0, 5, 1, 8, 3, 6], 3)

    result = mboot(inf_func, n_units=75, biters=99, cluster=cluster, random_state=7)

    np.testing.assert_allclose(np.abs(result["bres"]), np.full((99, 1), 4 / np.sqrt(10)), rtol=1e-12)
    np.testing.assert_allclose(result["se"], [8 / (30 * 1.3489795)], rtol=1e-12)


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf, -np.inf], ids=["zero", "nan", "inf", "neginf"])
def test_degenerate_column_gets_no_standard_error_and_stays_out_of_the_covariance_and_band(
    monkeypatch, draws_from_weights, rademacher_weights, value
):
    inf_func = np.random.default_rng(3).standard_normal((200, 3)) * [1.0, 0.0, 3.0]
    inf_func[10:20, 1] = value
    weights = rademacher_weights(99, 200)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = mboot(inf_func, n_units=200, biters=99)
    perturbations = weights @ inf_func[:, [0, 2]] / 200
    scale = np.subtract(*np.percentile(perturbations, [75, 25], axis=0, method="inverted_cdf")) / 1.3489795
    centered = np.sqrt(200) * (perturbations - perturbations.mean(axis=0))
    largest = np.max(np.abs(perturbations) / scale, axis=1)

    np.testing.assert_allclose(result["se"], [scale[0], np.nan, scale[1]], rtol=1e-12)
    np.testing.assert_allclose(result["V"], centered.T @ centered / 98, rtol=1e-12)
    np.testing.assert_allclose(result["crit_val"], np.percentile(largest, 95, method="inverted_cdf"), rtol=1e-12)


def test_random_state_reproduces_the_draws_and_another_seed_changes_them():
    inf_func = np.random.default_rng(3).standard_normal(100)

    first = mboot(inf_func, n_units=100, biters=99, random_state=42)
    again = mboot(inf_func, n_units=100, biters=99, random_state=42)
    other = mboot(inf_func, n_units=100, biters=99, random_state=43)

    np.testing.assert_array_equal(again["bres"], first["bres"])
    np.testing.assert_array_equal(again["se"], first["se"])
    assert again["crit_val"] == first["crit_val"]
    assert np.mean(other["bres"] != first["bres"]) > 0.9


@pytest.mark.parametrize("n_units", [100, 50])
def test_cluster_needs_one_entry_for_each_row_whatever_n_units(n_units):
    with pytest.raises(ValueError, match="cluster has 50 entries but inf_func has 100 rows"):
        mboot(np.zeros(100), n_units=n_units, cluster=np.arange(50))


@pytest.mark.parametrize("alpha, deviation", [(0.01, 99), (0.05, 95), (0.10, 90), (0.20, 80)])
def test_significance_level_picks_the_quantile_of_the_largest_standardized_draw(monkeypatch, alpha, deviation):
    draws = np.arange(-100.0, 101.0)[:, None]
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = mboot(np.zeros((1, 1)), n_units=1, biters=201, alpha=alpha)

    np.testing.assert_allclose(result["crit_val"], deviation / (100 / 1.3489795), rtol=1e-12)


@pytest.mark.parametrize(
    "rows, values",
    [(slice(None), [np.nan, np.nan, np.nan]), (slice(50, 60), [np.nan, np.inf, -np.inf])],
    ids=["all_nan", "mixed"],
)
def test_influence_function_without_a_usable_column_has_no_errors_covariance_or_band(rows, values):
    inf_func = np.random.default_rng(3).standard_normal((100, 3))
    inf_func[rows] = values

    result = mboot(inf_func, n_units=100, biters=99, random_state=7)

    np.testing.assert_array_equal(result["se"], [np.nan, np.nan, np.nan])
    assert np.isnan(result["crit_val"])
    assert result["V"].shape == (0, 0)


@pytest.mark.parametrize("n_rows, n_units", [(1, 1), (2, 8), (10, 10), (1000, 250)])
def test_draws_of_one_nonzero_row_are_its_value_with_a_random_sign_times_root_n_units_over_the_rows(n_rows, n_units):
    inf_func = np.zeros(n_rows)
    inf_func[0] = 2.5

    result = mboot(inf_func, n_units=n_units, biters=99, random_state=7)

    np.testing.assert_allclose(np.abs(result["bres"]), np.full((99, 1), 2.5 * np.sqrt(n_units) / n_rows), rtol=1e-12)
    np.testing.assert_allclose(result["se"], [5 / (n_rows * 1.3489795)], rtol=1e-12)


@pytest.mark.parametrize(
    "sizes",
    [[6] * 20, [5, 10, 15, 20, 25, 45], [1] * 120, [100, 20]],
    ids=["equal_sizes", "varying_sizes", "singleton_clusters", "unbalanced"],
)
def test_clustering_patterns_give_one_multiplier_to_the_sum_of_each_cluster_in_sorted_order(
    monkeypatch, draws_from_weights, rademacher_weights, sizes
):
    rng = np.random.default_rng(3)
    inf_func = rng.standard_normal((120, 2)) * [1.0, 4.0]
    cluster = rng.permutation(np.repeat(10 * rng.permutation(len(sizes)), sizes))
    weights = rademacher_weights(99, len(sizes))
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = mboot(inf_func, n_units=120, biters=99, cluster=cluster)
    sums = np.array([inf_func[cluster == label].sum(axis=0) for label in np.unique(cluster)])
    perturbations = weights @ sums / 120
    scale = np.subtract(*np.percentile(perturbations, [75, 25], axis=0, method="inverted_cdf")) / 1.3489795

    np.testing.assert_allclose(result["bres"], perturbations * 120 / np.sqrt(len(sizes)), rtol=1e-12)
    np.testing.assert_allclose(result["se"], scale, rtol=1e-12)


def test_clustered_bootstrap_over_every_sign_pattern_weights_cluster_sums(monkeypatch, draws_from_weights):
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    result = mboot(np.ones(6), n_units=6, biters=8, cluster=np.array([0, 1, 1, 2, 2, 2]))

    np.testing.assert_allclose(result["bres"], (signs @ [1.0, 2.0, 3.0] / np.sqrt(3))[:, None], rtol=1e-12)
    np.testing.assert_allclose(result["V"], [[16 / 3]], rtol=1e-12)
    np.testing.assert_allclose(result["se"], [1 / 1.3489795], rtol=1e-12)


def test_clustered_bootstrap_with_equal_cluster_sizes_keeps_the_results_of_cluster_means():
    inf_func = np.random.default_rng(3).standard_normal((120, 2))
    cluster = np.repeat(np.arange(20), 6)

    result = mboot(inf_func, n_units=120, biters=199, cluster=cluster, random_state=5)
    expected = mboot(inf_func.reshape(20, 6, 2).mean(axis=1), n_units=20, biters=199, random_state=5)

    assert np.all(np.isfinite(expected["se"]))
    assert np.isfinite(expected["crit_val"])
    np.testing.assert_allclose(result["se"], expected["se"], rtol=1e-12)
    np.testing.assert_allclose(result["crit_val"], expected["crit_val"], rtol=1e-12)


def test_clustered_bootstrap_with_one_row_per_cluster_is_the_unclustered_bootstrap():
    inf_func = np.random.default_rng(3).standard_normal((150, 3))

    clustered = mboot(inf_func, n_units=150, biters=199, cluster=np.arange(150), random_state=5)
    unclustered = mboot(inf_func, n_units=150, biters=199, random_state=5)

    np.testing.assert_array_equal(clustered["bres"], unclustered["bres"])
    np.testing.assert_allclose(clustered["se"], unclustered["se"], rtol=1e-12)
    assert clustered["crit_val"] == unclustered["crit_val"]


def test_clustered_standard_errors_scale_with_a_small_influence_function():
    inf_func = np.random.default_rng(3).standard_normal((500, 2))
    cluster = np.arange(500) % 29

    base = mboot(inf_func, n_units=500, biters=999, cluster=cluster, random_state=7)
    small = mboot(1e-5 * inf_func, n_units=500, biters=999, cluster=cluster, random_state=7)

    assert np.all(np.isfinite(small["se"]))
    assert np.isfinite(base["crit_val"])
    np.testing.assert_allclose(small["se"], 1e-5 * base["se"], rtol=1e-12)
    np.testing.assert_allclose(small["crit_val"], base["crit_val"], rtol=1e-12)


@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
def test_att_gt_and_aggte_state_clustered_standard_errors_sum_within_states(
    monkeypatch, draws_from_weights, rademacher_weights, mpdta_with_states, panel
):
    weights = rademacher_weights(199, 29)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    fit = att_gt(
        mpdta_with_states,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        panel=panel,
        est_method="reg",
        base_period="universal",
        boot=True,
        biters=199,
        cband=False,
        clustervars=["state"],
    )
    group = aggte(fit, type="group")

    _, codes = np.unique(fit.estimation_params["cluster"], return_inverse=True)
    influence = np.column_stack([np.asarray(fit.influence_func), np.ravel(group.influence_func_overall)])
    sums = np.column_stack([np.bincount(codes, weights=column, minlength=29) for column in influence.T])
    q75, q25 = np.percentile(weights @ sums / len(codes), [75, 25], axis=0, method="inverted_cdf")
    expected = (q75 - q25) / 1.3489795
    kept = ~np.isnan(fit.se_gt)

    assert kept.sum() == 12
    np.testing.assert_allclose(fit.se_gt[kept], expected[:-1][kept], rtol=1e-12)
    np.testing.assert_allclose(group.overall_se, expected[-1], rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e5, 1e10])
def test_scale_invariance(scale):
    inf_func = np.random.default_rng(3).standard_normal((100, 2)) * [1.0, 5.0]

    base = mboot(inf_func, n_units=100, biters=99, random_state=42)
    scaled = mboot(scale * inf_func, n_units=100, biters=99, random_state=42)

    assert np.all(np.isfinite(base["se"]))
    assert np.isfinite(base["crit_val"])
    np.testing.assert_allclose(scaled["se"], scale * base["se"], rtol=1e-12)
    np.testing.assert_allclose(scaled["crit_val"], base["crit_val"], rtol=1e-12)


def test_quantile_uses_inverted_cdf(monkeypatch):
    draws = np.array([[3.0], [-1.0], [7.0], [0.0], [-4.0], [12.0], [2.0], [-6.0]])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = mboot(np.zeros((1, 1)), n_units=1, biters=8)

    np.testing.assert_allclose(result["se"], [7 / 1.3489795], rtol=1e-12)
    np.testing.assert_allclose(result["crit_val"], 12 / (7 / 1.3489795), rtol=1e-12)


@pytest.mark.parametrize(
    "rule", [{}, {"skip_small_scales": True}, {"keep_infinite_draws": True}], ids=["default", "skip", "keep"]
)
def test_column_with_an_infinite_entry_gets_an_infinite_error_and_stays_out_of_the_band(
    monkeypatch, draws_from_weights, rademacher_weights, rule
):
    inf_func = np.random.default_rng(3).standard_normal((200, 3)) * [1.0, 2.0, 3.0]
    inf_func[10, 1] = np.inf
    weights = rademacher_weights(99, 200)
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = _mboot(inf_func, n_units=200, biters=99, **rule)
    perturbations = weights @ inf_func[:, [0, 2]] / 200
    scale = np.subtract(*np.percentile(perturbations, [75, 25], axis=0, method="inverted_cdf")) / 1.3489795
    largest = np.max(np.abs(perturbations) / scale, axis=1)

    np.testing.assert_allclose(result["se"], [scale[0], np.inf, scale[1]], rtol=1e-12)
    np.testing.assert_allclose(result["crit_val"], np.percentile(largest, 95, method="inverted_cdf"), rtol=1e-12)


def test_draw_with_an_infinite_deviation_leaves_the_critical_value_sample(monkeypatch):
    draws = np.arange(-10.0, 11.0)[:, None]
    draws[[0, -1]] = [[-np.inf], [np.inf]]
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = mboot(np.zeros((1, 1)), n_units=1, biters=21)

    np.testing.assert_allclose(result["se"], [10 / 1.3489795], rtol=1e-12)
    np.testing.assert_allclose(result["crit_val"], 9 / (10 / 1.3489795), rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_zero_scale_column_drops_the_draws_that_move_it(monkeypatch, zero_scale_draws):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: zero_scale_draws
    )

    result = mboot(np.zeros((1, 2)), n_units=1, biters=21)

    np.testing.assert_allclose(result["crit_val"], 5 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(result["se"], [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_critical_value_is_nan_when_every_draw_moves_a_zero_scale_column(monkeypatch):
    draws = np.column_stack([np.arange(-10.0, 11.0), np.ones(21)])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = mboot(np.zeros((1, 2)), n_units=1, biters=21)

    assert np.isnan(result["crit_val"])
    np.testing.assert_allclose(result["se"], [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_zero_scale_column_critical_value_from_bootstrap_draws(
    monkeypatch, draws_from_weights, offsetting_spike_weights, inf_func_with_zero_scale_column
):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"),
        "_run_multiplier_bootstrap",
        draws_from_weights(offsetting_spike_weights(200, 0, 1)),
    )

    result = mboot(inf_func_with_zero_scale_column, n_units=200, biters=999)

    bres = result["bres"]
    scale = (
        np.percentile(bres, 75, axis=0, method="inverted_cdf") - np.percentile(bres, 25, axis=0, method="inverted_cdf")
    ) / 1.3489795
    at_zero = bres[:, 1] == 0
    largest = np.max(np.abs(bres[at_zero][:, [0, 2]]) / scale[[0, 2]], axis=1)

    assert scale[1] == 0
    assert 0.5 < at_zero.mean() < 0.7
    np.testing.assert_allclose(result["crit_val"], np.percentile(largest, 95, method="inverted_cdf"), rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_critical_value_without_zero_scale_columns_is_quantile_of_largest_standardized_draw():
    inf_func = np.random.default_rng(3).standard_normal((150, 4))

    result = mboot(inf_func, n_units=150, biters=499, random_state=5)

    bres = result["bres"]
    scale = (
        np.percentile(bres, 75, axis=0, method="inverted_cdf") - np.percentile(bres, 25, axis=0, method="inverted_cdf")
    ) / 1.3489795
    expected = np.percentile(np.max(np.abs(bres / scale), axis=1), 95, method="inverted_cdf")

    assert result["crit_val"] == expected


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_skip_small_scales_ignores_a_zero_scale_column_in_every_draw(monkeypatch, zero_scale_draws):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: zero_scale_draws
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, skip_small_scales=True)

    np.testing.assert_allclose(result["crit_val"], 10 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(result["se"], [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_small_scale_column_is_standardized_unless_skipped(monkeypatch, zero_scale_draws):
    first = zero_scale_draws[:, 0]
    draws = np.column_stack([first, np.where(np.abs(first) == 10, 1.0, 1e-9 * first)])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    standardized = mboot(np.zeros((1, 2)), n_units=1, biters=21)
    skipped = _mboot(np.zeros((1, 2)), n_units=1, biters=21, skip_small_scales=True)

    np.testing.assert_allclose(standardized["crit_val"], 1 / (1e-8 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(skipped["crit_val"], 10 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_array_equal(standardized["se"], skipped["se"])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_skip_small_scales_critical_value_is_nan_when_every_column_is_skipped(monkeypatch, zero_scale_draws):
    first = zero_scale_draws[:, 0]
    column = np.where(np.abs(first) == 10, 1.0, 1e-9 * first)
    draws = np.column_stack([column, 2 * column])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, skip_small_scales=True)

    assert np.all(result["se"] > 0)
    assert np.isnan(result["crit_val"])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_skip_small_scales_with_bootstrap_draws(
    monkeypatch, draws_from_weights, offsetting_spike_weights, inf_func_with_zero_scale_column
):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"),
        "_run_multiplier_bootstrap",
        draws_from_weights(offsetting_spike_weights(200, 0, 1)),
    )

    result = _mboot(inf_func_with_zero_scale_column, n_units=200, biters=999, skip_small_scales=True)

    bres = result["bres"]
    scale = (
        np.percentile(bres, 75, axis=0, method="inverted_cdf") - np.percentile(bres, 25, axis=0, method="inverted_cdf")
    ) / 1.3489795
    largest = np.max(np.abs(bres[:, [0, 2]]) / scale[[0, 2]], axis=1)

    assert scale[1] == 0
    np.testing.assert_allclose(result["crit_val"], np.percentile(largest, 95, method="inverted_cdf"), rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_skip_small_scales_does_not_change_a_critical_value_without_small_scales():
    inf_func = np.random.default_rng(3).standard_normal((150, 4))

    default = mboot(inf_func, n_units=150, biters=499, random_state=5)
    skipped = _mboot(inf_func, n_units=150, biters=499, random_state=5, skip_small_scales=True)

    assert skipped["crit_val"] == default["crit_val"]


def test_mboot_keeps_its_public_signature():
    assert list(inspect.signature(mboot).parameters) == [
        "inf_func",
        "n_units",
        "biters",
        "alpha",
        "cluster",
        "random_state",
    ]


def test_mboot_takes_999_draws_by_default():
    inf_func = np.random.default_rng(3).standard_normal(100)

    result = mboot(inf_func, n_units=100, random_state=1)

    assert result["bres"].shape == (999, 1)


def test_mboot_passes_every_argument_to_the_bootstrap_without_skipping_small_scales():
    inf_func = np.random.default_rng(3).standard_normal((150, 4))
    cluster = np.arange(150) % 30

    result = mboot(inf_func, 150, 199, 0.1, cluster, 5)
    expected = _mboot(
        inf_func, n_units=150, biters=199, alpha=0.1, cluster=cluster, random_state=5, skip_small_scales=False
    )

    assert result.keys() == {"bres", "V", "se", "crit_val"}
    np.testing.assert_array_equal(result["bres"], expected["bres"])
    np.testing.assert_array_equal(result["V"], expected["V"])
    np.testing.assert_array_equal(result["se"], expected["se"])
    assert result["crit_val"] == expected["crit_val"]


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_keep_infinite_draws_ignores_a_column_that_every_draw_leaves_at_zero(monkeypatch, zero_scale_draws):
    draws = np.column_stack([zero_scale_draws[:, 0], np.zeros(21)])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, keep_infinite_draws=True)

    np.testing.assert_allclose(result["crit_val"], 10 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(result["se"], [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("alpha, expected", [(0.05, np.inf), (0.5, 5 / (10 / 1.3489795))])
def test_keep_infinite_draws_gives_an_infinite_critical_value_when_enough_draws_move_a_column(
    monkeypatch, zero_scale_draws, alpha, expected
):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: zero_scale_draws
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, alpha=alpha, keep_infinite_draws=True)

    np.testing.assert_allclose(result["crit_val"], expected, rtol=1e-12)
    np.testing.assert_allclose(result["se"], [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_keep_infinite_draws_counts_the_draws_that_move_a_column_in_the_quantile(monkeypatch):
    first = np.arange(1.0, 51.0)
    draws = np.column_stack([first, np.where(first <= 2, 1.0, 0.0)])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    kept = _mboot(np.zeros((1, 2)), n_units=1, biters=50, keep_infinite_draws=True)
    dropped = mboot(np.zeros((1, 2)), n_units=1, biters=50)

    np.testing.assert_allclose(kept["crit_val"], 50 / (25 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(dropped["crit_val"], 48 / (25 / 1.3489795), rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_keep_infinite_draws_standardizes_a_column_with_a_negligible_scale(monkeypatch, zero_scale_draws):
    first = zero_scale_draws[:, 0]
    draws = np.column_stack([first, 1e-9 * np.where(np.abs(first) == 10, 3 * first, first)])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    kept = _mboot(np.zeros((1, 2)), n_units=1, biters=21, keep_infinite_draws=True)
    screened = mboot(np.zeros((1, 2)), n_units=1, biters=21)

    np.testing.assert_allclose(kept["crit_val"], 3 * 1.3489795, rtol=1e-12)
    np.testing.assert_allclose(kept["se"], [10 / 1.3489795, 1e-8 / 1.3489795], rtol=1e-12)
    np.testing.assert_allclose(screened["crit_val"], 1.3489795, rtol=1e-12)
    assert np.isnan(screened["se"][1])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_keep_infinite_draws_ignores_a_column_of_missing_draws(monkeypatch, zero_scale_draws):
    draws = np.column_stack([zero_scale_draws[:, 0], np.full(21, np.nan)])
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, keep_infinite_draws=True)

    np.testing.assert_allclose(result["crit_val"], 10 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(result["se"], [10 / 1.3489795, np.nan], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_keep_infinite_draws_critical_value_is_negative_infinity_without_a_deviation_to_compare(monkeypatch):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: np.zeros((21, 2))
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, keep_infinite_draws=True)

    assert result["crit_val"] == -np.inf
    np.testing.assert_array_equal(result["se"], [0.0, 0.0])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_keep_infinite_draws_matches_the_default_rule_without_small_scales(monkeypatch):
    draws = np.random.default_rng(3).standard_normal((499, 4))
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: draws
    )
    cluster = np.arange(150) % 30

    default = _mboot(np.zeros((150, 4)), n_units=150, biters=499, alpha=0.1, cluster=cluster)
    kept = _mboot(np.zeros((150, 4)), n_units=150, biters=499, alpha=0.1, cluster=cluster, keep_infinite_draws=True)

    assert kept.keys() == {"bres", "V", "se", "crit_val"}
    np.testing.assert_array_equal(kept["bres"], default["bres"])
    np.testing.assert_array_equal(kept["V"], default["V"])
    np.testing.assert_array_equal(kept["se"], default["se"])
    assert kept["crit_val"] == default["crit_val"]


def test_keep_infinite_draws_and_skip_small_scales_exclude_each_other():
    with pytest.raises(ValueError, match="skip_small_scales and keep_infinite_draws can't both be True"):
        _mboot(np.zeros((5, 2)), n_units=5, biters=9, skip_small_scales=True, keep_infinite_draws=True)


def test_bootstrap_statistics_from_given_draws_reject_both_critical_value_rules():
    with pytest.raises(ValueError, match="skip_small_scales and keep_infinite_draws can't both be True"):
        _mboot_from_draws(np.ones((9, 2)), 1.0, skip_small_scales=True, keep_infinite_draws=True)


def test_mboot_rejects_the_removed_alp_argument():
    inf_func = np.random.default_rng(0).standard_normal((50, 2))

    with pytest.raises(TypeError, match="unexpected keyword argument 'alp'"):
        mboot(inf_func, n_units=50, biters=99, alp=0.1)
