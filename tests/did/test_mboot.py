"""Tests for the multiplier bootstrap function."""

import inspect
from importlib import import_module

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid.did import mboot
from moderndid.did.mboot import _mboot


def test_basic_functionality():
    n = 200
    inf_func = np.random.normal(0, 1, n)
    result = mboot(inf_func, n_units=n, biters=99)

    assert isinstance(result, dict)
    assert "bres" in result
    assert "V" in result
    assert "se" in result
    assert "crit_val" in result

    assert result["bres"].shape[0] == 99
    assert len(result["se"]) == 1
    assert isinstance(result["crit_val"], float | np.floating)


def test_multivariate_influence_function():
    n = 200
    k = 3
    inf_func = np.random.normal(0, 1, (n, k))
    result = mboot(inf_func, n_units=n, biters=99)

    assert result["bres"].shape == (99, k)
    assert result["V"].shape == (k, k)
    assert result["se"].shape == (k,)


def test_clustering():
    n_units = 100
    n_clusters = 20

    cluster = np.repeat(np.arange(n_clusters), n_units // n_clusters)
    inf_func = np.random.normal(0, 1, n_units)

    result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster)

    assert result["bres"].shape == (99, 1)
    assert result["se"].shape == (1,)


def test_degenerate_columns():
    n = 200
    inf_func = np.column_stack([np.random.normal(0, 1, n), np.zeros(n), np.random.normal(0, 1, n)])

    result = mboot(inf_func, n_units=n, biters=99)

    assert not np.isnan(result["se"][0])
    assert np.isnan(result["se"][1])
    assert not np.isnan(result["se"][2])


def test_reproducibility():
    n = 100
    inf_func = np.random.normal(0, 1, n)

    result1 = mboot(inf_func, n_units=n, biters=99, random_state=42)
    result2 = mboot(inf_func, n_units=n, biters=99, random_state=42)

    np.testing.assert_array_equal(result1["bres"], result2["bres"])
    assert result1["se"] == result2["se"]
    assert result1["crit_val"] == result2["crit_val"]


def test_invalid_inputs():
    with pytest.raises(ValueError, match="cluster must have length"):
        mboot(np.random.normal(0, 1, 100), n_units=100, cluster=np.arange(50))


@pytest.mark.parametrize(
    "alp,expected_range",
    [
        (0.01, (2.0, 4.0)),
        (0.05, (1.5, 3.0)),
        (0.10, (1.2, 2.5)),
        (0.20, (1.0, 2.2)),
    ],
)
def test_significance_levels(alp, expected_range):
    n = 200
    inf_func = np.random.normal(0, 1, n)
    result = mboot(inf_func, n_units=n, biters=999, alp=alp, random_state=42)

    assert isinstance(result["crit_val"], float | np.floating)
    assert expected_range[0] < result["crit_val"] < expected_range[1]


def test_significance_level_ordering():
    n = 200
    inf_func = np.random.normal(0, 1, n)

    result_01 = mboot(inf_func, n_units=n, biters=999, alp=0.01, random_state=42)
    result_10 = mboot(inf_func, n_units=n, biters=999, alp=0.10, random_state=42)

    assert result_01["crit_val"] > result_10["crit_val"]


@pytest.mark.parametrize(
    "contamination_type,contamination_indices",
    [
        ("nan", slice(10, 20)),
        ("inf", slice(30, 35)),
        ("neginf", slice(40, 45)),
        ("mixed", slice(50, 60)),
    ],
)
def test_contaminated_influence_functions(contamination_type, contamination_indices):
    n = 200
    k = 3
    inf_func = np.random.normal(0, 1, (n, k))

    if contamination_type == "nan":
        inf_func[contamination_indices, 0] = np.nan
    elif contamination_type == "inf":
        inf_func[contamination_indices, 1] = np.inf
    elif contamination_type == "neginf":
        inf_func[contamination_indices, 1] = -np.inf
    elif contamination_type == "mixed":
        inf_func[contamination_indices, 0] = np.nan
        inf_func[contamination_indices, 1] = np.inf
        inf_func[contamination_indices, 2] = -np.inf

    result = mboot(inf_func, n_units=n, biters=99)

    assert result["bres"].shape == (99, k)
    assert result["se"].shape == (k,)
    assert "V" in result
    assert "crit_val" in result


def test_all_nan_influence_function():
    n = 100
    k = 2
    inf_func = np.full((n, k), np.nan)

    result = mboot(inf_func, n_units=n, biters=99)

    assert np.all(np.isnan(result["se"]))
    assert np.isnan(result["crit_val"])
    assert result["V"].shape == (0, 0) or np.all(np.isnan(result["V"]))


@pytest.mark.parametrize(
    "n_units,expected_behavior",
    [
        (1, "single_unit"),
        (2, "minimal"),
        (10, "small"),
        (1000, "large"),
    ],
)
def test_different_sample_sizes(n_units, expected_behavior):
    inf_func = np.random.normal(0, 1, n_units)
    result = mboot(inf_func, n_units=n_units, biters=99)

    assert result["bres"].shape == (99, 1)
    assert result["se"].shape == (1,)
    assert len(result["se"]) == 1

    if expected_behavior == "single_unit":
        assert result["se"][0] >= 0
    elif expected_behavior == "large":
        assert result["se"][0] < 0.1


@pytest.mark.parametrize(
    "cluster_pattern",
    [
        "equal_sizes",
        "varying_sizes",
        "singleton_clusters",
        "unbalanced",
    ],
)
def test_clustering_patterns(cluster_pattern):
    n_units = 120

    if cluster_pattern == "equal_sizes":
        cluster = np.repeat(np.arange(20), 6)
    elif cluster_pattern == "varying_sizes":
        cluster_sizes = [5, 10, 15, 20, 25, 45]
        cluster = np.concatenate([np.full(size, i) for i, size in enumerate(cluster_sizes)])
    elif cluster_pattern == "singleton_clusters":
        cluster = np.arange(n_units)
    elif cluster_pattern == "unbalanced":
        cluster = np.concatenate([np.zeros(100), np.ones(20)])
    else:
        raise ValueError(f"Unknown cluster_pattern: {cluster_pattern}")

    inf_func = np.random.normal(0, 1, n_units)
    result = mboot(inf_func, n_units=n_units, biters=99, cluster=cluster)

    assert result["bres"].shape == (99, 1)
    assert result["se"].shape == (1,)
    assert not np.isnan(result["se"][0])


def test_mammen_weights_properties():
    n = 10000
    rng = np.random.default_rng(42)

    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    pkappa = 0.5 * (1 + sqrt5) / sqrt5

    v = rng.binomial(1, pkappa, size=n)
    v = np.where(v == 1, k1, k2)

    assert np.abs(np.mean(v)) < 0.1
    assert np.abs(np.var(v) - 1.0) < 0.1

    skewness = np.mean((v - np.mean(v)) ** 3) / np.std(v) ** 3
    assert np.abs(skewness) < 1.5


@pytest.mark.parametrize(
    "inf_func_type",
    [
        "constant",
        "linear_trend",
        "high_variance",
        "low_variance",
    ],
)
def test_different_influence_functions(inf_func_type):
    n = 200

    if inf_func_type == "constant":
        inf_func = np.full(n, 5.0)
    elif inf_func_type == "linear_trend":
        inf_func = np.linspace(0, 10, n)
    elif inf_func_type == "high_variance":
        inf_func = np.random.normal(0, 100, n)
    elif inf_func_type == "low_variance":
        inf_func = np.random.normal(0, 0.01, n)
    else:
        raise ValueError(f"Unknown inf_func_type: {inf_func_type}")

    result = mboot(inf_func, n_units=n, biters=99, random_state=42)

    assert result["se"][0] > 0
    assert not np.isnan(result["se"][0])

    if inf_func_type == "constant":
        assert result["se"][0] > 0
        assert result["se"][0] < 1.0
    elif inf_func_type == "high_variance":
        assert result["se"][0] > 1.0
    elif inf_func_type == "low_variance":
        assert result["se"][0] < 0.01


@pytest.mark.parametrize("scale", [1e-10, 1e-5, 1.0, 1e5, 1e10])
def test_scale_invariance(scale):
    n = 100
    base_inf_func = np.random.normal(0, 1, n)
    scaled_inf_func = base_inf_func * scale

    result_base = mboot(base_inf_func, n_units=n, biters=99, random_state=42)
    result_scaled = mboot(scaled_inf_func, n_units=n, biters=99, random_state=42)

    if scale >= 1e-3:
        assert np.abs(result_scaled["se"][0] / result_base["se"][0] - scale) < scale * 0.3
        assert not np.isnan(result_scaled["se"][0])
        assert not np.isinf(result_scaled["se"][0])
    else:
        assert result_scaled["se"][0] >= 0 or np.isnan(result_scaled["se"][0])


def test_variance_covariance_matrix_properties():
    n = 200
    k = 4

    np.random.seed(42)
    correlation_matrix = np.eye(k)
    correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.5
    correlation_matrix[2, 3] = correlation_matrix[3, 2] = -0.3

    inf_func = np.random.multivariate_normal(np.zeros(k), correlation_matrix, n)

    result = mboot(inf_func, n_units=n, biters=999)

    V = result["V"]
    assert V.shape == (k, k)
    assert np.allclose(V, V.T, rtol=1e-10)

    eigenvalues = np.linalg.eigvals(V)
    assert np.all(eigenvalues >= -1e-10)

    assert np.all(np.diag(V) >= 0)

    expected_corr_sign = np.sign(V[0, 1])
    assert expected_corr_sign == np.sign(0.5) or np.abs(V[0, 1]) < 0.1


@pytest.mark.parametrize(
    "distribution,df_or_param",
    [
        ("normal", None),
        ("uniform", None),
        ("t", 3),
        ("t", 10),
        ("chi2", 5),
        ("exponential", None),
    ],
)
def test_critical_values_by_distribution(distribution, df_or_param):
    n = 300
    biters = 999

    if distribution == "normal":
        inf_func = np.random.normal(0, 1, n)
    elif distribution == "uniform":
        inf_func = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    elif distribution == "t":
        inf_func = np.random.standard_t(df=df_or_param, size=n)
    elif distribution == "chi2":
        inf_func = np.random.chisquare(df=df_or_param, size=n) - df_or_param
    elif distribution == "exponential":
        inf_func = np.random.exponential(1, n) - 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    result = mboot(inf_func, n_units=n, biters=biters, random_state=42)

    assert result["crit_val"] > 0
    assert not np.isnan(result["crit_val"])
    assert not np.isinf(result["crit_val"])

    if distribution == "normal":
        assert 1.5 < result["crit_val"] < 2.5
    elif distribution == "t" and df_or_param == 3:
        assert result["crit_val"] > 1.5


@pytest.mark.parametrize(
    "n_obs,n_params",
    [
        (100, 1),
        (100, 5),
        (100, 10),
        (1000, 1),
        (1000, 20),
    ],
)
def test_dimensions_handling(n_obs, n_params):
    inf_func = np.random.normal(0, 1, (n_obs, n_params))

    result = mboot(inf_func, n_units=n_obs, biters=99)

    assert result["bres"].shape == (99, n_params)
    assert result["se"].shape == (n_params,)
    assert result["V"].shape[0] <= n_params
    assert result["V"].shape[1] <= n_params


def test_quantile_uses_inverted_cdf():
    rng = np.random.default_rng(123)
    n = 200
    inf_func = rng.normal(0, 1, n)
    result = mboot(inf_func, n_units=n, biters=999, random_state=42)

    bres = result["bres"]
    col_sums_sq = np.sum(bres**2, axis=0)
    ndg_dim = (~np.isnan(col_sums_sq)) & (col_sums_sq > np.sqrt(np.finfo(float).eps) * 10)
    bres_clean = bres[:, ndg_dim]

    q75_step = np.percentile(bres_clean, 75, axis=0, method="inverted_cdf")
    q25_step = np.percentile(bres_clean, 25, axis=0, method="inverted_cdf")
    se_step = (q75_step - q25_step) / 1.3489795

    q75_linear = np.percentile(bres_clean, 75, axis=0)
    q25_linear = np.percentile(bres_clean, 25, axis=0)
    se_linear = (q75_linear - q25_linear) / 1.3489795

    assert not np.allclose(se_step, se_linear), (
        "Step-function and linear quantiles should differ for discrete bootstrap samples"
    )

    expected_se = se_step / np.sqrt(n)
    np.testing.assert_allclose(result["se"][ndg_dim], expected_se, rtol=1e-10)


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
        "alp",
        "cluster",
        "random_state",
    ]


def test_mboot_passes_every_argument_to_the_bootstrap_without_skipping_small_scales():
    inf_func = np.random.default_rng(3).standard_normal((150, 4))
    cluster = np.arange(150) % 30

    result = mboot(inf_func, 150, 199, 0.1, cluster, 5)
    expected = _mboot(
        inf_func, n_units=150, biters=199, alp=0.1, cluster=cluster, random_state=5, skip_small_scales=False
    )

    assert result.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])


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
@pytest.mark.parametrize("alp, expected", [(0.05, np.inf), (0.5, 5 / (10 / 1.3489795))])
def test_keep_infinite_draws_gives_an_infinite_critical_value_when_enough_draws_move_a_column(
    monkeypatch, zero_scale_draws, alp, expected
):
    monkeypatch.setattr(
        import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", lambda *args, **kwargs: zero_scale_draws
    )

    result = _mboot(np.zeros((1, 2)), n_units=1, biters=21, alp=alp, keep_infinite_draws=True)

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

    default = _mboot(np.zeros((150, 4)), n_units=150, biters=499, alp=0.1, cluster=cluster)
    kept = _mboot(np.zeros((150, 4)), n_units=150, biters=499, alp=0.1, cluster=cluster, keep_infinite_draws=True)

    assert kept.keys() == default.keys()
    for key in default:
        np.testing.assert_array_equal(kept[key], default[key])


def test_keep_infinite_draws_and_skip_small_scales_exclude_each_other():
    with pytest.raises(ValueError, match="skip_small_scales"):
        _mboot(np.zeros((5, 2)), n_units=5, biters=9, skip_small_scales=True, keep_infinite_draws=True)
