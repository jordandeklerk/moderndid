"""Tests for processing aggregate treatment effects."""

import numpy as np
import pytest
import scipy.stats

from moderndid.didcont.panel import (
    PTEAggteResult,
    aggregate_att_gt,
    overall_weights,
)
from moderndid.didcont.panel.process_aggte import (
    check_critical_value,
    get_aggregated_influence_function,
    get_se,
    safe_normalize,
    set_small_se_to_nan,
    weight_influence_function_from_att_indices,
    weight_influence_function_from_groups,
)


def test_aggregate_att_gt_overall(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="overall")

    assert isinstance(result, PTEAggteResult)
    assert result.aggregation_type == "overall"
    assert result.overall_att is not None
    assert result.overall_se is not None
    assert result.critical_value is not None
    assert result.influence_func is not None
    assert "overall" in result.influence_func


def test_aggregate_att_gt_dynamic(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="dynamic", min_event_time=-2, max_event_time=2)

    assert isinstance(result, PTEAggteResult)
    assert result.aggregation_type == "dynamic"
    assert result.event_times is not None
    assert result.att_by_event is not None
    assert result.se_by_event is not None
    assert len(result.att_by_event) == len(result.event_times)
    assert len(result.se_by_event) == len(result.event_times)
    assert result.min_event_time == -2
    assert result.max_event_time == 2


def test_aggregate_att_gt_group(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="group")

    assert isinstance(result, PTEAggteResult)
    assert result.aggregation_type == "group"
    assert result.event_times is not None
    assert result.att_by_event is not None
    assert result.se_by_event is not None
    assert len(result.att_by_event) == len(np.unique(mock_att_gt_result.groups[mock_att_gt_result.groups > 0]))


def test_aggregate_att_gt_balanced_event(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="dynamic", balance_event=1)

    assert result.balance_event == 1
    assert result.aggregation_type == "dynamic"


def test_aggregate_att_gt_no_valid_periods(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="dynamic", min_event_time=100, max_event_time=101)

    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)
    assert len(result.att_by_event) == 0


def test_overall_weights_basic(att_gt_result_with_data):
    weights_dict = overall_weights(att_gt_result_with_data)

    assert "groups" in weights_dict
    assert "times" in weights_dict
    assert "weights" in weights_dict
    assert len(weights_dict["weights"]) == len(weights_dict["groups"])
    assert len(weights_dict["weights"]) == len(weights_dict["times"])
    assert np.isclose(np.sum(weights_dict["weights"]), 1.0, atol=1e-10)


def test_overall_weights_max_event_time(att_gt_result_with_data):
    weights_dict = overall_weights(att_gt_result_with_data, max_event_time=2)

    assert np.isclose(np.sum(weights_dict["weights"]), 1.0, atol=1e-10)


def test_overall_weights_no_data_error(mock_att_gt_result):
    mock_att_gt_result.pte_params = None

    with pytest.raises(ValueError, match="overall_weights requires pte_params"):
        overall_weights(mock_att_gt_result)


def test_safe_normalize_basic():
    x = np.array([1, 2, 3, 4])
    normalized = safe_normalize(x)

    assert np.isclose(np.sum(normalized), 1.0)
    assert np.allclose(normalized, x / 10.0)


def test_safe_normalize_zero_sum():
    x = np.array([0, 0, 0])
    normalized = safe_normalize(x)

    assert np.allclose(normalized, [1 / 3, 1 / 3, 1 / 3])


def test_safe_normalize_negative_sum():
    x = np.array([-1, -2, -3])
    normalized = safe_normalize(x)

    assert np.allclose(normalized, [1 / 3, 1 / 3, 1 / 3])


def test_safe_normalize_infinite():
    x = np.array([1, np.inf, 2])
    normalized = safe_normalize(x)

    assert np.allclose(normalized, [1 / 3, 1 / 3, 1 / 3])


def test_set_small_se_to_nan():
    threshold = np.sqrt(np.finfo(float).eps) * 10.0

    se_large = 0.1
    assert set_small_se_to_nan(se_large) == se_large

    se_small = 1e-15
    assert np.isnan(set_small_se_to_nan(se_small))

    se_threshold = threshold
    assert np.isnan(set_small_se_to_nan(se_threshold))


def test_check_critical_value_valid():
    alpha = 0.05

    crit_val = 2.5
    result = check_critical_value(crit_val, alpha)
    assert result == crit_val


def test_check_critical_value_too_small():
    alpha = 0.05
    pointwise = scipy.stats.norm.ppf(1 - alpha / 2)

    crit_val = 1.5
    with pytest.warns(UserWarning, match="Simultaneous band smaller than pointwise"):
        result = check_critical_value(crit_val, alpha)
    assert result == pointwise


def test_check_critical_value_nan():
    alpha = 0.05
    pointwise = scipy.stats.norm.ppf(1 - alpha / 2)

    with pytest.warns(UserWarning, match="Simultaneous critical value is NA/Inf"):
        result = check_critical_value(np.nan, alpha)
    assert result == pointwise


def test_check_critical_value_too_large():
    alpha = 0.05

    with pytest.warns(UserWarning, match="Simultaneous critical value is very large"):
        result = check_critical_value(8.0, alpha)
    assert result == 8.0


def test_get_se_bootstrap(simple_influence_func):
    se = get_se(simple_influence_func, bootstrap=True, bootstrap_iterations=100, alpha=0.05)

    assert isinstance(se, float)
    assert se > 0


def test_get_se_analytical(simple_influence_func):
    se = get_se(simple_influence_func[:, :1], bootstrap=False)

    assert isinstance(se, float)
    assert se > 0

    n = simple_influence_func.shape[0]
    expected_se = float(np.sqrt(np.mean(simple_influence_func[:, 0] ** 2) / n))
    assert np.isclose(se, expected_se)


def test_get_aggregated_influence_function_basic():
    np.random.seed(42)
    att = np.array([0.1, 0.2, 0.3, 0.4])
    influence_func = np.random.randn(100, 4)
    selected_indices = [0, 2]
    weights = np.array([0.6, 0.4])

    result = get_aggregated_influence_function(att, influence_func, selected_indices, weights, None)

    assert result.shape == (100,)
    expected = influence_func[:, 0] * 0.6 + influence_func[:, 2] * 0.4
    assert np.allclose(result, expected)


def test_get_aggregated_influence_function_with_weight_if():
    np.random.seed(42)
    att = np.array([0.1, 0.2, 0.3, 0.4])
    influence_func = np.random.randn(100, 4)
    selected_indices = [1, 3]
    weights = np.array([0.5, 0.5])
    weight_if = np.random.randn(100, 2) * 0.01

    result = get_aggregated_influence_function(att, influence_func, selected_indices, weights, weight_if)

    assert result.shape == (100,)
    expected = influence_func[:, 1] * 0.5 + influence_func[:, 3] * 0.5 + weight_if @ att[[1, 3]]
    assert np.allclose(result, expected)


def test_weight_influence_function_from_att_indices():
    np.random.seed(42)
    n_units = 50
    att_indices = np.array([0, 2, 3])
    pg_att = np.array([0.2, 0.1, 0.3, 0.25, 0.15])
    weights_ind = np.random.uniform(0.5, 2.0, n_units)
    g_units_idx = np.random.choice([1, 2, 3], n_units)
    group_idx = np.array([1, 1, 2, 3, 3])

    result = weight_influence_function_from_att_indices(att_indices, pg_att, weights_ind, g_units_idx, group_idx)

    if result is not None:
        assert result.shape == (n_units, len(att_indices))


def test_weight_influence_function_from_att_indices_zero_denom():
    n_units = 20
    att_indices = np.array([0, 1])
    pg_att = np.array([0, 0, 0.5, 0.5])
    weights_ind = np.ones(n_units)
    g_units_idx = np.ones(n_units)
    group_idx = np.array([1, 1, 2, 2])

    result = weight_influence_function_from_att_indices(att_indices, pg_att, weights_ind, g_units_idx, group_idx)

    assert result is None


def test_weight_influence_function_from_groups():
    np.random.seed(42)
    n_units = 50
    pg_comp = np.array([0.3, 0.4, 0.3])
    weights_ind = np.random.uniform(0.5, 2.0, n_units)
    g_units_idx = np.random.choice([1, 2, 3], n_units)
    group_labels = np.array([1, 2, 3])

    result = weight_influence_function_from_groups(pg_comp, weights_ind, g_units_idx, group_labels)

    if result is not None:
        assert result.shape == (n_units, len(group_labels))


def test_weight_influence_function_from_groups_zero_denom():
    n_units = 20
    pg_comp = np.array([0, 0, 0])
    weights_ind = np.ones(n_units)
    g_units_idx = np.ones(n_units)
    group_labels = np.array([1, 2, 3])

    result = weight_influence_function_from_groups(pg_comp, weights_ind, g_units_idx, group_labels)

    assert result is None


def test_pte_aggte_result_repr(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="overall")

    repr_str = repr(result)
    assert "Aggregate Treatment Effects" in repr_str
    assert "ATT" in repr_str
    assert "Std. Error" in repr_str
    assert "Conf. Interval" in repr_str


def test_pte_aggte_result_str(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="overall")

    str_repr = str(result)
    assert "Aggregate Treatment Effects" in str_repr
    assert str_repr == repr(result)


@pytest.mark.parametrize("aggregation_type", ["overall", "dynamic", "group"])
def test_aggregate_att_gt_all_types(mock_att_gt_result, aggregation_type):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type=aggregation_type)

    assert result.aggregation_type == aggregation_type
    assert result.overall_att is not None
    assert result.overall_se is not None
    assert result.att_gt_result is mock_att_gt_result


def test_aggregate_att_gt_preserves_original(mock_att_gt_result):
    original_att = mock_att_gt_result.att.copy()
    original_inf = mock_att_gt_result.influence_func.copy()

    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="overall")

    assert result is not None
    assert result.overall_att is not None
    assert result.overall_se is not None

    assert np.allclose(mock_att_gt_result.att, original_att)
    assert np.allclose(mock_att_gt_result.influence_func, original_inf)


def test_aggregate_att_gt_dynamic_empty_event_range(mock_att_gt_result):
    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="dynamic", min_event_time=50, max_event_time=51)

    assert result.aggregation_type == "dynamic"
    assert np.isnan(result.overall_att)
    assert np.isnan(result.overall_se)
    assert len(result.att_by_event) == 0
    assert len(result.se_by_event) == 0


def test_aggregate_att_gt_group_no_positive_groups(mock_att_gt_result):
    mock_att_gt_result.groups = np.zeros_like(mock_att_gt_result.groups)

    result = aggregate_att_gt(mock_att_gt_result, aggregation_type="group")

    assert result.aggregation_type == "group"
    assert len(result.att_by_event) == 0
