# pylint: disable=redefined-outer-name
"""Test helpers for Sun and Abraham estimator."""

import warnings

import numpy as np
import pytest

from pydid.honestdid.helpers import (
    aggregate_to_event_study,
    create_period_interactions,
    estimate_sunab_model,
    find_never_always_treated,
)


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n = 100
    y = np.random.randn(n)
    covariates = np.column_stack((np.ones(n), np.random.randn(n, 2)))
    interaction_matrix = np.random.randn(n, 3)
    cohort = np.repeat([0, 1, 2, np.inf], 25)
    period = np.tile([-1, 0, 1, 2], 25)
    period_values = np.array([-1, 0, 1, 2])
    weights = np.ones(n)
    return {
        "y": y,
        "covariates": covariates,
        "interaction_matrix": interaction_matrix,
        "cohort": cohort,
        "period": period,
        "period_values": period_values,
        "weights": weights,
    }


def test_estimate_sunab_model_basic(simple_data):
    result = estimate_sunab_model(
        simple_data["y"],
        simple_data["covariates"],
        simple_data["interaction_matrix"],
        simple_data["cohort"],
        simple_data["period"],
        simple_data["period_values"],
        simple_data["weights"],
        att=False,
        no_agg=False,
    )

    assert isinstance(result, dict)
    assert "att_by_event" in result
    assert "se_by_event" in result
    assert "vcov_event" in result
    assert "aggregated" in result
    assert result["aggregated"] is True
    assert len(result["att_by_event"]) == len(simple_data["period_values"])


def test_estimate_sunab_model_no_agg(simple_data):
    result = estimate_sunab_model(
        simple_data["y"],
        simple_data["covariates"],
        simple_data["interaction_matrix"],
        simple_data["cohort"],
        simple_data["period"],
        simple_data["period_values"],
        simple_data["weights"],
        att=False,
        no_agg=True,
    )

    assert result["aggregated"] is False
    assert "coefs" in result
    assert "vcov" in result
    assert "se" in result
    assert len(result["coefs"]) == simple_data["interaction_matrix"].shape[1]


def test_estimate_sunab_model_with_att(simple_data):
    result = estimate_sunab_model(
        simple_data["y"],
        simple_data["covariates"],
        simple_data["interaction_matrix"],
        simple_data["cohort"],
        simple_data["period"],
        simple_data["period_values"],
        simple_data["weights"],
        att=True,
        no_agg=False,
    )

    assert "att" in result
    assert "se_att" in result
    assert result["att"] is not None
    assert result["se_att"] is not None


def test_estimate_sunab_model_nan_handling(simple_data):
    interaction_with_nan = simple_data["interaction_matrix"].copy()
    interaction_with_nan[10:15, :] = np.nan

    result = estimate_sunab_model(
        simple_data["y"],
        simple_data["covariates"],
        interaction_with_nan,
        simple_data["cohort"],
        simple_data["period"],
        simple_data["period_values"],
        simple_data["weights"],
        att=False,
        no_agg=False,
    )

    assert result is not None
    assert "n_obs" in result


def test_estimate_sunab_model_insufficient_obs():
    n = 5
    y = np.random.randn(n)
    covariates = np.random.randn(n, 10)
    interaction_matrix = np.random.randn(n, 10)
    cohort = np.arange(n)
    period = np.arange(n)
    period_values = np.unique(period)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = estimate_sunab_model(
            y, covariates, interaction_matrix, cohort, period, period_values, None, False, False
        )
        assert result is None
        assert len(w) == 1
        assert "Insufficient observations" in str(w[0].message)


def test_estimate_sunab_model_singular_matrix(simple_data):
    covariates = np.column_stack((np.ones(100), np.arange(100), np.arange(100) * 2))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = estimate_sunab_model(
            simple_data["y"],
            covariates,
            simple_data["interaction_matrix"],
            simple_data["cohort"],
            simple_data["period"],
            simple_data["period_values"],
            simple_data["weights"],
            att=False,
            no_agg=False,
        )
        assert result is not None or len(w) > 0


def test_find_never_always_treated_basic():
    cohort_int = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    period = np.array([-1, 0, 1, -1, -2, -3, 0, 1, 2])
    n_cohorts = 3

    never_treated, always_treated_idx = find_never_always_treated(cohort_int, period, n_cohorts)

    assert len(never_treated) == 1
    assert 1 in never_treated
    assert len(always_treated_idx) == 3
    assert set(always_treated_idx) == {6, 7, 8}


def test_find_never_always_treated_all_treated():
    cohort_int = np.array([0, 0, 1, 1])
    period = np.array([0, 1, 2, 3])
    n_cohorts = 2

    never_treated, always_treated_idx = find_never_always_treated(cohort_int, period, n_cohorts)

    assert len(never_treated) == 0
    assert len(always_treated_idx) == 4


def test_find_never_always_treated_mixed():
    cohort_int = np.array([0, 0, 0, 1, 1, 1])
    period = np.array([-1, 0, 1, -2, -1, 0])
    n_cohorts = 2

    never_treated, always_treated_idx = find_never_always_treated(cohort_int, period, n_cohorts)

    assert len(never_treated) == 0
    assert len(always_treated_idx) == 0


def test_create_period_interactions_basic():
    period = np.array([-1, 0, 1, -1, 0, 1])
    period_values = np.array([-1, 0, 1])
    is_ref_cohort = np.array([False, False, False, True, True, True])
    is_always_treated = np.array([False, False, False, False, False, False])
    n_total = 6

    interactions = create_period_interactions(period, period_values, is_ref_cohort, is_always_treated, n_total, None)

    assert interactions.shape == (n_total, len(period_values))
    assert np.all(interactions[is_ref_cohort] == 0)
    assert interactions[0, 0] == 1
    assert interactions[1, 1] == 1
    assert interactions[2, 2] == 1


def test_create_period_interactions_with_always_treated():
    period = np.array([-1, 0, 1, 2])
    period_values = np.array([-1, 0, 1, 2])
    is_ref_cohort = np.array([False, False, False, False])
    is_always_treated = np.array([False, False, True, True])
    n_total = 4

    interactions = create_period_interactions(period, period_values, is_ref_cohort, is_always_treated, n_total, None)

    assert interactions.shape == (n_total, len(period_values))
    assert np.all(np.isnan(interactions[is_always_treated]))
    assert interactions[0, 0] == 1
    assert interactions[1, 1] == 1


def test_create_period_interactions_with_valid_mask():
    period = np.array([-1, 0, 1])
    period_values = np.array([-1, 0, 1])
    is_ref_cohort = np.array([False, False, False])
    is_always_treated = np.array([False, False, False])
    n_total = 5
    valid_obs_mask = np.array([True, True, True, False, False])

    interactions = create_period_interactions(
        period, period_values, is_ref_cohort, is_always_treated, n_total, valid_obs_mask
    )

    assert interactions.shape == (n_total, len(period_values))
    assert np.all(interactions[3:] == 0)


def test_aggregate_to_event_study_basic():
    coefs = np.array([0.1, 0.2, 0.3])
    vcov = np.diag([0.01, 0.02, 0.03])
    interactions = np.eye(3)
    cohort = np.array([0, 1, 2])
    period_values = np.array([-1, 0, 1])
    weights = np.ones(3)

    result = aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att=False)

    assert "att_by_event" in result
    assert "se_by_event" in result
    assert "vcov_event" in result
    assert "cohort_shares" in result
    assert np.allclose(result["att_by_event"], coefs)
    assert result["att"] is None


def test_aggregate_to_event_study_with_att():
    coefs = np.array([0.0, 0.5, 0.6])
    vcov = np.diag([0.01, 0.02, 0.03])
    interactions = np.eye(3)
    cohort = np.array([0, 1, 2])
    period_values = np.array([-1, 0, 1])
    weights = np.ones(3)

    result = aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att=True)

    assert result["att"] is not None
    assert result["se_att"] is not None
    expected_att = (0.5 + 0.6) / 2
    assert np.isclose(result["att"], expected_att)


def test_aggregate_to_event_study_cohort_shares():
    coefs = np.array([0.1, 0.2])
    vcov = np.diag([0.01, 0.02])
    interactions = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
        ]
    )
    cohort = np.array([0, 0, 1])
    period_values = np.array([0, 1])
    weights = np.ones(3)

    result = aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att=False)

    assert len(result["cohort_shares"]) == 2
    assert np.isclose(result["cohort_shares"][0], 2 / 3)
    assert np.isclose(result["cohort_shares"][1], 1 / 3)


def test_aggregate_to_event_study_weighted():
    coefs = np.array([0.1, 0.2])
    vcov = np.diag([0.01, 0.02])
    interactions = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    cohort = np.array([0, 1])
    period_values = np.array([0, 1])
    weights = np.array([2.0, 1.0])

    result = aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att=False)

    assert np.allclose(result["att_by_event"], coefs)


def test_aggregate_to_event_study_multi_cohort_period():
    n_periods = 2
    coefs = np.array([0.1, 0.2, 0.3, 0.4])
    vcov = np.diag([0.01, 0.02, 0.03, 0.04])

    interactions = np.zeros((4, 4))
    interactions[0, 0] = 1
    interactions[1, 1] = 1
    interactions[2, 2] = 1
    interactions[3, 3] = 1

    cohort = np.array([0, 0, 1, 1])
    period_values = np.array([0, 1])
    weights = np.ones(4)

    result = aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att=False)

    assert len(result["att_by_event"]) == n_periods
    expected_period0 = (0.1 + 0.3) / 2
    assert np.isclose(result["att_by_event"][0], expected_period0)


@pytest.mark.parametrize("compute_att", [True, False])
def test_aggregate_to_event_study_empty_period(compute_att):
    coefs = np.array([0.1, 0.0, 0.3])
    vcov = np.diag([0.01, 0.0, 0.03])
    interactions = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    cohort = np.array([0, 1])
    period_values = np.array([0, 1, 2])
    weights = np.ones(2)

    result = aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att)

    assert len(result["att_by_event"]) == len(period_values)
    assert result["att_by_event"][1] == 0
