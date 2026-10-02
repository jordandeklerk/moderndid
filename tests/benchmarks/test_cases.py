from types import SimpleNamespace

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from benchmarks.cases import Workload, make_panel
from benchmarks.common import check_aggregation, check_estimate, make_estimator


@pytest.mark.parametrize("estimator", ["att_gt", "ddd", "cont_did", "did_multiplegt"])
def test_panel_is_seeded_and_balanced(estimator):
    workload = Workload(n_units=240, n_covariates=2)
    data = make_panel(workload, estimator)
    assert_frame_equal(data, make_panel(workload, estimator))
    assert len(data) == workload.n_units * workload.n_periods
    assert data.group_by("id").len()["len"].to_list() == [workload.n_periods] * workload.n_units
    units = data.unique(subset="id")
    counts = units.group_by("group", "partition").len()
    assert counts.height == (workload.n_cohorts + 1) * 2
    assert counts["len"].to_list() == [40] * 6
    unique_covariates = data.group_by("id").agg("x1").get_column("x1").list.n_unique().to_list()
    expected = workload.n_periods if estimator == "did_multiplegt" else 1
    assert unique_covariates == [expected] * workload.n_units


def test_intertemporal_covariate_differences_are_full_rank():
    workload = Workload(n_units=240, n_covariates=4)
    data = make_panel(workload, "did_multiplegt")
    values = data.select("x1", "x2", "x3", "x4").to_numpy().reshape(workload.n_units, workload.n_periods, 4)
    groups = data.unique(subset="id", maintain_order=True)["group"].to_numpy()
    for horizon in (1, 2):
        for period in range(horizon, workload.n_periods):
            differences = values[groups == 0, period] - values[groups == 0, period - horizon]
            assert np.linalg.matrix_rank(differences) == workload.n_covariates


@pytest.mark.parametrize(
    ("estimator", "profile"),
    [("att_gt", "missing"), ("cont_did", "covariates"), ("cont_did", "normalized"), ("did_multiplegt", "ipw")],
)
def test_unknown_or_unsupported_workload_is_rejected(estimator, profile):
    with pytest.raises(ValueError):
        make_estimator(estimator, profile)


def test_intertemporal_bootstrap_has_fixed_refit_count():
    estimate = make_estimator("did_multiplegt", "bootstrap")
    assert estimate.keywords["biters"] == 199


@pytest.mark.parametrize(
    ("estimates", "errors"),
    [
        ([np.nan, 1.0], [0.2, 0.2]),
        ([1.0, 1.0], [np.inf, 0.2]),
        ([1.0, 1.0], [-0.2, 0.2]),
        ([1.0, 1.0], [0.2]),
        ([], []),
    ],
)
def test_estimate_check_rejects_invalid_values_or_shape(estimates, errors):
    result = SimpleNamespace(att_gt=np.array(estimates), se_gt=np.array(errors), groups=[3, 3], times=[2, 3])
    with pytest.raises(AssertionError):
        check_estimate("att_gt", result)


def test_estimate_check_accepts_finite_values():
    result = SimpleNamespace(att_gt=[0.0, 1.0], se_gt=[0.1, 0.2], groups=[3, 3], times=[2, 3])
    check_estimate("att_gt", result)


@pytest.mark.parametrize("errors", [[np.nan], [-0.2], []])
def test_intertemporal_check_rejects_invalid_placebo_errors(errors):
    result = SimpleNamespace(
        effects=SimpleNamespace(estimates=[1.0, 1.1], std_errors=[0.2, 0.2]),
        placebos=SimpleNamespace(estimates=[0.0], std_errors=errors),
    )
    with pytest.raises(AssertionError):
        check_estimate("did_multiplegt", result)


@pytest.mark.parametrize("reference_error", [0.0, np.nan])
def test_triple_difference_check_accepts_reference_period(reference_error):
    result = SimpleNamespace(att=[0.0, 1.0], se=[reference_error, 0.2], groups=[3, 3], times=[2, 3])
    check_estimate("ddd", result)


@pytest.mark.parametrize(
    ("estimates", "errors"),
    [([0.1, 1.0], [np.nan, 0.2]), ([0.0, 1.0], [-1.0, 0.2]), ([0.0, 1.0], [np.nan, np.nan])],
)
def test_triple_difference_check_rejects_invalid_reference_or_estimate(estimates, errors):
    result = SimpleNamespace(att=estimates, se=errors, groups=[3, 3], times=[2, 3])
    with pytest.raises(AssertionError):
        check_estimate("ddd", result)


@pytest.mark.parametrize(("estimate", "error"), [(np.nan, 0.2), (1.0, np.inf), (1.0, -0.2)])
def test_aggregation_check_rejects_invalid_values(estimate, error):
    result = SimpleNamespace(overall_att=estimate, overall_se=error)
    with pytest.raises(AssertionError):
        check_aggregation(result)
