"""Tests for result converters."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from moderndid.plotting.converters import (
    aggte_to_dataset,
    doseresult_to_dataset,
    honestdid_to_dataset,
    mpresult_to_dataset,
    pteresult_to_dataset,
    sensitivity_to_dataset,
)

from moderndid.did.aggte_obj import AGGTEResult
from moderndid.didcont.estimation.container import PTEResult
from moderndid.didhonest.honest_did import HonestDiDResult
from moderndid.didhonest.sensitivity import OriginalCSResult


def test_mpresult_to_dataset(mp_result):
    ds = mpresult_to_dataset(mp_result)

    assert "att" in ds.data_vars
    assert "se" in ds.data_vars
    assert "ci_lower" in ds.data_vars
    assert "ci_upper" in ds.data_vars

    assert ds["att"].dims == ("group", "time")
    assert ds["att"].shape == (2, 3)

    groups_coord = ds["att"].coords["group"]
    times_coord = ds["att"].coords["time"]

    assert len(groups_coord) == 2
    assert len(times_coord) == 3

    att_val = ds["att"].sel({"group": 2000, "time": 2004})
    assert att_val.item() == pytest.approx(0.5)


def test_aggte_dynamic_to_dataset(aggte_result_dynamic):
    ds = aggte_to_dataset(aggte_result_dynamic)

    assert "att" in ds.data_vars
    assert "se" in ds.data_vars
    assert "ci_lower" in ds.data_vars
    assert "ci_upper" in ds.data_vars

    assert ds["att"].dims == ("event",)
    assert len(ds["att"].coords["event"]) == 5

    att_val = ds["att"].sel({"event": 0})
    assert att_val.item() == pytest.approx(0.8)


def test_aggte_simple_to_dataset(aggte_result_simple):
    ds = aggte_to_dataset(aggte_result_simple)

    assert "overall_att" in ds.data_vars
    assert "overall_se" in ds.data_vars

    assert ds["overall_att"].values[0] == pytest.approx(0.75)
    assert ds["overall_se"].values[0] == pytest.approx(0.12)


def test_aggte_group_aggregation():
    result = AGGTEResult(
        overall_att=0.8,
        overall_se=0.15,
        aggregation_type="group",
        event_times=np.array([2000, 2007]),
        att_by_event=np.array([0.7, 0.9]),
        se_by_event=np.array([0.12, 0.18]),
    )

    ds = aggte_to_dataset(result)
    assert ds["att"].dims == ("group",)
    assert len(ds["att"].coords["group"]) == 2


def test_aggte_calendar_aggregation():
    result = AGGTEResult(
        overall_att=0.65,
        overall_se=0.14,
        aggregation_type="calendar",
        event_times=np.array([2004, 2006, 2007]),
        att_by_event=np.array([0.5, 0.7, 0.8]),
        se_by_event=np.array([0.10, 0.13, 0.16]),
    )

    ds = aggte_to_dataset(result)
    assert ds["att"].dims == ("time",)
    assert len(ds["att"].coords["time"]) == 3


def test_doseresult_to_dataset(dose_result):
    ds = doseresult_to_dataset(dose_result)

    assert "att_d" in ds.data_vars
    assert "se_d" in ds.data_vars
    assert "acrt_d" in ds.data_vars
    assert "se_acrt_d" in ds.data_vars
    assert "ci_lower_att" in ds.data_vars
    assert "ci_upper_att" in ds.data_vars
    assert "ci_lower_acrt" in ds.data_vars
    assert "ci_upper_acrt" in ds.data_vars
    assert "overall_att" in ds.data_vars
    assert "overall_acrt" in ds.data_vars

    assert ds["att_d"].dims == ("dose",)
    assert len(ds["att_d"].coords["dose"]) == 5

    att_val = ds["att_d"].sel({"dose": 3.0})
    assert att_val.item() == pytest.approx(1.5)


def test_honestdid_to_dataset(honest_result):
    ds = honestdid_to_dataset(honest_result)

    assert "lb" in ds.data_vars
    assert "ub" in ds.data_vars

    assert len(ds.dims) == 1
    dim_name = list(ds.dims)[0]
    assert dim_name == "m_value"

    lb_vals = ds["lb"].values
    assert len(lb_vals) == 3
    assert lb_vals[0] == pytest.approx(0.1)


def test_honestdid_empty_dataframe():
    df = pd.DataFrame()
    original_ci = OriginalCSResult(lb=0.3, ub=0.7)
    result = HonestDiDResult(df, original_ci, "smoothness")

    with pytest.raises(ValueError, match="empty robust_ci"):
        honestdid_to_dataset(result)


def test_aggte_missing_event_data():
    result = AGGTEResult(
        overall_att=0.8,
        overall_se=0.15,
        aggregation_type="dynamic",
        event_times=None,
        att_by_event=None,
        se_by_event=None,
    )

    with pytest.raises(ValueError, match="must have event_times"):
        aggte_to_dataset(result)


def test_mpresult_confidence_intervals(mp_result):
    ds = mpresult_to_dataset(mp_result)

    ci_lower = ds["ci_lower"].sel({"group": 2000, "time": 2004})
    ci_upper = ds["ci_upper"].sel({"group": 2000, "time": 2004})
    att = ds["att"].sel({"group": 2000, "time": 2004})
    se = ds["se"].sel({"group": 2000, "time": 2004})

    expected_lower = att.item() - 1.96 * se.item()
    expected_upper = att.item() + 1.96 * se.item()

    assert ci_lower.item() == pytest.approx(expected_lower)
    assert ci_upper.item() == pytest.approx(expected_upper)


def test_doseresult_confidence_intervals(dose_result):
    ds = doseresult_to_dataset(dose_result)

    ci_lower = ds["ci_lower_att"].sel({"dose": 2.0})
    ci_upper = ds["ci_upper_att"].sel({"dose": 2.0})
    att = ds["att_d"].sel({"dose": 2.0})
    se = ds["se_d"].sel({"dose": 2.0})

    expected_lower = att.item() - 1.96 * se.item()
    expected_upper = att.item() + 1.96 * se.item()

    assert ci_lower.item() == pytest.approx(expected_lower)
    assert ci_upper.item() == pytest.approx(expected_upper)


def test_aggte_with_critical_values():
    result = AGGTEResult(
        overall_att=0.7,
        overall_se=0.15,
        aggregation_type="dynamic",
        event_times=np.array([-1, 0, 1]),
        att_by_event=np.array([0.1, 0.8, 1.2]),
        se_by_event=np.array([0.1, 0.12, 0.15]),
        critical_values=np.array([2.5, 2.5, 2.5]),
    )

    ds = aggte_to_dataset(result)

    ci_lower = ds["ci_lower"].sel({"event": 0})
    ci_upper = ds["ci_upper"].sel({"event": 0})
    att = ds["att"].sel({"event": 0})
    se = ds["se"].sel({"event": 0})

    expected_lower = att.item() - 2.5 * se.item()
    expected_upper = att.item() + 2.5 * se.item()

    assert ci_lower.item() == pytest.approx(expected_lower)
    assert ci_upper.item() == pytest.approx(expected_upper)


def test_pteresult_to_dataset(pte_result_with_event_study):
    ds = pteresult_to_dataset(pte_result_with_event_study)

    assert "att" in ds.data_vars
    assert "se" in ds.data_vars
    assert "ci_lower" in ds.data_vars
    assert "ci_upper" in ds.data_vars
    assert "treatment_status" in ds.data_vars

    assert ds["att"].dims == ("event",)
    assert len(ds["att"].coords["event"]) == 5

    treatment_status = ds["treatment_status"].values
    assert treatment_status[0] == "pre"
    assert treatment_status[1] == "pre"
    assert treatment_status[2] == "post"
    assert treatment_status[3] == "post"
    assert treatment_status[4] == "post"


def test_pteresult_to_dataset_no_event_study():
    result = PTEResult(
        att_gt=None,
        overall_att=None,
        event_study=None,
        ptep=None,
    )

    with pytest.raises(ValueError, match="does not contain event study"):
        pteresult_to_dataset(result)


def test_sensitivity_to_dataset(sensitivity_robust_results, sensitivity_original_result):
    ds = sensitivity_to_dataset(sensitivity_robust_results, sensitivity_original_result, param_col="M")

    assert "lb" in ds.data_vars
    assert "ub" in ds.data_vars
    assert "midpoint" in ds.data_vars
    assert "halfwidth" in ds.data_vars

    assert ds["lb"].dims == ("param_value", "method")

    param_values = ds["lb"].coords["param_value"]
    methods = ds["lb"].coords["method"]

    assert len(param_values) == 4
    assert len(methods) == 3


def test_sensitivity_to_dataset_includes_original(sensitivity_robust_results, sensitivity_original_result):
    ds = sensitivity_to_dataset(sensitivity_robust_results, sensitivity_original_result, param_col="M")

    param_values = ds["lb"].coords["param_value"]
    min_robust_m = 0.5

    assert param_values[0] < min_robust_m


def test_sensitivity_to_dataset_midpoint_halfwidth(sensitivity_robust_results, sensitivity_original_result):
    ds = sensitivity_to_dataset(sensitivity_robust_results, sensitivity_original_result, param_col="M")

    lb = ds["lb"].values
    ub = ds["ub"].values
    midpoint = ds["midpoint"].values
    halfwidth = ds["halfwidth"].values

    valid_mask = ~np.isnan(lb) & ~np.isnan(ub)
    expected_midpoint = (lb[valid_mask] + ub[valid_mask]) / 2
    expected_halfwidth = (ub[valid_mask] - lb[valid_mask]) / 2

    assert np.allclose(midpoint[valid_mask], expected_midpoint)
    assert np.allclose(halfwidth[valid_mask], expected_halfwidth)


def test_mpresult_treatment_status(mp_result):
    ds = mpresult_to_dataset(mp_result)

    assert "treatment_status" in ds.data_vars
    assert ds["treatment_status"].dims == ("group", "time")

    status = ds["treatment_status"].values

    groups = ds["treatment_status"].coords["group"]
    times = ds["treatment_status"].coords["time"]
    group_idx = np.where(groups == 2007)[0][0]

    time_2004_idx = np.where(times == 2004)[0][0]
    time_2006_idx = np.where(times == 2006)[0][0]
    time_2007_idx = np.where(times == 2007)[0][0]

    assert status[group_idx, time_2004_idx] == "pre"
    assert status[group_idx, time_2006_idx] == "pre"
    assert status[group_idx, time_2007_idx] == "post"


def test_aggte_dynamic_treatment_status(aggte_result_dynamic):
    ds = aggte_to_dataset(aggte_result_dynamic)

    assert "treatment_status" in ds.data_vars

    status = ds["treatment_status"].values
    event_times = ds["treatment_status"].coords["event"]

    for i, e in enumerate(event_times):
        expected = "pre" if e < 0 else "post"
        assert status[i] == expected, f"Event time {e} should be {expected}, got {status[i]}"
