"""Tests for result converters."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from moderndid.did.aggte_obj import AGGTEResult
from moderndid.didhonest.honest_did import HonestDiDResult
from moderndid.didhonest.sensitivity import OriginalCSResult
from moderndid.plotting.converters import (
    aggte_to_dataset,
    doseresult_to_dataset,
    honestdid_to_dataset,
    mpresult_to_dataset,
)


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
