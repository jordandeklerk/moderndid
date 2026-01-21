"""Tests for result converters."""

import numpy as np
import polars as pl
import pytest

from moderndid.did.aggte_obj import AGGTEResult
from moderndid.didcont.estimation.container import PTEResult
from moderndid.didhonest.honest_did import HonestDiDResult
from moderndid.didhonest.sensitivity import OriginalCSResult
from moderndid.plots.converters import (
    aggteresult_to_polars,
    doseresult_to_polars,
    honestdid_to_polars,
    mpresult_to_polars,
    pteresult_to_polars,
)


def test_mpresult_to_polars(mp_result):
    df = mpresult_to_polars(mp_result)

    assert isinstance(df, pl.DataFrame)
    assert "group" in df.columns
    assert "time" in df.columns
    assert "att" in df.columns
    assert "se" in df.columns
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns
    assert "treatment_status" in df.columns

    assert len(df) == 6

    row = df.filter((pl.col("group") == 2000) & (pl.col("time") == 2004))
    assert row["att"].item() == pytest.approx(0.5)


def test_mpresult_treatment_status(mp_result):
    df = mpresult_to_polars(mp_result)

    post_row = df.filter((pl.col("group") == 2000) & (pl.col("time") == 2004))
    assert post_row["treatment_status"].item() == "Post"

    pre_row = df.filter((pl.col("group") == 2007) & (pl.col("time") == 2004))
    assert pre_row["treatment_status"].item() == "Pre"


def test_mpresult_confidence_intervals(mp_result):
    df = mpresult_to_polars(mp_result)

    row = df.filter((pl.col("group") == 2000) & (pl.col("time") == 2004))
    att = row["att"].item()
    se = row["se"].item()
    ci_lower = row["ci_lower"].item()
    ci_upper = row["ci_upper"].item()

    expected_lower = att - 1.96 * se
    expected_upper = att + 1.96 * se

    assert ci_lower == pytest.approx(expected_lower)
    assert ci_upper == pytest.approx(expected_upper)


def test_aggteresult_dynamic_to_polars(aggte_result_dynamic):
    df = aggteresult_to_polars(aggte_result_dynamic)

    assert isinstance(df, pl.DataFrame)
    assert "event_time" in df.columns
    assert "att" in df.columns
    assert "se" in df.columns
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns
    assert "treatment_status" in df.columns

    assert len(df) == 5

    row = df.filter(pl.col("event_time") == 0)
    assert row["att"].item() == pytest.approx(0.8)


def test_aggteresult_dynamic_treatment_status(aggte_result_dynamic):
    df = aggteresult_to_polars(aggte_result_dynamic)

    pre_rows = df.filter(pl.col("event_time") < 0)
    post_rows = df.filter(pl.col("event_time") >= 0)

    assert all(s == "Pre" for s in pre_rows["treatment_status"].to_list())
    assert all(s == "Post" for s in post_rows["treatment_status"].to_list())


def test_aggteresult_simple_raises(aggte_result_simple):
    with pytest.raises(ValueError, match="Simple aggregation"):
        aggteresult_to_polars(aggte_result_simple)


def test_aggteresult_missing_data_raises():
    result = AGGTEResult(
        overall_att=0.8,
        overall_se=0.15,
        aggregation_type="dynamic",
        event_times=None,
        att_by_event=None,
        se_by_event=None,
    )

    with pytest.raises(ValueError, match="must have event_times"):
        aggteresult_to_polars(result)


def test_aggte_group_aggregation():
    result = AGGTEResult(
        overall_att=0.8,
        overall_se=0.15,
        aggregation_type="group",
        event_times=np.array([2000, 2007]),
        att_by_event=np.array([0.7, 0.9]),
        se_by_event=np.array([0.12, 0.18]),
    )

    df = aggteresult_to_polars(result)
    assert len(df) == 2
    assert "event_time" in df.columns


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

    df = aggteresult_to_polars(result)

    row = df.filter(pl.col("event_time") == 0)
    att = row["att"].item()
    se = row["se"].item()
    ci_lower = row["ci_lower"].item()
    ci_upper = row["ci_upper"].item()

    expected_lower = att - 2.5 * se
    expected_upper = att + 2.5 * se

    assert ci_lower == pytest.approx(expected_lower)
    assert ci_upper == pytest.approx(expected_upper)


def test_doseresult_to_polars_att(dose_result):
    df = doseresult_to_polars(dose_result, effect_type="att")

    assert isinstance(df, pl.DataFrame)
    assert "dose" in df.columns
    assert "effect" in df.columns
    assert "se" in df.columns
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns

    assert len(df) == 5

    row = df.filter(pl.col("dose") == 3.0)
    assert row["effect"].item() == pytest.approx(1.5)


def test_doseresult_to_polars_acrt(dose_result):
    df = doseresult_to_polars(dose_result, effect_type="acrt")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5

    row = df.filter(pl.col("dose") == 3.0)
    assert row["effect"].item() == pytest.approx(1.2)


def test_doseresult_invalid_type(dose_result):
    with pytest.raises(ValueError, match="effect_type must be"):
        doseresult_to_polars(dose_result, effect_type="invalid")


def test_doseresult_confidence_intervals(dose_result):
    df = doseresult_to_polars(dose_result, effect_type="att")

    row = df.filter(pl.col("dose") == 2.0)
    effect = row["effect"].item()
    se = row["se"].item()
    ci_lower = row["ci_lower"].item()
    ci_upper = row["ci_upper"].item()

    expected_lower = effect - 1.96 * se
    expected_upper = effect + 1.96 * se

    assert ci_lower == pytest.approx(expected_lower)
    assert ci_upper == pytest.approx(expected_upper)


def test_pteresult_to_polars(pte_result_with_event_study):
    df = pteresult_to_polars(pte_result_with_event_study)

    assert isinstance(df, pl.DataFrame)
    assert "event_time" in df.columns
    assert "att" in df.columns
    assert "se" in df.columns
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns
    assert "treatment_status" in df.columns

    assert len(df) == 5

    pre_rows = df.filter(pl.col("treatment_status") == "Pre")
    post_rows = df.filter(pl.col("treatment_status") == "Post")

    assert len(pre_rows) == 2
    assert len(post_rows) == 3


def test_pteresult_no_event_study_raises():
    result = PTEResult(
        att_gt=None,
        overall_att=None,
        event_study=None,
        ptep=None,
    )

    with pytest.raises(ValueError, match="does not contain event study"):
        pteresult_to_polars(result)


def test_honestdid_to_polars(honest_result):
    df = honestdid_to_polars(honest_result)

    assert isinstance(df, pl.DataFrame)
    assert "param_value" in df.columns
    assert "method" in df.columns
    assert "lb" in df.columns
    assert "ub" in df.columns
    assert "midpoint" in df.columns

    methods = df["method"].unique().to_list()
    assert "Original" in methods
    assert "FLCI" in methods


def test_honestdid_empty_dataframe():
    df = pl.DataFrame()
    original_ci = OriginalCSResult(lb=0.3, ub=0.7)
    result = HonestDiDResult(df, original_ci, "smoothness")

    with pytest.raises(ValueError, match="empty robust_ci"):
        honestdid_to_polars(result)


def test_honestdid_midpoint():
    df = pl.DataFrame({"M": [1.0], "lb": [0.0], "ub": [1.0], "method": ["FLCI"]})
    original_ci = OriginalCSResult(lb=0.2, ub=0.8, method="Original")
    result = HonestDiDResult(df, original_ci, "smoothness")

    df_result = honestdid_to_polars(result)

    original_row = df_result.filter(pl.col("method") == "Original")
    flci_row = df_result.filter(pl.col("method") == "FLCI")

    assert original_row["midpoint"].item() == pytest.approx(0.5)
    assert flci_row["midpoint"].item() == pytest.approx(0.5)
