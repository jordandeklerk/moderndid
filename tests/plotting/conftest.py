"""Shared fixtures for plotting tests."""

import numpy as np
import polars as pl
import pytest

from moderndid.did.aggte_obj import AGGTEResult
from moderndid.did.multiperiod_obj import MPResult
from moderndid.didcont.estimation.container import DoseResult
from moderndid.didhonest.honest_did import HonestDiDResult
from moderndid.didhonest.sensitivity import OriginalCSResult
from moderndid.didtriple.agg_ddd_obj import DDDAggResult
from moderndid.didtriple.estimators.ddd_mp import DDDMultiPeriodResult


@pytest.fixture
def mp_result():
    groups = np.array([2000, 2000, 2000, 2007, 2007, 2007])
    times = np.array([2004, 2006, 2007, 2004, 2006, 2007])
    att_gt = np.array([0.5, 0.8, 1.2, 0.3, 0.6, 0.9])
    se_gt = np.array([0.1, 0.12, 0.15, 0.08, 0.11, 0.13])
    vcov = np.eye(6)
    influence_func = np.random.randn(100, 6)

    return MPResult(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=influence_func,
    )


@pytest.fixture
def aggte_result_dynamic():
    event_times = np.array([-2, -1, 0, 1, 2])
    att = np.array([0.1, 0.05, 0.8, 1.2, 1.5])
    se = np.array([0.1, 0.09, 0.12, 0.15, 0.18])

    return AGGTEResult(
        overall_att=0.7,
        overall_se=0.15,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att,
        se_by_event=se,
    )


@pytest.fixture
def aggte_result_simple():
    return AGGTEResult(
        overall_att=0.75,
        overall_se=0.12,
        aggregation_type="simple",
    )


@pytest.fixture
def dose_result():
    dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    att_d = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    att_d_se = np.array([0.1, 0.12, 0.15, 0.18, 0.20])
    acrt_d = np.array([0.4, 0.8, 1.2, 1.6, 2.0])
    acrt_d_se = np.array([0.08, 0.10, 0.12, 0.14, 0.16])

    return DoseResult(
        dose=dose,
        overall_att=1.5,
        overall_att_se=0.15,
        overall_att_inf_func=np.random.randn(100),
        overall_acrt=1.2,
        overall_acrt_se=0.12,
        overall_acrt_inf_func=np.random.randn(100),
        att_d=att_d,
        att_d_se=att_d_se,
        acrt_d=acrt_d,
        acrt_d_se=acrt_d_se,
    )


@pytest.fixture
def honest_result():
    df = pl.DataFrame({"M": [0.5, 1.0, 1.5], "lb": [0.1, 0.0, -0.1], "ub": [0.9, 0.8, 0.7], "method": ["FLCI"] * 3})

    original_ci = OriginalCSResult(
        lb=0.3,
        ub=0.7,
        method="Original",
    )

    return HonestDiDResult(
        robust_ci=df,
        original_ci=original_ci,
        sensitivity_type="smoothness",
    )


@pytest.fixture
def pte_result_with_event_study():
    from moderndid.didcont.estimation.container import PTEAggteResult, PTEResult

    event_study = PTEAggteResult(
        overall_att=0.5,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-2, -1, 0, 1, 2]),
        att_by_event=np.array([0.1, 0.05, 0.5, 0.8, 1.0]),
        se_by_event=np.array([0.08, 0.07, 0.1, 0.12, 0.15]),
        critical_value=1.96,
    )

    return PTEResult(
        att_gt=None,
        overall_att=None,
        event_study=event_study,
        ptep=None,
    )


@pytest.fixture
def sensitivity_robust_results():
    return pl.DataFrame(
        {
            "M": [0.5, 1.0, 1.5, 0.5, 1.0, 1.5],
            "lb": [0.2, 0.1, 0.0, 0.15, 0.05, -0.05],
            "ub": [0.8, 0.9, 1.0, 0.75, 0.85, 0.95],
            "method": ["FLCI", "FLCI", "FLCI", "Conditional", "Conditional", "Conditional"],
        }
    )


@pytest.fixture
def sensitivity_original_result():
    return OriginalCSResult(lb=0.3, ub=0.7, method="Original")


@pytest.fixture
def ddd_mp_result():
    rng = np.random.default_rng(42)
    groups = np.array([2, 2, 2, 3, 3, 3])
    times = np.array([1, 2, 3, 1, 2, 3])
    att = np.array([0.0, 0.5, 0.8, 0.0, 0.0, 0.6])
    se = np.array([0.1, 0.12, 0.15, 0.1, 0.11, 0.13])
    uci = att + 1.96 * se
    lci = att - 1.96 * se

    return DDDMultiPeriodResult(
        att=att,
        se=se,
        uci=uci,
        lci=lci,
        groups=groups,
        times=times,
        glist=np.array([2, 3]),
        tlist=np.array([1, 2, 3]),
        inf_func_mat=rng.standard_normal((100, 6)),
        n=100,
        args={"control_group": "nevertreated", "est_method": "dr"},
        unit_groups=rng.choice([0, 2, 3], size=100),
    )


@pytest.fixture
def ddd_agg_eventstudy():
    return DDDAggResult(
        overall_att=0.7,
        overall_se=0.15,
        aggregation_type="eventstudy",
        egt=np.array([-2, -1, 0, 1, 2]),
        att_egt=np.array([0.1, 0.05, 0.8, 1.2, 1.5]),
        se_egt=np.array([0.1, 0.09, 0.12, 0.15, 0.18]),
        crit_val=1.96,
        args={"alpha": 0.05},
    )


@pytest.fixture
def ddd_agg_group():
    return DDDAggResult(
        overall_att=0.75,
        overall_se=0.12,
        aggregation_type="group",
        egt=np.array([2, 3, 4]),
        att_egt=np.array([0.5, 0.8, 1.0]),
        se_egt=np.array([0.1, 0.12, 0.15]),
        crit_val=1.96,
        args={"alpha": 0.05},
    )


@pytest.fixture
def ddd_agg_calendar():
    return DDDAggResult(
        overall_att=0.65,
        overall_se=0.11,
        aggregation_type="calendar",
        egt=np.array([2, 3, 4]),
        att_egt=np.array([0.4, 0.7, 0.9]),
        se_egt=np.array([0.09, 0.11, 0.13]),
        crit_val=1.96,
        args={"alpha": 0.05},
    )


@pytest.fixture
def ddd_agg_simple():
    return DDDAggResult(
        overall_att=0.8,
        overall_se=0.14,
        aggregation_type="simple",
        args={"alpha": 0.05},
    )
