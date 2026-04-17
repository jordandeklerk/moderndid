"""Shared fixtures for plotting tests."""

import numpy as np
import polars as pl
import pytest

from moderndid.did.container import AGGTEResult, MPResult
from moderndid.didcont.container import DoseResult
from moderndid.diddynamic.container import DynBalancingHetResult, DynBalancingHistoryResult, DynBalancingResult
from moderndid.didhonest.honest_did import HonestDiDResult
from moderndid.didhonest.sensitivity import OriginalCSResult
from moderndid.didinter.container import DIDInterResult, EffectsResult, PlacebosResult
from moderndid.didtriple.container import DDDAggResult, DDDMultiPeriodResult


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
def aggte_result_group():
    return AGGTEResult(
        overall_att=0.8,
        overall_se=0.14,
        aggregation_type="group",
        event_times=np.array([2000, 2007]),
        att_by_event=np.array([0.7, 0.9]),
        se_by_event=np.array([0.12, 0.16]),
    )


@pytest.fixture
def aggte_result_calendar():
    return AGGTEResult(
        overall_att=0.75,
        overall_se=0.13,
        aggregation_type="calendar",
        event_times=np.array([2004, 2005, 2006]),
        att_by_event=np.array([0.6, 0.8, 0.85]),
        se_by_event=np.array([0.11, 0.13, 0.14]),
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
    from moderndid.didcont.container import PTEAggteResult, PTEResult

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


@pytest.fixture
def didinter_result():
    effects = EffectsResult(
        horizons=np.array([1, 2, 3, 4]),
        estimates=np.array([0.5, 0.8, 1.0, 1.2]),
        std_errors=np.array([0.1, 0.12, 0.15, 0.18]),
        ci_lower=np.array([0.3, 0.56, 0.71, 0.85]),
        ci_upper=np.array([0.7, 1.04, 1.29, 1.55]),
        n_switchers=np.array([50, 45, 40, 35]),
        n_observations=np.array([500, 450, 400, 350]),
    )
    return DIDInterResult(
        effects=effects,
        n_units=100,
        n_switchers=50,
        n_never_switchers=50,
        ci_level=95.0,
    )


@pytest.fixture
def didinter_result_with_placebos():
    effects = EffectsResult(
        horizons=np.array([1, 2, 3]),
        estimates=np.array([0.5, 0.8, 1.0]),
        std_errors=np.array([0.1, 0.12, 0.15]),
        ci_lower=np.array([0.3, 0.56, 0.71]),
        ci_upper=np.array([0.7, 1.04, 1.29]),
        n_switchers=np.array([50, 45, 40]),
        n_observations=np.array([500, 450, 400]),
    )
    placebos = PlacebosResult(
        horizons=np.array([-2, -1]),
        estimates=np.array([0.05, 0.02]),
        std_errors=np.array([0.08, 0.07]),
        ci_lower=np.array([-0.11, -0.12]),
        ci_upper=np.array([0.21, 0.16]),
        n_switchers=np.array([50, 50]),
        n_observations=np.array([500, 500]),
    )
    return DIDInterResult(
        effects=effects,
        placebos=placebos,
        n_units=100,
        n_switchers=50,
        n_never_switchers=50,
        ci_level=95.0,
    )


@pytest.fixture
def dyn_balancing_result():
    return DynBalancingResult(
        att=0.30,
        var_att=0.04,
        mu1=8.00,
        mu2=7.70,
        var_mu1=0.02,
        var_mu2=0.022,
        robust_quantile=2.45,
        gaussian_quantile=1.96,
        gammas={"ds1": np.array([]), "ds2": np.array([])},
        coefficients={"ds1": [], "ds2": []},
        imbalances={},
        estimation_params={"alpha": 0.05, "n_periods": 2},
    )


@pytest.fixture
def dyn_balancing_history_result():
    summary = pl.DataFrame(
        {
            "period_length": [1, 2, 3, 4, 5],
            "att": [0.27, 0.30, 0.24, 0.28, 0.25],
            "var_att": [0.040, 0.041, 0.047, 0.049, 0.053],
            "mu1": [7.97, 8.00, 8.01, 8.01, 8.01],
            "var_mu1": [0.018, 0.019, 0.022, 0.024, 0.027],
            "mu2": [7.70, 7.70, 7.77, 7.73, 7.76],
            "var_mu2": [0.022, 0.022, 0.025, 0.025, 0.026],
            "robust_quantile": [2.45, 2.45, 2.45, 2.45, 2.45],
            "gaussian_quantile": [1.96, 1.96, 1.96, 1.96, 1.96],
        }
    )
    return DynBalancingHistoryResult(summary=summary, results=[])


@pytest.fixture
def dyn_balancing_het_result():
    summary = pl.DataFrame(
        {
            "final_period": [3, 4, 5],
            "att": [0.25, 0.30, 0.28],
            "var_att": [0.038, 0.041, 0.045],
            "mu1": [7.98, 8.00, 8.01],
            "var_mu1": [0.019, 0.020, 0.022],
            "mu2": [7.73, 7.70, 7.73],
            "var_mu2": [0.019, 0.021, 0.023],
            "robust_quantile": [2.45, 2.45, 2.45],
            "gaussian_quantile": [1.96, 1.96, 1.96],
        }
    )
    return DynBalancingHetResult(summary=summary, results=[])


@pytest.fixture
def dyn_balancing_result_with_coefs():
    return DynBalancingResult(
        att=0.30,
        var_att=0.04,
        mu1=8.00,
        mu2=7.70,
        var_mu1=0.02,
        var_mu2=0.022,
        robust_quantile=2.45,
        gaussian_quantile=1.96,
        gammas={"ds1": np.array([]), "ds2": np.array([])},
        coefficients={
            "ds1": [
                np.array([0.5, 0.3, -0.1, 0.0, 0.0, 0.2]),
                np.array([0.4, 0.25, 0.0, -0.15, 0.0, 0.18]),
            ],
            "ds2": [
                np.array([0.45, 0.28, 0.0, 0.0, 0.0, 0.19]),
                np.array([0.38, 0.22, -0.12, 0.0, 0.0, 0.16]),
            ],
        },
        imbalances={},
        estimation_params={"alpha": 0.05, "n_periods": 2},
    )
