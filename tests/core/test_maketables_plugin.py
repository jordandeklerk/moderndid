"""Tests for maketables plug-in attributes."""

from types import SimpleNamespace

import numpy as np
import pytest

from moderndid.core.result import extract_n_obs, extract_vcov_info
from moderndid.did.container import AGGTEResult, MPResult
from moderndid.didinter.container import ATEResult, DIDInterResult, EffectsResult, PlacebosResult
from moderndid.didtriple.container import (
    DDDAggResult,
    DDDMultiPeriodRCResult,
    DDDMultiPeriodResult,
    DDDPanelResult,
    DDDRCResult,
)
from moderndid.drdid.container import DRDIDResult, IPWDIDResult, ORDIDResult
from tests.helpers import importorskip

importorskip("formulaic")
from moderndid.didcont.container import DoseResult, PTEAggteResult, PTEResult


def test_drdid_result_exposes_maketables_plugin():
    result = DRDIDResult(
        att=1.2,
        se=0.3,
        uci=1.8,
        lci=0.6,
        boots=None,
        att_inf_func=None,
        call_params={"yname": "outcome", "data_shape": (120, 6)},
        args={"boot": False, "est_method": "imp"},
    )

    coef_table = result.__maketables_coef_table__
    assert list(coef_table.columns[:4]) == ["b", "se", "t", "p"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert coef_table.index.tolist() == ["ATT"]
    assert result.__maketables_stat__("N") == 120
    assert result.__maketables_default_stat_keys__ == ["N", "se_type"]


def test_mp_result_exposes_maketables_plugin():
    result = MPResult(
        groups=np.array([2000, 2001]),
        times=np.array([2001, 2002]),
        att_gt=np.array([0.2, 0.4]),
        vcov_analytical=np.eye(2),
        se_gt=np.array([0.1, 0.15]),
        critical_value=1.96,
        influence_func=np.zeros((10, 2)),
        n_units=10,
        wald_pvalue=0.12,
        estimation_params={"bootstrap": False, "control_group": "nevertreated", "yname": "lemp"},
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["ATT(g=2000, t=2001)", "ATT(g=2001, t=2002)"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == 10
    assert result.__maketables_stat__("se_type") == "Analytical"
    assert result.__maketables_depvar__ == "lemp"
    assert "wald_pvalue" in result.__maketables_default_stat_keys__


def test_aggte_result_exposes_maketables_plugin():
    result = AGGTEResult(
        overall_att=0.3,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-1, 0, 1]),
        att_by_event=np.array([0.0, 0.2, 0.4]),
        se_by_event=np.array([0.1, 0.12, 0.15]),
        influence_func=np.zeros((20, 3)),
        estimation_params={"bootstrap": True, "control_group": "notyettreated", "alpha": 0.05, "yname": "lemp"},
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["Overall ATT", "Event -1", "Event 0", "Event 1"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("aggregation") == "dynamic"
    assert result.__maketables_stat__("se_type") == "Bootstrap"
    assert result.__maketables_depvar__ == "lemp"


def test_didinter_result_exposes_maketables_plugin():
    effects = EffectsResult(
        horizons=np.array([1, 2]),
        estimates=np.array([0.5, 0.7]),
        std_errors=np.array([0.1, 0.2]),
        ci_lower=np.array([0.3, 0.3]),
        ci_upper=np.array([0.7, 1.1]),
        n_switchers=np.array([50, 45]),
        n_observations=np.array([100, 90]),
    )
    placebos = PlacebosResult(
        horizons=np.array([-1]),
        estimates=np.array([0.01]),
        std_errors=np.array([0.05]),
        ci_lower=np.array([-0.08]),
        ci_upper=np.array([0.10]),
        n_switchers=np.array([60]),
        n_observations=np.array([110]),
    )
    ate = ATEResult(estimate=0.6, std_error=0.15, ci_lower=0.3, ci_upper=0.9, n_observations=150, n_switchers=70)
    result = DIDInterResult(
        effects=effects,
        placebos=placebos,
        ate=ate,
        n_units=120,
        n_switchers=70,
        n_never_switchers=50,
        effects_equal_test={"p_value": 0.2},
        placebo_joint_test={"p_value": 0.4},
        estimation_params={"cluster": "id", "yname": "outcome"},
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["ATE", "Effect h=1", "Effect h=2", "Placebo h=-1"]
    assert {"ci95l", "ci95u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == 120
    assert result.__maketables_stat__("se_type") == "Clustered"
    assert result.__maketables_depvar__ == "outcome"
    assert "placebo_joint_pvalue" in result.__maketables_default_stat_keys__


def test_ddd_mp_result_exposes_maketables_plugin():
    result = DDDMultiPeriodResult(
        att=np.array([0.1, 0.2]),
        se=np.array([0.05, 0.08]),
        uci=np.array([0.2, 0.35]),
        lci=np.array([0.0, 0.05]),
        groups=np.array([2, 3]),
        times=np.array([2, 3]),
        glist=np.array([2, 3]),
        tlist=np.array([1, 2, 3]),
        inf_func_mat=np.zeros((12, 2)),
        n=12,
        args={"yname": "y", "control_group": "nevertreated", "base_period": "universal", "est_method": "dr"},
        unit_groups=np.array([0, 0, 2, 2, 3, 3]),
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["ATT(g=2, t=2)", "ATT(g=3, t=3)"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == 12
    assert result.__maketables_stat__("n_cohorts") == 2


def test_ddd_agg_result_exposes_maketables_plugin():
    result = DDDAggResult(
        overall_att=0.4,
        overall_se=0.15,
        aggregation_type="eventstudy",
        egt=np.array([-1, 0, 1]),
        att_egt=np.array([0.0, 0.3, 0.5]),
        se_egt=np.array([0.1, 0.1, 0.2]),
        inf_func=np.zeros((30, 3)),
        args={"boot": True, "control_group": "nevertreated", "est_method": "dr", "alpha": 0.05},
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["Overall ATT", "Event -1", "Event 0", "Event 1"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("aggregation") == "eventstudy"
    assert result.__maketables_stat__("se_type") == "Bootstrap"


def test_dose_result_exposes_maketables_plugin():
    pte_params = SimpleNamespace(
        yname="y",
        alp=0.05,
        control_group="notyettreated",
        dose_est_method="parametric",
        data=[1, 2, 3, 4],
        idname=None,
    )
    result = DoseResult(
        dose=np.array([0.2, 0.8]),
        overall_att=0.3,
        overall_att_se=0.1,
        overall_acrt=0.4,
        overall_acrt_se=0.15,
        att_d=np.array([0.25, 0.35]),
        att_d_se=np.array([0.1, 0.11]),
        acrt_d=np.array([0.3, 0.5]),
        acrt_d_se=np.array([0.12, 0.18]),
        pte_params=pte_params,
    )

    coef_table = result.__maketables_coef_table__
    assert "Overall ATT" in coef_table.index
    assert "ATT(d=0.2)" in coef_table.index
    assert "ACRT(d=0.8)" in coef_table.index
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == 4
    assert result.__maketables_stat__("dose_est_method") == "parametric"


@pytest.mark.parametrize(
    "row_name, att, se, crit",
    [
        ("Overall ATT", 0.3, 0.1, 1.96),
        ("Event -1", 0.0, 0.1, 2.5),
        ("Event 0", 0.2, 0.12, 2.5),
        ("Event 1", 0.4, 0.15, 2.5),
    ],
)
def test_aggte_ci_matches_critical_values(row_name, att, se, crit):
    result = AGGTEResult(
        overall_att=0.3,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-1, 0, 1]),
        att_by_event=np.array([0.0, 0.2, 0.4]),
        se_by_event=np.array([0.1, 0.12, 0.15]),
        critical_values=np.array([2.5, 2.5, 2.5]),
        estimation_params={"alpha": 0.05},
    )
    row = result.__maketables_coef_table__.loc[row_name]
    np.testing.assert_allclose(row["b"], att)
    np.testing.assert_allclose(row["se"], se)
    np.testing.assert_allclose(row["ci95l"], att - crit * se, atol=1e-4)
    np.testing.assert_allclose(row["ci95u"], att + crit * se, atol=1e-4)


@pytest.mark.parametrize("idx", [0, 1])
def test_mp_ci_matches_critical_value(idx):
    result = MPResult(
        groups=np.array([2000, 2001]),
        times=np.array([2001, 2002]),
        att_gt=np.array([0.2, 0.4]),
        vcov_analytical=np.eye(2),
        se_gt=np.array([0.1, 0.15]),
        critical_value=2.66,
        influence_func=np.zeros((10, 2)),
    )
    row = result.__maketables_coef_table__.iloc[idx]
    att, se = result.att_gt[idx], result.se_gt[idx]
    np.testing.assert_allclose(row["ci95l"], att - 2.66 * se)
    np.testing.assert_allclose(row["ci95u"], att + 2.66 * se)


@pytest.mark.parametrize(
    "row_name, att, se, crit",
    [
        ("Overall ATT", 0.4, 0.15, 1.96),
        ("Event -1", 0.0, 0.1, 2.8),
        ("Event 0", 0.3, 0.1, 2.8),
        ("Event 1", 0.5, 0.2, 2.8),
    ],
)
def test_ddd_agg_ci_matches_crit_val(row_name, att, se, crit):
    result = DDDAggResult(
        overall_att=0.4,
        overall_se=0.15,
        aggregation_type="eventstudy",
        egt=np.array([-1, 0, 1]),
        att_egt=np.array([0.0, 0.3, 0.5]),
        se_egt=np.array([0.1, 0.1, 0.2]),
        crit_val=2.8,
        args={"alpha": 0.05},
    )
    row = result.__maketables_coef_table__.loc[row_name]
    np.testing.assert_allclose(row["ci95l"], att - crit * se, atol=1e-4)
    np.testing.assert_allclose(row["ci95u"], att + crit * se, atol=1e-4)


def test_ddd_mp_ci_matches_precomputed():
    lci = np.array([-0.05, -0.02])
    uci = np.array([0.25, 0.42])
    result = DDDMultiPeriodResult(
        att=np.array([0.1, 0.2]),
        se=np.array([0.05, 0.08]),
        uci=uci,
        lci=lci,
        groups=np.array([2, 3]),
        times=np.array([2, 3]),
        glist=np.array([2, 3]),
        tlist=np.array([1, 2, 3]),
        inf_func_mat=np.zeros((12, 2)),
        n=12,
        args={},
        unit_groups=np.array([0, 0, 2, 2, 3, 3]),
    )
    coef_table = result.__maketables_coef_table__
    np.testing.assert_allclose(coef_table["ci95l"].values, lci)
    np.testing.assert_allclose(coef_table["ci95u"].values, uci)


def test_drdid_ci_matches_precomputed():
    result = DRDIDResult(
        att=1.2,
        se=0.3,
        uci=1.85,
        lci=0.55,
        boots=None,
        att_inf_func=None,
        call_params={},
        args={},
    )
    coef_table = result.__maketables_coef_table__
    np.testing.assert_allclose(coef_table.loc["ATT", "ci95l"], 0.55)
    np.testing.assert_allclose(coef_table.loc["ATT", "ci95u"], 1.85)


def test_pte_result_delegates_maketables_plugin():
    dummy_pte_params = SimpleNamespace(yname="y", alp=0.05, data=[1, 2], idname=None)
    event_study = PTEAggteResult(
        overall_att=0.2,
        overall_se=0.05,
        aggregation_type="dynamic",
        event_times=np.array([0, 1]),
        att_by_event=np.array([0.1, 0.25]),
        se_by_event=np.array([0.05, 0.08]),
        att_gt_result=SimpleNamespace(pte_params=dummy_pte_params),
    )
    result = PTEResult(att_gt=None, overall_att=None, event_study=event_study, ptep=dummy_pte_params)

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["Overall ATT", "Event 0", "Event 1"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_depvar__ == "y"


@pytest.mark.parametrize(
    "params, bootstrap_key, result_key, expected",
    [
        ({"bootstrap": True}, "bootstrap", "vcov_type", "bootstrap"),
        ({"bootstrap": False}, "bootstrap", "vcov_type", "analytical"),
        ({"bootstrap": False, "cluster": "county"}, "bootstrap", "clustervar", "county"),
        ({"bootstrap": False, "clustervars": "state"}, "bootstrap", "clustervar", "state"),
        ({"boot": True}, "boot", "vcov_type", "bootstrap"),
    ],
)
def test_extract_vcov_info(params, bootstrap_key, result_key, expected):
    info = extract_vcov_info(params, bootstrap_key=bootstrap_key)
    assert expected in str(info[result_key])


@pytest.mark.parametrize(
    "influence, params, expected",
    [
        (None, {"n_obs": 500}, 500),
        (None, {"n_units": 200}, 200),
        (np.zeros((100, 5)), None, 100),
        (None, {}, None),
        (np.zeros((100, 5)), {"n_obs": 50}, 50),
    ],
)
def test_extract_n_obs(influence, params, expected):
    args = (influence,) if influence is not None else ()
    assert extract_n_obs(*args, params=params) == expected


@pytest.mark.parametrize(
    "cls, call_params, args, expected_n, expected_se_type, expected_depvar, expected_vcov",
    [
        (
            IPWDIDResult,
            {"yname": "wage", "data_shape": (200, 4)},
            {"boot": True},
            200,
            "Bootstrap",
            "wage",
            "bootstrap",
        ),
        (
            ORDIDResult,
            {"yname": "income", "data_shape": (300, 5)},
            {"boot": False},
            300,
            "Analytical",
            "income",
            "analytical",
        ),
    ],
)
def test_drdid_variants_expose_maketables_plugin(
    cls, call_params, args, expected_n, expected_se_type, expected_depvar, expected_vcov
):
    result = cls(att=1.0, se=0.3, uci=1.6, lci=0.4, boots=None, att_inf_func=None, call_params=call_params, args=args)

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["ATT"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == expected_n
    assert result.__maketables_stat__("se_type") == expected_se_type
    assert result.__maketables_depvar__ == expected_depvar
    assert result.__maketables_fixef_string__ is None
    assert result.__maketables_vcov_info__["vcov_type"] == expected_vcov
    assert result.__maketables_default_stat_keys__ == ["N", "se_type"]
    np.testing.assert_allclose(coef_table.loc["ATT", "ci95l"], 0.4)
    np.testing.assert_allclose(coef_table.loc["ATT", "ci95u"], 1.6)


@pytest.mark.parametrize(
    "cls, subgroup_counts, args, expected_n, expected_se_type, expected_vcov",
    [
        (
            DDDPanelResult,
            {"treated_target": 50, "treated_other": 40, "control_target": 60, "control_other": 55},
            {"yname": "y", "boot": False, "est_method": "dr"},
            205,
            "Analytical",
            "analytical",
        ),
        (
            DDDRCResult,
            {"treated_target": 100, "control_target": 120},
            {"yname": "y", "boot": True, "est_method": "ipw"},
            220,
            "Bootstrap",
            "bootstrap",
        ),
    ],
)
def test_ddd_two_period_results_expose_maketables_plugin(
    cls, subgroup_counts, args, expected_n, expected_se_type, expected_vcov
):
    result = cls(
        att=0.5,
        se=0.1,
        uci=0.7,
        lci=0.3,
        boots=None,
        att_inf_func=None,
        did_atts={"a": 0.2, "b": 0.8},
        subgroup_counts=subgroup_counts,
        args=args,
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["ATT"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == expected_n
    assert result.__maketables_stat__("se_type") == expected_se_type
    assert result.__maketables_stat__("est_method") == args["est_method"]
    assert result.__maketables_stat__("unknown_key") is None
    assert result.__maketables_depvar__ == "y"
    assert result.__maketables_fixef_string__ is None
    assert result.__maketables_vcov_info__["vcov_type"] == expected_vcov
    assert result.__maketables_stat_labels__ == {"est_method": "Estimation Method"}
    assert result.__maketables_default_stat_keys__ == ["N", "se_type", "est_method"]


def test_ddd_mp_rc_result_exposes_maketables_plugin():
    result = DDDMultiPeriodRCResult(
        att=np.array([0.1, 0.2]),
        se=np.array([0.05, 0.08]),
        uci=np.array([0.2, 0.35]),
        lci=np.array([0.0, 0.05]),
        groups=np.array([2, 3]),
        times=np.array([2, 3]),
        glist=np.array([2, 3]),
        tlist=np.array([1, 2, 3]),
        inf_func_mat=np.zeros((50, 2)),
        n=50,
        args={"yname": "y", "control_group": "nevertreated", "base_period": "varying", "est_method": "dr"},
        unit_groups=np.zeros(50),
    )

    coef_table = result.__maketables_coef_table__
    assert coef_table.index.tolist() == ["ATT(g=2, t=2)", "ATT(g=3, t=3)"]
    assert {"ci95l", "ci95u", "ci90l", "ci90u"}.issubset(set(coef_table.columns))
    assert result.__maketables_stat__("N") == 50
    assert result.__maketables_stat__("n_cohorts") == 2
    assert result.__maketables_stat__("n_periods") == 3
    assert result.__maketables_stat__("se_type") == "Analytical"
    assert result.__maketables_stat__("control_group") == "Never Treated"
    assert result.__maketables_stat__("base_period") == "varying"
    assert result.__maketables_stat__("est_method") == "dr"
    assert result.__maketables_stat__("unknown_key") is None
    assert result.__maketables_depvar__ == "y"
    assert result.__maketables_fixef_string__ is None
    assert "n_cohorts" in result.__maketables_stat_labels__
    assert "N" in result.__maketables_default_stat_keys__


def test_mp_result_n_units_none():
    result = MPResult(
        groups=np.array([2000]),
        times=np.array([2001]),
        att_gt=np.array([0.2]),
        vcov_analytical=np.eye(1),
        se_gt=np.array([0.1]),
        critical_value=1.96,
        influence_func=np.zeros((10, 1)),
    )

    assert result.__maketables_stat__("N") is None


def test_aggte_result_no_critical_values():
    result = AGGTEResult(
        overall_att=0.3,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-1, 0, 1]),
        att_by_event=np.array([0.0, 0.2, 0.4]),
        se_by_event=np.array([0.1, 0.12, 0.15]),
        estimation_params={"alpha": 0.05},
    )

    coef_table = result.__maketables_coef_table__
    assert len(coef_table) == 4
    assert {"ci95l", "ci95u"}.issubset(set(coef_table.columns))


def test_aggte_result_with_estimation_method():
    result = AGGTEResult(
        overall_att=0.3,
        overall_se=0.1,
        aggregation_type="simple",
        estimation_params={"estimation_method": "dr", "n_obs": 500, "n_units": 100},
    )

    assert result.__maketables_stat__("N") == 500
    assert result.__maketables_stat__("n_units") == 100
    keys = result.__maketables_default_stat_keys__
    assert "N" in keys
    assert "n_units" in keys
    assert "estimation_method" in keys


def test_ddd_agg_default_stat_keys_with_n():
    result = DDDAggResult(
        overall_att=0.4,
        overall_se=0.15,
        aggregation_type="simple",
        inf_func_overall=np.zeros((30,)),
        args={"est_method": "dr"},
    )

    keys = result.__maketables_default_stat_keys__
    assert "N" in keys
    assert "est_method" in keys
