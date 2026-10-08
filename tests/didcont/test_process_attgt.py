"""Tests for processing ATT(g,t) results."""

import numpy as np
import pytest
import scipy.stats

from moderndid.did.mboot import mboot
from moderndid.didcont.estimation import (
    GroupTimeATTResult,
    process_att_gt,
)


def test_mboot_basic(simple_influence_func):
    n = simple_influence_func.shape[0]
    result = mboot(simple_influence_func, n_units=n, biters=20, alp=0.05)

    assert "se" in result
    assert "crit_val" in result
    assert len(result["se"]) == simple_influence_func.shape[1]


def test_mboot_single_param():
    np.random.seed(42)
    influence_func = np.random.randn(100, 1)

    result = mboot(influence_func, n_units=100, biters=500, alp=0.05, random_state=np.random.default_rng(42))

    assert len(result["se"]) == 1


def test_process_att_gt_basic(att_gt_raw_results, pte_params_basic):
    result = process_att_gt(att_gt_raw_results, pte_params_basic)

    assert isinstance(result, GroupTimeATTResult)
    assert len(result.groups) == len(att_gt_raw_results["attgt_list"])
    assert len(result.times) == len(att_gt_raw_results["attgt_list"])
    assert len(result.att) == len(att_gt_raw_results["attgt_list"])
    assert result.n_units == att_gt_raw_results["influence_func"].shape[0]
    assert result.vcov_analytical.shape == (12, 12)
    assert result.cband == pte_params_basic.cband
    assert result.alpha == pte_params_basic.alp


def test_process_att_gt_pre_treatment_test(att_gt_raw_results, pte_params_basic):
    result = process_att_gt(att_gt_raw_results, pte_params_basic)

    pre_treatment_mask = result.groups > result.times
    n_pre = np.sum(pre_treatment_mask)

    if n_pre > 0:
        assert result.wald_stat is not None or result.wald_pvalue is not None


def test_process_att_gt_no_pre_treatment(pte_params_basic):
    attgt_list = []
    for g in [2004]:
        for t in [2004, 2005, 2006]:
            attgt_list.append({"att": 0.1, "group": g, "time_period": t})

    att_gt_results = {"attgt_list": attgt_list, "influence_func": np.random.randn(100, 3), "extra_gt_returns": []}

    with pytest.warns(UserWarning, match="No pre-treatment periods"):
        result = process_att_gt(att_gt_results, pte_params_basic)

    assert result.wald_stat is None
    assert result.wald_pvalue is None


@pytest.mark.filterwarnings("ignore:Simultaneous confidence band:UserWarning")
def test_process_att_gt_singular_vcov(pte_params_basic):
    attgt_list = []
    for i in range(3):
        attgt_list.append({"att": 0.0, "group": 2005, "time_period": 2003 + i})

    influence_func = np.zeros((100, 3))
    influence_func[:, 0] = np.random.randn(100)
    influence_func[:, 1] = influence_func[:, 0]
    influence_func[:, 2] = influence_func[:, 0] * 2

    att_gt_results = {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": []}

    with pytest.warns(UserWarning, match="singular covariance matrix"):
        result = process_att_gt(att_gt_results, pte_params_basic)

    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_mboot_returns_crit_val():
    np.random.seed(42)
    influence_func = np.random.randn(100, 3)

    result = mboot(influence_func, n_units=100, biters=50, alp=0.05, random_state=42)

    assert "crit_val" in result
    assert "se" in result
    assert len(result["se"]) == 3


def test_process_att_gt_with_extra_returns(pte_params_basic):
    att_gt_results = {
        "attgt_list": [
            {"att": 0.1, "group": 2004, "time_period": 2003},
            {"att": 0.2, "group": 2004, "time_period": 2005},
        ],
        "influence_func": np.random.randn(100, 2),
        "extra_gt_returns": [
            {"group": 2004, "time_period": 2003, "extra_data": "test1"},
            {"group": 2004, "time_period": 2005, "extra_data": "test2"},
        ],
    }

    result = process_att_gt(att_gt_results, pte_params_basic)

    assert result.extra_gt_returns is not None
    assert len(result.extra_gt_returns) == 2
    assert result.extra_gt_returns[0]["extra_data"] == "test1"


def test_process_att_gt_with_real_mp_result(att_gt_result):
    att_gt_raw = {
        "attgt_list": [
            {"att": att, "group": g, "time_period": t}
            for att, g, t in zip(att_gt_result.att_gt, att_gt_result.groups, att_gt_result.times)
        ],
        "influence_func": att_gt_result.influence_func,
        "extra_gt_returns": [],
    }

    from moderndid.didcont.estimation import PTEParams

    pte_params = PTEParams(
        yname="lemp",
        gname="first.treat",
        tname="year",
        idname="countyreal",
        data={"year": att_gt_result.times},
        g_list=np.unique(att_gt_result.groups),
        t_list=np.unique(att_gt_result.times),
        cband=att_gt_result.estimation_params.get("uniform_bands", False),
        alp=att_gt_result.alpha,
        boot_type="multiplier",
        anticipation=att_gt_result.estimation_params.get("anticipation_periods", 0),
        base_period=att_gt_result.estimation_params.get("base_period", "varying"),
        weightsname=None,
        control_group=att_gt_result.estimation_params.get("control_group", "nevertreated"),
        gt_type="att",
        ret_quantile=0.5,
        biters=20,
        dname=None,
        degree=None,
        num_knots=None,
        knots=None,
        dvals=None,
        target_parameter=None,
        aggregation=None,
        treatment_type=None,
        xformula="~1",
    )

    result = process_att_gt(att_gt_raw, pte_params)

    assert isinstance(result, GroupTimeATTResult)
    assert len(result.groups) == len(att_gt_result.groups)
    assert len(result.times) == len(att_gt_result.times)
    assert len(result.att) == len(att_gt_result.att_gt)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_process_att_gt_critical_value_is_infinite_when_draws_move_a_zero_scale_cell(
    fix_bootstrap_draws, zero_scale_draws, two_cell_results, pte_params_basic
):
    fix_bootstrap_draws(zero_scale_draws)

    result = process_att_gt(two_cell_results, pte_params_basic)

    assert result.critical_value == np.inf
    np.testing.assert_allclose(result.se, [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_process_att_gt_standard_error_is_zero_for_a_cell_that_every_draw_leaves_at_zero(
    fix_bootstrap_draws, central_zero_scale_draws, two_cell_results, pte_params_basic
):
    fix_bootstrap_draws(np.column_stack([central_zero_scale_draws[:, 0], np.zeros(21)]))

    result = process_att_gt(two_cell_results, pte_params_basic)

    np.testing.assert_allclose(result.critical_value, 50 / (10 / 1.3489795), rtol=1e-12)
    np.testing.assert_allclose(result.se, [10 / 1.3489795, 0.0], rtol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_process_att_gt_critical_value_is_never_below_the_pointwise_value(
    fix_bootstrap_draws, two_cell_results, pte_params_basic
):
    fix_bootstrap_draws(np.linspace(-1.0, 1.0, 99)[:, None])

    with pytest.warns(UserWarning, match="smaller than pointwise"):
        result = process_att_gt(two_cell_results, pte_params_basic)

    assert result.critical_value == pytest.approx(scipy.stats.norm.ppf(0.975))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_process_att_gt_critical_value_is_pointwise_when_no_draw_has_a_deviation(
    fix_bootstrap_draws, two_cell_results, pte_params_basic
):
    fix_bootstrap_draws(np.zeros((21, 1)))

    with pytest.warns(UserWarning, match="smaller than pointwise"):
        result = process_att_gt(two_cell_results, pte_params_basic)

    assert result.critical_value == pytest.approx(scipy.stats.norm.ppf(0.975))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_process_att_gt_keeps_a_critical_value_above_the_pointwise_value(
    fix_bootstrap_draws, central_zero_scale_draws, two_cell_results, pte_params_basic, recwarn
):
    fix_bootstrap_draws(central_zero_scale_draws[:, [0]])

    result = process_att_gt(two_cell_results, pte_params_basic)

    assert result.critical_value == pytest.approx(50 / (10 / 1.3489795))
    assert not [w for w in recwarn if "smaller than pointwise" in str(w.message)]


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_process_att_gt_pointwise_value_needs_no_floor_without_a_band(
    fix_bootstrap_draws, two_cell_results, pte_params_basic, recwarn
):
    fix_bootstrap_draws(np.linspace(-1.0, 1.0, 99)[:, None])

    result = process_att_gt(two_cell_results, pte_params_basic._replace(cband=False))

    assert result.critical_value == pytest.approx(scipy.stats.norm.ppf(0.975))
    assert not [w for w in recwarn if "smaller than pointwise" in str(w.message)]


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("base_period, expected", [("universal", np.nan), ("varying", 0.0)])
def test_process_att_gt_leaves_the_standard_error_of_a_reference_cell_undefined(
    fix_bootstrap_draws, central_zero_scale_draws, two_cell_results, pte_params_basic, base_period, expected
):
    fix_bootstrap_draws(np.column_stack([np.zeros(21), central_zero_scale_draws[:, 0]]))

    result = process_att_gt(two_cell_results, pte_params_basic._replace(base_period=base_period))

    np.testing.assert_allclose(result.se, [expected, 10 / 1.3489795], rtol=1e-12)
