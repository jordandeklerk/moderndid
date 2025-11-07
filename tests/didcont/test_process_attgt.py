"""Tests for processing ATT(g,t) results."""

import numpy as np
import pytest
import scipy.stats

from moderndid.didcont.estimation import (
    GroupTimeATTResult,
    multiplier_bootstrap,
    process_att_gt,
)


def test_multiplier_bootstrap_basic(simple_influence_func):
    result = multiplier_bootstrap(simple_influence_func, biters=20, alpha=0.05)

    assert "se" in result
    assert "critical_value" in result
    assert len(result["se"]) == simple_influence_func.shape[1]
    assert np.all(result["se"] > 0)
    assert result["critical_value"] >= scipy.stats.norm.ppf(0.975)


def test_multiplier_bootstrap_single_param():
    np.random.seed(42)
    influence_func = np.random.randn(100, 1)

    result = multiplier_bootstrap(influence_func, biters=500, alpha=0.05, rng=np.random.default_rng(42))

    assert len(result["se"]) == 1
    assert result["se"][0] > 0
    assert 1.95 <= result["critical_value"] < 4.0


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


def test_multiplier_bootstrap_critical_value_checks():
    np.random.seed(42)
    influence_func = np.ones((100, 1)) * 1e-10

    with pytest.warns(UserWarning, match="Simultaneous confidence band is smaller than pointwise"):
        result = multiplier_bootstrap(influence_func, biters=20, alpha=0.05, rng=np.random.default_rng(42))

    assert np.isclose(result["critical_value"], scipy.stats.norm.ppf(0.975), rtol=1e-3)


def test_multiplier_bootstrap_large_critical_value():
    np.random.seed(42)
    influence_func = np.random.randn(20, 3) * 100
    influence_func[0, :] = 1000

    with pytest.warns(UserWarning, match="Simultaneous confidence band is smaller than pointwise"):
        result = multiplier_bootstrap(influence_func, biters=20, alpha=0.05, rng=np.random.default_rng(42))

    assert result["critical_value"] >= scipy.stats.norm.ppf(0.975)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_multiplier_bootstrap_alpha_levels(simple_influence_func, alpha):
    seed = int(alpha * 10_000) + 7
    result = multiplier_bootstrap(simple_influence_func, biters=500, alpha=alpha, rng=np.random.default_rng(seed))

    pointwise_crit = scipy.stats.norm.ppf(1 - alpha / 2)
    assert result["critical_value"] >= pointwise_crit


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
    """Test processing with real att_gt results from moderndid.did."""
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
