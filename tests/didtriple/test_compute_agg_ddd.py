import numpy as np
import pytest
from scipy import stats

from moderndid.didtriple.agg_ddd_obj import DDDAggResult
from moderndid.didtriple.compute_agg_ddd import compute_agg_ddd


@pytest.mark.parametrize("agg_type", ["simple", "eventstudy", "group", "calendar"])
def test_compute_agg_ddd_types(mp_ddd_result, agg_type):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type=agg_type, boot=False, cband=False)

    assert isinstance(result, DDDAggResult)
    assert result.aggregation_type == agg_type
    assert np.isfinite(result.overall_att)
    assert np.isfinite(result.overall_se)
    assert result.overall_se > 0


def test_compute_agg_ddd_simple(mp_ddd_result):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type="simple", boot=False, cband=False)

    assert result.egt is None
    assert result.att_egt is None
    assert result.se_egt is None


def test_compute_agg_ddd_eventstudy(mp_ddd_result):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type="eventstudy", boot=False, cband=False)

    assert len(result.egt) == len(result.att_egt) == len(result.se_egt)
    assert np.all(np.diff(result.egt) > 0)
    valid_se = result.se_egt[~np.isnan(result.se_egt)]
    assert np.all(valid_se > 0)


def test_compute_agg_ddd_group(mp_ddd_result):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type="group", boot=False, cband=False)

    unique_groups = np.unique(mp_ddd_result.groups)
    finite_groups = unique_groups[np.isfinite(unique_groups)]
    assert len(result.egt) == len(finite_groups)
    assert np.all(result.se_egt > 0)


def test_compute_agg_ddd_calendar(mp_ddd_result):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type="calendar", boot=False, cband=False)

    assert len(result.egt) > 0
    assert len(np.unique(result.egt)) == len(result.egt)
    assert np.all(result.se_egt > 0)


def test_compute_agg_ddd_invalid_type(mp_ddd_result):
    with pytest.raises(ValueError, match="aggregation_type must be"):
        compute_agg_ddd(mp_ddd_result, aggregation_type="invalid")


def test_compute_agg_ddd_cband_requires_boot(mp_ddd_result):
    with pytest.raises(ValueError, match="cband=True requires boot=True"):
        compute_agg_ddd(mp_ddd_result, boot=False, cband=True)


def test_compute_agg_ddd_with_bootstrap(mp_ddd_result):
    result = compute_agg_ddd(
        mp_ddd_result,
        aggregation_type="simple",
        boot=True,
        biters=50,
        cband=False,
        random_state=42,
    )

    assert np.isfinite(result.overall_se)
    assert result.overall_se > 0
    ci_width = result.crit_val * result.overall_se
    assert ci_width > 0, "Confidence interval width should be positive"


def test_compute_agg_ddd_eventstudy_min_max_e(mp_ddd_result):
    result = compute_agg_ddd(
        mp_ddd_result,
        aggregation_type="eventstudy",
        min_e=-1,
        max_e=2,
        boot=False,
        cband=False,
    )

    if result.egt is not None:
        assert np.all(result.egt >= -1), "All event times should be >= min_e"
        assert np.all(result.egt <= 2), "All event times should be <= max_e"


def test_compute_agg_ddd_alpha(mp_ddd_result):
    result_05 = compute_agg_ddd(mp_ddd_result, aggregation_type="simple", boot=False, cband=False, alpha=0.05)
    result_10 = compute_agg_ddd(mp_ddd_result, aggregation_type="simple", boot=False, cband=False, alpha=0.10)

    expected_crit_05 = stats.norm.ppf(1 - 0.05 / 2)
    expected_crit_10 = stats.norm.ppf(1 - 0.10 / 2)
    assert np.isclose(result_05.crit_val, expected_crit_05, rtol=1e-3)
    assert np.isclose(result_10.crit_val, expected_crit_10, rtol=1e-3)
    assert result_05.crit_val > result_10.crit_val


def test_compute_agg_ddd_inf_func_overall(mp_ddd_result):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type="simple", boot=False, cband=False)

    assert len(result.inf_func_overall) == mp_ddd_result.n
    assert np.abs(np.mean(result.inf_func_overall)) < 0.5
    assert np.var(result.inf_func_overall) > 0


def test_compute_agg_ddd_se_derivation(mp_ddd_result):
    result = compute_agg_ddd(mp_ddd_result, aggregation_type="simple", boot=False, cband=False)

    inf_func_var = np.var(result.inf_func_overall, ddof=0)
    expected_se = np.sqrt(inf_func_var / mp_ddd_result.n)
    assert np.isclose(result.overall_se, expected_se, rtol=0.1)
