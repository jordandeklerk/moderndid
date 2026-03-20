"""Tests for the EMFX aggregation function."""

import numpy as np
import pytest
from scipy import stats

from tests.helpers import importorskip

pl = importorskip("polars")
importorskip("pyfixest")

from moderndid import emfx, etwfe
from moderndid.etwfe.container import EmfxResult


def test_emfx_simple_returns_emfx_result(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert isinstance(result, EmfxResult)


def test_emfx_simple_overall_att(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    np.testing.assert_allclose(result.overall_att, -0.04771, atol=1e-4)
    np.testing.assert_allclose(result.overall_se, 0.012341, atol=1e-4)


def test_emfx_simple_no_disaggregated_arrays(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert result.event_times is None
    assert result.att_by_event is None
    assert result.se_by_event is None
    assert result.ci_lower is None
    assert result.ci_upper is None


@pytest.mark.parametrize("agg_type", ["simple", "group", "calendar", "event"])
def test_emfx_aggregation_type_field(etwfe_baseline, agg_type):
    result = emfx(etwfe_baseline, type=agg_type)
    assert result.aggregation_type == agg_type


def test_emfx_event_times(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    np.testing.assert_array_equal(result.event_times, [0.0, 1.0, 2.0, 3.0])


def test_emfx_event_att_values(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    expected = np.array([-0.031067, -0.052235, -0.136078, -0.104707])
    np.testing.assert_allclose(result.att_by_event, expected, atol=1e-4)


def test_emfx_event_se_values(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    expected = np.array([0.013204, 0.017089, 0.030408, 0.032949])
    np.testing.assert_allclose(result.se_by_event, expected, atol=1e-4)


@pytest.mark.parametrize("agg_type", ["event", "group", "calendar"])
def test_emfx_overall_consistent_across_types(etwfe_baseline, agg_type):
    simple = emfx(etwfe_baseline, type="simple")
    result = emfx(etwfe_baseline, type=agg_type)
    np.testing.assert_allclose(result.overall_att, simple.overall_att, atol=1e-10)
    np.testing.assert_allclose(result.overall_se, simple.overall_se, atol=1e-10)


def test_emfx_event_ci_computation(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    z = result.critical_value
    np.testing.assert_allclose(result.ci_lower, result.att_by_event - z * result.se_by_event, atol=1e-10)
    np.testing.assert_allclose(result.ci_upper, result.att_by_event + z * result.se_by_event, atol=1e-10)


@pytest.mark.parametrize("agg_type", ["event", "group", "calendar"])
def test_emfx_se_positive(etwfe_baseline, agg_type):
    result = emfx(etwfe_baseline, type=agg_type)
    assert np.all(result.se_by_event > 0)


@pytest.mark.parametrize("agg_type", ["event", "group", "calendar"])
def test_emfx_ci_arrays_present(etwfe_baseline, agg_type):
    result = emfx(etwfe_baseline, type=agg_type)
    assert result.ci_lower is not None
    assert result.ci_upper is not None
    assert len(result.ci_lower) == len(result.att_by_event)


def test_emfx_group_values(etwfe_baseline):
    result = emfx(etwfe_baseline, type="group")
    np.testing.assert_array_equal(result.event_times, [2004.0, 2006.0, 2007.0])
    np.testing.assert_allclose(result.att_by_event, [-0.084619, -0.018339, -0.043106], atol=1e-4)
    np.testing.assert_allclose(result.se_by_event, [0.025016, 0.015958, 0.017891], atol=1e-4)


def test_emfx_calendar_values(etwfe_baseline):
    result = emfx(etwfe_baseline, type="calendar")
    np.testing.assert_array_equal(result.event_times, [2004.0, 2005.0, 2006.0, 2007.0])
    np.testing.assert_allclose(result.att_by_event, [-0.019372, -0.078319, -0.043683, -0.048737], atol=1e-4)
    np.testing.assert_allclose(result.se_by_event, [0.030820, 0.027551, 0.016639, 0.015147], atol=1e-4)


def test_emfx_event_window(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event", window=(0, 2))
    np.testing.assert_array_equal(result.event_times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.att_by_event, [-0.031067, -0.052235, -0.136078], atol=1e-4)


def test_emfx_event_window_excludes_outside(etwfe_baseline):
    full = emfx(etwfe_baseline, type="event")
    windowed = emfx(etwfe_baseline, type="event", window=(0, 2))
    assert len(windowed.event_times) < len(full.event_times)
    assert 3.0 not in windowed.event_times


def test_emfx_never_event_has_pretreatment(etwfe_never):
    result = emfx(etwfe_never, type="event", post_only=False)
    pre_times = result.event_times[result.event_times < 0]
    assert len(pre_times) > 0


def test_emfx_never_event_pretreatment_values(etwfe_never):
    result = emfx(etwfe_never, type="event", post_only=False)
    np.testing.assert_array_equal(result.event_times, [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    pre_mask = result.event_times < 0
    pre_atts = result.att_by_event[pre_mask]
    np.testing.assert_allclose(pre_atts, [0.003306, 0.025022, 0.024459, 0.0], atol=1e-4)


def test_emfx_never_post_only_fewer_event_times(etwfe_never):
    post = emfx(etwfe_never, type="event", post_only=True)
    full = emfx(etwfe_never, type="event", post_only=False)
    assert len(post.event_times) < len(full.event_times)


def test_emfx_never_event_window_with_pretreatment(etwfe_never):
    result = emfx(etwfe_never, type="event", post_only=False, window=(-3, 1))
    assert result.event_times[0] == -3.0
    assert result.event_times[-1] == 1.0


def test_emfx_custom_alpha(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", alp=0.10)
    result = emfx(mod, type="simple")
    np.testing.assert_allclose(result.critical_value, stats.norm.ppf(0.95), atol=1e-6)


def test_emfx_alpha_05_wider_than_10(mpdta_data):
    mod05 = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", alp=0.05)
    mod10 = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", alp=0.10)
    e05 = emfx(mod05, type="event")
    e10 = emfx(mod10, type="event")
    assert e05.critical_value > e10.critical_value
    assert (e05.ci_upper[0] - e05.ci_lower[0]) > (e10.ci_upper[0] - e10.ci_lower[0])


def test_emfx_invalid_type(etwfe_baseline):
    with pytest.raises(ValueError, match="type must be"):
        emfx(etwfe_baseline, type="invalid")


def test_emfx_wrong_input_type():
    with pytest.raises(TypeError, match="Expected EtwfeResult"):
        emfx("not_a_result", type="simple")


def test_emfx_preserves_n_obs(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert result.n_obs == etwfe_baseline.n_obs


def test_emfx_estimation_params_propagated(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    assert "yname" in result.estimation_params
    assert "alpha" in result.estimation_params
    assert result.estimation_params["yname"] == "lemp"
