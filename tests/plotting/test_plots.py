"""Tests for plotnine-based plotting functions."""

import pytest

plotnine = pytest.importorskip("plotnine")

from plotnine import ggplot

from moderndid.plots import (
    plot_agg,
    plot_dose_response,
    plot_event_study,
    plot_gt,
    plot_multiplegt,
    plot_sensitivity,
)


def test_plot_gt_mp_result(mp_result):
    plot = plot_gt(mp_result)
    assert isinstance(plot, ggplot)


def test_plot_gt_ddd_result(ddd_mp_result):
    plot = plot_gt(ddd_mp_result)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("show_ci", [True, False])
def test_plot_gt_show_ci(mp_result, show_ci):
    plot = plot_gt(mp_result, show_ci=show_ci)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("ref_line", [0, None, 0.5])
def test_plot_gt_ref_line(mp_result, ref_line):
    plot = plot_gt(mp_result, ref_line=ref_line)
    assert isinstance(plot, ggplot)


def test_plot_gt_custom_labels(mp_result):
    plot = plot_gt(mp_result, xlab="Time Period", ylab="Effect", title="Cohort")
    assert isinstance(plot, ggplot)


def test_plot_gt_invalid_input_raises():
    with pytest.raises(TypeError, match="plot_gt requires"):
        plot_gt("invalid")


def test_plot_event_study_aggte_result(aggte_result_dynamic):
    plot = plot_event_study(aggte_result_dynamic)
    assert isinstance(plot, ggplot)


def test_plot_event_study_ddd_result(ddd_agg_eventstudy):
    plot = plot_event_study(ddd_agg_eventstudy)
    assert isinstance(plot, ggplot)


def test_plot_event_study_pte_result(pte_result_with_event_study):
    plot = plot_event_study(pte_result_with_event_study)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("show_ci", [True, False])
def test_plot_event_study_show_ci(aggte_result_dynamic, show_ci):
    plot = plot_event_study(aggte_result_dynamic, show_ci=show_ci)
    assert isinstance(plot, ggplot)


def test_plot_event_study_custom_labels(aggte_result_dynamic):
    plot = plot_event_study(aggte_result_dynamic, xlab="Relative Time", ylab="Treatment Effect", title="Dynamic ATT")
    assert isinstance(plot, ggplot)


def test_plot_event_study_wrong_aggte_aggregation_raises(aggte_result_simple):
    with pytest.raises(ValueError, match="dynamic aggregation"):
        plot_event_study(aggte_result_simple)


def test_plot_event_study_wrong_ddd_aggregation_raises(ddd_agg_simple):
    with pytest.raises(ValueError, match="eventstudy aggregation"):
        plot_event_study(ddd_agg_simple)


def test_plot_event_study_invalid_input_raises():
    with pytest.raises(TypeError, match="plot_event_study requires"):
        plot_event_study("invalid")


@pytest.mark.parametrize("fixture_name", ["aggte_result_group", "aggte_result_calendar"])
def test_plot_agg_aggte_result(fixture_name, request):
    result = request.getfixturevalue(fixture_name)
    plot = plot_agg(result)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("fixture_name", ["ddd_agg_group", "ddd_agg_calendar"])
def test_plot_agg_ddd_result(fixture_name, request):
    result = request.getfixturevalue(fixture_name)
    plot = plot_agg(result)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("show_ci", [True, False])
def test_plot_agg_show_ci(ddd_agg_group, show_ci):
    plot = plot_agg(ddd_agg_group, show_ci=show_ci)
    assert isinstance(plot, ggplot)


def test_plot_agg_custom_labels(ddd_agg_group):
    plot = plot_agg(ddd_agg_group, xlab="Group", ylab="Effect", title="Custom")
    assert isinstance(plot, ggplot)


def test_plot_agg_wrong_aggte_aggregation_raises(aggte_result_dynamic):
    with pytest.raises(ValueError, match="group or calendar aggregation"):
        plot_agg(aggte_result_dynamic)


def test_plot_agg_wrong_ddd_aggregation_raises(ddd_agg_eventstudy):
    with pytest.raises(ValueError, match="group or calendar aggregation"):
        plot_agg(ddd_agg_eventstudy)


def test_plot_agg_invalid_input_raises():
    with pytest.raises(TypeError, match="plot_agg requires"):
        plot_agg("invalid")


def test_plot_dose_response_returns_ggplot(dose_result):
    plot = plot_dose_response(dose_result)
    assert isinstance(plot, ggplot)


def test_plot_dose_response_acrt(dose_result):
    plot = plot_dose_response(dose_result, effect_type="acrt")
    assert isinstance(plot, ggplot)


def test_plot_dose_response_no_ci(dose_result):
    plot = plot_dose_response(dose_result, show_ci=False)
    assert isinstance(plot, ggplot)


def test_plot_dose_response_custom_labels(dose_result):
    plot = plot_dose_response(dose_result, xlab="Dose Level", ylab="Effect", title="Custom Title")
    assert isinstance(plot, ggplot)


def test_plot_sensitivity_returns_ggplot(honest_result):
    plot = plot_sensitivity(honest_result)
    assert isinstance(plot, ggplot)


def test_plot_sensitivity_no_ref_line(honest_result):
    plot = plot_sensitivity(honest_result, ref_line=None)
    assert isinstance(plot, ggplot)


def test_plot_sensitivity_custom_labels(honest_result):
    plot = plot_sensitivity(honest_result, xlab="Parameter", ylab="CI", title="Sensitivity")
    assert isinstance(plot, ggplot)


def test_plot_multiplegt_returns_ggplot(didinter_result):
    plot = plot_multiplegt(didinter_result)
    assert isinstance(plot, ggplot)


def test_plot_multiplegt_with_placebos(didinter_result_with_placebos):
    plot = plot_multiplegt(didinter_result_with_placebos)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("show_ci", [True, False])
def test_plot_multiplegt_show_ci(didinter_result, show_ci):
    plot = plot_multiplegt(didinter_result, show_ci=show_ci)
    assert isinstance(plot, ggplot)


@pytest.mark.parametrize("ref_line", [0, None, 0.5])
def test_plot_multiplegt_ref_line(didinter_result, ref_line):
    plot = plot_multiplegt(didinter_result, ref_line=ref_line)
    assert isinstance(plot, ggplot)


def test_plot_multiplegt_custom_labels(didinter_result):
    plot = plot_multiplegt(didinter_result, xlab="Event Horizon", ylab="Treatment Effect", title="Custom Title")
    assert isinstance(plot, ggplot)
