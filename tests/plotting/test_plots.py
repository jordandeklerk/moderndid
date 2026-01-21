"""Tests for plotnine-based plotting functions."""

import pytest

plotnine = pytest.importorskip("plotnine")

from plotnine import ggplot

from moderndid.plots import (
    plot_att_gt,
    plot_dose_response,
    plot_event_study,
    plot_sensitivity,
)


def test_plot_att_gt_returns_ggplot(mp_result):
    plot = plot_att_gt(mp_result)
    assert isinstance(plot, ggplot)


def test_plot_att_gt_no_ci(mp_result):
    plot = plot_att_gt(mp_result, show_ci=False)
    assert isinstance(plot, ggplot)


def test_plot_att_gt_no_ref_line(mp_result):
    plot = plot_att_gt(mp_result, ref_line=None)
    assert isinstance(plot, ggplot)


def test_plot_att_gt_custom_labels(mp_result):
    plot = plot_att_gt(mp_result, xlab="Time Period", ylab="Effect", title="Cohort")
    assert isinstance(plot, ggplot)


def test_plot_event_study_returns_ggplot(aggte_result_dynamic):
    plot = plot_event_study(aggte_result_dynamic)
    assert isinstance(plot, ggplot)


def test_plot_event_study_no_ci(aggte_result_dynamic):
    plot = plot_event_study(aggte_result_dynamic, show_ci=False)
    assert isinstance(plot, ggplot)


def test_plot_event_study_custom_labels(aggte_result_dynamic):
    plot = plot_event_study(aggte_result_dynamic, xlab="Relative Time", ylab="Treatment Effect", title="Dynamic ATT")
    assert isinstance(plot, ggplot)


def test_plot_event_study_pte_result(pte_result_with_event_study):
    plot = plot_event_study(pte_result_with_event_study)
    assert isinstance(plot, ggplot)


def test_plot_event_study_wrong_aggregation_raises(aggte_result_simple):
    with pytest.raises(ValueError, match="dynamic aggregation"):
        plot_event_study(aggte_result_simple)


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
