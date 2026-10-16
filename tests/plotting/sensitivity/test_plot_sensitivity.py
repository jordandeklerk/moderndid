"""Tests for the sensitivity plot."""

from tests.helpers import importorskip

importorskip("plotnine")

from plotnine import ggplot

from moderndid.plots import plot_sensitivity


def test_plot_sensitivity_returns_ggplot(honest_result):
    plot = plot_sensitivity(honest_result)
    assert isinstance(plot, ggplot)


def test_plot_sensitivity_no_ref_line(honest_result):
    plot = plot_sensitivity(honest_result, ref_line=None)
    assert isinstance(plot, ggplot)


def test_plot_sensitivity_custom_labels(honest_result):
    plot = plot_sensitivity(honest_result, xlab="Parameter", ylab="CI", title="Sensitivity")
    assert isinstance(plot, ggplot)
