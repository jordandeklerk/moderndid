"""Tests for drawing didml results with plot_gt and plot_event_study."""

import numpy as np
import pytest

from tests.helpers import importorskip

importorskip("plotnine")

from plotnine import ggplot

from moderndid import plot_event_study, plot_gt


def test_plot_gt_draws_a_didml_fit_from_its_group_time_cells(mpdta_analytic_fit):
    fit = mpdta_analytic_fit

    plot = plot_gt(fit)

    assert isinstance(plot, ggplot)
    np.testing.assert_array_equal(plot.data["group"].to_numpy(), fit.groups)
    np.testing.assert_array_equal(plot.data["time"].to_numpy(), fit.times)
    np.testing.assert_allclose(plot.data["att"].to_numpy(), fit.att_gt, rtol=1e-12)
    assert plot.data["group_label"].to_list() == [str(int(group)) for group in fit.groups]


def test_plot_event_study_draws_a_didml_dynamic_aggregation(mpdta_dynamic_agg):
    event_times = mpdta_dynamic_agg.event_times
    agg = mpdta_dynamic_agg._replace(se_by_event=np.where(event_times == -1, np.nan, mpdta_dynamic_agg.se_by_event))
    keep = np.isfinite(agg.se_by_event)

    plot = plot_event_study(agg)

    assert isinstance(plot, ggplot)
    assert len(plot.data) == len(event_times) - 1
    np.testing.assert_array_equal(plot.data["event_time"].to_numpy(), event_times[keep])
    np.testing.assert_allclose(plot.data["att"].to_numpy(), agg.att_by_event[keep], rtol=1e-12)


def test_plot_event_study_rejects_a_group_didml_aggregation(mpdta_group_agg):
    with pytest.raises(ValueError, match="requires dynamic aggregation, got group"):
        plot_event_study(mpdta_group_agg)
