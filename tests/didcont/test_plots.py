"""Tests for continuous DID plotting utilities."""

from __future__ import annotations

from pathlib import Path

from moderndid.didcont.cont_did import cont_did
from moderndid.didcont.plots import plot_cont_did
from moderndid.plots import PlotCollection
from tests.helpers import importorskip

np = importorskip("numpy")
pd = importorskip("pandas")
plt = importorskip("matplotlib.pyplot")


DATA_DIR = Path(__file__).resolve().parent / "data"


def test_dose_response_plot_produces_finite_outputs():
    df_raw = _load_base_data()
    df_dose = df_raw.copy()
    df_dose.loc[df_dose["time_period"] < df_dose["G"], "D"] = 0

    result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df_dose,
        gname="G",
        target_parameter="slope",
        aggregation="dose",
        treatment_type="continuous",
        control_group="notyettreated",
        biters=40,
        cband=True,
        num_knots=1,
        degree=3,
    )

    assert result.dose.ndim == 1
    assert result.att_d.ndim == 1
    assert result.dose.shape == result.att_d.shape
    assert np.all(np.isfinite(result.dose))
    assert np.all(np.isfinite(result.att_d))

    pc_att = plot_cont_did(result, type="att", show_ci=True)
    assert isinstance(pc_att, PlotCollection)
    fig_att = pc_att.viz["figure"]
    assert fig_att.axes
    plt.close(fig_att)

    pc_acrt = plot_cont_did(result, type="acrt", show_ci=True)
    assert isinstance(pc_acrt, PlotCollection)
    fig_acrt = pc_acrt.viz["figure"]
    assert fig_acrt.axes
    plt.close(fig_acrt)


def test_event_study_plots_have_consistent_dimensions():
    df_raw = _load_base_data()

    level_result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df_raw,
        gname="G",
        target_parameter="level",
        aggregation="eventstudy",
        treatment_type="continuous",
        control_group="notyettreated",
        biters=40,
        cband=True,
        num_knots=1,
        degree=3,
    )

    level_event = level_result.event_study
    assert level_event is not None
    assert level_event.event_times.shape == level_event.att_by_event.shape
    assert level_event.att_by_event.shape == level_event.se_by_event.shape
    assert np.all(np.isfinite(level_event.att_by_event))
    assert np.all(level_event.se_by_event >= 0)

    pc_level = plot_cont_did(level_result)
    assert isinstance(pc_level, PlotCollection)
    fig_level = pc_level.viz["figure"]
    assert fig_level.axes
    plt.close(fig_level)

    slope_result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df_raw,
        gname="G",
        target_parameter="slope",
        aggregation="eventstudy",
        treatment_type="continuous",
        control_group="notyettreated",
        biters=40,
        cband=True,
        num_knots=1,
        degree=3,
    )

    slope_event = slope_result.event_study
    assert slope_event is not None
    assert slope_event.event_times.shape == slope_event.att_by_event.shape
    assert slope_event.att_by_event.shape == slope_event.se_by_event.shape
    assert np.all(np.isfinite(slope_event.att_by_event))

    pc_slope = plot_cont_did(slope_result, type="acrt")
    assert isinstance(pc_slope, PlotCollection)
    fig_slope = pc_slope.viz["figure"]
    assert fig_slope.axes
    plt.close(fig_slope)


def test_cck_dose_response_is_well_defined():
    df_cck = _load_cck_data()

    result = cont_did(
        yname="Y",
        tname="time_period",
        idname="id",
        dname="D",
        data=df_cck,
        gname="G",
        target_parameter="level",
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="cck",
        control_group="notyettreated",
        biters=40,
        cband=True,
    )

    assert result.dose.shape == result.att_d.shape
    assert np.all(np.isfinite(result.dose))
    assert np.all(np.isfinite(result.att_d))
    assert np.isfinite(result.overall_att)
    assert np.isfinite(result.overall_acrt)

    pc = plot_cont_did(result, type="att", show_ci=True)
    assert isinstance(pc, PlotCollection)
    fig = pc.viz["figure"]
    assert fig.axes
    plt.close(fig)


def _load_base_data():
    df_raw = pd.read_csv(DATA_DIR / "cont_test_data.csv.gz")
    df_raw.loc[df_raw["G"] == 0, "D"] = 0
    return df_raw


def _load_cck_data():
    df_cck = pd.read_csv(DATA_DIR / "cont_test_data_cck.csv.gz")
    df_cck.loc[df_cck["G"] == 0, "D"] = 0
    return df_cck
