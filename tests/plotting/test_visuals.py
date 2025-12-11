"""Tests for visual functions."""

import pytest

np = pytest.importorskip("numpy")

from moderndid.plots.containers import DataArray
from moderndid.plots.visuals import (
    errorbar,
    fill_between,
    hline,
    line,
    scatter,
    vline,
)


def test_scatter_basic(simple_data, ax):
    artist = scatter(simple_data, ax)
    assert artist is not None
    assert len(artist.get_offsets()) == 5


def test_scatter_with_color(simple_data, ax):
    artist = scatter(simple_data, ax, color="red")
    assert artist is not None


def test_scatter_with_marker(simple_data, ax):
    artist = scatter(simple_data, ax, marker="s")
    assert artist is not None


def test_scatter_with_custom_coords(simple_data, ax):
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([5, 4, 3, 2, 1])
    artist = scatter(simple_data, ax, x=x, y=y)
    assert artist is not None
    assert len(artist.get_offsets()) == 5


def test_line_basic(simple_data, ax):
    artist = line(simple_data, ax)
    assert artist is not None
    xdata, ydata = artist.get_data()
    assert len(xdata) == 5
    assert len(ydata) == 5


def test_line_with_style(simple_data, ax):
    artist = line(simple_data, ax, color="blue", linestyle="--", linewidth=2)
    assert artist is not None
    assert artist.get_linestyle() == "--"
    assert artist.get_linewidth() == 2


def test_line_with_custom_coords(simple_data, ax):
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 2, 3, 4, 5])
    artist = line(simple_data, ax, x=x, y=y)
    assert artist is not None


def test_errorbar_basic(simple_data, ax):
    artist = errorbar(simple_data, ax, yerr=0.5)
    assert artist is not None


def test_errorbar_with_arrays(simple_data, ax):
    yerr = np.array([0.1, 0.2, 0.1, 0.3, 0.2])
    artist = errorbar(simple_data, ax, yerr=yerr)
    assert artist is not None


def test_errorbar_with_style(simple_data, ax):
    artist = errorbar(simple_data, ax, yerr=0.5, color="red", marker="o", capsize=5)
    assert artist is not None


def test_fill_between_basic(simple_data, ax):
    artist = fill_between(simple_data, ax, y2=0)
    assert artist is not None


def test_fill_between_with_bounds(simple_data, ax):
    y1 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    artist = fill_between(simple_data, ax, y1=y1, y2=y2, alpha=0.3)
    assert artist is not None


def test_hline_basic(ax):
    data = DataArray(np.array([2.5]), ["_"], {"_": np.array([0])})
    artist = hline(data, ax)
    assert artist is not None


def test_hline_with_value(ax):
    data = DataArray(np.array([1]), ["_"], {"_": np.array([0])})
    artist = hline(data, ax, y=3.0, color="red", linestyle="--")
    assert artist is not None


def test_vline_basic(ax):
    data = DataArray(np.array([2.5]), ["_"], {"_": np.array([0])})
    artist = vline(data, ax)
    assert artist is not None


def test_vline_with_value(ax):
    data = DataArray(np.array([1]), ["_"], {"_": np.array([0])})
    artist = vline(data, ax, x=3.0, color="blue", linestyle=":")
    assert artist is not None


def test_scatter_with_all_aesthetics(simple_data, ax):
    artist = scatter(
        simple_data,
        ax,
        color="green",
        marker="^",
        alpha=0.5,
        s=100,
    )
    assert artist is not None


def test_line_with_alpha(simple_data, ax):
    artist = line(simple_data, ax, alpha=0.7)
    assert artist is not None
    assert artist.get_alpha() == 0.7


def test_errorbar_with_xerr(simple_data, ax):
    artist = errorbar(simple_data, ax, xerr=0.2, yerr=0.3)
    assert artist is not None


def test_fill_between_with_color(simple_data, ax):
    artist = fill_between(simple_data, ax, color="lightblue", alpha=0.5)
    assert artist is not None


def test_hline_scalar_extraction(ax):
    data = DataArray(np.array([[[2.5]]]), ["x", "y", "z"], {"x": [0], "y": [0], "z": [0]})
    artist = hline(data, ax)
    assert artist is not None


def test_vline_scalar_extraction(ax):
    data = DataArray(np.array([[[1.5]]]), ["x", "y", "z"], {"x": [0], "y": [0], "z": [0]})
    artist = vline(data, ax)
    assert artist is not None
