"""Tests for PlotCollection class."""

# pylint: disable=redefined-outer-name

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from moderndid.plotting.collection import PlotCollection
from moderndid.plotting.containers import Dataset

matplotlib.use("Agg")


@pytest.fixture
def simple_dataset():
    data_vars = {"a": {"values": np.array([1, 2, 3]), "dims": ["x"], "coords": {"x": np.array([0, 1, 2])}}}
    return Dataset(data_vars)


@pytest.fixture
def multidim_dataset():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        }
    }
    return Dataset(data_vars)


@pytest.fixture
def multivar_dataset():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
    }
    return Dataset(data_vars)


def test_plotcollection_grid_simple(simple_dataset):
    pc = PlotCollection.grid(simple_dataset)

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    assert isinstance(pc.viz["figure"], matplotlib.figure.Figure)
    plt.close(pc.viz["figure"])


def test_plotcollection_grid_with_rows(multidim_dataset):
    pc = PlotCollection.grid(multidim_dataset, rows=["x"])

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    assert pc.viz["plot"].shape == (2,)
    plt.close(pc.viz["figure"])


def test_plotcollection_grid_with_cols(multidim_dataset):
    pc = PlotCollection.grid(multidim_dataset, cols=["y"])

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    assert pc.viz["plot"].shape == (2,)
    plt.close(pc.viz["figure"])


def test_plotcollection_grid_rows_and_cols(multidim_dataset):
    pc = PlotCollection.grid(multidim_dataset, rows=["x"], cols=["y"])

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    assert pc.viz["plot"].shape == (2, 2)
    plt.close(pc.viz["figure"])


def test_plotcollection_grid_with_aesthetics(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, aes={"color": ["x"]})

    assert "color" in pc.aes
    assert pc.aes_set == {"color"}
    plt.close(pc.viz["figure"])


def test_plotcollection_grid_duplicate_dims_raises():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        }
    }
    ds = Dataset(data_vars)

    with pytest.raises(ValueError, match="Same dimension cannot be in both rows and cols"):
        PlotCollection.grid(ds, rows=["x"], cols=["x"])


def test_plotcollection_wrap_simple(simple_dataset):
    pc = PlotCollection.wrap(simple_dataset, cols=["x"], col_wrap=2)

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    plt.close(pc.viz["figure"])


def test_plotcollection_wrap_no_wrapping(simple_dataset):
    pc = PlotCollection.wrap(simple_dataset, cols=["x"], col_wrap=5)

    assert "figure" in pc.viz
    plt.close(pc.viz["figure"])


def test_plotcollection_wrap_with_wrapping():
    data_vars = {"a": {"values": np.arange(6), "dims": ["x"], "coords": {"x": np.arange(6)}}}
    ds = Dataset(data_vars)

    pc = PlotCollection.wrap(ds, cols=["x"], col_wrap=3)

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    plt.close(pc.viz["figure"])


def test_plotcollection_facet_dims(multidim_dataset):
    pc = PlotCollection.grid(multidim_dataset, rows=["x"], cols=["y"])

    facet_dims = pc.facet_dims
    assert "x" in facet_dims
    assert "y" in facet_dims
    plt.close(pc.viz["figure"])


def test_plotcollection_map_simple(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"])

    call_count = [0]

    def scatter(data, target, **_kwargs):
        call_count[0] += 1
        return target.scatter([0], [data.values.mean()])

    pc.map(scatter, "points", data="a")

    assert "points" in pc.viz
    assert "a" in pc.viz["points"]
    assert call_count[0] == 3
    plt.close(pc.viz["figure"])


def test_plotcollection_map_with_aesthetics(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"], aes={"color": ["x"]}, color=["red", "blue", "green"])

    received_colors = []

    def scatter(data, target, color=None, **_kwargs):
        received_colors.append(color)
        val = data.values.item() if data.values.ndim == 0 else data.values[0]
        return target.scatter([0], [val], c=color)

    pc.map(scatter, "points", data="a")

    assert len(received_colors) == 3
    assert set(received_colors) == {"red", "blue", "green"}
    plt.close(pc.viz["figure"])


def test_plotcollection_map_ignore_aes(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"], aes={"color": ["x"], "marker": ["x"]})

    received_kwargs = []

    def scatter(data, target, **kwargs):
        received_kwargs.append(kwargs.copy())
        val = data.values.item() if data.values.ndim == 0 else data.values[0]
        return target.scatter([0], [val])

    pc.map(scatter, "points", data="a", ignore_aes={"marker"})

    assert all("color" in kw for kw in received_kwargs)
    assert all("marker" not in kw for kw in received_kwargs)
    plt.close(pc.viz["figure"])


def test_plotcollection_map_ignore_all_aes(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"], aes={"color": ["x"]})

    received_kwargs = []

    def scatter(data, target, **kwargs):
        received_kwargs.append(kwargs.copy())
        val = data.values.item() if data.values.ndim == 0 else data.values[0]
        return target.scatter([0], [val])

    pc.map(scatter, "points", data="a", ignore_aes="all")

    assert all("color" not in kw for kw in received_kwargs)
    plt.close(pc.viz["figure"])


def test_plotcollection_map_with_extra_kwargs(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"])

    def scatter(data, target, s=10, **_kwargs):
        val = data.values.item() if data.values.ndim == 0 else data.values[0]
        return target.scatter([0], [val], s=s)

    pc.map(scatter, "points", data="a", s=50)

    assert "points" in pc.viz
    plt.close(pc.viz["figure"])


def test_plotcollection_map_chaining(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"])

    def scatter(data, target, **_kwargs):
        return target.scatter([0], [data.values.mean()])

    def line(data, target, **_kwargs):
        return target.plot([0, 1], [data.values.mean()] * 2)[0]

    pc.map(scatter, "points", data="a").map(line, "lines", data="a")

    assert "points" in pc.viz
    assert "lines" in pc.viz
    plt.close(pc.viz["figure"])


def test_plotcollection_map_no_store(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, cols=["x"])

    def scatter(data, target, **_kwargs):
        return target.scatter([0], [data.values.mean()])

    pc.map(scatter, "points", data="a", store_artist=False)

    assert "points" not in pc.viz
    plt.close(pc.viz["figure"])


def test_plotcollection_coords_property(simple_dataset):
    pc = PlotCollection.grid(simple_dataset)

    assert pc.coords is None

    pc.coords = {"x": 0}
    assert pc.coords == {"x": 0}
    plt.close(pc.viz["figure"])


def test_plotcollection_backend_attribute(simple_dataset):
    pc = PlotCollection.grid(simple_dataset)

    assert pc.backend == "matplotlib"
    plt.close(pc.viz["figure"])


def test_plotcollection_variable_faceting(multivar_dataset):
    pc = PlotCollection.grid(multivar_dataset, cols=["__variable__"])

    assert "figure" in pc.viz
    assert "plot" in pc.viz
    assert isinstance(pc.viz["plot"], dict)
    assert "a" in pc.viz["plot"]
    assert "b" in pc.viz["plot"]
    plt.close(pc.viz["figure"])


def test_plotcollection_savefig(simple_dataset, tmp_path):
    pc = PlotCollection.grid(simple_dataset)

    output_file = tmp_path / "test.png"
    pc.savefig(str(output_file))

    assert output_file.exists()
    plt.close(pc.viz["figure"])


def test_plotcollection_aes_set_property(simple_dataset):
    pc = PlotCollection.grid(simple_dataset, aes={"color": ["x"], "marker": ["x"]})

    assert pc.aes_set == {"color", "marker"}
    plt.close(pc.viz["figure"])
