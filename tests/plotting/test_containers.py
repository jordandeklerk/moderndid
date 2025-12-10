"""Tests for plotting data containers."""

import numpy as np
import pytest
from moderndid.plotting.containers import (
    DataArray,
    Dataset,
    iterate_over_selection,
    process_facet_dims,
)


def test_dataarray_basic_creation():
    values = np.array([1, 2, 3])
    dims = ["x"]
    coords = {"x": np.array([0, 1, 2])}

    da = DataArray(values, dims, coords, name="test")

    assert da.shape == (3,)
    assert da.ndim == 1
    assert da.size == 3
    assert da.dims == ("x",)
    assert da.name == "test"
    assert np.array_equal(da.coords["x"], coords["x"])


def test_dataarray_multidimensional():
    values = np.array([[1, 2], [3, 4]])
    dims = ["x", "y"]
    coords = {"x": np.array([0, 1]), "y": np.array([10, 20])}

    da = DataArray(values, dims, coords)

    assert da.shape == (2, 2)
    assert da.ndim == 2
    assert len(da.dims) == 2


def test_dataarray_sel_single_value():
    values = np.array([[1, 2], [3, 4]])
    dims = ["x", "y"]
    coords = {"x": np.array([0, 1]), "y": np.array([10, 20])}

    da = DataArray(values, dims, coords)
    subset = da.sel({"x": 0})

    assert subset.shape == (2,)
    assert np.array_equal(subset.values, np.array([1, 2]))
    assert "x" not in subset.dims
    assert "y" in subset.dims


def test_dataarray_sel_multiple_dims():
    values = np.arange(8).reshape(2, 2, 2)
    dims = ["x", "y", "z"]
    coords = {"x": np.array([0, 1]), "y": np.array([0, 1]), "z": np.array([0, 1])}

    da = DataArray(values, dims, coords)
    subset = da.sel({"x": 1, "y": 0})

    assert subset.shape == (2,)
    assert np.array_equal(subset.values, np.array([4, 5]))


def test_dataarray_item():
    da = DataArray(np.array([42]), ["x"], {"x": np.array([0])})
    subset = da.sel({"x": 0})
    assert subset.item() == 42


def test_dataarray_item_raises_on_multivalue():
    da = DataArray(np.array([1, 2, 3]), ["x"], {"x": np.array([0, 1, 2])})

    with pytest.raises(ValueError, match="Can only convert arrays of size 1"):
        da.item()


def test_dataarray_dimension_validation():
    values = np.array([[1, 2], [3, 4]])
    dims = ["x"]

    with pytest.raises(ValueError, match="Number of dims"):
        DataArray(values, dims, {})


def test_dataarray_coordinate_validation():
    values = np.array([1, 2, 3])
    dims = ["x"]
    coords = {"x": np.array([0, 1])}

    with pytest.raises(ValueError, match="Coordinate x has length"):
        DataArray(values, dims, coords)


def test_dataset_creation():
    data_vars = {
        "a": {"values": np.array([1, 2, 3]), "dims": ["x"], "coords": {"x": np.array([0, 1, 2])}},
        "b": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        },
    }

    ds = Dataset(data_vars)

    assert "a" in ds
    assert "b" in ds
    assert ds.dims == {"x", "y"}
    assert len(list(ds.keys())) == 2


def test_dataset_with_dataarray():
    da1 = DataArray(np.array([1, 2]), ["x"], {"x": np.array([0, 1])}, name="var1")
    da2 = DataArray(np.array([3, 4]), ["y"], {"y": np.array([5, 6])}, name="var2")

    ds = Dataset({"var1": da1, "var2": da2})

    assert "var1" in ds
    assert "var2" in ds


def test_dataset_sel():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        }
    }

    ds = Dataset(data_vars)
    subset = ds.sel({"x": 0})

    assert np.array_equal(subset["a"].values, np.array([1, 2]))


def test_dataset_sel_partial():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["y"], "coords": {"y": np.array([10, 20])}},
    }

    ds = Dataset(data_vars)
    subset = ds.sel({"x": 0})

    assert subset["a"].size == 1
    assert subset["b"].size == 2


def test_iterate_over_selection_simple():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}

    ds = Dataset(data_vars)
    results = iterate_over_selection(ds)

    assert len(results) == 2
    assert results[0] == ("a", {"x": 0}, {"x": 0})
    assert results[1] == ("a", {"x": 1}, {"x": 1})


def test_iterate_over_selection_multidim():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        }
    }

    ds = Dataset(data_vars)
    results = iterate_over_selection(ds)

    assert len(results) == 4
    var_names = [r[0] for r in results]
    assert all(name == "a" for name in var_names)


def test_iterate_over_selection_skip_dims():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        }
    }

    ds = Dataset(data_vars)
    results = iterate_over_selection(ds, skip_dims={"y"})

    assert len(results) == 2


def test_iterate_over_selection_no_coords():
    data_vars = {
        "a": {
            "values": np.array([1, 2, 3]),
            "dims": ["x"],
        }
    }

    ds = Dataset(data_vars)
    results = iterate_over_selection(ds)

    assert len(results) == 3


def test_process_facet_dims_no_faceting():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}
    ds = Dataset(data_vars)

    n_facets, facets_per_var = process_facet_dims(ds, [])

    assert n_facets == 1
    assert not facets_per_var


def test_process_facet_dims_simple():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        }
    }
    ds = Dataset(data_vars)

    n_facets, facets_per_var = process_facet_dims(ds, ["x"])

    assert n_facets == 2
    assert not facets_per_var


def test_process_facet_dims_with_variable():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
    }
    ds = Dataset(data_vars)

    n_facets, facets_per_var = process_facet_dims(ds, ["__variable__"])

    assert n_facets == 2
    assert facets_per_var == {"a": 1, "b": 1}


def test_process_facet_dims_variable_with_dims():
    data_vars = {
        "a": {
            "values": np.array([[1, 2], [3, 4]]),
            "dims": ["x", "y"],
            "coords": {"x": np.array([0, 1]), "y": np.array([10, 20])},
        },
        "b": {
            "values": np.array([5, 6]),
            "dims": ["x"],
            "coords": {"x": np.array([0, 1])},
        },
    }
    ds = Dataset(data_vars)

    n_facets, facets_per_var = process_facet_dims(ds, ["__variable__", "y"])

    assert n_facets == 3
    assert facets_per_var == {"a": 2, "b": 1}


def test_process_facet_dims_missing_dimension_raises():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["y"], "coords": {"y": np.array([10, 20])}},
    }
    ds = Dataset(data_vars)

    with pytest.raises(ValueError, match="All variables must have all faceting dimensions"):
        process_facet_dims(ds, ["x"])
