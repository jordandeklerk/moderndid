"""Tests for aesthetic mapping system."""

import numpy as np
from moderndid.plotting.aesthetics import (
    generate_aes_mappings,
    get_aes_kwargs,
    get_default_aes_values,
)
from moderndid.plotting.containers import Dataset


def test_get_default_aes_values_from_defaults():
    values = get_default_aes_values("color", 3, {})

    assert len(values) == 3
    assert all(v.startswith("C") for v in values)


def test_get_default_aes_values_from_user():
    user_colors = ["red", "blue", "green"]
    values = get_default_aes_values("color", 3, {"color": user_colors})

    assert values == user_colors


def test_get_default_aes_values_cycles():
    values = get_default_aes_values("color", 5, {"color": ["red", "blue"]})

    assert len(values) == 5
    assert values[:2] == ["red", "blue"]
    assert values[2] == "red"


def test_generate_aes_mappings_simple_dimension():
    data_vars = {"a": {"values": np.array([1, 2, 3]), "dims": ["x"], "coords": {"x": np.array([0, 1, 2])}}}
    ds = Dataset(data_vars)

    aes = {"color": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds)

    assert "color" in aes_dict
    assert "mapping" in aes_dict["color"]
    assert aes_dict["color"]["mapping"].shape == (3,)


def test_generate_aes_mappings_variable():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
    }
    ds = Dataset(data_vars)

    aes = {"color": ["__variable__"]}
    aes_dict = generate_aes_mappings(aes, ds)

    assert "color" in aes_dict
    assert "a" in aes_dict["color"]
    assert "b" in aes_dict["color"]
    assert aes_dict["color"]["a"].size == 1
    assert aes_dict["color"]["b"].size == 1


def test_generate_aes_mappings_variable_with_dims():
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

    aes = {"marker": ["__variable__", "y"]}
    aes_dict = generate_aes_mappings(aes, ds)

    assert "marker" in aes_dict
    assert "a" in aes_dict["marker"]
    assert "b" in aes_dict["marker"]
    assert aes_dict["marker"]["a"].shape == (2,)
    assert aes_dict["marker"]["b"].size == 1


def test_generate_aes_mappings_neutral_element():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["y"], "coords": {"y": np.array([0, 1])}},
    }
    ds = Dataset(data_vars)

    aes = {"color": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds)

    assert "neutral_element" in aes_dict["color"]
    assert "mapping" in aes_dict["color"]


def test_generate_aes_mappings_all_vars_have_dim():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
    }
    ds = Dataset(data_vars)

    aes = {"color": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds)

    assert "mapping" in aes_dict["color"]
    assert "neutral_element" not in aes_dict["color"]


def test_generate_aes_mappings_disabled():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}
    ds = Dataset(data_vars)

    aes = {"color": ["x"], "marker": False}
    aes_dict = generate_aes_mappings(aes, ds)

    assert "color" in aes_dict
    assert "marker" not in aes_dict


def test_generate_aes_mappings_with_user_values():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}
    ds = Dataset(data_vars)

    aes = {"color": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds, color=["red", "blue"])

    mapping = aes_dict["color"]["mapping"]
    assert np.array_equal(mapping.values, np.array(["red", "blue"]))


def test_get_aes_kwargs_simple():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}
    ds = Dataset(data_vars)

    aes = {"color": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds, color=["red", "blue"])

    kwargs = get_aes_kwargs(aes_dict, ["color"], "a", {"x": 0})
    assert kwargs["color"] == "red"

    kwargs = get_aes_kwargs(aes_dict, ["color"], "a", {"x": 1})
    assert kwargs["color"] == "blue"


def test_get_aes_kwargs_variable_specific():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
    }
    ds = Dataset(data_vars)

    aes = {"color": ["__variable__"]}
    aes_dict = generate_aes_mappings(aes, ds, color=["red", "blue"])

    kwargs_a = get_aes_kwargs(aes_dict, ["color"], "a", {})
    kwargs_b = get_aes_kwargs(aes_dict, ["color"], "b", {})

    assert kwargs_a["color"] != kwargs_b["color"]


def test_get_aes_kwargs_neutral_element():
    data_vars = {
        "a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}},
        "b": {"values": np.array([3, 4]), "dims": ["y"], "coords": {"y": np.array([0, 1])}},
    }
    ds = Dataset(data_vars)

    aes = {"color": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds, color=["red", "blue", "green"])

    kwargs_a = get_aes_kwargs(aes_dict, ["color"], "a", {"x": 0})
    assert kwargs_a["color"] in ["red", "blue", "green"]

    kwargs_b = get_aes_kwargs(aes_dict, ["color"], "b", {})
    assert kwargs_b["color"] in ["red", "blue", "green"]


def test_get_aes_kwargs_skips_overlay():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}
    ds = Dataset(data_vars)

    aes = {"overlay": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds)

    kwargs = get_aes_kwargs(aes_dict, ["overlay"], "a", {"x": 0})
    assert "overlay" not in kwargs


def test_get_aes_kwargs_multiple_aesthetics():
    data_vars = {"a": {"values": np.array([1, 2]), "dims": ["x"], "coords": {"x": np.array([0, 1])}}}
    ds = Dataset(data_vars)

    aes = {"color": ["x"], "marker": ["x"]}
    aes_dict = generate_aes_mappings(aes, ds, color=["red", "blue"], marker=["o", "s"])

    kwargs = get_aes_kwargs(aes_dict, ["color", "marker"], "a", {"x": 0})

    assert kwargs["color"] == "red"
    assert kwargs["marker"] == "o"
