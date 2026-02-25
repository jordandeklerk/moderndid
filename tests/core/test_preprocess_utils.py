"""Tests for preprocessing utility functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess.utils import (
    add_intercept,
    check_partition_collinearity,
    choose_knots_quantile,
    create_ddd_subgroups,
    create_dose_grid,
    extract_covariates,
    extract_ddd_covariates,
    extract_vars_from_formula,
    get_covariate_names_from_formula,
    get_first_difference,
    get_group,
    is_balanced_panel,
    make_balanced_panel,
    map_to_idx,
    parse_formula,
    remove_collinear,
    two_by_two_subset,
    validate_dose_values,
    validate_subgroup_sizes,
)


@pytest.mark.parametrize(
    "val, time_map, expected_check",
    [
        (2004.0, {2004.0: 1, 2006.0: 2}, lambda r: r == 1),
        (999.0, {}, lambda r: r == 999.0),
        (float("inf"), {1.0: 10}, lambda r: np.isinf(r)),
    ],
)
def test_map_to_idx_scalar(val, time_map, expected_check):
    assert expected_check(map_to_idx(val, time_map))


def test_map_to_idx_array_no_inf():
    time_map = {1.0: 10, 2.0: 20}
    result = map_to_idx([1.0, 2.0], time_map)
    np.testing.assert_array_equal(result, [10, 20])
    assert result.dtype == int


def test_map_to_idx_array_with_inf():
    time_map = {1.0: 10}
    result = map_to_idx([1.0, float("inf")], time_map)
    assert result[0] == 10.0
    assert np.isinf(result[1])
    assert result.dtype == float


def test_make_balanced_panel():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "time": [1, 2, 1, 2, 1],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = make_balanced_panel(df, "id", "time")
    assert set(result["id"].unique().to_list()) == {1, 2}


def test_make_balanced_panel_empty():
    df = pl.DataFrame({"id": [], "time": [], "y": []}).cast({"id": pl.Int64, "time": pl.Int64, "y": pl.Float64})
    result = make_balanced_panel(df, "id", "time")
    assert result.is_empty()


def test_get_first_difference():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "y": [10.0, 12.0, 15.0, 20.0, 22.0, 25.0],
        }
    )
    result = get_first_difference(df, "id", "y", "time")
    assert "dy" in result.columns
    dy_unit1 = result.filter(pl.col("id") == 1).sort("time")["dy"].to_list()
    assert dy_unit1[1] == pytest.approx(2.0)
    assert dy_unit1[2] == pytest.approx(3.0)


def test_get_group_with_treat_period():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "treat": [0, 1, 0, 0],
        }
    )
    result = get_group(df, "id", "time", "treat", treat_period=2)
    g_vals = result.sort(["id", "time"])["G"].to_list()
    assert g_vals == [2, 2, 0, 0]


def test_get_group_first_switch_detection():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treat": [0, 0, 1, 0, 1, 1, 0, 0, 0],
        }
    )
    result = get_group(df, "id", "time", "treat")
    groups = result.group_by("id").first().sort("id")["G"].to_list()
    assert groups == [3, 2, 0]


@pytest.mark.parametrize(
    "control_group, base_period, g, tp, check_fn",
    [
        ("notyettreated", "varying", 2.0, 2, lambda r: r["n1"] > 0 and not r["gt_data"].is_empty()),
        ("nevertreated", "varying", 2.0, 2, lambda r: r["n1"] == 2),
        ("notyettreated", "universal", 3.0, 3, lambda r: not r["gt_data"].is_empty()),
    ],
)
def test_two_by_two_subset_control_groups(control_group, base_period, g, tp, check_fn):
    if base_period == "universal":
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "period": [1, 2, 3, 1, 2, 3],
                "G": [3.0, 3.0, 3.0, float("inf"), float("inf"), float("inf")],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
    elif control_group == "nevertreated":
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "G": [2.0, 2.0, float("inf"), float("inf")],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "period": [1, 2, 1, 2, 1, 2],
                "G": [2.0, 2.0, 3.0, 3.0, float("inf"), float("inf")],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
    result = two_by_two_subset(df, g=g, tp=tp, control_group=control_group, base_period=base_period)
    assert check_fn(result)


def test_two_by_two_subset_insufficient_variation():
    df = pl.DataFrame(
        {
            "id": [1, 1],
            "period": [1, 2],
            "G": [2.0, 2.0],
            "y": [1.0, 2.0],
        }
    )
    result = two_by_two_subset(df, g=2.0, tp=2)
    assert result["n1"] == 0
    assert result["gt_data"].is_empty()


@pytest.mark.parametrize(
    "x, num_knots, expected_len",
    [
        (np.linspace(0, 10, 100), 3, 3),
        (np.array([1, 2, 3]), 0, 0),
        (np.array([]), 3, 0),
    ],
)
def test_choose_knots_quantile(x, num_knots, expected_len):
    assert len(choose_knots_quantile(x, num_knots)) == expected_len


@pytest.mark.parametrize(
    "doses, expected_len, check_fn",
    [
        (np.array([0.0, 0.5, 1.0, 2.0]), 50, lambda g: g[0] == pytest.approx(0.5) and g[-1] == pytest.approx(2.0)),
        (np.array([0.0, 0.0, -1.0]), 0, lambda g: True),
    ],
)
def test_create_dose_grid(doses, expected_len, check_fn):
    grid = create_dose_grid(doses)
    assert len(grid) == expected_len
    assert check_fn(grid)


@pytest.mark.parametrize(
    "dose, groups, is_valid, error_substr, warning_substr",
    [
        (np.array([0.0, 0.5, 1.0]), np.array([0, 2, 2]), True, None, None),
        (np.array([-1.0, 0.5]), np.array([2, 2]), False, "Negative", None),
        (np.array([0.5, 0.0]), np.array([float("inf"), 2]), True, None, "never-treated"),
        (np.array([0.0, 0.5]), np.array([2, 2]), True, None, "zero dose"),
    ],
)
def test_validate_dose_values(dose, groups, is_valid, error_substr, warning_substr):
    result = validate_dose_values(dose, groups)
    assert result["is_valid"] == is_valid
    if error_substr:
        assert any(error_substr in e for e in result["errors"])
    if warning_substr:
        assert any(warning_substr in w for w in result["warnings"])


@pytest.mark.parametrize(
    "formula, expected_outcome, expected_predictors",
    [
        ("y ~ x1 + x2", "y", ["x1", "x2"]),
    ],
)
def test_parse_formula_basic(formula, expected_outcome, expected_predictors):
    parsed = parse_formula(formula)
    assert parsed["outcome"] == expected_outcome
    assert parsed["predictors"] == expected_predictors


def test_parse_formula_with_functions():
    parsed = parse_formula("y ~ log(x1) + I(x2**2)")
    assert "x1" in parsed["predictors"]
    assert "x2" in parsed["predictors"]
    assert "log" not in parsed["predictors"]
    assert "I" not in parsed["predictors"]


def test_parse_formula_invalid():
    with pytest.raises(ValueError, match="must be in the form"):
        parse_formula("x1 + x2")


def test_extract_vars_from_formula():
    result = extract_vars_from_formula("y ~ x1 + x2 + x3")
    assert result == ["y", "x1", "x2", "x3"]


@pytest.mark.parametrize(
    "ids, times, expected",
    [
        ([1, 1, 2, 2], [1, 2, 1, 2], True),
        ([1, 1, 2], [1, 2, 1], False),
    ],
)
def test_is_balanced_panel(ids, times, expected):
    df = pl.DataFrame({"id": ids, "time": times})
    assert is_balanced_panel(df, "time", "id") == expected


@pytest.mark.parametrize(
    "X, expected_shape, expected_first_col",
    [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), (2, 3), [1.0, 1.0]),
        (None, None, None),
        (np.empty((5, 0)), None, None),
    ],
)
def test_add_intercept(X, expected_shape, expected_first_col):
    result = add_intercept(X)
    if expected_shape is None:
        assert result is None
    else:
        assert result.shape == expected_shape
        np.testing.assert_array_equal(result[:, 0], expected_first_col)


def test_extract_covariates_with_formula():
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [3.0, 4.0], "x2": [5.0, 6.0]})
    result = extract_covariates(df, "~ x1 + x2")
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[:, 0], [1.0, 1.0])


@pytest.mark.parametrize(
    "formula",
    [None, "~1"],
)
def test_extract_covariates_returns_none(formula):
    assert extract_covariates(pl.DataFrame({"y": [1.0]}), formula) is None


def test_extract_covariates_missing_column():
    df = pl.DataFrame({"y": [1.0], "x1": [2.0]})
    with pytest.raises(ValueError, match="not found"):
        extract_covariates(df, "~ x1 + missing_col")


@pytest.mark.parametrize(
    "formula, expected",
    [
        (None, None),
        ("~1", None),
    ],
)
def test_get_covariate_names_from_formula_none_cases(formula, expected):
    assert get_covariate_names_from_formula(formula) is expected


def test_get_covariate_names_from_formula_with_vars():
    result = get_covariate_names_from_formula("y ~ x1 + x2")
    assert "x1" in result
    assert "x2" in result


def test_remove_collinear_no_collinearity():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    result, kept = remove_collinear(X, ["a", "b", "c"])
    assert result.shape[1] == 3
    assert kept == ["a", "b", "c"]


def test_remove_collinear_with_collinearity():
    X = np.column_stack([np.ones(50), np.ones(50) * 2, np.arange(50, dtype=float)])
    result, kept = remove_collinear(X, ["a", "b", "c"])
    assert result.shape[1] == 2
    assert len(kept) == 2


def test_remove_collinear_empty():
    X = np.empty((10, 0))
    result, kept = remove_collinear(X, [])
    assert result.shape[1] == 0
    assert kept == []


def test_check_partition_collinearity_no_collinearity():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    subgroup = np.concatenate([np.full(25, 4), np.full(25, 3), np.full(25, 2), np.full(25, 1)])
    collinear_map, collinear_list = check_partition_collinearity(X, subgroup, ["x1", "x2"])
    assert len(collinear_map) == 0
    assert len(collinear_list) == 0


def test_check_partition_collinearity_empty_vars():
    collinear_map, collinear_list = check_partition_collinearity(np.empty((10, 0)), np.ones(10), [])
    assert collinear_map == {}
    assert collinear_list == []


def test_create_ddd_subgroups():
    treat = np.array([1, 1, 0, 0])
    partition = np.array([1, 0, 1, 0])
    result = create_ddd_subgroups(treat, partition, treat_val=1)
    np.testing.assert_array_equal(result, [4, 3, 2, 1])


@pytest.mark.parametrize(
    "sizes, should_raise",
    [
        ({1: 10, 2: 20, 3: 15, 4: 12}, False),
        ({1: 10, 2: 3, 3: 15, 4: 12}, True),
    ],
)
def test_validate_subgroup_sizes(sizes, should_raise):
    if should_raise:
        with pytest.raises(ValueError, match="Subgroup 2 has only 3"):
            validate_subgroup_sizes(sizes)
    else:
        validate_subgroup_sizes(sizes)


@pytest.mark.filterwarnings("ignore:Missing values in covariates:UserWarning")
def test_extract_ddd_covariates_intercept_only():
    df = pl.DataFrame({"_post": [0, 0, 1, 1], "y": [1.0, 2.0, 3.0, 4.0]})
    cov, names = extract_ddd_covariates(df, "~1")
    assert cov.shape == (2, 0)
    assert names == []


def test_extract_ddd_covariates_with_vars():
    df = pl.DataFrame(
        {
            "_post": [0, 0, 0, 0, 1, 1, 1, 1],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    cov, names = extract_ddd_covariates(df, "~ x1 + x2")
    assert cov.shape[0] == 4
    assert len(names) >= 1


def test_extract_ddd_covariates_with_nan_raises():
    df = pl.DataFrame(
        {
            "_post": [0, 0, 1, 1],
            "x1": [1.0, float("nan"), 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError):
        extract_ddd_covariates(df, "~ x1")
