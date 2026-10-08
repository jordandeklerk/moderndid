"""Tests for preprocessing utility functions."""

import re

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
    get_column_terms,
    get_covariate_names_from_formula,
    get_first_difference,
    get_formula_columns,
    get_group,
    get_transformed_terms,
    is_balanced_panel,
    make_balanced_panel,
    map_to_idx,
    nonfinite_to_null,
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


@pytest.mark.parametrize("name", ["_group", "_ever", "_is_treated", "_treat_cumsum", "_first_treat"])
@pytest.mark.parametrize(
    "treat_period, expected",
    [
        (None, [3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0]),
        (2, [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]),
    ],
    ids=["first_switch", "treat_period"],
)
def test_get_group_keeps_user_column_named_like_a_helper(staggered_panel, name, treat_period, expected):
    data = staggered_panel.with_columns(pl.lit(9).alias(name))
    result = get_group(data, "id", "time", "treat", treat_period=treat_period)
    assert result.columns == [*data.columns, "G"]
    assert result[name].to_list() == [9] * 12
    assert result["G"].to_list() == expected


@pytest.mark.parametrize(
    "treat_period, expected",
    [
        (None, [3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0]),
        (2, [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]),
    ],
    ids=["first_switch", "treat_period"],
)
def test_get_group_replaces_existing_G_column(staggered_panel, treat_period, expected):
    data = staggered_panel.with_columns(pl.lit(9).alias("G")).select("id", "G", "time", "y", "treat")
    result = get_group(data, "id", "time", "treat", treat_period=treat_period)
    assert result.columns == ["id", "G", "time", "y", "treat"]
    assert result["G"].to_list() == expected


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), float("-inf")])
def test_get_group_skips_rows_with_missing_period(missing):
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2],
            "time": [missing, 1.0, 2.0, 3.0, missing, 1.0, 2.0],
            "treat": [1, 0, 1, 1, 1, 0, 0],
        }
    )
    result = get_group(data, "id", "time", "treat")
    assert result["G"].to_list() == [2, 2, 2, 2, 0, 0, 0]


@pytest.mark.parametrize(
    "treat_period, by_id",
    [
        (None, {1: 3, 2: 2, 3: 0}),
        (2, {1: 2, 2: 2, 3: 0}),
    ],
    ids=["first_switch", "treat_period"],
)
def test_get_group_keeps_row_order(staggered_panel, treat_period, by_id):
    shuffled = staggered_panel.sample(fraction=1.0, shuffle=True, seed=1)
    result = get_group(shuffled, "id", "time", "treat", treat_period=treat_period)
    assert result.drop("G").equals(shuffled)
    assert result["G"].to_list() == [by_id[unit] for unit in shuffled["id"]]


@pytest.mark.parametrize("treat_period", [None, 2], ids=["first_switch", "treat_period"])
def test_get_group_assigns_no_group_to_rows_without_unit_id(treat_period):
    data = pl.DataFrame(
        {
            "id": [1, 1, None, None, 2, 2],
            "time": [1, 2, 1, 2, 1, 2],
            "treat": [0, 1, 0, 1, 0, 0],
        }
    )
    result = get_group(data, "id", "time", "treat", treat_period=treat_period)
    assert result.filter(pl.col("id") == 1)["G"].to_list() == [2, 2]
    assert result.filter(pl.col("id") == 2)["G"].to_list() == [0, 0]
    assert result.filter(pl.col("id").is_null())["G"].fill_null(0).to_list() == [0, 0]


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


@pytest.mark.parametrize(
    "kwargs, expected_ids",
    [
        ({"g": 3, "tp": 3, "control_group": "notyettreated", "anticipation": 1}, [0, 1, 4, 5, 6, 7]),
        ({"g": 4, "tp": 2, "control_group": "notyettreated", "base_period": "universal"}, [2, 3, 4, 5, 6, 7]),
        ({"g": 3, "tp": 3, "control_group": "notyettreated"}, [0, 1, 2, 3, 4, 5, 6, 7]),
        ({"g": 3, "tp": 3, "control_group": "nevertreated"}, [0, 1, 4, 5, 6, 7]),
    ],
)
def test_two_by_two_subset_keeps_controls_untreated_in_both_periods(two_by_two_panel, kwargs, expected_ids):
    result = two_by_two_subset(two_by_two_panel, **kwargs)

    assert sorted(result["gt_data"]["id"].unique().to_list()) == expected_ids
    assert result["n1"] == len(expected_ids)
    np.testing.assert_array_equal(result["disidx"], np.isin(np.arange(8), expected_ids))


def test_two_by_two_subset_accepts_pandas(two_by_two_panel):
    result = two_by_two_subset(two_by_two_panel.to_pandas(), g=3, tp=3, anticipation=1)

    assert sorted(result["gt_data"]["id"].unique().to_list()) == [0, 1, 4, 5, 6, 7]


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
        ("~ x1 + x2", "", ["x1", "x2"]),
        ("~ log.pop + x2", "", ["log.pop", "x2"]),
        ("~ lag1.Value1 + .G + _x", "", ["lag1.Value1", ".G", "_x"]),
        ("~ `log pop` + `lag1.Value1`", "", ["log pop", "lag1.Value1"]),
        ("~ `a+b` + `c~d` + `I(x)`", "", ["a+b", "c~d", "I(x)"]),
        ("`my y` ~ x1", "my y", ["x1"]),
        ("~ 1 + x1 + 1", "", ["x1"]),
        ("~ x1 + x2 + x1 + `x2`", "", ["x1", "x2"]),
        ("~1", "", []),
        ("~ 1", "", []),
    ],
)
def test_parse_formula_basic(formula, expected_outcome, expected_predictors):
    parsed = parse_formula(formula)
    assert parsed["outcome"] == expected_outcome
    assert parsed["predictors"] == expected_predictors


@pytest.mark.parametrize(
    "formula, term",
    [
        ("y ~ x1 + log(x2)", "log(x2)"),
        ("~ x1 + I(x1**2)", "I(x1**2)"),
        ("~ I(x1 + x2)", "I(x1 + x2)"),
        ("~ np.log(x1)", "np.log(x1)"),
        ("~ C(group)", "C(group)"),
        ("~ x1:x2", "x1:x2"),
        ("~ x1*x2", "x1*x2"),
        ("~ x1 - x2", "x1 - x2"),
        ("~ I(x1 - 1)", "I(x1 - 1)"),
        ("~ my col", "my col"),
        ("~ 2x", "2x"),
    ],
)
def test_parse_formula_rejects_terms_that_are_not_columns(formula, term):
    with pytest.raises(ValueError, match=re.escape(f"xformla term '{term}' is not a column name")):
        parse_formula(formula)


@pytest.mark.parametrize(
    "formula, term",
    [
        ("~ 0 + x1", "0"),
        ("~ -1 + x1", "-1"),
        ("~ x1 - 1", "x1 - 1"),
        ("~ x1 + x2-1", "x2-1"),
    ],
)
def test_parse_formula_rejects_intercept_removal(formula, term):
    with pytest.raises(ValueError, match=re.escape(f"xformla term '{term}' drops the intercept")):
        parse_formula(formula)


@pytest.mark.parametrize("formula", ["~ x1 +", "~ x1 + + x2", "~ + x1"])
def test_parse_formula_rejects_empty_terms(formula):
    with pytest.raises(ValueError, match="xformla has an empty term"):
        parse_formula(formula)


@pytest.mark.parametrize("formula", ["x1 + x2", "y ~ x1 ~ x2"])
def test_parse_formula_invalid(formula):
    with pytest.raises(ValueError, match="must be in the form"):
        parse_formula(formula)


def test_extract_vars_from_formula():
    result = extract_vars_from_formula("~ x1 + x2 + x3")
    assert result == ["x1", "x2", "x3"]


@pytest.mark.parametrize("formula", ["y ~ x1 + x2 + x3", "lpop ~ 1", "`my y` ~ x1"])
def test_extract_vars_from_formula_rejects_left_hand_side(formula):
    with pytest.raises(ValueError, match=re.escape(f"xformla='{formula}' has a left-hand side")):
        extract_vars_from_formula(formula)


def test_extract_vars_from_formula_keeps_dotted_names():
    assert extract_vars_from_formula("~ lag1.Value1 + `log pop`") == ["lag1.Value1", "log pop"]


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("~ age + I(age**2) + educ", ["age", "educ"]),
        ("~ np.log(age) + C(group, Treatment('a'))", ["age", "group"]),
        ("~ age:educ + center(age)", ["age", "educ"]),
        ("~ age.yrs + I(`age yrs`**2)", ["age.yrs", "age yrs"]),
        ("~ I(age.clip(0)) + educ", ["age", "educ"]),
        ("~ I(x1 * 1e5)", ["x1"]),
        ("~ missing + educ", ["educ"]),
        ("~ poly(age, degree=2)", ["age"]),
        ("~ bs(age, df = 3) + degree", ["age", "degree"]),
        ("~ I(age == 2) + I(educ>=1)", ["age", "educ"]),
    ],
)
def test_get_formula_columns(formula, expected):
    columns = ["age", "educ", "group", "a", "age.yrs", "age yrs", "x1", "e5", "degree", "df", "d"]
    assert get_formula_columns(formula, columns) == expected


@pytest.mark.parametrize("formula", ["age + educ", "y ~ age ~ educ"])
def test_get_formula_columns_invalid(formula):
    with pytest.raises(ValueError, match="must be in the form"):
        get_formula_columns(formula, ["y", "age", "educ"])


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("~ x1 + x2", []),
        ("~ 1", []),
        ("~1", []),
        ("~ log.pop + `log pop` + 1", []),
        ("~ x1 + I(x1**2)", ["I(x1**2)"]),
        ("~ C(group) + x1 + x1:x2", ["C(group)", "x1:x2"]),
        ("~ np.log(x1 + 1)", ["np.log(x1 + 1)"]),
        ("~ x1 - x2", ["x1 - x2"]),
    ],
)
def test_get_transformed_terms(formula, expected):
    assert get_transformed_terms(formula) == expected


@pytest.mark.parametrize(
    "formula, message",
    [
        ("y ~ I(x1**2)", "has a left-hand side"),
        ("x1 ~ 1", "has a left-hand side"),
        ("~ I(x1**2) - 1", "drops the intercept"),
        ("~ 0 + x1", "drops the intercept"),
        ("~ I(x1**2) + + x1", "xformla has an empty term"),
        ("x1 + I(x1**2)", "must be in the form"),
    ],
)
def test_get_transformed_terms_rejects_invalid_formulas(formula, message):
    with pytest.raises(ValueError, match=message):
        get_transformed_terms(formula)


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("~1", []),
        ("~ x1 + x2 + x1", ["x1", "x2"]),
        ("~ x1 + I(x2**2) + C(g)", ["x1"]),
        ("~ `log pop` + 1 + x.2", ["log pop", "x.2"]),
        ("y ~ x1", ["x1"]),
    ],
)
def test_get_column_terms(formula, expected):
    assert get_column_terms(formula) == expected


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
    assert get_covariate_names_from_formula("~ x1 + x2") == ["x1", "x2"]


def test_get_covariate_names_from_formula_rejects_left_hand_side():
    with pytest.raises(ValueError, match="has a left-hand side"):
        get_covariate_names_from_formula("y ~ x1 + x2")


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


@pytest.mark.filterwarnings("ignore:Missing values in covariates:UserWarning")
def test_extract_ddd_covariates_with_nan_raises():
    df = pl.DataFrame(
        {
            "_post": [0, 0, 1, 1],
            "x1": [1.0, float("nan"), 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError):
        extract_ddd_covariates(df, "~ x1")


def test_nonfinite_to_null_turns_nan_and_infinity_into_null():
    df = pl.DataFrame(
        {
            "y": [1.0, float("nan"), float("inf")],
            "x": pl.Series([float("nan"), 2.0, -float("inf")], dtype=pl.Float32),
            "g": [0, 1, 2],
        }
    )
    result = nonfinite_to_null(df)

    assert result["y"].to_list() == [1.0, None, None]
    assert result["x"].to_list() == [None, 2.0, None]
    assert result["g"].to_list() == [0, 1, 2]
    assert result.schema == df.schema
    assert result.equals(nonfinite_to_null(df.to_pandas()))


def test_nonfinite_to_null_keeps_infinity_in_named_columns():
    df = pl.DataFrame({"y": [float("inf"), 2.0, 3.0], "g": [float("inf"), float("nan"), -float("inf")]})
    result = nonfinite_to_null(df, keep_infinite=["g"])

    assert result["y"].to_list() == [None, 2.0, 3.0]
    assert result["g"].to_list() == [float("inf"), None, -float("inf")]


def test_make_balanced_panel_drops_units_with_two_rows_in_a_period(panel_with_duplicates):
    complete = pl.DataFrame({"id": [3, 3, 3], "time": [1, 2, 3], "y": [30.0, 31.0, 32.0], "cat": ["e", "e", "e"]})

    result = make_balanced_panel(pl.concat([panel_with_duplicates, complete]), "id", "time")

    assert result.equals(complete)
