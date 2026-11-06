"""Tests for the drdid numerical helpers."""

import re
import warnings

import numpy as np
import pytest

from moderndid.cupy.backend import to_numpy
from moderndid.drdid.utils import (
    _check_wls_condition_number,
    _independent_columns,
    _propensity_columns,
    _scaled_columns,
    _scaled_inverse,
    _scaled_pseudo_inverse,
)


def test_scaled_inverse_inverts_a_gram_matrix_and_rejects_one_with_a_zero_column(xp):
    x = np.column_stack([np.ones(50), np.random.default_rng(3).normal(size=(50, 2))])
    gram = x.T @ x
    padded = np.column_stack([x, np.zeros(50)])

    inverse = _scaled_inverse(xp.asarray(gram), "unused")

    assert isinstance(inverse, xp.ndarray)
    np.testing.assert_allclose(to_numpy(inverse), np.linalg.inv(gram), rtol=1e-10)
    with pytest.raises(ValueError, match="^The design matrix is singular\\.$"):
        _scaled_inverse(xp.asarray(padded.T @ padded), "The design matrix is singular.")


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("full_rank", None),
        ("duplicate", [0, 1, 3]),
        ("intercept", [0, 1, 3]),
        ("zero", [0, 1, 3]),
        ("twice", [0, 1, 3]),
        ("affine", [0, 1, 2]),
        ("large_affine", [0, 1, 2]),
        ("two_redundant", [0, 1, 3]),
    ],
)
def test_independent_columns_keep_each_column_that_raises_the_rank_of_the_earlier_ones(
    propensity_designs, case, expected
):
    design, weights = propensity_designs[case]
    factor = np.linalg.qr(np.sqrt(weights)[:, None] * design, mode="r")

    kept = _independent_columns(factor, design.shape[0])

    assert kept == expected


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("full_rank", None),
        ("duplicate", [0, 1, 3]),
        ("intercept", [0, 1, 3]),
        ("zero", [0, 1, 3]),
        ("twice", [0, 1, 3]),
        ("affine", [0, 1, 2]),
        ("large_affine", [0, 1, 2]),
        ("two_redundant", [0, 1, 3]),
    ],
)
def test_propensity_columns_keep_each_covariate_that_raises_the_weighted_rank(xp, propensity_designs, case, expected):
    design, weights = propensity_designs[case]

    kept = _propensity_columns(xp.asarray(design), xp.asarray(weights))

    assert kept == expected


def test_propensity_columns_drop_a_column_that_is_zero_on_every_row_with_positive_weight(xp, propensity_designs):
    design, weights = propensity_designs["full_rank"]
    padded = np.column_stack([design, np.where(weights > 1.4, 5.0, 0.0)])
    zeroed = np.where(weights > 1.4, 0.0, weights)

    kept = _propensity_columns(xp.asarray(padded), xp.asarray(zeroed))
    weighted = _propensity_columns(xp.asarray(padded), xp.asarray(weights))

    assert kept == [0, 1, 2]
    assert weighted is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_column_helpers_keep_every_column_of_covariates_that_are_not_finite(xp, propensity_designs, value):
    design, weights = propensity_designs["duplicate"]
    covariates = design.copy()
    covariates[5, 3] = value

    kept = _propensity_columns(xp.asarray(covariates), xp.asarray(weights))
    factor_kept = _independent_columns(np.linalg.qr(covariates, mode="r"), covariates.shape[0])
    finite_kept = _propensity_columns(xp.asarray(design), xp.asarray(weights))

    assert kept is None
    assert factor_kept is None
    assert finite_kept == [0, 1, 3]


@pytest.mark.parametrize(
    ("covariates", "weights"),
    [
        (np.empty((0, 3)), np.empty(0)),
        (np.empty((10, 0)), np.ones(10)),
        (np.arange(10.0), np.ones(10)),
        (np.column_stack([np.ones(10), np.arange(10.0), np.arange(10.0)]), np.zeros(10)),
    ],
    ids=["no_rows", "no_columns", "one_dimension", "no_weight"],
)
def test_propensity_columns_keep_every_column_of_a_design_without_rows_columns_or_weight(covariates, weights):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        kept = _propensity_columns(covariates, weights)

    assert kept is None
    assert record == []


@pytest.mark.parametrize(("n_rows", "expected"), [(3, None), (4, [0, 1, 2])])
def test_column_helpers_keep_every_column_when_the_kept_columns_would_fill_the_rows(n_rows, expected):
    columns = np.random.default_rng(n_rows).standard_normal((n_rows, 2))
    design = np.column_stack([np.ones(n_rows), columns, columns[:, 0]])

    kept = _independent_columns(np.linalg.qr(design, mode="r"), n_rows)
    screened = _propensity_columns(design, np.ones(n_rows))

    assert kept == expected
    assert screened == expected


def test_propensity_columns_agree_with_the_factor_rule_on_designs_near_the_gram_threshold(gram_screen_designs):
    factors = [np.linalg.qr(np.sqrt(w)[:, None] * _scaled_columns(x), mode="r") for x, w in gram_screen_designs]

    kept = [_propensity_columns(x, w) for x, w in gram_screen_designs]
    expected = [_independent_columns(factor, x.shape[0]) for factor, (x, _) in zip(factors, gram_screen_designs)]

    assert kept == expected
    assert None in expected
    assert any(columns is not None for columns in expected)


def test_scaled_pseudo_inverse_returns_a_generalized_inverse_of_a_singular_gram_matrix(xp):
    z = np.random.default_rng(4).normal(size=(50, 2))
    x = np.column_stack([np.ones(50), z, z[:, 0] + z[:, 1]])
    gram = x.T @ x

    inverse = _scaled_pseudo_inverse(xp.asarray(gram))

    assert isinstance(inverse, xp.ndarray)
    np.testing.assert_allclose(gram @ to_numpy(inverse) @ gram, gram, rtol=1e-10, atol=0)


@pytest.mark.parametrize(
    ("condition_number", "messages"),
    [
        (1e9, []),
        (
            1e11,
            [
                "Potential multicollinearity detected (condition number: 1.00e+11). Consider removing or combining "
                "covariates."
            ],
        ),
    ],
)
def test_check_wls_condition_number_warns_only_above_the_warning_threshold(condition_number, messages):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_wls_condition_number(condition_number)

    assert [str(w.message) for w in caught] == messages


def test_check_wls_condition_number_rejects_a_design_above_the_error_threshold():
    message = (
        "Failed to solve linear system: The covariate matrix may be singular or ill-conditioned "
        "(condition number: 1.00e+16)."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _check_wls_condition_number(1e16)
