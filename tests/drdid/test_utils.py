"""Tests for the drdid numerical helpers."""

import re
import warnings

import numpy as np
import pytest

from moderndid.cupy.backend import to_numpy
from moderndid.drdid.utils import _check_wls_condition_number, _scaled_inverse, _scaled_pseudo_inverse


def test_scaled_inverse_inverts_a_gram_matrix_and_rejects_one_with_a_zero_column(xp):
    x = np.column_stack([np.ones(50), np.random.default_rng(3).normal(size=(50, 2))])
    gram = x.T @ x
    padded = np.column_stack([x, np.zeros(50)])

    inverse = _scaled_inverse(xp.asarray(gram), "unused")

    assert isinstance(inverse, xp.ndarray)
    np.testing.assert_allclose(to_numpy(inverse), np.linalg.inv(gram), rtol=1e-10)
    with pytest.raises(ValueError, match="^The design matrix is singular\\.$"):
        _scaled_inverse(xp.asarray(padded.T @ padded), "The design matrix is singular.")


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
