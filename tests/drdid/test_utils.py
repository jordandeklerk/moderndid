"""Tests for the drdid numerical helpers."""

import numpy as np
import pytest

from moderndid.drdid.utils import _scaled_inverse


def test_scaled_inverse_runs_on_an_array_module_without_a_condition_number(array_module_without_cond):
    x = np.column_stack([np.ones(50), np.random.default_rng(3).normal(size=(50, 2))])
    gram = x.T @ x
    collinear = np.column_stack([x, 2 * x[:, 1]])

    inverse = _scaled_inverse(gram, "unused")

    np.testing.assert_allclose(inverse, np.linalg.inv(gram), rtol=1e-12)
    with pytest.raises(ValueError, match="^The design matrix is singular\\.$"):
        _scaled_inverse(collinear.T @ collinear, "The design matrix is singular.")
