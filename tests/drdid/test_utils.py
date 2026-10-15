"""Tests for the drdid numerical helpers."""

import numpy as np
import pytest

from moderndid.cupy.backend import to_numpy
from moderndid.drdid.utils import _scaled_inverse


def test_scaled_inverse_inverts_a_gram_matrix_and_rejects_one_with_a_zero_column(xp):
    x = np.column_stack([np.ones(50), np.random.default_rng(3).normal(size=(50, 2))])
    gram = x.T @ x
    padded = np.column_stack([x, np.zeros(50)])

    inverse = _scaled_inverse(xp.asarray(gram), "unused")

    assert isinstance(inverse, xp.ndarray)
    np.testing.assert_allclose(to_numpy(inverse), np.linalg.inv(gram), rtol=1e-10)
    with pytest.raises(ValueError, match="^The design matrix is singular\\.$"):
        _scaled_inverse(xp.asarray(padded.T @ padded), "The design matrix is singular.")
