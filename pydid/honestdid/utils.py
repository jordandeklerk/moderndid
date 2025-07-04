"""Utility functions for sensitivity analysis."""

import warnings

import numpy as np

from .numba import compute_bounds, selection_matrix

__all__ = [
    "basis_vector",
    "validate_symmetric_psd",
    "validate_conformable",
    "selection_matrix",
    "compute_bounds",
]


def basis_vector(index=1, size=1):
    """Create a standard basis vector.

    Parameters
    ----------
    index : int, default=1
        Position for the 1 value.
    size : int, default=1
        Length of the vector.

    Returns
    -------
    ndarray
        Column vector with shape (size, 1).
    """
    if index < 1 or index > size:
        raise ValueError(f"index must be between 1 and {size}, got {index}")

    v = np.zeros((size, 1))
    v[index - 1] = 1
    return v


def validate_symmetric_psd(sigma):
    """Check if a matrix is symmetric and positive semi-definite.

    Issues warnings if the matrix is not exactly symmetric or not
    numerically positive semi-definite.

    Parameters
    ----------
    sigma : ndarray
        Matrix to validate.

    Warnings
    --------
    UserWarning
        If the matrix is not symmetric or not positive semi-definite.

    Notes
    -----
    This function only issues warnings and does not raise exceptions.
    """
    sigma = np.asarray(sigma)

    asymmetry = np.max(np.abs(sigma - sigma.T))
    if asymmetry > 1e-10:
        warnings.warn(
            f"Matrix sigma not exactly symmetric (largest asymmetry was {asymmetry:.6g})",
            UserWarning,
        )

    eigenvalues = np.linalg.eigvals(sigma)
    min_eigenvalue = np.min(eigenvalues.real)

    if min_eigenvalue < -1e-10:
        warnings.warn(
            f"Matrix sigma not numerically positive semi-definite (smallest eigenvalue was {min_eigenvalue:.6g})",
            UserWarning,
        )


def validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec):
    """Validate dimensions of inputs for sensitivity analysis.

    Parameters
    ----------
    betahat : ndarray
        Estimated coefficients vector.
    sigma : ndarray
        Covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : array-like
        Weight vector for post-treatment periods.

    Raises
    ------
    ValueError
        If any dimensions are incompatible.
    """
    betahat = np.asarray(betahat)
    sigma = np.asarray(sigma)
    l_vec = np.asarray(l_vec)

    # Check betahat is a vector
    if betahat.ndim > 2:
        raise ValueError(f"Expected a vector but betahat has shape {betahat.shape}")

    betahat_flat = betahat.flatten()
    beta_length = len(betahat_flat)

    # Check sigma is square
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Expected a square matrix but sigma was {sigma.shape[0]} by {sigma.shape[1]}")

    # Check betahat and sigma are conformable
    if sigma.shape[0] != beta_length:
        raise ValueError(f"betahat ({betahat.shape}) and sigma ({sigma.shape}) were non-conformable")

    # Check periods match betahat length
    num_periods = num_pre_periods + num_post_periods
    if num_periods != beta_length:
        raise ValueError(
            f"betahat ({betahat.shape}) and pre + post periods "
            f"({num_pre_periods} + {num_post_periods}) were non-conformable"
        )

    # Check l_vec length
    if len(l_vec) != num_post_periods:
        raise ValueError(f"l_vec (length {len(l_vec)}) and post periods ({num_post_periods}) were non-conformable")
