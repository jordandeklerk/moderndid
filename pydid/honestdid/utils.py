"""Utility functions for sensitivity analysis."""

import warnings

import numpy as np


def selection_matrix(selection, size, select="columns"):
    """Create a selection matrix for extracting specific rows or columns.

    Constructs a matrix that can be used to select specific elements from
    a vector or specific rows/columns from a matrix through matrix
    multiplication.

    Parameters
    ----------
    selection : array-like
        Indices to select.
    size : int
        Size of the target dimension.
    select : {'columns', 'rows'}, default='columns'
        Whether to select columns or rows.

    Returns
    -------
    ndarray
        Selection matrix of appropriate dimensions.
    """
    selection = np.asarray(selection)

    selection_0idx = selection - 1

    if select == "rows":
        m = np.zeros((len(selection), size))
        for i, idx in enumerate(selection_0idx):
            m[i, idx] = 1
    else:  # columns
        m = np.zeros((size, len(selection)))
        for i, idx in enumerate(selection_0idx):
            m[idx, i] = 1

    return m


def lee_coefficient(eta, sigma):
    """Compute coefficient for constructing confidence intervals.

    Parameters
    ----------
    eta : ndarray
        Direction vector.
    sigma : ndarray
        Covariance matrix.

    Returns
    -------
    ndarray
        Coefficient vector.
    """
    eta = np.asarray(eta).flatten()
    sigma = np.asarray(sigma)

    sigma_eta = sigma @ eta
    eta_sigma_eta = eta.T @ sigma_eta

    if np.abs(eta_sigma_eta) < 1e-10:
        raise ValueError("Estimated coefficient is effectively zero, cannot compute coefficient.")

    c = sigma_eta / eta_sigma_eta
    return c


def compute_bounds(eta, sigma, A, b, z):
    """Compute lower and upper bounds for confidence intervals.

    Calculates lower and upper bounds used in constructing confidence
    intervals under shape restrictions.

    Parameters
    ----------
    eta : ndarray
        Direction vector.
    sigma : ndarray
        Covariance matrix.
    A : ndarray
        Constraint matrix.
    b : ndarray
        Constraint bounds.
    z : ndarray
        Current point.

    Returns
    -------
    tuple
        (lower_bound, upper_bound).

    Notes
    -----
    Returns (-inf, inf) when no constraints are active in the respective direction.
    """
    eta = np.asarray(eta).flatten()
    sigma = np.asarray(sigma)
    A = np.asarray(A)
    b = np.asarray(b).flatten()
    z = np.asarray(z).flatten()

    c = lee_coefficient(eta, sigma)

    # Compute objective: (b - Az) / Ac
    Az = A @ z
    Ac = A @ c

    nonzero_mask = np.abs(Ac) > 1e-10
    objective = np.full_like(Ac, np.nan)
    objective[nonzero_mask] = (b[nonzero_mask] - Az[nonzero_mask]) / Ac[nonzero_mask]

    # Find indices where Ac is negative and positive
    ac_negative_idx = Ac < 0
    ac_positive_idx = Ac > 0

    # Compute lower and upper bounds
    if np.any(ac_negative_idx):
        VLo = np.max(objective[ac_negative_idx])
    else:
        VLo = -np.inf

    if np.any(ac_positive_idx):
        VUp = np.min(objective[ac_positive_idx])
    else:
        VUp = np.inf

    return VLo, VUp


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
        warnings.warn(f"Matrix sigma not exactly symmetric (largest asymmetry was {asymmetry:.6g})", UserWarning)

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
