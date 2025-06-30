"""Utility functions for sensitivity analysis."""

import warnings

import numpy as np

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


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
    n_selections = len(selection)
    select_rows = select == "rows"

    return _selection_matrix_impl(selection_0idx, size, n_selections, select_rows)


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
    eta = np.asarray(eta, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    z = np.asarray(z, dtype=np.float64).flatten()

    return _compute_bounds_impl(eta, sigma, A, b, z)


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


if HAS_NUMBA:

    @nb.jit(nopython=True, cache=True)
    def _compute_bounds_impl(eta, sigma, A, b, z):
        """Compute bounds (Numba-jitted)."""
        sigma_eta = np.dot(sigma, eta)
        eta_sigma_eta = np.dot(eta, sigma_eta)
        c = sigma_eta / eta_sigma_eta

        Az = np.dot(A, z)
        Ac = np.dot(A, c)

        lower_bound = -np.inf
        upper_bound = np.inf

        for i, ac_val in enumerate(Ac):
            if abs(ac_val) > 1e-10:
                obj_val = (b[i] - Az[i]) / ac_val
                if ac_val < 0:
                    lower_bound = max(lower_bound, obj_val)
                elif obj_val < upper_bound:
                    upper_bound = obj_val
        return lower_bound, upper_bound

    @nb.jit(nopython=True, cache=True)
    def _selection_matrix_impl(selection_0idx, size, n_selections, select_rows):
        """Create selection matrix (Numba-jitted)."""
        if select_rows:
            m = np.zeros((n_selections, size))
            for i in range(n_selections):
                m[i, selection_0idx[i]] = 1.0
        else:
            m = np.zeros((size, n_selections))
            for i in range(n_selections):
                m[selection_0idx[i], i] = 1.0
        return m


else:

    def _compute_bounds_impl(eta, sigma, A, b, z):
        """Compute bounds (pure Python)."""
        c = lee_coefficient(eta, sigma)
        Az = A @ z
        Ac = A @ c

        nonzero_mask = np.abs(Ac) > 1e-10
        objective = np.full_like(Ac, np.nan)
        objective[nonzero_mask] = (b[nonzero_mask] - Az[nonzero_mask]) / Ac[nonzero_mask]

        ac_negative_idx = Ac < 0
        ac_positive_idx = Ac > 0

        lower_bound = np.max(objective[ac_negative_idx]) if np.any(ac_negative_idx) else -np.inf
        upper_bound = np.min(objective[ac_positive_idx]) if np.any(ac_positive_idx) else np.inf
        return lower_bound, upper_bound

    def _selection_matrix_impl(selection_0idx, size, n_selections, select_rows):
        """Create selection matrix (pure Python)."""
        if select_rows:
            m = np.zeros((n_selections, size))
            for i, idx in enumerate(selection_0idx):
                m[i, idx] = 1
        else:
            m = np.zeros((size, n_selections))
            for i, idx in enumerate(selection_0idx):
                m[idx, i] = 1
        return m
