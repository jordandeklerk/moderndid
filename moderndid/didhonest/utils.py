"""Utility functions for sensitivity analysis."""

import re
import warnings

import numpy as np


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

    if betahat.ndim > 2:
        raise ValueError(f"Expected a vector but betahat has shape {betahat.shape}")

    betahat_flat = betahat.flatten()
    beta_length = len(betahat_flat)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Expected a square matrix but sigma was {sigma.shape[0]} by {sigma.shape[1]}")

    if sigma.shape[0] != beta_length:
        raise ValueError(f"betahat ({betahat.shape}) and sigma ({sigma.shape}) were non-conformable")

    num_periods = num_pre_periods + num_post_periods
    if num_periods != beta_length:
        raise ValueError(
            f"betahat ({betahat.shape}) and pre + post periods "
            f"({num_pre_periods} + {num_post_periods}) were non-conformable"
        )

    if len(l_vec) != num_post_periods:
        raise ValueError(f"l_vec (length {len(l_vec)}) and post periods ({num_post_periods}) were non-conformable")


def bin_factor(bin_spec, values, name=""):
    """Apply binning transformation to array values.

    Groups values according to various binning specifications including
    consecutive grouping, pattern matching, and custom mappings.

    Parameters
    ----------
    bin_spec : str, list, or dict
        Binning specification:

        - ``"bin::n"``: Group every n consecutive values
        - ``"!bin::n"``: Group n values starting from first
        - ``"!!bin::n"``: Group n values starting from last
        - ``"@pattern"``: Regex pattern for matching values
        - list: Values to group together (all map to first)
        - dict: Custom mapping of new names to old values

    values : array-like
        Values to bin.
    name : str, optional
        Variable name for error messages.

    Returns
    -------
    ndarray
        Array with binned values.
    """
    values = np.asarray(values)
    unique_values = np.unique(values)

    if isinstance(bin_spec, str):
        return _handle_string_bin_spec(bin_spec, values, unique_values)

    if isinstance(bin_spec, list):
        return _handle_list_bin_spec(bin_spec, values, unique_values)

    if isinstance(bin_spec, dict):
        return _handle_dict_bin_spec(bin_spec, values, unique_values)

    return bin_factor(list(bin_spec), values, name)


def create_interactions(
    factor_var, var=None, ref=None, keep=None, ref2=None, keep2=None, bin=None, bin2=None, name=None, return_dict=False
):
    """Create dummy variables from factors with optional interactions.

    Parameters
    ----------
    factor_var : array-like
        Primary factor variable to create dummies from.
    var : array-like, optional
        Variable to interact with factor_var. Can be numeric or categorical.
    ref : scalar, list, or True, optional
        Reference levels to exclude. If True, drops first level.
    keep : list, optional
        Levels to keep (all others dropped).
    ref2 : scalar or list, optional
        Reference levels for var (if categorical).
    keep2 : list, optional
        Levels to keep from var (if categorical).
    bin : str, list, or dict, optional
        Binning specification for factor_var.
    bin2 : str, list, or dict, optional
        Binning specification for var.
    name : str, optional
        Name prefix for generated variables.
    return_dict : bool, optional
        If True, return dictionary with matrix and metadata.

    Returns
    -------
    ndarray or dict
        Matrix of dummy variables. If return_dict=True, returns dict with
        'matrix', 'names', and 'reference_info' keys.
    """
    factor_var = np.asarray(factor_var)
    n = len(factor_var)

    if bin is not None:
        factor_var = bin_factor(bin, factor_var, name or "factor")

    is_interaction = var is not None
    is_numeric_interaction = False
    is_factor_interaction = False

    if is_interaction:
        var = np.asarray(var)
        if len(var) != len(factor_var):
            raise ValueError("factor_var and var must have the same length")

        if bin2 is not None:
            var = bin_factor(bin2, var, "var")

        if np.issubdtype(var.dtype, np.number):
            is_numeric_interaction = True
        else:
            is_factor_interaction = True

    factor_unique = np.unique(factor_var)

    if ref is not None:
        if ref is True:
            ref = [factor_unique[0]]
        elif not isinstance(ref, list):
            ref = [ref]
        factor_unique = factor_unique[~np.isin(factor_unique, ref)]

    if keep is not None:
        if not isinstance(keep, list):
            keep = [keep]
        factor_unique = factor_unique[np.isin(factor_unique, keep)]

    if is_factor_interaction:
        var_unique = np.unique(var)

        if ref2 is not None:
            if not isinstance(ref2, list):
                ref2 = [ref2]
            var_unique = var_unique[~np.isin(var_unique, ref2)]

        if keep2 is not None:
            if not isinstance(keep2, list):
                keep2 = [keep2]
            var_unique = var_unique[np.isin(var_unique, keep2)]

        interactions = []
        col_names = []

        for f_val in factor_unique:
            for v_val in var_unique:
                mask = (factor_var == f_val) & (var == v_val)
                if np.any(mask):
                    col = np.zeros(n)
                    col[mask] = 1
                    interactions.append(col)

                    if name is not None:
                        col_names.append(f"{name}::{f_val}:var::{v_val}")
                    else:
                        col_names.append(f"{f_val}:{v_val}")

        if not interactions:
            matrix = np.zeros((n, 0))
        else:
            matrix = np.column_stack(interactions)

    elif is_numeric_interaction:
        k = len(factor_unique)
        matrix = np.zeros((n, k))
        col_names = []

        for j, f_val in enumerate(factor_unique):
            mask = factor_var == f_val
            matrix[mask, j] = var[mask]

            if name is not None:
                col_names.append(f"{name}::{f_val}:var")
            else:
                col_names.append(f"{f_val}:var")

    else:
        k = len(factor_unique)
        matrix = np.zeros((n, k))
        col_names = []

        for j, f_val in enumerate(factor_unique):
            matrix[factor_var == f_val, j] = 1

            if name is not None:
                col_names.append(f"{name}::{f_val}")
            else:
                col_names.append(str(f_val))

    if return_dict:
        return {
            "matrix": matrix,
            "names": col_names,
            "reference_info": {
                "ref": ref,
                "ref2": ref2 if is_factor_interaction else None,
                "is_interaction": is_interaction,
                "is_numeric": is_numeric_interaction,
                "is_factor": is_factor_interaction,
            },
        }

    return matrix


def _handle_string_bin_spec(bin_spec, values, unique_values):
    """Handle string-based binning specifications."""
    if bin_spec.startswith("@"):
        return _apply_regex_binning(bin_spec[1:], values, unique_values)

    if "bin::" in bin_spec:
        return _apply_consecutive_binning(bin_spec, values, unique_values)

    raise ValueError(f"Unknown string bin specification: {bin_spec}")


def _apply_regex_binning(pattern, values, unique_values):
    """Apply regex pattern matching to group values."""
    regex = re.compile(pattern)

    matches = [val for val in unique_values if regex.search(str(val))]

    if not matches:
        raise ValueError(f"No values matched regex pattern '{pattern}'")

    if len(matches) > 1:
        mapping = {val: matches[0] for val in matches[1:]}
        result = values.copy()
        for old_val, new_val in mapping.items():
            result[values == old_val] = new_val
        return result

    return values


def _apply_consecutive_binning(bin_spec, values, unique_values):
    """Apply consecutive binning based on specification."""
    match = re.match(r"^(!!?)?bin::(\d+)$", bin_spec)
    if not match:
        raise ValueError(f"Invalid bin specification: {bin_spec}")

    prefix, bin_size = match.groups()
    bin_size = int(bin_size)

    if prefix is None and np.issubdtype(values.dtype, np.number):
        return (values // bin_size) * bin_size

    groups = _create_consecutive_groups(unique_values, bin_size, prefix)

    return _apply_grouping(values, groups)


def _create_consecutive_groups(unique_values, bin_size, prefix):
    """Create groups of consecutive values based on prefix."""
    n_values = len(unique_values)
    groups = []

    if prefix == "!!":
        for i in range(n_values - 1, -1, -bin_size):
            group_start = max(0, i - bin_size + 1)
            groups.append(unique_values[group_start : i + 1])
    elif prefix == "!":
        for i in range(0, n_values, bin_size):
            groups.append(unique_values[i : i + bin_size])
    else:
        for i in range(0, n_values, bin_size):
            groups.append(unique_values[i : i + bin_size])

    return groups


def _apply_grouping(values, groups):
    """Apply grouping by mapping values to group representatives."""
    result = values.copy()

    for group in groups:
        if len(group) > 1:
            group_representative = group[0]
            for val in group[1:]:
                result[values == val] = group_representative

    return result


def _handle_list_bin_spec(bin_spec, values, unique_values):
    """Handle list-based binning specifications."""
    if not bin_spec:
        return values

    existing_values = [val for val in bin_spec if val in unique_values]

    if not existing_values:
        return values

    result = values.copy()
    first_value = existing_values[0]

    for val in existing_values[1:]:
        result[values == val] = first_value

    return result


def _handle_dict_bin_spec(bin_spec, values, unique_values):
    """Handle dictionary-based binning specifications."""
    result = values.copy()

    for new_name, old_values in bin_spec.items():
        if isinstance(new_name, str) and new_name.startswith("@"):
            pattern = re.compile(new_name[1:])
            old_values = [val for val in unique_values if pattern.search(str(val))]
        else:
            if not isinstance(old_values, list):
                old_values = [old_values]

        for val in old_values:
            if val in unique_values:
                result[values == val] = new_name

    return result
