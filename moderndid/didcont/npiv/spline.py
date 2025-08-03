# pylint: disable=too-many-nested-blocks
"""Multivariate spline construction for continuous treatment DiD estimation."""

import warnings
from typing import NamedTuple

import numpy as np

from .gsl_bspline import gsl_bs, predict_gsl_bs


class MultivariateBasis(NamedTuple):
    """Result from multivariate spline basis construction."""

    basis: np.ndarray
    dim_no_tensor: int
    degree_matrix: np.ndarray
    n_segments: np.ndarray
    basis_type: str


def prodspline(
    x,
    K,
    z=None,
    indicator=None,
    xeval=None,
    zeval=None,
    knots="quantiles",
    basis="additive",
    x_min=None,
    x_max=None,
    deriv_index=1,
    deriv=0,
):
    r"""Create multivariate spline basis with B-spline components.

    Constructs additive, tensor product, or generalized linear product (GLP)
    basis functions for multivariate continuous and discrete predictors.

    Parameters
    ----------
    x : ndarray
        Continuous predictor matrix of shape (n, p).
    K : ndarray
        Matrix of shape (p, 2) containing spline specifications:

        - Column 0: degree for each continuous variable
        - Column 1: number of segments - 1 for each variable
    z : ndarray, optional
        Discrete predictor matrix of shape (n, q).
    indicator : ndarray, optional
        Indicator vector of length q for discrete variables (1 to include).
    xeval : ndarray, optional
        Evaluation points for continuous variables. If None, uses x.
    zeval : ndarray, optional
        Evaluation points for discrete variables. If None, uses z.
    knots : {"quantiles", "uniform"}, default="quantiles"
        Method for knot placement:
        - "quantiles": Knots at data quantiles
        - "uniform": Uniformly spaced knots
    basis : {"additive", "tensor", "glp"}, default="additive"
        Type of basis construction:

        - "additive": Sum of univariate bases
        - "tensor": Full tensor product of all bases
        - "glp": Generalized linear product (hierarchical interactions)
    x_min : ndarray, optional
        Minimum values for each continuous variable.
    x_max : ndarray, optional
        Maximum values for each continuous variable.
    deriv_index : int, default=1
        Index (1-based) of variable for derivative computation.
    deriv : int, default=0
        Order of derivative to compute.

    Returns
    -------
    MultivariateBasis
        NamedTuple containing:

        - basis: Complete basis matrix
        - dim_no_tensor: Number of columns before tensor product
        - degree_matrix: Copy of K matrix
        - n_segments: Number of segments for each variable
        - basis_type: Type of basis used

    References
    ----------

    .. [1] Wood, S. N. (2017). Generalized Additive Models: An Introduction
        with R. Chapman and Hall/CRC.
    """
    if x is None or K is None:
        raise ValueError("Must provide x and K.")

    if not isinstance(K, np.ndarray) or K.ndim != 2 or K.shape[1] != 2:
        raise ValueError("K must be a two-column matrix.")

    x = np.atleast_2d(x)
    K = np.round(K).astype(int)

    num_x = x.shape[1]
    num_K = K.shape[0]

    if num_K != num_x:
        raise ValueError(f"Dimension of x and K incompatible ({num_x}, {num_K}).")

    if deriv < 0:
        raise ValueError("deriv is invalid.")
    if deriv_index < 1 or deriv_index > num_x:
        raise ValueError("deriv_index is invalid.")
    if deriv > K[deriv_index - 1, 0]:
        warnings.warn("deriv order too large, result will be zero.", UserWarning)

    num_z = 0
    if z is not None:
        z = np.atleast_2d(z)
        num_z = z.shape[1]
        if indicator is None:
            raise ValueError("Must provide indicator when z is specified.")
        indicator = np.asarray(indicator)
        num_indicator = len(indicator)
        if num_indicator != num_z:
            raise ValueError(f"Dimension of z and indicator incompatible ({num_z}, {num_indicator}).")

    if xeval is None:
        xeval = x.copy()
    else:
        xeval = np.atleast_2d(xeval)
        if xeval.shape[1] != num_x:
            raise ValueError("xeval must be of the same dimension as x.")

    if z is not None and zeval is None:
        zeval = z.copy()
    elif z is not None:
        zeval = np.atleast_2d(zeval)

    gsl_intercept = basis not in ("additive", "glp")

    if np.any(K[:, 0] > 0) or (indicator is not None and np.any(indicator != 0)):
        tp = []

        for i in range(num_x):
            if K[i, 0] > 0:
                if knots == "uniform":
                    knots_vec = None
                else:
                    probs = np.linspace(0, 1, K[i, 1] + 1)
                    knots_vec = np.quantile(x[:, i], probs)
                    knots_vec = knots_vec + np.linspace(
                        0,
                        1e-10 * (np.max(x[:, i]) - np.min(x[:, i])),
                        len(knots_vec),
                    )

                if i == deriv_index - 1 and deriv != 0:
                    basis_obj = gsl_bs(
                        x=x[:, i],
                        degree=K[i, 0],
                        nbreak=K[i, 1] + 1,
                        knots=knots_vec,
                        deriv=deriv,
                        x_min=x_min[i] if x_min is not None else None,
                        x_max=x_max[i] if x_max is not None else None,
                        intercept=gsl_intercept,
                    )
                else:
                    basis_obj = gsl_bs(
                        x=x[:, i],
                        degree=K[i, 0],
                        nbreak=K[i, 1] + 1,
                        knots=knots_vec,
                        x_min=x_min[i] if x_min is not None else None,
                        x_max=x_max[i] if x_max is not None else None,
                        intercept=gsl_intercept,
                    )

                tp.append(predict_gsl_bs(basis_obj, xeval[:, i]))

        if z is not None:
            for i in range(num_z):
                if indicator[i] == 1:
                    if zeval is None:
                        unique_vals = np.unique(z[:, i])
                        if len(unique_vals) > 1:
                            dummies = np.column_stack([(z[:, i] == val).astype(float) for val in unique_vals[1:]])
                            tp.append(dummies)
                    else:
                        unique_vals = np.unique(z[:, i])
                        if len(unique_vals) > 1:
                            dummies = np.column_stack([(zeval[:, i] == val).astype(float) for val in unique_vals[1:]])
                            tp.append(dummies)

        if len(tp) > 1:
            P = np.hstack(tp)
            dim_P_no_tensor = P.shape[1]

            if basis == "tensor":
                P = tensor_prod_model_matrix(tp)
            elif basis == "glp":
                P = glp_model_matrix(tp)
                if deriv != 0:
                    p_deriv_list = [np.zeros((1, b.shape[1])) for b in tp]

                    # Find the index in `tp` that corresponds to the derivative variable.
                    # `deriv_index` is 1-based for `x`. `tp` only contains bases for
                    # variables with `K[i,0] > 0` or `indicator[i] == 1`. Derivatives are
                    # only for continuous variables, so we only care about `K`.
                    tp_idx = -1
                    spline_count = 0
                    if deriv_index > 0:
                        for i in range(deriv_index - 1):
                            if K[i, 0] > 0:
                                spline_count += 1
                        if K[deriv_index - 1, 0] > 0:
                            tp_idx = spline_count

                    if tp_idx != -1 and tp_idx < len(p_deriv_list):
                        p_deriv_list[tp_idx] = np.full((1, tp[tp_idx].shape[1]), np.nan)

                        mask_basis = glp_model_matrix(p_deriv_list)

                        mask = np.isnan(mask_basis.flatten())

                        P[:, ~mask] = 0
        else:
            P = tp[0] if tp else np.ones((xeval.shape[0], 1))
            dim_P_no_tensor = P.shape[1]

    else:
        dim_P_no_tensor = 0
        P = np.ones((xeval.shape[0], 1))

    return MultivariateBasis(
        basis=P,
        dim_no_tensor=dim_P_no_tensor,
        degree_matrix=K.copy(),
        n_segments=K[:, 1] + 1 if K.size > 0 else np.array([]),
        basis_type=basis,
    )


def tensor_prod_model_matrix(bases: list[np.ndarray]) -> np.ndarray:
    """Construct tensor product of basis matrices.

    Parameters
    ----------
    bases : list of ndarray
        List of basis matrices for each variable.

    Returns
    -------
    ndarray
        Full tensor product basis matrix.
    """
    if not bases:
        return np.ones((1, 1))

    n = bases[0].shape[0]

    dims = [b.shape[1] for b in bases]
    total_cols = np.prod(dims)

    result = np.ones((n, total_cols))

    for row in range(n):
        rows = [b[row, :] for b in bases]

        tensor_row = rows[0]
        for r in rows[1:]:
            tensor_row = np.kron(tensor_row, r)

        result[row, :] = tensor_row

    return result


def glp_model_matrix(bases: list[np.ndarray]) -> np.ndarray:
    """Construct generalized linear product (GLP) basis matrix.

    Parameters
    ----------
    bases : list of ndarray
        List of basis matrices for each variable.

    Returns
    -------
    ndarray
        GLP basis matrix with hierarchical structure.
    """
    if not bases:
        return np.array([[]])

    n = bases[0].shape[0]

    P = np.hstack(bases)

    num_bases = len(bases)
    if num_bases > 1:
        for i in range(num_bases):
            for j in range(i + 1, num_bases):
                interaction = bases[i][:, :, np.newaxis] * bases[j][:, np.newaxis, :]
                interaction = interaction.reshape(n, -1)
                P = np.hstack([P, interaction])

    return P
