# pylint: disable=too-many-nested-blocks
"""Multivariate spline construction for continuous treatment DiD estimation."""

import warnings
from itertools import combinations, product
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

        - **basis**: Complete basis matrix
        - **dim_no_tensor**: Number of columns before tensor product
        - **degree_matrix**: Copy of K matrix
        - **n_segments**: Number of segments for each variable
        - **basis_type**: Type of basis used

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
                    probs = np.linspace(0, 1, K[i, 1] + 2)
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
                        nbreak=K[i, 1] + 2,
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
                        nbreak=K[i, 1] + 2,
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


def tensor_prod_model_matrix(bases):
    r"""Construct tensor product of marginal basis model matrices.

    Produces model matrices for tensor product smooths from marginal basis
    model matrices. The tensor product is computed row-wise using Kronecker
    products.

    Parameters
    ----------
    bases : list of ndarray
        List of model matrices for marginal bases. Each matrix must have
        the same number of rows (observations).

    Returns
    -------
    ndarray
        Tensor product model matrix of shape (n, prod(dims)) where n is the
        number of observations and dims are the dimensions of input matrices.

    References
    ----------

    .. [1] Wood, S. N. (2006). Low-rank scale-invariant tensor product smooths
        for generalized additive mixed models. Biometrics, 62(4), 1025-1036.
    """
    if not bases:
        raise ValueError("bases cannot be empty")

    for i, basis in enumerate(bases):
        if not isinstance(basis, np.ndarray):
            raise TypeError(f"bases[{i}] must be a NumPy array")
        if basis.ndim != 2:
            raise ValueError(f"bases[{i}] must be 2-dimensional")

    n_obs = bases[0].shape[0]
    for i, basis in enumerate(bases[1:], 1):
        if basis.shape[0] != n_obs:
            raise ValueError(
                f"All matrices must have same number of rows. bases[0] has {n_obs}, bases[{i}] has {basis.shape[0]}"
            )

    dims = [basis.shape[1] for basis in bases]
    total_cols = int(np.prod(dims))
    result = np.empty((n_obs, total_cols), dtype=np.float64)

    for row in range(n_obs):
        row_vectors = [basis[row, :] for basis in bases]

        tensor_row = row_vectors[0].copy()
        for vec in row_vectors[1:]:
            tensor_row = np.kron(tensor_row, vec)

        result[row, :] = tensor_row

    return result


def glp_model_matrix(bases):
    r"""Construct generalized linear product (GLP) model matrix.

    Produces model matrices for generalized polynomial smooths from marginal
    basis model matrices. The GLP creates a hierarchical polynomial structure
    where terms of different orders can be included, providing a more
    parsimonious alternative to full tensor products while retaining good
    approximation capabilities.

    Parameters
    ----------
    bases : list of ndarray
        List of model matrices for marginal bases. Each matrix must have
        the same number of rows (observations).

    Returns
    -------
    ndarray
        GLP model matrix with hierarchical polynomial structure.

    References
    ----------

    .. [1] Hall, P., & Racine, J. S. (2015). Infinite order cross-validated
        local polynomial regression. Journal of Econometrics, 185(2), 510-525.
    """
    if not bases:
        raise ValueError("bases cannot be empty")

    for i, basis in enumerate(bases):
        if not isinstance(basis, np.ndarray):
            raise TypeError(f"bases[{i}] must be a NumPy array")
        if basis.ndim != 2:
            raise ValueError(f"bases[{i}] must be 2-dimensional")

    n_obs = bases[0].shape[0]

    for i, basis in enumerate(bases[1:], 1):
        if basis.shape[0] != n_obs:
            raise ValueError(
                f"All matrices must have same number of rows. bases[0] has {n_obs}, bases[{i}] has {basis.shape[0]}"
            )

    if n_obs == 0:
        return np.empty((0, 0))

    num_bases = len(bases)
    result_matrices = []

    for basis in bases:
        result_matrices.append(basis)

    for i, j in combinations(range(num_bases), 2):
        interaction = _compute_basis_interaction([bases[i], bases[j]])
        result_matrices.append(interaction)

    for order in range(3, num_bases + 1):
        for indices in combinations(range(num_bases), order):
            selected_bases = [bases[idx] for idx in indices]
            interaction = _compute_basis_interaction(selected_bases)
            result_matrices.append(interaction)

    if result_matrices:
        return np.hstack(result_matrices)
    return np.ones((n_obs, 1))


def _compute_basis_interaction(bases):
    """Compute interaction terms between basis functions.

    Parameters
    ----------
    bases : list of ndarray
        List of basis matrices to interact.

    Returns
    -------
    ndarray
        Matrix of interaction terms.
    """
    if len(bases) == 1:
        return bases[0]

    n_obs = bases[0].shape[0]

    dims = [basis.shape[1] for basis in bases]
    total_interactions = int(np.prod(dims))

    result = np.empty((n_obs, total_interactions))

    col_idx = 0

    for indices in product(*[range(dim) for dim in dims]):
        interaction_col = np.ones(n_obs)
        for basis_idx, func_idx in enumerate(indices):
            interaction_col *= bases[basis_idx][:, func_idx]

        result[:, col_idx] = interaction_col
        col_idx += 1

    return result
