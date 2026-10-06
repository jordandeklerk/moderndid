"""Multivariate spline construction for nonparametric estimation."""

import warnings

import numpy as np

from ..cupy.backend import get_backend, to_numpy
from .container import MultivariateBasis
from .gsl_bspline import gsl_bs, predict_gsl_bs


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

    Constructs additive, tensor product, or generalized polynomial (glp)
    basis functions for multivariate continuous and discrete predictors.

    The additive basis stacks the univariate B-spline bases and the tensor
    basis multiplies them in every combination. The glp basis treats the
    column index within each univariate basis like a polynomial degree and
    keeps the main effects and only the interactions of low combined order.
    It therefore grows more slowly than the tensor product. It needs
    univariate bases with enough columns and raises an error when they are
    too small. Cubic or higher-degree splines always have enough columns.

    With ``deriv`` above zero, the basis holds the derivatives of the basis
    functions with respect to the variable ``deriv_index``. Columns that do
    not depend on that variable are zero. The whole basis is zero when that
    variable enters with degree 0.

    Parameters
    ----------
    x : ndarray
        Continuous predictor matrix of shape (n, p).
    K : ndarray
        Matrix of shape (p, 2). Column 0 holds the spline degree of each
        continuous variable and column 1 holds its number of segments minus one.
    z : ndarray, optional
        Discrete predictor matrix of shape (n, q).
    indicator : ndarray, optional
        Indicator vector of length q for discrete variables (1 to include).
    xeval : ndarray, optional
        Evaluation points for continuous variables. If None, uses x.
    zeval : ndarray, optional
        Evaluation points for discrete variables. If None, uses z.
    knots : {"quantiles", "uniform"}, default="quantiles"
        Knot placement, either at the quantiles of the data or uniformly
        spaced over its range.
    basis : {"additive", "tensor", "glp"}, default="additive"
        Multivariate basis construction.
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
    """
    xp = get_backend()

    if x is None or K is None:
        raise ValueError("Must provide x and K.")

    if not isinstance(K, np.ndarray) or K.ndim != 2 or K.shape[1] != 2:
        raise ValueError("K must be a two-column matrix.")

    x = xp.atleast_2d(xp.asarray(x))
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
        xeval = xp.atleast_2d(xp.asarray(xeval))
        if xeval.shape[1] != num_x:
            raise ValueError("xeval must be of the same dimension as x.")

    if z is not None and zeval is None:
        zeval = z.copy()
    elif z is not None:
        zeval = xp.atleast_2d(xp.asarray(zeval))

    gsl_intercept = basis not in ("additive", "glp")

    if np.any(K[:, 0] > 0) or (indicator is not None and np.any(indicator != 0)):
        tp = []

        for i in range(num_x):
            if K[i, 0] > 0:
                if knots == "uniform":
                    knots_vec = None
                else:
                    probs = np.linspace(0, 1, K[i, 1] + 2)
                    x_col_np = to_numpy(x[:, i])
                    knots_vec = np.quantile(x_col_np, probs)
                    knots_vec = knots_vec + np.linspace(
                        0,
                        1e-10 * (np.max(x_col_np) - np.min(x_col_np)),
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
            P = xp.hstack(tp)
            dim_P_no_tensor = P.shape[1]

            if basis == "tensor":
                P = tensor_prod_model_matrix(tp)
            elif basis == "glp":
                P = glp_model_matrix(tp)

            if deriv != 0 and K[deriv_index - 1, 0] > 0 and basis != "tensor":
                # Since tp holds one block per continuous variable with a positive degree, tp_idx is the
                # derivative variable's block. Additive and glp columns without it have zero derivative.
                tp_idx = int(np.sum(K[: deriv_index - 1, 0] > 0))
                widths = [b.shape[1] for b in tp]
                if basis == "glp":
                    keep = _glp_index_sets(widths)[:, tp_idx] != 0
                else:
                    keep = np.repeat(np.arange(len(tp)) == tp_idx, widths)
                P[:, ~keep] = 0
        else:
            P = tp[0] if tp else np.ones((xeval.shape[0], 1))
            dim_P_no_tensor = P.shape[1]

    else:
        dim_P_no_tensor = 0
        P = xp.ones((xeval.shape[0], 1))

    if deriv != 0 and K[deriv_index - 1, 0] == 0:
        # A variable that enters with degree 0 leaves the basis unchanged and has a zero derivative.
        P = xp.zeros_like(P)

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
    """
    xp = get_backend()
    if not bases:
        raise ValueError("bases cannot be empty")

    for i, basis in enumerate(bases):
        if not hasattr(basis, "ndim"):
            raise TypeError(f"bases[{i}] must be an array, got {type(basis)}")
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
    result = xp.empty((n_obs, total_cols), dtype=np.float64)

    for row in range(n_obs):
        row_vectors = [basis[row, :] for basis in bases]

        tensor_row = row_vectors[0].copy()
        for vec in row_vectors[1:]:
            tensor_row = xp.kron(tensor_row, vec)

        result[row, :] = tensor_row

    return result


def glp_model_matrix(bases):
    r"""Construct a generalized polynomial (glp) model matrix.

    Each column of the glp matrix multiplies one column from some of the
    marginal bases. The column index within a marginal basis plays the role
    of a polynomial degree. Keeping all main effects and only the
    interactions of low combined order makes the matrix more parsimonious
    and better conditioned than the tensor product while it keeps good
    approximation properties.

    The construction needs marginal bases with enough columns. With two
    bases each needs at least two columns. With more bases, small ones such
    as two bases of two columns make the construction repeat or drop
    columns. The function raises an error in those cases.

    Parameters
    ----------
    bases : list of ndarray
        List of model matrices for marginal bases. Each matrix must have
        the same number of rows (observations).

    Returns
    -------
    ndarray
        glp model matrix with one column per kept product of marginal columns.
    """
    xp = get_backend()
    if not bases:
        raise ValueError("bases cannot be empty")

    for i, basis in enumerate(bases):
        if not hasattr(basis, "ndim"):
            raise TypeError(f"bases[{i}] must be an array, got {type(basis)}")
        if basis.ndim != 2:
            raise ValueError(f"bases[{i}] must be 2-dimensional")

    n_obs = bases[0].shape[0]

    for i, basis in enumerate(bases[1:], 1):
        if basis.shape[0] != n_obs:
            raise ValueError(
                f"All matrices must have same number of rows. bases[0] has {n_obs}, bases[{i}] has {basis.shape[0]}"
            )

    if n_obs == 0:
        return xp.empty((0, 0))

    if len(bases) == 1:
        return bases[0]

    sets = _glp_index_sets([basis.shape[1] for basis in bases])
    result = xp.ones((n_obs, sets.shape[0]))
    for k, basis in enumerate(bases):
        used = sets[:, k] > 0
        result[:, used] = result[:, used] * basis[:, sets[used, k] - 1]

    return result


def _glp_index_sets(dims):
    """Column index sets of the glp basis for marginal bases with ``dims`` columns.

    Row r holds, for each marginal basis, the 1-based column that enters column r of the
    glp matrix, or 0 when that basis is left out of the product.
    """
    dims = np.asarray(dims, dtype=int)
    if np.any(dims < 1):
        raise ValueError("Each marginal basis of a glp basis needs at least one column.")

    order = np.argsort(-dims, kind="stable")
    dimen = dims[order]
    d1 = int(dimen[0])
    # nd1[s - 1] counts the index sets with index sum s that can still enter interactions.
    nd1 = np.ones(d1, dtype=int)
    nd1[d1 - 1] = 0
    sets = np.arange(1, d1 + 1).reshape(-1, 1)
    d2p = 0
    for d2 in dimen[1:]:
        sets, nd1 = _glp_add_dimension(d1, int(d2), d2p, nd1, sets)
        d2p = int(d2)
    for i in range(1, len(dims)):
        row = np.zeros(sets.shape[1], dtype=int)
        row[i - 1] = dimen[i - 1]
        sets = np.vstack([sets, row])

    sets = sets[:, np.argsort(order, kind="stable")]
    sets = sets[np.lexsort(tuple(sets[:, j] for j in range(sets.shape[1])))]

    rows = {tuple(r) for r in sets.tolist()}
    mains = {
        tuple(j if i == q else 0 for i in range(len(dims))) for q in range(len(dims)) for j in range(1, dims[q] + 1)
    }
    if len(rows) != sets.shape[0] or not mains <= rows or np.any(sets > dims):
        raise ValueError(
            f"basis='glp' cannot be built from marginal bases with {dims.tolist()} columns because the "
            "construction repeats or drops columns at these sizes. Use basis='tensor' or 'additive', "
            "or raise the spline degree or the number of segments."
        )
    return sets


def _glp_add_dimension(d1, d2, d2p, nd1, d1sets):
    """Extend the glp index sets by one marginal basis with ``d2`` columns."""
    if d2 == 1:
        return np.column_stack([d1sets, np.zeros(d1sets.shape[0], dtype=int)]), nd1

    n_main = min(d1sets.shape[0], d2)
    d2sets = np.column_stack([np.zeros((n_main, d1sets.shape[1]), dtype=int), np.resize(np.arange(1, d2 + 1), n_main)])

    # Since the previous basis's top column enters only as a main effect added at the end, it is left out here.
    candidates = d1sets[d1sets[:, -1] != d2p] if d1sets.shape[1] > 1 and d2p > 0 else d1sets
    if candidates.shape[1] > 1 and candidates.shape[0] == 1:
        raise ValueError("basis='glp' cannot be built from marginal bases of these sizes.")
    for total in range(1, d1 - d2 + 1):
        d2sets = np.vstack([d2sets, _glp_expand(candidates, total, d2, nd1[total - 1])])
    for i in range(1, d2 + 1):
        if nd1[d1 - i] > 0:
            d2sets = np.vstack([d2sets, _glp_expand(candidates, d1 - i + 1, i, nd1[d1 - i])])

    nd2 = nd1.copy()
    for j in range(1, d1):
        nd2[j - 1] = sum(nd1[i - 1] if i > 0 else 1 for i in range(j, max(0, j - d2 + 1) - 1, -1))
    nd2[d1 - 1] = nd1[d1 - 1] + sum(nd1[i - 1] for i in range(d1 - d2 + 1, d1))
    return d2sets, nd2


def _glp_expand(candidates, total, times, n_repeat):
    """Pair the sets that sum to ``total`` with the new basis's columns 0 to ``times - 1``."""
    block = candidates[candidates.sum(axis=1) == total]
    if candidates.shape[1] > 1 and block.shape[0] == 1:
        # A lone matching set is recycled across the columns so the index sets follow the standard glp construction.
        block = np.tile(block[0].reshape(-1, 1), (1, candidates.shape[1]))
    if block.shape[0] == 0:
        raise ValueError("basis='glp' cannot be built from marginal bases of these sizes.")
    stacked = block[np.tile(np.arange(block.shape[0]), times)]
    new_col = np.resize(np.repeat(np.arange(times), n_repeat), stacked.shape[0])
    return np.column_stack([stacked, new_col])
