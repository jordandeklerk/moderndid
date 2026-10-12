"""Data-driven dimension selection for nonparametric instrumental variables."""

from .lepski import npiv_j, npiv_jhat_max


def npiv_choose_j(
    y,
    x,
    w,
    x_grid=None,
    j_x_degree=3,
    k_w_degree=4,
    k_w_smooth=2,
    knots="uniform",
    basis="tensor",
    x_min=None,
    x_max=None,
    w_min=None,
    w_max=None,
    grid_num=50,
    biters=99,
    check_is_fullrank=False,
    seed=None,
):
    r"""Select optimal B-spline dimensions.

    Implements the data-driven choice of sieve dimension in [1]_. The selected
    dimension :math:`\tilde{J}` adapts to the unknown smoothness of :math:`h_0` and
    the strength of the instruments. With it, :math:`\hat{h}_{\tilde{J}}` converges
    to :math:`h_0` at the minimax rate in sup-norm.

    The selection runs in two steps. :func:`npiv_jhat_max` first sets the grid of
    candidate dimensions from the sample size and the estimated strength of the
    instruments. :func:`npiv_j` then keeps the smallest dimension whose fit no
    larger dimension rejects in a bootstrap Lepski test. It returns the smaller of
    that dimension and the second-largest grid dimension.

    Two details differ from [1]_. The grid starts at one segment and has no lower
    cutoff. The truncation at the second-largest grid dimension also applies when
    ``w`` equals ``x``. The regression version of the procedure in [1]_ keeps the
    Lepski choice instead.

    When ``w`` equals ``x``, the instrument basis takes the degree and segments of
    the basis for :math:`X` at every dimension. The function then ignores
    ``k_w_degree`` and ``k_w_smooth``. Unless the range limits on ``w`` differ from
    those on ``x``, the fits are least squares regressions on the basis for
    :math:`X`.

    Parameters
    ----------
    y : ndarray
        Dependent variable vector.
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    x_grid : ndarray, optional
        Points over which the fits are compared. If None, ``grid_num`` equally
        spaced values between the smallest and largest value of each regressor.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`, at least 1.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`.
    k_w_smooth : int, default=2
        Number of dyadic refinements of the :math:`W` basis relative to the
        :math:`X` basis at each grid point.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate :math:`X`, either the full tensor
        product of the univariate bases, their sum, or the generalized
        polynomial basis described in :func:`prodspline`.
    x_min, x_max, w_min, w_max : float, optional
        Range limits for basis construction.
    grid_num : int, default=50
        Number of grid points for evaluation.
    biters : int, default=99
        Number of bootstrap replications for confidence bands.
    check_is_fullrank : bool, default=False
        Whether to check if basis matrices have full rank.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - **j_hat_max**: Largest grid dimension :math:`\hat{J}_{\max}`
        - **j_hat_n**: Truncation point, the dimension of the second-largest grid entry
        - **j_hat**: Dimension chosen by the Lepski test before truncation
        - **j_tilde**: Selected dimension, the smaller of ``j_hat`` and ``j_hat_n``
        - **j_x_seg**: Number of segments of the :math:`X` basis at ``j_tilde``
        - **k_w_seg**: Number of segments of the :math:`W` basis paired with ``j_x_seg``
        - **j_x_segments_set**: Grid of segments for :math:`X`
        - **k_w_segments_set**: Grid of segments for :math:`W`
        - **theta_star**: Bootstrap critical value of the Lepski test

    Examples
    --------
    Select the sieve dimension adaptively for the Engel curve IV problem.
    The procedure determines the largest grid dimension and then applies
    the Lepski test to choose the number of B-spline segments:

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import load_engel
           ...: from moderndid.npiv import npiv_choose_j
           ...:
           ...: df = load_engel()
           ...: y = df["food"].to_numpy()
           ...: x = df["logexp"].to_numpy().reshape(-1, 1)
           ...: w = df["logwages"].to_numpy().reshape(-1, 1)
           ...: sel = npiv_choose_j(y=y, x=x, w=w, biters=50, seed=42)
           ...: print(f"Selected segments: {sel['j_x_seg']}")
           ...: print(f"Selected dimension (J_tilde): {sel['j_tilde']}")
           ...: print(f"Max feasible dimension: {sel['j_hat_max']}")
           ...: print(f"Bootstrap critical value: {sel['theta_star']:.3f}")

    See Also
    --------
    npiv_jhat_max : Compute maximum feasible dimension
    npiv_j : Lepski-style test for dimension selection

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024). Adaptive estimation and
        uniform confidence bands for nonparametric structural functions and elasticities.
        *Review of Economic Studies*, 92(1), 162-196. https://doi.org/10.1093/restud/rdae025.
    """
    tmp1 = npiv_jhat_max(
        x=x,
        w=w,
        j_x_degree=j_x_degree,
        k_w_degree=k_w_degree,
        k_w_smooth=k_w_smooth,
        knots=knots,
        basis=basis,
        x_min=x_min,
        x_max=x_max,
        w_min=w_min,
        w_max=w_max,
    )

    j_x_segments_set = tmp1["j_x_segments_set"]
    k_w_segments_set = tmp1["k_w_segments_set"]
    alpha_hat = tmp1["alpha_hat"]

    tmp2 = npiv_j(
        y=y,
        x=x,
        w=w,
        x_grid=x_grid,
        j_x_degree=j_x_degree,
        k_w_degree=k_w_degree,
        j_x_segments_set=j_x_segments_set,
        k_w_segments_set=k_w_segments_set,
        knots=knots,
        basis=basis,
        x_min=x_min,
        x_max=x_max,
        w_min=w_min,
        w_max=w_max,
        grid_num=grid_num,
        biters=biters,
        alpha=alpha_hat,
        check_is_fullrank=check_is_fullrank,
        seed=seed,
    )

    return {
        "j_hat_max": tmp1["j_hat_max"],
        "j_hat_n": tmp2["j_hat_n"],
        "j_hat": tmp2["j_hat"],
        "j_tilde": tmp2["j_tilde"],
        "j_x_seg": tmp2["j_x_seg"],
        "k_w_seg": tmp2["k_w_seg"],
        "j_x_segments_set": j_x_segments_set,
        "k_w_segments_set": k_w_segments_set,
        "theta_star": tmp2["theta_star"],
    }
