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
    boot_num=99,
    check_is_fullrank=False,
    seed=None,
):
    r"""Select optimal B-spline dimensions.

    Implements the full data-driven selection procedure from [1]_, combining a maximum
    dimension selection step with a Lepski-style test. This procedure selects a
    data-driven sieve dimension, :math:`\tilde{J}`, that is sup-norm rate-adaptive. This means
    it adapts to unknown features of the data-generating process (e.g., smoothness of
    :math:`h_0`, instrument strength) to achieve the minimax optimal convergence rate in sup-norm,

    .. math::
        \sup_x |\hat{h}_{\tilde{J}}(x) - h_0(x)|.

    The procedure involves two main steps. First, determine a maximum feasible dimension, :math:`\hat{J}_{\max}`,
    based on the sample size and an estimate of the sieve measure of ill-posedness, :math:`\hat{s}_J^{-1}`.
    This step defines the search grid :math:`\hat{\mathcal{J}}` as

    .. math::
        \hat{J}_{\max} = \min \left\{ J \in \mathcal{T} : J \sqrt{\log J} \hat{s}_J^{-1} \leq c \sqrt{n} \right\}.

    Second, use a Lepski-style method with a multiplier bootstrap to select the optimal dimension
    :math:`\hat{J}` from the grid :math:`\hat{\mathcal{J}}`. This is done by comparing estimates across different
    dimensions and selecting the smallest dimension that is not statistically different from estimates
    at larger dimensions. The final choice is :math:`\tilde{J} = \min\{\hat{J}, \hat{J}_n\}`,
    where :math:`\hat{J}_n` is a slightly smaller, more conservative dimension for stability.

    .. math::
        \hat{J} = \min \left\{ J \in \hat{\mathcal{J}} : \sup_{x, J_2 > J} \left|
        \frac{\hat{h}_J(x) - \hat{h}_{J_2}(x)}{\hat{\sigma}_{J, J_2}(x)} \right|
        \leq \theta_{1-\hat{\alpha}}^* \right\}.

    Parameters
    ----------
    y : ndarray
        Dependent variable vector.
    x : ndarray
        Endogenous regressor matrix.
    w : ndarray
        Instrument matrix.
    x_grid : ndarray, optional
        Grid points for evaluation. If None, created automatically.
    j_x_degree : int, default=3
        Degree of B-spline basis for :math:`X`.
    k_w_degree : int, default=4
        Degree of B-spline basis for :math:`W`.
    k_w_smooth : int, default=2
        Smoothness parameter for :math:`K` selection.
    knots : {"uniform", "quantiles"}, default="uniform"
        Knot placement method.
    basis : {"tensor", "additive", "glp"}, default="tensor"
        Type of basis for multivariate :math:`X`:

        - "tensor": Full tensor product of univariate bases
        - "additive": Sum of univariate bases
        - "glp": Generalized linear product (hierarchical)
    x_min, x_max, w_min, w_max : float, optional
        Range limits for basis construction.
    grid_num : int, default=50
        Number of grid points for evaluation.
    boot_num : int, default=99
        Number of bootstrap replications for confidence bands.
    check_is_fullrank : bool, default=False
        Whether to check if basis matrices have full rank.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - **j_x_segments**: Selected number of segments for :math:`X`
        - **k_w_segments**: Corresponding segments for :math:`W`
        - **j_tilde**: Selected dimension
        - **theta_star**: Bootstrap critical value
        - **j_hat_max**: Maximum feasible dimension
        - Additional diagnostic information

    See Also
    --------
    npiv_jhat_max : Compute maximum feasible dimension
    npiv_j : Lepski-style test for dimension selection

    References
    ----------

    .. [1] Chen, X., Christensen, T. M., & Kankanala, S. (2024).
        Adaptive Estimation and Uniform Confidence Bands for Nonparametric
        Structural Functions and Elasticities. https://arxiv.org/abs/2107.11869.
    """
    # Step 1: Compute maximum feasible J and data-driven grid
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

    # Step 2: Run Lepski test with bootstrap
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
        boot_num=boot_num,
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
