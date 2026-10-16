"""Minimax weighting solver for the doubly robust ML DiD estimator."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def amle_weights(
    covariates,
    post_indicator,
    cohort_indicator,
    *,
    zeta=0.5,
    sample_weights=None,
    solver=None,
    verbose=False,
):
    r"""Solve the augmented minimax-linear balancing weights for one group-time cell.

    Computes the weight vector :math:`\hat{\gamma} \in \mathbb{R}^n` that
    enters the doubly robust ATT estimator as the residual correction term
    :math:`\hat{\gamma}_i (Y_i - \hat{y}_i)`.

    The weights solve the Augmented Minimax Linear Estimation (AMLE)
    program of [1]_ in the version that [2]_ adapts to the doubly robust
    DiD setting, with the exact balance of the intercepts that [3]_ states.

    The weights sum to one within the treated cell :math:`(T = G = 1)` and
    to zero within the whole sample, the post period, and the treated
    cohort. Covariate moments and their interactions with :math:`T` and
    :math:`G` are balanced up to slack variables. The covariate moments of
    the treated cell are balanced against that cell's own covariate mean.
    That mean uses ``sample_weights`` when given. The weights therefore
    target the average conditional effect over the treated cell.

    Parameters
    ----------
    covariates : ndarray of shape (n, p)
        Covariate design matrix without the intercept column.
    post_indicator : ndarray of shape (n,)
        Binary 0/1 indicator equal to 1 for post-treatment observations.
    cohort_indicator : ndarray of shape (n,)
        Binary 0/1 indicator equal to 1 for units in the treated cohort
        :math:`g`.
    zeta : float, default=0.5
        Mixing weight in :math:`(0, 1)`. Higher values prioritize tighter
        moment balance; lower values prioritize smaller raw weights.
    sample_weights : ndarray of shape (n,), optional
        Non-negative sampling weight of each observation. Defaults to equal
        weights.
    solver : str, optional
        Name of the cvxpy solver to use. Defaults to ``"CLARABEL"``.
    verbose : bool, default=False
        Whether to print solver progress information.

    Returns
    -------
    ndarray of shape (n,)
        The minimax weight vector :math:`n \cdot g_{1:n}^\star`.

    Notes
    -----
    The decision vector
    :math:`g = (g_1, \ldots, g_n, g_{n+1}, \ldots, g_{n+4})` has length
    :math:`n + 4`. The first :math:`n` entries are the per-observation
    balancing weights and the final four entries are slack variables
    bounding the worst-case imbalance in each balance condition. The
    program is

    .. math::

        \min_{g} \; (1 - \zeta) \sum_{i=1}^n g_i^2
        + \zeta \sum_{k=1}^{4} g_{n+k}^2

    subject to

    .. math::

        \begin{aligned}
            \sum_{i=1}^n g_i = \sum_{i=1}^n T_i g_i
            = \sum_{i=1}^n G_i g_i &= 0,\\
            \big|X^\top g_{1:n}\big|_j &\le g_{n+1},\\
            \big|X^\top (T \odot g_{1:n})\big|_j &\le g_{n+2},\\
            \big|X^\top (G \odot g_{1:n})\big|_j &\le g_{n+3},\\
            \sum_{i=1}^n T_i G_i g_i &= 1,\\
            \big|X^\top (T \odot G \odot g_{1:n}) - \bar{X}_{1}\big|_j
            &\le g_{n+4},
        \end{aligned}

    where :math:`T \in \{0, 1\}^n` is the post-period indicator,
    :math:`G \in \{0, 1\}^n` is the treated-cohort indicator, and
    :math:`\bar{X}_{1}` is the column mean of ``covariates`` over the rows
    with :math:`T_i G_i = 1`, weighted by ``sample_weights`` when given. The
    objective
    trades off raw weight magnitude (small :math:`\zeta`) against tighter
    moment balance (large :math:`\zeta`). The returned weights are scaled
    to :math:`\hat{\gamma} = n \cdot g_{1:n}^\star`.

    References
    ----------

    .. [1] Hirshberg, D. A., & Wager, S. (2021). "Augmented minimax linear
           estimation." The Annals of Statistics, 49(6), 3206-3227.
           https://doi.org/10.1214/21-AOS2080

    .. [2] Nie, X., Lu, C., & Wager, S. (2024). "Nonparametric heterogeneous
           treatment effect estimation in repeated cross sectional designs."
           In E. Laber, B. Chakraborty, E. E. M. Moodie, T. Cai, & M. van der
           Laan (Eds.), Handbook of Statistical Methods for Precision Medicine
           (Ch. 9). Chapman and Hall/CRC.
           https://doi.org/10.1201/9781003216223-9

    .. [3] Lu, C., Nie, X., & Wager, S. (2019). "Robust nonparametric
           difference-in-differences estimation." arXiv:1905.11622v1.
           https://arxiv.org/abs/1905.11622v1
    """
    if not 0 < zeta < 1:
        raise ValueError(f"zeta must lie in the open interval (0, 1), got {zeta!r}.")

    covariates = np.asarray(covariates, dtype=float)
    post_indicator = np.asarray(post_indicator, dtype=float)
    cohort_indicator = np.asarray(cohort_indicator, dtype=float)

    if covariates.ndim != 2:
        raise ValueError(f"covariates must be 2-D, got shape {covariates.shape}.")

    n, _ = covariates.shape
    if post_indicator.shape != (n,):
        raise ValueError(f"post_indicator length {post_indicator.shape} does not match covariates row count {n}.")
    if cohort_indicator.shape != (n,):
        raise ValueError(f"cohort_indicator length {cohort_indicator.shape} does not match covariates row count {n}.")
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape != (n,):
            raise ValueError(f"sample_weights length {sample_weights.shape} does not match covariates row count {n}.")
        if np.any(sample_weights < 0) or not np.sum(sample_weights) > 0:
            raise ValueError("sample_weights must be non-negative with a positive sum.")
    if np.sum(post_indicator * cohort_indicator) <= 0:
        raise ValueError(
            "No observations with post_indicator == 1 and cohort_indicator == 1; "
            "the equality constraint sum(T*G*g) = 1 cannot be satisfied."
        )

    treated_cell = post_indicator * cohort_indicator
    row_weights = np.ones(n) if sample_weights is None else sample_weights
    treated_means = np.average(covariates, axis=0, weights=treated_cell * row_weights)

    g = cp.Variable(n + 4)
    g_main = g[:n]
    s1, s2, s3, s4 = g[n], g[n + 1], g[n + 2], g[n + 3]

    objective = cp.Minimize((1 - zeta) * cp.sum_squares(g_main) + zeta * cp.sum_squares(g[n:]))

    main_balance = covariates.T @ g_main
    post_balance = covariates.T @ cp.multiply(post_indicator, g_main)
    cohort_balance = covariates.T @ cp.multiply(cohort_indicator, g_main)
    interaction_balance = covariates.T @ cp.multiply(treated_cell, g_main)

    # Since cvxpy infers the shape of cp.sum by summing an uninitialized array that can overflow and warn, the
    # sums below are inner products.
    constraints = [
        np.ones(n) @ g_main == 0,
        post_indicator @ g_main == 0,
        cohort_indicator @ g_main == 0,
        main_balance <= s1,
        -main_balance <= s1,
        post_balance <= s2,
        -post_balance <= s2,
        cohort_balance <= s3,
        -cohort_balance <= s3,
        treated_cell @ g_main == 1,
        interaction_balance - treated_means <= s4,
        -(interaction_balance - treated_means) <= s4,
    ]

    problem = cp.Problem(objective, constraints)
    chosen_solver = solver or cp.CLARABEL

    try:
        problem.solve(solver=chosen_solver, verbose=verbose)
    except cp.error.SolverError as exc:
        raise RuntimeError(f"Minimax solver {chosen_solver!r} failed: {exc}") from exc

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Minimax problem did not reach an optimal solution (status={problem.status!r}).")

    solution = g.value

    if solution is None:
        raise RuntimeError("Solver returned a null solution.")

    return n * np.asarray(solution[:n], dtype=float)
