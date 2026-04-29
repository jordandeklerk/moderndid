"""Minimax weighting solver for the doubly-robust ML DiD estimator."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_minimax_weights(
    covariates,
    post_indicator,
    cohort_indicator,
    *,
    zeta=0.5,
    solver=None,
    verbose=False,
):
    r"""Solve the augmented minimax-linear balancing weights for one group-time cell.

    Computes the weight vector :math:`\hat{\gamma} \in \mathbb{R}^n` that
    enters the doubly-robust ATT estimator

    .. math::

        \widehat{ATT}(g, t) = \frac{1}{n} \sum_{i=1}^{n}
        \left[\hat{\tau}(X_i)
        + \hat{\gamma}_i \, \bigl(Y_i - \hat{y}(X_i, G_i, T_i)\bigr)\right],

    where :math:`\hat{\tau}` is the conditional ATT and :math:`\hat{y}` is
    the orthogonal-decomposition prediction for :math:`Y`. The weights solve
    the Augmented Minimax Linear Estimation (AMLE) program of [1]_, adapted
    to the doubly-robust DiD setting by [2]_, over a length ``n + 4``
    decision vector
    :math:`g = (g_1, \ldots, g_n, g_{n+1}, \ldots, g_{n+4})`:

    .. math::

        \min_{g} \; (1 - \zeta) \sum_{i=1}^n g_i^2
        + \zeta \sum_{k=1}^{4} g_{n+k}^2

    subject to

    .. math::

        \begin{aligned}
            \sum_{i=1}^n g_i &= 0,\\
            \big|X^\top g_{1:n}\big|_j &\le g_{n+1},\\
            \big|X^\top (T \odot g_{1:n})\big|_j &\le g_{n+2},\\
            \big|X^\top (G \odot g_{1:n})\big|_j &\le g_{n+3},\\
            \sum_{i=1}^n T_i G_i g_i &= 1,\\
            \big|X^\top (T \odot G \odot g_{1:n}) - \bar{X}\big|_j
            &\le g_{n+4},
        \end{aligned}

    where :math:`T \in \{0, 1\}^n` is the post-period indicator,
    :math:`G \in \{0, 1\}^n` is the treated-cohort indicator, and
    :math:`\bar{X}` is the column mean of ``covariates``. The four slack
    variables :math:`g_{n+1}, \ldots, g_{n+4}` represent worst-case
    imbalance levels in each balance condition; the objective trades off
    raw weight magnitude (small :math:`\zeta`) against tighter moment
    balance (large :math:`\zeta`). The returned weights are scaled to
    :math:`\hat{\gamma} = n \cdot g_{1:n}^\star`.

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
    solver : str, optional
        Name of the cvxpy solver to use. Defaults to ``"ECOS"``.
    verbose : bool, default=False
        Whether to print solver progress information.

    Returns
    -------
    ndarray of shape (n,)
        The minimax weight vector :math:`n \cdot g[:n]^\star`.

    Raises
    ------
    ValueError
        If ``zeta`` is outside :math:`(0, 1)`, the inputs are inconsistent,
        or no units satisfy ``post_indicator == 1`` and
        ``cohort_indicator == 1``.
    RuntimeError
        If the convex program fails to reach an optimal solution.

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
    if np.sum(post_indicator * cohort_indicator) <= 0:
        raise ValueError(
            "No observations with post_indicator == 1 and cohort_indicator == 1; "
            "the equality constraint sum(T*G*g) = 1 cannot be satisfied."
        )

    column_means = covariates.mean(axis=0)

    g = cp.Variable(n + 4)
    g_main = g[:n]
    s1, s2, s3, s4 = g[n], g[n + 1], g[n + 2], g[n + 3]

    objective = cp.Minimize((1 - zeta) * cp.sum_squares(g_main) + zeta * cp.sum_squares(g[n:]))

    main_balance = covariates.T @ g_main
    post_balance = covariates.T @ cp.multiply(post_indicator, g_main)
    cohort_balance = covariates.T @ cp.multiply(cohort_indicator, g_main)
    interaction = cp.multiply(post_indicator * cohort_indicator, g_main)
    interaction_balance = covariates.T @ interaction

    constraints = [
        cp.sum(g_main) == 0,
        main_balance <= s1,
        -main_balance <= s1,
        post_balance <= s2,
        -post_balance <= s2,
        cohort_balance <= s3,
        -cohort_balance <= s3,
        cp.sum(interaction) == 1,
        interaction_balance - column_means <= s4,
        -(interaction_balance - column_means) <= s4,
    ]

    problem = cp.Problem(objective, constraints)
    chosen_solver = solver or "ECOS"
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
