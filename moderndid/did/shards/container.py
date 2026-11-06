"""Containers for shards and the statistics they contribute to each pass."""

from typing import NamedTuple

import numpy as np


class Shard(NamedTuple):
    """Container for the panel data of a block of whole units.

    Every unit carries all of its periods in the order of the fit's sorted time
    periods.

    Attributes
    ----------
    unit_keys : ndarray
        Canonical 64-bit key of each unit's identifier, of shape (m,).
    cohort : ndarray
        Period in which each unit is first treated or infinity for a
        never-treated unit, of shape (m,).
    y : ndarray
        Outcome of each unit in each period, of shape (m, T).
    weights : ndarray, optional
        Normalized sampling weight of each unit in each period, of shape (m, T),
        or None for an unweighted fit.
    covariates : ndarray, optional
        Covariates of each unit in each period without the intercept, of shape
        (m, T, p), or None without covariates.
    cluster_keys : ndarray, optional
        Canonical 64-bit key of each unit's cluster, of shape (m,), or None
        without a cluster variable besides the unit identifier.
    """

    #: Canonical key of each unit's identifier.
    unit_keys: np.ndarray
    #: Period in which each unit is first treated.
    cohort: np.ndarray
    #: Outcome of each unit in each period.
    y: np.ndarray
    #: Normalized sampling weight of each unit in each period.
    weights: np.ndarray | None = None
    #: Covariates of each unit in each period.
    covariates: np.ndarray | None = None
    #: Canonical key of each unit's cluster.
    cluster_keys: np.ndarray | None = None


class RowShard(NamedTuple):
    """Container for the m rows of a block of u whole units.

    Each row of repeated cross-sections is its own unit. A shard of an
    unbalanced panel holds the rows of its units in the order of
    ``unit_keys``. The rows of each unit stay together in period order.

    Attributes
    ----------
    period : ndarray
        Index of each row's period among the sorted periods, of shape (m,).
    cohort : ndarray
        Period in which each row's unit is first treated or infinity for a
        never-treated unit, of shape (m,).
    y : ndarray
        Outcome of each row, of shape (m,).
    weights : ndarray, optional
        Normalized sampling weight of each row, of shape (m,), or None for an
        unweighted fit.
    covariates : ndarray, optional
        Covariates of each row without the intercept, of shape (m, p), or None
        without covariates.
    unit_keys : ndarray, optional
        Canonical key of each unit's identifier or the
        :func:`~moderndid.did.shards.layout.row_keys` of each row of repeated
        cross-sections, of shape (u,), or None for rows without copy indices.
    units : ndarray, optional
        Position of each row's unit among the shard's units, of shape (m,), or
        None when every row is its own unit.
    cluster_keys : ndarray, optional
        Canonical 64-bit key of each unit's cluster, of shape (u,), or None
        without a cluster variable besides the unit identifier.
    """

    #: Index of each row's period.
    period: np.ndarray
    #: Period in which each row's unit is first treated.
    cohort: np.ndarray
    #: Outcome of each row.
    y: np.ndarray
    #: Normalized sampling weight of each row.
    weights: np.ndarray | None = None
    #: Covariates of each row.
    covariates: np.ndarray | None = None
    #: 64-bit key of each unit.
    unit_keys: np.ndarray | None = None
    #: Position of each row's unit among the shard's units.
    units: np.ndarray | None = None
    #: Canonical key of each unit's cluster.
    cluster_keys: np.ndarray | None = None


class Partial(NamedTuple):
    """Container for the statistics that shards contribute to one pass.

    Each field maps a name to statistics that
    :func:`~moderndid.did.shards.partials.combine_partials` merges across
    shards by the field's own rule.

    Attributes
    ----------
    sums : dict
        Arrays that combine by addition.
    maxima : dict
        Arrays that combine by their elementwise maximum.
    factors : dict
        Pairs :math:`(R, Q'B)` from the QR factorization :math:`A = QR` of a
        design :math:`A` with right-hand sides :math:`B`.
    keyed : dict
        Pairs of an array of keys, such as cluster keys, and a matrix with one
        row of sums for each key.
    """

    #: Arrays that combine by addition.
    sums: dict
    #: Arrays that combine by their elementwise maximum.
    maxima: dict
    #: Pairs of a triangular factor and the transformed right-hand sides.
    factors: dict
    #: Pairs of an array of keys and a matrix with one row of sums for each key.
    keyed: dict


class FitTable(NamedTuple):
    """Container for the units and the period of each fit that cells share.

    :func:`~moderndid.did.shards.plan.plan_cells` decides which cells share a
    fit.

    Attributes
    ----------
    cohorts : ndarray
        Cohorts of the data in increasing order with infinity for the
        never-treated units, of shape (C,).
    group : ndarray
        Position of each fit's treated cohort in ``cohorts``, of shape (F,).
    controls : ndarray
        Whether each cohort is in each fit's comparison group, of shape (F, C).
    earlier : ndarray
        Index of the period of each fit's covariates and weights, of shape (F,).
    """

    #: Cohorts of the data in increasing order.
    cohorts: np.ndarray
    #: Position of each fit's treated cohort.
    group: np.ndarray
    #: Whether each cohort belongs to each fit's comparison group.
    controls: np.ndarray
    #: Index of the period that supplies each fit's covariates and weights.
    earlier: np.ndarray


class CellTable(NamedTuple):
    """Container for the fit and the two periods of each cell that a pass estimates.

    Attributes
    ----------
    fit : ndarray
        Position of each cell's fit in its :class:`FitTable`, of shape (K,).
    time : ndarray
        Index of each cell's period, of shape (K,).
    base : ndarray
        Index of each cell's base period, of shape (K,).
    """

    #: Position of each cell's fit.
    fit: np.ndarray
    #: Index of each cell's period.
    time: np.ndarray
    #: Index of each cell's base period.
    base: np.ndarray


class ShardPlan(NamedTuple):
    """Container for every group-time cell and the fits the estimable cells share.

    Attributes
    ----------
    specs : tuple
        One :class:`~moderndid.did.container.CellSpec`, or None for a cell
        skipped without a warning, for each group-time pair in the order in
        which att_gt reports the cells.
    estimable : ndarray
        Position in ``specs`` of each cell that an estimator fits, of shape (K,).
    fits : FitTable
        The fits that the estimable cells share.
    cells : CellTable
        The fit and the periods of each estimable cell.
    """

    #: Description of each group-time pair.
    specs: tuple
    #: Position of each cell that an estimator fits.
    estimable: np.ndarray
    #: The fits that the estimable cells share.
    fits: FitTable
    #: The fit and the periods of each estimable cell.
    cells: CellTable


class SummaryState(NamedTuple):
    """Container for what the summary pass needs to describe every fit.

    Attributes
    ----------
    fits : FitTable
        The fits to describe.
    cells : CellTable
        The cells whose outcome changes are the regressions' right-hand sides.
    logit : bool
        Whether the fits estimate a propensity score.
    regression : bool
        Whether the fits estimate an outcome regression.
    """

    #: The fits to describe.
    fits: FitTable
    #: The cells whose outcome changes are the regressions' right-hand sides.
    cells: CellTable
    #: Whether the fits estimate a propensity score.
    logit: bool
    #: Whether the fits estimate an outcome regression.
    regression: bool


class FitState(NamedTuple):
    """Container for the propensity score fits that a pass evaluates.

    Attributes
    ----------
    table : FitTable
        The fits.
    normalizer : ndarray
        Mean sampling weight of each fit's units, or 1 when their weights sum
        to zero, of shape (F,).
    scale : ndarray
        Power of two that divides each column of each fit's logit design, of
        shape (F, p + 1).
    coefficients : ndarray
        Logit coefficients of each fit on its scaled design, of shape
        (F, p + 1).
    """

    #: The fits.
    table: FitTable
    #: Mean sampling weight of each fit's units.
    normalizer: np.ndarray
    #: Power of two that divides each column of each fit's logit design.
    scale: np.ndarray
    #: Logit coefficients of each fit on its scaled design.
    coefficients: np.ndarray


class CellSumsState(NamedTuple):
    """Container for the state of the pass that sums the parts of each cell's estimate.

    Attributes
    ----------
    fits : FitState
        The fits at their final coefficients.
    cells : CellTable
        The cells to sum.
    regression : ndarray
        Outcome regression coefficients of each cell on its unscaled design, of
        shape (K, p + 1).
    method : {"dr", "ipw", "reg"}
        Estimator of the cells.
    """

    #: The fits at their final coefficients.
    fits: FitState
    #: The cells to sum.
    cells: CellTable
    #: Outcome regression coefficients of each cell.
    regression: np.ndarray
    #: Estimator of the cells.
    method: str


class MomentsState(NamedTuple):
    """Container for the state of the pass that sums each unit's influence function row.

    The K kept cells and their influence function columns in
    :class:`~moderndid.InfluenceMoments` follow the order of the result.

    Attributes
    ----------
    fits : FitState
        The fits at their final coefficients.
    cells : CellTable
        The fit and the periods of each kept cell, read only for estimates.
    kind : ndarray
        0 for an estimate, 1 for a reference cell with a zero influence
        function, or 2 for a failed cell with a NaN influence function, of
        shape (K,).
    regression : ndarray
        Outcome regression coefficients of each cell on its unscaled design, of
        shape (K, p + 1).
    constants : ndarray
        Scale of the influence function as the number of units over the number
        of the cell's units, the treated and comparison terms of the effect,
        and the mean treated and comparison weights of each cell, of shape
        (K, 5).
    directions : ndarray
        Products of the inverse Gram or Hessian matrices with the derivatives of
        the treated term, of the comparison term through the propensity score,
        and of the comparison term through the regression, of shape
        (K, 3, p + 1).
    method : {"dr", "ipw", "reg"}
        Estimator of the cells.
    cohorts : ndarray
        Treated cohorts that get a weight column, in increasing order.
    biters : int
        Number of bootstrap draws, or 0 for none.
    random_state : int, optional
        Seed of the bootstrap multipliers, or None without draws.
    """

    #: The fits at their final coefficients.
    fits: FitState
    #: The fit and the periods of each kept cell.
    cells: CellTable
    #: Code of each cell as an estimate, a reference cell, or a failed cell.
    kind: np.ndarray
    #: Outcome regression coefficients of each cell.
    regression: np.ndarray
    #: Scale, effect terms, and mean weights of each cell.
    constants: np.ndarray
    #: Inverse Gram or Hessian matrices times the derivatives of each cell.
    directions: np.ndarray
    #: Estimator of the cells.
    method: str
    #: Treated cohorts that get a weight column.
    cohorts: np.ndarray
    #: Number of bootstrap draws.
    biters: int
    #: Seed of the bootstrap multipliers.
    random_state: int | None = None


class LogitFit(NamedTuple):
    """Container for a propensity score fit that the iteration passes finished.

    Attributes
    ----------
    coefficients : ndarray
        Coefficients on the scaled design, of shape (p + 1,).
    converged : bool
        Whether the deviance settled within 100 iterations.
    iterations : int
        Number of iterations the fit ran.
    separated : tuple
        Whether the fitted scores of each iteration all lay within 1e-8 of
        the treatment indicator.
    overflowed : tuple
        Whether the exponential in the scores of each iteration overflows for
        some row. The fits of a balanced panel record no overflow.
    error : str, optional
        Why the fit failed, or None for a fit that finished.
    kept : tuple, optional
        Positions of the design columns that the fit keeps, or None when it
        keeps every column. A finished fit gives each dropped column a zero
        coefficient.
    dropped : int, default=0
        Number of covariates that the fit drops.
    """

    #: Coefficients on the scaled design.
    coefficients: np.ndarray
    #: Whether the deviance settled within 100 iterations.
    converged: bool
    #: Number of iterations the fit ran.
    iterations: int
    #: Whether each iteration's scores separated the units.
    separated: tuple
    #: Whether each iteration's scores overflowed the exponential.
    overflowed: tuple
    #: Why the fit failed.
    error: str | None = None
    #: Positions of the design columns that the fit keeps.
    kept: tuple | None = None
    #: Number of covariates that the fit drops.
    dropped: int = 0


class CellEstimate(NamedTuple):
    """Container for the estimate of one cell and the constants of its influence function.

    Attributes
    ----------
    att : float
        Estimated average treatment effect on the treated of the cell.
    terms : ndarray
        Treated and comparison terms of the effect and the mean treated and
        comparison weights, of shape (4,).
    directions : ndarray
        Products of the inverse Gram or Hessian matrices with the derivatives
        that :class:`MomentsState` describes, of shape (3, p + 1).
    coefficients : ndarray
        Outcome regression coefficients of the cell on the unscaled design, of
        shape (p + 1,).
    """

    #: Estimated average treatment effect on the treated of the cell.
    att: float
    #: Effect terms and mean weights of the cell.
    terms: np.ndarray
    #: Inverse Gram or Hessian matrices times the derivatives of the cell.
    directions: np.ndarray
    #: Outcome regression coefficients of the cell.
    coefficients: np.ndarray


class RegressionFit(NamedTuple):
    """Container for the outcome regressions that share one weighted design.

    Attributes
    ----------
    coefficients : ndarray
        Coefficients of each of the k right-hand sides on the unscaled design,
        of shape (k, p + 1).
    collinear : bool
        Whether the weighted design lacks full column rank.
    condition_number : float
        Condition number of the weighted design with its columns scaled.
    error : str, optional
        Why the solve failed, or None for a solve that finished.
    """

    #: Coefficients of each right-hand side on the unscaled design.
    coefficients: np.ndarray
    #: Whether the weighted design lacks full column rank.
    collinear: bool
    #: Condition number of the weighted design with its columns scaled.
    condition_number: float
    #: Why the solve failed.
    error: str | None = None


class RowSummaryState(NamedTuple):
    """Container for what the summary pass over rows needs to describe every cell.

    Cell :math:`j` takes row :math:`j` of ``fits`` and of ``cells``.

    Attributes
    ----------
    fits : FitTable
        The treated cohort and the comparison group of each cell.
    cells : CellTable
        The period and the base period of each cell.
    method : {"dr", "ipw", "reg"}
        Estimator of the cells.
    """

    #: The treated cohort and the comparison group of each cell.
    fits: FitTable
    #: The period and the base period of each cell.
    cells: CellTable
    #: Estimator of the cells.
    method: str


class RowFitState(NamedTuple):
    """Container for the cells of rows that a pass evaluates at their propensity score coefficients.

    Cell :math:`j` takes row :math:`j` of every field.

    Attributes
    ----------
    fits : FitTable
        The treated cohort and the comparison group of each cell.
    cells : CellTable
        The period and the base period of each cell.
    normalizer : ndarray
        Mean sampling weight of each cell's rows, or 1 when their weights sum
        to zero, of shape (K,).
    scale : ndarray
        Power of two that divides each column of each cell's logit design, of
        shape (K, p + 1).
    coefficients : ndarray
        Logit coefficients of each cell on its scaled design, of shape
        (K, p + 1).
    """

    #: The treated cohort and the comparison group of each cell.
    fits: FitTable
    #: The period and the base period of each cell.
    cells: CellTable
    #: Mean sampling weight of each cell's rows.
    normalizer: np.ndarray
    #: Power of two that divides each column of each cell's logit design.
    scale: np.ndarray
    #: Logit coefficients of each cell on its scaled design.
    coefficients: np.ndarray


class RowCellSumsState(NamedTuple):
    """Container for the state of the pass that sums the parts of each cell's estimate over rows.

    The four groups of a cell follow the order of its comparison group in the
    base period, its comparison group in its own period, its treated group in
    the base period, and its treated group in its own period.

    Attributes
    ----------
    fits : RowFitState
        The cells at their final propensity score coefficients.
    regression : ndarray
        Outcome regression coefficients of each cell's four groups on the
        unscaled design, of shape (K, 4, p + 1). A regression that the
        estimator doesn't fit has zeros.
    centers : ndarray
        Value that each cell's two comparison groups subtract from their effect
        terms before they multiply them by the covariates, of shape (K, 2).
    method : {"dr", "ipw", "reg"}
        Estimator of the cells.
    """

    #: The cells at their final propensity score coefficients.
    fits: RowFitState
    #: Outcome regression coefficients of each cell's four groups.
    regression: np.ndarray
    #: Value that the comparison groups subtract before the covariate products.
    centers: np.ndarray
    #: Estimator of the cells.
    method: str


class RowMomentsState(NamedTuple):
    """Container for the state of the pass that sums the influence function row of each unit of rows.

    The kept cells follow the order of the result as in :class:`MomentsState`.
    The groups of a cell follow the order of :class:`RowCellSumsState`.

    Attributes
    ----------
    fits : RowFitState
        The kept cells at their final propensity score coefficients, read only
        for estimates.
    kind : ndarray
        Kind of each cell as :class:`MomentsState` codes it, of shape (K,).
    regression : ndarray
        Outcome regression coefficients of each cell's four groups on the
        unscaled design, of shape (K, 4, p + 1).
    ratio : ndarray
        Scale of each cell's influence function as the number of units of the
        data over the number of the cell's rows, of shape (K,).
    terms : ndarray
        Terms of each cell's effect, of shape (K, 8). The doubly robust
        estimator holds the mean residuals of its four groups and its four
        adjustment terms. The inverse probability estimator holds the mean
        outcomes of its four groups. The outcome regression estimator holds
        the mean outcomes of its treated groups and its comparison term in the
        third to fifth columns.
    means : ndarray
        Mean weight of each cell's four groups and of its treated rows in both
        periods, of shape (K, 5).
    directions : ndarray
        Products of the inverse Gram matrix of each group's regression and of
        the inverse logit Hessian with the derivatives of the effect, of shape
        (K, 5, p + 1).
    method : {"dr", "ipw", "reg"}
        Estimator of the cells.
    cohorts : ndarray
        Treated cohorts that get a weight column, in increasing order.
    biters : int
        Number of bootstrap draws, or 0 for none.
    random_state : int, optional
        Seed of the bootstrap multipliers, or None without draws.
    """

    #: The kept cells at their final propensity score coefficients.
    fits: RowFitState
    #: Code of each cell as an estimate, a reference cell, or a failed cell.
    kind: np.ndarray
    #: Outcome regression coefficients of each cell's four groups.
    regression: np.ndarray
    #: Scale of each cell's influence function.
    ratio: np.ndarray
    #: Terms of each cell's effect.
    terms: np.ndarray
    #: Mean weights of each cell's groups and treated rows.
    means: np.ndarray
    #: Inverse Gram or Hessian matrices times the derivatives of each cell.
    directions: np.ndarray
    #: Estimator of the cells.
    method: str
    #: Treated cohorts that get a weight column.
    cohorts: np.ndarray
    #: Number of bootstrap draws.
    biters: int
    #: Seed of the bootstrap multipliers.
    random_state: int | None = None


class RowCellEstimate(NamedTuple):
    """Container for the estimate of one cell of rows and the constants of its influence function.

    Attributes
    ----------
    att : float
        Estimated average treatment effect on the treated of the cell.
    terms : ndarray
        Terms of the effect that :class:`RowMomentsState` describes, of shape
        (8,).
    means : ndarray
        Mean weights of the cell's four groups and of its treated rows, of
        shape (5,).
    directions : ndarray
        Products of the inverse Gram or Hessian matrices with the derivatives
        that :class:`RowMomentsState` describes, of shape (5, p + 1).
    coefficients : ndarray
        Outcome regression coefficients of the cell's four groups on the
        unscaled design, of shape (4, p + 1).
    """

    #: Estimated average treatment effect on the treated of the cell.
    att: float
    #: Terms of the effect.
    terms: np.ndarray
    #: Mean weights of the cell's groups and treated rows.
    means: np.ndarray
    #: Inverse Gram or Hessian matrices times the derivatives of the cell.
    directions: np.ndarray
    #: Outcome regression coefficients of the cell's four groups.
    coefficients: np.ndarray
