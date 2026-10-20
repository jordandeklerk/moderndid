"""Group-time cells of a fit on shards and the fits they share."""

import numpy as np

from moderndid.did.compute_att_gt import cell_spec

from .container import CellTable, FitTable, ShardPlan


def plan_cells(config, cohort_counts):
    """Describe every group-time cell of a balanced panel and the fits the cells share.

    The cells follow :func:`~moderndid.did.compute_att_gt.cell_spec` in the
    order in which :func:`~moderndid.att_gt` reports them. Since estimable cells
    that share a treated cohort, a comparison group, and an earlier period use
    the same units, covariates, and weights, they share one propensity score
    fit and one outcome regression design.

    Parameters
    ----------
    config : DIDConfig
        Preprocessed configuration of the fit.
    cohort_counts : DataFrame
        Number of units of each cohort in the columns ``cohort`` and
        ``cohort_size``.

    Returns
    -------
    ShardPlan
        Object containing the cells and the fits they share:

        - **specs**: Description of each group-time pair, or None for a cell skipped without a warning
        - **estimable**: Position in ``specs`` of each cell that an estimator fits
        - **fits**: The fits that the estimable cells share
        - **cells**: The fit and the periods of each estimable cell
    """
    n_cells = len(config.time_periods) if config.base_period == "universal" else len(config.time_periods) - 1
    specs = tuple(
        cell_spec(group_idx, time_idx, config, cohort_counts)
        for group_idx in range(config.treated_groups_count)
        for time_idx in range(n_cells)
    )

    fit_index = {}
    estimable = []
    cell_fit = []
    for position, spec in enumerate(specs):
        if spec is None or spec.skip_warning is not None or spec.reference:
            continue
        key = (float(spec.group), tuple(sorted(float(c) for c in spec.control_cohorts)), int(spec.earlier_index))
        cell_fit.append(fit_index.setdefault(key, len(fit_index)))
        estimable.append(position)

    cohorts = np.unique(np.asarray(cohort_counts["cohort"].to_numpy(), dtype=float))
    keys = list(fit_index)
    fits = FitTable(
        cohorts=cohorts,
        group=np.array([np.searchsorted(cohorts, group) for group, _, _ in keys], dtype=np.int64),
        controls=np.array([np.isin(cohorts, controls) for _, controls, _ in keys], dtype=bool).reshape(
            len(keys), len(cohorts)
        ),
        earlier=np.array([earlier for _, _, earlier in keys], dtype=np.int64),
    )
    cells = CellTable(
        fit=np.array(cell_fit, dtype=np.int64),
        time=np.array([specs[position].time_index for position in estimable], dtype=np.int64),
        base=np.array([specs[position].base_index for position in estimable], dtype=np.int64),
    )
    return ShardPlan(specs=specs, estimable=np.array(estimable, dtype=np.int64), fits=fits, cells=cells)
