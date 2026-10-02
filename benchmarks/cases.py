"""Build seeded panel data for estimator timing."""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Workload:
    """Describe one estimator workload."""

    n_units: int = 1000
    n_periods: int = 6
    n_cohorts: int = 2
    n_covariates: int = 0
    boot: bool = False
    biters: int = 199


WORKLOADS = {
    "small": Workload(n_units=400, biters=99),
    "short": Workload(n_periods=4),
    "baseline": Workload(),
    "rows": Workload(n_units=10_000),
    "periods": Workload(n_periods=12),
    "cohorts": Workload(n_periods=10, n_cohorts=5),
    "covariates": Workload(n_covariates=4),
    "bootstrap": Workload(boot=True, biters=999),
}


def make_panel(workload, estimator):
    """Build a balanced panel for one estimator."""
    rng = np.random.default_rng(42)
    last_cohort = max(2, workload.n_periods - 1)
    cohorts = np.linspace(min(3, last_cohort), last_cohort, workload.n_cohorts, dtype=int)
    cohort_levels = np.concatenate(([0], cohorts))
    group = cohort_levels[(np.arange(workload.n_units) // 2) % len(cohort_levels)]
    partition = np.arange(workload.n_units) % 2
    order = rng.permutation(workload.n_units)
    group = group[order]
    partition = partition[order]
    dose = np.where(group > 0, rng.uniform(0.5, 2.0, workload.n_units), 0.0)
    if estimator == "did_multiplegt":
        dose = np.where(group > 0, rng.integers(1, 4, workload.n_units), 0)

    covariates = rng.normal(size=(workload.n_units, workload.n_covariates))
    ids = np.repeat(np.arange(workload.n_units), workload.n_periods)
    times = np.tile(np.arange(1, workload.n_periods + 1), workload.n_units)
    groups = np.repeat(group, workload.n_periods)
    eligibility = np.repeat(partition, workload.n_periods)
    treated = (groups > 0) & (times >= groups)
    exposure = np.maximum(times - groups, 0)
    treatment = treated.astype(float)
    if estimator in {"cont_did", "did_multiplegt"}:
        treatment *= np.repeat(dose, workload.n_periods)
    elif estimator == "ddd":
        treatment *= eligibility

    row_covariates = np.repeat(covariates, workload.n_periods, axis=0)
    if estimator == "did_multiplegt":
        row_covariates += rng.normal(scale=0.25, size=row_covariates.shape)
    unit_effect = rng.normal(size=workload.n_units)
    outcome = np.repeat(unit_effect, workload.n_periods) + 0.2 * times
    outcome += 0.25 * row_covariates.sum(axis=1)
    outcome += treatment * (1.0 + 0.1 * exposure)
    outcome += rng.normal(scale=0.5, size=len(ids))
    columns = {
        "id": ids,
        "time": times,
        "group": groups,
        "partition": eligibility,
        "dose": np.repeat(dose, workload.n_periods),
        "treatment": treatment,
        "y": outcome,
    }
    columns.update({f"x{index + 1}": row_covariates[:, index] for index in range(workload.n_covariates)})
    if estimator in {"drdid", "ipwdid", "ordid"}:
        columns["treated"] = (groups > 0).astype(int)
    return pl.DataFrame(columns)
