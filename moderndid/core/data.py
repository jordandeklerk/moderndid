"""Datasets."""

import gzip
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from .dataframe import to_polars

__all__ = ["load_nsw", "load_mpdta", "load_ehec", "load_engel", "simulate_cont_did_data"]


def load_nsw() -> pl.DataFrame:
    """Load the NSW (National Supported Work) demonstration dataset.

    This dataset is from the National Supported Work (NSW) Demonstration,
    a randomized employment training program operated in the mid-1970s.
    It has been widely used in the causal inference literature, particularly
    for demonstrating difference-in-differences methods.

    The dataset is in long format with observations for 1975 (pre-treatment)
    and 1978 (post-treatment) periods.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - *id*: Individual identifier
        - *year*: Year (1975 or 1978)
        - *experimental*: Treatment indicator (1 if treated, 0 if control)
        - *re*: Real earnings (outcome variable)
        - *age*: Age in years
        - *educ*: Years of education
        - *black*: Indicator for Black race
        - *married*: Indicator for married status
        - *nodegree*: Indicator for no high school degree
        - *hisp*: Indicator for Hispanic ethnicity
        - *re74*: Real earnings in 1974

    Notes
    -----
    This dataset was used in Lalonde (1986) and has been extensively analyzed
    in the treatment effects literature. The version included here is formatted
    for panel data difference-in-differences analysis.

    References
    ----------

    .. [1] Lalonde, R. (1986). Evaluating the econometric evaluations of
        training programs with experimental data. American Economic Review,
        76(4), 604-620.
    """
    data_path = Path(__file__).parent / "datasets" / "nsw_long.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"NSW data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    with gzip.open(data_path, "rb") as f:
        nsw_data = pickle.load(f)

    return to_polars(nsw_data)


def load_mpdta() -> pl.DataFrame:
    """Load the County Teen Employment dataset for multiple time period DiD analysis.

    This dataset contains county-level teen employment rates from 2003-2007
    with staggered treatment timing (minimum wage increases). States were first
    treated in 2004, 2006, or 2007.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - *year*: Year (2003-2007)
        - *countyreal*: County identifier
        - *lpop*: Log of county population
        - *lemp*: Log of county-level teen employment (outcome variable)
        - *first.treat*: Period when state first increased minimum wage (2004, 2006, 2007, or 0 for never-treated)
        - *treat*: Treatment indicator (1 if treated, 0 if control)

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
        with multiple time periods. Journal of Econometrics, 225(2), 200-230.
    """
    data_path = Path(__file__).parent / "datasets" / "mpdta_long.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"MPDTA data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    with gzip.open(data_path, "rb") as f:
        mpdta_data = pickle.load(f)

    mpdta_data["first.treat"] = mpdta_data["first.treat"].astype(np.int64)

    return to_polars(mpdta_data)


def load_ehec() -> pl.DataFrame:
    """Load the EHEC dataset for Medicaid expansion analysis.

    This dataset contains state-level data on health insurance coverage rates
    among low-income childless adults from 2008-2019, used to study the effects
    of Medicaid expansion under the Affordable Care Act.

    The dataset tracks states that expanded Medicaid at different times
    (2014, 2015, 2016, 2017, or 2019) as well as states that never expanded
    during the sample period.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - *stfips*: State FIPS code identifier
        - *year*: Year (2008-2019)
        - *dins*: Share of low-income childless adults with health insurance (outcome variable)
        - *yexp2*: Year that state expanded Medicaid (2014, 2015, 2016, 2017, 2019, or NaN for never-expanded)
        - *W*: State population weights

    Notes
    -----
    This dataset is commonly used in staggered adoption difference-in-differences
    settings and for demonstrating methods that account for treatment effect
    heterogeneity across time and cohorts.

    The data comes from the Mixtape Sessions Advanced DID workshop and is used
    in examples demonstrating the HonestDiD method for sensitivity analysis.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    data_path = Path(__file__).parent / "datasets" / "ehec_data.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"EHEC data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    with gzip.open(data_path, "rb") as f:
        ehec_data = pickle.load(f)

    return to_polars(ehec_data)


def load_engel() -> pl.DataFrame:
    """Load the Engel household expenditure dataset.

    This dataset contains household expenditure data used to study Engel curves,
    which describe how household expenditure on different goods varies with income.
    The data includes expenditure shares on various categories and household
    characteristics.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - *food*: Food expenditure share
        - *catering*: Catering expenditure share
        - *alcohol*: Alcohol expenditure share
        - *fuel*: Fuel expenditure share
        - *motor*: Motor expenditure share
        - *fares*: Transportation fares expenditure share
        - *leisure*: Leisure expenditure share
        - *logexp*: Log of total expenditure
        - *logwages*: Log of wages
        - *nkids*: Number of children

    Notes
    -----
    This dataset is commonly used for demonstrating nonparametric methods,
    particularly for estimating Engel curves and testing for monotonicity
    or shape restrictions in consumer demand.

    References
    ----------

    .. [1] Engel, E. (1857). Die Lebenskosten belgischer Arbeiter-Familien.
        Dresden: C. Heinrich.
    """
    data_path = Path(__file__).parent / "datasets" / "engel.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Engel data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    with gzip.open(data_path, "rb") as f:
        engel_data = pickle.load(f)

    return to_polars(engel_data)


def simulate_cont_did_data(
    n: int = 500,
    num_time_periods: int = 4,
    num_groups: int | None = None,
    p_group: list | None = None,
    p_untreated: float | None = None,
    dose_linear_effect: float = 0.5,
    dose_quadratic_effect: float = 0,
    seed: int = 42,
) -> pl.DataFrame:
    """Simulate panel data for difference-in-differences with continuous treatment.

    Parameters
    ----------
    n : int, default=500
        Number of cross-sectional units.
    num_time_periods : int, default=4
        Number of time periods.
    num_groups : int, optional
        Number of timing groups. Defaults to ``num_time_periods``.
        Groups consist of a never-treated group (G=0) and groups that
        become treated in periods 2, 3, ..., num_time_periods.
    p_group : list, optional
        Probabilities for each treated group. Defaults to equal probabilities.
    p_untreated : float, optional
        Probability of being in the never-treated group.
        Defaults to ``1/num_groups``.
    dose_linear_effect : float, default=0.5
        True linear effect of treatment dose on the outcome.
    dose_quadratic_effect : float, default=0
        True quadratic effect of treatment dose on the outcome.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        A balanced panel DataFrame with columns:

        - *id*: Unit identifier
        - *time_period*: Time period (1, 2, ..., num_time_periods)
        - *Y*: Outcome variable
        - *G*: Timing group (0 for never-treated, or period when treatment starts)
        - *D*: Treatment dose (0 for untreated unit-periods, positive otherwise)
    """
    rng = np.random.default_rng(seed)

    if num_groups is None:
        num_groups = num_time_periods

    time_periods = np.arange(1, num_time_periods + 1)
    groups = np.concatenate(([0], time_periods[1:]))

    if p_untreated is None:
        p_untreated = 1 / num_groups

    if p_group is None:
        p_group_len = num_groups - 1
        p_group = np.repeat((1 - p_untreated) / p_group_len, p_group_len)

    p = np.concatenate(([p_untreated], p_group))
    p /= p.sum()

    group = rng.choice(groups, n, replace=True, p=p)
    dose = rng.uniform(0, 1, n)

    eta = rng.normal(loc=group, scale=1, size=n)
    time_effects = np.arange(1, num_time_periods + 1)
    y0_t = time_effects + eta[:, np.newaxis] + rng.normal(size=(n, num_time_periods))

    y1_t = (
        dose_linear_effect * dose[:, np.newaxis]
        + dose_quadratic_effect * (dose**2)[:, np.newaxis]
        + time_effects
        + eta[:, np.newaxis]
        + rng.normal(size=(n, num_time_periods))
    )

    post_matrix = (group[:, np.newaxis] <= time_periods) & (group[:, np.newaxis] != 0)
    y = post_matrix * y1_t + (1 - post_matrix) * y0_t

    df = pl.DataFrame(
        {
            **{f"Y_{t}": y[:, i] for i, t in enumerate(time_periods)},
            "id": np.arange(1, n + 1),
            "G": group,
            "D": dose,
        }
    )

    df_long = df.unpivot(
        index=["id", "G", "D"],
        on=[f"Y_{t}" for t in time_periods],
        variable_name="time_period",
        value_name="Y",
    )

    df_long = df_long.with_columns(pl.col("time_period").str.replace("Y_", "").cast(pl.Int64))
    df_long = df_long.with_columns(pl.when(pl.col("G") == 0).then(pl.lit(0.0)).otherwise(pl.col("D")).alias("D"))

    return df_long.sort(["id", "time_period"])
