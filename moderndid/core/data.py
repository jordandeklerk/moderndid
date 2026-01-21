"""Datasets."""

import gzip
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from .dataframe import to_polars

__all__ = ["load_nsw", "load_mpdta", "load_ehec", "load_engel"]


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
