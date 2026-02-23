"""Datasets."""

import gzip
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from ..didtriple.dgp import (
    _build_cov_dict,
    _fps2,
    _freg,
    _generate_ps_coefficients,
    _select_covars,
    _transform_covariates,
)
from .dataframe import to_polars

__all__ = [
    "gen_did_scalable",
    "load_cai2016",
    "load_ehec",
    "load_engel",
    "load_favara_imbs",
    "load_mpdta",
    "load_nsw",
    "simulate_cont_did_data",
]


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


def load_favara_imbs() -> pl.DataFrame:
    """Load the Favara and Imbs banking deregulation dataset.

    This dataset contains county-level data on bank lending and interstate
    branching deregulation from 1994-2005, used to study the effects of
    banking deregulation on credit supply. The treatment (interstate branching)
    is non-binary and potentially non-absorbing, making it suitable for
    intertemporal treatment effects estimation.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - *year*: Year (1994-2005)
        - *county*: County identifier
        - *state_n*: State number
        - *Dl_vloans_b*: Change in log volume of loans (outcome variable)
        - *inter_bra*: Interstate branching indicator (treatment variable)
        - *w1*: Sampling weight
        - *Dl_hpi*: Change in log house price index

    Notes
    -----
    This dataset is commonly used for demonstrating difference-in-differences
    methods with time-varying treatments, particularly the de Chaisemartin and
    D'Haultfoeuille (2024) estimator for intertemporal treatment effects.

    References
    ----------

    .. [1] Favara, G., & Imbs, J. (2015). Credit supply and the price of
        housing. American Economic Review, 105(3), 958-992.

    .. [2] de Chaisemartin, C., & D'Haultfoeuille, X. (2024). Difference-in-
        Differences Estimators of Intertemporal Treatment Effects.
        Review of Economics and Statistics, 106(6), 1723-1736.
    """
    data_path = Path(__file__).parent / "datasets" / "favara_imbs.csv.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Favara-Imbs data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    return pl.read_csv(data_path)


def load_cai2016() -> pl.DataFrame:
    """Load the Cai (2016) agricultural insurance dataset.

    This dataset contains household-level panel data from rural Jiangxi province
    in China (2000-2008), used to study the effects of weather-indexed crop
    insurance on household saving behavior. The People's Insurance Company of
    China (PICC) introduced crop insurance for tobacco farmers in select counties
    in 2003, creating a triple difference-in-differences (DDD) design with three
    sources of variation: treatment region, household eligibility (tobacco vs
    non-tobacco farmers), and time (pre/post 2003).

    The dataset includes all households with non-missing outcome and covariate
    values, forming an unbalanced panel of 3,659 households (32,391
    observations). Most households are observed in all 9 years, but some have
    fewer observations.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - *hhno*: Household identifier
        - *year*: Year (2000-2008)
        - *treatment*: Treatment region indicator (1 if in treated county, 0 otherwise)
        - *sector*: Eligibility indicator (1 for tobacco farmers, 0 for non-tobacco)
        - *checksaving_ratio*: Flexible-term saving ratio (outcome variable)
        - *savingtotal_rate*: Total saving rate
        - *hhsize*: Household size
        - *age*: Age of head of household
        - *educ_scale*: Education level of head of household
        - *county*: County identifier (for clustering)

    References
    ----------

    .. [1] Cai, J. (2016). The impact of insurance provision on household
        production and financial decisions. American Economic Journal:
        Economic Policy, 8(2), 44-88.

    .. [2] Ortiz-Villavicencio, J. & Sant'Anna, P. H. C. (2025). Triple
        Differences with Multiple Periods. arXiv preprint arXiv:2505.09942.
    """
    data_path = Path(__file__).parent / "datasets" / "cai2016.csv.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cai (2016) data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    return pl.read_csv(data_path)


def gen_did_scalable(
    n: int,
    dgp_type: int = 1,
    n_periods: int = 10,
    n_cohorts: int = 8,
    n_covariates: int = 20,
    att_base: float = 10.0,
    panel: bool = True,
    random_state=None,
) -> dict:
    """Generate configurable staggered DiD data for stress-testing.

    Parameters
    ----------
    n : int
        Number of units (panel) or observations per period (repeated
        cross-section).
    dgp_type : {1, 2, 3, 4}, default=1
        Controls nuisance function specification:

        - 1: Both propensity score and outcome regression use Z (both correct)
        - 2: Propensity score uses X, outcome regression uses Z (OR correct)
        - 3: Propensity score uses Z, outcome regression uses X (PS correct)
        - 4: Both use X (both misspecified when estimating with Z)

    n_periods : int, default=10
        Total number of time periods (labeled 1..T). Must be >= 2.
    n_cohorts : int, default=8
        Number of treated cohorts (excludes never-treated g=0). Must be >= 1
        and < n_periods. Cohorts adopt treatment at times 2, 3, ...,
        n_cohorts+1.
    n_covariates : int, default=20
        Total covariates. Must be >= 4. First 4 get nonlinear transform via
        ``_transform_covariates``; rest are raw standard normals.
    att_base : float, default=10.0
        Base treatment effect. Cohort g at period t >= g gets
        ``att_base * g * (t - g + 1)``.
    panel : bool, default=True
        If True, generate panel data. If False, generate repeated
        cross-section data with disjoint units per period.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - *data*: pl.DataFrame in long format with columns [id, group,
          time, y, cov1..covK, cluster]
        - *data_wide*: pl.DataFrame in wide format (panel with
          n_periods <= 20 only)
        - *att_config*: dict mapping each treated cohort g to
          ``att_base * g``
        - *cohort_values*: list of all cohort values
          [0, 2, 3, ..., n_cohorts+1]
        - *n_periods*: number of periods
        - *n_covariates*: number of covariates
    """
    if dgp_type not in {1, 2, 3, 4}:
        raise ValueError(f"dgp_type must be 1, 2, 3, or 4, got {dgp_type}")
    if n_periods < 2:
        raise ValueError(f"n_periods must be >= 2, got {n_periods}")
    if n_cohorts < 1:
        raise ValueError(f"n_cohorts must be >= 1, got {n_cohorts}")
    if n_cohorts >= n_periods:
        raise ValueError(f"n_cohorts must be < n_periods, got n_cohorts={n_cohorts}, n_periods={n_periods}")
    if n_covariates < 4:
        raise ValueError(f"n_covariates must be >= 4, got {n_covariates}")

    rng = np.random.default_rng(random_state)
    xsi_ps = 0.4
    b1 = np.array([27.4, 13.7, 13.7, 13.7])

    cohort_values = np.array([0, *list(range(2, n_cohorts + 2))])
    n_free = n_cohorts  # treated cohorts as free categories, never-treated as reference
    coef_rng = np.random.default_rng(12345)
    ws, psis, cs = _generate_ps_coefficients(coef_rng, n_free)

    if panel:
        x_first4 = rng.standard_normal((n, 4))
        z_first4 = _transform_covariates(x_first4)
        x_extra = rng.standard_normal((n, n_covariates - 4)) if n_covariates > 4 else None

        ps_covars, or_covars = _select_covars(dgp_type, x_first4, z_first4)
        cohort = _assign_did_cohort(rng, n, n_free, ws, psis, cs, ps_covars, cohort_values, xsi_ps)

        index_lin = _freg(b1, or_covars)
        index_unobs_het = cohort * index_lin
        index_trend = index_lin

        v = rng.normal(loc=index_unobs_het, scale=1.0)
        index_pt_violation = v / 10
        baseline = index_lin + v

        clusters = rng.integers(1, 51, size=n)
        cov_dict = _build_cov_dict(z_first4, x_extra, n_covariates)

        y_all = {}
        df_list = []
        for t in range(1, n_periods + 1):
            y_t = _compute_did_outcome(
                t, baseline, index_trend, index_pt_violation, cohort, cohort_values, att_base, n, rng
            )
            y_all[t] = y_t
            row_dict = {
                "id": np.arange(1, n + 1),
                "group": cohort,
                "time": np.full(n, t, dtype=int),
                "y": y_t,
            }
            row_dict.update(cov_dict)
            row_dict["cluster"] = clusters
            df_list.append(pl.DataFrame(row_dict))

        data = pl.concat(df_list).sort(["id", "time"])

        if n_periods <= 20:
            wide_dict = {
                "id": np.arange(1, n + 1),
                "group": cohort,
            }
            for t in range(1, n_periods + 1):
                wide_dict[f"y_t{t}"] = y_all[t]
            wide_dict.update(cov_dict)
            wide_dict["cluster"] = clusters
            data_wide = pl.DataFrame(wide_dict)
        else:
            data_wide = None

    else:
        df_list = []
        id_offset = 0

        for t in range(1, n_periods + 1):
            x_first4 = rng.standard_normal((n, 4))
            z_first4 = _transform_covariates(x_first4)
            x_extra = rng.standard_normal((n, n_covariates - 4)) if n_covariates > 4 else None

            ps_covars, or_covars = _select_covars(dgp_type, x_first4, z_first4)
            cohort = _assign_did_cohort(rng, n, n_free, ws, psis, cs, ps_covars, cohort_values, xsi_ps)

            index_lin = _freg(b1, or_covars)
            index_unobs_het = cohort * index_lin
            index_trend = index_lin

            v = rng.normal(loc=index_unobs_het, scale=1.0)
            index_pt_violation = v / 10
            baseline = index_lin + v

            y_t = _compute_did_outcome(
                t, baseline, index_trend, index_pt_violation, cohort, cohort_values, att_base, n, rng
            )

            clusters = rng.integers(1, 51, size=n)
            cov_dict = _build_cov_dict(z_first4, x_extra, n_covariates)

            row_dict = {
                "id": np.arange(id_offset + 1, id_offset + n + 1),
                "group": cohort,
                "time": np.full(n, t, dtype=int),
                "y": y_t,
            }
            row_dict.update(cov_dict)
            row_dict["cluster"] = clusters
            df_list.append(pl.DataFrame(row_dict))
            id_offset += n

        data = pl.concat(df_list)
        data_wide = None

    att_config = {int(g): att_base * g for g in cohort_values if g != 0}

    return {
        "data": data,
        "data_wide": data_wide,
        "att_config": att_config,
        "cohort_values": cohort_values.tolist(),
        "n_periods": n_periods,
        "n_covariates": n_covariates,
    }


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


def _assign_did_cohort(rng, n, n_free, ws, psis, cs, ps_covars, cohort_values, xsi_ps):
    """Multinomial draw to cohort array."""
    exp_vals = np.empty((n, n_free))
    for i in range(n_free):
        exp_vals[:, i] = np.exp(_fps2(xsi_ps * psis[i], ws[i], ps_covars, cs[i]))

    sum_exp = 1.0 + exp_vals.sum(axis=1, keepdims=True)
    probs = exp_vals / sum_exp
    prob_ref = 1.0 / sum_exp

    all_probs = np.column_stack([probs, prob_ref])
    cum_probs = np.cumsum(all_probs, axis=1)
    u = rng.uniform(size=n)
    group_types = (u[:, None] >= cum_probs).sum(axis=1)

    treated_cohorts = cohort_values[cohort_values != 0]
    all_cohorts = np.concatenate([treated_cohorts, [0]])
    return all_cohorts[group_types]


def _compute_did_outcome(t, baseline, index_trend, index_pt_violation, cohort, cohort_values, att_base, n, rng):
    """Per-period outcome with treatment effects."""
    baseline_t = baseline + (t - 1) * index_trend + (t - 1) * index_pt_violation
    y = baseline_t + rng.standard_normal(n)

    for g in cohort_values:
        if g == 0 or t < g:
            continue
        k = t - g + 1
        y_g = baseline_t + rng.standard_normal(n) + att_base * g * k
        mask = cohort == g
        y[mask] = y_g[mask]

    return y
