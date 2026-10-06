"""Datasets."""

import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from ..didtriple.dgp import (
    _assign_cohort_partition,
    _build_cov_dict,
    _compute_scalable_outcome,
    _fps,
    _fps2,
    _freg,
    _generate_ps_coefficients,
    _select_covars,
    _transform_covariates,
)
from .dataframe import to_polars

__all__ = [
    "gen_cont_did_data",
    "gen_ddd_2periods",
    "gen_ddd_mult_periods",
    "gen_ddd_scalable",
    "gen_did_scalable",
    "gen_simple_ddd_data",
    "load_acemoglu",
    "load_cai2016",
    "load_ehec",
    "load_engel",
    "load_favara_imbs",
    "load_fracking",
    "load_mpdta",
    "load_nsw",
]


def load_acemoglu():
    """Load the democracy and economic growth panel of 141 countries.

    This dataset is a six-year extract of the country panel that Acemoglu, Naidu,
    Restrepo, and Robinson [1]_ assembled to study whether democracy causes growth. It
    follows 141 countries over six consecutive years for 846 rows in all.

    Democracy can switch on and off, although only 12 of the countries change status at
    least once. The four lagged outcomes let an estimator hold past GDP per capita fixed
    when democracy responds to it, as dynamic covariate balancing does.

    See the :ref:`dynamic covariate balancing example <example_dyn_balancing>` for a full
    analysis of democracy and growth with this data.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - **Y**: Log GDP per capita (outcome). It's missing in the last year for four countries
        - **D**: Democracy indicator, 1 in the years a country was a democracy (treatment)
        - **Unit**: Country identifier
        - **Time**: Year, numbered 0 to 5
        - **region**: One of seven region codes, from AFR to SAS
        - **V1** through **V158**: Further country-level columns, of which **V47** through
          **V52** mark the six years and **V2** through **V46** are zero in every row
        - **lag1.Value1**: Log GDP per capita one year earlier
        - **lag2.Value1**: Log GDP per capita two years earlier
        - **lag3.Value1**: Log GDP per capita three years earlier
        - **lag4.Value1**: Log GDP per capita four years earlier

    References
    ----------

    .. [1] Acemoglu, D., Naidu, S., Restrepo, P., and Robinson, J.A.
       (2019). "Democracy does cause growth." Journal of Political
       Economy, 127(1), 47-100.
    """
    data_path = Path(__file__).parent / "datasets" / "acemoglu.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Acemoglu data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    with gzip.open(data_path, "rb") as f:
        data = pickle.load(f)

    df = to_polars(data)

    # Convert string Unit identifiers to numeric for compatibility with
    # estimators that require numeric panel IDs.
    if df["Unit"].dtype == pl.String:
        units = sorted(df["Unit"].unique().to_list())
        unit_map = {u: i for i, u in enumerate(units)}
        df = df.with_columns(pl.col("Unit").replace(unit_map).cast(pl.Int64))

    return df


def load_nsw():
    """Load the NSW (National Supported Work) demonstration dataset.

    This dataset is from the National Supported Work (NSW) Demonstration,
    a randomized employment training program operated in the mid-1970s.
    Lalonde [1]_ used the demonstration to check nonexperimental estimators
    against its experimental estimate. It has been widely used in the causal
    inference literature, particularly for demonstrating difference-in-differences
    methods.

    The dataset is a balanced panel in long format with 16,417 individuals
    observed in 1975 (pre-treatment) and 1978 (post-treatment), for a total
    of 32,834 observations.

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


def load_mpdta():
    """Load the County Teen Employment dataset for multiple time period DiD analysis.

    This dataset contains county-level teen employment rates from 2003-2007
    with staggered treatment timing (minimum wage increases). States were first
    treated in 2004, 2006, or 2007.

    The dataset is a balanced panel of 500 counties observed across 5 years,
    for a total of 2,500 observations. It is a subset of the data that Callaway
    and Sant'Anna [1]_ use in their application.

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


def load_ehec():
    """Load the EHEC dataset for Medicaid expansion analysis.

    This dataset holds the share of low-income adults without children who had health
    insurance in each state from 2008 to 2019. The shares come from the American Community
    Survey and are used to study the Medicaid expansion under the Affordable Care Act.

    The panel is balanced and covers 46 states over 12 years, 552 rows in all. Of those
    states, 30 expanded Medicaid in 2014, 2015, 2016, 2017, or 2019. The other 16 had not
    expanded by 2019.

    See the :ref:`sensitivity analysis example <example_honest_did>` for a full analysis of
    the Medicaid expansions with this data.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - **stfips**: State FIPS code identifier
        - **year**: Year (2008-2019)
        - **dins**: Share of low-income childless adults with health insurance (outcome variable)
        - **yexp2**: Year the state expanded Medicaid (2014, 2015, 2016, 2017, or 2019), missing for
          states that had not expanded by 2019
        - **W**: State population weights
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


def load_engel():
    """Load the Engel household expenditure dataset.

    Engel curves describe how household spending on a good varies with income.
    This dataset records how 1,655 households from the 1995 British Family
    Expenditure Survey split their budgets across seven groups of goods. It also
    holds each household's total expenditure, its earnings, and whether it has
    children.

    The households are married or cohabiting couples with an employed head and at
    most two children [1]_.

    See the :ref:`nonparametric IV example <example_npiv>` for the Engel curve for
    food estimated from this data.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - **food**: Food expenditure share
        - **catering**: Catering expenditure share
        - **alcohol**: Alcohol expenditure share
        - **fuel**: Fuel expenditure share
        - **motor**: Motor expenditure share
        - **fares**: Transportation fares expenditure share
        - **leisure**: Leisure expenditure share
        - **logexp**: Log of total expenditure
        - **logwages**: Log of total earnings
        - **nkids**: Indicator for children, 0 for none and 1 for one or two

    References
    ----------

    .. [1] Blundell, R., Chen, X., & Kristensen, D. (2007). Semi-nonparametric IV
        estimation of shape-invariant Engel curves. *Econometrica*, 75(6),
        1613-1669.
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


def load_favara_imbs():
    """Load the county panel of interstate branching deregulation and bank lending.

    After the Interstate Banking and Branching Efficiency Act of 1994, each US state
    chose when to lift four restrictions on branching by banks from other states.
    This dataset follows the counties that Favara and Imbs [1]_ studied from 1994 to
    2005. For each county and year it records how many restrictions the state had
    lifted and the growth of mortgage lending by banks. de Chaisemartin and
    D'Haultfoeuille [2]_ revisit it with intertemporal treatment effects.

    The file holds 12,538 rows for 1,048 counties in 50 states. All but five counties
    appear in each of the 12 years. The lending outcome is missing in 200 rows.

    See the :ref:`intertemporal treatment example <example_inter_did>` for a full
    analysis of the deregulations with this data.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:

        - **year**: Year (1994-2005)
        - **county**: County FIPS code
        - **state_n**: State FIPS code
        - **Dl_vloans_b**: Change in the log volume of mortgage loans originated by
          banks (outcome variable)
        - **inter_bra**: Number of the four restrictions on interstate branching that
          the state had lifted, from 0 to 4 (treatment variable)
        - **w1**: Inverse of the number of counties per state, the weight Favara and
          Imbs use for house prices, scaled to average one
        - **Dl_hpi**: Change in the log house price index

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


def load_fracking():
    """Load the county employment panel for the fracking application.

    The panel follows 402 US counties from 1990 through 2014 for 10,050 rows.
    Callaway, Goodman-Bacon, and Sant'Anna [1]_ use this county data from Bartik,
    Currie, Greenstone, and Knittel [2]_ in their continuous treatment application.
    The processed source file comes from their replication archive [3]_.

    The loader retains counties observed in all 25 years after removing missing
    employment outcomes. Each county's dose is a fixed geological prospectivity
    score. The source preparation sets missing scores to zero without retaining
    an imputation flag. Those zeros therefore cannot be distinguished from
    observed zero scores. The scores measure prospectivity within shale plays and are
    not comparable across plays.

    See the :ref:`continuous treatment example <example_cont_did>` for an
    analysis of geological prospectivity and county employment with this data.

    Returns
    -------
    pl.DataFrame
        A balanced county panel sorted by county and year.

        - **i**: County FIPS code
        - **t**: Year (1990-2014)
        - **y**: Log total county employment
        - **d**: Time-invariant geological prospectivity score
        - **G**: Adoption year based on publicity about successful fracking or 0 for zero-dose counties
        - **shale_basin1**: Shale-basin identifier
        - **G_original**: Formation adoption year retained for every county from the source file

    Notes
    -----
    The bundled extract is prepared from ``bcgk_replication.dta`` in [3]_.
    Preparation removes 2015 and missing employment outcomes, retains counties
    observed in every year from 1990 through 2014, and sets ``G=0`` when ``d=0``.
    The ``G_original`` column preserves the source dates used to restrict the
    comparison sample in the paper's time-averaged dose curves.
    Identifiers and years are integers. Outcomes and scores retain the source
    values without rescaling.

    Adoption dates follow the original study's first publicity about successful
    fracking in a shale play. Announcements after June are assigned to the
    following year to match the annual outcomes.

    The prospectivity data are county-level aggregates released with [2]_ and
    its replication package [4]_. Data reuse follows the AEA's published
    guidance for older replication deposits at
    https://aeadataeditor.github.io/aea-de-guidance/FAQ.html#licensing.
    The source data's rights are separate from ModernDiD's software license.

    References
    ----------

    .. [1] Callaway, B., Goodman-Bacon, A., and Sant'Anna, P. H. C. (2024).
       "Event Studies with a Continuous Treatment." AEA Papers and
       Proceedings, 114, 601-605. https://doi.org/10.1257/pandp.20241047.

    .. [2] Bartik, A. W., Currie, J., Greenstone, M., and Knittel, C. R. (2019).
       "The Local Economic and Welfare Consequences of Hydraulic Fracturing."
       American Economic Journal: Applied Economics, 11(4), 105-155.
       https://doi.org/10.1257/app.20170487.

    .. [3] Callaway, B., Goodman-Bacon, A., and Sant'Anna, P. H. C. (2024).
       "Data and Code for: Event-Studies with a Continuous Treatment."
       https://doi.org/10.3886/E201785V1.

    .. [4] Bartik, A. W., Currie, J., Greenstone, M., and Knittel, C. R.
       "Replication data for: The Local Economic and Welfare Consequences
       of Hydraulic Fracturing." https://doi.org/10.3886/E231454V1.
    """
    data_path = Path(__file__).parent / "datasets" / "fracking.csv.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Fracking data file not found at {data_path}. "
            "Please ensure the data file is included in the moderndid installation."
        )

    return pl.read_csv(
        data_path,
        schema_overrides={
            "i": pl.Int64,
            "t": pl.Int64,
            "y": pl.Float64,
            "d": pl.Float64,
            "G": pl.Int64,
            "shale_basin1": pl.Int64,
            "G_original": pl.Int64,
        },
    )


def load_cai2016():
    """Load the Cai (2016) agricultural insurance dataset.

    This dataset contains household-level panel data from rural Jiangxi province
    in China (2000-2008), used to study the effects of weather-indexed crop
    insurance on household saving behavior [1]_. The People's Insurance Company of
    China (PICC) introduced crop insurance for tobacco farmers in select counties
    in 2003, creating a triple difference-in-differences (DDD) design with three
    sources of variation: treatment region, household eligibility (tobacco vs
    non-tobacco farmers), and time (pre/post 2003). Ortiz-Villavicencio and
    Sant'Anna [2]_ revisit these households with their triple differences
    estimators.

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

    .. [2] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). Better
        Understanding Triple Differences Estimators. arXiv preprint
        arXiv:2505.09942.
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


def gen_cont_did_data(
    n=500,
    num_time_periods=4,
    num_groups=None,
    p_group=None,
    p_untreated=None,
    dose_linear_effect=0.5,
    dose_quadratic_effect=0,
    seed=42,
):
    """Simulate panel data for difference-in-differences with continuous treatment.

    Each unit falls into a cohort that starts treatment in one of the periods
    after the first or never starts it. Treated units draw a dose uniformly
    between 0 and 1. Once its treatment starts, a unit at dose :math:`d` gains
    ``dose_linear_effect * d + dose_quadratic_effect * d**2`` in every period.
    A unit fixed effect centered on the cohort shifts outcome levels without
    changing their trends.

    You can use this data to check whether an estimator recovers the planted
    effects. See the :ref:`continuous treatment example <example_cont_did>`
    for an analysis of the county employment data from :func:`load_fracking`.

    Parameters
    ----------
    n : int, default=500
        Number of cross-sectional units.
    num_time_periods : int, default=4
        Number of time periods.
    num_groups : int, optional
        Number of timing groups. The never-treated group (G=0) counts as one
        of them. The treated groups start treatment in periods 2, 3, ...,
        num_groups. Must be between 2 and ``num_time_periods``. Defaults to
        ``num_time_periods``.
    p_group : list, optional
        Probabilities for the treated groups, one per treated group.
        Defaults to equal probabilities.
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
        A balanced panel with one row per unit and period.

        - **id**: Unit identifier
        - **time_period**: Time period (1, 2, ..., num_time_periods)
        - **Y**: Outcome variable
        - **G**: Timing group, 0 for never-treated units or the period when treatment starts
        - **D**: The unit's dose, the same in every period before and after treatment starts.
          Never-treated units have a dose of 0.
    """
    rng = np.random.default_rng(seed)

    if num_groups is None:
        num_groups = num_time_periods
    if not 2 <= num_groups <= num_time_periods:
        raise ValueError(
            f"num_groups={num_groups} is not valid. Since it counts the never-treated group and groups that start "
            f"treatment in periods 2 to {num_time_periods}, it must be between 2 and {num_time_periods}."
        )

    time_periods = np.arange(1, num_time_periods + 1)
    groups = np.concatenate(([0], time_periods[1:num_groups]))

    if p_untreated is None:
        p_untreated = 1 / num_groups

    if p_group is None:
        p_group_len = num_groups - 1
        p_group = np.repeat((1 - p_untreated) / p_group_len, p_group_len)
    elif len(p_group) != num_groups - 1:
        raise ValueError(f"p_group needs one probability for each of the {num_groups - 1} treated groups.")

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


def gen_ddd_2periods(
    n,
    dgp_type,
    panel=True,
    random_state=None,
) -> dict:
    """Generate synthetic data for 2-period DDD estimation.

    Four subgroups are created based on treatment and partition status:

    - Subgroup 4: Treated AND Eligible (state=1, partition=1)
    - Subgroup 3: Treated BUT Ineligible (state=1, partition=0)
    - Subgroup 2: Eligible BUT Untreated (state=0, partition=1)
    - Subgroup 1: Untreated AND Ineligible (state=0, partition=0)

    Parameters
    ----------
    n : int, default=5000
        Number of units to simulate. For panel data, this is the total number of
        units observed in both periods. For repeated cross-section data, this is
        the number of observations per period.
    dgp_type : {1, 2, 3, 4}, default=1
        Controls nuisance function specification:

        - 1: Both propensity score and outcome regression use Z (both correct)
        - 2: Propensity score uses X, outcome regression uses Z (OR correct)
        - 3: Propensity score uses Z, outcome regression uses X (PS correct)
        - 4: Both use X (both misspecified when estimating with Z)

    panel : bool, default=True
        If True, generate panel data where each unit is observed in both periods.
        If False, generate repeated cross-section data where different units are
        sampled in each period.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - *data*: pl.DataFrame in long format with columns [id, state, partition,
          time, y, cov1, cov2, cov3, cov4, cluster]
        - *true_att*: True ATT (always 0)
        - *oracle_att*: Oracle ATT from potential outcomes
        - *efficiency_bound*: Theoretical efficiency bound
    """
    if dgp_type not in [1, 2, 3, 4]:
        raise ValueError(f"dgp_type must be 1, 2, 3, or 4, got {dgp_type}")

    rng = np.random.default_rng(random_state)
    att = 0.0

    w1 = np.array([-1.0, 0.5, -0.25, -0.1])
    w2 = np.array([-0.5, 2.0, 0.5, -0.2])
    w3 = np.array([3.0, -1.5, 0.75, -0.3])
    b1 = np.array([27.4, 13.7, 13.7, 13.7])
    b2 = np.array([6.85, 3.43, 3.43, 3.43])

    if dgp_type == 1:
        efficiency_bound = 32.82
    elif dgp_type == 2:
        efficiency_bound = 32.52
    elif dgp_type == 3:
        efficiency_bound = 32.82
    else:
        efficiency_bound = 32.52

    if panel:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x = np.column_stack([x1, x2, x3, x4])
        z = _transform_covariates(x)

        if dgp_type == 1:
            ps_covars, or_covars = z, z
        elif dgp_type == 2:
            ps_covars, or_covars = x, z
        elif dgp_type == 3:
            ps_covars, or_covars = z, x
        else:
            ps_covars, or_covars = x, x

        fps1 = _fps(0.2, w1, ps_covars)
        fps2_val = _fps(0.2, w2, ps_covars)
        fps3 = _fps(0.05, w3, ps_covars)
        freg1 = _freg(b1, or_covars)
        freg0 = _freg(b2, or_covars)

        exp_f1 = np.exp(fps1)
        exp_f2 = np.exp(fps2_val)
        exp_f3 = np.exp(fps3)
        sum_exp_f = exp_f1 + exp_f2 + exp_f3

        p1 = exp_f1 / (1 + sum_exp_f)
        p2 = exp_f2 / (1 + sum_exp_f)
        p4 = 1 / (1 + sum_exp_f)

        u = rng.uniform(size=n)
        pa = np.zeros(n, dtype=int)
        pa[u <= p1] = 1
        pa[(u > p1) & (u <= p1 + p2)] = 2
        pa[(u > p1 + p2) & (u <= 1 - p4)] = 3
        pa[u > 1 - p4] = 4

        state = np.where((pa == 3) | (pa == 4), 1, 0)
        partition = np.where((pa == 2) | (pa == 4), 1, 0)

        unobs_het = state * partition * freg1 + (1 - state) * partition * freg0
        or_lin = state * freg1 + (1 - state) * freg0
        v = rng.normal(loc=unobs_het, scale=1.0)

        y0 = or_lin + v + rng.standard_normal(n)
        y10 = or_lin + v + rng.standard_normal(n) + or_lin
        y11 = or_lin + v + rng.standard_normal(n) + or_lin + att

        treated_eligible = state * partition
        if np.sum(treated_eligible) > 0:
            oracle_att = (np.sum(treated_eligible * y11) - np.sum(treated_eligible * y10)) / np.sum(treated_eligible)
        else:
            oracle_att = np.nan

        y1 = treated_eligible * y11 + (1 - treated_eligible) * y10
        clusters = rng.integers(1, 51, size=n)

        df_t1 = pl.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "state": state,
                "partition": partition,
                "time": np.ones(n, dtype=int),
                "y": y0,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )

        df_t2 = pl.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "state": state,
                "partition": partition,
                "time": np.full(n, 2, dtype=int),
                "y": y1,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )

        df = pl.concat([df_t1, df_t2])
        df = df.sort(["id", "time"])

    else:
        df_list = []
        oracle_att = np.nan
        id_offset = 0

        for t in [1, 2]:
            x1 = rng.standard_normal(n)
            x2 = rng.standard_normal(n)
            x3 = rng.standard_normal(n)
            x4 = rng.standard_normal(n)
            x = np.column_stack([x1, x2, x3, x4])
            z = _transform_covariates(x)

            if dgp_type == 1:
                ps_covars, or_covars = z, z
            elif dgp_type == 2:
                ps_covars, or_covars = x, z
            elif dgp_type == 3:
                ps_covars, or_covars = z, x
            else:
                ps_covars, or_covars = x, x

            fps1 = _fps(0.2, w1, ps_covars)
            fps2_val = _fps(0.2, w2, ps_covars)
            fps3 = _fps(0.05, w3, ps_covars)
            freg1 = _freg(b1, or_covars)
            freg0 = _freg(b2, or_covars)

            exp_f1 = np.exp(fps1)
            exp_f2 = np.exp(fps2_val)
            exp_f3 = np.exp(fps3)
            sum_exp_f = exp_f1 + exp_f2 + exp_f3

            p1 = exp_f1 / (1 + sum_exp_f)
            p2 = exp_f2 / (1 + sum_exp_f)
            p4 = 1 / (1 + sum_exp_f)

            u = rng.uniform(size=n)
            pa = np.zeros(n, dtype=int)
            pa[u <= p1] = 1
            pa[(u > p1) & (u <= p1 + p2)] = 2
            pa[(u > p1 + p2) & (u <= 1 - p4)] = 3
            pa[u > 1 - p4] = 4

            state = np.where((pa == 3) | (pa == 4), 1, 0)
            partition = np.where((pa == 2) | (pa == 4), 1, 0)

            unobs_het = state * partition * freg1 + (1 - state) * partition * freg0
            or_lin = state * freg1 + (1 - state) * freg0
            v = rng.normal(loc=unobs_het, scale=1.0)

            if t == 1:
                y = or_lin + v + rng.standard_normal(n)
            else:
                treated_eligible = state * partition
                y10 = or_lin + v + rng.standard_normal(n) + or_lin
                y11 = or_lin + v + rng.standard_normal(n) + or_lin + att
                y = treated_eligible * y11 + (1 - treated_eligible) * y10

                if np.sum(treated_eligible) > 0:
                    oracle_att = (np.sum(treated_eligible * y11) - np.sum(treated_eligible * y10)) / np.sum(
                        treated_eligible
                    )

            clusters = rng.integers(1, 51, size=n)

            df_t = pl.DataFrame(
                {
                    "id": np.arange(id_offset + 1, id_offset + n + 1),
                    "state": state,
                    "partition": partition,
                    "time": np.full(n, t, dtype=int),
                    "y": y,
                    "cov1": z[:, 0],
                    "cov2": z[:, 1],
                    "cov3": z[:, 2],
                    "cov4": z[:, 3],
                    "cluster": clusters,
                }
            )
            df_list.append(df_t)
            id_offset += n

        df = pl.concat(df_list)

    return {
        "data": df,
        "true_att": att,
        "oracle_att": oracle_att,
        "efficiency_bound": efficiency_bound,
    }


def gen_ddd_mult_periods(
    n: int,
    dgp_type: int = 1,
    panel: bool = True,
    random_state=None,
) -> dict:
    """Generate data with staggered treatment adoption for multi-period DDD.

    Generates data where units adopt treatment at different times across
    three periods. The DGP has 3 timing groups (cohort=0 never treated, 2=treated
    at period 2, 3=treated at period 3) and two partitions (eligible/ineligible).

    Parameters
    ----------
    n : int
        Number of units to simulate. For panel data, this is the total number of
        units observed in all periods. For repeated cross-section data, this is
        the number of observations per period.
    dgp_type : {1, 2, 3, 4}, default=1
        Controls nuisance function specification:

        - 1: Both propensity score and outcome regression use Z (both correct)
        - 2: Propensity score uses X, outcome regression uses Z (OR correct)
        - 3: Propensity score uses Z, outcome regression uses X (PS correct)
        - 4: Both use X (both misspecified when estimating with Z)

    panel : bool, default=True
        If True, generate panel data where each unit is observed in all periods.
        If False, generate repeated cross-section data where different units are
        sampled in each period.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - *data*: pl.DataFrame in long format with columns [id, group, partition,
          time, y, cov1, cov2, cov3, cov4, cluster]
        - *data_wide*: pl.DataFrame in wide format with one row per unit (only for panel=True)
        - *es_0_oracle*: Oracle event-study parameter at event time 0
        - *prob_g2_p1*: Proportion of units with cohort=2 and eligibility
        - *prob_g3_p1*: Proportion of units with cohort=3 and eligibility
    """
    if dgp_type not in [1, 2, 3, 4]:
        raise ValueError(f"dgp_type must be 1, 2, 3, or 4, got {dgp_type}")

    rng = np.random.default_rng(random_state)
    xsi_ps = 0.4

    w1 = np.array([-1.0, 0.5, -0.25, -0.1])
    w2 = np.array([-0.5, 1.0, -0.1, -0.25])
    w3 = np.array([-0.25, 0.1, -1.0, -0.1])
    b1 = np.array([27.4, 13.7, 13.7, 13.7])

    index_att_g2 = 10
    index_att_g3 = 25

    if panel:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x = np.column_stack([x1, x2, x3, x4])
        z = _transform_covariates(x)

        if dgp_type == 1:
            ps_covars, or_covars = z, z
        elif dgp_type == 2:
            ps_covars, or_covars = x, z
        elif dgp_type == 3:
            ps_covars, or_covars = z, x
        else:
            ps_covars, or_covars = x, x

        pi_2a = np.exp(_fps2(xsi_ps, w1, ps_covars, 1.25))
        pi_2b = np.exp(_fps2(-xsi_ps, w1, ps_covars, -0.5))
        pi_3a = np.exp(_fps2(xsi_ps, w2, ps_covars, 2.0))
        pi_3b = np.exp(_fps2(-xsi_ps, w2, ps_covars, -1.25))
        pi_0a = np.exp(_fps2(xsi_ps, w3, ps_covars, -0.5))

        sum_pi = 1 + pi_2a + pi_2b + pi_3a + pi_3b + pi_0a
        pi_2a = pi_2a / sum_pi
        pi_2b = pi_2b / sum_pi
        pi_3a = pi_3a / sum_pi
        pi_3b = pi_3b / sum_pi
        pi_0a = pi_0a / sum_pi
        pi_0b = 1 - (pi_2a + pi_2b + pi_3a + pi_3b + pi_0a)

        probs_pscore = np.column_stack([pi_2a, pi_2b, pi_3a, pi_3b, pi_0a, pi_0b])
        cum_probs = np.cumsum(probs_pscore, axis=1)
        u = rng.uniform(size=n)
        group_types = (u[:, None] >= cum_probs).sum(axis=1) + 1

        partition = np.isin(group_types, [1, 3, 5]).astype(int)
        cohort = np.where(
            np.isin(group_types, [1, 2]),
            2,
            np.where(np.isin(group_types, [3, 4]), 3, 0),
        )

        index_lin = _freg(b1, or_covars)
        index_partition = partition * index_lin
        index_unobs_het = cohort * index_lin + index_partition
        index_trend = index_lin

        v = rng.normal(loc=index_unobs_het, scale=1.0)
        index_pt_violation = v / 10

        baseline_t1 = index_lin + index_partition + v
        y_t1 = baseline_t1 + rng.standard_normal(n)

        baseline_t2 = baseline_t1 + index_pt_violation + index_trend
        y_t2_never = baseline_t2 + rng.standard_normal(n)
        y_t2_g2 = baseline_t2 + rng.standard_normal(n) + index_att_g2 * partition

        baseline_t3 = baseline_t1 + 2 * index_trend + 2 * index_pt_violation
        y_t3_never = baseline_t3 + rng.standard_normal(n)
        y_t3_g2 = baseline_t3 + rng.standard_normal(n) + 2 * index_att_g2 * partition
        y_t3_g3 = baseline_t3 + rng.standard_normal(n) + index_att_g3 * partition

        y_t2 = np.where((cohort == 2) & (partition == 1), y_t2_g2, y_t2_never)
        y_t3 = np.where(
            (cohort == 2) & (partition == 1),
            y_t3_g2,
            np.where((cohort == 3) & (partition == 1), y_t3_g3, y_t3_never),
        )

        mask_g2_p1 = group_types == 1
        mask_g3_p1 = group_types == 3

        if np.sum(mask_g2_p1) > 0:
            att_g2_t2_unf = (np.sum(mask_g2_p1 * y_t2_g2) - np.sum(mask_g2_p1 * y_t2_never)) / np.sum(mask_g2_p1)
        else:
            att_g2_t2_unf = np.nan

        if np.sum(mask_g3_p1) > 0:
            att_g3_t3_unf = (np.sum(mask_g3_p1 * y_t3_g3) - np.sum(mask_g3_p1 * y_t3_never)) / np.sum(mask_g3_p1)
        else:
            att_g3_t3_unf = np.nan

        prob_g2_p1 = np.mean(pi_2a / (pi_2a + pi_3a))
        prob_g3_p1 = np.mean(pi_3a / (pi_2a + pi_3a))
        es_0_oracle = att_g2_t2_unf * prob_g2_p1 + att_g3_t3_unf * prob_g3_p1

        clusters = rng.integers(1, 51, size=n)

        data_wide = pl.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "group": cohort,
                "partition": partition,
                "y_t1": y_t1,
                "y_t2": y_t2,
                "y_t3": y_t3,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )

        df_list = []
        for t, y_vals in enumerate([y_t1, y_t2, y_t3], start=1):
            df_t = pl.DataFrame(
                {
                    "id": np.arange(1, n + 1),
                    "group": cohort,
                    "partition": partition,
                    "time": np.full(n, t, dtype=int),
                    "y": y_vals,
                    "cov1": z[:, 0],
                    "cov2": z[:, 1],
                    "cov3": z[:, 2],
                    "cov4": z[:, 3],
                    "cluster": clusters,
                }
            )
            df_list.append(df_t)

        data = pl.concat(df_list)
        data = data.sort(["id", "time"])

        return {
            "data": data,
            "data_wide": data_wide,
            "es_0_oracle": es_0_oracle,
            "prob_g2_p1": prob_g2_p1,
            "prob_g3_p1": prob_g3_p1,
        }

    df_list = []
    id_offset = 0
    all_pi_2a = []
    all_pi_3a = []

    for t in [1, 2, 3]:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x = np.column_stack([x1, x2, x3, x4])
        z = _transform_covariates(x)

        if dgp_type == 1:
            ps_covars, or_covars = z, z
        elif dgp_type == 2:
            ps_covars, or_covars = x, z
        elif dgp_type == 3:
            ps_covars, or_covars = z, x
        else:
            ps_covars, or_covars = x, x

        pi_2a = np.exp(_fps2(xsi_ps, w1, ps_covars, 1.25))
        pi_2b = np.exp(_fps2(-xsi_ps, w1, ps_covars, -0.5))
        pi_3a = np.exp(_fps2(xsi_ps, w2, ps_covars, 2.0))
        pi_3b = np.exp(_fps2(-xsi_ps, w2, ps_covars, -1.25))
        pi_0a = np.exp(_fps2(xsi_ps, w3, ps_covars, -0.5))

        sum_pi = 1 + pi_2a + pi_2b + pi_3a + pi_3b + pi_0a
        pi_2a = pi_2a / sum_pi
        pi_2b = pi_2b / sum_pi
        pi_3a = pi_3a / sum_pi
        pi_3b = pi_3b / sum_pi
        pi_0a = pi_0a / sum_pi
        pi_0b = 1 - (pi_2a + pi_2b + pi_3a + pi_3b + pi_0a)

        all_pi_2a.extend(pi_2a)
        all_pi_3a.extend(pi_3a)

        probs_pscore = np.column_stack([pi_2a, pi_2b, pi_3a, pi_3b, pi_0a, pi_0b])
        cum_probs = np.cumsum(probs_pscore, axis=1)
        u = rng.uniform(size=n)
        group_types = (u[:, None] >= cum_probs).sum(axis=1) + 1

        partition = np.isin(group_types, [1, 3, 5]).astype(int)
        cohort = np.where(
            np.isin(group_types, [1, 2]),
            2,
            np.where(np.isin(group_types, [3, 4]), 3, 0),
        )

        index_lin = _freg(b1, or_covars)
        index_partition = partition * index_lin
        index_unobs_het = cohort * index_lin + index_partition
        index_trend = index_lin

        v = rng.normal(loc=index_unobs_het, scale=1.0)
        index_pt_violation = v / 10

        baseline = index_lin + index_partition + v

        if t == 1:
            y = baseline + rng.standard_normal(n)
        elif t == 2:
            baseline_t2 = baseline + index_pt_violation + index_trend
            y_never = baseline_t2 + rng.standard_normal(n)
            y_treated = baseline_t2 + rng.standard_normal(n) + index_att_g2 * partition
            y = np.where((cohort == 2) & (partition == 1), y_treated, y_never)
        else:
            baseline_t3 = baseline + 2 * index_trend + 2 * index_pt_violation
            y_never = baseline_t3 + rng.standard_normal(n)
            y_g2 = baseline_t3 + rng.standard_normal(n) + 2 * index_att_g2 * partition
            y_g3 = baseline_t3 + rng.standard_normal(n) + index_att_g3 * partition
            y = np.where(
                (cohort == 2) & (partition == 1),
                y_g2,
                np.where((cohort == 3) & (partition == 1), y_g3, y_never),
            )

        clusters = rng.integers(1, 51, size=n)

        df_t = pl.DataFrame(
            {
                "id": np.arange(id_offset + 1, id_offset + n + 1),
                "group": cohort,
                "partition": partition,
                "time": np.full(n, t, dtype=int),
                "y": y,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )
        df_list.append(df_t)
        id_offset += n

    data = pl.concat(df_list)

    all_pi_2a = np.array(all_pi_2a)
    all_pi_3a = np.array(all_pi_3a)
    prob_g2_p1 = np.mean(all_pi_2a / (all_pi_2a + all_pi_3a))
    prob_g3_p1 = np.mean(all_pi_3a / (all_pi_2a + all_pi_3a))

    return {
        "data": data,
        "data_wide": None,
        "es_0_oracle": np.nan,
        "prob_g2_p1": prob_g2_p1,
        "prob_g3_p1": prob_g3_p1,
    }


def gen_simple_ddd_data(
    n,
    att,
    random_state=None,
) -> pl.DataFrame:
    """Generate simple DDD panel data with a known treatment effect.

    Parameters
    ----------
    n : int, default=500
        Number of units to simulate.
    att : float, default=5.0
        True average treatment effect on the treated.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:

        - *id*: Unit identifier
        - *state*: Treatment indicator (1=treated, 0=control)
        - *partition*: Eligibility indicator (1=eligible, 0=ineligible)
        - *time*: Time period (1=pre, 2=post)
        - *y*: Outcome variable
        - *x1*, *x2*: Covariates
    """
    rng = np.random.default_rng(random_state)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    state = rng.binomial(1, 0.5, n)
    partition = rng.binomial(1, 0.5, n)
    alpha_i = rng.standard_normal(n)

    y0 = 2 + 5 * state - 2 * partition + 0.5 * x1 + 0.3 * x2 + 4 * state * partition + alpha_i + rng.standard_normal(n)

    y1 = (
        2
        + 5 * state
        - 2 * partition
        + 3
        + 0.5 * x1
        + 0.3 * x2
        + 4 * state * partition
        + 2 * state
        + 3 * partition
        + att * state * partition
        + alpha_i
        + rng.standard_normal(n)
    )

    df_t1 = pl.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": np.ones(n, dtype=int),
            "y": y0,
            "x1": x1,
            "x2": x2,
        }
    )

    df_t2 = pl.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": np.full(n, 2, dtype=int),
            "y": y1,
            "x1": x1,
            "x2": x2,
        }
    )

    df = pl.concat([df_t1, df_t2])
    df = df.sort(["id", "time"])

    return df


def gen_ddd_scalable(
    n: int,
    dgp_type: int = 1,
    n_periods: int = 10,
    n_cohorts: int = 8,
    n_covariates: int = 20,
    att_base: float = 10.0,
    panel: bool = True,
    random_state=None,
) -> dict:
    """Generate configurable staggered DDD data for stress-testing.

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
        ``att_base * g * (t - g + 1) * partition``.
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
          partition, time, y, cov1..covK, cluster]
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
    n_free = 2 * (n_cohorts + 1) - 1
    coef_rng = np.random.default_rng(12345)
    ws, psis, cs = _generate_ps_coefficients(coef_rng, n_free)

    if panel:
        x_first4 = rng.standard_normal((n, 4))
        z_first4 = _transform_covariates(x_first4)
        x_extra = rng.standard_normal((n, n_covariates - 4)) if n_covariates > 4 else None

        ps_covars, or_covars = _select_covars(dgp_type, x_first4, z_first4)
        cohort, partition = _assign_cohort_partition(
            rng,
            n,
            n_free,
            ws,
            psis,
            cs,
            ps_covars,
            cohort_values,
            xsi_ps,
        )

        index_lin = _freg(b1, or_covars)
        index_partition = partition * index_lin
        index_unobs_het = cohort * index_lin + index_partition
        index_trend = index_lin

        v = rng.normal(loc=index_unobs_het, scale=1.0)
        index_pt_violation = v / 10
        baseline = index_lin + index_partition + v

        clusters = rng.integers(1, 51, size=n)
        cov_dict = _build_cov_dict(z_first4, x_extra, n_covariates)

        y_all = {}
        df_list = []
        for t in range(1, n_periods + 1):
            y_t = _compute_scalable_outcome(
                t,
                baseline,
                index_trend,
                index_pt_violation,
                cohort,
                partition,
                cohort_values,
                att_base,
                n,
                rng,
            )
            y_all[t] = y_t
            row_dict = {
                "id": np.arange(1, n + 1),
                "group": cohort,
                "partition": partition,
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
                "partition": partition,
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
            cohort, partition = _assign_cohort_partition(
                rng,
                n,
                n_free,
                ws,
                psis,
                cs,
                ps_covars,
                cohort_values,
                xsi_ps,
            )

            index_lin = _freg(b1, or_covars)
            index_partition = partition * index_lin
            index_unobs_het = cohort * index_lin + index_partition
            index_trend = index_lin

            v = rng.normal(loc=index_unobs_het, scale=1.0)
            index_pt_violation = v / 10
            baseline = index_lin + index_partition + v

            y_t = _compute_scalable_outcome(
                t,
                baseline,
                index_trend,
                index_pt_violation,
                cohort,
                partition,
                cohort_values,
                att_base,
                n,
                rng,
            )

            clusters = rng.integers(1, 51, size=n)
            cov_dict = _build_cov_dict(z_first4, x_extra, n_covariates)

            row_dict = {
                "id": np.arange(id_offset + 1, id_offset + n + 1),
                "group": cohort,
                "partition": partition,
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


def simulate_cont_did_data(*args, **kwargs):
    """Call :func:`gen_cont_did_data` instead (deprecated)."""
    warnings.warn(
        "simulate_cont_did_data is deprecated, use gen_cont_did_data instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return gen_cont_did_data(*args, **kwargs)


def generate_simple_ddd_data(*args, **kwargs):
    """Call :func:`gen_simple_ddd_data` instead (deprecated)."""
    warnings.warn(
        "generate_simple_ddd_data is deprecated, use gen_simple_ddd_data instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return gen_simple_ddd_data(*args, **kwargs)


def gen_dgp_2periods(*args, **kwargs):
    """Call :func:`gen_ddd_2periods` instead (deprecated)."""
    warnings.warn(
        "gen_dgp_2periods is deprecated, use gen_ddd_2periods instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return gen_ddd_2periods(*args, **kwargs)


def gen_dgp_mult_periods(*args, **kwargs):
    """Call :func:`gen_ddd_mult_periods` instead (deprecated)."""
    warnings.warn(
        "gen_dgp_mult_periods is deprecated, use gen_ddd_mult_periods instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return gen_ddd_mult_periods(*args, **kwargs)


def gen_dgp_scalable(*args, **kwargs):
    """Call :func:`gen_ddd_scalable` instead (deprecated)."""
    warnings.warn(
        "gen_dgp_scalable is deprecated, use gen_ddd_scalable instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return gen_ddd_scalable(*args, **kwargs)
