"""Heterogeneity analysis (BLP and CLAN) for ML-based DiD."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from formulaic import Formula
from scipy import stats

from .container import BLPResult, CLANResult


def het_prep(data, dynamic_cates_df, *, idname, tname, gname):
    r"""Merge dynamic per-unit CATTs back to the unit-level panel by event time.

    Produces the long-format panel frame consumed by :func:`blp_eventtimes`,
    :func:`clan_glhtest`, and :func:`clan_ttest`. Restricts to treated units
    (``gname > 0``), computes :math:`e = t - g` per row, and left-joins the
    output of :func:`dynamic_cates` so each unit-period row carries its
    aggregated CATT (or DR score) at the matching event time. Pre-treatment
    rows where ``e < 0`` carry the corresponding pre-treatment aggregate;
    rows with no matching event time receive a null.

    Parameters
    ----------
    data : DataFrame
        Original long-format panel passed to :func:`didml`. Accepts polars
        and pandas DataFrames.
    dynamic_cates_df : polars.DataFrame
        Output of :func:`dynamic_cates`. Must have columns ``id``,
        ``event_time``, and either ``cate`` or ``score``.
    idname : str
        Name of the unit identifier column in ``data``.
    tname : str
        Name of the time period column in ``data``.
    gname : str
        Name of the cohort first-treatment column in ``data``.

    Returns
    -------
    polars.DataFrame
        Long-format frame keyed by ``(idname, tname)`` with the original
        covariates plus:

        - **event_time**: Integer :math:`e = t - g`
        - **dynamic_cate**: Aggregated per-unit CATT at this event time, joined from
          ``dynamic_cates_df``; present when the input holds CATTs
        - **dynamic_score**: Aggregated per-unit DR score at this event time, joined from
          ``dynamic_cates_df``; present when the input holds scores
        - **e{k}**: 0/1 dummy columns, one per distinct event time, named ``e0``, ``e1``,
          and so on, with negative event times written with an underscore as ``e_1``, ``e_2``
    """
    if isinstance(data, pl.DataFrame):
        df = data
    else:
        df = pl.from_pandas(data) if hasattr(data, "to_pandas") or hasattr(data, "columns") else pl.DataFrame(data)

    df = df.filter(pl.col(gname) > 0).with_columns(
        (pl.col(tname) - pl.col(gname)).alias("event_time"),
        pl.col(tname).alias("period"),
    )

    value_col = "cate" if "cate" in dynamic_cates_df.columns else "score"
    output_value_col = "dynamic_cate" if value_col == "cate" else "dynamic_score"

    cates_for_join = dynamic_cates_df.rename({"id": idname, value_col: output_value_col})
    merged = df.join(cates_for_join, on=[idname, "event_time"], how="left")

    unique_events = sorted(merged["event_time"].unique().to_list())
    dummy_cols = []

    for e in unique_events:
        name = f"e{e}" if e >= 0 else f"e_{abs(e)}"
        dummy_cols.append((pl.col("event_time") == e).cast(pl.Int64).alias(name))

    if dummy_cols:
        merged = merged.with_columns(dummy_cols)

    return merged


def blp_eventtimes(het_data, *, n_periods, rhs_formula=None, value_col=None):
    r"""Fit Best Linear Predictor regressions of CATTs on covariates by event time.

    Implements the per-event-time BLP analysis of [1]_, which regresses the
    per-unit dynamic CATTs on observed covariates by ordinary least squares
    separately within each post-treatment exposure horizon. For each event
    time :math:`e \in \{0, 1, \ldots, E\}`, where :math:`E` is set by
    ``n_periods``, the paper's specification is

    .. math::

        \hat{\tau}_i(e) = \alpha_e + X_i^\top \beta_e + \varepsilon_i.

    Significant components of :math:`\beta_e` identify covariates that
    drive treatment effect heterogeneity at exposure horizon :math:`e`.

    This implementation augments each within-event-time regression with
    calendar-period dummies :math:`\lambda_t`, because cohorts treated at
    different times contribute different calendar periods at the same
    horizon. The dummies are included only for event times strictly below
    ``n_periods - 1``.

    When outcomes are noisy, [1]_ recommends regressing the doubly robust
    scores rather than the raw CATTs, which corresponds to
    ``value_col="dynamic_score"``.

    Parameters
    ----------
    het_data : polars.DataFrame
        Output of :func:`het_prep`. Must have ``event_time``, the value
        column (``dynamic_cate`` or ``dynamic_score``), and the covariates
        named in ``rhs_formula``.
    n_periods : int
        Highest event time at which to fit the regression. Regressions are
        fit for every event time from 0 through ``n_periods``.
    rhs_formula : str, optional
        Right-hand-side covariate formula in formulaic syntax, e.g.
        ``"cov1 + cov2 + cov3"``. ``None`` fits the intercept-only model.
    value_col : str, optional
        Column to use as the dependent variable. Defaults to
        ``dynamic_cate`` if present, else ``dynamic_score``.

    Returns
    -------
    BLPResult
        Object containing per-event-time BLP regression results:

        - **event_times**: Array ``[0, 1, ..., n_periods]`` of event times at which regressions were fit
        - **coefs**: Per-covariate arrays of coefficients indexed by event time
        - **ses**: Per-covariate arrays of standard errors indexed by event time
        - **pvalues**: Per-covariate arrays of p-values indexed by event time
        - **rhs_formula**: Right-hand-side formula used for the regression

    References
    ----------

    .. [1] Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023).
           "Machine learning for staggered difference-in-differences and
           dynamic treatment effect heterogeneity." arXiv:2310.11962.
           https://arxiv.org/abs/2310.11962
    """
    if value_col is None:
        value_col = "dynamic_cate" if "dynamic_cate" in het_data.columns else "dynamic_score"

    event_times = np.arange(0, n_periods + 1, dtype=int)
    coefs = {}
    ses = {}
    pvalues = {}

    for k, e in enumerate(event_times):
        subset = het_data.filter(pl.col("event_time") == int(e)).drop_nulls(subset=[value_col])

        if subset.is_empty():
            continue

        df_pd = subset.to_pandas()
        y = df_pd[value_col].to_numpy(dtype=float)

        if rhs_formula is None:
            x_block = np.ones((len(df_pd), 1))
            term_names = ["Intercept"]
        else:
            spec = Formula(f"~ 1 + {rhs_formula}").get_model_matrix(df_pd)
            x_block = np.asarray(spec, dtype=float)
            term_names = list(spec.columns)

        period_present = "period" in df_pd.columns and df_pd["period"].nunique() > 1
        if e < n_periods - 1 and period_present:
            period_dummies = pl.DataFrame({"period": df_pd["period"]}).to_dummies("period", drop_first=True).to_numpy()
            design = np.column_stack([x_block, period_dummies])
        else:
            design = x_block

        ols = sm.OLS(y, design).fit()

        for j, name in enumerate(term_names):
            coefs.setdefault(name, np.full(len(event_times), np.nan))[k] = ols.params[j]
            ses.setdefault(name, np.full(len(event_times), np.nan))[k] = ols.bse[j]
            pvalues.setdefault(name, np.full(len(event_times), np.nan))[k] = ols.pvalues[j]

    return BLPResult(
        event_times=event_times,
        coefs=coefs,
        ses=ses,
        pvalues=pvalues,
        rhs_formula=rhs_formula or "",
    )


def clan_ttest(het_data, *, affected, threshold=0.2, value_col=None):
    r"""Run Welch t-tests of covariate means in the top vs. bottom CATT-quantile groups.

    Implements the simple variant of Classification Analysis (CLAN) from
    [1]_. For each event time, defines the high-effect group as units with
    dynamic CATT above the :math:`(1 - \mathrm{threshold})` quantile and
    the low-effect group as units below the ``threshold`` quantile, then
    runs a Welch two-sample t-test on each affected covariate. Covariates
    with significantly different means across the two groups identify
    candidate drivers of treatment effect heterogeneity.

    Parameters
    ----------
    het_data : polars.DataFrame
        Output of :func:`het_prep`.
    affected : list[str]
        Covariate names to test.
    threshold : float, default=0.2
        Tail proportion used to define top and bottom groups. ``0.2``
        compares the top and bottom quintiles.
    value_col : str, optional
        Column to use for ranking. Defaults to ``dynamic_cate`` if present,
        else ``dynamic_score``.

    Returns
    -------
    list[CLANResult]
        One :class:`CLANResult` per event time, in increasing order. Each
        result has ``test_type='t'``.

    References
    ----------

    .. [1] Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023).
           "Machine learning for staggered difference-in-differences and
           dynamic treatment effect heterogeneity." arXiv:2310.11962.
           https://arxiv.org/abs/2310.11962
    """
    if value_col is None:
        value_col = "dynamic_cate" if "dynamic_cate" in het_data.columns else "dynamic_score"

    event_times = sorted(het_data["event_time"].unique().to_list())
    event_times = [e for e in event_times if e >= 0]

    results = []

    for e in event_times:
        subset = het_data.filter(pl.col("event_time") == int(e)).drop_nulls(subset=[value_col])

        if subset.is_empty():
            continue

        df_pd = subset.to_pandas()
        cates = df_pd[value_col].to_numpy(dtype=float)
        high_q = np.quantile(cates, 1 - threshold)
        low_q = np.quantile(cates, threshold)
        high_mask = cates > high_q
        low_mask = cates < low_q

        n_aff = len(affected)
        high_means = np.zeros(n_aff)
        low_means = np.zeros(n_aff)
        diffs = np.zeros(n_aff)
        pvals = np.zeros(n_aff)

        for j, var in enumerate(affected):
            x = df_pd[var].to_numpy(dtype=float)
            high_x = x[high_mask]
            low_x = x[low_mask]
            high_means[j] = np.nanmean(high_x)
            low_means[j] = np.nanmean(low_x)
            diffs[j] = high_means[j] - low_means[j]
            pvals[j] = stats.ttest_ind(high_x, low_x, equal_var=False, nan_policy="omit").pvalue

        results.append(
            CLANResult(
                affected=list(affected),
                threshold=float(threshold),
                high_means=high_means,
                low_means=low_means,
                diffs=diffs,
                pvalues=pvals,
                test_type="t",
            )
        )

    return results


def clan_glhtest(het_data, *, affected, threshold=0.2, alpha=0.05, value_col=None):
    r"""Test equality of high and low CATT-quantile group means with a general linear hypothesis.

    Implements the GLH variant of Classification Analysis (CLAN) from [1]_,
    adapted from [2]_. For each event time and each affected covariate, fits

    .. math::

        \mathrm{covariate}_i = \beta_h H_i + \beta_l L_i + \varepsilon_i,

    where :math:`H_i` and :math:`L_i` are 0/1 indicators for membership in
    the top-:math:`\mathrm{threshold}` and bottom-:math:`\mathrm{threshold}`
    CATT-quantile groups respectively. The high-minus-low contrast
    :math:`\hat\beta_h - \hat\beta_l` is tested for equality to zero with a
    general linear hypothesis test on the OLS coefficients, implementing
    the most-versus-least-affected comparison of [2]_.

    Parameters
    ----------
    het_data : polars.DataFrame
        Output of :func:`het_prep`.
    affected : list[str]
        Covariate names to test.
    threshold : float, default=0.2
        Tail proportion used to define top and bottom groups.
    alpha : float, default=0.05
        Significance level used implicitly by downstream summaries.
    value_col : str, optional
        Column to use for ranking. Defaults to ``dynamic_cate`` if present,
        else ``dynamic_score``.

    Returns
    -------
    list[CLANResult]
        One :class:`CLANResult` per event time, in increasing order. Each
        result has ``test_type='glh'``.

    References
    ----------

    .. [1] Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023).
           "Machine learning for staggered difference-in-differences and
           dynamic treatment effect heterogeneity." arXiv:2310.11962.
           https://arxiv.org/abs/2310.11962

    .. [2] Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I.
           (2018). "Generic machine learning inference on heterogeneous
           treatment effects in randomized experiments." NBER Working
           Paper 24678. https://doi.org/10.3386/w24678
    """
    if value_col is None:
        value_col = "dynamic_cate" if "dynamic_cate" in het_data.columns else "dynamic_score"

    event_times = sorted(het_data["event_time"].unique().to_list())
    event_times = [e for e in event_times if e >= 0]

    results = []

    for e in event_times:
        subset = het_data.filter(pl.col("event_time") == int(e)).drop_nulls(subset=[value_col])

        if subset.is_empty():
            continue

        df_pd = subset.to_pandas()
        cates = df_pd[value_col].to_numpy(dtype=float)
        high_q = np.quantile(cates, 1 - threshold)
        low_q = np.quantile(cates, threshold)
        h_ind = (cates > high_q).astype(float)
        l_ind = (cates < low_q).astype(float)
        keep = (h_ind == 1) | (l_ind == 1)

        n_aff = len(affected)
        high_means = np.full(n_aff, np.nan)
        low_means = np.full(n_aff, np.nan)
        diffs = np.full(n_aff, np.nan)
        pvals = np.full(n_aff, np.nan)

        if keep.sum() == 0 or h_ind[keep].sum() == 0 or l_ind[keep].sum() == 0:
            results.append(
                CLANResult(
                    affected=list(affected),
                    threshold=float(threshold),
                    high_means=high_means,
                    low_means=low_means,
                    diffs=diffs,
                    pvalues=pvals,
                    test_type="glh",
                )
            )
            continue

        for j, var in enumerate(affected):
            y = df_pd[var].to_numpy(dtype=float)[keep]
            h = h_ind[keep]
            l_ = l_ind[keep]
            design = np.column_stack([h, l_])
            ols = sm.OLS(y, design).fit()
            high_means[j] = ols.params[0]
            low_means[j] = ols.params[1]
            diffs[j] = high_means[j] - low_means[j]
            contrast = np.array([[1.0, -1.0]])
            test = ols.t_test(contrast)
            pvals[j] = float(test.pvalue)

        results.append(
            CLANResult(
                affected=list(affected),
                threshold=float(threshold),
                high_means=high_means,
                low_means=low_means,
                diffs=diffs,
                pvalues=pvals,
                test_type="glh",
            )
        )

    _ = alpha
    return results
