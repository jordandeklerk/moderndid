"""ETWFE estimation via saturated cohort-time interactions.

Inspired by https://github.com/armandkapllani/etwfe/
"""

import warnings

import numpy as np

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess.config import EtwfeConfig
from moderndid.core.preprocess.validators import check_columns

from .compute import (
    _cell_column,
    _format_cells,
    build_etwfe_formula,
    clean_etwfe_data,
    prepare_etwfe_data,
    run_etwfe_regression,
    set_references,
    treatment_cells,
)
from .container import EtwfeResult


def etwfe(
    data,
    yname,
    tname,
    gname,
    idname=None,
    xformla=None,
    xvar=None,
    tref=None,
    gref=None,
    cgroup="notyet",
    fe="vs",
    family=None,
    weightsname=None,
    vcov=None,
    alp=0.05,
    backend=None,
):
    r"""Estimate the Extended Two-Way Fixed Effects model.

    Implements the extended two-way fixed effects (ETWFE) estimator for
    difference-in-differences with staggered adoption and heterogeneous
    treatment effects [1]_ [2]_. Rather than discarding the two-way fixed
    effects regression, the estimator adds an indicator for every treated
    cohort-time cell. The coefficient on each indicator is the average
    treatment effect on the treated for that cohort and period. Pooled least
    squares on this saturated regression gives the same estimates as cohort
    imputation (Proposition 5.2 in [1]_).

    Use :func:`~moderndid.etwfe.emfx.emfx` to average the cell estimates into
    overall, group, calendar, or event-study summaries.

    Rows with a null, NaN, or infinite value in the period, cohort, unit,
    control, moderator, weight, or cluster variable leave the sample with a
    warning before the controls are demeaned. Since an infinite cohort marks a
    never-treated unit, it stays. The weights that remain must be non-negative
    with a positive mean.

    A column that the call names may not take the name of a column the
    regression adds. These are ``_g``, ``_t``, ``_Dtreat``, a name that starts
    with ``__etwfe_``, a control's name followed by ``_dm``, and a name that
    ends in ``_xdm`` and starts with the moderator's name or with ``_t``.

    Since no untreated period identifies their effects, units already treated in
    the first period leave with a warning as well. So do the units of any other
    treated cohort with no untreated row in the sample. Under
    ``cgroup="never"`` that row must lie in period :math:`g - 1`.

    The covariate in ``xvar`` is demeaned within each cohort-time cell before it
    interacts with the treatment cells. Because each category indicator of a
    string covariate gets its own year terms, the estimates do not depend on
    which category comes first. The controls in ``xformla`` already interact
    with every cell. A covariate that is constant within the cells, or that
    varies there only as a linear function of those controls, therefore adds no
    terms and draws a warning.

    See the :ref:`extended TWFE example <example_etwfe>` for ``etwfe`` on the
    minimum wage data with not-yet-treated and never-treated controls, a
    covariate, and standard errors clustered by county and by state.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome variable.
    tname : str
        The name of the column containing the time periods.
    gname : str
        The name of the column that holds the first period in which each unit
        is treated. Never-treated units have 0, infinity, or a value after the
        last period.
    idname : str or None, default=None
        The individual (cross-sectional unit) id name. When provided, linear
        models absorb unit fixed effects and the default standard errors
        cluster by unit.
    xformla : str or None, default=None
        A formula for the controls, such as ``"~ x1 + x2"``. Each term is a
        column name. A name with spaces or symbols goes in backticks, as in
        ``"~ `log pop` + x"``. Since every term must be a column, a
        transformation such as ``I(x**2)`` or an interaction such as ``x1:x2``
        raises an error and belongs in the data as a column of its own.
    xvar : str or None, default=None
        Name of a covariate to interact with the treatment cells for
        heterogeneous treatment effects. A string covariate enters as an
        indicator for each category but the first.
    tref : numeric or None, default=None
        Reference period, a value of ``tname``. Defaults to the first period.
    gref : numeric or None, default=None
        Reference cohort. Any never-treated code, such as 0, refers to every
        never-treated unit. Another value must be a cohort in ``gname`` first
        treated after the first period. Defaults to the never-treated units.
        Without them, ``cgroup="notyet"`` uses the latest-treated cohort.
        ``cgroup="never"`` then raises an error. Under ``cgroup="notyet"``, any
        treated reference cohort drops every row from its first treated period
        on.
    cgroup : {'notyet', 'never'}, default='notyet'
        Control group. ``"notyet"`` compares the treated cells with the
        never-treated units, the reference cohort, and the rows of other
        cohorts before they are treated. ``"never"`` leaves out those
        not-yet-treated rows.
    fe : {'vs', 'feo', 'none'}, default='vs'
        Fixed effects specification. ``"vs"`` and ``"feo"`` fit the same
        regression. It absorbs unit and time fixed effects, or cohort and time
        fixed effects when ``idname`` is None. Each control enters with its
        cohort and time interactions. ``"none"`` absorbs nothing and adds cohort
        and time dummies instead.
    family : {None, 'gaussian', 'poisson', 'logit', 'probit'}, default=None
        Model family. ``None`` and ``"gaussian"`` fit a linear regression.
        ``"poisson"`` fits a Poisson quasi-maximum likelihood model. ``"logit"``
        and ``"probit"`` fit binary response models. Nonlinear families replace
        the absorbed fixed effects with cohort and time dummies [2]_ and use
        ``idname`` only for clustering and for counting units. The binary
        families cannot drop collinear columns. Their controls must therefore
        vary within every cohort and period.
    weightsname : str or None, default=None
        The name of the column containing sampling weights. If not set, all
        observations have equal weight.
    vcov : str or dict or None, default=None
        Variance-covariance specification, such as ``"iid"``, ``"hetero"``,
        ``"HC1"``, or ``{"CRV1": "cluster_var"}``. The default clusters by
        ``idname`` when it is given and is heteroskedasticity-robust otherwise.
        Heteroskedasticity-robust errors ignore the correlation of a unit's
        outcomes over time. Pass a cluster variable for panel data fitted
        without ``idname``.
    alp : float, default=0.05
        The significance level.
    backend : {'cupy', 'jax', 'numba', 'rust', 'scipy'} or None, default=None
        Backend that absorbs the fixed effects. ``"numba"``, ``"rust"``, and
        ``"scipy"`` run on the CPU. ``None`` selects ``"numba"``. ``"cupy"`` and
        ``"jax"`` run on a GPU when CuPy or JAX finds one and otherwise use a
        CPU solver.

    Returns
    -------
    EtwfeResult
        Object containing ETWFE regression results:

        - **coefficients**: estimate for each treatment cell (index scale for
          nonlinear families), NaN for a cell the regression dropped as collinear
        - **std_errors**: standard error of each cell estimate
        - **vcov**: variance-covariance matrix of all regression coefficients
        - **coef_names**: names of all regression coefficients
        - **gt_pairs**: (group, time) pair of each treatment cell
        - **n_obs**: number of observations
        - **n_units**: number of units in the estimation sample, or of
          observations when ``idname`` is None
        - **r_squared**: R-squared of the regression
        - **data**: fitted data (used internally by ``emfx``)
        - **config**: configuration object (used internally by ``emfx``)
        - **estimation_params**: dictionary of estimation details whose
          ``formula`` names internal columns
        - **model_coefficients**: estimates of all regression coefficients,
          ordered as ``coef_names``

    See Also
    --------
    emfx : Aggregate ETWFE cell-level estimates into treatment effect summaries.
    att_gt : Group-time ATT estimation of Callaway and Sant'Anna (2021).

    Notes
    -----
    Each cell's coefficient targets the cohort-time average treatment effect on
    the treated,

    .. math::

        \tau_{g,t} \equiv E[y_t(g) - y_t(\infty) \mid d_g = 1],
        \quad t \ge g.

    Under no anticipation, conditional parallel trends, and linearity, the
    conditional expectation of the never-treated potential outcome is

    .. math::

        E[y_t(\infty) \mid \mathbf{d}, \mathbf{x}]
        = \alpha + \sum_g \beta_g d_g + \mathbf{x}\boldsymbol{\kappa}
        + \sum_g (d_g \cdot \mathbf{x})\boldsymbol{\xi}_g
        + \sum_s \gamma_s f_{s,t}
        + \sum_s (f_{s,t} \cdot \mathbf{x})\boldsymbol{\pi}_s,

    where :math:`d_g` are treatment cohort indicators, :math:`f_{s,t}` are
    time dummies, and :math:`\mathbf{x}` are time-constant covariates. The
    ATTs are then identified as

    .. math::

        \tau_{g,t}
        = E(y_t \mid d_g = 1)
        - \bigl[(\alpha + \beta_g + \gamma_t)
        + E(\mathbf{x} \mid d_g = 1)
        \cdot (\boldsymbol{\kappa} + \boldsymbol{\xi}_g
        + \boldsymbol{\pi}_t)\bigr].

    With :math:`w_t` the treatment indicator, the regression includes the full
    set of treatment interactions :math:`w_t \cdot d_g \cdot f_{s,t}`. Their
    interactions with the covariates use the covariates demeaned about their
    cohort means, :math:`\dot{\mathbf{x}}_g = \mathbf{x} - \bar{\mathbf{x}}_g`.

    References
    ----------

    .. [1] Wooldridge, J. M. (2025). "Two-Way Fixed Effects, the Two-Way
       Mundlak Regression, and Difference-in-Differences Estimators."
       Empirical Economics, 69, 2545-2587.

    .. [2] Wooldridge, J. M. (2023). "Simple Approaches to Nonlinear
       Difference-in-Differences with Panel Data." The Econometrics
       Journal, 26(3), C31-C66.

    """
    if family not in (None, "gaussian", "poisson", "logit", "probit"):
        raise ValueError(f"family must be None, 'gaussian', 'poisson', 'logit', or 'probit', got '{family}'")

    # Counterfactual predictions in emfx need every effect as a regressor. Nonlinear families
    # therefore fit cohort and time dummies instead of absorbed fixed effects.
    if family not in (None, "gaussian"):
        fe = "none"

    if cgroup not in ("notyet", "never"):
        raise ValueError(f"cgroup must be 'notyet' or 'never', got '{cgroup}'")

    if fe not in ("vs", "feo", "none"):
        raise ValueError(f"fe must be 'vs', 'feo', or 'none', got '{fe}'")

    df = to_polars(data)
    check_columns(
        df,
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=xformla,
        xvar=xvar,
        weightsname=weightsname,
        vcov=vcov if isinstance(vcov, dict) else None,
    )

    # A unit's outcomes are correlated over time. The default clusters by unit whenever the data name one.
    if vcov is None:
        vcov = {"CRV1": idname} if idname else "hetero"

    config = EtwfeConfig(
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=xformla or "~1",
        xvar=xvar,
        tref=tref,
        gref=gref,
        cgroup=cgroup,
        fe=fe,
        family=family,
        weightsname=weightsname,
        alp=alp,
        panel=idname is not None,
    )

    df = clean_etwfe_data(df, config, vcov)
    config = set_references(config, df)

    df_prepared = prepare_etwfe_data(df, config)
    if xvar and not config._xvar_dm_cols:
        warnings.warn(
            f"xvar='{xvar}' adds no terms. Within each cohort-time cell it is constant or a linear "
            "function of the controls in xformla.",
            UserWarning,
            stacklevel=2,
        )

    cells = treatment_cells(df_prepared, config)
    formula = build_etwfe_formula(config, df_prepared)
    config._formula = formula

    reg = run_etwfe_regression(formula, df_prepared, config, vcov=vcov, backend=backend)

    model = reg["model"]
    fit_data = reg["fit_data"]

    coef_names = [str(c) for c in model._coefnames]
    beta = np.asarray(model._beta_hat, dtype=float)
    se = np.asarray(model._se, dtype=float)
    vcov_mat = np.asarray(model._vcov, dtype=float)
    n_obs = int(model._N)
    r2 = model._r2
    r2_adj = model._r2_adj if hasattr(model, "_r2_adj") else None

    coef_pos = {name: i for i, name in enumerate(coef_names)}
    cell_pos = [coef_pos.get(f"_Dtreat:{_cell_column(g, t)}") for g, t in cells]
    treat_beta = np.array([np.nan if i is None else beta[i] for i in cell_pos])
    treat_se = np.array([np.nan if i is None else se[i] for i in cell_pos])
    dropped = [cell for cell, i in zip(cells, cell_pos, strict=True) if i is None]
    if dropped:
        warnings.warn(
            f"The regression dropped the treatment cells {_format_cells(dropped)} as collinear with its other "
            "terms. Controls that span a cell or a cohort with no untreated period cause this. Their estimates "
            "are NaN.",
            UserWarning,
            stacklevel=2,
        )

    n_units = fit_data[idname].n_unique() if idname else n_obs
    config.n_units = n_units
    config.n_obs = n_obs

    return EtwfeResult(
        coefficients=treat_beta,
        std_errors=treat_se,
        vcov=vcov_mat,
        coef_names=coef_names,
        gt_pairs=cells,
        n_obs=n_obs,
        n_units=n_units,
        r_squared=r2,
        adj_r_squared=r2_adj,
        data=fit_data,
        config=config,
        estimation_params={
            "yname": yname,
            "tname": tname,
            "gname": gname,
            "idname": idname,
            "cgroup": cgroup,
            "fe": fe,
            "alpha": alp,
            "formula": formula,
            "fe_spec": f"{idname or gname} + {tname}" if fe != "none" else None,
            "vcov_type": _vcov_type_label(vcov),
            "vcov_spec": vcov,
            "clustervar": next(iter(vcov.values())) if isinstance(vcov, dict) else None,
            "backend": backend,
            "n_units": n_units,
            "n_obs": n_obs,
            "family": family,
        },
        model_coefficients=beta,
    )


def _vcov_type_label(vcov_spec):
    """Convert vcov spec to human-readable label."""
    if vcov_spec is None:
        return "iid"
    if isinstance(vcov_spec, str):
        return vcov_spec
    if isinstance(vcov_spec, dict):
        return next(iter(vcov_spec.keys()))
    return str(vcov_spec)
