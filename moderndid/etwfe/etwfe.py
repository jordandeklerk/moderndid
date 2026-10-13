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
    alpha=0.05,
    backend=None,
):
    r"""Estimate the Extended Two-Way Fixed Effects model.

    The extended two-way fixed effects (ETWFE) regression of [1]_ adds an indicator for every
    treated cohort-time cell to the two-way fixed effects model. The coefficient on the cell of
    cohort :math:`g` in period :math:`t` estimates the average treatment effect on the treated,

    .. math::

        \tau_{g,t} \equiv E[y_t(g) - y_t(\infty) \mid d_g = 1],
        \quad t \ge g.

    :func:`~moderndid.emfx` averages the cell estimates into overall, group, calendar, or
    event-study summaries.

    Rows with a missing or infinite value in any column the call names except the outcome leave
    the sample with a warning before the controls are demeaned. Since an infinite cohort marks a
    never-treated unit, it stays. The units of a cohort with no untreated row, such as one treated
    in the first period, also leave with a warning. Under ``cgroup="never"``, that row must lie in
    period :math:`g - 1`.

    See the :ref:`extended TWFE example <example_etwfe>` for a full analysis of the minimum wage
    data.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow PyCapsule Interface
        (``__arrow_c_stream__``), including polars, pandas, pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column.
    gname : str
        The name of the column with each unit's first treated period. Never-treated units have 0,
        infinity, or a value after the last period.
    idname : str, optional
        The name of the unit identifier column. When given, linear models absorb unit fixed effects
        and the default standard errors cluster by unit.
    xformla : str, optional
        A formula for the controls, such as ``"~ x1 + x2"``. Each term names a column, in backticks
        if the name holds spaces or symbols. Terms such as ``I(x**2)`` or ``x1:x2`` raise an error.
    xvar : str, optional
        The name of a covariate that moderates the treatment effects. A string covariate enters as
        an indicator for each category but the first. A covariate that is constant within the
        cohort-time cells, or a linear function of the controls there, draws a warning.
    tref : int or float, optional
        The reference period, a value of ``tname``. If None, the first period.
    gref : int or float, optional
        The reference cohort, a never-treated code such as 0 or, under ``cgroup="notyet"``, a
        cohort first treated after the first period. If None, the never-treated units, or the
        latest-treated cohort when there are none.
    cgroup : {"notyet", "never"}, default="notyet"
        The control group. ``"notyet"`` uses the never-treated units, the reference cohort, and
        other cohorts before they are treated. ``"never"`` uses the never-treated units alone and
        raises an error when there are none.
    fe : {"vs", "feo", "none"}, default="vs"
        The fixed effects. ``"vs"`` and ``"feo"`` absorb unit and time fixed effects, or cohort and
        time fixed effects without ``idname``. ``"none"`` adds cohort and time dummies instead.
    family : {"gaussian", "poisson", "logit", "probit"}, optional
        The model family. None and ``"gaussian"`` fit a linear regression, ``"poisson"`` a Poisson
        quasi-maximum likelihood model, and ``"logit"`` and ``"probit"`` binary response models
        [2]_. Nonlinear families ignore ``fe`` and use cohort and time dummies. Under the binary
        ones, each control must vary within every cohort and period.
    weightsname : str, optional
        The name of the sampling weights column. The weights must be non-negative with a positive
        mean. If None, observations have equal weight. Nonlinear families take none.
    vcov : str or dict, optional
        The variance-covariance specification, such as ``"iid"``, ``"hetero"``, or
        ``{"CRV1": "cluster_var"}``. If None, it clusters by ``idname`` when given and is
        heteroskedasticity-robust otherwise.
    alpha : float, default=0.05
        The significance level.
    backend : {"cupy", "cupy32", "cupy64", "jax", "numba", "rust", "scipy"}, optional
        The solver that absorbs the fixed effects. If None, ``"numba"``. The CuPy and JAX options
        use a GPU when they find one and the CPU otherwise. On a GPU, ``"cupy32"`` solves in single
        precision. Fits that absorb no fixed effects ignore it.

    Returns
    -------
    EtwfeResult
        Object containing ETWFE regression results:

        - **coefficients**: estimate for each treatment cell, on the index scale for nonlinear
          families and NaN for a cell the regression dropped as collinear
        - **std_errors**: standard error of each cell estimate
        - **vcov**: variance-covariance matrix of all regression coefficients
        - **coef_names**: names of all regression coefficients
        - **gt_pairs**: (group, time) pair of each treatment cell
        - **n_obs**: number of observations
        - **n_units**: number of units in the sample, or of observations without ``idname``
        - **r_squared**: R-squared of the regression, NaN for nonlinear families
        - **adj_r_squared**: adjusted R-squared, NaN for nonlinear families
        - **data**: fitted data (used internally by ``emfx``)
        - **config**: configuration object (used internally by ``emfx``)
        - **estimation_params**: estimation details whose ``formula`` names internal columns
        - **model_coefficients**: all regression coefficient estimates, ordered as ``coef_names``

    See Also
    --------
    emfx : Aggregate ETWFE cell-level estimates into treatment effect summaries.
    att_gt : Group-time ATT estimation of Callaway and Sant'Anna (2021).

    Notes
    -----
    Under no anticipation, conditional parallel trends, and linearity, the conditional
    expectation of the never-treated potential outcome is

    .. math::

        E[y_t(\infty) \mid \mathbf{d}, \mathbf{x}]
        = \alpha + \sum_g \beta_g d_g + \mathbf{x}\boldsymbol{\kappa}
        + \sum_g (d_g \cdot \mathbf{x})\boldsymbol{\xi}_g
        + \sum_s \gamma_s f_{s,t}
        + \sum_s (f_{s,t} \cdot \mathbf{x})\boldsymbol{\pi}_s,

    where :math:`d_g` are cohort indicators, :math:`f_{s,t}` are time dummies, and
    :math:`\mathbf{x}` are time-constant covariates. The ATTs are then identified as

    .. math::

        \tau_{g,t}
        = E(y_t \mid d_g = 1)
        - \bigl[(\alpha + \beta_g + \gamma_t)
        + E(\mathbf{x} \mid d_g = 1)
        \cdot (\boldsymbol{\kappa} + \boldsymbol{\xi}_g
        + \boldsymbol{\pi}_t)\bigr].

    With :math:`w_t` the treatment indicator, the regression includes every treatment interaction
    :math:`w_t \cdot d_g \cdot f_{s,t}`. Their interactions with the covariates use the covariates
    demeaned about their cohort means, :math:`\dot{\mathbf{x}}_g = \mathbf{x} - \bar{\mathbf{x}}_g`.
    Since the covariate in ``xvar`` is demeaned within each cohort-time cell instead, each cell's
    coefficient is the effect at the cell's mean of that covariate. The order of a string
    covariate's categories does not change the estimates.

    A column that the call names may not take the name of a column the regression adds, such as
    ``_g``, ``_t``, ``_Dtreat``, or one that starts with ``__etwfe_``. The
    :ref:`extended TWFE background <background-etwfe>` derives the untreated outcome model and
    shows why the regression reproduces cohort imputation.

    References
    ----------

    .. [1] Wooldridge, J. M. (2025). "Two-Way Fixed Effects, the Two-Way
       Mundlak Regression, and Difference-in-Differences Estimators."
       Empirical Economics, 69(5), 2545-2587.
       https://doi.org/10.1007/s00181-025-02807-z

    .. [2] Wooldridge, J. M. (2023). "Simple Approaches to Nonlinear
       Difference-in-Differences with Panel Data." The Econometrics
       Journal, 26(3), C31-C66.
       https://doi.org/10.1093/ectj/utad016
    """
    if family not in (None, "gaussian", "poisson", "logit", "probit"):
        raise ValueError(f"family must be None, 'gaussian', 'poisson', 'logit', or 'probit', got '{family}'")

    if weightsname is not None and family not in (None, "gaussian"):
        raise ValueError(
            f"family='{family}' takes no weights. Drop weightsname or pass family=None to fit a weighted linear "
            "regression."
        )

    # Counterfactual predictions in emfx need every effect as a regressor. Nonlinear families
    # therefore fit cohort and time dummies instead of absorbed fixed effects.
    if family not in (None, "gaussian"):
        fe = "none"

    if cgroup not in ("notyet", "never"):
        raise ValueError(f"cgroup must be 'notyet' or 'never', got '{cgroup}'")

    if fe not in ("vs", "feo", "none"):
        raise ValueError(f"fe must be 'vs', 'feo', or 'none', got '{fe}'")

    # The regression sees the backend only when it absorbs fixed effects. This check covers every fit.
    if backend not in (None, "cupy", "cupy32", "cupy64", "jax", "numba", "rust", "scipy"):
        raise ValueError(
            f"backend must be None, 'cupy', 'cupy32', 'cupy64', 'jax', 'numba', 'rust', or 'scipy', got '{backend}'"
        )

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
        alpha=alpha,
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
    r2_adj = model._adj_r2

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
            "alpha": alpha,
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
