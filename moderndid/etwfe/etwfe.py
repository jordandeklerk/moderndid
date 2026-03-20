"""ETWFE estimation via saturated cohort-time interactions.

Inspired by https://github.com/armandkapllani/etwfe/
"""

from __future__ import annotations

import re
import warnings

import numpy as np

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess.config import EtwfeConfig

from .compute import build_etwfe_formula, prepare_etwfe_data, run_etwfe_regression, set_references
from .container import EtwfeResult


def etwfe(
    data,
    yname: str,
    tname: str,
    gname: str,
    idname: str | None = None,
    xformla: str | None = None,
    xvar: str | None = None,
    tref=None,
    gref=None,
    cgroup: str = "notyet",
    fe: str = "vs",
    family=None,
    weightsname: str | None = None,
    vcov: str | dict | None = None,
    alp: float = 0.05,
    backend=None,
) -> EtwfeResult:
    r"""Estimate the Extended Two-Way Fixed Effects model.

    Implements the ETWFE methodology for difference-in-differences with
    staggered treatment adoption and heterogeneous treatment effects
    [1]_ [2]_. Rather than discarding the TWFE estimator, the approach
    saturates the model with cohort-by-time interaction terms so that the
    coefficients on the treatment indicators directly recover the
    cohort-time-specific average treatment effects on the treated,

    .. math::

        \tau_{g,t} \equiv E[y_t(g) - y_t(\infty) \mid d_g = 1],
        \quad t \ge g.

    Under no anticipation (NA), conditional parallel trends (CPT), and
    linearity (LIN), the conditional expectation of the never-treated
    potential outcome is

    .. math::

        E[y_t(\infty) \mid \mathbf{d}, \mathbf{x}]
        = \alpha + \sum_g \beta_g d_g + \mathbf{x}\boldsymbol{\kappa}
        + \sum_g (d_g \cdot \mathbf{x})\boldsymbol{\xi}_g
        + \sum_s \gamma_s f_{s,t}
        + \sum_s (f_{s,t} \cdot \mathbf{x})\boldsymbol{\pi}_s,

    where :math:`d_g` are treatment cohort indicators, :math:`f_{s,t}` are
    time dummies, and :math:`\mathbf{x}` are time-constant covariates.
    The ATTs are identified as

    .. math::

        \tau_{g,t}
        = E(y_t \mid d_g = 1)
        - \bigl[(\alpha + \beta_g + \gamma_t)
        + E(\mathbf{x} \mid d_g = 1)
        \cdot (\boldsymbol{\kappa} + \boldsymbol{\xi}_g
        + \boldsymbol{\pi}_t)\bigr].

    The ETWFE regression includes the full set of
    :math:`w_t \cdot d_g \cdot f_{s,t}` treatment interactions, with
    covariates demeaned about their cohort means
    :math:`\dot{\mathbf{x}}_g = \mathbf{x} - \bar{\mathbf{x}}_g`.
    The pooled OLS estimates of :math:`\tau_{g,t}` from this saturated
    regression are numerically identical to a cohort imputation
    procedure (Proposition 5.2 in [2]_). Use
    :func:`~moderndid.etwfe.emfx.emfx` to aggregate the cell-level
    estimates into overall, group, calendar, or event-study summaries.

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
        The name of the variable that contains the first period when a
        particular observation is treated. This should be a positive number
        for all observations in treated groups. It should be 0 for units in
        the untreated group. It defines which "cohort" a unit belongs to.
    idname : str or None, default=None
        The individual (cross-sectional unit) id name. When provided, unit
        fixed effects are absorbed in the regression.
    xformla : str or None, default=None
        A formula for the covariates to include in the model. It should be of
        the form ``"~ x1 + x2"``. Controls are demeaned within cohort groups
        following the Mundlak device so that the parallel trends assumption
        need only hold conditional on covariates.
    xvar : str or None, default=None
        Name of a covariate to interact with treatment for heterogeneous
        treatment effect analysis. The variable is demeaned within cohorts and
        interacted with the treatment and time indicators.
    tref : numeric or None, default=None
        Reference time period. Defaults to the minimum time period in the data.
    gref : numeric or None, default=None
        Reference cohort (control group). Auto-detected based on ``cgroup``.
        For ``"never"``, selects the group beyond the last observed period.
        For ``"notyet"``, defaults to the latest-treated cohort.
    cgroup : {'notyet', 'never'}, default='notyet'
        Control group strategy:

        - ``"notyet"``: use not-yet-treated units as controls (drops
          observations once the reference cohort enters treatment)
        - ``"never"``: use never-treated units as controls
    fe : {'vs', 'feo', 'none'}, default='vs'
        Fixed effects specification:

        - ``"vs"``: varying slopes (controls interact with cohort and time FE)
        - ``"feo"``: fixed effects only
        - ``"none"``: no absorbed fixed effects
    family : {None, 'gaussian', 'poisson', 'logit', 'probit'}, default=None
        GLM family for nonlinear models. ``None`` and ``"gaussian"`` use
        OLS via ``feols``. ``"poisson"`` uses Poisson QMLE via ``fepois``.
        ``"logit"`` and ``"probit"`` use ``feglm``. For non-Gaussian
        families, ``fe`` is set to ``"none"`` and ``idname`` is ignored
        (unit FE absorption is not supported for GLM).
    weightsname : str or None, default=None
        The name of the column containing sampling weights. If not set, all
        observations have equal weight.
    vcov : str or dict or None, default=None
        Variance-covariance specification passed to pyfixest. Defaults to
        ``"hetero"`` (heteroskedasticity-robust). Examples: ``"iid"``,
        ``"hetero"``, ``"HC1"``, ``{"CRV1": "cluster_var"}``.
    alp : float, default=0.05
        The significance level.
    backend : {'cupy', 'jax', 'numba', 'rust', 'scipy'} or None, default=None
        Demeaner backend for pyfixest's fixed-effects absorption.
        ``"cupy"`` and ``"jax"`` enable GPU acceleration (require CuPy or
        JAX with GPU support; without a GPU, pyfixest falls back to CPU).
        ``"numba"`` (the default), ``"rust"``, and ``"scipy"`` are CPU-only.
        ``None`` uses pyfixest's default.

    Returns
    -------
    EtwfeResult
        Object containing ETWFE regression results:

        - **coefficients**: coefficient estimates for each interaction term
        - **std_errors**: standard errors for each coefficient
        - **vcov**: variance-covariance matrix
        - **coef_names**: coefficient names from pyfixest
        - **gt_pairs**: list of (group, time) pairs for each coefficient
        - **n_obs**: number of observations
        - **n_units**: number of unique cross-sectional units
        - **r_squared**: R-squared of the regression
        - **data**: fitted data (used internally by ``emfx``)
        - **config**: configuration object (used internally by ``emfx``)
        - **estimation_params**: dictionary with estimation details

    See Also
    --------
    emfx : Aggregate ETWFE cell-level estimates into treatment effect summaries.
    att_gt : Group-time ATT estimation via Callaway and Sant'Anna (2021).

    References
    ----------

    .. [1] Wooldridge, J. M. (2021). "Two-Way Fixed Effects, the Two-Way
       Mundlak Regression, and Difference-in-Differences Estimators."

    .. [2] Wooldridge, J. M. (2023). "Simple Approaches to Nonlinear
       Difference-in-Differences with Panel Data." The Econometrics
       Journal, 26(3), C31-C66.

    Examples
    --------
    The dataset below contains 500 counties observed from 2003 to 2007.
    Some counties are first treated in 2004, some in 2006, and some in 2007.
    The variable ``first.treat`` indicates the first period of treatment:

    .. ipython::
        :okwarning:

        In [1]: from moderndid import etwfe, emfx, load_mpdta
           ...:
           ...: df = load_mpdta()
           ...: print(df.head())

    Estimate the saturated ETWFE model and print the cohort-time ATTs:

    .. ipython::
        :okwarning:

        In [2]: mod = etwfe(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...: )
           ...: print(mod)

    Aggregate into an event study with ``emfx``:

    .. ipython::
        :okwarning:

        In [3]: es = emfx(mod, type="event")
           ...: print(es)
    """
    if family not in (None, "gaussian", "poisson", "logit", "probit"):
        raise ValueError(f"family must be None, 'gaussian', 'poisson', 'logit', or 'probit', got '{family}'")

    if family is not None and family not in (None, "gaussian"):
        if idname is not None:
            warnings.warn(
                f"Non-linear family '{family}' does not support unit FE absorption. Setting idname=None.",
                UserWarning,
                stacklevel=2,
            )
            idname = None
        if fe != "none":
            fe = "none"

    if cgroup not in ("notyet", "never"):
        raise ValueError(f"cgroup must be 'notyet' or 'never', got '{cgroup}'")

    if fe not in ("vs", "feo", "none"):
        raise ValueError(f"fe must be 'vs', 'feo', or 'none', got '{fe}'")

    df = to_polars(data)

    for col_name, col_label in [(yname, "yname"), (tname, "tname"), (gname, "gname")]:
        if col_name not in df.columns:
            raise ValueError(f"{col_label}='{col_name}' not found in data columns")

    if idname and idname not in df.columns:
        raise ValueError(f"idname='{idname}' not found in data columns")

    if weightsname and weightsname not in df.columns:
        raise ValueError(f"weightsname='{weightsname}' not found in data columns")

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

    config = set_references(config, df)

    df_prepared = prepare_etwfe_data(df, config)

    formula = build_etwfe_formula(config)
    config._formula = formula

    reg = run_etwfe_regression(formula, df_prepared, config, vcov=vcov, backend=backend)

    model = reg["model"]
    fit_data = reg["fit_data"]

    coef_names = [str(c) for c in model._coefnames]
    beta = np.asarray(model._beta_hat, dtype=float)
    se = np.asarray(model._se, dtype=float)
    vcov_mat = np.asarray(model._vcov, dtype=float)
    n_obs = int(model._N)
    r2 = float(model._r2) if model._r2 is not None else None
    r2_adj = float(model._r2_adj) if hasattr(model, "_r2_adj") and model._r2_adj is not None else None

    gt_pairs = _extract_gt_pairs(coef_names)

    n_units = df[idname].n_unique() if idname else n_obs
    config.n_units = n_units
    config.n_obs = n_obs

    vcov_label = _vcov_type_label(vcov if vcov else "hetero")

    return EtwfeResult(
        coefficients=beta,
        std_errors=se,
        vcov=vcov_mat,
        coef_names=coef_names,
        gt_pairs=gt_pairs,
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
            "vcov_type": vcov_label,
            "clustervar": next(iter(vcov.values())) if isinstance(vcov, dict) else None,
            "backend": backend,
            "n_units": n_units,
            "n_obs": n_obs,
            "family": family,
        },
    )


def _extract_gt_pairs(coef_names: list[str]) -> list[tuple[float, float]]:
    """Extract (group, time) pairs from pyfixest coefficient names."""
    pattern = re.compile(
        r"_Dtreat:C\(__etwfe_gcat\)\[([^\]]+)\]"
        r":C\(__etwfe_tcat\)\[([^\]]+)\]"
    )

    gt_pairs = []
    for name in coef_names:
        m = pattern.search(name)
        if m:
            try:
                g = float(m.group(1))
                t = float(m.group(2))
                gt_pairs.append((g, t))
            except ValueError:
                continue

    return gt_pairs


def _vcov_type_label(vcov_spec) -> str:
    """Convert vcov spec to human-readable label."""
    if vcov_spec is None:
        return "iid"
    if isinstance(vcov_spec, str):
        return vcov_spec
    if isinstance(vcov_spec, dict):
        return next(iter(vcov_spec.keys()))
    return str(vcov_spec)
