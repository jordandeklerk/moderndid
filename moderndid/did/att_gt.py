"""Multi-period difference-in-differences group-time average treatment effects estimation."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as la
import scipy.stats

from moderndid.core.preprocess import (
    BasePeriod,
    ControlGroup,
    DIDConfig,
    EstimationMethod,
    PreprocessDataBuilder,
)

from .compute_att_gt import compute_att_gt
from .mboot import mboot
from .multiperiod_obj import mp


def att_gt(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    xformla=None,
    weightsname=None,
    alp=0.05,
    bstrap=True,
    cband=True,
    biters=1000,
    clustervars=None,
    est_method="dr",
    panel=True,
    allow_unbalanced_panel=False,
    control_group="nevertreated",
    anticipation=0,
    base_period="varying",
    random_state=None,
):
    r"""Compute group-time average treatment effects.

    Computes average treatment effects in DID setups where there are more than two
    periods of data and allowing for treatment to occur at different points in time
    and allowing for treatment effect heterogeneity and dynamics.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        The DataFrame containing the data. Accepts both pandas and polars DataFrames.
    yname : str
        The name of the outcome variable.
    tname : str
        The name of the column containing the time periods.
    idname : str, optional
        The individual (cross-sectional unit) id name. Required for panel data.
    gname : str
        The name of the variable that contains the first period when a particular
        observation is treated. This should be a positive number for all observations
        in treated groups. It defines which "group" a unit belongs to. It should be 0
        for units in the untreated group.
    xformla : str, optional
        A formula for the covariates to include in the model. It should be of the
        form "~ X1 + X2". Default is None which is equivalent to xformla="~1".
    weightsname : str, optional
        The name of the column containing the sampling weights. If not set, all
        observations have same weight.
    alp : float, default=0.05
        The significance level.
    bstrap : bool, default=True
        Whether or not to compute standard errors using the multiplier bootstrap.
        If standard errors are clustered, then one must set bstrap=True.
    cband : bool, default=True
        Whether or not to compute a uniform confidence band that covers all of the
        group-time average treatment effects with fixed probability 1-alp.
    biters : int, default=1000
        The number of bootstrap iterations to use. Only applicable if bstrap=True.
    clustervars : list[str], optional
        A list of variables names to cluster on. At most, there can be two variables
        (otherwise will throw an error) and one of these must be the same as idname
        which allows for clustering at the individual level.
    est_method : {"dr", "ipw", "reg"} or callable, default="dr"
        The method to compute group-time average treatment effects. The default is
        "dr" which uses the doubly robust approach. Other built-in methods include
        "ipw" for inverse probability weighting and "reg" for first step regression
        estimators. The user can also pass their own function for estimating group
        time average treatment effects.
    panel : bool, default=True
        Whether or not the data is a panel dataset. The panel dataset should be
        provided in long format.
    allow_unbalanced_panel : bool, default=False
        Whether or not function should "balance" the panel with respect to time and
        id. The default values if False which means that att_gt will drop all units
        where data is not observed in all periods.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Which units to use the control group. The default is "nevertreated" which
        sets the control group to be the group of units that never participate in
        the treatment.
    anticipation : int, default=0
        The number of time periods before participating in the treatment where units
        can anticipate participating in the treatment and therefore it can affect
        their untreated potential outcomes.
    base_period : {"varying", "universal"}, default="varying"
        Whether to use a "varying" base period or a "universal" base period.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    MPResult
        Object containing all the results for group-time average treatment effects.

    Examples
    --------
    The dataset below contains 500 observations of county-level teen employment rates from 2003-2007.
    Some states are first treated in 2004, some in 2006, and some in 2007. The variable ``first.treat``
    indicates the first period in which a state is treated:

    .. ipython::

        In [1]: import numpy as np
           ...: from moderndid import att_gt, load_mpdta
           ...:
           ...: df = load_mpdta()
           ...: print(df.head())

    We can compute group-time average treatment effects for a staggered adoption design
    where different units adopt treatment at different time periods. The output is an object of type
    ``MPResult`` which is a container for the results:

    .. ipython::
        :okwarning:

        In [2]: result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     bstrap=False
           ...: )
           ...: print(result)

    See Also
    --------
    aggte : Aggregate group-time average treatment effects.

    References
    ----------
    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-differences
           with multiple time periods." Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")

    control_group_enum = ControlGroup(control_group)
    est_method_enum = EstimationMethod(est_method) if isinstance(est_method, str) else est_method
    base_period_enum = BasePeriod(base_period)

    config = DIDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group_enum,
        anticipation=anticipation,
        weightsname=weightsname,
        alp=alp,
        bstrap=bstrap,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        est_method=est_method_enum,
        base_period=base_period_enum,
    )

    builder = PreprocessDataBuilder()
    dp = builder.with_data(data).with_config(config).validate().transform().build()
    results = compute_att_gt(dp)

    att_gt_list = results.attgt_list
    influence_functions = results.influence_functions

    groups = np.array([att.group for att in att_gt_list])
    times = np.array([att.year for att in att_gt_list])
    att_values = np.array([att.att for att in att_gt_list])

    if hasattr(influence_functions, "toarray"):
        influence_functions_dense = influence_functions.toarray()
    else:
        influence_functions_dense = np.array(influence_functions)

    n_units = dp.config.id_count
    variance_matrix = influence_functions_dense.T @ influence_functions_dense / n_units
    standard_errors = np.sqrt(np.diag(variance_matrix) / n_units)
    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    # If clustering along another dimension we require using the bootstrap
    if (clustervars is not None and len(clustervars) > 0) and not bstrap:
        warnings.warn(
            "Clustering the standard errors requires using the bootstrap, "
            "resulting standard errors are NOT accounting for clustering",
            UserWarning,
        )

    zero_na_sd_indices = np.unique(np.where(np.isnan(standard_errors))[0])

    if bstrap:
        cluster = None
        if clustervars is not None and len(clustervars) > 0:
            if len(clustervars) > 2:
                raise ValueError("Can cluster on at most 2 variables.")

            if not hasattr(dp, "time_invariant_data"):
                raise RuntimeError(
                    "Clustering requires 'time_invariant_data' in the pre-processed data, but it was not found."
                )
            cluster_data = dp.time_invariant_data

            if len(clustervars) == 1:
                cluster = cluster_data[clustervars[0]].to_numpy()
            else:
                combined = cluster_data[clustervars[0]].cast(str) + "_" + cluster_data[clustervars[1]].cast(str)
                unique_vals = combined.unique()
                val_to_code = {v: i for i, v in enumerate(unique_vals.to_list())}
                cluster = np.array([val_to_code[v] for v in combined.to_list()])

        bootstrap_results = mboot(
            inf_func=influence_functions_dense,
            n_units=n_units,
            biters=biters,
            alp=alp,
            cluster=cluster,
            random_state=random_state,
        )

        if len(zero_na_sd_indices) > 0:
            standard_errors[~np.isin(np.arange(len(standard_errors)), zero_na_sd_indices)] = bootstrap_results["se"][
                ~np.isin(np.arange(len(standard_errors)), zero_na_sd_indices)
            ]
        else:
            standard_errors = bootstrap_results["se"]

    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    # Wald pre-test
    pre_treatment_indices = np.where(groups > times)[0]

    if len(zero_na_sd_indices) > 0:
        pre_treatment_indices = pre_treatment_indices[~np.isin(pre_treatment_indices, zero_na_sd_indices)]

    # Pseudo-atts in pre-treatment periods
    pre_treatment_att = att_values[pre_treatment_indices]
    pre_treatment_variance = variance_matrix[np.ix_(pre_treatment_indices, pre_treatment_indices)]

    if len(pre_treatment_indices) == 0:
        warnings.warn("No pre-treatment periods to test", UserWarning)
        wald_statistic = None
        wald_pvalue = None
    if np.any(np.isnan(pre_treatment_variance)):
        warnings.warn(
            "Not returning pre-test Wald statistic due to NA pre-treatment values",
            UserWarning,
        )
        wald_statistic = None
        wald_pvalue = None
    if (
        la.norm(pre_treatment_variance) == 0
        or np.linalg.matrix_rank(pre_treatment_variance) < pre_treatment_variance.shape[0]
    ):
        warnings.warn(
            "Not returning pre-test Wald statistic due to singular covariance matrix",
            UserWarning,
        )
        wald_statistic = None
        wald_pvalue = None
    else:
        try:
            wald_statistic = n_units * pre_treatment_att.T @ np.linalg.solve(pre_treatment_variance, pre_treatment_att)
            q = len(pre_treatment_indices)
            wald_pvalue = round(1 - scipy.stats.chi2.cdf(wald_statistic, q), 5)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Not returning pre-test Wald statistic due to numerical issues",
                UserWarning,
            )
            wald_statistic = None
            wald_pvalue = None

    critical_value = scipy.stats.norm.ppf(1 - alp / 2)

    if bstrap and cband:
        critical_value = bootstrap_results["crit_val"]
        if not np.isnan(critical_value) and critical_value >= 7:
            warnings.warn(
                "Simultaneous critical value is arguably 'too large' to be reliable. "
                "This usually happens when number of observations per group is small "
                "and/or there is not much variation in outcomes.",
                UserWarning,
            )

    estimation_params = {
        "control_group": control_group,
        "anticipation_periods": anticipation,
        "estimation_method": est_method if isinstance(est_method, str) else "custom",
        "bootstrap": bstrap,
        "uniform_bands": cband,
        "base_period": base_period,
        "panel": panel,
        "clustervars": clustervars,
        "biters": biters,
        "random_state": random_state,
    }

    group_assignments = None
    sampling_weights = None

    if hasattr(dp, "time_invariant_data"):
        if gname in dp.time_invariant_data.columns:
            group_assignments = dp.time_invariant_data[gname]
        if weightsname is not None and weightsname in dp.time_invariant_data.columns:
            sampling_weights = dp.time_invariant_data[weightsname]

    return mp(
        groups=groups,
        times=times,
        att_gt=att_values,
        vcov_analytical=variance_matrix,
        se_gt=standard_errors,
        critical_value=critical_value,
        influence_func=influence_functions_dense,
        n_units=n_units,
        wald_stat=wald_statistic,
        wald_pvalue=wald_pvalue,
        alpha=alp,
        estimation_params=estimation_params,
        G=group_assignments,
        weights_ind=sampling_weights,
    )
