"""Compute Aggregated Treatment Effect Parameters."""

from __future__ import annotations

import warnings

import numpy as np

from .aggte_obj import AGGTEResult
from .mboot import mboot


def compute_aggte(
    multi_period_result,
    aggregation_type="group",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    dropna=False,
    bootstrap=None,
    bootstrap_iterations=None,
    confidence_band=None,
    alpha=None,
    clustervars=None,
    random_state=None,
):
    """Compute aggregated treatment effect parameters.

    Aggregates group-time average treatment effects into different summary
    measures based on the specified aggregation type.

    Parameters
    ----------
    multi_period_result : MPResult
        Multi-period result object containing group-time ATTs and their
        influence functions.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}, default='group'
        Type of aggregation to perform:

        - 'simple': Simple weighted average of all post-treatment ATTs
        - 'dynamic': Event-study aggregation by relative time
        - 'group': Aggregation by treatment group
        - 'calendar': Aggregation by calendar time
    balance_e : int, optional
        For event studies, the relative time period to balance the sample on.
        This ensures a balanced panel with respect to event time.
    min_e : float, default=-np.inf
        Minimum event time to include in dynamic aggregation.
    max_e : float, default=np.inf
        Maximum event time to include in aggregation.
    dropna : bool, default=False
        Whether to remove NA values before aggregation.
    bootstrap : bool, optional
        Whether to use bootstrap inference. If None, uses the value from mp_result.
    bootstrap_iterations : int, optional
        Number of bootstrap iterations. If None, uses the value from mp_result.
    confidence_band : bool, optional
        Whether to compute uniform confidence bands. If None, uses value from mp_result.
    alpha : float, optional
        Significance level. If None, uses the value from mp_result.
    clustervars : list[str], optional
        Variables to cluster on. If None, uses the value from mp_result.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    AGGTEResult
        Aggregated treatment effect results with overall ATT, standard errors,
        and potentially disaggregated effects by event time.

    References
    ----------
    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    groups = multi_period_result.groups
    times = multi_period_result.times
    att = multi_period_result.att_gt
    influence_function = multi_period_result.influence_func
    n_units = multi_period_result.n_units or len(influence_function)

    estimation_params = multi_period_result.estimation_params.copy()
    if bootstrap is not None:
        estimation_params["bootstrap"] = bootstrap
    if bootstrap_iterations is not None:
        estimation_params["biters"] = bootstrap_iterations
    if confidence_band is not None:
        estimation_params["uniform_bands"] = confidence_band
    if alpha is not None:
        estimation_params["alpha"] = alpha
    else:
        alpha = multi_period_result.alpha
    if clustervars is not None:
        estimation_params["clustervars"] = clustervars
    elif "clustervars" not in estimation_params:
        estimation_params["clustervars"] = None

    if random_state is not None:
        estimation_params["random_state"] = random_state
    elif "random_state" not in estimation_params:
        estimation_params["random_state"] = None

    bootstrap = estimation_params.get("bootstrap", False)
    bootstrap_iterations = bootstrap_iterations or estimation_params.get("biters", 999)
    random_state = estimation_params.get("random_state")

    if aggregation_type not in ["simple", "dynamic", "group", "calendar"]:
        raise ValueError(
            f"`aggregation_type` must be one of ['simple', 'dynamic', 'group', 'calendar'], got {aggregation_type}"
        )

    if dropna:
        notna = ~np.isnan(att)
        groups = groups[notna]
        times = times[notna]
        att = att[notna]
        influence_function = influence_function[:, notna]

        if aggregation_type == "group":
            unique_groups = np.unique(groups)
            groups_with_att = []
            for g in unique_groups:
                mask = (groups == g) & (groups <= times)
                if mask.any() and not np.isnan(att[mask].mean()):
                    groups_with_att.append(g)

            keep = np.isin(groups, groups_with_att)
            groups = groups[keep]
            times = times[keep]
            att = att[keep]
            influence_function = influence_function[:, keep]

    elif np.any(np.isnan(att)):
        raise ValueError("Missing values at att_gt found. If you want to remove these, set `dropna = True'.")

    original_groups = groups.copy()
    original_times = times.copy()

    unique_original_groups = np.unique(original_groups)
    unique_original_times = np.unique(original_times)

    # Recode time periods to ensure they are 1 unit apart
    # This handles cases where time periods are not sequential integers
    unique_original_times_and_groups = np.sort(
        np.unique(np.concatenate([unique_original_times, unique_original_groups]))
    )
    recoded_times = np.arange(1, len(unique_original_times_and_groups) + 1)

    # Remap time-related variables to sequential integers
    groups = np.array([_orig2t(g, unique_original_times_and_groups, recoded_times) for g in original_groups])
    times = np.array([_orig2t(t, unique_original_times_and_groups, recoded_times) for t in original_times])
    unique_groups_recoded = np.array(
        [_orig2t(g, unique_original_times_and_groups, recoded_times) for g in unique_original_groups]
    )
    unique_times_recoded = np.unique(times)

    unit_level_groups = multi_period_result.G
    unit_level_weights = multi_period_result.weights_ind

    # If weights not provided, use uniform weights
    if unit_level_weights is None:
        if unit_level_groups is not None:
            unit_level_weights = np.ones(len(unit_level_groups))
        else:
            warnings.warn(
                "Unit-level data (unit_level_groups and unit_level_weights) not provided. "
                "Standard errors may be incorrect. Please update the MPResult object."
            )

            group_probabilities = np.array([np.mean(groups == g) for g in unique_groups_recoded])
            observation_group_probabilities = np.zeros(len(groups))
            for i, g in enumerate(unique_groups_recoded):
                mask = groups == g
                observation_group_probabilities[mask] = group_probabilities[i]
            unit_level_groups = None

    if unit_level_groups is not None and unit_level_weights is not None:
        unit_level_groups_recoded = np.array(
            [_orig2t(g, unique_original_times_and_groups, recoded_times) for g in unit_level_groups]
        )

        group_probabilities = np.array(
            [np.mean(unit_level_weights * (unit_level_groups_recoded == g)) for g in unique_groups_recoded]
        )
        observation_group_probabilities = np.zeros(len(groups))
        for i, g in enumerate(unique_groups_recoded):
            mask = groups == g
            observation_group_probabilities[mask] = group_probabilities[i]
    else:
        unit_level_groups_recoded = None

    post_treatment = (groups <= times) & (times <= groups + max_e)

    if aggregation_type == "simple":
        return _compute_simple_att(
            att=att,
            influence_function=influence_function,
            post_treatment=post_treatment,
            observation_group_probabilities=observation_group_probabilities,
            groups=groups,
            n_units=n_units,
            estimation_params=estimation_params,
            alpha=alpha,
            bootstrap_iterations=bootstrap_iterations,
            unique_groups_recoded=unique_groups_recoded,
            group_probabilities=group_probabilities,
            unit_level_groups=unit_level_groups_recoded,
            unit_level_weights=unit_level_weights,
            random_state=random_state,
        )

    if aggregation_type == "group":
        return _compute_group_att(
            att=att,
            influence_function=influence_function,
            groups=groups,
            times=times,
            unique_groups_recoded=unique_groups_recoded,
            unique_original_groups=unique_original_groups,
            group_probabilities=group_probabilities,
            n_units=n_units,
            max_e=max_e,
            estimation_params=estimation_params,
            alpha=alpha,
            bootstrap_iterations=bootstrap_iterations,
            unit_level_groups=unit_level_groups_recoded,
            unit_level_weights=unit_level_weights,
            random_state=random_state,
        )

    if aggregation_type == "dynamic":
        return _compute_dynamic_att(
            att=att,
            influence_function=influence_function,
            groups=groups,
            times=times,
            original_groups=original_groups,
            original_times=original_times,
            observation_group_probabilities=observation_group_probabilities,
            n_units=n_units,
            balance_e=balance_e,
            min_e=min_e,
            max_e=max_e,
            estimation_params=estimation_params,
            alpha=alpha,
            bootstrap_iterations=bootstrap_iterations,
            unique_groups_recoded=unique_groups_recoded,
            group_probabilities=group_probabilities,
            unit_level_groups=unit_level_groups_recoded,
            unit_level_weights=unit_level_weights,
            unique_original_times_and_groups=unique_original_times_and_groups,
            recoded_times=recoded_times,
            random_state=random_state,
        )

    if aggregation_type == "calendar":
        return _compute_calendar_att(
            att=att,
            influence_function=influence_function,
            groups=groups,
            times=times,
            unique_times_recoded=unique_times_recoded,
            observation_group_probabilities=observation_group_probabilities,
            n_units=n_units,
            estimation_params=estimation_params,
            alpha=alpha,
            bootstrap_iterations=bootstrap_iterations,
            unique_groups_recoded=unique_groups_recoded,
            group_probabilities=group_probabilities,
            unit_level_groups=unit_level_groups_recoded,
            unit_level_weights=unit_level_weights,
            unique_original_times_and_groups=unique_original_times_and_groups,
            recoded_times=recoded_times,
            random_state=random_state,
        )

    raise ValueError(f"Unexpected aggregation_type: {aggregation_type}")


def _compute_simple_att(
    att,
    influence_function,
    post_treatment,
    observation_group_probabilities,
    groups,
    n_units,
    estimation_params,
    alpha,
    bootstrap_iterations,
    unique_groups_recoded,
    group_probabilities,
    unit_level_groups,
    unit_level_weights,
    random_state=None,
):
    """Compute simple ATT by averaging all post-treatment effects."""
    # Simple ATT: weighted average of all post-treatment ATT(g,t)
    weights = observation_group_probabilities[post_treatment] / observation_group_probabilities[post_treatment].sum()
    simple_att = np.sum(att[post_treatment] * weights)

    if np.isnan(simple_att):
        simple_att = np.nan
        simple_se = np.nan
        simple_influence_function = np.zeros(n_units)
    else:
        # Get influence function for weights
        if unit_level_groups is not None and unit_level_weights is not None:
            weight_influence_function = _compute_weight_inf_func(
                keepers=np.where(post_treatment)[0],
                group_probabilities=group_probabilities,
                unit_level_weights=unit_level_weights,
                unit_level_groups=unit_level_groups,
                groups=groups,
                unique_groups_recoded=unique_groups_recoded,
            )
        else:
            weight_influence_function = None

        simple_influence_function = _get_agg_inf_func(
            att=att,
            influence_function=influence_function,
            keepers=post_treatment,
            weights=weights,
            weight_influence_function=weight_influence_function,
        )

        simple_se = _compute_se(
            influence_function=simple_influence_function,
            n_units=n_units,
            estimation_params=estimation_params,
            bootstrap_iterations=bootstrap_iterations,
            alpha=alpha,
            random_state=random_state,
        )

    return AGGTEResult(
        overall_att=simple_att,
        overall_se=simple_se,
        aggregation_type="simple",
        influence_func=simple_influence_function,
        influence_func_overall=simple_influence_function,
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "simple"},
    )


def _compute_group_att(
    att,
    influence_function,
    groups,
    times,
    unique_groups_recoded,
    unique_original_groups,
    group_probabilities,
    n_units,
    max_e,
    estimation_params,
    alpha,
    bootstrap_iterations,
    unit_level_groups,
    unit_level_weights,
    random_state=None,
):
    """Compute group-specific ATTs."""
    group_att = np.zeros(len(unique_groups_recoded))
    group_influence_functions = []

    for i, g in enumerate(unique_groups_recoded):
        mask = (groups == g) & ((g - 1) <= times) & (times <= g + max_e)

        if mask.any():
            group_att[i] = np.mean(att[mask])

            # Group-specific influence function
            weights_g = np.ones(mask.sum()) / mask.sum()
            influence_function_g = _get_agg_inf_func(
                att=att,
                influence_function=influence_function,
                keepers=mask,
                weights=weights_g,
                weight_influence_function=None,
            )
        else:
            group_att[i] = np.nan
            influence_function_g = np.zeros(n_units)

        group_influence_functions.append(influence_function_g)

    group_influence_functions = np.column_stack(group_influence_functions)
    group_att_clean = np.where(np.isnan(group_att), 0, group_att)

    group_se = np.zeros(len(unique_groups_recoded))
    for i in range(len(unique_groups_recoded)):
        if not np.isnan(group_att[i]):
            group_se[i] = _compute_se(
                influence_function=group_influence_functions[:, i],
                n_units=n_units,
                estimation_params=estimation_params,
                bootstrap_iterations=bootstrap_iterations,
                alpha=alpha,
                random_state=random_state,
            )
        else:
            group_se[i] = np.nan

    critical_value = _get_z_critical(alpha / 2)
    if estimation_params.get("uniform_bands", False):
        if not estimation_params.get("bootstrap", False):
            warnings.warn("Used bootstrap procedure to compute simultaneous confidence band")

        mboot_result = mboot(
            inf_func=group_influence_functions,
            n_units=n_units,
            biters=bootstrap_iterations,
            alp=alpha,
            random_state=random_state,
        )
        critical_value = mboot_result["crit_val"]

        if np.isnan(critical_value) or np.isinf(critical_value):
            warnings.warn(
                "Simultaneous critical value is NA. This probably happened because "
                "we cannot compute t-statistic (std errors are NA). "
                "We then report pointwise conf. intervals."
            )
            critical_value = _get_z_critical(alpha / 2)
            estimation_params["uniform_bands"] = False

    # Overall ATT: weighted average across groups
    overall_att = np.sum(group_att_clean * group_probabilities) / np.sum(group_probabilities)

    if unit_level_groups is not None and unit_level_weights is not None:
        weight_influence_function = _compute_weight_inf_func(
            keepers=np.arange(len(unique_groups_recoded)),
            group_probabilities=group_probabilities,
            unit_level_weights=unit_level_weights,
            unit_level_groups=unit_level_groups,
            groups=unique_groups_recoded,
            unique_groups_recoded=unique_groups_recoded,
        )
    else:
        weight_influence_function = None

    overall_influence_function = _get_agg_inf_func(
        att=group_att_clean,
        influence_function=group_influence_functions,
        keepers=np.ones(len(unique_groups_recoded), dtype=bool),
        weights=group_probabilities / group_probabilities.sum(),
        weight_influence_function=weight_influence_function,
    )

    overall_se = _compute_se(
        influence_function=overall_influence_function,
        n_units=n_units,
        estimation_params=estimation_params,
        bootstrap_iterations=bootstrap_iterations,
        alpha=alpha,
        random_state=random_state,
    )

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="group",
        event_times=unique_original_groups,
        att_by_event=group_att,
        se_by_event=group_se,
        critical_values=np.full(len(unique_groups_recoded), critical_value),
        influence_func=group_influence_functions,
        influence_func_overall=overall_influence_function,
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "group"},
    )


def _compute_dynamic_att(
    att,
    influence_function,
    groups,
    times,
    original_groups,
    original_times,
    observation_group_probabilities,
    n_units,
    balance_e,
    min_e,
    max_e,
    estimation_params,
    alpha,
    bootstrap_iterations,
    unique_groups_recoded,
    group_probabilities,
    unit_level_groups,
    unit_level_weights,
    unique_original_times_and_groups,
    recoded_times,
    random_state=None,
):
    """Compute dynamic (event-study) treatment effects."""
    event_times = original_times - original_groups
    unique_event_times = np.unique(event_times)
    unique_event_times = unique_event_times[np.isfinite(unique_event_times)]
    unique_event_times = np.sort(unique_event_times)

    include_balanced = np.ones(len(groups), dtype=bool)
    if balance_e is not None:
        max_recoded_time = times.max()
        max_original_time = _t2orig(max_recoded_time, unique_original_times_and_groups, recoded_times)
        # Only keep observations where we can observe balance_e periods after treatment
        include_balanced = (max_original_time - original_groups) >= balance_e

        event_times_balanced = event_times[include_balanced]
        unique_event_times = np.unique(event_times_balanced)
        unique_event_times = unique_event_times[np.isfinite(unique_event_times)]
        unique_event_times = np.sort(unique_event_times)

        min_recoded_time = times.min()
        min_original_time = _t2orig(min_recoded_time, unique_original_times_and_groups, recoded_times)
        min_possible_e = balance_e - max_original_time + min_original_time
        unique_event_times = unique_event_times[
            (unique_event_times >= min_possible_e) & (unique_event_times <= balance_e)
        ]

    unique_event_times = unique_event_times[(unique_event_times >= min_e) & (unique_event_times <= max_e)]

    dynamic_att = np.zeros(len(unique_event_times))
    dynamic_influence_functions = []

    for i, e in enumerate(unique_event_times):
        mask = (event_times == e) & include_balanced

        if mask.any():
            pg_e = observation_group_probabilities[mask]
            weights_e = pg_e / pg_e.sum()

            dynamic_att[i] = np.sum(att[mask] * weights_e)

            if unit_level_groups is not None and unit_level_weights is not None:
                weight_influence_function_e = _compute_weight_inf_func(
                    keepers=np.where(mask)[0],
                    group_probabilities=group_probabilities,
                    unit_level_weights=unit_level_weights,
                    unit_level_groups=unit_level_groups,
                    groups=groups,
                    unique_groups_recoded=unique_groups_recoded,
                )
            else:
                weight_influence_function_e = None

            influence_function_e = _get_agg_inf_func(
                att=att,
                influence_function=influence_function,
                keepers=mask,
                weights=weights_e,
                weight_influence_function=weight_influence_function_e,
            )
        else:
            dynamic_att[i] = np.nan
            influence_function_e = np.zeros(n_units)

        dynamic_influence_functions.append(influence_function_e)

    dynamic_influence_functions = np.column_stack(dynamic_influence_functions)

    dynamic_se = np.zeros(len(unique_event_times))
    for i in range(len(unique_event_times)):
        if not np.isnan(dynamic_att[i]):
            dynamic_se[i] = _compute_se(
                influence_function=dynamic_influence_functions[:, i],
                n_units=n_units,
                estimation_params=estimation_params,
                bootstrap_iterations=bootstrap_iterations,
                alpha=alpha,
                random_state=random_state,
            )
        else:
            dynamic_se[i] = np.nan

    critical_value = _get_z_critical(alpha / 2)
    if estimation_params.get("uniform_bands", False):
        if not estimation_params.get("bootstrap", False):
            warnings.warn("Used bootstrap procedure to compute simultaneous confidence band")

        mboot_result = mboot(
            inf_func=dynamic_influence_functions,
            n_units=n_units,
            biters=bootstrap_iterations,
            alp=alpha,
            random_state=random_state,
        )
        critical_value = mboot_result["crit_val"]

        if np.isnan(critical_value) or np.isinf(critical_value):
            warnings.warn(
                "Simultaneous critical value is NA. This probably happened because "
                "we cannot compute t-statistic (std errors are NA). "
                "We then report pointwise conf. intervals."
            )
            critical_value = _get_z_critical(alpha / 2)
            estimation_params["uniform_bands"] = False

    # Overall ATT: average over non-negative event times
    post_mask = unique_event_times >= 0
    if post_mask.any():
        overall_att = np.nanmean(dynamic_att[post_mask])

        # Overall influence function
        overall_influence_function = _get_agg_inf_func(
            att=dynamic_att[post_mask],
            influence_function=dynamic_influence_functions[:, post_mask],
            keepers=np.ones(post_mask.sum(), dtype=bool),
            weights=np.ones(post_mask.sum()) / post_mask.sum(),
            weight_influence_function=None,
        )

        # Overall standard error
        overall_se = _compute_se(
            influence_function=overall_influence_function,
            n_units=n_units,
            estimation_params=estimation_params,
            bootstrap_iterations=bootstrap_iterations,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        overall_att = np.nan
        overall_se = np.nan
        overall_influence_function = np.zeros(n_units)

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="dynamic",
        event_times=unique_event_times,
        att_by_event=dynamic_att,
        se_by_event=dynamic_se,
        critical_values=np.full(len(unique_event_times), critical_value),
        influence_func=dynamic_influence_functions,
        influence_func_overall=overall_influence_function,
        min_event_time=int(min_e) if np.isfinite(min_e) else None,
        max_event_time=int(max_e) if np.isfinite(max_e) else None,
        balanced_event_threshold=balance_e,
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "dynamic"},
    )


def _compute_calendar_att(
    att,
    influence_function,
    groups,
    times,
    unique_times_recoded,
    observation_group_probabilities,
    n_units,
    estimation_params,
    alpha,
    bootstrap_iterations,
    unique_groups_recoded,
    group_probabilities,
    unit_level_groups,
    unit_level_weights,
    unique_original_times_and_groups,
    recoded_times,
    random_state=None,
):
    """Compute calendar time effects."""
    min_group = groups.min()
    calendar_times = unique_times_recoded[unique_times_recoded >= min_group]

    original_calendar_times = np.array(
        [_t2orig(t, unique_original_times_and_groups, recoded_times) for t in calendar_times]
    )

    calendar_att = np.zeros(len(calendar_times))
    calendar_influence_functions = []

    for i, t in enumerate(calendar_times):
        mask = (times == t) & (groups <= t)

        if mask.any():
            pg_t = observation_group_probabilities[mask]
            weights_t = pg_t / pg_t.sum()

            calendar_att[i] = np.sum(att[mask] * weights_t)

            if unit_level_groups is not None and unit_level_weights is not None:
                weight_influence_function_t = _compute_weight_inf_func(
                    keepers=np.where(mask)[0],
                    group_probabilities=group_probabilities,
                    unit_level_weights=unit_level_weights,
                    unit_level_groups=unit_level_groups,
                    groups=groups,
                    unique_groups_recoded=unique_groups_recoded,
                )
            else:
                weight_influence_function_t = None

            influence_function_t = _get_agg_inf_func(
                att=att,
                influence_function=influence_function,
                keepers=mask,
                weights=weights_t,
                weight_influence_function=weight_influence_function_t,
            )
        else:
            calendar_att[i] = np.nan
            influence_function_t = np.zeros(n_units)

        calendar_influence_functions.append(influence_function_t)

    calendar_influence_functions = np.column_stack(calendar_influence_functions)

    calendar_se = np.zeros(len(calendar_times))
    for i in range(len(calendar_times)):
        if not np.isnan(calendar_att[i]):
            calendar_se[i] = _compute_se(
                influence_function=calendar_influence_functions[:, i],
                n_units=n_units,
                estimation_params=estimation_params,
                bootstrap_iterations=bootstrap_iterations,
                alpha=alpha,
                random_state=random_state,
            )
        else:
            calendar_se[i] = np.nan

    critical_value = _get_z_critical(alpha / 2)
    if estimation_params.get("uniform_bands", False):
        if not estimation_params.get("bootstrap", False):
            warnings.warn("Used bootstrap procedure to compute simultaneous confidence band")

        mboot_result = mboot(
            inf_func=calendar_influence_functions,
            n_units=n_units,
            biters=bootstrap_iterations,
            alp=alpha,
            random_state=random_state,
        )
        critical_value = mboot_result["crit_val"]

        if np.isnan(critical_value) or np.isinf(critical_value):
            warnings.warn(
                "Simultaneous critical value is NA. This probably happened because "
                "we cannot compute t-statistic (std errors are NA). "
                "We then report pointwise conf. intervals."
            )
            critical_value = _get_z_critical(alpha / 2)
            estimation_params["uniform_bands"] = False

    # Overall ATT: simple average across time periods
    overall_att = np.nanmean(calendar_att)

    # Overall influence function
    n_valid = (~np.isnan(calendar_att)).sum()
    if n_valid > 0:
        overall_influence_function = _get_agg_inf_func(
            att=calendar_att,
            influence_function=calendar_influence_functions,
            keepers=~np.isnan(calendar_att),
            weights=np.ones(n_valid) / n_valid,
            weight_influence_function=None,
        )

        # Overall standard error
        overall_se = _compute_se(
            influence_function=overall_influence_function,
            n_units=n_units,
            estimation_params=estimation_params,
            bootstrap_iterations=bootstrap_iterations,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        overall_att = np.nan
        overall_se = np.nan
        overall_influence_function = np.zeros(n_units)

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="calendar",
        event_times=original_calendar_times,
        att_by_event=calendar_att,
        se_by_event=calendar_se,
        critical_values=np.full(len(calendar_times), critical_value),
        influence_func=calendar_influence_functions,
        influence_func_overall=overall_influence_function,
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "calendar"},
    )


def _compute_weight_inf_func(
    keepers,
    group_probabilities,
    unit_level_weights,
    unit_level_groups,
    groups,
    unique_groups_recoded,
):
    """Compute influence function adjustment for estimated weights.

    This computes the extra term in the influence function that arises
    from having to estimate the weights (group probabilities).

    Parameters
    ----------
    keepers : ndarray
        Indices of observations used in aggregation.
    group_probabilities : ndarray
        Probability of being in each group (length = len(glist)).
    unit_level_weights : ndarray
        Unit-level sampling weights (length = n_units).
    unit_level_groups : ndarray
        Unit-level group assignments (length = n_units).
    groups : ndarray
        Group indicators for ATT(g,t) observations.
    unique_groups_recoded : ndarray
        Unique group values.

    Returns
    -------
    ndarray
        Weight influence function of shape (n_units, len(keepers)).
    """
    if unit_level_groups is None or unit_level_weights is None:
        raise ValueError(
            "Unit-level group assignments (unit_level_groups) and weights (unit_level_weights) are required"
        )

    n_units = len(unit_level_groups)
    n_keepers = len(keepers)

    weight_influence_function = np.zeros((n_units, n_keepers))

    pg_keepers = np.zeros(n_keepers)
    for i, k in enumerate(keepers):
        group_k = groups[k]
        group_idx = np.where(unique_groups_recoded == group_k)[0][0]
        pg_keepers[i] = group_probabilities[group_idx]

    sum_pg = pg_keepers.sum()

    for i, k in enumerate(keepers):
        group_k = groups[k]
        group_idx = np.where(unique_groups_recoded == group_k)[0][0]
        pg_k = group_probabilities[group_idx]

        # Effect of estimating weight in numerator
        # This is unit_level_weights * 1(unit_level_groups_i = g) - group_probabilities[g]
        # for each unit
        if1 = unit_level_weights * (unit_level_groups == group_k).astype(float) - pg_k
        if1 = if1 / sum_pg

        weight_influence_function[:, i] = if1

    # Effect of estimating weights in denominator
    sum_term = np.zeros(n_units)
    for k in keepers:
        group_k = groups[k]
        group_idx = np.where(unique_groups_recoded == group_k)[0][0]
        sum_term += unit_level_weights * (unit_level_groups == group_k).astype(float) - group_probabilities[group_idx]

    weight_influence_function -= np.outer(sum_term / (sum_pg**2), pg_keepers)

    return weight_influence_function


def _get_agg_inf_func(
    att,
    influence_function,
    keepers,
    weights,
    weight_influence_function=None,
):
    """Combine influence functions to get aggregated influence function.

    Parameters
    ----------
    att : ndarray
        Vector of ATT estimates.
    influence_function : ndarray
        Matrix of influence functions (n_units x n_att).
    keepers : ndarray
        Boolean mask or indices of which ATTs to include.
    weights : ndarray
        Weights to apply to each kept ATT.
    weight_influence_function : ndarray, optional
        Weight influence function from estimating group probabilities.

    Returns
    -------
    ndarray
        Aggregated influence function of shape (n_units,).
    """
    if keepers.dtype == bool:
        keep_idx = np.where(keepers)[0]
    else:
        keep_idx = keepers

    influence_function_kept = influence_function[:, keep_idx]
    att_kept = att[keep_idx]

    weighted_inf = influence_function_kept @ weights

    if weight_influence_function is not None:
        weighted_inf += weight_influence_function @ att_kept

    return weighted_inf


def _compute_se(
    influence_function,
    n_units,
    estimation_params,
    bootstrap_iterations,
    alpha,
    random_state=None,
):
    """Compute standard error from influence function.

    Parameters
    ----------
    influence_function : ndarray
        Influence function vector.
    n_units : int
        Number of cross-sectional units.
    estimation_params : dict
        Dictionary of estimation parameters.
    bootstrap_iterations : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    float
        Standard error estimate.
    """
    if estimation_params.get("bootstrap", False):
        # Use multiplier bootstrap
        clustervars = estimation_params.get("clustervars")
        cluster = None
        if clustervars:
            if len(clustervars) > 1:
                raise NotImplementedError("Clustering on multiple variables is not supported yet.")

            cluster_var_name = clustervars[0]

            warnings.warn(
                f"Clustering requested on '{cluster_var_name}' but data not available. "
                "Using standard bootstrap without clustering."
            )

        mboot_result = mboot(
            inf_func=influence_function.reshape(-1, 1),
            n_units=n_units,
            biters=bootstrap_iterations,
            alp=alpha,
            cluster=cluster,
            random_state=random_state,
        )
        standard_error = mboot_result["se"][0]
    else:
        # Analytical standard error
        standard_error = np.sqrt(np.mean(influence_function**2) / n_units)

    if standard_error <= np.sqrt(np.finfo(float).eps) * 10:
        standard_error = np.nan

    return standard_error


def _get_z_critical(alpha):
    """Get critical value from standard normal distribution."""
    from scipy import stats

    return stats.norm.ppf(1 - alpha)


def _orig2t(orig, unique_original_times_and_groups, recoded_times):
    """Convert original time value to sequential time value."""
    try:
        idx = np.where(unique_original_times_and_groups == orig)[0][0]
        return recoded_times[idx]
    except IndexError:
        return np.nan


def _t2orig(t, unique_original_times_and_groups, recoded_times):
    """Convert sequential time value back to original time value."""
    try:
        idx = np.where(recoded_times == t)[0][0]
        return unique_original_times_and_groups[idx]
    except IndexError:
        return 0
