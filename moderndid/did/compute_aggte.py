"""Compute Aggregated Treatment Effect Parameters."""

import warnings

import numpy as np
from scipy import stats

from .container import AGGTEResult
from .mboot import _cluster_sums, mboot


def compute_aggte(
    multi_period_result,
    aggregation_type="group",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    dropna=False,
    bootstrap=None,
    biters=None,
    confidence_band=None,
    alpha=None,
    clustervars=None,
    random_state=None,
):
    """Compute aggregated treatment effect parameters.

    Aggregates group-time average treatment effects into different summary
    measures based on the specified aggregation type. :func:`~moderndid.aggte`
    describes each aggregation and what it reports when no post-treatment
    effect is left.

    Parameters
    ----------
    multi_period_result : MPResult
        Multi-period result object containing group-time ATTs and their
        influence functions.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}, default='group'
        Type of aggregation to perform. 'simple' takes a weighted average of
        all post-treatment ATTs. 'dynamic' aggregates by event time, 'group'
        by treatment group, and 'calendar' by calendar time.
    balance_e : int, optional
        Event time to balance the 'dynamic' aggregation on. The other
        aggregations ignore it.
    min_e : float, default=-np.inf
        Smallest event time of the 'dynamic' aggregation. The other
        aggregations ignore it.
    max_e : float, default=np.inf
        Largest event time to include. It limits the 'dynamic', 'simple', and
        'group' aggregations.
    dropna : bool, default=False
        Whether to remove NaN estimates before aggregation.
    bootstrap : bool, optional
        Whether to use bootstrap inference. Defaults to the setting stored in
        multi_period_result.
    biters : int, optional
        Number of bootstrap iterations. Defaults to the setting stored in
        multi_period_result.
    confidence_band : bool, optional
        Whether to compute a uniform confidence band. Defaults to the setting
        stored in multi_period_result.
    alpha : float, optional
        Significance level. Defaults to the level stored in multi_period_result.
    clustervars : list[str], optional
        Variables to cluster the standard errors on. Each must be the idname or
        one of the clustervars stored in multi_period_result. An empty list, or
        the idname alone, gives unit-level inference. Defaults to the
        clustervars stored in multi_period_result.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    AGGTEResult
        Object containing the aggregated treatment effect results:

        - **overall_att**: Overall average treatment effect on the treated
        - **overall_se**: Standard error of the overall effect
        - **aggregation_type**: The aggregation that ``aggregation_type`` selected
        - **event_times**: Event times, groups, or periods of the disaggregated effects
        - **att_by_event**: Effect at each event time, group, or period
        - **se_by_event**: Standard error of each disaggregated effect
        - **critical_values**: Critical value of the confidence band for each disaggregated effect
        - **influence_func**: Influence functions of the disaggregated effects, or of the overall effect for 'simple'
        - **influence_func_overall**: Influence function of the overall effect
        - **min_event_time**: Lower end of an event study's window, or None without one
        - **max_event_time**: Upper end of an event study's window, or None without one
        - **balanced_event_threshold**: The ``balance_e`` that balanced an event study, or None
        - **estimation_params**: Dictionary of the estimation settings
        - **call_info**: Dictionary describing the call that produced the result
    """
    groups = multi_period_result.groups
    times = multi_period_result.times
    att = multi_period_result.att_gt
    influence_function = multi_period_result.influence_func
    n_units = multi_period_result.n_units or len(influence_function)

    estimation_params = multi_period_result.estimation_params.copy()
    if bootstrap is not None:
        estimation_params["bootstrap"] = bootstrap
    if biters is not None:
        estimation_params["biters"] = biters
    if confidence_band is not None:
        estimation_params["uniform_bands"] = confidence_band
    if alpha is not None:
        estimation_params["alpha"] = alpha
    else:
        alpha = multi_period_result.alpha
    estimation_params["clustervars"], estimation_params["cluster"] = _resolve_clusters(
        multi_period_result.estimation_params, clustervars
    )

    if random_state is not None:
        estimation_params["random_state"] = random_state
    elif "random_state" not in estimation_params:
        estimation_params["random_state"] = None

    bootstrap = estimation_params.get("bootstrap", False)
    biters = biters or estimation_params.get("biters", 999)
    random_state = estimation_params.get("random_state")

    if aggregation_type not in ["simple", "dynamic", "group", "calendar"]:
        raise ValueError(
            f"`aggregation_type` must be one of ['simple', 'dynamic', 'group', 'calendar'], got {aggregation_type}"
        )

    if len(att) == 0:
        raise ValueError(
            "The att_gt result holds no group-time effects to aggregate. The att_gt warnings name the cells that "
            "failed to estimate. Fewer covariates or control_group='notyettreated' can recover them."
        )

    if dropna:
        notna = ~np.isnan(att)
        if not notna.any():
            raise ValueError(
                "All att_gt estimates are NaN. The att_gt warnings name the cells that failed to estimate. Fewer "
                "covariates or control_group='notyettreated' can recover them."
            )
        groups = groups[notna]
        times = times[notna]
        att = att[notna]
        influence_function = influence_function[:, notna]
    elif np.any(np.isnan(att)):
        raise ValueError("Missing values at att_gt found. If you want to remove these, set na_rm=True in aggte.")

    original_groups = groups.copy()
    original_times = times.copy()

    unique_original_groups = np.unique(original_groups)
    unique_original_times = np.unique(original_times)

    # Since the max_e window counts periods, periods and cohorts are recoded to consecutive integers.
    unique_original_times_and_groups = np.sort(
        np.unique(np.concatenate([unique_original_times, unique_original_groups]))
    )
    recoded_times = np.arange(1, len(unique_original_times_and_groups) + 1)

    groups = np.array([_orig2t(g, unique_original_times_and_groups, recoded_times) for g in original_groups])
    times = np.array([_orig2t(t, unique_original_times_and_groups, recoded_times) for t in original_times])
    unique_groups_recoded = np.array(
        [_orig2t(g, unique_original_times_and_groups, recoded_times) for g in unique_original_groups]
    )
    unique_times_recoded = np.unique(times)

    unit_level_groups = multi_period_result.G
    unit_level_weights = multi_period_result.weights_ind

    if unit_level_groups is not None and not isinstance(unit_level_groups, np.ndarray):
        unit_level_groups = np.asarray(unit_level_groups)
    if unit_level_weights is not None and not isinstance(unit_level_weights, np.ndarray):
        unit_level_weights = np.asarray(unit_level_weights)

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
        unit_level_groups_recoded = _orig2t_array(unit_level_groups, unique_original_times_and_groups, recoded_times)

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
            times=times,
            n_units=n_units,
            max_e=max_e,
            estimation_params=estimation_params,
            alpha=alpha,
            biters=biters,
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
            biters=biters,
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
            biters=biters,
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
            biters=biters,
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
    times,
    n_units,
    max_e,
    estimation_params,
    alpha,
    biters,
    unique_groups_recoded,
    group_probabilities,
    unit_level_groups,
    unit_level_weights,
    random_state=None,
):
    """Compute simple ATT by averaging all post-treatment effects."""
    if not post_treatment.any():
        _warn_without_post_treatment_effects(times - groups, max_e)
        return AGGTEResult(
            overall_att=np.nan,
            overall_se=np.nan,
            aggregation_type="simple",
            influence_func=np.zeros(n_units),
            influence_func_overall=np.zeros(n_units),
            estimation_params=estimation_params,
            call_info={"function": "compute_aggte", "type": "simple"},
        )

    weights = observation_group_probabilities[post_treatment] / observation_group_probabilities[post_treatment].sum()
    simple_att = np.sum(att[post_treatment] * weights)

    if np.isnan(simple_att):
        simple_att = np.nan
        simple_se = np.nan
        simple_influence_function = np.zeros(n_units)
    else:
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
            biters=biters,
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
    biters,
    unit_level_groups,
    unit_level_weights,
    random_state=None,
):
    """Compute group-specific ATTs."""
    # A cohort without a post-treatment effect in the window has nothing to average, whether the varying base
    # dropped its failed cells or na_rm removed them.
    has_effect = np.array(
        [np.any((groups == g) & (g <= times) & (times <= g + max_e)) for g in unique_groups_recoded], dtype=bool
    )
    unique_groups_recoded = unique_groups_recoded[has_effect]
    unique_original_groups = unique_original_groups[has_effect]
    group_probabilities = group_probabilities[has_effect]

    group_att = np.zeros(len(unique_groups_recoded))
    group_influence_functions = []

    for i, g in enumerate(unique_groups_recoded):
        mask = (groups == g) & (g <= times) & (times <= g + max_e)
        group_att[i] = np.mean(att[mask])
        group_influence_functions.append(
            _get_agg_inf_func(
                att=att,
                influence_function=influence_function,
                keepers=mask,
                weights=np.ones(mask.sum()) / mask.sum(),
                weight_influence_function=None,
            )
        )

    group_influence_functions = _stack_columns(group_influence_functions, n_units)
    group_att_clean = np.where(np.isnan(group_att), 0, group_att)
    valid_group_mask = ~np.isnan(group_att)

    group_se, critical_value, _ = _batched_se(
        group_influence_functions,
        group_att,
        n_units,
        estimation_params,
        biters,
        alpha,
        random_state,
    )

    if valid_group_mask.any():
        valid_probs = group_probabilities[valid_group_mask]
        overall_att = np.sum(group_att_clean[valid_group_mask] * valid_probs) / np.sum(valid_probs)

        valid_indices = np.where(valid_group_mask)[0]
        if unit_level_groups is not None and unit_level_weights is not None:
            weight_influence_function = _compute_weight_inf_func(
                keepers=valid_indices,
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
            keepers=valid_group_mask,
            weights=valid_probs / valid_probs.sum(),
            weight_influence_function=weight_influence_function,
        )
    else:
        _warn_without_post_treatment_effects(times - groups, max_e)
        overall_att = np.nan
        overall_influence_function = np.zeros(n_units)

    overall_se = _compute_se(
        influence_function=overall_influence_function,
        n_units=n_units,
        estimation_params=estimation_params,
        biters=biters,
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
    biters,
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
    if len(unique_event_times) == 0:
        window = f"min_e={min_e} and max_e={max_e}" + ("" if balance_e is None else f" with balance_e={balance_e}")
        raise ValueError(
            f"No group-time effect has an event time within {window}. Widen the window or check the att_gt "
            "warnings for cells that failed to estimate."
        )

    dynamic_att = np.zeros(len(unique_event_times))
    dynamic_influence_functions = []

    for i, e in enumerate(unique_event_times):
        mask = (event_times == e) & include_balanced
        weights_e = observation_group_probabilities[mask] / observation_group_probabilities[mask].sum()
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

        dynamic_influence_functions.append(
            _get_agg_inf_func(
                att=att,
                influence_function=influence_function,
                keepers=mask,
                weights=weights_e,
                weight_influence_function=weight_influence_function_e,
            )
        )

    dynamic_influence_functions = _stack_columns(dynamic_influence_functions, n_units)

    dynamic_se, critical_value, _ = _batched_se(
        dynamic_influence_functions,
        dynamic_att,
        n_units,
        estimation_params,
        biters,
        alpha,
        random_state,
    )

    post_mask = unique_event_times >= 0
    if post_mask.any():
        overall_att = np.nanmean(dynamic_att[post_mask])

        overall_influence_function = _get_agg_inf_func(
            att=dynamic_att[post_mask],
            influence_function=dynamic_influence_functions[:, post_mask],
            keepers=np.ones(post_mask.sum(), dtype=bool),
            weights=np.ones(post_mask.sum()) / post_mask.sum(),
            weight_influence_function=None,
        )

        overall_se = _compute_se(
            influence_function=overall_influence_function,
            n_units=n_units,
            estimation_params=estimation_params,
            biters=biters,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        _warn_without_post_treatment_effects(event_times[include_balanced], max_e)
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
    biters,
    unique_groups_recoded,
    group_probabilities,
    unit_level_groups,
    unit_level_weights,
    unique_original_times_and_groups,
    recoded_times,
    random_state=None,
):
    """Compute calendar time effects."""
    # A period enters only when one of its cells is treated. Since a universal base keeps each cohort's reference
    # cell, a cohort whose treated cells all failed would otherwise add periods with nothing to average.
    calendar_times = np.array([t for t in unique_times_recoded if np.any((times == t) & (groups <= t))])

    original_calendar_times = np.array(
        [_t2orig(t, unique_original_times_and_groups, recoded_times) for t in calendar_times]
    )

    calendar_att = np.zeros(len(calendar_times))
    calendar_influence_functions = []

    for i, t in enumerate(calendar_times):
        mask = (times == t) & (groups <= t)
        weights_t = observation_group_probabilities[mask] / observation_group_probabilities[mask].sum()
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

        calendar_influence_functions.append(
            _get_agg_inf_func(
                att=att,
                influence_function=influence_function,
                keepers=mask,
                weights=weights_t,
                weight_influence_function=weight_influence_function_t,
            )
        )

    calendar_influence_functions = _stack_columns(calendar_influence_functions, n_units)

    calendar_se, critical_value, _ = _batched_se(
        calendar_influence_functions,
        calendar_att,
        n_units,
        estimation_params,
        biters,
        alpha,
        random_state,
    )

    n_valid = (~np.isnan(calendar_att)).sum()
    if n_valid > 0:
        overall_att = np.nanmean(calendar_att)
        overall_influence_function = _get_agg_inf_func(
            att=calendar_att,
            influence_function=calendar_influence_functions,
            keepers=~np.isnan(calendar_att),
            weights=np.ones(n_valid) / n_valid,
            weight_influence_function=None,
        )

        overall_se = _compute_se(
            influence_function=overall_influence_function,
            n_units=n_units,
            estimation_params=estimation_params,
            biters=biters,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        _warn_without_post_treatment_effects(times - groups, np.inf)
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


def _stack_columns(columns, n_units):
    """Stack influence functions as the columns of a matrix.

    Parameters
    ----------
    columns : list of ndarray
        Influence functions, one for each aggregated effect. The list may be empty.
    n_units : int
        Number of cross-sectional units.

    Returns
    -------
    ndarray
        Matrix of shape (n_units, len(columns)).
    """
    return np.column_stack(columns) if columns else np.zeros((n_units, 0))


def _warn_without_post_treatment_effects(event_times, max_e):
    """Warn that the overall effect is NaN and name what left out the post-treatment effects.

    Parameters
    ----------
    event_times : ndarray
        Event time of each group-time effect that reached the aggregation.
    max_e : float
        Largest event time the aggregation includes.
    """
    if np.any((event_times >= 0) & (event_times > max_e)):
        message = (
            f"The overall effect is NaN because max_e={max_e} leaves out every post-treatment group-time effect. "
            "Pass a larger max_e to estimate it."
        )
    else:
        message = (
            "The overall effect is NaN because no post-treatment group-time effect is left to average. The att_gt "
            "warnings name the cells that failed to estimate. Fewer covariates or control_group='notyettreated' "
            "can recover them."
        )
    warnings.warn(message, UserWarning)


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
        Indices of the effects that the aggregation averages.
    group_probabilities : ndarray
        Probability of each cohort, one entry for each cohort in
        unique_groups_recoded.
    unit_level_weights : ndarray
        Sampling weight of each unit.
    unit_level_groups : ndarray
        Cohort of each unit.
    groups : ndarray
        Cohort of each effect that keepers indexes.
    unique_groups_recoded : ndarray
        Unique cohorts.

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

    # Since each weight divides an estimated cohort probability by the sum over the kept effects, its influence
    # function has a term from the numerator and a term from the shared denominator.
    for i, k in enumerate(keepers):
        group_k = groups[k]
        group_idx = np.where(unique_groups_recoded == group_k)[0][0]
        pg_k = group_probabilities[group_idx]

        if1 = unit_level_weights * (unit_level_groups == group_k).astype(float) - pg_k
        if1 = if1 / sum_pg

        weight_influence_function[:, i] = if1

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
    keep_idx = np.where(keepers)[0] if keepers.dtype == bool else keepers

    influence_function_kept = influence_function[:, keep_idx]
    att_kept = att[keep_idx]

    weighted_inf = influence_function_kept @ weights

    if weight_influence_function is not None:
        weighted_inf += weight_influence_function @ att_kept

    return weighted_inf


def _resolve_clusters(fit_params, clustervars):
    """Return the cluster variables and cluster vector of an aggregation.

    Parameters
    ----------
    fit_params : dict
        Estimation settings of the att_gt fit.
    clustervars : list[str] or None
        Cluster variables passed to the aggregation, or None for those of
        the fit.

    Returns
    -------
    tuple of list[str] or None, and ndarray or None
        Tuple containing:

        - **clustervars**: Cluster variables of the aggregation besides the unit id
        - **cluster**: Cluster of each unit, or None for unit-level inference
    """
    data = fit_params.get("data")
    config = getattr(data, "config", None)
    idname = fit_params.get("idname", getattr(config, "idname", None))
    # A result built without att_gt keeps its cluster variables in its preprocessed data.
    fitted = fit_params["clustervars"] if "clustervars" in fit_params else getattr(config, "clustervars", None)
    fitted = [name for name in fitted or [] if name != idname]

    if clustervars is None:
        # A result that records no clustervars stays unclustered, whatever its preprocessed data names.
        clustervars = None if fit_params.get("clustervars") is None else fitted
    else:
        # In a panel, clusters of one unit each are the units themselves. Since att_gt draws a multiplier for each
        # row of a repeated cross section even when rows share an id, the aggregation drops the id there too.
        clustervars = list(dict.fromkeys(name for name in clustervars if name != idname))
        unknown = [name for name in clustervars if name not in fitted]
        if unknown:
            if fitted:
                choice = f"pass clustervars={fitted!r} to reuse its clusters, or an empty list for unit-level inference"
            else:
                choice = "pass an empty list for unit-level inference"
            raise ValueError(
                f"The att_gt fit did not cluster on {' or '.join(repr(name) for name in unknown)}. Since the "
                "aggregation reuses the clusters of the att_gt fit, pass the cluster variable to att_gt instead. To "
                f"aggregate this fit, {choice}."
            )

    if not clustervars:
        return clustervars, None
    cluster = fit_params.get("cluster")
    if cluster is None:
        cluster = getattr(data, "cluster", None)
    if cluster is None:
        raise ValueError(
            f"The att_gt result records clustervars={clustervars!r} but no cluster for its units. Rerun att_gt or "
            "store the cluster of each unit in estimation_params['cluster']. To aggregate the result as it is, pass "
            "an empty list for unit-level inference."
        )
    return clustervars, cluster


def _cluster_vector(estimation_params):
    """Return the cluster of each unit for the standard errors.

    Parameters
    ----------
    estimation_params : dict
        Estimation settings of the aggregation.

    Returns
    -------
    ndarray or None
        Cluster of each unit, or None for standard errors at the unit level.
    """
    cluster = estimation_params.get("cluster")
    # A cluster variable with one value for each unit, such as a copy of the unit id, makes each unit its own
    # cluster. Since such clusters are the units themselves, the inference stays at the unit level.
    if cluster is None or len(np.unique(cluster)) == len(cluster):
        return None
    return cluster


def _compute_se(
    influence_function,
    n_units,
    estimation_params,
    biters,
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
    biters : int
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
    cluster = _cluster_vector(estimation_params)
    if estimation_params.get("bootstrap", False):
        mboot_result = mboot(
            inf_func=influence_function.reshape(-1, 1),
            n_units=n_units,
            biters=biters,
            alpha=alpha,
            cluster=cluster,
            random_state=random_state,
        )
        standard_error = mboot_result["se"][0]
    elif cluster is None:
        standard_error = np.sqrt(np.mean(influence_function**2) / n_units)
    else:
        standard_error = np.sqrt(np.sum(_cluster_sums(influence_function.reshape(-1, 1), cluster) ** 2)) / n_units

    if standard_error <= np.sqrt(np.finfo(float).eps) * 10:
        standard_error = np.nan

    return standard_error


def _batched_se(inf_func_mat, att_vec, n_units, estimation_params, biters, alpha, random_state):
    """Compute per-column SEs and critical value in a single bootstrap pass.

    Parameters
    ----------
    inf_func_mat : ndarray
        Influence function matrix of shape (n_units, n_cols).
    att_vec : ndarray
        ATT estimates, length n_cols. Columns with NaN ATT are skipped.
    n_units : int
        Number of cross-sectional units.
    estimation_params : dict
        Estimation parameters (bootstrap, uniform_bands, ...).
    biters : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    random_state : int, Generator, or None
        Controls bootstrap randomness.

    Returns
    -------
    tuple of ndarray, float, and ndarray or None
        Tuple containing:

        - **se**: Standard error of each column, NaN where the ATT is NaN
        - **critical_value**: Critical value of the confidence band
        - **bres**: Bootstrap draws, or None without the bootstrap
    """
    n_cols = inf_func_mat.shape[1]
    eps_thresh = np.sqrt(np.finfo(float).eps) * 10

    nan_mask = np.isnan(att_vec)
    valid_cols = ~nan_mask
    se = np.full(n_cols, np.nan)

    if not valid_cols.any():
        return se, _get_z_critical(alpha / 2), None

    valid_inf = inf_func_mat[:, valid_cols]
    use_bootstrap = estimation_params.get("bootstrap", False)
    use_uniform = estimation_params.get("uniform_bands", False)
    cluster = _cluster_vector(estimation_params)
    bres = None

    if not use_bootstrap:
        if cluster is None:
            se_valid = np.sqrt(np.mean(valid_inf**2, axis=0) / n_units)
        else:
            se_valid = np.sqrt(np.sum(_cluster_sums(valid_inf, cluster) ** 2, axis=0)) / n_units
        se_valid = np.where(se_valid <= eps_thresh, np.nan, se_valid)
        se[valid_cols] = se_valid
    else:
        mboot_result = mboot(
            inf_func=valid_inf,
            n_units=n_units,
            biters=biters,
            alpha=alpha,
            cluster=cluster,
            random_state=random_state,
        )
        se_valid = mboot_result["se"]
        se_valid = np.where((~np.isnan(se_valid)) & (se_valid <= eps_thresh), np.nan, se_valid)
        se[valid_cols] = se_valid
        bres = mboot_result["bres"]

    critical_value = _get_z_critical(alpha / 2)
    if use_uniform:
        if not use_bootstrap:
            warnings.warn("Used bootstrap procedure to compute simultaneous confidence band")

        if use_bootstrap:
            cv = mboot_result["crit_val"]
        else:
            # Since the analytic standard errors sum within the clusters, the draws behind their band do too.
            band_result = mboot(
                inf_func=valid_inf,
                n_units=n_units,
                biters=biters,
                alpha=alpha,
                cluster=cluster,
                random_state=random_state,
            )
            cv = band_result["crit_val"]
            bres = band_result["bres"]

        if not np.isfinite(cv):
            warnings.warn(
                "Simultaneous critical value is NaN, likely because the standard errors "
                "could not be computed. Falling back to pointwise confidence intervals."
            )
            estimation_params["uniform_bands"] = False
        elif cv < critical_value:
            warnings.warn(
                "The simultaneous confidence band is narrower than the pointwise "
                "confidence interval. Since a band that covers every effect at once can't "
                "be narrower, aggte falls back to pointwise confidence intervals."
            )
            estimation_params["uniform_bands"] = False
        else:
            critical_value = cv
            if cv >= 7:
                warnings.warn(
                    "Simultaneous critical value is very large and may be unreliable. "
                    "This typically happens when the number of observations per group is "
                    "small or the outcomes vary little. Pass cband=False for pointwise "
                    "confidence intervals instead."
                )

    return se, critical_value, bres


def _get_z_critical(alpha):
    """Get critical value from standard normal distribution."""
    return stats.norm.ppf(1 - alpha)


def _orig2t(orig, unique_original_times_and_groups, recoded_times):
    """Convert original time value to sequential time value."""
    try:
        idx = np.where(unique_original_times_and_groups == orig)[0][0]
        return recoded_times[idx]
    except IndexError:
        return np.nan


def _orig2t_array(arr, unique_original_times_and_groups, recoded_times):
    """Vectorized version of _orig2t for array inputs."""
    arr = np.asarray(arr)
    n_lookup = len(unique_original_times_and_groups)
    if n_lookup == 0 or len(arr) == 0:
        return np.full(len(arr), np.nan)
    indices = np.searchsorted(unique_original_times_and_groups, arr)
    clamped = np.minimum(indices, n_lookup - 1)
    valid = unique_original_times_and_groups[clamped] == arr
    result = np.full(len(arr), np.nan)
    result[valid] = recoded_times[clamped[valid]]
    return result


def _t2orig(t, unique_original_times_and_groups, recoded_times):
    """Convert sequential time value back to original time value."""
    try:
        idx = np.where(recoded_times == t)[0][0]
        return unique_original_times_and_groups[idx]
    except IndexError:
        return 0
