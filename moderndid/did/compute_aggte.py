"""Compute Aggregated Treatment Effect Parameters."""

import warnings

import numpy as np
from scipy import stats

from .container import AGGTEResult
from .mboot import _cluster_sums, _mboot_from_draws, mboot


def compute_aggte(
    multi_period_result,
    aggregation_type="group",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    na_rm=False,
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

    Each aggregated influence function takes the same linear combination of
    every unit's row that :class:`~moderndid.InfluenceMoments` defines. Since a
    fit on a Spark DataFrame keeps only the sums of those rows, a call that
    needs draws the fit did not store, such as more draws, another seed, or
    draws by unit from a clustered fit, raises an error.

    Parameters
    ----------
    multi_period_result : MPResult
        Multi-period result object containing group-time ATTs and their
        influence functions or influence moments.
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
    na_rm : bool, default=False
        Whether to drop group-time effects that are NaN before aggregating.
        Without it, a NaN effect stops the aggregation with an error.
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
        - **vcov_analytical**: Asymptotic variance-covariance matrix of the disaggregated effects, or of the overall
          effect for 'simple'
    """
    groups = multi_period_result.groups
    times = multi_period_result.times
    att = multi_period_result.att_gt
    influence_function = multi_period_result.influence_func
    # A fit on a Spark DataFrame keeps sums over its units in place of the influence function.
    moments = getattr(multi_period_result, "influence_moments", None) if influence_function is None else None
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
    if moments is None:
        estimation_params["clustervars"], estimation_params["cluster"] = _resolve_clusters(
            multi_period_result.estimation_params, clustervars
        )
    else:
        estimation_params["clustervars"] = _resolve_clustervars(multi_period_result.estimation_params, clustervars)
        estimation_params["cluster"] = None

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

    cell_index = np.arange(len(att))
    if na_rm:
        notna = ~np.isnan(att)
        if not notna.any():
            raise ValueError(
                "All att_gt estimates are NaN. The att_gt warnings name the cells that failed to estimate. Fewer "
                "covariates or control_group='notyettreated' can recover them."
            )
        groups = groups[notna]
        times = times[notna]
        att = att[notna]
        cell_index = cell_index[notna]
        if moments is None:
            influence_function = influence_function[:, notna]
    elif np.any(np.isnan(att)):
        raise ValueError("Missing values at att_gt found. If you want to remove these, set na_rm=True in aggte.")

    if moments is not None:
        # As for an in-memory fit, clusters of one unit each leave the inference at the unit level.
        fit_clustered = moments.cluster_gram is not None and moments.n_clusters < n_units
        clustered = bool(estimation_params["clustervars"]) and fit_clustered
        # Since the simple aggregation has no disaggregated effects, it draws only for bootstrap standard errors.
        if bootstrap or (estimation_params.get("uniform_bands", False) and aggregation_type != "simple"):
            _check_stored_draws(moments, clustervars, clustered, fit_clustered, bootstrap, biters, random_state)
        estimation_params["n_units"] = n_units
        estimation_params["n_clusters"] = (
            (moments.n_clusters if moments.cluster_gram is not None else n_units)
            if estimation_params["clustervars"]
            else None
        )

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

    cohort_weights = None
    if moments is not None:
        rows = _moment_rows(moments, len(multi_period_result.att_gt), cell_index, unique_original_groups)
        # Since every unit's row ends in a one, the last row of the Gram matrix sums each cohort's weight column.
        group_probabilities = moments.gram[-1, rows[len(att) : -1]] / n_units
    else:
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
                unit_level_groups = None

        if unit_level_groups is not None and unit_level_weights is not None:
            unit_level_groups_recoded = _orig2t_array(
                unit_level_groups, unique_original_times_and_groups, recoded_times
            )

            cohort_weight_columns = [
                unit_level_weights * (unit_level_groups_recoded == g) for g in unique_groups_recoded
            ]
            group_probabilities = np.array([np.mean(column) for column in cohort_weight_columns])
            cohort_weights = _stack_columns(cohort_weight_columns, len(unit_level_groups))

    observation_group_probabilities = np.zeros(len(groups))
    for i, g in enumerate(unique_groups_recoded):
        mask = groups == g
        observation_group_probabilities[mask] = group_probabilities[i]

    # The coefficients of an effect weigh the kept cells, the weight column of each of their cohorts, and the
    # constant, in that order.
    n_basis = len(att) + len(unique_groups_recoded) + 1
    cell_coefficients = np.eye(n_basis, len(att))
    cohort_rows = None
    if moments is not None or cohort_weights is not None:
        cohort_rows = len(att) + np.arange(len(unique_groups_recoded))

    if moments is not None:
        evaluator = _MomentEvaluator(moments, rows, n_units, clustered)
    else:
        evaluator = _UnitLevelEvaluator(influence_function, cohort_weights, n_units, _cluster_vector(estimation_params))

    post_treatment = (groups <= times) & (times <= groups + max_e)

    if aggregation_type == "simple":
        return _compute_simple_att(
            att=att,
            cell_coefficients=cell_coefficients,
            evaluator=evaluator,
            post_treatment=post_treatment,
            observation_group_probabilities=observation_group_probabilities,
            groups=groups,
            times=times,
            max_e=max_e,
            estimation_params=estimation_params,
            alpha=alpha,
            biters=biters,
            unique_groups_recoded=unique_groups_recoded,
            group_probabilities=group_probabilities,
            cohort_rows=cohort_rows,
            random_state=random_state,
        )

    if aggregation_type == "group":
        return _compute_group_att(
            att=att,
            cell_coefficients=cell_coefficients,
            evaluator=evaluator,
            groups=groups,
            times=times,
            unique_groups_recoded=unique_groups_recoded,
            unique_original_groups=unique_original_groups,
            group_probabilities=group_probabilities,
            max_e=max_e,
            estimation_params=estimation_params,
            alpha=alpha,
            biters=biters,
            cohort_rows=cohort_rows,
            random_state=random_state,
        )

    if aggregation_type == "dynamic":
        return _compute_dynamic_att(
            att=att,
            cell_coefficients=cell_coefficients,
            evaluator=evaluator,
            groups=groups,
            times=times,
            original_groups=original_groups,
            original_times=original_times,
            observation_group_probabilities=observation_group_probabilities,
            balance_e=balance_e,
            min_e=min_e,
            max_e=max_e,
            estimation_params=estimation_params,
            alpha=alpha,
            biters=biters,
            unique_groups_recoded=unique_groups_recoded,
            group_probabilities=group_probabilities,
            cohort_rows=cohort_rows,
            unique_original_times_and_groups=unique_original_times_and_groups,
            recoded_times=recoded_times,
            random_state=random_state,
        )

    if aggregation_type == "calendar":
        return _compute_calendar_att(
            att=att,
            cell_coefficients=cell_coefficients,
            evaluator=evaluator,
            groups=groups,
            times=times,
            unique_times_recoded=unique_times_recoded,
            observation_group_probabilities=observation_group_probabilities,
            estimation_params=estimation_params,
            alpha=alpha,
            biters=biters,
            unique_groups_recoded=unique_groups_recoded,
            group_probabilities=group_probabilities,
            cohort_rows=cohort_rows,
            unique_original_times_and_groups=unique_original_times_and_groups,
            recoded_times=recoded_times,
            random_state=random_state,
        )

    raise ValueError(f"Unexpected aggregation_type: {aggregation_type}")


def _compute_simple_att(
    att,
    cell_coefficients,
    evaluator,
    post_treatment,
    observation_group_probabilities,
    groups,
    times,
    max_e,
    estimation_params,
    alpha,
    biters,
    unique_groups_recoded,
    group_probabilities,
    cohort_rows,
    random_state=None,
):
    """Compute simple ATT by averaging all post-treatment effects."""
    n_basis = cell_coefficients.shape[0]
    if not post_treatment.any():
        _warn_without_post_treatment_effects(times - groups, max_e)
        no_effect = evaluator.evaluate(np.zeros((n_basis, 1)))
        return AGGTEResult(
            overall_att=np.nan,
            overall_se=np.nan,
            aggregation_type="simple",
            influence_func=evaluator.unit_values(no_effect[:, 0]),
            influence_func_overall=evaluator.unit_values(no_effect[:, 0]),
            estimation_params=estimation_params,
            call_info={"function": "compute_aggte", "type": "simple"},
            vcov_analytical=evaluator.vcov(no_effect),
        )

    weights = observation_group_probabilities[post_treatment] / observation_group_probabilities[post_treatment].sum()
    simple_att = np.sum(att[post_treatment] * weights)

    if np.isnan(simple_att):
        simple_att = np.nan
        simple_se = np.nan
        simple_influence = evaluator.evaluate(np.zeros((n_basis, 1)))
    else:
        if cohort_rows is not None:
            weight_coefficients = _weight_coefficients(
                keepers=np.where(post_treatment)[0],
                group_probabilities=group_probabilities,
                groups=groups,
                unique_groups_recoded=unique_groups_recoded,
                cohort_rows=cohort_rows,
                n_basis=n_basis,
            )
        else:
            weight_coefficients = None

        simple_coefficients = _aggregate_coefficients(
            att=att,
            coefficients=cell_coefficients,
            keepers=post_treatment,
            weights=weights,
            weight_coefficients=weight_coefficients,
        )
        simple_influence = evaluator.evaluate(simple_coefficients[:, None])

        simple_se = _compute_se(
            evaluator=evaluator,
            influence=simple_influence,
            estimation_params=estimation_params,
            biters=biters,
            alpha=alpha,
            random_state=random_state,
        )

    return AGGTEResult(
        overall_att=simple_att,
        overall_se=simple_se,
        aggregation_type="simple",
        influence_func=evaluator.unit_values(simple_influence[:, 0]),
        influence_func_overall=evaluator.unit_values(simple_influence[:, 0]),
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "simple"},
        vcov_analytical=evaluator.vcov(simple_influence),
    )


def _compute_group_att(
    att,
    cell_coefficients,
    evaluator,
    groups,
    times,
    unique_groups_recoded,
    unique_original_groups,
    group_probabilities,
    max_e,
    estimation_params,
    alpha,
    biters,
    cohort_rows,
    random_state=None,
):
    """Compute group-specific ATTs."""
    n_basis = cell_coefficients.shape[0]
    # A cohort without a post-treatment effect in the window has nothing to average, whether the varying base
    # dropped its failed cells or na_rm removed them.
    has_effect = np.array(
        [np.any((groups == g) & (g <= times) & (times <= g + max_e)) for g in unique_groups_recoded], dtype=bool
    )
    unique_groups_recoded = unique_groups_recoded[has_effect]
    unique_original_groups = unique_original_groups[has_effect]
    group_probabilities = group_probabilities[has_effect]
    if cohort_rows is not None:
        cohort_rows = cohort_rows[has_effect]

    group_att = np.zeros(len(unique_groups_recoded))
    group_coefficients = []

    for i, g in enumerate(unique_groups_recoded):
        mask = (groups == g) & (g <= times) & (times <= g + max_e)
        group_att[i] = np.mean(att[mask])
        group_coefficients.append(
            _aggregate_coefficients(
                att=att,
                coefficients=cell_coefficients,
                keepers=mask,
                weights=np.ones(mask.sum()) / mask.sum(),
                weight_coefficients=None,
            )
        )

    group_coefficients = _stack_columns(group_coefficients, n_basis)
    group_influence = evaluator.evaluate(group_coefficients)
    group_att_clean = np.where(np.isnan(group_att), 0, group_att)
    valid_group_mask = ~np.isnan(group_att)

    group_se, critical_value, _ = _batched_se(
        evaluator,
        group_influence,
        group_att,
        estimation_params,
        biters,
        alpha,
        random_state,
    )

    if valid_group_mask.any():
        valid_probs = group_probabilities[valid_group_mask]
        overall_att = np.sum(group_att_clean[valid_group_mask] * valid_probs) / np.sum(valid_probs)

        valid_indices = np.where(valid_group_mask)[0]
        if cohort_rows is not None:
            weight_coefficients = _weight_coefficients(
                keepers=valid_indices,
                group_probabilities=group_probabilities,
                groups=unique_groups_recoded,
                unique_groups_recoded=unique_groups_recoded,
                cohort_rows=cohort_rows,
                n_basis=n_basis,
            )
        else:
            weight_coefficients = None

        overall_coefficients = _aggregate_coefficients(
            att=group_att_clean,
            coefficients=group_coefficients,
            keepers=valid_group_mask,
            weights=valid_probs / valid_probs.sum(),
            weight_coefficients=weight_coefficients,
        )
    else:
        _warn_without_post_treatment_effects(times - groups, max_e)
        overall_att = np.nan
        overall_coefficients = np.zeros(n_basis)

    overall_influence = evaluator.evaluate(overall_coefficients[:, None])
    overall_se = _compute_se(
        evaluator=evaluator,
        influence=overall_influence,
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
        influence_func=evaluator.unit_values(group_influence),
        influence_func_overall=evaluator.unit_values(overall_influence[:, 0]),
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "group"},
        vcov_analytical=evaluator.vcov(group_influence),
    )


def _compute_dynamic_att(
    att,
    cell_coefficients,
    evaluator,
    groups,
    times,
    original_groups,
    original_times,
    observation_group_probabilities,
    balance_e,
    min_e,
    max_e,
    estimation_params,
    alpha,
    biters,
    unique_groups_recoded,
    group_probabilities,
    cohort_rows,
    unique_original_times_and_groups,
    recoded_times,
    random_state=None,
):
    """Compute dynamic (event-study) treatment effects."""
    n_basis = cell_coefficients.shape[0]
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
    dynamic_coefficients = []

    for i, e in enumerate(unique_event_times):
        mask = (event_times == e) & include_balanced
        weights_e = observation_group_probabilities[mask] / observation_group_probabilities[mask].sum()
        dynamic_att[i] = np.sum(att[mask] * weights_e)

        if cohort_rows is not None:
            weight_coefficients_e = _weight_coefficients(
                keepers=np.where(mask)[0],
                group_probabilities=group_probabilities,
                groups=groups,
                unique_groups_recoded=unique_groups_recoded,
                cohort_rows=cohort_rows,
                n_basis=n_basis,
            )
        else:
            weight_coefficients_e = None

        dynamic_coefficients.append(
            _aggregate_coefficients(
                att=att,
                coefficients=cell_coefficients,
                keepers=mask,
                weights=weights_e,
                weight_coefficients=weight_coefficients_e,
            )
        )

    dynamic_coefficients = _stack_columns(dynamic_coefficients, n_basis)
    dynamic_influence = evaluator.evaluate(dynamic_coefficients)

    dynamic_se, critical_value, _ = _batched_se(
        evaluator,
        dynamic_influence,
        dynamic_att,
        estimation_params,
        biters,
        alpha,
        random_state,
    )

    post_mask = unique_event_times >= 0
    if post_mask.any():
        overall_att = np.nanmean(dynamic_att[post_mask])

        overall_coefficients = _aggregate_coefficients(
            att=dynamic_att[post_mask],
            coefficients=dynamic_coefficients[:, post_mask],
            keepers=np.ones(post_mask.sum(), dtype=bool),
            weights=np.ones(post_mask.sum()) / post_mask.sum(),
            weight_coefficients=None,
        )
        overall_influence = evaluator.evaluate(overall_coefficients[:, None])

        overall_se = _compute_se(
            evaluator=evaluator,
            influence=overall_influence,
            estimation_params=estimation_params,
            biters=biters,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        _warn_without_post_treatment_effects(event_times[include_balanced], max_e)
        overall_att = np.nan
        overall_se = np.nan
        overall_influence = evaluator.evaluate(np.zeros((n_basis, 1)))

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="dynamic",
        event_times=unique_event_times,
        att_by_event=dynamic_att,
        se_by_event=dynamic_se,
        critical_values=np.full(len(unique_event_times), critical_value),
        influence_func=evaluator.unit_values(dynamic_influence),
        influence_func_overall=evaluator.unit_values(overall_influence[:, 0]),
        min_event_time=int(min_e) if np.isfinite(min_e) else None,
        max_event_time=int(max_e) if np.isfinite(max_e) else None,
        balanced_event_threshold=balance_e,
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "dynamic"},
        vcov_analytical=evaluator.vcov(dynamic_influence),
    )


def _compute_calendar_att(
    att,
    cell_coefficients,
    evaluator,
    groups,
    times,
    unique_times_recoded,
    observation_group_probabilities,
    estimation_params,
    alpha,
    biters,
    unique_groups_recoded,
    group_probabilities,
    cohort_rows,
    unique_original_times_and_groups,
    recoded_times,
    random_state=None,
):
    """Compute calendar time effects."""
    n_basis = cell_coefficients.shape[0]
    # A period enters only when one of its cells is treated. Since a universal base keeps each cohort's reference
    # cell, a cohort whose treated cells all failed would otherwise add periods with nothing to average.
    calendar_times = np.array([t for t in unique_times_recoded if np.any((times == t) & (groups <= t))])

    original_calendar_times = np.array(
        [_t2orig(t, unique_original_times_and_groups, recoded_times) for t in calendar_times]
    )

    calendar_att = np.zeros(len(calendar_times))
    calendar_coefficients = []

    for i, t in enumerate(calendar_times):
        mask = (times == t) & (groups <= t)
        weights_t = observation_group_probabilities[mask] / observation_group_probabilities[mask].sum()
        calendar_att[i] = np.sum(att[mask] * weights_t)

        if cohort_rows is not None:
            weight_coefficients_t = _weight_coefficients(
                keepers=np.where(mask)[0],
                group_probabilities=group_probabilities,
                groups=groups,
                unique_groups_recoded=unique_groups_recoded,
                cohort_rows=cohort_rows,
                n_basis=n_basis,
            )
        else:
            weight_coefficients_t = None

        calendar_coefficients.append(
            _aggregate_coefficients(
                att=att,
                coefficients=cell_coefficients,
                keepers=mask,
                weights=weights_t,
                weight_coefficients=weight_coefficients_t,
            )
        )

    calendar_coefficients = _stack_columns(calendar_coefficients, n_basis)
    calendar_influence = evaluator.evaluate(calendar_coefficients)

    calendar_se, critical_value, _ = _batched_se(
        evaluator,
        calendar_influence,
        calendar_att,
        estimation_params,
        biters,
        alpha,
        random_state,
    )

    n_valid = (~np.isnan(calendar_att)).sum()
    if n_valid > 0:
        overall_att = np.nanmean(calendar_att)
        overall_coefficients = _aggregate_coefficients(
            att=calendar_att,
            coefficients=calendar_coefficients,
            keepers=~np.isnan(calendar_att),
            weights=np.ones(n_valid) / n_valid,
            weight_coefficients=None,
        )
        overall_influence = evaluator.evaluate(overall_coefficients[:, None])

        overall_se = _compute_se(
            evaluator=evaluator,
            influence=overall_influence,
            estimation_params=estimation_params,
            biters=biters,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        _warn_without_post_treatment_effects(times - groups, np.inf)
        overall_att = np.nan
        overall_se = np.nan
        overall_influence = evaluator.evaluate(np.zeros((n_basis, 1)))

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="calendar",
        event_times=original_calendar_times,
        att_by_event=calendar_att,
        se_by_event=calendar_se,
        critical_values=np.full(len(calendar_times), critical_value),
        influence_func=evaluator.unit_values(calendar_influence),
        influence_func_overall=evaluator.unit_values(overall_influence[:, 0]),
        estimation_params=estimation_params,
        call_info={"function": "compute_aggte", "type": "calendar"},
        vcov_analytical=evaluator.vcov(calendar_influence),
    )


def _stack_columns(columns, n_rows):
    """Stack vectors as the columns of a matrix.

    Parameters
    ----------
    columns : list of ndarray
        Vectors of length n_rows, one for each column. The list may be empty.
    n_rows : int
        Number of rows of the matrix.

    Returns
    -------
    ndarray
        Matrix of shape (n_rows, len(columns)).
    """
    return np.column_stack(columns) if columns else np.zeros((n_rows, 0))


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


def _weight_coefficients(
    keepers,
    group_probabilities,
    groups,
    unique_groups_recoded,
    cohort_rows,
    n_basis,
):
    """Compute the coefficients of the influence function adjustment for estimated weights.

    This computes the extra term in the influence function that arises
    from having to estimate the weights (group probabilities). Since the
    estimated probability of a cohort is the mean of its weight column, the
    term combines the weight columns of the kept cohorts and the constant.

    Parameters
    ----------
    keepers : ndarray
        Indices of the effects that the aggregation averages.
    group_probabilities : ndarray
        Probability of each cohort, one entry for each cohort in
        unique_groups_recoded.
    groups : ndarray
        Cohort of each effect that keepers indexes.
    unique_groups_recoded : ndarray
        Unique cohorts.
    cohort_rows : ndarray
        Basis row of the weight column of each cohort in unique_groups_recoded.
    n_basis : int
        Number of basis rows. The last row is the constant.

    Returns
    -------
    ndarray
        Coefficients of the weight influence function of shape (n_basis, len(keepers)).
    """
    n_keepers = len(keepers)

    weight_coefficients = np.zeros((n_basis, n_keepers))

    pg_keepers = np.zeros(n_keepers)
    for i, k in enumerate(keepers):
        group_k = groups[k]
        group_idx = np.where(unique_groups_recoded == group_k)[0][0]
        pg_keepers[i] = group_probabilities[group_idx]

    sum_pg = pg_keepers.sum()

    # Since each weight divides an estimated cohort probability by the sum over the kept effects, its influence
    # function has a term from the numerator and a term from the shared denominator.
    sum_term = np.zeros(n_basis)
    for i, k in enumerate(keepers):
        group_k = groups[k]
        group_idx = np.where(unique_groups_recoded == group_k)[0][0]

        centered = np.zeros(n_basis)
        centered[cohort_rows[group_idx]] = 1.0
        centered[-1] = -group_probabilities[group_idx]

        weight_coefficients[:, i] = centered / sum_pg
        sum_term += centered

    weight_coefficients -= np.outer(sum_term / (sum_pg**2), pg_keepers)

    return weight_coefficients


def _aggregate_coefficients(
    att,
    coefficients,
    keepers,
    weights,
    weight_coefficients=None,
):
    """Combine the coefficients of influence functions into those of an aggregated effect.

    Parameters
    ----------
    att : ndarray
        Vector of ATT estimates.
    coefficients : ndarray
        Coefficients of the influence function of each ATT, of shape (n_basis, n_att).
    keepers : ndarray
        Boolean mask or indices of which ATTs to include.
    weights : ndarray
        Weights to apply to each kept ATT.
    weight_coefficients : ndarray, optional
        Coefficients of the weight influence function from estimating group probabilities.

    Returns
    -------
    ndarray
        Coefficients of the aggregated influence function of shape (n_basis,).
    """
    keep_idx = np.where(keepers)[0] if keepers.dtype == bool else keepers

    coefficients_kept = coefficients[:, keep_idx]
    att_kept = att[keep_idx]

    aggregated = coefficients_kept @ weights

    if weight_coefficients is not None:
        aggregated += weight_coefficients @ att_kept

    return aggregated


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
    clustervars = _resolve_clustervars(fit_params, clustervars)

    if not clustervars:
        return clustervars, None
    cluster = fit_params.get("cluster")
    if cluster is None:
        cluster = getattr(fit_params.get("data"), "cluster", None)
    if cluster is None:
        raise ValueError(
            f"The att_gt result records clustervars={clustervars!r} but no cluster for its units. Rerun att_gt or "
            "store the cluster of each unit in estimation_params['cluster']. To aggregate the result as it is, pass "
            "an empty list for unit-level inference."
        )
    return clustervars, cluster


def _resolve_clustervars(fit_params, clustervars):
    """Return the cluster variables of an aggregation besides the unit id after checking them against the fit."""
    data = fit_params.get("data")
    config = getattr(data, "config", None)
    idname = fit_params.get("idname", getattr(config, "idname", None))
    # A result built without att_gt keeps its cluster variables in its preprocessed data.
    fitted = fit_params["clustervars"] if "clustervars" in fit_params else getattr(config, "clustervars", None)
    fitted = [name for name in fitted or [] if name != idname]

    if clustervars is None:
        # A result that records no clustervars stays unclustered, whatever its preprocessed data names.
        return None if fit_params.get("clustervars") is None else fitted

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
    return clustervars


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


def _check_stored_draws(moments, clustervars, clustered, fit_clustered, bootstrap, biters, random_state):
    """Raise a ValueError when an aggregation needs bootstrap draws that a fit on a Spark DataFrame did not store."""
    if moments.bootstrap_sums is None:
        raise ValueError(
            f"{'boot' if bootstrap else 'cband'}=True needs bootstrap draws that this att_gt fit on a Spark "
            "DataFrame did not store. Pass boot=False and cband=False for analytic standard errors. To bootstrap, "
            "rerun att_gt with boot=True."
        )
    if clustered != fit_clustered:
        raise ValueError(
            f"clustervars={clustervars!r} asks for unit-level bootstrap draws. Since this att_gt fit on a Spark "
            "DataFrame drew one multiplier for each cluster, it holds no draws for single units. Pass boot=False "
            "and cband=False for unit-level analytic standard errors. To bootstrap by unit, rerun att_gt without "
            "clustervars."
        )
    n_draws = moments.bootstrap_sums.shape[0]
    if biters > n_draws:
        raise ValueError(
            f"biters={biters} is more than the {n_draws} bootstrap draws that this att_gt fit on a Spark DataFrame "
            f"stored. Pass biters={n_draws} or fewer. For more draws, rerun att_gt with biters={biters}."
        )
    if random_state is not None and random_state != moments.random_state:
        raise ValueError(
            f"random_state={random_state!r} is not the seed {moments.random_state!r} of the bootstrap draws that "
            f"this att_gt fit on a Spark DataFrame stored. Pass random_state={moments.random_state!r} or leave it "
            "out. For other draws, rerun att_gt with that random_state."
        )


def _moment_rows(moments, n_cells, cell_index, cohorts):
    """Return the rows of the influence moments that the basis of an aggregation uses.

    Parameters
    ----------
    moments : InfluenceMoments
        Sums over units of the fit.
    n_cells : int
        Number of group-time effects of the fit.
    cell_index : ndarray
        Index of each group-time effect that the aggregation keeps.
    cohorts : ndarray
        Distinct cohorts of the kept effects in increasing order.

    Returns
    -------
    ndarray
        Rows of the kept effects, their cohorts' weight columns, and the constant, in that order.
    """
    columns = np.searchsorted(moments.cohorts, cohorts)
    found = columns < len(moments.cohorts)
    found[found] = moments.cohorts[columns[found]] == cohorts[found]
    if not found.all():
        raise ValueError(
            f"The influence moments of the att_gt result hold no weight column for cohort {cohorts[~found][0]}. "
            "Rerun att_gt to rebuild them."
        )
    return np.concatenate([cell_index, n_cells + columns, [n_cells + len(moments.cohorts)]])


class _UnitLevelEvaluator:
    """Evaluate aggregation coefficients on the unit rows of an in-memory fit."""

    def __init__(self, influence_function, cohort_weights, n_units, cluster):
        self.influence_function = influence_function
        self.cohort_weights = cohort_weights
        self.n_units = n_units
        self.cluster = cluster

    def evaluate(self, coefficients):
        """Return the influence function of each effect that a column of coefficients describes."""
        n_cells = self.influence_function.shape[1]
        influence = np.empty((self.influence_function.shape[0], coefficients.shape[1]))
        for j in range(coefficients.shape[1]):
            # Since an effect averages only some cells, the product reads only the columns it weighs.
            used = np.flatnonzero(coefficients[:n_cells, j])
            influence[:, j] = self.influence_function[:, used] @ coefficients[used, j]
        if self.cohort_weights is not None:
            influence += self.cohort_weights @ coefficients[n_cells:-1]
        influence += coefficients[-1]
        return influence

    def unit_values(self, influence):
        """Return the influence functions for the result."""
        return influence

    def standard_errors(self, influence):
        """Return the analytic standard error of the effect that each column describes."""
        if self.cluster is None:
            return np.sqrt(np.mean(influence**2, axis=0) / self.n_units)
        return np.sqrt(np.sum(_cluster_sums(influence, self.cluster) ** 2, axis=0)) / self.n_units

    def vcov(self, influence):
        """Return the asymptotic variance-covariance matrix of the effects that the columns describe."""
        sums = influence if self.cluster is None else _cluster_sums(influence, self.cluster)
        return sums.T @ sums / self.n_units

    def bootstrap(self, influence, biters, alpha, random_state):
        """Return the multiplier bootstrap of the columns."""
        return mboot(
            inf_func=influence,
            n_units=self.n_units,
            biters=biters,
            alpha=alpha,
            cluster=self.cluster,
            random_state=random_state,
        )


class _MomentEvaluator:
    """Evaluate aggregation coefficients on the sums over units that a fit on a Spark DataFrame keeps."""

    def __init__(self, moments, rows, n_units, clustered):
        gram = moments.cluster_gram if clustered else moments.gram
        self.gram = gram[np.ix_(rows, rows)]
        self.bootstrap_sums = None if moments.bootstrap_sums is None else moments.bootstrap_sums[:, rows]
        self.n_units = n_units
        # As in mboot, a clustered draw gives each cluster's sum one multiplier.
        self.n_draw_rows = moments.n_clusters if clustered else n_units
        self.draw_scale = n_units / np.sqrt(moments.n_clusters) if clustered else np.sqrt(n_units)

    def evaluate(self, coefficients):
        """Return the coefficients unchanged for the sums to evaluate."""
        return coefficients

    def unit_values(self, influence):
        """Return None since the fit keeps no influence function for each unit."""
        return None

    def standard_errors(self, coefficients):
        """Return the analytic standard error of the effect that each column describes."""
        quadratic = np.sum(coefficients * (self.gram @ coefficients), axis=0)
        # Since rounding can leave the variance of a constant effect just below zero, it stops at zero.
        return np.sqrt(np.maximum(quadratic, 0)) / self.n_units

    def vcov(self, coefficients):
        """Return the asymptotic variance-covariance matrix of the effects that the columns describe."""
        covariance = coefficients.T @ (self.gram @ coefficients) / self.n_units
        # Since the two products round differently, the average with the transpose keeps the matrix symmetric.
        return (covariance + covariance.T) / 2

    def bootstrap(self, coefficients, biters, alpha, random_state):
        """Return the multiplier bootstrap of the columns from the stored draws."""
        draws = self.bootstrap_sums[:biters] @ coefficients / np.sqrt(self.n_draw_rows)
        return _mboot_from_draws(draws, self.draw_scale, alpha)


def _compute_se(
    evaluator,
    influence,
    estimation_params,
    biters,
    alpha,
    random_state=None,
):
    """Compute the standard error of one aggregated effect.

    Parameters
    ----------
    evaluator : _UnitLevelEvaluator or _MomentEvaluator
        Evaluator of the influence functions of the aggregation.
    influence : ndarray
        Column that ``evaluator.evaluate`` returns for the effect.
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
    if estimation_params.get("bootstrap", False):
        standard_error = evaluator.bootstrap(influence, biters, alpha, random_state)["se"][0]
    else:
        standard_error = evaluator.standard_errors(influence)[0]

    if standard_error <= np.sqrt(np.finfo(float).eps) * 10:
        standard_error = np.nan

    return standard_error


def _batched_se(evaluator, influence, att_vec, estimation_params, biters, alpha, random_state):
    """Compute per-column SEs and critical value in a single bootstrap pass.

    Parameters
    ----------
    evaluator : _UnitLevelEvaluator or _MomentEvaluator
        Evaluator of the influence functions of the aggregation.
    influence : ndarray
        Columns that ``evaluator.evaluate`` returns, one for each effect.
    att_vec : ndarray
        ATT estimates, one for each column. Columns with NaN ATT are skipped.
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
    n_cols = influence.shape[1]
    eps_thresh = np.sqrt(np.finfo(float).eps) * 10

    nan_mask = np.isnan(att_vec)
    valid_cols = ~nan_mask
    se = np.full(n_cols, np.nan)

    if not valid_cols.any():
        return se, _get_z_critical(alpha / 2), None

    valid_inf = influence[:, valid_cols]
    use_bootstrap = estimation_params.get("bootstrap", False)
    use_uniform = estimation_params.get("uniform_bands", False)
    bres = None

    if not use_bootstrap:
        se_valid = evaluator.standard_errors(valid_inf)
        se_valid = np.where(se_valid <= eps_thresh, np.nan, se_valid)
        se[valid_cols] = se_valid
    else:
        mboot_result = evaluator.bootstrap(valid_inf, biters, alpha, random_state)
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
            band_result = evaluator.bootstrap(valid_inf, biters, alpha, random_state)
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
