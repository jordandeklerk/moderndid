"""Utility functions for panel and repeated cross-section data."""

import re
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd


def complete_data(
    data: pd.DataFrame,
    *args: str,
    formula: str | None = None,
    variables: list[str] | None = None,
    min_periods: int | str = "all",
) -> pd.DataFrame:
    """Filter out entities with too few observations.

    This function allows you to define a minimum number of periods
    and exclude all entities with fewer observations than that threshold.
    Only complete cases (no missing values) are considered when counting
    observations.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time).
    *args : str
        Variable names to check for completeness. If none provided,
        all columns are checked.
    formula : str | None, default None
        Formula string to extract variables from (alternative to args).
    variables : list[str] | None, default None
        List of variable names (alternative to args and formula).
    min_periods : int | str, default "all"
        Minimum number of observations to keep. Can be:

        - "all": Keep only entities observed in all time periods
        - int: Keep entities with at least this many observations

    Returns
    -------
    pd.DataFrame
        Filtered panel data containing only entities with sufficient observations.

    See Also
    --------
    make_panel_balanced : Create balanced panel from unbalanced data.
    fill_panel_gaps : Fill gaps in panel data time series.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, _ = data.index.names

    if args:
        cols = list(args)
        for col in cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")
    elif formula is not None:
        cols = extract_vars_from_formula(formula)
        missing_cols = [col for col in cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
    elif variables is not None:
        cols = variables
        for col in cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")
    else:
        cols = data.columns.tolist()

    selected_data = data[cols]
    complete_mask = selected_data.notna().all(axis=1)
    complete_obs = data[complete_mask]
    entity_counts = complete_obs.groupby(entity_name).size()

    if min_periods == "all":
        if len(entity_counts) == 0:
            warnings.warn("No complete observations found in data.")
            return data.iloc[0:0]
        min_periods_val = entity_counts.max()
    else:
        min_periods_val = int(min_periods)

    keep_entities = entity_counts[entity_counts >= min_periods_val].index

    if len(keep_entities) == 0:
        warnings.warn(f"No entities have {min_periods_val} complete observations.")
        return data.iloc[0:0]

    result = data[data.index.get_level_values(entity_name).isin(keep_entities)]

    return result


def widen_panel(
    data: pd.DataFrame,
    separator: str = "_",
    ignore_attributes: bool = False,
    varying: list[str] | None = None,
) -> pd.DataFrame:
    """Convert long panel data to wide format.

    Transforms panel data from long format (one row per entity-time) to wide
    format (one row per entity, time-varying variables as separate columns).
    Automatically detects time-varying vs time-invariant variables.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time).
    separator : str, default "_"
        Character(s) to separate variable name and time period in wide format.
        For example, "var" becomes "var_1", "var_2", etc.
    ignore_attributes : bool, default False
        If True, re-checks which variables are time-varying rather than using
        stored attributes.
    varying : list[str] | None, default None
        Explicitly specify time-varying variables. If None, automatically detects.

    Returns
    -------
    pd.DataFrame
        Wide format data with one row per entity.

    See Also
    --------
    long_panel : Convert wide format to long format.
    identify_time_varying_covariates : Identify time-varying variables.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, time_name = data.index.names

    if hasattr(data, "_varying") and not ignore_attributes:
        varying_vars = getattr(data, "_varying")
        constant_vars = getattr(data, "_constant")
    else:
        all_vars = data.columns.tolist()

        if varying is None:
            varying_vars = are_varying(data, all_vars, return_names=True)
            constant_vars = [v for v in all_vars if v not in varying_vars]
        else:
            varying_vars = varying
            constant_vars = [v for v in all_vars if v not in varying]

    if not varying_vars:
        warnings.warn("No time-varying variables found. Returning data grouped by entity.")
        return data.groupby(level=entity_name).first()

    wide_data = data.reset_index()

    if constant_vars:
        constant_df = wide_data.groupby(entity_name)[constant_vars].first()
    else:
        unique_entities = wide_data[entity_name].unique()
        constant_df = pd.DataFrame(index=pd.Index(unique_entities, name=entity_name))

    wide_df = wide_data.pivot(index=entity_name, columns=time_name, values=varying_vars)

    if isinstance(wide_df.columns, pd.MultiIndex):
        wide_df.columns = [f"{var}{separator}{time}" for var, time in wide_df.columns]

    result = pd.concat([constant_df, wide_df], axis=1)

    return result


def long_panel(
    data: pd.DataFrame,
    entity_col: str,
    stub_names: list[str] | None = None,
    suffix: str = r"\d+",
    separator: str = "_",
    time_name: str = "time",
    time_values: list[int | str] | None = None,
) -> pd.DataFrame:
    r"""Convert wide panel data to long format.

    Transforms panel data from wide format (one row per entity) to long
    format (one row per entity-time observation). Handles both balanced
    and unbalanced panels.

    Parameters
    ----------
    data : pd.DataFrame
        Wide format data with time-varying variables as separate columns.
    entity_col : str
        Name of entity/unit identifier column.
    stub_names : list[str] | None, default None
        Base names of time-varying variables. If None, infers from column names.
    suffix : str, default r"\\d+"
        Regex pattern for time period suffix. Default matches digits.
    separator : str, default "_"
        Character(s) separating variable name from time period.
    time_name : str, default "time"
        Name for the created time period column.
    time_values : list[int | str] | None, default None
        Explicit time period values. If None, infers from column suffixes.

    Returns
    -------
    pd.DataFrame
        Long format panel data with MultiIndex (entity, time).

    See Also
    --------
    widen_panel : Convert long format to wide format.
    prepare_data_for_did : Prepare data for DiD analysis.
    """
    data = data.copy()

    if entity_col not in data.columns:
        if data.index.name == entity_col:
            data = data.reset_index()
        else:
            raise ValueError(f"Entity column '{entity_col}' not found in data.")

    if stub_names is None:
        pattern = f"(.+){separator}({suffix})$"
        stub_names = []
        for col in data.columns:
            if col == entity_col:
                continue
            match = re.match(pattern, col)
            if match:
                stub = match.group(1)
                if stub not in stub_names:
                    stub_names.append(stub)

    if not stub_names:
        raise ValueError("No time-varying variables detected. Check column naming pattern.")

    id_vars = [entity_col]
    value_vars = []
    var_name_map = {}

    if time_values is None:
        time_values = set()
        pattern = f"{separator}({suffix})$"
        for col in data.columns:
            match = re.search(pattern, col)
            if match:
                time_values.add(match.group(1))
        time_values = sorted(time_values, key=lambda x: int(x) if x.isdigit() else x)

    for stub in stub_names:
        stub_cols = []
        for t in time_values:
            col_name = f"{stub}{separator}{t}"
            if col_name in data.columns:
                stub_cols.append(col_name)
                var_name_map[col_name] = (stub, t)
        if stub_cols:
            value_vars.extend(stub_cols)

    constant_vars = [col for col in data.columns if col not in value_vars and col != entity_col]
    id_vars.extend(constant_vars)

    long_df = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name="variable", value_name="value")

    long_df[["var_name", time_name]] = pd.DataFrame(long_df["variable"].map(var_name_map).tolist(), index=long_df.index)

    long_df = long_df.drop(columns=["variable"])

    pivot_df = long_df.pivot_table(
        index=[entity_col, time_name] + constant_vars,
        columns="var_name",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivot_df = pivot_df.set_index([entity_col, time_name]).sort_index()

    return pivot_df


def unpanel(data: pd.DataFrame) -> pd.DataFrame:
    """Convert panel data with MultiIndex to regular DataFrame.

    This convenience function removes the MultiIndex structure from panel
    data, converting it to a regular DataFrame with the index levels as
    columns.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with former index levels as regular columns.

    See Also
    --------
    prepare_data_for_did : Create panel data with MultiIndex.
    """
    if isinstance(data.index, pd.MultiIndex):
        return data.reset_index()
    return data.copy()


def is_panel_balanced(data: pd.DataFrame) -> bool:
    """Check if panel data is balanced.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time).

    Returns
    -------
    bool
        True if panel is balanced, False otherwise.

    See Also
    --------
    make_panel_balanced : Create a balanced panel from unbalanced data.
    panel_has_gaps : Check for gaps in time series within entities.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    time_name = data.index.names[1]
    _, counts = np.unique(data.index.get_level_values(time_name), return_counts=True)

    return len(set(counts)) == 1


def panel_has_gaps(data: pd.DataFrame) -> dict[int, list[int]]:
    """Check for gaps in time series within entities.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time).

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping entity IDs to lists of missing time periods.
        Empty dict if no gaps found.

    See Also
    --------
    fill_panel_gaps : Fill gaps in panel data.
    is_panel_balanced : Check if panel is balanced.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, time_name = data.index.names
    gaps = {}

    for entity in data.index.get_level_values(entity_name).unique():
        entity_data = data.loc[entity]
        times = entity_data.index.get_level_values(time_name).values

        if len(times) > 1:
            full_range = np.arange(times.min(), times.max() + 1)
            missing = set(full_range) - set(times)
            if missing:
                gaps[entity] = sorted(list(missing))

    return gaps


def is_repeated_cross_section(data: pd.DataFrame) -> bool:
    """Check if data is repeated cross-section format.

    Parameters
    ----------
    data : pd.DataFrame
        Data with MultiIndex (entity, time).

    Returns
    -------
    bool
        True if repeated cross-section, False if panel.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name = data.index.names[0]
    return data.index.get_level_values(entity_name).nunique() == len(data)


def datetime_to_int(
    dates: pd.Series | pd.DatetimeIndex, freq: str = "YS", start_value: int = 1000
) -> dict[pd.Timestamp, int]:
    """Convert datetime values to integers preserving time spacing.

    For yearly frequencies, uses the year directly. For other frequencies,
    assigns sequential integers starting from start_value.

    Parameters
    ----------
    dates : pd.Series | pd.DatetimeIndex
        Datetime values to convert.
    freq : str, default "YS"
        Pandas frequency string (e.g., "YS", "QS", "MS", "D").
    start_value : int, default 1000
        Starting integer for non-yearly frequencies.

    Returns
    -------
    dict[pd.Timestamp, int]
        Mapping from datetime to integer values.

    See Also
    --------
    convert_panel_time_to_int : Convert time index in panel data.
    """
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be pd.Series or pd.DatetimeIndex with datetime values.")

    start, end = dates.min(), dates.max()
    full_range = pd.date_range(start=start, end=end, freq=freq)

    if freq in ["A", "AS", "Y", "YS", "A-DEC", "AS-JAN"]:
        return {date: date.year for date in full_range}

    return {date: idx for idx, date in enumerate(full_range, start=start_value)}


def convert_panel_time_to_int(
    data: pd.DataFrame, freq: str = "YS", start_value: int = 1000
) -> tuple[pd.DataFrame, dict[int, pd.Timestamp]]:
    """Convert datetime time index to integers in panel data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time) where time is datetime.
    freq : str, default "YS"
        Pandas frequency string for the time series.
    start_value : int, default 1000
        Starting integer for non-yearly frequencies.

    Returns
    -------
    pd.DataFrame
        Data with integer time index.
    dict[int, pd.Timestamp]
        Reverse mapping from integers back to datetime.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, time_name = data.index.names
    time_values = data.index.get_level_values(time_name)

    if not pd.api.types.is_datetime64_any_dtype(time_values):
        raise TypeError(f"Time index '{time_name}' must be datetime type.")

    date_map = datetime_to_int(time_values, freq=freq, start_value=start_value)
    reverse_map = {v: k for k, v in date_map.items()}

    new_data = data.copy()
    new_times = time_values.map(date_map)
    new_index = pd.MultiIndex.from_arrays(
        [data.index.get_level_values(entity_name), new_times], names=[entity_name, time_name]
    )
    new_data.index = new_index

    return new_data, reverse_map


def panel_to_cross_section_diff(
    data: pd.DataFrame,
    y_col: str,
    x_base_cols: list[str] | None = None,
    x_delta_cols: list[str] | None = None,
    pre_period: int | None = None,
    post_period: int | None = None,
) -> pd.DataFrame:
    """Transform panel data to cross-section with differences.

    Creates a cross-sectional dataset with outcome differences and
    optionally co-variate differences between two time periods.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data with 2 time periods.
    y_col : str
        Name of outcome variable column.
    x_base_cols : list[str] | None, default None
        Covariates to keep at baseline values.
    x_delta_cols : list[str] | None, default None
        Covariates to difference between periods.
    pre_period : int | None, default None
        Pre-treatment period (defaults to min time).
    post_period : int | None, default None
        Post-treatment period (defaults to max time).

    Returns
    -------
    pd.DataFrame
        Cross-sectional data with differences.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, time_name = data.index.names
    time_values = data.index.get_level_values(time_name).unique()

    if len(time_values) != 2:
        raise ValueError(f"Panel must have exactly 2 time periods, found {len(time_values)}.")

    if pre_period is None:
        pre_period = time_values.min()
    if post_period is None:
        post_period = time_values.max()

    if pre_period not in time_values or post_period not in time_values:
        raise ValueError("pre_period and post_period must be in the data.")

    pre_data = data.xs(pre_period, level=time_name)
    post_data = data.xs(post_period, level=time_name)

    common_entities = pre_data.index.intersection(post_data.index)
    pre_data = pre_data.loc[common_entities]
    post_data = post_data.loc[common_entities]

    result = pd.DataFrame(index=common_entities)
    result.index.name = entity_name

    result[y_col] = post_data[y_col] - pre_data[y_col]

    if x_base_cols:
        for col in x_base_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")
            result[col] = pre_data[col]

    if x_delta_cols:
        for col in x_delta_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")
            delta_name = f"delta_{col}" if col in (x_base_cols or []) else col
            result[delta_name] = post_data[col] - pre_data[col]

    return result


def are_varying(
    data: pd.DataFrame,
    covariates: list[str] | None = None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    variation_type: Literal["time", "individual", "both"] = "time",
    return_names: bool = True,
) -> list[str] | dict[str, bool] | pd.DataFrame:
    """Identify which covariates vary within entities over time or across individuals.

    This function checks whether variables vary within entities over time
    (time variation) or vary in their within-entity patterns across entities
    (individual variation).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time).
    covariates : list[str] | None, default None
        List of covariate column names to check. If None, checks all columns.
    rtol : float, default 1e-05
        Relative tolerance for float comparison.
    atol : float, default 1e-08
        Absolute tolerance for float comparison.
    variation_type : {"time", "individual", "both"}, default "time"
        Type of variation to check:

        - "time": Check if variables change within entities over time
        - "individual": Check if within-entity patterns differ across entities
        - "both": Check both types of variation

    return_names : bool, default True
        If True and type="time", returns list of varying variable names (backward compatible).
        If False, returns dict mapping variable names to bool.

    Returns
    -------
    list[str] | dict[str, bool] | pd.DataFrame
        Depends on parameters:

        - If return_names=True and variation_type="time": List of time-varying variable names
        - If return_names=False and variation_type="time" or "individual": Dict mapping names to bool
        - If variation_type="both": DataFrame with rows for each type and columns for variables
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, _ = data.index.names

    if covariates is None:
        var_list = data.columns.tolist()
    else:
        var_list = covariates
        for col in var_list:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

    results = {}

    if variation_type in ["time", "both"]:
        time_results = {}
        for col in var_list:
            grouped = data.groupby(entity_name)[col]

            if pd.api.types.is_numeric_dtype(data[col]):
                varies = grouped.apply(lambda x: not np.allclose(x.values, x.values[0], rtol=rtol, atol=atol)).any()
            else:
                varies = (grouped.nunique() > 1).any()

            time_results[col] = varies

        if variation_type == "time":
            if return_names:
                return [col for col, varies in time_results.items() if varies]
            return time_results
        results["time"] = time_results

    if variation_type in ["individual", "both"]:
        individual_results = {}
        for col in var_list:
            if pd.api.types.is_numeric_dtype(data[col]):
                grouped = data.groupby(entity_name)[col]
                variances = grouped.var()
                varies = variances.nunique() > 1
            else:
                grouped = data.groupby(entity_name)[col]
                varies = (grouped.nunique() > 1).any()
            individual_results[col] = varies

        if variation_type == "individual":
            return individual_results
        results["individual"] = individual_results

    return pd.DataFrame(results).T


def fill_panel_gaps(
    data: pd.DataFrame,
    fill_value: Any = np.nan,
    method: Literal["value", "ffill", "bfill"] = "value",
) -> pd.DataFrame:
    """Fill gaps in panel data time series.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with potential gaps.
    fill_value : Any, default np.nan
        Value to use for filling when method="value".
    method : {"value", "ffill", "bfill"}, default "value"
        Filling method: 'value' uses fill_value, 'ffill' forward fills,
        'bfill' backward fills.

    Returns
    -------
    pd.DataFrame
        Panel data with gaps filled.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, time_name = data.index.names

    new_idx_list = []
    for entity in data.index.get_level_values(entity_name).unique():
        entity_data = data.loc[entity]
        times = entity_data.index.values

        if len(times) > 1:
            time_range = np.arange(times.min(), times.max() + 1)
            for t in time_range:
                new_idx_list.append((entity, t))
        else:
            new_idx_list.append((entity, times[0]))

    new_index = pd.MultiIndex.from_tuples(new_idx_list, names=[entity_name, time_name])

    if method == "value":
        result = data.reindex(new_index, fill_value=fill_value)
    elif method == "ffill":
        result = data.reindex(new_index).groupby(entity_name).ffill()
    elif method == "bfill":
        result = data.reindex(new_index).groupby(entity_name).bfill()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'value', 'ffill', or 'bfill'.")

    return result.sort_index()


def make_panel_balanced(
    data: pd.DataFrame,
    min_periods: int | None = None,
    method: Literal["drop", "fill"] = "drop",
    fill_value: Any = np.nan,
) -> pd.DataFrame:
    """Create balanced panel from unbalanced data.

    Parameters
    ----------
    data : pd.DataFrame
        Unbalanced panel data.
    min_periods : int | None, default None
        Minimum number of periods an entity must have.
        If None, uses the maximum observed periods.
    method : {"drop", "fill"}, default "drop"
        How to handle imbalance: 'drop' removes entities,
        'fill' adds missing observations.
    fill_value : Any, default np.nan
        Value for filling if method="fill".

    Returns
    -------
    pd.DataFrame
        Balanced panel data.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    entity_name, time_name = data.index.names

    entity_counts = data.groupby(entity_name).size()

    if min_periods is None:
        min_periods = entity_counts.max()

    if method == "drop":
        keep_entities = entity_counts[entity_counts >= min_periods].index
        result = data.loc[data.index.get_level_values(entity_name).isin(keep_entities)]

        if len(keep_entities) == 0:
            warnings.warn(f"No entities have {min_periods} periods. Returning empty DataFrame.")

    elif method == "fill":
        keep_entities = entity_counts[entity_counts >= min_periods].index
        filtered = data.loc[data.index.get_level_values(entity_name).isin(keep_entities)]

        all_times = filtered.index.get_level_values(time_name).unique()
        selected_times = sorted(all_times)[:min_periods]

        new_idx = pd.MultiIndex.from_product([keep_entities, selected_times], names=[entity_name, time_name])
        result = filtered.reindex(new_idx, fill_value=fill_value)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'drop' or 'fill'.")

    return result.sort_index()


def create_relative_time_indicators(data: pd.DataFrame, cohort_col: str, base_period: int = -1) -> pd.DataFrame:
    """Create relative time indicators for DiD analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with treatment cohort information.
    cohort_col : str
        Column name containing treatment timing.
    base_period : int, default -1
        Relative time period to use as reference (excluded).

    Returns
    -------
    pd.DataFrame
        DataFrame with relative time indicator columns.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    if cohort_col not in data.columns:
        raise ValueError(f"Column '{cohort_col}' not found in data.")

    _, time_name = data.index.names

    time_values = data.index.get_level_values(time_name).values
    relative_time = time_values - data[cohort_col].values

    never_treated_mask = data[cohort_col].isna()
    relative_time = pd.Series(relative_time, index=data.index)
    relative_time[never_treated_mask] = np.nan

    unique_rel_times = sorted(relative_time.dropna().unique())
    indicators = pd.DataFrame(index=data.index)

    for rel_time in unique_rel_times:
        if int(rel_time) != base_period:
            col_name = f"rel_time_{int(rel_time):+d}"
            indicators[col_name] = (relative_time == rel_time).astype(int)

    return indicators


def validate_treatment_timing(data: pd.DataFrame, treat_col: str, cohort_col: str | None = None) -> dict[str, Any]:
    """Validate treatment timing assumptions for DiD.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with treatment information.
    treat_col : str
        Binary treatment status column.
    cohort_col : str | None, default None
        Treatment timing/cohort column.

    Returns
    -------
    dict[str, Any]
        Validation results with keys:

        - 'has_reversals': bool indicating treatment reversals
        - 'always_treated': list of always-treated entities
        - 'never_treated': list of never-treated entities
        - 'timing_consistent': bool for treat/cohort consistency
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Data must have a 2-level MultiIndex (entity, time).")

    if treat_col not in data.columns:
        raise ValueError(f"Column '{treat_col}' not found in data.")

    entity_name, _ = data.index.names
    results = {}

    reversals = []
    for entity in data.index.get_level_values(entity_name).unique():
        entity_data = data.loc[entity, treat_col].sort_index()
        if len(entity_data) > 1:
            if (entity_data.diff().dropna() < 0).any():
                reversals.append(entity)

    results["has_reversals"] = len(reversals) > 0
    results["entities_with_reversals"] = reversals

    treat_by_entity = data.groupby(entity_name)[treat_col]
    results["always_treated"] = treat_by_entity.all()[treat_by_entity.all()].index.tolist()
    results["never_treated"] = (~treat_by_entity.any())[~treat_by_entity.any()].index.tolist()

    if cohort_col is not None:
        if cohort_col not in data.columns:
            raise ValueError(f"Column '{cohort_col}' not found in data.")

        timing_consistent = True
        inconsistent_entities = []

        for entity in data.index.get_level_values(entity_name).unique():
            entity_data = data.loc[entity].sort_index()

            if entity_data[cohort_col].isna().all():
                continue

            cohort_time = entity_data[cohort_col].iloc[0]

            treated_times = entity_data[entity_data[treat_col] == 1].index
            if len(treated_times) > 0:
                first_treat_time = treated_times.min()
                if first_treat_time != cohort_time:
                    timing_consistent = False
                    inconsistent_entities.append(entity)

        results["timing_consistent"] = timing_consistent
        results["inconsistent_entities"] = inconsistent_entities

    return results


def prepare_data_for_did(
    data: pd.DataFrame,
    y_col: str,
    entity_col: str,
    time_col: str,
    treat_col: str | None = None,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare data for difference-in-differences analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data in any format.
    y_col : str
        Outcome variable column name.
    entity_col : str
        Entity/unit identifier column.
    time_col : str
        Time period column.
    treat_col : str | None, default None
        Treatment status column.
    covariates : list[str] | None, default None
        List of covariate columns to include.

    Returns
    -------
    pd.DataFrame
        Data with MultiIndex (entity, time) ready for DiD analysis.
    """
    required_cols = [y_col, entity_col, time_col]
    if treat_col:
        required_cols.append(treat_col)
    if covariates:
        required_cols.extend(covariates)

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    keep_cols = [entity_col, time_col, y_col]
    if treat_col:
        keep_cols.append(treat_col)
    if covariates:
        keep_cols.extend(covariates)

    result = data[keep_cols].copy()

    if result[[entity_col, time_col, y_col]].isna().any().any():
        warnings.warn("Missing values found in entity, time, or outcome columns.")

    result = result.set_index([entity_col, time_col]).sort_index()

    if treat_col and treat_col in result.columns:
        unique_vals = result[treat_col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            warnings.warn(f"Treatment column '{treat_col}' contains non-binary values: {unique_vals}")

    return result


def parse_formula(formula: str) -> dict[str, Any]:
    """Parse formula string to extract components."""
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError("Formula must be in the form 'y ~ x1 + x2 + ...'")

    outcome = parts[0].strip()
    predictors_str = parts[1].strip()

    var_pattern = r"\b[a-zA-Z_]\w*\b"
    all_vars = re.findall(var_pattern, predictors_str)

    exclude = {"C", "I", "Q", "bs", "ns", "log", "exp", "sqrt", "abs", "np"}
    predictors = [v for v in all_vars if v not in exclude]

    seen = set()
    predictors = [x for x in predictors if not (x in seen or seen.add(x))]

    return {
        "outcome": outcome,
        "predictors": predictors,
        "formula": formula,
    }


def extract_vars_from_formula(formula: str) -> list[str]:
    """Extract all variable names from formula string."""
    parsed = parse_formula(formula)
    vars_list = []
    if parsed["outcome"]:
        vars_list.append(parsed["outcome"])
    vars_list.extend(parsed["predictors"])
    return vars_list
