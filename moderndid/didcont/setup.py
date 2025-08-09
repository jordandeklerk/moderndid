"""Functions for panel treatment effects."""

from typing import NamedTuple

import formulaic as fml
import numpy as np
import pandas as pd


class PTEParams(NamedTuple):
    """Parameters for panel treatment effects.

    Attributes
    ----------
    y : ndarray
        Outcome variable values.
    d : ndarray
        Continuous treatment variable values.
    x : ndarray
        Covariate matrix including intercept if desired.
    time_ids : ndarray
        Time period identifiers.
    unit_ids : ndarray
        Unit (individual) identifiers.
    weights : ndarray
        Sample weights for each observation.
    n_units : int
        Number of unique units in the panel.
    n_periods : int
        Number of time periods.
    n_obs : int
        Total number of observations.
    base_period : int
        Base period used for treatment effect calculation.
    anticipation : int
        Number of periods of treatment anticipation allowed.
    control_group : str
        Type of control group used ("never_treated" or "not_yet_treated").
    balanced : bool
        Whether the panel is balanced.
    orig_time_periods : ndarray
        Original time period values before recoding.
    time_map : dict
        Mapping from original to recoded time periods.
    """

    y: np.ndarray
    d: np.ndarray
    x: np.ndarray
    time_ids: np.ndarray
    unit_ids: np.ndarray
    weights: np.ndarray
    n_units: int
    n_periods: int
    n_obs: int
    base_period: int
    anticipation: int
    control_group: str
    balanced: bool
    orig_time_periods: np.ndarray
    time_map: dict


def setup_pte(
    data,
    yname,
    tname,
    dname,
    idname,
    xformla="~1",
    weightsname=None,
    base_period="varying",
    anticipation=0,
    control_group="never_treated",
    allow_unbalanced_panel=True,
):
    """Set up data for panel treatment effects with validation.

    Parameters
    ----------
    data : DataFrame
        Panel data with observations indexed by unit and time.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    dname : str
        Name of continuous treatment variable column.
    idname : str
        Name of unit identifier column.
    xformla : str, default="~1"
        Formula string for covariates. Default includes only intercept.
    weightsname : str, optional
        Name of sample weights column. If None, uniform weights used.
    base_period : {"varying", "universal"} or int, default="varying"
        Base period specification:

        - "varying": Use the last pre-treatment period for each unit
        - "universal": Use a common base period for all units
        - int: Specific time period to use as base
    anticipation : int, default=0
        Number of periods of treatment anticipation allowed.
    control_group : {"never_treated", "not_yet_treated"}, default="never_treated"
        Definition of control group for comparison.
    allow_unbalanced_panel : bool, default=True
        Whether to allow unbalanced panels. If False, raises error for unbalanced data.

    Returns
    -------
    PTEParams
        NamedTuple containing prepared data and parameters.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If required columns are missing, data has insufficient observations,
        panel is unbalanced when balanced required, or other data issues.
    KeyError
        If specified column names are not found in data.

    See Also
    --------
    setup_pte_basic : Lightweight version with minimal error checking.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("data cannot be empty")

    required_cols = [yname, tname, dname, idname]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Required columns missing from data: {missing_cols}")

    key_vars = [yname, tname, dname, idname]
    for var in key_vars:
        if data[var].isna().any():
            n_missing = data[var].isna().sum()
            raise ValueError(f"Missing values found in {var}: {n_missing} observations")

    if weightsname is not None:
        if weightsname not in data.columns:
            raise KeyError(f"Weights column '{weightsname}' not found in data")
        if data[weightsname].isna().any():
            raise ValueError(f"Missing values found in weights column '{weightsname}'")
        if (data[weightsname] < 0).any():
            raise ValueError("All weights must be non-negative")

    n_obs = len(data)
    n_units = data[idname].nunique()
    n_periods = data[tname].nunique()

    if n_obs < 10:
        raise ValueError(f"Insufficient observations: {n_obs}. Need at least 10.")

    if n_units < 2:
        raise ValueError(f"Insufficient units: {n_units}. Need at least 2.")

    if n_periods < 2:
        raise ValueError(f"Insufficient time periods: {n_periods}. Need at least 2.")

    if not isinstance(anticipation, int) or anticipation < 0:
        raise ValueError("anticipation must be a non-negative integer")

    if control_group not in ["never_treated", "not_yet_treated"]:
        raise ValueError("control_group must be 'never_treated' or 'not_yet_treated'")

    unit_time_counts = data.groupby(idname).size()
    balanced = len(unit_time_counts.unique()) == 1

    if not balanced and not allow_unbalanced_panel:
        raise ValueError("Panel data is unbalanced but allow_unbalanced_panel=False")

    if not np.issubdtype(data[dname].dtype, np.number):
        raise ValueError(f"Treatment variable '{dname}' must be numeric")

    if not np.issubdtype(data[yname].dtype, np.number):
        raise ValueError(f"Outcome variable '{yname}' must be numeric")

    if np.isinf(data[yname]).any():
        raise ValueError(f"Infinite values found in outcome variable '{yname}'")

    if np.isinf(data[dname]).any():
        raise ValueError(f"Infinite values found in treatment variable '{dname}'")

    try:
        params = setup_pte_basic(
            data=data,
            yname=yname,
            tname=tname,
            dname=dname,
            idname=idname,
            xformla=xformla,
            weightsname=weightsname,
            base_period=base_period,
            anticipation=anticipation,
            control_group=control_group,
        )
    except Exception as e:
        raise ValueError(f"Error in data setup: {str(e)}") from e

    return params


def setup_pte_basic(
    data,
    yname,
    tname,
    dname,
    idname,
    xformla="~1",
    weightsname=None,
    base_period="varying",
    anticipation=0,
    control_group="never_treated",
):
    """Set up data for panel treatment effects with minimal validation.

    Parameters
    ----------
    data : DataFrame
        Panel data with observations indexed by unit and time.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    dname : str
        Name of continuous treatment variable column.
    idname : str
        Name of unit identifier column.
    xformla : str, default="~1"
        Formula string for covariates. Default includes only intercept.
    weightsname : str, optional
        Name of sample weights column. If None, uniform weights used.
    base_period : {"varying", "universal"} or int, default="varying"
        Base period specification:

        - "varying": Use the last pre-treatment period for each unit
        - "universal": Use a common base period for all units
        - int: Specific time period to use as base
    anticipation : int, default=0
        Number of periods of treatment anticipation allowed.
    control_group : {"never_treated", "not_yet_treated"}, default="never_treated"
        Definition of control group for comparison.

    Returns
    -------
    PTEParams
        NamedTuple containing prepared data and parameters.

    See Also
    --------
    setup_pte : More robust version with comprehensive error checking.
    """
    y = data[yname].values
    d = data[dname].values
    time_ids = data[tname].values
    unit_ids = data[idname].values

    if weightsname is not None:
        weights = data[weightsname].values
    else:
        weights = np.ones(len(data))

    if xformla == "~1":
        x = np.ones((len(data), 1))
    else:
        try:
            model_matrix_result = fml.model_matrix(xformla, data, ensure_full_rank=True)
            x = np.asarray(model_matrix_result)
        except Exception as e:
            raise ValueError(f"Error processing xformla '{xformla}' with formulaic: {e}") from e

    unique_times = np.sort(np.unique(time_ids))
    time_map = {orig: i + 1 for i, orig in enumerate(unique_times)}
    sequential_time_ids = np.array([time_map[t] for t in time_ids])

    n_units = len(np.unique(unit_ids))
    n_periods = len(unique_times)
    n_obs = len(data)

    if base_period == "varying":
        base_period_val = 1
    elif base_period == "universal":
        base_period_val = n_periods - 1
    elif isinstance(base_period, int | float):
        base_period_val = _original_to_new_time(base_period, time_map)
        if np.isnan(base_period_val):
            raise ValueError(f"base_period {base_period} not found in data")
        base_period_val = int(base_period_val)
    else:
        raise ValueError("base_period must be 'varying', 'universal', or int")

    unit_time_counts = data.groupby(idname).size()
    balanced = len(unit_time_counts.unique()) == 1

    return PTEParams(
        y=y,
        d=d,
        x=x,
        time_ids=sequential_time_ids,
        unit_ids=unit_ids,
        weights=weights,
        n_units=n_units,
        n_periods=n_periods,
        n_obs=n_obs,
        base_period=base_period_val,
        anticipation=anticipation,
        control_group=control_group,
        balanced=balanced,
        orig_time_periods=unique_times,
        time_map=time_map,
    )


def _original_to_new_time(orig_time, time_map):
    """Convert original time values to sequential time values."""
    if isinstance(orig_time, list | tuple | np.ndarray):
        orig_time = np.asarray(orig_time)
        return np.array([_convert_single_time(t, time_map) for t in orig_time])
    return _convert_single_time(orig_time, time_map)


def _convert_single_time(single_time, time_map):
    """Convert a single original time value to sequential time value."""
    return time_map.get(single_time, np.nan)
