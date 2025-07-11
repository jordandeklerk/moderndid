"""Extract aggregated coefficients from Sun and Abraham regression models."""

import re
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


class SunAbrahamCoefficients(NamedTuple):
    """Result from extracting Sun & Abraham aggregated coefficients.

    Attributes
    ----------
    beta : ndarray
        Aggregated event-study coefficients.
    sigma : ndarray
        Variance-covariance matrix of the aggregated coefficients.
    event_times : ndarray
        Event times (relative periods) corresponding to beta and sigma.
    """

    beta: np.ndarray
    sigma: np.ndarray
    event_times: np.ndarray


@runtime_checkable
class RegressionModelProtocol(Protocol):
    """Protocol for regression model result objects."""

    coefficients: np.ndarray | dict
    cov_scaled: np.ndarray | None


def extract_sunab_coefficients(model_result, pattern=None):
    r"""Extract aggregated event-study coefficients from regression model with Sun & Abraham interactions.

    This function takes a regression model result that includes Sun & Abraham
    interaction terms and extracts the aggregated event-study coefficients and
    their variance-covariance matrix.

    The aggregation uses cohort shares as weights, computed from the model matrix
    to ensure proper weighting of cohort-specific effects:

    .. math::

        \hat{\tau}_e = \sum_g w_{g,e} \hat{\tau}_{g,e},

    where :math:`w_{g,e}` are the cohort-specific weights for event time :math:`e`
    and cohort :math:`g`.

    Parameters
    ----------
    model_result : RegressionModelProtocol
        Regression model result object. Must have:

        - params : Series or array-like with coefficient values
        - cov_params() : method returning variance-covariance matrix
        - model.exog : design matrix (for computing weights)
        - model.weights : observation weights (optional)

    pattern : str, optional
        Regular expression pattern to identify Sun & Abraham interaction terms.
        If None, attempts to detect pattern automatically from coefficient names.
        Pattern should capture event time in group 2 (e.g., r'sunab::(\w+)::([-\d]+)').

    Returns
    -------
    SunAbrahamCoefficients
        NamedTuple containing:

        - beta : Aggregated event-study coefficients
        - sigma : Variance-covariance matrix of aggregated coefficients
        - event_times : Event times corresponding to beta and sigma

    Notes
    -----
    The function assumes that interaction terms follow a specific naming pattern
    that includes cohort and event time information. The aggregation weights
    are computed from the model matrix by summing the (weighted) signs of
    observations contributing to each interaction term.

    The transformation matrix :math:`T` is constructed such that:

    .. math::

        \hat{\tau}_{aggregated} = T \cdot \hat{\tau}_{interactions}

        \Sigma_{aggregated} = T \cdot \Sigma_{interactions} \cdot T'

    References
    ----------

    .. [1] Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects
           in event studies with heterogeneous treatment effects. Journal of
           Econometrics, 225(2), 175-199.
    """
    coefficients, cov_scaled, coef_names = _extract_model_attributes(model_result)

    if pattern is None:
        pattern = _detect_sunab_pattern(coef_names)

    sunab_indices, sunab_names = _extract_sunab_indices(coef_names, pattern)

    if not sunab_indices:
        raise ValueError(f"No coefficients matching pattern '{pattern}' found")

    event_times_raw = _extract_event_times(sunab_names, pattern)
    unique_event_times = sorted(set(event_times_raw))

    aggregation_weights = _compute_aggregation_weights(model_result, sunab_indices)

    transformation_matrix = _create_transformation_matrix(event_times_raw, unique_event_times, aggregation_weights)

    sunab_coefs = coefficients[sunab_indices]
    sunab_vcov = cov_scaled[np.ix_(sunab_indices, sunab_indices)]

    aggregated_coefs = transformation_matrix @ sunab_coefs
    aggregated_vcov = transformation_matrix @ sunab_vcov @ transformation_matrix.T

    aggregated_vcov = (aggregated_vcov + aggregated_vcov.T) / 2

    return SunAbrahamCoefficients(
        beta=aggregated_coefs, sigma=aggregated_vcov, event_times=np.array(unique_event_times)
    )


def _extract_model_attributes(model_result):
    """Extract coefficients, covariance, and names from statsmodels result."""
    if hasattr(model_result, "params") and hasattr(model_result, "cov_params"):
        coefficients = np.asarray(model_result.params)
        cov_scaled = np.asarray(model_result.cov_params())

        if hasattr(model_result.params, "index"):
            coef_names = list(model_result.params.index)
        else:
            coef_names = [f"coef_{i}" for i in range(len(coefficients))]
    else:
        raise AttributeError(
            "Model result must be a statsmodels RegressionResults object with params and cov_params() attributes"
        )

    if coefficients.shape[0] != cov_scaled.shape[0]:
        raise ValueError("Coefficient vector and covariance matrix dimensions do not match")

    return coefficients, cov_scaled, coef_names


def _detect_sunab_pattern(coef_names):
    """Detect Sun & Abraham interaction pattern from coefficient names."""
    common_patterns = [
        r"sunab::(\w+)::([-\d]+)",
        r"cohort_(\w+)_period_([-\d]+)",
        r"cohort(\w+):period([-\d]+)",
        r"g(\d+)_t([-\d]+)",
        r"treat_(\w+)_rel_([-\d]+)",
        r"C\(cohort\)\[T\.(\w+)\]:C\(period\)\[T\.([-\d]+)\]",
    ]

    for pattern in common_patterns:
        if any(re.search(pattern, str(name)) for name in coef_names):
            return pattern

    raise ValueError(
        "Could not detect Sun & Abraham interaction pattern. "
        "Please provide explicit pattern parameter. "
        "Expected patterns like 'sunab::cohort::period' or 'cohort_X_period_Y'"
    )


def _extract_sunab_indices(coef_names, pattern):
    """Extract indices and names of Sun & Abraham interaction terms."""
    sunab_indices = []
    sunab_names = []

    for i, name in enumerate(coef_names):
        if re.search(pattern, str(name)):
            sunab_indices.append(i)
            sunab_names.append(str(name))

    return sunab_indices, sunab_names


def _extract_event_times(sunab_names, pattern):
    """Extract event times from Sun & Abraham coefficient names."""
    event_times = []
    for name in sunab_names:
        match = re.search(pattern, name)
        if match:
            event_time = int(match.group(2))
            event_times.append(event_time)
        else:
            raise ValueError(f"Could not extract event time from coefficient name: {name}")

    return event_times


def _compute_aggregation_weights(model_result, sunab_indices):
    """Compute aggregation weights from model matrix using statsmodels."""
    if hasattr(model_result, "model"):
        model = model_result.model

        if hasattr(model, "exog"):
            model_matrix = np.asarray(model.exog)
        else:
            raise AttributeError("Model does not have design matrix (exog)")

        if hasattr(model, "weights") and model.weights is not None:
            weights = np.asarray(model.weights)
        else:
            weights = np.ones(model_matrix.shape[0])
    else:
        raise AttributeError("Model result does not have associated model object. Cannot compute aggregation weights.")

    try:
        sunab_matrix = model_matrix[:, sunab_indices]
    except IndexError as e:
        raise IndexError(
            f"Could not extract Sun & Abraham columns from design matrix. "
            f"Matrix shape: {model_matrix.shape}, indices: {sunab_indices}"
        ) from e

    aggregation_weights = np.sum(weights[:, np.newaxis] * np.sign(sunab_matrix), axis=0)

    aggregation_weights = np.where(aggregation_weights == 0, 1.0, aggregation_weights)

    return aggregation_weights


def _create_transformation_matrix(event_times_raw, unique_event_times, weights):
    """Create transformation matrix for aggregating coefficients."""
    n_event_times = len(unique_event_times)
    n_interactions = len(event_times_raw)

    event_indicator = np.zeros((n_interactions, n_event_times))
    for i, event_time in enumerate(event_times_raw):
        j = unique_event_times.index(event_time)
        event_indicator[i, j] = 1.0

    W = np.diag(weights)
    X = event_indicator

    XtW = X.T @ W
    XtWX = XtW @ X

    if np.linalg.matrix_rank(XtWX) < XtWX.shape[0]:
        XtWX_inv = np.linalg.pinv(XtWX)
    else:
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)

    transformation_matrix = XtWX_inv @ XtW

    return transformation_matrix
