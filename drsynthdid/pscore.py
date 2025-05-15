"""Propensity scores for DRDiD methods."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


def _estimate_propensity_score_mle(
    treatment_indicator: np.ndarray,
    covariates: np.ndarray,
    observation_weights: np.ndarray,
    is_frequency_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Estimate propensity scores using Maximum Likelihood Estimation (Logit).

    Fit a logistic regression model to estimate the probability of treatment
    given a set of covariates. This function can handle both sampling
    (variance) weights and frequency weights.

    Parameters
    ----------
    treatment_indicator : ndarray
        Binary array indicating treatment status (1 for treated, 0 for control).
    covariates : ndarray
        Matrix of exogenous variables (covariates).
    observation_weights : ndarray
        Array of observation weights.
    is_frequency_weights : bool, default False
        Specify if `observation_weights` are frequency weights (True) or
        variance/sampling weights (False).

    Returns
    -------
    tuple[ndarray, ndarray | None]
        Return a tuple containing the fitted propensity scores and the
        covariance matrix of the parameters. Return None for the covariance
        matrix if an error occurs during estimation.
    """
    weights_arg: dict[str, Any] = (
        {"freq_weights": observation_weights} if is_frequency_weights else {"var_weights": observation_weights}
    )

    try:
        logit_model = sm.GLM(treatment_indicator, covariates, family=sm.families.Binomial(), **weights_arg)
        logit_results = logit_model.fit()
        propensity_scores = logit_results.predict()
        propensity_scores = np.clip(propensity_scores, 1e-16, 1.0 - 1e-16)
        return propensity_scores, logit_results.cov_params()
    except Exception as e:
        warnings.warn(f"MLE for propensity score estimation failed: {e}", UserWarning)
        return np.full(treatment_indicator.shape, np.nan), None


def _calculate_ipt_loss_graham(
    gamma_params: np.ndarray,
    treatment_indicator: np.ndarray,
    covariates: np.ndarray,
    observation_weights: np.ndarray,
    n_observations: int,
) -> float:
    """Calculate loss for Inverse Probability Tilting (IPT).

    Loss function based on the method described by Graham, Pinto, and Egel
    (2012) [1]_. It is used in the optimization process for estimating propensity
    scores via IPT.

    Parameters
    ----------
    gamma_params : ndarray
        Parameters of the propensity score model.
    treatment_indicator : ndarray
        Binary array indicating treatment status.
    covariates : ndarray
        Matrix of covariates.
    observation_weights : ndarray
        Array of observation weights.
    n_observations : int
        Total number of observations, used for quadratic extrapolation.

    Returns
    -------
    float
        The calculated loss value.

    References
    ----------

    .. [1] Graham, B. S., Pinto, C. C., & Egel, D. (2012). *Inverse Probability
       Tilting for Moment Condition Models with Missing Data.* Review of
       Economic Studies, 79 (3), 1053-1079.
       https://doi.org/10.1093/restud/rdr047
    """
    if n_observations <= 1:
        return np.inf

    v_star = np.log(n_observations - 1)
    c_n = -(n_observations - 1)
    b_n = -n_observations + (n_observations - 1) * v_star
    a_n = -(n_observations - 1) * (1 - v_star + 0.5 * (v_star**2))

    linear_predictor = covariates @ gamma_params
    phi = np.zeros_like(linear_predictor)

    mask_v_lt_vstar = linear_predictor < v_star

    phi[mask_v_lt_vstar] = -linear_predictor[mask_v_lt_vstar] - np.exp(linear_predictor[mask_v_lt_vstar])
    phi[~mask_v_lt_vstar] = (
        a_n + b_n * linear_predictor[~mask_v_lt_vstar] + 0.5 * c_n * (linear_predictor[~mask_v_lt_vstar] ** 2)
    )

    loss_value = -np.sum(observation_weights * ((1 - treatment_indicator) * phi + linear_predictor))
    return loss_value


def _calculate_ipt_gradient_graham(
    gamma_params: np.ndarray,
    treatment_indicator: np.ndarray,
    covariates: np.ndarray,
    observation_weights: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    """Calculate gradient of the loss for Inverse Probability Tilting (IPT).

    This gradient function corresponds to the loss function described by
    Graham, Pinto, and Egel (2012) [1]_. It is used in the optimization
    process for estimating propensity scores via IPT.

    Parameters
    ----------
    gamma_params : ndarray
        Parameters of the propensity score model.
    treatment_indicator : ndarray
        Binary array indicating treatment status.
    covariates : ndarray
        Matrix of covariates.
    observation_weights : ndarray
        Array of observation weights.
    n_observations : int
        Total number of observations, used for quadratic extrapolation.

    Returns
    -------
    ndarray
        The gradient of the loss function.

    References
    ----------

    .. [1] Graham, B. S., Pinto, C. C., & Egel, D. (2012). *Inverse Probability
       Tilting for Moment Condition Models with Missing Data.* Review of
       Economic Studies, 79 (3), 1053-1079.
       https://doi.org/10.1093/restud/rdr047
    """
    if n_observations <= 1:
        return np.full_like(gamma_params, np.nan)

    v_star = np.log(n_observations - 1)
    c_n = -(n_observations - 1)
    b_n = -n_observations + (n_observations - 1) * v_star

    linear_predictor = covariates @ gamma_params
    phi_derivative = np.zeros_like(linear_predictor)
    mask_v_lt_vstar = linear_predictor < v_star

    phi_derivative[mask_v_lt_vstar] = -1 - np.exp(linear_predictor[mask_v_lt_vstar])
    phi_derivative[~mask_v_lt_vstar] = b_n + c_n * linear_predictor[~mask_v_lt_vstar]

    gradient = -covariates.T @ (observation_weights * ((1 - treatment_indicator) * phi_derivative + 1))
    return gradient


def _calculate_ipt_hessian_graham(
    gamma_params: np.ndarray,
    treatment_indicator: np.ndarray,
    covariates: np.ndarray,
    observation_weights: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    """Calculate Hessian of the loss for Inverse Probability Tilting (IPT).

    This Hessian function corresponds to the loss function described by
    Graham, Pinto, and Egel (2012) [1]_. It is used in the optimization
    process for estimating propensity scores via IPT, particularly with
    optimizers that can utilize Hessian information.

    Parameters
    ----------
    gamma_params : ndarray
        Parameters of the propensity score model.
    treatment_indicator : ndarray
        Binary array indicating treatment status.
    covariates : ndarray
        Matrix of covariates.
    observation_weights : ndarray
        Array of observation weights.
    n_observations : int
        Total number of observations, used for quadratic extrapolation.

    Returns
    -------
    ndarray
        The Hessian matrix of the loss function.

    References
    ----------

    .. [1] Graham, B. S., Pinto, C. C., & Egel, D. (2012). *Inverse Probability
       Tilting for Moment Condition Models with Missing Data.* Review of
       Economic Studies, 79 (3), 1053-1079.
       https://doi.org/10.1093/restud/rdr047
    """
    if n_observations <= 1:
        return np.full((len(gamma_params), len(gamma_params)), np.nan)

    v_star = np.log(n_observations - 1)
    c_n = -(n_observations - 1)

    linear_predictor = covariates @ gamma_params
    phi_second_derivative = np.zeros_like(linear_predictor)
    mask_v_lt_vstar = linear_predictor < v_star

    phi_second_derivative[mask_v_lt_vstar] = -np.exp(linear_predictor[mask_v_lt_vstar])
    phi_second_derivative[~mask_v_lt_vstar] = c_n

    hessian = -covariates.T @ (
        np.diag((1 - treatment_indicator) * observation_weights * phi_second_derivative) @ covariates
    )
    return hessian


def _objective_calibrated_propensity_score_tan(
    gamma_params: np.ndarray, treatment_indicator: np.ndarray, covariates: np.ndarray, observation_weights: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate objective, gradient, and Hessian for calibrated propensity score.

    This objective function, along with its gradient and Hessian, is used for
    estimating propensity scores using the calibration method described by
    Tan (2020) [1]_.

    Parameters
    ----------
    gamma_params : ndarray
        Parameters of the propensity score model.
    treatment_indicator : ndarray
        Binary array indicating treatment status.
    covariates : ndarray
        Matrix of covariates.
    observation_weights : ndarray
        Array of observation weights.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        A tuple containing the objective function value, the gradient, and the Hessian.

    References
    ----------

    .. [1] Tan, Z. (2020). *Regularized calibrated estimation of propensity scores
       with model misspecification and high-dimensional data.* Biometrika, 107 (1),
       137-158. https://doi.org/10.1093/biomet/asz059
    """
    if np.any(np.isnan(gamma_params)):
        k_covariates = covariates.shape[1]
        return np.inf, np.full(k_covariates, np.nan), np.full((k_covariates, k_covariates), np.nan)

    linear_predictor = covariates @ gamma_params
    exp_linear_predictor = np.exp(linear_predictor)

    objective_value = -np.mean(
        np.where(treatment_indicator == 1, linear_predictor, -exp_linear_predictor) * observation_weights
    )

    grad_terms = np.where(treatment_indicator == 1, 1, -exp_linear_predictor)
    gradient = -np.mean(grad_terms[:, np.newaxis] * observation_weights[:, np.newaxis] * covariates, axis=0)

    hess_diag_terms = np.where(treatment_indicator == 1, 0, exp_linear_predictor) * observation_weights
    hessian = (covariates.T @ np.diag(hess_diag_terms) @ covariates) / len(treatment_indicator)
    return objective_value, gradient, hessian


def calculate_propensity_score(
    treatment_indicator: np.ndarray, covariates: np.ndarray, observation_weights: np.ndarray, n_observations: int
) -> tuple[np.ndarray, int]:
    """Calculate propensity scores using different methods with fallbacks.

    Attempt estimation first with the calibrated method described by Tan (2020) [1]_
    using the ``trust-constr`` optimizer. If that fails, attempt the Inverse
    Probability Tilting (IPT) method described by Graham, Pinto, and Egel (2012) [2]_
    using the ``L-BFGS-B`` optimizer. If both fail, fall back to standard
    Maximum Likelihood Estimation (MLE) via a Logit model.

    Parameters
    ----------
    treatment_indicator : ndarray
        Treatment indicator array (0 or 1).
    covariates : ndarray
        Covariate matrix (n x k), including intercept.
    observation_weights : ndarray
        Observation weights.
    n_observations : int
        Total number of observations, used by the IPT method for extrapolation.

    Returns
    -------
    tuple[ndarray, int]
        A tuple containing the estimated propensity scores and an integer flag
        indicating the estimation outcome:
        0: Calibrated method (Tan) converged.
        1: IPT method (Graham et al.) converged.
        2: MLE (Logit) was used.
        3: All methods failed.

    References
    ----------

    .. [1] Tan, Z. (2020). *Regularized calibrated estimation of propensity scores
       with model misspecification and high-dimensional data.* Biometrika, 107 (1),
       137-158. https://doi.org/10.1093/biomet/asz059
    .. [2] Graham, B. S., Pinto, C. C., & Egel, D. (2012). *Inverse Probability
       Tilting for Moment Condition Models with Missing Data.* Review of
       Economic Studies, 79 (3), 1053-1079.
       https://doi.org/10.1093/restud/rdr047
    """
    estimation_flag = 3
    propensity_scores = np.full(len(treatment_indicator), np.nan)

    try:
        gamma_init_model = sm.GLM(
            treatment_indicator, covariates, family=sm.families.Binomial(), var_weights=observation_weights
        )
        gamma_init_results = gamma_init_model.fit()
        initial_gamma = np.asarray(gamma_init_results.params)
    except Exception as e:
        warnings.warn(
            f"Initial MLE for propensity score parameters failed: {e}. Using zeros as initial guess.", UserWarning
        )
        initial_gamma = np.zeros(covariates.shape[1])

    try:

        def obj_fun_cal(g):
            return _objective_calibrated_propensity_score_tan(g, treatment_indicator, covariates, observation_weights)[
                0
            ]

        def jac_fun_cal(g):
            return _objective_calibrated_propensity_score_tan(g, treatment_indicator, covariates, observation_weights)[
                1
            ]

        def hess_fun_cal(g):
            return _objective_calibrated_propensity_score_tan(g, treatment_indicator, covariates, observation_weights)[
                2
            ]

        cal_res = minimize(
            fun=obj_fun_cal,
            x0=initial_gamma,
            method="trust-constr",
            jac=jac_fun_cal,
            hess=hess_fun_cal,
            options={"maxiter": 1000, "gtol": 1e-6},
        )
        if cal_res.success:
            gamma_calibrated = cal_res.x
            linear_predictor_cal = covariates @ gamma_calibrated
            propensity_scores = 1 / (1 + np.exp(-linear_predictor_cal))
            estimation_flag = 0
    except Exception as e:
        warnings.warn(f"Calibrated propensity score estimation (Tan method) encountered an error: {e}", UserWarning)

    if estimation_flag != 0:
        try:
            ipt_res = minimize(
                fun=_calculate_ipt_loss_graham,
                x0=initial_gamma,
                args=(treatment_indicator, covariates, observation_weights, n_observations),
                method="L-BFGS-B",
                jac=_calculate_ipt_gradient_graham,
                options={"maxiter": 10000, "gtol": 1e-6},
            )
            if ipt_res.success:
                gamma_ipt = ipt_res.x
                linear_predictor_ipt = covariates @ gamma_ipt
                propensity_scores = 1 / (1 + np.exp(-linear_predictor_ipt))
                estimation_flag = 1
        except Exception as e:
            warnings.warn(
                f"IPT propensity score estimation (Graham et al. method) encountered an error: {e}", UserWarning
            )

    if estimation_flag not in [0, 1]:
        warnings.warn(
            "Calibrated and IPT propensity score methods failed or did not converge. Falling back to MLE (Logit).",
            UserWarning,
        )
        try:
            ps_fit_mle, _ = _estimate_propensity_score_mle(treatment_indicator, covariates, observation_weights)
            if not np.any(np.isnan(ps_fit_mle)):
                propensity_scores = ps_fit_mle
                estimation_flag = 2
            else:
                warnings.warn(
                    "Fallback MLE (Logit) for propensity score also failed to produce valid scores.", UserWarning
                )
        except Exception as e:
            warnings.warn(f"Fallback MLE for propensity score encountered an error: {e}", UserWarning)

    if estimation_flag in [0, 1, 2]:
        propensity_scores = np.clip(propensity_scores, 1e-16, 1.0 - 1e-16)
        if np.any(np.isnan(propensity_scores)):
            warnings.warn(
                "NaNs found in propensity scores despite a reported successful estimation method. "
                "This may indicate issues with input data or numerical instability.",
                UserWarning,
            )
            estimation_flag = 3

    if estimation_flag == 3:
        warnings.warn(
            "All propensity score estimation methods failed or resulted in invalid scores. "
            "Returning NaNs for propensity scores.",
            UserWarning,
        )
    return propensity_scores, estimation_flag
