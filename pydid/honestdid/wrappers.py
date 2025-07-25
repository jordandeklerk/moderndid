"""Interface for delta method selection."""

from .delta.rm import (
    compute_conditional_cs_rm,
    compute_conditional_cs_rmb,
    compute_conditional_cs_rmm,
)
from .delta.sd import (
    compute_conditional_cs_sd,
    compute_conditional_cs_sdb,
    compute_conditional_cs_sdm,
)
from .delta.sdrm import (
    compute_conditional_cs_sdrm,
    compute_conditional_cs_sdrmb,
    compute_conditional_cs_sdrmm,
)


class DeltaMethodSelector:
    """Select appropriate delta method based on configuration."""

    @staticmethod
    def get_smoothness_method(
        monotonicity_direction,
        bias_direction,
    ):
        """Get delta method for smoothness restrictions.

        Parameters
        ----------
        monotonicity_direction : str, optional
            Direction of monotonicity restriction: "increasing" or "decreasing".
        bias_direction : str, optional
            Direction of bias restriction: "positive" or "negative".

        Returns
        -------
        tuple
            A tuple of (method_function, delta_type_name).
        """
        if monotonicity_direction is None and bias_direction is None:
            return compute_conditional_cs_sd, "DeltaSD"
        if monotonicity_direction is not None:
            delta_type = "DeltaSDI" if monotonicity_direction == "increasing" else "DeltaSDD"
            return compute_conditional_cs_sdm, delta_type
        delta_type = "DeltaSDPB" if bias_direction == "positive" else "DeltaSDNB"
        return compute_conditional_cs_sdb, delta_type

    @staticmethod
    def get_relative_magnitude_method(
        bound_type,
        monotonicity_direction,
        bias_direction,
    ):
        """Get delta method for relative magnitude restrictions.

        Parameters
        ----------
        bound_type : str
            Type of bound: "deviation from parallel trends" or "deviation from linear trend".
        monotonicity_direction : str, optional
            Direction of monotonicity restriction: "increasing" or "decreasing".
        bias_direction : str, optional
            Direction of bias restriction: "positive" or "negative".

        Returns
        -------
        tuple
            A tuple of (method_function, delta_type_name).
        """
        if bound_type == "deviation from parallel trends":
            if monotonicity_direction is None and bias_direction is None:
                return compute_conditional_cs_rm, "DeltaRM"
            if monotonicity_direction is not None:
                delta_type = "DeltaRMI" if monotonicity_direction == "increasing" else "DeltaRMD"
                return compute_conditional_cs_rmm, delta_type
            delta_type = "DeltaRMPB" if bias_direction == "positive" else "DeltaRMNB"
            return compute_conditional_cs_rmb, delta_type
        if monotonicity_direction is None and bias_direction is None:
            return compute_conditional_cs_sdrm, "DeltaSDRM"
        if monotonicity_direction is not None:
            delta_type = "DeltaSDRMI" if monotonicity_direction == "increasing" else "DeltaSDRMD"
            return compute_conditional_cs_sdrmm, delta_type
        delta_type = "DeltaSDRMPB" if bias_direction == "positive" else "DeltaSDRMNB"
        return compute_conditional_cs_sdrmb, delta_type


def get_delta_method(
    sensitivity_type,
    bound_type,
    monotonicity_direction,
    bias_direction,
):
    """Get appropriate delta method based on parameters.

    Parameters
    ----------
    sensitivity_type : {'smoothness', 'relative_magnitude'}
        Type of sensitivity analysis.
    bound_type : str, default="deviation from parallel trends"
        Type of bound for relative magnitude methods.
    monotonicity_direction : str, optional
        Direction of monotonicity restriction: "increasing" or "decreasing".
    bias_direction : str, optional
        Direction of bias restriction: "positive" or "negative".

    Returns
    -------
    tuple
        A tuple of (method_function, delta_type_name).

    Raises
    ------
    ValueError
        If invalid sensitivity_type is provided.
    """
    if sensitivity_type == "smoothness":
        return DeltaMethodSelector.get_smoothness_method(
            monotonicity_direction=monotonicity_direction,
            bias_direction=bias_direction,
        )
    if sensitivity_type == "relative_magnitude":
        return DeltaMethodSelector.get_relative_magnitude_method(
            bound_type=bound_type,
            monotonicity_direction=monotonicity_direction,
            bias_direction=bias_direction,
        )
    raise ValueError(f"sensitivity_type must be 'smoothness' or 'relative_magnitude', got {sensitivity_type}")
