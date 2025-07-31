"""Print formatting for DiD estimation results."""

from typing import NamedTuple

import numpy as np


def format_did_result(result: NamedTuple) -> str:
    """Format DiD estimation results."""
    att = result.att
    se = result.se
    lci = result.lci
    uci = result.uci
    args = getattr(result, "args", {})

    result_type_name = type(result).__name__
    estimator_type = args.get("type", _infer_estimator_type(result_type_name))
    est_method = args.get("est_method", args.get("estMethod", "default"))
    panel = args.get("panel", _infer_panel_type(result_type_name))
    boot = args.get("boot", False)
    nboot = args.get("nboot", 999)
    boot_type = args.get("boot_type", "weighted")

    t_stat = att / se if se > 0 and not np.isnan(se) else np.nan
    p_value = 2 * (1 - _norm_cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan

    att_label_str = "ATT"
    att_val_str = f"{att:.4f}" if not np.isnan(att) else "NaN"
    se_val_str = f"{se:.4f}" if not np.isnan(se) else "NaN"
    t_val_str = f"{t_stat:.4f}" if not np.isnan(t_stat) else "NaN"

    if np.isnan(p_value):
        p_val_str = "NaN"
    elif p_value < 0.001:
        p_val_str = "<0.001"
    else:
        p_val_str = f"{p_value:.4f}"

    lci_s = f"{lci:.4f}" if not np.isnan(lci) else "NaN"
    uci_s = f"{uci:.4f}" if not np.isnan(uci) else "NaN"
    conf_int_val_str = f"[{lci_s}, {uci_s}]"

    headers = ["", "Estimate", "Std. Error", "t-value", "Pr(>|t|)", "[95% Conf. Interval]"]
    values = [att_label_str, att_val_str, se_val_str, t_val_str, p_val_str, conf_int_val_str]

    col_padding = 2
    w = [max(len(h), len(v)) for h, v in zip(headers, values)]
    w[0] += 1
    for i in range(1, len(w)):
        w[i] += col_padding

    total_width = sum(w)

    lines = []

    title = _get_estimator_title(estimator_type, est_method)
    lines.append("=" * total_width)
    lines.append(f" {title}")
    lines.append("=" * total_width)

    if hasattr(result, "call_params") and "data_shape" in result.call_params:
        n_obs, n_cols = result.call_params["data_shape"]
        lines.append(f" Computed from {n_obs} observations and {n_cols - 4} covariates.")

    lines.append("")

    header = (
        f"{headers[0]:<{w[0]}}"
        f"{headers[1]:>{w[1]}}"
        f"{headers[2]:>{w[2]}}"
        f"{headers[3]:>{w[3]}}"
        f"{headers[4]:>{w[4]}}"
        f"{headers[5]:>{w[5]}}"
    )
    lines.append(header)
    lines.append("-" * total_width)

    values_line = (
        f"{att_label_str:<{w[0]}}"
        f"{att_val_str:>{w[1]}}"
        f"{se_val_str:>{w[2]}}"
        f"{t_val_str:>{w[3]}}"
        f"{p_val_str:>{w[4]}}"
        f"{conf_int_val_str:>{w[5]}}"
    )
    lines.append(values_line)

    lines.append("")
    lines.append("-" * total_width)
    lines.append(" Method Details:")

    if panel is not None:
        data_type = "Panel data" if panel else "Repeated cross-sections"
        lines.append(f"   Data structure: {data_type}")

    method_details = _get_method_description(estimator_type, est_method, args)
    for detail in method_details:
        lines.append(f"   {detail}")

    lines.append("")
    lines.append(" Inference:")
    if boot:
        lines.append(f"   Standard errors: Bootstrapped ({nboot} replications)")
        lines.append(f"   Bootstrap type: {boot_type.capitalize()}")
    else:
        lines.append("   Standard errors: Analytical")

    if "trim_level" in args and estimator_type in ["ipw", "dr"]:
        lines.append(f"   Propensity score trimming: {args['trim_level']}")

    if np.isnan(att) or (se is not None and np.isnan(se)):
        lines.append("")
        lines.append("Warning: Estimation failed. Check for data issues:")
        if "warning_msg" in args:
            lines.append(f"     {args['warning_msg']}")
        else:
            lines.append("     - Insufficient variation in treatment/control groups")
            lines.append("     - Perfect separation in propensity score")
            lines.append("     - Collinearity in covariates")

    lines.append("=" * total_width)

    if estimator_type == "dr":
        lines.append(" Reference: Sant'Anna and Zhao (2020), Journal of Econometrics")
    elif estimator_type == "ipw":
        lines.append(" Reference: Abadie (2005), Review of Economic Studies")
    elif estimator_type == "or":
        lines.append(" Reference: Heckman et al. (1997), Review of Economic Studies")

    return "\n".join(lines)


def _infer_estimator_type(result_type_name: str) -> str:
    """Infer estimator type from result class name."""
    name_lower = result_type_name.lower()
    if "drdid" in name_lower:
        return "dr"
    if "ipwdid" in name_lower or "stdipw" in name_lower:
        return "ipw"
    if "regdid" in name_lower or "ordid" in name_lower:
        return "or"
    if "twfe" in name_lower:
        return "twfe"
    return "unknown"


def _infer_panel_type(result_type_name: str) -> bool | None:
    """Infer whether panel data was used from result class name."""
    name_lower = result_type_name.lower()
    if "panel" in name_lower:
        return True
    if "rc" in name_lower:
        return False
    return None


def _get_estimator_title(estimator_type: str, est_method: str) -> str:
    """Get descriptive title for the estimator."""
    titles = {
        "dr": {
            "imp": "Doubly Robust DiD Estimator (Improved Method)",
            "trad": "Doubly Robust DiD Estimator (Traditional Method)",
            "imp_local": "Doubly Robust DiD Estimator (Improved Locally Efficient Method)",
            "trad_local": "Doubly Robust DiD Estimator (Traditional Local Method)",
            "default": "Doubly Robust DiD Estimator",
        },
        "ipw": {
            "ipw": "Inverse Probability Weighted DiD Estimator",
            "std_ipw": "Standardized IPW DiD Estimator (Hajek-type)",
            "default": "Inverse Probability Weighted DiD Estimator",
        },
        "or": {"default": "Outcome Regression DiD Estimator"},
        "twfe": {"default": "Two-Way Fixed Effects DiD Estimator"},
    }

    if estimator_type in titles:
        return titles[estimator_type].get(est_method, titles[estimator_type]["default"])
    return "Difference-in-Differences Estimator"


def _get_method_description(estimator_type: str, est_method: str, args: dict) -> list[str]:
    """Get method-specific description lines."""
    details = []

    if estimator_type == "dr":
        if est_method == "imp":
            details.append("Outcome regression: Weighted least squares")
            details.append("Propensity score: Inverse probability tilting")
        elif est_method == "trad":
            details.append("Outcome regression: Ordinary least squares")
            details.append("Propensity score: Logistic regression")
        elif est_method == "imp_local":
            details.append("Outcome regression: Locally weighted least squares")
            details.append("Propensity score: Inverse probability tilting")
        elif est_method == "trad_local":
            details.append("Outcome regression: Local linear regression")
            details.append("Propensity score: Logistic regression")
        else:
            details.append("Outcome regression and propensity score models")

    elif estimator_type == "ipw":
        normalized = args.get("normalized", True)
        if est_method == "std_ipw" or normalized:
            details.append("Weight type: Normalized (Hajek-type estimator)")
        else:
            details.append("Weight type: Unnormalized (Horvitz-Thompson-type)")
        details.append("Propensity score: Logistic regression")

    elif estimator_type == "or":
        details.append("Estimation method: Ordinary least squares")

    elif estimator_type == "twfe":
        details.append("Fixed effects: Unit and time")
        details.append("Estimation method: Within transformation")

    return details


def _norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def print_did_result(result_class):
    """Add __repr__ and __str__ methods to a NamedTuple result class."""

    def __repr__(self):
        return format_did_result(self)

    def __str__(self):
        return format_did_result(self)

    result_class.__repr__ = __repr__
    result_class.__str__ = __str__

    return result_class
