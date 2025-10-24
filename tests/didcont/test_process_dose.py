"""Tests for continuous treatment dose-response processing."""

import numpy as np
import pytest
import scipy.stats as st

from moderndid.didcont.panel import (
    DoseResult,
    PTEParams,
    _summary_dose_result,
    process_dose_gt,
)
from moderndid.didcont.panel.process_dose import (
    _compute_dose_influence_functions,
    _compute_overall_att_inf_func,
    _multiplier_bootstrap_dose,
    _weighted_combine_arrays,
)
from tests.didcont.conftest import mock_gt_results


def test_dose_result_creation():
    dose_vals = np.array([0.5, 1.0, 1.5])
    result = DoseResult(
        dose=dose_vals,
        overall_att=0.15,
        overall_att_se=0.02,
        overall_att_inf_func=np.random.randn(100),
        overall_acrt=0.08,
        overall_acrt_se=0.01,
        overall_acrt_inf_func=np.random.randn(100),
        att_d=np.array([0.1, 0.15, 0.2]),
        att_d_se=np.array([0.02, 0.025, 0.03]),
        att_d_crit_val=1.96,
        att_d_inf_func=np.random.randn(100, 3),
        acrt_d=np.array([0.05, 0.08, 0.10]),
        acrt_d_se=np.array([0.01, 0.012, 0.015]),
        acrt_d_crit_val=1.96,
        acrt_d_inf_func=np.random.randn(100, 3),
        pte_params=None,
    )

    assert isinstance(result, DoseResult)
    assert len(result.dose) == 3
    assert result.overall_att == 0.15
    assert result.overall_att_se == 0.02
    assert len(result.att_d) == 3
    assert len(result.acrt_d) == 3


def test_dose_result_minimal():
    result = DoseResult(dose=np.array([]))

    assert isinstance(result, DoseResult)
    assert len(result.dose) == 0
    assert result.overall_att is None
    assert result.overall_att_se is None


def test_process_dose_gt_basic(mock_gt_results_with_dose, mock_pte_params_with_dose):
    result = process_dose_gt(mock_gt_results_with_dose, mock_pte_params_with_dose)

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None
    assert result.overall_att_se is not None
    assert result.overall_acrt is not None
    assert result.overall_acrt_se is not None
    assert len(result.dose) == len(mock_pte_params_with_dose.dvals)
    assert result.att_d is not None
    assert result.acrt_d is not None
    assert len(result.att_d) == len(result.dose)
    assert len(result.acrt_d) == len(result.dose)


def test_process_dose_gt_no_extra_returns(mock_gt_results_no_dose, mock_pte_params_with_dose):
    with pytest.raises(ValueError, match="No dose-specific results found"):
        process_dose_gt(mock_gt_results_no_dose, mock_pte_params_with_dose)


def test_process_dose_gt_no_dose_values(mock_gt_results_with_dose, pte_params_basic):
    params_dict = pte_params_basic._asdict()
    params_dict["dvals"] = None
    params = PTEParams(**params_dict)

    with pytest.warns(UserWarning, match="No dose values provided, returning overall results only"):
        result = process_dose_gt(mock_gt_results_with_dose, params)

    assert len(result.dose) == 0
    assert result.overall_att is not None
    assert result.overall_att_se is not None


def test_process_dose_gt_empty_dose_values(mock_gt_results_with_dose, pte_params_basic):
    params_dict = pte_params_basic._asdict()
    params_dict["dvals"] = np.array([])
    params = PTEParams(**params_dict)

    with pytest.warns(UserWarning, match="No dose values provided, returning overall results only"):
        result = process_dose_gt(mock_gt_results_with_dose, params)

    assert len(result.dose) == 0


def test_process_dose_gt_zero_degree(mock_pte_params_with_dose):
    params_dict = mock_pte_params_with_dose._asdict()
    params_dict["degree"] = 0
    params = PTEParams(**params_dict)

    n_basis = 3
    n_groups = 3
    n_times = 4
    n_gt = n_groups * n_times
    n_units = 200
    n_doses = 20

    np.random.seed(42)
    attgt_list = []
    extra_gt_returns = []

    for g in [2004, 2006, 2007]:
        for t in [2003, 2004, 2005, 2006]:
            att_value = np.random.normal(0.1, 0.05) if g > t else np.random.normal(0, 0.02)
            attgt_list.append({"att": att_value, "group": g, "time_period": t})

            dose_results = {
                "group": g,
                "time_period": t,
                "extra_gt_returns": {
                    "att_dose": np.random.normal(0.1, 0.02, n_doses),
                    "acrt_dose": np.random.normal(0.05, 0.01, n_doses),
                    "att_overall": np.random.normal(0.1, 0.02),
                    "acrt_overall": np.random.normal(0.05, 0.01),
                    "beta": np.random.randn(n_basis),
                    "bread": np.random.randn(n_basis, n_basis),
                    "x_expanded": np.random.randn(50, n_basis),
                },
            }
            extra_gt_returns.append(dose_results)

    influence_func = np.random.randn(n_units, n_gt) * 0.1

    gt_results = {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": extra_gt_returns}

    result = process_dose_gt(gt_results, params)

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None


def test_process_dose_gt_no_knots(mock_pte_params_with_dose):
    params_dict = mock_pte_params_with_dose._asdict()
    params_dict["knots"] = None
    params = PTEParams(**params_dict)

    gt_results = mock_gt_results(degree=params_dict["degree"], knots=None, n_doses=len(params_dict["dvals"]))

    result = process_dose_gt(gt_results, params)

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None


def test_process_dose_gt_confidence_band_false(mock_gt_results_with_dose, mock_pte_params_with_dose):
    params_dict = mock_pte_params_with_dose._asdict()
    params_dict["cband"] = False
    params = PTEParams(**params_dict)

    result = process_dose_gt(mock_gt_results_with_dose, params)

    assert result.att_d_crit_val == st.norm.ppf(1 - params.alp / 2)
    assert result.acrt_d_crit_val == st.norm.ppf(1 - params.alp / 2)


def test_weighted_combine_arrays_basic():
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    weights = np.array([0.2, 0.3, 0.5])

    result = _weighted_combine_arrays(arrays, weights)

    expected = 0.2 * arrays[0] + 0.3 * arrays[1] + 0.5 * arrays[2]
    assert np.allclose(result, expected)


def test_weighted_combine_arrays_with_none():
    arrays = [np.array([1, 2, 3]), None, np.array([7, 8, 9])]
    weights = np.array([0.3, 0.2, 0.5])

    result = _weighted_combine_arrays(arrays, weights)

    normalized_weights = np.array([0.3, 0.5]) / 0.8
    expected = normalized_weights[0] * arrays[0] + normalized_weights[1] * arrays[2]
    assert np.allclose(result, expected)


def test_weighted_combine_arrays_empty():
    result = _weighted_combine_arrays([], np.array([]))

    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_weighted_combine_arrays_all_none():
    arrays = [None, None, None]
    weights = np.array([0.3, 0.3, 0.4])

    result = _weighted_combine_arrays(arrays, weights)

    assert len(result) == 0


def test_compute_overall_att_inffunc_basic():
    np.random.seed(42)
    n_obs = 100
    n_groups = 5
    att_influence_matrix = np.random.randn(n_obs, n_groups)
    weights = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

    result = _compute_overall_att_inf_func(weights, att_influence_matrix)

    assert result.shape == (n_obs,)
    expected = np.sum(att_influence_matrix * weights[np.newaxis, :], axis=1)
    assert np.allclose(result, expected)


def test_compute_overall_att_inffunc_none():
    weights = np.array([0.5, 0.5])

    result = _compute_overall_att_inf_func(weights, None)

    assert result is None


def test_compute_dose_influence_functions_basic():
    np.random.seed(42)
    n_obs = 100
    n_groups = 3
    n_doses = 10
    n_basis = 4

    x_expanded_by_group = [np.random.randn(30, n_basis) for _ in range(n_groups)]
    bread_matrices = [np.random.randn(n_basis, n_basis) for _ in range(n_groups)]
    basis_matrix = np.random.randn(n_doses, n_basis)
    derivative_matrix = np.random.randn(n_doses, n_basis)
    acrt_influence_matrix = np.random.randn(n_obs, n_groups)
    att_influence_matrix = np.random.randn(n_obs, n_groups)
    weights = np.array([0.3, 0.4, 0.3])

    att_dose_inf, acrt_dose_inf = _compute_dose_influence_functions(
        x_expanded_by_group,
        bread_matrices,
        basis_matrix,
        derivative_matrix,
        acrt_influence_matrix,
        att_influence_matrix,
        weights,
        n_obs,
    )

    assert att_dose_inf.shape == (n_obs, n_doses)
    assert acrt_dose_inf.shape == (n_obs, n_doses)


def test_compute_dose_influence_functions_with_none():
    np.random.seed(42)
    n_obs = 100
    n_groups = 3
    n_doses = 10
    n_basis = 4

    x_expanded_by_group = [None, np.random.randn(30, n_basis), None]
    bread_matrices = [None, np.random.randn(n_basis, n_basis), None]
    basis_matrix = np.random.randn(n_doses, n_basis)
    derivative_matrix = np.random.randn(n_doses, n_basis)
    acrt_influence_matrix = np.random.randn(n_obs, n_groups)
    att_influence_matrix = np.random.randn(n_obs, n_groups)
    weights = np.array([0.3, 0.4, 0.3])

    att_dose_inf, acrt_dose_inf = _compute_dose_influence_functions(
        x_expanded_by_group,
        bread_matrices,
        basis_matrix,
        derivative_matrix,
        acrt_influence_matrix,
        att_influence_matrix,
        weights,
        n_obs,
    )

    assert att_dose_inf.shape == (n_obs, n_doses)
    assert acrt_dose_inf.shape == (n_obs, n_doses)


def test_compute_dose_influence_functions_zero_weights():
    np.random.seed(42)
    n_obs = 100
    n_groups = 3
    n_doses = 10
    n_basis = 4

    x_expanded_by_group = [np.random.randn(30, n_basis) for _ in range(n_groups)]
    bread_matrices = [np.random.randn(n_basis, n_basis) for _ in range(n_groups)]
    basis_matrix = np.random.randn(n_doses, n_basis)
    derivative_matrix = np.random.randn(n_doses, n_basis)
    acrt_influence_matrix = np.random.randn(n_obs, n_groups)
    att_influence_matrix = np.random.randn(n_obs, n_groups)
    weights = np.array([0.0, 1.0, 0.0])

    att_dose_inf, acrt_dose_inf = _compute_dose_influence_functions(
        x_expanded_by_group,
        bread_matrices,
        basis_matrix,
        derivative_matrix,
        acrt_influence_matrix,
        att_influence_matrix,
        weights,
        n_obs,
    )

    assert att_dose_inf.shape == (n_obs, n_doses)
    assert acrt_dose_inf.shape == (n_obs, n_doses)


def test_multiplier_bootstrap_dose_basic():
    np.random.seed(42)
    n_obs = 200
    n_doses = 15
    influence_function = np.random.randn(n_obs, n_doses) * 0.1

    result = _multiplier_bootstrap_dose(influence_function, biters=20, alpha=0.05)

    assert "se" in result
    assert "crit_val" in result
    assert result["se"].shape == (n_doses,)
    assert np.all(result["se"] > 0)
    assert result["crit_val"] > 0


def test_multiplier_bootstrap_dose_different_alpha():
    np.random.seed(42)
    n_obs = 150
    n_doses = 10
    influence_function = np.random.randn(n_obs, n_doses) * 0.1

    result_05 = _multiplier_bootstrap_dose(influence_function, biters=1000, alpha=0.05)
    result_10 = _multiplier_bootstrap_dose(influence_function, biters=1000, alpha=0.10)

    assert result_10["crit_val"] < result_05["crit_val"]


def test_multiplier_bootstrap_dose_single_dose():
    np.random.seed(42)
    n_obs = 100
    influence_function = np.random.randn(n_obs, 1) * 0.1

    result = _multiplier_bootstrap_dose(influence_function, biters=20, alpha=0.05)

    assert result["se"].shape == (1,)
    assert result["crit_val"] > 0


def test_summary_dose_result():
    dose_vals = np.array([0.5, 1.0, 1.5])
    result = DoseResult(
        dose=dose_vals,
        overall_att=0.15,
        overall_att_se=0.02,
        overall_att_inf_func=None,
        overall_acrt=0.08,
        overall_acrt_se=0.01,
        overall_acrt_inf_func=None,
        att_d=np.array([0.1, 0.15, 0.2]),
        att_d_se=np.array([0.02, 0.025, 0.03]),
        att_d_crit_val=1.96,
        att_d_inf_func=None,
        acrt_d=np.array([0.05, 0.08, 0.10]),
        acrt_d_se=np.array([0.01, 0.012, 0.015]),
        acrt_d_crit_val=1.96,
        acrt_d_inf_func=None,
        pte_params=None,
    )

    summary = _summary_dose_result(result)

    assert "dose" in summary
    assert "overall_att" in summary
    assert "overall_att_se" in summary
    assert "overall_acrt" in summary
    assert "overall_acrt_se" in summary
    assert "att_d" in summary
    assert "att_d_se" in summary
    assert "acrt_d" in summary
    assert "acrt_d_se" in summary
    assert summary["overall_att"] == 0.15
    assert summary["overall_acrt"] == 0.08


def test_summary_dose_result_with_params(mock_pte_params_with_dose):
    result = DoseResult(
        dose=np.array([0.5, 1.0]), overall_att=0.1, overall_att_se=0.02, pte_params=mock_pte_params_with_dose
    )

    summary = _summary_dose_result(result)

    assert "alpha" in summary
    assert "cband" in summary
    assert "biters" in summary
    assert summary["alpha"] == mock_pte_params_with_dose.alp
    assert summary["cband"] == mock_pte_params_with_dose.cband
    assert summary["biters"] == mock_pte_params_with_dose.biters


def test_dose_result_repr():
    result = DoseResult(
        dose=np.array([0.5, 1.0]),
        overall_att=0.15,
        overall_att_se=0.02,
        overall_acrt=0.08,
        overall_acrt_se=0.01,
        pte_params=None,
    )

    repr_str = repr(result)

    assert "Continuous Treatment Dose-Response Results" in repr_str
    assert "Overall ATT:" in repr_str
    assert "Overall ACRT:" in repr_str
    assert "0.1500" in repr_str
    assert "0.0800" in repr_str


def test_dose_result_str():
    result = DoseResult(
        dose=np.array([0.5, 1.0]),
        overall_att=0.15,
        overall_att_se=0.02,
        overall_acrt=0.08,
        overall_acrt_se=0.01,
        pte_params=None,
    )

    str_repr = str(result)

    assert "Continuous Treatment Dose-Response Results" in str_repr
    assert str_repr == repr(result)


def test_dose_result_repr_with_params(mock_pte_params_with_dose):
    result = DoseResult(
        dose=np.array([0.5, 1.0]),
        overall_att=0.15,
        overall_att_se=0.02,
        overall_acrt=0.08,
        overall_acrt_se=0.01,
        pte_params=mock_pte_params_with_dose,
    )

    repr_str = repr(result)

    assert "Spline Degree:" in repr_str
    assert "Number of Knots:" in repr_str
    assert "Control Group:" in repr_str


def test_dose_result_repr_no_effects():
    result = DoseResult(dose=np.array([]))

    repr_str = repr(result)

    assert "Continuous Treatment Dose-Response Results" in repr_str


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_process_dose_gt_different_degrees(mock_pte_params_with_dose, degree):
    params_dict = mock_pte_params_with_dose._asdict()
    params_dict["degree"] = degree
    params = PTEParams(**params_dict)

    gt_results = mock_gt_results(degree=degree, knots=params_dict["knots"], n_doses=len(params_dict["dvals"]))

    result = process_dose_gt(gt_results, params)

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None
    assert result.overall_att_se is not None


@pytest.mark.parametrize("n_knots", [0, 1, 3, 5])
def test_process_dose_gt_different_knots(mock_pte_params_with_dose, n_knots):
    params_dict = mock_pte_params_with_dose._asdict()
    if n_knots > 0:
        params_dict["knots"] = np.linspace(0.5, 1.5, n_knots)
    else:
        params_dict["knots"] = np.array([])
    params_dict["num_knots"] = n_knots
    params = PTEParams(**params_dict)

    gt_results = mock_gt_results(
        degree=params_dict["degree"], knots=params_dict["knots"], n_doses=len(params_dict["dvals"])
    )

    result = process_dose_gt(gt_results, params)

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None


def test_process_dose_gt_nan_in_dose_results(pte_params_basic):
    np.random.seed(42)
    n_groups = 2
    n_times = 2
    n_gt = n_groups * n_times
    n_units = 100
    n_doses = 10

    attgt_list = []
    extra_gt_returns = []

    for i, (g, t) in enumerate([(2004, 2003), (2004, 2004), (2006, 2003), (2006, 2004)]):
        att_value = np.random.normal(0.1, 0.05)
        attgt_list.append({"att": att_value, "group": g, "time_period": t})

        dose_results = {
            "group": g,
            "time_period": t,
            "extra_gt_returns": {
                "att_dose": np.full(n_doses, np.nan) if i == 0 else np.random.normal(0.1, 0.02, n_doses),
                "acrt_dose": np.full(n_doses, np.nan) if i == 0 else np.random.normal(0.05, 0.01, n_doses),
                "att_overall": np.nan if i == 0 else np.random.normal(0.1, 0.02),
                "acrt_overall": np.nan if i == 0 else np.random.normal(0.05, 0.01),
                "beta": np.random.randn(2),
                "bread": np.random.randn(2, 2),
                "x_expanded": np.random.randn(20, 2),
            },
        }
        extra_gt_returns.append(dose_results)

    influence_func = np.random.randn(n_units, n_gt) * 0.1

    gt_results = {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": extra_gt_returns}

    params_dict = pte_params_basic._asdict()
    params_dict.update(
        {
            "dname": "dose",
            "degree": 1,
            "num_knots": 0,
            "knots": np.array([]),
            "dvals": np.linspace(0.1, 1.0, n_doses),
        }
    )
    pte_params = PTEParams(**params_dict)

    result = process_dose_gt(gt_results, pte_params)

    assert isinstance(result, DoseResult)
    assert result.overall_att is not None


def test_process_dose_gt_mismatched_groups_times(pte_params_basic):
    np.random.seed(42)
    n_units = 100

    attgt_list = [
        {"att": 0.1, "group": 2004, "time_period": 2003},
        {"att": 0.2, "group": 2006, "time_period": 2003},
    ]

    extra_gt_returns = [
        {"group": 2006, "time_period": 2003, "extra_gt_returns": {}},
        {"group": 2004, "time_period": 2003, "extra_gt_returns": {}},
    ]

    influence_func = np.random.randn(n_units, 2) * 0.1

    gt_results = {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": extra_gt_returns}

    params_dict = pte_params_basic._asdict()
    params_dict.update(
        {
            "dname": "dose",
            "degree": 1,
            "num_knots": 0,
            "knots": np.array([]),
            "dvals": np.linspace(0.1, 1.0, 10),
        }
    )
    pte_params = PTEParams(**params_dict)

    with pytest.raises(ValueError, match="Mismatch between order of groups and time periods"):
        process_dose_gt(gt_results, pte_params)
