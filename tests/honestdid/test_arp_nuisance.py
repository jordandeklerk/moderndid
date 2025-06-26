# pylint: disable=redefined-outer-name
"""Tests for ARP CI with nuisance parameters."""

import numpy as np
import pytest

from pydid.honestdid.arp_nuisance import (
    ARPNuisanceCIResult,
    _check_if_solution,
    _compute_flci_vlo_vup,
    _compute_least_favorable_cv,
    _compute_vlo_vup_dual,
    _construct_gamma,
    _find_leading_one_column,
    _lp_conditional_test,
    _lp_dual_wrapper,
    _round_eps,
    _solve_max_program,
    _test_delta_lp,
    compute_arp_nuisance_ci,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_data(rng):
    n = 10
    return {
        "y_t": rng.normal(0, 1, n),
        "x_t": rng.normal(0, 1, (n, 2)),
        "sigma": np.eye(n),
        "n": n,
    }


@pytest.fixture
def panel_data(rng):
    num_pre = 3
    num_post = 2
    n = num_pre + num_post
    return {
        "betahat": np.concatenate([np.zeros(num_pre), rng.normal(0.5, 0.1, num_post)]),
        "sigma": 0.1 * np.eye(n),
        "l_vec": np.ones(num_post) / num_post,
        "num_pre": num_pre,
        "num_post": num_post,
        "n": n,
    }


@pytest.fixture
def constraint_data():
    return {
        "v": np.array([1.0, -1.0]),
        "a_matrix": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
        "d_vec": np.array([2.0, 2.0, 2.0, 2.0]),
    }


@pytest.fixture
def hybrid_list(simple_data):
    return {
        "hybrid_kappa": 0.05,
        "lf_cv": 2.0,
        "vbar": np.ones(simple_data["n"]) / simple_data["n"],
        "dbar": np.array([1.0, 1.0]),
        "flci_halflength": 0.5,
    }


@pytest.mark.parametrize(
    "value,eps,expected",
    [
        (1e-20, None, 0.0),
        (0.5, None, 0.5),
        (-0.5, None, -0.5),
        (1e-10, 1e-9, 0.0),
        (1e-10, 1e-11, 1e-10),
    ],
)
def test_round_eps(value, eps, expected):
    if eps is None:
        assert _round_eps(value) == expected
    else:
        assert _round_eps(value, eps) == expected


def test_find_leading_one_column():
    rref_matrix = np.array(
        [
            [1, 0, 0, 2],
            [0, 1, 0, 3],
            [0, 0, 1, 4],
        ]
    )
    assert _find_leading_one_column(0, rref_matrix) == 0
    assert _find_leading_one_column(1, rref_matrix) == 1
    assert _find_leading_one_column(2, rref_matrix) == 2

    with pytest.raises(ValueError):
        _find_leading_one_column(0, np.zeros((3, 3)))


@pytest.mark.parametrize(
    "l_vec",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.5]),
    ],
)
def test_construct_gamma(l_vec):
    gamma = _construct_gamma(l_vec)
    assert gamma.shape == (len(l_vec), len(l_vec))
    np.testing.assert_array_almost_equal(gamma[0], l_vec)
    assert abs(np.linalg.det(gamma)) > 1e-10


def test_test_delta_lp(simple_data):
    result = _test_delta_lp(simple_data["y_t"], simple_data["x_t"], simple_data["sigma"])
    assert "eta_star" in result
    assert "delta_star" in result
    assert "lambda" in result
    assert "success" in result
    assert result["success"]
    assert isinstance(result["eta_star"], float)
    assert result["delta_star"].shape == (2,)


def test_solve_max_program():
    s_t = np.array([1.0, 2.0, 3.0])
    gamma_tilde = np.array([1 / 3, 1 / 3, 1 / 3])
    sigma = np.eye(3)
    w_t = np.eye(3)
    c = 0.5

    result = _solve_max_program(s_t, gamma_tilde, sigma, w_t, c)
    assert result.success
    assert hasattr(result, "objective_value")


def test_check_if_solution():
    s_t = np.array([1.0, 2.0, 3.0])
    gamma_tilde = np.array([0.5, 0.3, 0.2])
    sigma = np.eye(3)
    w_t = np.column_stack([np.ones(3), np.eye(3)])

    result, is_solution = _check_if_solution(1.0, 1e-6, s_t, gamma_tilde, sigma, w_t)
    assert isinstance(is_solution, bool)
    assert hasattr(result, "success")


def test_compute_vlo_vup_dual(rng):
    n = 5
    s_t = rng.normal(0, 1, n)
    gamma_tilde = rng.dirichlet(np.ones(n))
    sigma = np.eye(n) + 0.1 * rng.normal(0, 1, (n, n))
    sigma = (sigma + sigma.T) / 2
    w_t = np.column_stack([np.ones(n), rng.normal(0, 1, (n, 2))])
    eta = 0.5

    result = _compute_vlo_vup_dual(eta, s_t, gamma_tilde, sigma, w_t)
    assert "vlo" in result
    assert "vup" in result
    assert result["vlo"] <= eta <= result["vup"]


def test_lp_dual_wrapper(simple_data):
    lp_result = _test_delta_lp(simple_data["y_t"], simple_data["x_t"], simple_data["sigma"])
    if lp_result["success"]:
        dual_result = _lp_dual_wrapper(
            simple_data["y_t"], simple_data["x_t"], lp_result["eta_star"], lp_result["lambda"], simple_data["sigma"]
        )
        assert "vlo" in dual_result
        assert "vup" in dual_result
        assert "eta" in dual_result
        assert "gamma_tilde" in dual_result


def test_compute_least_favorable_cv():
    np.random.seed(42)
    sigma = np.eye(5)

    cv_no_nuisance = _compute_least_favorable_cv(None, sigma, 0.1, sims=100)
    assert isinstance(cv_no_nuisance, float)
    assert cv_no_nuisance > 0

    x_t = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.6], [0.7, 0.1], [0.2, 0.3]])
    try:
        cv_with_nuisance = _compute_least_favorable_cv(x_t, sigma, 0.1, sims=50, seed=42)
        assert isinstance(cv_with_nuisance, float)
    except RuntimeError:
        pass


def test_compute_flci_vlo_vup():
    vbar = np.array([0.5, 0.5])
    dbar = np.array([1.0, 1.0])
    s_vec = np.array([0.2, 0.3])
    c_vec = np.array([0.1, 0.2])

    result = _compute_flci_vlo_vup(vbar, dbar, s_vec, c_vec)
    assert "vlo" in result
    assert "vup" in result
    assert result["vlo"] <= result["vup"]


@pytest.mark.parametrize("x_t", [None, np.random.normal(0, 1, (10, 2))])
def test_lp_conditional_test_with_without_nuisance(simple_data, x_t):
    result = _lp_conditional_test(
        y_t=simple_data["y_t"],
        x_t=x_t,
        sigma=simple_data["sigma"],
        alpha=0.05,
        hybrid_flag="ARP",
    )

    assert isinstance(result["reject"], bool)
    assert isinstance(result["eta"], float)
    if x_t is None:
        assert len(result["delta"]) == 0
        assert len(result["lambda"]) == 0
    else:
        assert result["delta"].shape == (2,)
        assert len(result["lambda"]) > 0


@pytest.mark.parametrize("hybrid_flag", ["ARP", "LF", "FLCI"])
def test_lp_conditional_test_hybrid_types(simple_data, hybrid_list, hybrid_flag):
    result = _lp_conditional_test(
        y_t=simple_data["y_t"],
        x_t=simple_data["x_t"],
        sigma=simple_data["sigma"],
        alpha=0.10,
        hybrid_flag=hybrid_flag,
        hybrid_list=hybrid_list,
    )
    assert isinstance(result["reject"], bool)


def test_compute_arp_nuisance_ci_basic(panel_data):
    a_matrix = np.vstack([np.eye(panel_data["num_post"]), -np.eye(panel_data["num_post"])])
    a_matrix = np.column_stack([np.zeros((a_matrix.shape[0], panel_data["num_pre"])), a_matrix])
    d_vec = np.ones(2 * panel_data["num_post"])

    result = compute_arp_nuisance_ci(
        betahat=panel_data["betahat"],
        sigma=panel_data["sigma"],
        l_vec=panel_data["l_vec"],
        a_matrix=a_matrix,
        d_vec=d_vec,
        num_pre_periods=panel_data["num_pre"],
        num_post_periods=panel_data["num_post"],
        alpha=0.05,
        grid_lb=-2.0,
        grid_ub=2.0,
        grid_points=50,
    )

    assert isinstance(result, ARPNuisanceCIResult)
    assert isinstance(result.ci_lb, float)
    assert isinstance(result.ci_ub, float)
    assert result.ci_lb <= result.ci_ub
    assert result.accept_grid.shape[1] == 2
    assert isinstance(result.length, float)


@pytest.mark.parametrize("return_length", [True, False])
def test_compute_arp_nuisance_ci_return_length(panel_data, return_length):
    a_matrix = np.eye(panel_data["num_post"])
    a_matrix = np.column_stack([np.zeros((a_matrix.shape[0], panel_data["num_pre"])), a_matrix])
    d_vec = np.ones(panel_data["num_post"])

    result = compute_arp_nuisance_ci(
        betahat=panel_data["betahat"],
        sigma=panel_data["sigma"],
        l_vec=panel_data["l_vec"],
        a_matrix=a_matrix,
        d_vec=d_vec,
        num_pre_periods=panel_data["num_pre"],
        num_post_periods=panel_data["num_post"],
        alpha=0.05,
        grid_points=20,
        return_length=return_length,
    )

    assert isinstance(result.length, float)
    if not return_length:
        assert result.length == pytest.approx(result.ci_ub - result.ci_lb, rel=1e-6)


def test_compute_arp_nuisance_ci_with_rows_subset(rng):
    num_pre = 4
    num_post = 3
    n = num_pre + num_post

    betahat = rng.normal(0, 0.1, n)
    sigma = 0.05 * np.eye(n)
    l_vec = np.ones(num_post) / num_post
    a_matrix = np.eye(n)
    d_vec = 0.5 * np.ones(n)
    rows_for_arp = np.array([0, 1, 2, 5, 6])

    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=a_matrix,
        d_vec=d_vec,
        num_pre_periods=num_pre,
        num_post_periods=num_post,
        alpha=0.05,
        grid_points=30,
        rows_for_arp=rows_for_arp,
    )

    assert isinstance(result, ARPNuisanceCIResult)
    assert not np.isnan(result.ci_lb)
    assert not np.isnan(result.ci_ub)


def test_all_hybrid_flags():
    y_t = np.array([1.0, 2.0])
    sigma = np.eye(2)

    result = _lp_conditional_test(
        y_t=y_t,
        x_t=None,
        sigma=sigma,
        alpha=0.05,
        hybrid_flag="ARP",
    )
    assert isinstance(result, dict)
    assert "reject" in result

    result = _lp_conditional_test(
        y_t=y_t,
        x_t=None,
        sigma=sigma,
        alpha=0.05,
        hybrid_flag="LF",
        hybrid_list={"hybrid_kappa": 0.05, "lf_cv": 2.0},
    )
    assert isinstance(result, dict)
    assert "reject" in result

    result = _lp_conditional_test(
        y_t=y_t,
        x_t=None,
        sigma=sigma,
        alpha=0.05,
        hybrid_flag="FLCI",
        hybrid_list={"hybrid_kappa": 0.05, "vbar": np.ones(2) / 2, "dbar": np.array([1.0, 1.0])},
    )
    assert isinstance(result, dict)
    assert "reject" in result
