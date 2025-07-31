# pylint: disable=redefined-outer-name
"""Tests for ARP nuisance CI."""

import numpy as np
import pytest

from doublediff.didhonest.arp_nuisance import (
    ARPNuisanceCIResult,
    _check_if_solution,
    _compute_flci_vlo_vup,
    _construct_gamma,
    _find_leading_one_column,
    _lp_dual_wrapper,
    _round_eps,
    _solve_max_program,
    _test_delta_lp,
    compute_arp_nuisance_ci,
    compute_least_favorable_cv,
    compute_vlo_vup_dual,
    lp_conditional_test,
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
def hybrid_list():
    return {
        "hybrid_kappa": 0.05,
        "lf_cv": 2.0,
        "vbar": np.ones(2) / 2,
        "dbar": np.array([1.0, 1.0]),
        "flci_halflength": 0.5,
    }


@pytest.fixture
def edge_case_data(rng):
    return {
        "singular_sigma": np.array([[1.0, 1.0], [1.0, 1.0]]),
        "near_singular_sigma": np.array([[1.0, 0.999], [0.999, 1.0]]),
        "ill_conditioned_x": np.array([[1.0, 1e-10], [1.0, 2e-10], [1.0, 3e-10]]),
        "extreme_values": rng.normal(0, 1000, 5),
        "tiny_values": rng.normal(0, 1e-10, 5),
    }


@pytest.fixture
def high_dim_data(rng):
    n = 50
    k = 10
    return {
        "y_t": rng.normal(0, 1, n),
        "x_t": rng.normal(0, 1, (n, k)),
        "sigma": np.eye(n) + 0.1 * rng.normal(0, 1, (n, n)),
        "n": n,
        "k": k,
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

    result = compute_vlo_vup_dual(eta, s_t, gamma_tilde, sigma, w_t)
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

    cv_no_nuisance = compute_least_favorable_cv(None, sigma, 0.1, sims=100)
    assert isinstance(cv_no_nuisance, float)
    assert cv_no_nuisance > 0

    x_t = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.6], [0.7, 0.1], [0.2, 0.3]])
    try:
        cv_with_nuisance = compute_least_favorable_cv(x_t, sigma, 0.1, sims=50, seed=42)
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
    result = lp_conditional_test(
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
def test_lp_conditional_test_hybrid_types(simple_data, hybrid_flag):
    hybrid_list_test = {
        "hybrid_kappa": 0.05,
        "lf_cv": 2.0,
        "vbar": np.ones(simple_data["n"]) / simple_data["n"],
        "dbar": np.array([1.0, 1.0]),
        "flci_halflength": 0.5,
    }
    result = lp_conditional_test(
        y_t=simple_data["y_t"],
        x_t=simple_data["x_t"],
        sigma=simple_data["sigma"],
        alpha=0.10,
        hybrid_flag=hybrid_flag,
        hybrid_list=hybrid_list_test,
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

    result = lp_conditional_test(
        y_t=y_t,
        x_t=None,
        sigma=sigma,
        alpha=0.05,
        hybrid_flag="ARP",
    )
    assert isinstance(result, dict)
    assert "reject" in result

    result = lp_conditional_test(
        y_t=y_t,
        x_t=None,
        sigma=sigma,
        alpha=0.05,
        hybrid_flag="LF",
        hybrid_list={"hybrid_kappa": 0.05, "lf_cv": 2.0},
    )
    assert isinstance(result, dict)
    assert "reject" in result

    result = lp_conditional_test(
        y_t=y_t,
        x_t=None,
        sigma=sigma,
        alpha=0.05,
        hybrid_flag="FLCI",
        hybrid_list={"hybrid_kappa": 0.05, "vbar": np.ones(2) / 2, "dbar": np.array([1.0, 1.0])},
    )
    assert isinstance(result, dict)
    assert "reject" in result


def test_empty_ci_case(panel_data):
    a_matrix = np.eye(panel_data["num_post"])
    a_matrix = np.column_stack([np.zeros((a_matrix.shape[0], panel_data["num_pre"])), a_matrix])
    d_vec = -10 * np.ones(panel_data["num_post"])

    result = compute_arp_nuisance_ci(
        betahat=panel_data["betahat"],
        sigma=panel_data["sigma"],
        l_vec=panel_data["l_vec"],
        a_matrix=a_matrix,
        d_vec=d_vec,
        num_pre_periods=panel_data["num_pre"],
        num_post_periods=panel_data["num_post"],
        alpha=0.05,
        grid_lb=-1.0,
        grid_ub=1.0,
        grid_points=20,
    )

    assert np.isnan(result.ci_lb)
    assert np.isnan(result.ci_ub)
    assert np.isnan(result.length)


def test_ci_open_at_endpoints(panel_data):
    a_matrix = 0.1 * np.eye(panel_data["num_post"])
    a_matrix = np.column_stack([np.zeros((a_matrix.shape[0], panel_data["num_pre"])), a_matrix])
    d_vec = 10 * np.ones(panel_data["num_post"])

    with pytest.warns(UserWarning, match="CI is open at one of the endpoints"):
        compute_arp_nuisance_ci(
            betahat=panel_data["betahat"],
            sigma=panel_data["sigma"],
            l_vec=panel_data["l_vec"],
            a_matrix=a_matrix,
            d_vec=d_vec,
            num_pre_periods=panel_data["num_pre"],
            num_post_periods=panel_data["num_post"],
            alpha=0.05,
            grid_lb=-0.1,
            grid_ub=0.1,
            grid_points=10,
        )


@pytest.mark.parametrize("invalid_hybrid_flag", ["invalid", "INVALID", ""])
def test_invalid_hybrid_flag(simple_data, invalid_hybrid_flag):
    with pytest.raises(ValueError, match=f"Invalid hybrid_flag: {invalid_hybrid_flag}"):
        lp_conditional_test(
            y_t=simple_data["y_t"],
            x_t=simple_data["x_t"],
            sigma=simple_data["sigma"],
            alpha=0.05,
            hybrid_flag=invalid_hybrid_flag,
        )


def test_singular_covariance_matrix(edge_case_data):
    y_t = np.array([1.0, 1.0])
    x_t = np.array([[1.0], [1.0]])

    result = lp_conditional_test(
        y_t=y_t,
        x_t=x_t,
        sigma=edge_case_data["singular_sigma"],
        alpha=0.05,
    )
    assert isinstance(result, dict)


def test_near_singular_covariance(edge_case_data):
    y_t = np.array([1.0, 0.5])
    x_t = np.array([[1.0, 0.5], [0.5, 1.0]])

    result = _test_delta_lp(y_t, x_t, edge_case_data["near_singular_sigma"])
    assert result["success"] or np.isnan(result["eta_star"])


def test_ill_conditioned_constraints(edge_case_data):
    n = edge_case_data["ill_conditioned_x"].shape[0]
    y_t = np.random.normal(0, 1, n)
    sigma = np.eye(n)

    result = _test_delta_lp(y_t, edge_case_data["ill_conditioned_x"], sigma)
    assert "success" in result
    if not result["success"]:
        assert np.isnan(result["eta_star"])


def test_extreme_value_handling(edge_case_data):
    n = len(edge_case_data["extreme_values"])
    x_t = np.random.normal(0, 1, (n, 2))
    sigma = np.eye(n)

    result = _test_delta_lp(edge_case_data["extreme_values"], x_t, sigma)
    assert "eta_star" in result
    assert not np.isinf(result["eta_star"]) if result["success"] else True


def test_high_dimensional_case(high_dim_data):
    sigma = high_dim_data["sigma"]
    sigma = (sigma + sigma.T) / 2
    eigenvalues = np.linalg.eigvalsh(sigma)
    if eigenvalues.min() < 0.1:
        sigma += (0.1 - eigenvalues.min()) * np.eye(high_dim_data["n"])

    result = lp_conditional_test(
        y_t=high_dim_data["y_t"],
        x_t=high_dim_data["x_t"],
        sigma=sigma,
        alpha=0.05,
    )
    assert isinstance(result["reject"], bool)
    assert result["delta"].shape == (high_dim_data["k"],)


def test_binding_constraint_configurations(rng):
    n = 6
    k = 2

    y_t = np.zeros(n)
    x_t = rng.normal(0, 1, (n, k))
    sigma = np.eye(n)

    result1 = lp_conditional_test(y_t, x_t, sigma, alpha=0.05)

    y_t = -10 * np.ones(n)
    result2 = lp_conditional_test(y_t, x_t, sigma, alpha=0.05)

    y_t = np.concatenate([np.zeros(3), -10 * np.ones(3)])
    result3 = lp_conditional_test(y_t, x_t, sigma, alpha=0.05)

    for result in [result1, result2, result3]:
        assert isinstance(result["reject"], bool)
        assert isinstance(result["eta"], float)


def test_bisection_convergence():
    n = 5
    s_t = np.array([1.0, -0.5, 0.3, -0.2, 0.8])
    gamma_tilde = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    sigma = np.eye(n) + 0.1 * np.ones((n, n))
    w_t = np.column_stack([np.ones(n), np.eye(n)])

    result = compute_vlo_vup_dual(0.5, s_t, gamma_tilde, sigma, w_t)
    assert result["vlo"] < result["vup"]
    assert not np.isinf(result["vlo"]) or not np.isinf(result["vup"])


def test_least_favorable_cv_reproducibility():
    sigma = np.eye(4)
    x_t = np.random.RandomState(42).normal(0, 1, (4, 2))

    cv1 = compute_least_favorable_cv(x_t, sigma, 0.05, sims=100, seed=123)
    cv2 = compute_least_favorable_cv(x_t, sigma, 0.05, sims=100, seed=123)

    if not (isinstance(cv1, float) and isinstance(cv2, float)):
        pytest.skip("LF CV computation failed")

    assert cv1 == cv2


def test_flci_bounds_computation():
    vbar = np.array([[0.5, 0.3, 0.2], [-0.3, 0.4, -0.3]])
    dbar = np.array([1.0, 0.5, 1.0, 0.5])
    s_vec = np.array([0.1, 0.2, 0.3])
    c_vec = np.array([0.2, 0.1, 0.15])

    result = _compute_flci_vlo_vup(vbar.T, dbar, s_vec, c_vec)

    assert result["vlo"] <= result["vup"]

    result_inf = _compute_flci_vlo_vup(np.zeros((3, 2)), np.array([1.0, 1.0, 1.0, 1.0]), s_vec, c_vec)
    assert np.isinf(result_inf["vlo"]) or np.isinf(result_inf["vup"])


@pytest.mark.parametrize(
    "method,needs_hybrid_list",
    [
        ("ARP", False),
        ("LF", True),
        ("FLCI", True),
    ],
)
def test_hybrid_method_transitions(panel_data, hybrid_list, method, needs_hybrid_list):
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
        hybrid_flag=method,
        hybrid_list=hybrid_list if needs_hybrid_list else None,
        grid_points=30,
    )

    assert isinstance(result.ci_lb, float)
    assert isinstance(result.ci_ub, float)
    assert result.ci_lb <= result.ci_ub


def test_ci_monotonicity_in_alpha():
    num_pre, num_post = 3, 2
    betahat = np.concatenate([np.zeros(num_pre), [0.5, 0.3]])
    sigma = 0.1 * np.eye(num_pre + num_post)
    l_vec = np.ones(num_post) / num_post
    a_matrix = np.column_stack([np.zeros((num_post, num_pre)), np.eye(num_post)])
    d_vec = np.ones(num_post)

    results = {}
    for alpha in [0.01, 0.05, 0.10]:
        result = compute_arp_nuisance_ci(
            betahat=betahat,
            sigma=sigma,
            l_vec=l_vec,
            a_matrix=a_matrix,
            d_vec=d_vec,
            num_pre_periods=num_pre,
            num_post_periods=num_post,
            alpha=alpha,
            grid_points=50,
        )
        results[alpha] = result.length

    assert results[0.01] >= results[0.05]
    assert results[0.05] >= results[0.10]


def test_dual_primal_consistency(simple_data):
    lp_result = _test_delta_lp(simple_data["y_t"], simple_data["x_t"], simple_data["sigma"])

    if not lp_result["success"]:
        pytest.skip("LP failed to converge")

    dual_result = _lp_dual_wrapper(
        simple_data["y_t"], simple_data["x_t"], lp_result["eta_star"], lp_result["lambda"], simple_data["sigma"]
    )

    assert dual_result["eta"] == pytest.approx(lp_result["eta_star"], rel=1e-6)


def test_gamma_matrix_properties():
    test_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.33, 0.33, 0.34]),
        np.ones(5) / 5,
    ]

    for l_vec in test_vectors:
        gamma = _construct_gamma(l_vec)

        np.testing.assert_array_almost_equal(gamma[0], l_vec)
        assert np.linalg.cond(gamma) < 1e10

        gamma_inv = np.linalg.inv(gamma)
        identity = gamma @ gamma_inv
        np.testing.assert_array_almost_equal(identity, np.eye(len(l_vec)))


def test_influence_function_properties(panel_data):
    a_matrix = np.eye(panel_data["num_post"])
    a_matrix = np.column_stack([np.zeros((a_matrix.shape[0], panel_data["num_pre"])), a_matrix])
    d_vec = 0.5 * np.ones(panel_data["num_post"])

    y_t = a_matrix @ panel_data["betahat"] - d_vec
    x_t = np.random.normal(0, 1, (len(y_t), 2))
    sigma_y = a_matrix @ panel_data["sigma"] @ a_matrix.T

    result = lp_conditional_test(
        y_t=y_t,
        x_t=x_t,
        sigma=sigma_y,
        alpha=0.05,
        hybrid_flag="ARP",
    )

    assert len(result["lambda"]) == len(y_t)
    assert np.all(result["lambda"] >= -1e-10)


def test_numerical_precision_handling():
    assert _round_eps(1e-20) == 0.0
    assert _round_eps(1e-13) == 0.0
    assert _round_eps(1e-5) != 0.0

    assert _round_eps(1e-5, eps=1e-4) == 0.0
    assert _round_eps(1e-5, eps=1e-6) != 0.0

    y_t = np.array([1e-15, 1e-14, 1e-13])
    x_t = np.eye(3)
    sigma = np.eye(3)

    result = _test_delta_lp(y_t, x_t, sigma)
    assert "success" in result
