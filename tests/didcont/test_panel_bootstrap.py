# pylint: disable=redefined-outer-name, unused-argument
"""Tests for panel empirical bootstrap."""

import numpy as np
import pandas as pd
import pytest

from moderndid.didcont.panel.bootstrap import (
    _combine_ecdfs,
    _convert_to_original_time,
    _make_ecdf,
    attgt_pte_aggregations,
    block_boot_sample,
    panel_empirical_bootstrap,
    qott_pte_aggregations,
    qtt_pte_aggregations,
)
from moderndid.didcont.panel.container import PteEmpBootResult


def test_block_boot_sample_balanced(balanced_panel_data_bootstrap):
    np.random.seed(42)
    boot_data = block_boot_sample(balanced_panel_data_bootstrap, "id")

    assert len(boot_data) == len(balanced_panel_data_bootstrap)
    assert set(boot_data["id"].unique()) == {0, 1, 2}
    assert all(boot_data.groupby("id")["time"].count().values == 4)
    assert set(boot_data["time"].unique()) == set(balanced_panel_data_bootstrap["time"].unique())


def test_block_boot_sample_unbalanced(unbalanced_panel_data_bootstrap):
    np.random.seed(42)
    boot_data = block_boot_sample(unbalanced_panel_data_bootstrap, "id")

    original_units = unbalanced_panel_data_bootstrap["id"].unique()
    assert len(boot_data["id"].unique()) == len(original_units)


@pytest.mark.parametrize(
    "n_units,n_periods",
    [
        (5, 3),
        (10, 2),
        (3, 10),
        (20, 4),
    ],
)
def test_block_boot_sample_various_sizes(n_units, n_periods):
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_periods),
            "time": np.tile(np.arange(n_periods), n_units),
            "y": np.random.randn(n_units * n_periods),
        }
    )

    boot_data = block_boot_sample(data, "id")
    assert len(boot_data) == n_units * n_periods
    assert len(boot_data["id"].unique()) == n_units


def test_make_ecdf_basic():
    y_values = np.array([0, 1, 2, 3, 4])
    cdf_values = np.array([0, 0.25, 0.5, 0.75, 1.0])
    ecdf = _make_ecdf(y_values, cdf_values)

    test_cases = [
        (0, 0),
        (2, 0.5),
        (4, 1.0),
        (-1, 0),
        (5, 1.0),
    ]

    for x, expected in test_cases:
        assert ecdf(x) == expected

    assert np.isclose(ecdf(1.5), 0.375)


@pytest.mark.parametrize(
    "x_array",
    [
        np.array([0, 2, 4]),
        np.linspace(-1, 5, 10),
        np.array([1.5, 2.5, 3.5]),
    ],
)
def test_make_ecdf_vectorized(x_array):
    y_values = np.array([0, 1, 2, 3, 4])
    cdf_values = np.array([0, 0.25, 0.5, 0.75, 1.0])
    ecdf = _make_ecdf(y_values, cdf_values)

    results = ecdf(x_array)
    assert len(results) == len(x_array)
    assert all(0 <= r <= 1 for r in results)


def test_combine_ecdfs_weighted():
    np.random.seed(42)
    y_seq = np.linspace(0, 10, 100)

    ecdf_list = [
        lambda x: np.clip(x / 5, 0, 1),
        lambda x: np.clip(x / 10, 0, 1),
    ]
    weights = np.array([0.7, 0.3])

    combined = _combine_ecdfs(y_seq, ecdf_list, weights)

    assert np.isclose(combined(5), 0.7 * 1.0 + 0.3 * 0.5, rtol=0.1)
    assert combined(0) == 0
    assert combined(10) == 1.0


def test_combine_ecdfs_no_weights():
    y_seq = np.linspace(0, 10, 20)

    ecdf_list = [
        lambda x: np.clip(x / 10, 0, 1),
        lambda x: np.clip(x / 10, 0, 1),
    ]

    combined = _combine_ecdfs(y_seq, ecdf_list)

    for y in [0, 5, 10]:
        assert np.isclose(combined(y), ecdf_list[0](y))


@pytest.mark.parametrize(
    "n_ecdfs,weight_type",
    [
        (2, "equal"),
        (3, "random"),
        (5, "skewed"),
    ],
)
def test_combine_ecdfs_various_configs(n_ecdfs, weight_type):
    np.random.seed(42)
    y_seq = np.linspace(0, 10, 50)
    ecdf_list = [lambda x, i=i: np.clip(x / (5 + i), 0, 1) for i in range(n_ecdfs)]

    if weight_type == "equal":
        weights = None
    elif weight_type == "random":
        weights = np.random.rand(n_ecdfs)
    else:
        weights = np.array([2**i for i in range(n_ecdfs)])

    combined = _combine_ecdfs(y_seq, ecdf_list, weights)

    assert combined(0) == 0
    assert combined(20) == 1.0
    assert 0 <= combined(5) <= 1.0


@pytest.mark.parametrize(
    "original_periods,expected_map",
    [
        (np.array([2010, 2011, 2012]), {1: 2010, 2: 2011, 3: 2012}),
        (np.array([2000, 2005, 2010]), {1: 2000, 2: 2005, 3: 2010}),
        (np.array([1, 2, 3]), {1: 1, 2: 2, 3: 3}),
    ],
)
def test_convert_to_original_time(original_periods, expected_map):
    extra_gt_returns = [
        {"group": i, "time_period": i + 1, "extra_gt_returns": {"value": 0.5}} for i in range(1, len(original_periods))
    ]

    converted = _convert_to_original_time(extra_gt_returns, original_periods)

    for i, item in enumerate(converted):
        assert item["group"] == expected_map[i + 1]
        assert item["time_period"] == expected_map[i + 2]


def test_attgt_pte_aggregations_basic(basic_attgt_data, create_pte_params):
    pte_params = create_pte_params()
    result = attgt_pte_aggregations(basic_attgt_data, pte_params)

    assert all(
        key in result
        for key in [
            "attgt_results",
            "dyn_results",
            "group_results",
            "overall_results",
            "overall_weights",
            "dyn_weights",
            "group_weights",
        ]
    )

    assert len(result["attgt_results"]) == 5
    assert result["dyn_results"] is not None
    assert result["group_results"] is not None
    assert not np.isnan(result["overall_results"])

    assert -1.0 <= result["overall_results"] <= 1.0

    overall_weights = result["overall_weights"]
    assert len(overall_weights) == len(result["attgt_results"])

    post_treatment_weights = overall_weights[overall_weights > 0]
    if len(post_treatment_weights) > 0:
        assert np.isclose(post_treatment_weights.sum(), 1.0, rtol=0.01)

    if result["dyn_results"] is not None:
        dyn_att = result["dyn_results"]["att_e"].values
        assert all(np.isfinite(dyn_att))
        assert all(-10 <= att <= 10 for att in dyn_att)

    if result["group_results"] is not None:
        group_att = result["group_results"]["att_g"].values
        assert all(np.isfinite(group_att))
        for _, group_row in result["group_results"].iterrows():
            g = group_row["group"]
            group_attgt = result["attgt_results"][
                (result["attgt_results"]["group"] == g) & (result["attgt_results"]["time_period"] >= g)
            ]["att"]
            if len(group_attgt) > 0:
                expected_att = group_attgt.mean()
                assert np.isclose(group_row["att_g"], expected_att, rtol=0.01)


@pytest.mark.parametrize(
    "missing_indices",
    [
        [1],
        [1, 3],
        [0, 2, 4],
    ],
)
def test_attgt_pte_aggregations_with_missing(basic_attgt_data, create_pte_params, missing_indices):
    attgt_list = basic_attgt_data.copy()
    for idx in missing_indices:
        attgt_list[idx]["att"] = np.nan

    pte_params = create_pte_params()
    result = attgt_pte_aggregations(attgt_list, pte_params)

    assert len(result["attgt_results"]) == len(attgt_list) - len(missing_indices)
    assert not result["attgt_results"]["att"].isna().any()


def test_overall_weights_e_mask(basic_attgt_data, create_pte_params):
    pte_params = create_pte_params()
    result = attgt_pte_aggregations(basic_attgt_data, pte_params)

    overall_weights = result["overall_weights"]
    attgt_df = pd.DataFrame(basic_attgt_data)
    attgt_df["e"] = attgt_df["time_period"] - attgt_df["group"]

    assert len(overall_weights) == len(basic_attgt_data)
    assert all(overall_weights[attgt_df["e"] < 0] == 0)
    assert any(overall_weights[attgt_df["e"] >= 0] > 0)

    if not np.isnan(result["overall_results"]):
        attgt_df = result["attgt_results"].copy()
        attgt_df["weight"] = overall_weights
        attgt_df["e"] = attgt_df["time_period"] - attgt_df["group"]
        post_treatment = attgt_df[attgt_df["e"] >= 0]

        if len(post_treatment) > 0 and post_treatment["weight"].sum() > 0:
            computed_overall = (post_treatment["att"] * post_treatment["weight"]).sum()
            assert np.isclose(result["overall_results"], computed_overall, rtol=0.01)


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
def test_qtt_pte_aggregations(quantile_test_data, create_pte_params, quantile):
    pte_params = create_pte_params(ret_quantile=quantile)
    result = qtt_pte_aggregations(
        quantile_test_data["attgt_list"], pte_params, quantile_test_data["extra_gt_returns_qtt"]
    )

    assert "attgt_results" in result
    assert len(result["attgt_results"]) == 2
    assert not result["attgt_results"]["att"].isna().any()
    assert result["overall_results"] is not None

    assert np.isfinite(result["overall_results"])
    assert -10 <= result["overall_results"] <= 10

    qtt_values = result["attgt_results"]["att"].values
    assert all(np.isfinite(qtt_values))
    assert all(-10 <= val <= 10 for val in qtt_values)


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
def test_qott_pte_aggregations(quantile_test_data, create_pte_params, quantile):
    pte_params = create_pte_params(ret_quantile=quantile)
    result = qott_pte_aggregations(
        quantile_test_data["attgt_list"], pte_params, quantile_test_data["extra_gt_returns_qott"]
    )

    assert "attgt_results" in result
    assert len(result["attgt_results"]) == 2
    assert not result["attgt_results"]["att"].isna().any()
    assert result["overall_results"] is not None

    assert np.isfinite(result["overall_results"])
    assert -10 <= result["overall_results"] <= 10

    qott_values = result["attgt_results"]["att"].values
    assert all(np.isfinite(qott_values))
    assert all(-10 <= val <= 10 for val in qott_values)


@pytest.mark.parametrize(
    "n_boot,gt_type",
    [
        (5, "att"),
        (10, "att"),
        (5, "qtt"),
        (5, "qott"),
    ],
)
def test_panel_empirical_bootstrap(create_pte_params, n_boot, gt_type):
    np.random.seed(42)

    attgt_list = [
        {"att": 0.1, "group": 2004, "time_period": 2003},
        {"att": 0.2, "group": 2004, "time_period": 2004},
        {"att": 0.15, "group": 2006, "time_period": 2005},
        {"att": 0.12, "group": 2006, "time_period": 2006},
    ]

    extra_gt_returns = []
    if gt_type == "qtt":
        extra_gt_returns = [
            {
                "group": att["group"],
                "time_period": att["time_period"],
                "extra_gt_returns": {
                    "F0": np.random.randn(50),
                    "F1": np.random.randn(50) + 0.3,
                },
            }
            for att in attgt_list
        ]
    elif gt_type == "qott":
        extra_gt_returns = [
            {
                "group": att["group"],
                "time_period": att["time_period"],
                "extra_gt_returns": {"Fte": np.random.randn(50)},
            }
            for att in attgt_list
        ]

    pte_params = create_pte_params(gt_type=gt_type)._replace(biters=n_boot)

    def mock_setup_pte(**kwargs):
        return pte_params

    def mock_subset_fun(data, g, t):
        return data

    def mock_attgt_fun(data):
        return {"att": np.random.normal(0.1, 0.02)}

    def mock_compute_pte(ptep, subset_fun, attgt_fun, **kwargs):
        boot_attgt_list = [
            {"att": att["att"] + np.random.normal(0, 0.01), "group": att["group"], "time_period": att["time_period"]}
            for att in attgt_list
        ]
        return {
            "attgt_list": boot_attgt_list,
            "extra_gt_returns": extra_gt_returns,
        }

    result = panel_empirical_bootstrap(
        attgt_list=attgt_list,
        pte_params=pte_params,
        setup_pte_fun=mock_setup_pte,
        subset_fun=mock_subset_fun,
        attgt_fun=mock_attgt_fun,
        extra_gt_returns=extra_gt_returns,
        compute_pte_fun=mock_compute_pte,
    )

    assert isinstance(result, PteEmpBootResult)
    assert result.attgt_results is not None
    assert "se" in result.attgt_results.columns
    assert result.overall_results is not None
    assert "se" in result.overall_results

    se_values = result.attgt_results["se"].dropna()
    if len(se_values) > 0:
        assert all(se_values >= 0)
        assert all(se_values < 1.0)
        if n_boot >= 10:
            assert all(se_values > 0)

    assert np.isfinite(result.overall_results["att"])
    assert -10 <= result.overall_results["att"] <= 10
    assert result.overall_results["se"] >= 0
    assert result.overall_results["se"] < 1.0

    if gt_type in ["qtt", "qott"]:
        assert np.isfinite(result.overall_results["att"])


def test_panel_empirical_bootstrap_with_warnings(create_pte_params):
    np.random.seed(42)

    attgt_list = [
        {"att": 0.1, "group": 2004, "time_period": 2003},
        {"att": 0.2, "group": 2004, "time_period": 2004},
    ]

    pte_params = create_pte_params()._replace(biters=10)

    def mock_setup_pte(**kwargs):
        return pte_params

    def mock_subset_fun(data, g, t):
        return data

    def mock_attgt_fun(data):
        return {"att": np.random.normal(0.1, 0.02)}

    counter = [0]

    def mock_compute_pte(ptep, subset_fun, attgt_fun, **kwargs):
        counter[0] += 1
        if counter[0] <= 5:
            return {
                "attgt_list": attgt_list,
                "extra_gt_returns": [],
            }
        return {
            "attgt_list": [attgt_list[0]],
            "extra_gt_returns": [],
        }

    with pytest.warns(UserWarning, match="dropping some"):
        result = panel_empirical_bootstrap(
            attgt_list=attgt_list,
            pte_params=pte_params,
            setup_pte_fun=mock_setup_pte,
            subset_fun=mock_subset_fun,
            attgt_fun=mock_attgt_fun,
            extra_gt_returns=[],
            compute_pte_fun=mock_compute_pte,
        )

        assert isinstance(result, PteEmpBootResult)
