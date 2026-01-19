"""Tests for DDD data generation functions."""

import pytest

from moderndid import gen_dgp_2periods, gen_dgp_mult_periods, generate_simple_ddd_data

from ..helpers import importorskip

np = importorskip("numpy")
pd = importorskip("pandas")


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_returns_expected_keys(dgp_type):
    result = gen_dgp_2periods(n=500, dgp_type=dgp_type, random_state=42)

    assert "data" in result
    assert "true_att" in result
    assert "oracle_att" in result
    assert "efficiency_bound" in result


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_data_structure(dgp_type):
    result = gen_dgp_2periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 1000
    expected_cols = ["id", "state", "partition", "time", "y", "cov1", "cov2", "cov3", "cov4", "cluster"]
    assert list(data.columns) == expected_cols


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_time_periods(dgp_type):
    result = gen_dgp_2periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert set(data["time"].unique()) == {1, 2}
    assert len(data[data["time"] == 1]) == 500
    assert len(data[data["time"] == 2]) == 500


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_binary_variables(dgp_type):
    result = gen_dgp_2periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert set(data["state"].unique()).issubset({0, 1})
    assert set(data["partition"].unique()).issubset({0, 1})


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_true_att_is_zero(dgp_type):
    result = gen_dgp_2periods(n=500, dgp_type=dgp_type, random_state=42)

    assert result["true_att"] == 0.0


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_2period_oracle_att_near_zero(dgp_type):
    result = gen_dgp_2periods(n=5000, dgp_type=dgp_type, random_state=42)

    assert np.abs(result["oracle_att"]) < 1.0


def test_2period_reproducibility():
    result1 = gen_dgp_2periods(n=500, dgp_type=1, random_state=42)
    result2 = gen_dgp_2periods(n=500, dgp_type=1, random_state=42)

    pd.testing.assert_frame_equal(result1["data"], result2["data"])
    assert result1["oracle_att"] == result2["oracle_att"]


def test_2period_invalid_dgp_type():
    with pytest.raises(ValueError, match="dgp_type must be 1, 2, 3, or 4"):
        gen_dgp_2periods(n=500, dgp_type=5, random_state=42)


@pytest.mark.parametrize("dgp_type,expected_bound", [(1, 32.82), (2, 32.52), (3, 32.82), (4, 32.52)])
def test_2period_efficiency_bounds(dgp_type, expected_bound):
    result = gen_dgp_2periods(n=500, dgp_type=dgp_type, random_state=42)

    assert result["efficiency_bound"] == expected_bound


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_returns_expected_keys(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)

    assert "data" in result
    assert "data_wide" in result
    assert "es_0_oracle" in result
    assert "prob_g2_p1" in result
    assert "prob_g3_p1" in result


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_long_data_structure(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 1500
    expected_cols = ["id", "group", "partition", "time", "y", "cov1", "cov2", "cov3", "cov4", "cluster"]
    assert list(data.columns) == expected_cols


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_wide_data_structure(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data_wide = result["data_wide"]

    assert isinstance(data_wide, pd.DataFrame)
    assert len(data_wide) == 500
    expected_cols = ["id", "group", "partition", "y_t1", "y_t2", "y_t3", "cov1", "cov2", "cov3", "cov4", "cluster"]
    assert list(data_wide.columns) == expected_cols


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_time_periods(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert set(data["time"].unique()) == {1, 2, 3}
    for t in [1, 2, 3]:
        assert len(data[data["time"] == t]) == 500


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_cohort_values(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert set(data["group"].unique()).issubset({0, 2, 3})


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_partition_binary(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)
    data = result["data"]

    assert set(data["partition"].unique()).issubset({0, 1})


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_probabilities_valid(dgp_type):
    result = gen_dgp_mult_periods(n=500, dgp_type=dgp_type, random_state=42)

    assert 0 < result["prob_g2_p1"] < 1
    assert 0 < result["prob_g3_p1"] < 1
    assert np.isclose(result["prob_g2_p1"] + result["prob_g3_p1"], 1.0)


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_multiperiod_oracle_es_finite(dgp_type):
    result = gen_dgp_mult_periods(n=1000, dgp_type=dgp_type, random_state=42)

    assert np.isfinite(result["es_0_oracle"])
    assert result["es_0_oracle"] > 0


def test_multiperiod_reproducibility():
    result1 = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    result2 = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)

    pd.testing.assert_frame_equal(result1["data"], result2["data"])
    pd.testing.assert_frame_equal(result1["data_wide"], result2["data_wide"])
    assert result1["es_0_oracle"] == result2["es_0_oracle"]


def test_multiperiod_invalid_dgp_type():
    with pytest.raises(ValueError, match="dgp_type must be 1, 2, 3, or 4"):
        gen_dgp_mult_periods(n=500, dgp_type=5, random_state=42)


def test_multiperiod_long_wide_consistency():
    result = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    data = result["data"]
    data_wide = result["data_wide"]

    for unit_id in [1, 100, 500]:
        unit_long = data[data["id"] == unit_id]
        unit_wide = data_wide[data_wide["id"] == unit_id].iloc[0]

        assert unit_long[unit_long["time"] == 1]["y"].values[0] == unit_wide["y_t1"]
        assert unit_long[unit_long["time"] == 2]["y"].values[0] == unit_wide["y_t2"]
        assert unit_long[unit_long["time"] == 3]["y"].values[0] == unit_wide["y_t3"]


@pytest.mark.parametrize("n", [100, 500, 1000])
def test_simple_ddd_data_size(n):
    data = generate_simple_ddd_data(n=n, att=5.0, random_state=42)

    assert len(data) == 2 * n


def test_simple_ddd_data_structure():
    data = generate_simple_ddd_data(n=500, att=5.0, random_state=42)

    expected_cols = ["id", "state", "partition", "time", "y", "x1", "x2"]
    assert list(data.columns) == expected_cols


def test_simple_ddd_time_periods():
    data = generate_simple_ddd_data(n=500, att=5.0, random_state=42)

    assert set(data["time"].unique()) == {1, 2}


def test_simple_ddd_binary_variables():
    data = generate_simple_ddd_data(n=500, att=5.0, random_state=42)

    assert set(data["state"].unique()).issubset({0, 1})
    assert set(data["partition"].unique()).issubset({0, 1})


def test_simple_ddd_reproducibility():
    data1 = generate_simple_ddd_data(n=500, att=5.0, random_state=42)
    data2 = generate_simple_ddd_data(n=500, att=5.0, random_state=42)

    pd.testing.assert_frame_equal(data1, data2)
