"""Tests for the data generating processes."""

import numpy as np
import pytest

from .helpers import importorskip

pd = importorskip("pandas")

from .dgp import BaseDGP, SantAnnaZhaoDRDiD


def test_abstract_methods():
    with pytest.raises(TypeError):
        BaseDGP()  # pylint: disable=abstract-class-instantiated


def test_santannazhaodrdid_init():
    dgp = SantAnnaZhaoDRDiD(
        n_units=500,
        treatment_fraction=0.7,
        common_support_strength=0.9,
        random_seed=4321,
    )

    assert dgp.n_units == 500
    assert dgp.treatment_fraction == 0.7
    assert dgp.common_support_strength == 0.9
    assert dgp.random_seed == 4321


def test_santannazhaodrdid_generate_data():
    dgp = SantAnnaZhaoDRDiD(n_units=200, random_seed=1234)
    data = dgp.generate_data(att=0.5)

    assert "df" in data
    assert "true_att" in data
    assert "raw_covariates" in data
    assert "transformed_covariates" in data
    assert "propensity_scores" in data
    assert "potential_outcomes_pre" in data
    assert "potential_outcomes_post" in data

    df = data["df"]
    assert len(df) == 200
    assert set(df.columns) == {"id", "post", "y", "d", "x1", "x2", "x3", "x4"}

    assert data["true_att"] == 0.5
    assert data["raw_covariates"].shape == (200, 4)
    assert data["transformed_covariates"].shape == (200, 4)
    assert data["propensity_scores"].shape == (200,)
    assert all(0 <= p <= 1 for p in data["propensity_scores"])

    assert all(d in [0, 1] for d in df["d"])
    assert all(p in [0, 1] for p in df["post"])

    assert "y0" in data["potential_outcomes_pre"]
    assert "y0" in data["potential_outcomes_post"]
    assert "y1" in data["potential_outcomes_post"]
    assert len(data["potential_outcomes_pre"]["y0"]) == 200
    assert len(data["potential_outcomes_post"]["y0"]) == 200
    assert len(data["potential_outcomes_post"]["y1"]) == 200


def test_santannazhaodrdid_set_seed():
    dgp1 = SantAnnaZhaoDRDiD(n_units=100, random_seed=5678)
    data1 = dgp1.generate_data(att=0.0)

    dgp2 = SantAnnaZhaoDRDiD(n_units=100, random_seed=5678)
    data2 = dgp2.generate_data(att=0.0)

    pd.testing.assert_frame_equal(data1["df"], data2["df"])

    np.testing.assert_array_equal(data1["raw_covariates"], data2["raw_covariates"])
    np.testing.assert_array_equal(data1["transformed_covariates"], data2["transformed_covariates"])
    np.testing.assert_array_equal(data1["propensity_scores"], data2["propensity_scores"])

    dgp3 = SantAnnaZhaoDRDiD(n_units=100, random_seed=9999)
    data3 = dgp3.generate_data(att=0.0)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(data1["raw_covariates"], data3["raw_covariates"])


def test_santannazhaodrdid_att_parameter():
    dgp = SantAnnaZhaoDRDiD(n_units=500, random_seed=1234)

    data_att0 = dgp.generate_data(att=0.0)
    dgp.set_seed(1234)
    data_att1 = dgp.generate_data(att=1.0)

    diff_y1 = data_att1["potential_outcomes_post"]["y1"] - data_att0["potential_outcomes_post"]["y1"]

    treated_units = data_att0["df"]["d"] == 1
    assert np.allclose(diff_y1[treated_units], 1.0)

    np.testing.assert_array_equal(
        data_att0["potential_outcomes_post"]["y0"], data_att1["potential_outcomes_post"]["y0"]
    )


@pytest.mark.parametrize(
    "args, kwargs, match_pattern",
    [
        ((1, 2), {}, "Unexpected positional arguments"),
        ((), {"unknown_param": 5}, "Unexpected keyword arguments"),
    ],
)
def test_santannazhaodrdid_value_errors(args, kwargs, match_pattern):
    dgp = SantAnnaZhaoDRDiD()
    with pytest.raises(ValueError, match=match_pattern):
        dgp.generate_data(*args, **kwargs)
