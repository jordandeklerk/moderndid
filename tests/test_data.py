"""Test the datasets."""

import numpy as np
import pytest

from moderndid.core.data import (
    gen_did_scalable,
    load_cai2016,
    load_ehec,
    load_engel,
    load_favara_imbs,
    load_mpdta,
    load_nsw,
    simulate_cont_did_data,
)
from tests.helpers import importorskip

pl = importorskip("polars")


def test_load_nsw():
    nsw_data = load_nsw()

    assert isinstance(nsw_data, pl.DataFrame)
    assert nsw_data.shape == (32834, 16)

    expected_columns = [
        "id",
        "year",
        "experimental",
        "re",
        "age",
        "educ",
        "black",
        "married",
        "nodegree",
        "hisp",
        "re74",
    ]
    for col in expected_columns:
        assert col in nsw_data.columns

    assert set(nsw_data["year"].unique().to_list()) == {1975, 1978}
    assert set(nsw_data["experimental"].unique().to_list()) == {0, 1}

    assert nsw_data["experimental"].sum() > 0
    assert (nsw_data["experimental"] == 0).sum() > 0


def test_load_nsw_data_integrity():
    nsw_data = load_nsw()

    key_columns = ["id", "year", "experimental"]
    for col in key_columns:
        assert nsw_data[col].is_not_null().all()

    assert nsw_data["id"].dtype.is_integer()
    assert nsw_data["year"].dtype.is_integer()
    assert nsw_data["experimental"].dtype.is_integer()
    assert nsw_data["re"].dtype.is_float()

    id_counts = nsw_data.group_by("id").len()
    assert (id_counts["len"] == 2).all()


def test_import_from_package():
    import moderndid

    assert hasattr(moderndid, "data")
    assert hasattr(moderndid.data, "load_nsw")

    nsw_data = moderndid.data.load_nsw()

    assert isinstance(nsw_data, pl.DataFrame)
    assert nsw_data.shape[0] > 0


def test_load_nsw_returns_copy():
    nsw_data1 = load_nsw()
    nsw_data2 = load_nsw()

    nsw_data1 = nsw_data1.with_columns(pl.lit(1).alias("test_column"))

    assert "test_column" not in nsw_data2.columns


def test_load_mpdta():
    mpdta_data = load_mpdta()

    assert isinstance(mpdta_data, pl.DataFrame)
    assert mpdta_data.shape == (2500, 6)

    expected_columns = [
        "year",
        "countyreal",
        "lpop",
        "lemp",
        "first.treat",
        "treat",
    ]
    assert list(mpdta_data.columns) == expected_columns

    assert set(mpdta_data["year"].unique().to_list()) == {2003, 2004, 2005, 2006, 2007}
    assert set(mpdta_data["treat"].unique().to_list()) == {0, 1}
    assert set(mpdta_data["first.treat"].unique().to_list()) == {0, 2004, 2006, 2007}


def test_load_mpdta_data_integrity():
    mpdta_data = load_mpdta()

    key_columns = ["year", "countyreal", "lpop", "lemp", "first.treat", "treat"]
    for col in key_columns:
        assert mpdta_data[col].is_not_null().all()

    assert mpdta_data["year"].dtype.is_integer()
    assert mpdta_data["countyreal"].dtype.is_integer()
    assert mpdta_data["lpop"].dtype.is_float()
    assert mpdta_data["lemp"].dtype.is_float()
    assert mpdta_data["first.treat"].dtype.is_integer()
    assert mpdta_data["treat"].dtype.is_integer()

    county_counts = mpdta_data.group_by("countyreal").len()
    assert (county_counts["len"] == 5).all()

    for county in mpdta_data["countyreal"].unique().to_list():
        county_data = mpdta_data.filter(pl.col("countyreal") == county)
        assert county_data["first.treat"].n_unique() == 1


def test_import_mpdta_from_package():
    import moderndid

    assert hasattr(moderndid, "load_mpdta")
    assert hasattr(moderndid.data, "load_mpdta")

    mpdta_data = moderndid.load_mpdta()

    assert isinstance(mpdta_data, pl.DataFrame)
    assert mpdta_data.shape[0] == 2500


def test_load_mpdta_returns_copy():
    mpdta_data1 = load_mpdta()
    mpdta_data2 = load_mpdta()

    mpdta_data1 = mpdta_data1.with_columns(pl.lit(1).alias("test_column"))

    assert "test_column" not in mpdta_data2.columns


def test_load_ehec():
    ehec_data = load_ehec()

    assert isinstance(ehec_data, pl.DataFrame)
    assert ehec_data.shape == (552, 5)

    expected_columns = ["stfips", "year", "dins", "yexp2", "W"]
    assert list(ehec_data.columns) == expected_columns

    assert set(ehec_data["year"].unique().to_list()) == set(range(2008, 2020))

    assert ehec_data["yexp2"].is_not_null().sum() == 360
    assert ehec_data["yexp2"].is_null().sum() == 192

    assert set(ehec_data["yexp2"].drop_nulls().unique().to_list()) == {2014.0, 2015.0, 2016.0, 2017.0, 2019.0}


def test_load_ehec_data_integrity():
    ehec_data = load_ehec()

    key_columns = ["stfips", "year", "dins", "W"]
    for col in key_columns:
        assert ehec_data[col].is_not_null().all()

    assert ehec_data["stfips"].dtype.is_integer()
    assert ehec_data["year"].dtype.is_integer()
    assert ehec_data["dins"].dtype.is_float()
    assert ehec_data["yexp2"].dtype.is_float()
    assert ehec_data["W"].dtype.is_float()

    state_counts = ehec_data.group_by("stfips").len()
    assert (state_counts["len"] == 12).all()

    assert ehec_data["dins"].min() > 0.4
    assert ehec_data["dins"].max() < 1.0

    assert ehec_data["W"].min() > 0


def test_import_ehec_from_package():
    import moderndid

    assert hasattr(moderndid, "load_ehec")
    assert hasattr(moderndid.data, "load_ehec")

    ehec_data = moderndid.load_ehec()

    assert isinstance(ehec_data, pl.DataFrame)
    assert ehec_data.shape[0] == 552


def test_load_ehec_returns_copy():
    ehec_data1 = load_ehec()
    ehec_data2 = load_ehec()

    ehec_data1 = ehec_data1.with_columns(pl.lit(1).alias("test_column"))

    assert "test_column" not in ehec_data2.columns


@pytest.mark.parametrize(
    "loader, min_rows, min_cols, required_cols",
    [
        (
            load_engel,
            1,
            10,
            {"food", "catering", "alcohol", "fuel", "motor", "fares", "leisure", "logexp", "logwages", "nkids"},
        ),
        (load_favara_imbs, 1, 7, {"year", "county", "state_n", "Dl_vloans_b", "inter_bra"}),
        (load_cai2016, 1, 10, {"hhno", "year", "treatment", "sector", "checksaving_ratio"}),
    ],
)
def test_loader_shape_and_columns(loader, min_rows, min_cols, required_cols):
    df = loader()
    assert df.shape[0] >= min_rows
    assert df.shape[1] >= min_cols
    assert required_cols.issubset(set(df.columns))


@pytest.mark.parametrize(
    "col_name",
    ["food", "logexp"],
)
def test_load_engel_no_nulls_in_key_columns(col_name):
    df = load_engel()
    assert df[col_name].null_count() == 0


@pytest.mark.parametrize(
    "loader, col, min_val, max_val",
    [
        (load_favara_imbs, "year", 1994, 2005),
        (load_cai2016, "year", 2000, 2008),
    ],
)
def test_loader_year_range(loader, col, min_val, max_val):
    df = loader()
    years = df[col].unique().sort().to_list()
    assert min(years) >= min_val
    assert max(years) <= max_val


def test_load_cai2016_treatment_binary():
    df = load_cai2016()
    unique_vals = set(df["treatment"].unique().to_list())
    assert unique_vals.issubset({0, 1})


@pytest.mark.parametrize(
    "key",
    ["data", "data_wide", "att_config", "cohort_values"],
)
def test_gen_did_scalable_panel_basic_keys(key):
    result = gen_did_scalable(n=100, random_state=42)
    assert key in result


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("n_periods", 10),
        ("n_covariates", 20),
    ],
)
def test_gen_did_scalable_panel_basic_defaults(attr, expected):
    result = gen_did_scalable(n=100, random_state=42)
    assert result[attr] == expected


@pytest.mark.parametrize(
    "col",
    ["id", "group", "time", "y", "cluster"],
)
def test_gen_did_scalable_panel_data_columns(col):
    result = gen_did_scalable(n=50, n_periods=5, n_cohorts=3, random_state=42)
    assert col in result["data"].columns


def test_gen_did_scalable_panel_data_shape():
    result = gen_did_scalable(n=50, n_periods=5, n_cohorts=3, random_state=42)
    assert result["data"].shape[0] == 50 * 5


def test_gen_did_scalable_panel_wide_data():
    result = gen_did_scalable(n=50, n_periods=5, n_cohorts=3, random_state=42)
    wide = result["data_wide"]
    assert wide is not None
    assert wide.shape[0] == 50
    assert "y_t1" in wide.columns
    assert "y_t5" in wide.columns


def test_gen_did_scalable_panel_no_wide_when_many_periods():
    result = gen_did_scalable(n=50, n_periods=25, n_cohorts=3, random_state=42)
    assert result["data_wide"] is None


def test_gen_did_scalable_cohort_values():
    result = gen_did_scalable(n=50, n_periods=5, n_cohorts=3, random_state=42)
    cohorts = result["cohort_values"]
    assert 0 in cohorts
    assert len(cohorts) == 4


def test_gen_did_scalable_att_config():
    result = gen_did_scalable(n=50, n_periods=5, n_cohorts=3, att_base=5.0, random_state=42)
    att = result["att_config"]
    for g, val in att.items():
        assert val == 5.0 * g


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_gen_did_scalable_all_dgp_types(dgp_type):
    result = gen_did_scalable(n=50, dgp_type=dgp_type, n_periods=5, n_cohorts=3, random_state=42)
    assert result["data"].shape[0] == 250


def test_gen_did_scalable_repeated_cross_section():
    result = gen_did_scalable(n=50, n_periods=5, n_cohorts=3, panel=False, random_state=42)
    df = result["data"]
    assert df.shape[0] == 50 * 5
    assert result["data_wide"] is None
    ids_per_period = df.group_by("time").agg(pl.col("id").n_unique())
    assert (ids_per_period["id"] == 50).all()


def test_gen_did_scalable_rc_unique_ids_across_periods():
    result = gen_did_scalable(n=30, n_periods=3, n_cohorts=2, panel=False, random_state=42)
    df = result["data"]
    for t in range(1, 4):
        period_ids = set(df.filter(pl.col("time") == t)["id"].to_list())
        for t2 in range(t + 1, 4):
            other_ids = set(df.filter(pl.col("time") == t2)["id"].to_list())
            assert len(period_ids & other_ids) == 0


def test_gen_did_scalable_custom_covariates():
    result = gen_did_scalable(n=50, n_covariates=8, n_periods=5, n_cohorts=3, random_state=42)
    df = result["data"]
    cov_cols = [c for c in df.columns if c.startswith("cov")]
    assert len(cov_cols) == 8


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dgp_type": 5}, "dgp_type must be"),
        ({"n_periods": 1}, "n_periods must be"),
        ({"n_cohorts": 0}, "n_cohorts must be >= 1"),
        ({"n_cohorts": 10, "n_periods": 5}, "n_cohorts must be < n_periods"),
        ({"n_covariates": 2}, "n_covariates must be >= 4"),
    ],
)
def test_gen_did_scalable_validation_errors(kwargs, match):
    base = {"n": 50, "random_state": 42}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        gen_did_scalable(**base)


@pytest.mark.parametrize(
    "col",
    ["id", "time_period", "Y", "G", "D"],
)
def test_simulate_cont_did_data_default_columns(col):
    df = simulate_cont_did_data()
    assert isinstance(df, pl.DataFrame)
    assert col in df.columns


def test_simulate_cont_did_data_shape():
    df = simulate_cont_did_data(n=100, num_time_periods=3)
    assert df.shape[0] == 100 * 3


def test_simulate_cont_did_data_sorted():
    df = simulate_cont_did_data(n=50, seed=123)
    ids = df["id"].to_list()
    times = df["time_period"].to_list()
    for i in range(len(ids) - 1):
        assert (ids[i], times[i]) <= (ids[i + 1], times[i + 1])


def test_simulate_cont_did_data_never_treated_zero_dose():
    df = simulate_cont_did_data(n=200, seed=99)
    never_treated = df.filter(pl.col("G") == 0)
    assert (never_treated["D"] == 0.0).all()


def test_simulate_cont_did_data_custom_params():
    df = simulate_cont_did_data(
        n=100,
        num_time_periods=5,
        num_groups=5,
        dose_linear_effect=1.0,
        dose_quadratic_effect=0.2,
        seed=7,
    )
    assert df.shape[0] == 100 * 5
    groups = set(df["G"].unique().to_list())
    assert 0 in groups


def test_simulate_cont_did_data_custom_probabilities():
    df = simulate_cont_did_data(
        n=300,
        num_time_periods=3,
        p_untreated=0.5,
        p_group=[0.25, 0.25],
        seed=10,
    )
    assert df.shape[0] == 300 * 3


def test_simulate_cont_did_data_reproducible():
    df1 = simulate_cont_did_data(n=50, seed=42)
    df2 = simulate_cont_did_data(n=50, seed=42)
    np.testing.assert_array_equal(df1["Y"].to_numpy(), df2["Y"].to_numpy())
