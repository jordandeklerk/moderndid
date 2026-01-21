# pylint: disable=redefined-outer-name
"""Tests for group-time average treatment effects."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import MPResult, att_gt, load_mpdta


@pytest.fixture
def mpdta_data():
    df = load_mpdta()
    return df


def test_att_gt_basic_functionality(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ 1",
        est_method="reg",
    )

    assert isinstance(result, MPResult)
    assert hasattr(result, "groups")
    assert hasattr(result, "times")
    assert hasattr(result, "att_gt")
    assert hasattr(result, "se_gt")
    assert len(result.groups) == len(result.times)
    assert len(result.att_gt) == len(result.groups)
    assert len(result.se_gt) == len(result.groups)


def test_att_gt_with_covariates(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        control_group="nevertreated",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert len(result.groups) == len(result.times)
    assert np.all(~np.isnan(result.att_gt))


def test_att_gt_notyettreated_control(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["control_group"] == "notyettreated"


def test_att_gt_with_anticipation(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        anticipation=1,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["anticipation_periods"] == 1


def test_att_gt_bootstrap_inference(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=True,
        biters=20,
        cband=True,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["bootstrap"] is True
    assert result.estimation_params["uniform_bands"] is True
    assert result.critical_value > 0
    assert np.all(result.se_gt > 0)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_estimation_methods(est_method, mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method=est_method,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["estimation_method"] == est_method


def test_att_gt_universal_base_period(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        base_period="universal",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["base_period"] == "universal"


def test_att_gt_with_weights(mpdta_data):
    mpdta_data = mpdta_data.with_columns(pl.Series("weights", np.random.uniform(0.5, 1.5, len(mpdta_data))))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="weights",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.weights_ind is not None


def test_att_gt_repeated_cross_section(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        panel=False,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["panel"] is False


def test_att_gt_unbalanced_panel(mpdta_data):
    mpdta_data = mpdta_data.filter(~((pl.col("countyreal") < 1010) & (pl.col("year") == 2005)))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
        bstrap=False,
    )

    assert isinstance(result, MPResult)


def test_att_gt_clustering(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))
    mpdta_data = mpdta_data.with_columns((pl.col("countyreal") // 10).alias("cluster"))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        clustervars=["cluster"],
        bstrap=True,
        biters=10,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["clustervars"] == ["cluster"]


def test_att_gt_wald_pretest(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    pre_treatment_periods = np.any(result.groups > result.times)
    if pre_treatment_periods:
        assert hasattr(result, "wald_stat")
        assert hasattr(result, "wald_pvalue")


def test_att_gt_invalid_control_group(mpdta_data):
    with pytest.raises(ValueError):
        att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            control_group="invalid",
        )


def test_att_gt_missing_column(mpdta_data):
    with pytest.raises(ValueError, match="yname"):
        att_gt(
            data=mpdta_data,
            yname="missing_column",
            tname="year",
            gname="first.treat",
            idname="countyreal",
        )


def test_att_gt_all_treated_notyettreated(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert len(result.att_gt) > 0


def test_att_gt_summary_output(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=False,
    )

    summary_str = str(result)
    assert "Group-Time Average Treatment Effects" in summary_str
    assert "ATT(g,t)" in summary_str
    assert "Std. Error" in summary_str


def test_att_gt_influence_functions(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=False,
    )

    assert hasattr(result, "influence_func")
    assert isinstance(result.influence_func, np.ndarray)
    assert result.influence_func.shape[0] == result.n_units
    assert result.influence_func.shape[1] == len(result.att_gt)


def test_att_gt_variance_matrix(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=False,
    )

    assert hasattr(result, "vcov_analytical")
    assert isinstance(result.vcov_analytical, np.ndarray)
    n_groups_times = len(result.att_gt)
    assert result.vcov_analytical.shape == (n_groups_times, n_groups_times)
    assert np.allclose(result.vcov_analytical, result.vcov_analytical.T)


def test_att_gt_custom_alpha(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        alp=0.10,
        bstrap=False,
    )

    assert result.alpha == 0.10
    assert result.critical_value < 1.96


def test_att_gt_print_details(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=False,
    )

    assert isinstance(result, MPResult)


def test_att_gt_bootstrap_reproducibility(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties)).sort(["countyreal", "year"])

    result1 = att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=True,
        biters=50,
        random_state=42,
    )

    result2 = att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=True,
        biters=50,
        random_state=42,
    )

    np.testing.assert_array_equal(result1.se_gt, result2.se_gt)
    assert result1.critical_value == result2.critical_value
    assert result1.estimation_params.get("random_state") == 42
