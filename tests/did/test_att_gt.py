"""Tests for group-time average treatment effects."""

import numpy as np
import pytest

from pydid import MPResult, att_gt, load_mpdta


def test_att_gt_basic_functionality():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
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


def test_att_gt_with_covariates():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        xformla="~ lpop",
        control_group="nevertreated",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert len(result.groups) == len(result.times)
    assert np.all(~np.isnan(result.att_gt))


def test_att_gt_notyettreated_control():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        control_group="notyettreated",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["control_group"] == "notyettreated"


def test_att_gt_with_anticipation():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        anticipation=1,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["anticipation_periods"] == 1


def test_att_gt_bootstrap_inference():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)
    unique_counties = df["countyreal"].unique()[:100]
    df = df[df["countyreal"].isin(unique_counties)]

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        bstrap=True,
        biters=100,
        cband=True,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["bootstrap"] is True
    assert result.estimation_params["uniform_bands"] is True
    assert result.critical_value > 0
    assert np.all(result.se_gt > 0)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_estimation_methods(est_method):
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        est_method=est_method,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["estimation_method"] == est_method


def test_att_gt_universal_base_period():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        base_period="universal",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["base_period"] == "universal"


def test_att_gt_with_weights():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)
    df["weights"] = np.random.uniform(0.5, 1.5, len(df))

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        weightsname="weights",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.weights_ind is not None


def test_att_gt_repeated_cross_section():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        panel=False,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["panel"] is False


def test_att_gt_unbalanced_panel():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)
    df = df[~((df["countyreal"] < 1010) & (df["year"] == 2005))]

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        allow_unbalanced_panel=True,
        bstrap=False,
    )

    assert isinstance(result, MPResult)


def test_att_gt_clustering():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)
    unique_counties = df["countyreal"].unique()[:100]
    df = df[df["countyreal"].isin(unique_counties)]
    df["cluster"] = df["countyreal"] // 10

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        clustervars=["cluster"],
        bstrap=True,
        biters=50,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["clustervars"] == ["cluster"]


def test_att_gt_wald_pretest():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    pre_treatment_periods = np.any(result.groups > result.times)
    if pre_treatment_periods:
        assert hasattr(result, "wald_stat")
        assert hasattr(result, "wald_pvalue")


def test_att_gt_invalid_control_group():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    with pytest.raises(ValueError):
        att_gt(
            data=df,
            yname="lemp",
            tname="year",
            gname="first_treat",
            idname="countyreal",
            control_group="invalid",
        )


def test_att_gt_missing_column():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    with pytest.raises(ValueError, match="yname"):
        att_gt(
            data=df,
            yname="missing_column",
            tname="year",
            gname="first_treat",
            idname="countyreal",
        )


def test_att_gt_all_treated_notyettreated():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, 2008)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        control_group="notyettreated",
        bstrap=False,
    )

    assert isinstance(result, MPResult)
    assert len(result.att_gt) > 0


def test_att_gt_summary_output():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        bstrap=False,
    )

    summary_str = str(result)
    assert "Group-Time Average Treatment Effects" in summary_str
    assert "ATT(g,t)" in summary_str
    assert "Std. Error" in summary_str


def test_att_gt_influence_functions():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        bstrap=False,
    )

    assert hasattr(result, "influence_func")
    assert isinstance(result.influence_func, np.ndarray)
    assert result.influence_func.shape[0] == result.n_units
    assert result.influence_func.shape[1] == len(result.att_gt)


def test_att_gt_variance_matrix():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        bstrap=False,
    )

    assert hasattr(result, "vcov_analytical")
    assert isinstance(result.vcov_analytical, np.ndarray)
    n_groups_times = len(result.att_gt)
    assert result.vcov_analytical.shape == (n_groups_times, n_groups_times)
    assert np.allclose(result.vcov_analytical, result.vcov_analytical.T)


def test_att_gt_custom_alpha():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        alp=0.10,
        bstrap=False,
    )

    assert result.alpha == 0.10
    assert result.critical_value < 1.96


def test_att_gt_print_details():
    df = load_mpdta()
    df["first_treat"] = df["first.treat"].replace(0, np.inf)

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first_treat",
        idname="countyreal",
        print_details=True,
        bstrap=False,
    )

    assert isinstance(result, MPResult)
