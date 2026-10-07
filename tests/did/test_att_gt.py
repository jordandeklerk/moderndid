"""Tests for group-time average treatment effects."""

import re
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import MPResult, aggte, att_gt
from moderndid.core.preprocess import preprocess_did
from moderndid.did.compute_att_gt import ATTgtResult, ComputeATTgtResult


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
        boot=False,
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
        boot=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["control_group"] == "notyettreated"


@pytest.mark.filterwarnings("ignore:Dropped.*units that were already treated:UserWarning")
def test_att_gt_with_anticipation(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        anticipation=1,
        boot=False,
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
        boot=True,
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
        boot=False,
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
        boot=False,
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
        boot=False,
    )

    assert isinstance(result, MPResult)
    assert result.weights_ind is not None


def test_att_gt_weights_any_column_name(mpdta_pop_weighted):
    spec = dict(yname="lemp", tname="year", idname="countyreal", gname="first.treat", boot=False, cband=False)
    named = att_gt(data=mpdta_pop_weighted, weightsname="pop", **spec)
    renamed = att_gt(data=mpdta_pop_weighted.rename({"pop": "weights"}), weightsname="weights", **spec)

    assert named.weights_ind is not None
    np.testing.assert_array_equal(np.asarray(named.weights_ind), np.asarray(renamed.weights_ind))
    for agg_type in ("simple", "group", "dynamic", "calendar"):
        by_name = aggte(named, type=agg_type)
        by_rename = aggte(renamed, type=agg_type)
        assert by_name.overall_att == by_rename.overall_att
        assert by_name.overall_se == by_rename.overall_se


def test_att_gt_unbalanced_weights_ind_averages_each_unit(mpdta_unbalanced_varying_weights):
    spec = dict(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="w",
        allow_unbalanced_panel=True,
    )
    result = att_gt(data=mpdta_unbalanced_varying_weights, boot=False, cband=False, **spec)
    units = preprocess_did(mpdta_unbalanced_varying_weights, **spec).time_invariant_data["countyreal"]
    rows = mpdta_unbalanced_varying_weights.with_columns(pl.col("w") / pl.col("w").mean())
    unit_means = rows.group_by("countyreal").agg(pl.col("w").mean())
    expected = units.replace_strict(unit_means["countyreal"], unit_means["w"]).to_numpy()

    np.testing.assert_allclose(np.asarray(result.weights_ind), expected, rtol=1e-12)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("est_method", ["reg", "dr"])
def test_att_gt_time_varying_weights_use_earlier_period(
    mpdta_varying_weights, mpdta_weights_by_year, base_period, est_method
):
    spec = dict(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        est_method=est_method,
        base_period=base_period,
        boot=False,
        cband=False,
    )
    result = att_gt(data=mpdta_varying_weights, weightsname="w", **spec)
    fixed = {year: att_gt(data=data, weightsname="w_fixed", **spec) for year, data in mpdta_weights_by_year.items()}

    for i, (group, time) in enumerate(zip(result.groups, result.times)):
        base = group - 1 if time >= group or base_period == "universal" else time - 1
        earlier = fixed[int(min(base, time))]
        assert (earlier.groups[i], earlier.times[i]) == (group, time)
        np.testing.assert_allclose(result.att_gt[i], earlier.att_gt[i], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.se_gt[i], earlier.se_gt[i], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("weightsname", "panel"), [("pop", True), ("w", False)])
def test_att_gt_weights_message_needs_panel_weights_that_vary(mpdta_varying_weights, weightsname, panel, recwarn):
    att_gt(
        data=mpdta_varying_weights.with_columns(pl.col("lpop").exp().alias("pop")),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname=weightsname,
        panel=panel,
        boot=False,
        cband=False,
    )

    assert not [w for w in recwarn if "Time-varying weights" in str(w.message)]


@pytest.mark.parametrize("data_fixture", ["mpdta_varying_weights", "mpdta_unbalanced_varying_weights"])
def test_att_gt_time_varying_weights_warn(request, data_fixture):
    with pytest.warns(UserWarning, match="Time-varying weights detected"):
        att_gt(
            data=request.getfixturevalue(data_fixture),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            weightsname="w",
            allow_unbalanced_panel=True,
            boot=False,
            cband=False,
        )


@pytest.mark.parametrize(("column", "yname", "xformla"), [("lemp", "weights", "~ lpop"), ("lpop", "lemp", "~ weights")])
def test_att_gt_user_column_named_weights(mpdta_data, column, yname, xformla):
    spec = dict(tname="year", idname="countyreal", gname="first.treat", boot=False, cband=False)
    named = att_gt(data=mpdta_data, yname="lemp", xformla="~ lpop", **spec)
    renamed = att_gt(data=mpdta_data.rename({column: "weights"}), yname=yname, xformla=xformla, **spec)

    np.testing.assert_array_equal(renamed.att_gt, named.att_gt)
    np.testing.assert_array_equal(renamed.se_gt, named.se_gt)


@pytest.mark.parametrize("reserved", [".w", ".rowid"])
@pytest.mark.parametrize("panel", [True, False])
def test_att_gt_rejects_reserved_outcome_name(mpdta_data, reserved, panel):
    with pytest.raises(ValueError, match=re.escape(f"yname names the column '{reserved}'")):
        att_gt(
            data=mpdta_data.rename({"lemp": reserved}),
            yname=reserved,
            tname="year",
            idname="countyreal",
            gname="first.treat",
            panel=panel,
        )


def test_att_gt_rejects_reserved_covariate_name(mpdta_data):
    with pytest.raises(ValueError, match=re.escape("xformla names the column '.w'")):
        att_gt(
            data=mpdta_data.rename({"lpop": ".w"}),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            xformla="~ .w",
        )


def test_att_gt_repeated_cross_section(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        panel=False,
        boot=False,
    )

    assert isinstance(result, MPResult)
    assert result.estimation_params["panel"] is False


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_att_gt_repeated_cross_section_ignores_idname(mpdta_data, allow_unbalanced_panel):
    spec = dict(yname="lemp", tname="year", gname="first.treat", est_method="reg", panel=False, boot=False, cband=False)

    plain = att_gt(data=mpdta_data, **spec)
    with_id = att_gt(data=mpdta_data, idname="countyreal", allow_unbalanced_panel=allow_unbalanced_panel, **spec)

    assert with_id.n_units == plain.n_units == mpdta_data.height
    np.testing.assert_allclose(with_id.att_gt, plain.att_gt, rtol=1e-12)
    np.testing.assert_allclose(with_id.se_gt, plain.se_gt, rtol=1e-12)
    np.testing.assert_allclose(with_id.se_gt[:4], [0.475829, 0.482270, 0.485621, 0.478955], atol=1e-6)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
def test_att_gt_repeated_cross_section_idname_universal_base(mpdta_data):
    spec = dict(
        yname="lemp",
        tname="year",
        gname="first.treat",
        est_method="reg",
        panel=False,
        base_period="universal",
        boot=False,
        cband=False,
    )

    plain = att_gt(data=mpdta_data, **spec)
    with_id = att_gt(data=mpdta_data, idname="countyreal", **spec)

    np.testing.assert_allclose(with_id.att_gt, plain.att_gt, rtol=1e-12)
    np.testing.assert_allclose(with_id.se_gt, plain.se_gt, rtol=1e-12)


def test_att_gt_rotating_cross_sections_count_rows(mpdta_rotating):
    spec = dict(yname="lemp", tname="year", gname="first.treat", est_method="reg", panel=False, boot=False, cband=False)

    plain = att_gt(data=mpdta_rotating, **spec)
    with_id = att_gt(data=mpdta_rotating, idname="countyreal", **spec)

    assert with_id.n_units == mpdta_rotating.height == 1500
    np.testing.assert_allclose(with_id.att_gt, plain.att_gt, rtol=1e-12)
    np.testing.assert_allclose(with_id.se_gt, plain.se_gt, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.filterwarnings("ignore:The Wald pre-test is not reported:UserWarning")
def test_att_gt_repeated_cross_section_idname_in_clustervars(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = dict(
        yname="lemp",
        tname="year",
        gname="first.treat",
        est_method="reg",
        panel=False,
        boot=True,
        biters=99,
        cband=False,
        random_state=7,
    )

    plain = att_gt(data=data, clustervars=["state"], **spec)
    with_id = att_gt(data=data, idname="countyreal", clustervars=["countyreal", "state"], **spec)

    np.testing.assert_array_equal(with_id.estimation_params["cluster"], plain.estimation_params["cluster"])
    np.testing.assert_allclose(with_id.se_gt, plain.se_gt, rtol=1e-12)
    np.testing.assert_allclose(
        aggte(with_id, type="simple", random_state=7).overall_se,
        aggte(plain, type="simple", random_state=7).overall_se,
        rtol=1e-12,
    )


def test_att_gt_unbalanced_panel(mpdta_data):
    mpdta_data = mpdta_data.filter(~((pl.col("countyreal") % 7 == 0) & (pl.col("year") == 2005)))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
        boot=False,
    )

    assert isinstance(result, MPResult)
    assert result.n_units == 500
    np.testing.assert_allclose(result.att_gt[:4], [-0.0105, -0.14462, -0.13726, -0.10081], atol=1e-4)
    np.testing.assert_allclose(result.se_gt[:4], [0.02325, 0.06288, 0.03644, 0.03436], atol=1e-4)


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
        boot=True,
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
        boot=False,
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


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tname": "yeer"}, "tname='yeer' is not a column in the data. Did you mean 'year'?"),
        (
            {"gname": "first_treat"},
            "gname='first_treat' is not a column in the data. Did you mean 'first.treat' or 'treat'?",
        ),
        ({"idname": "countyrel"}, "idname='countyrel' is not a column in the data. Did you mean 'countyreal'?"),
        (
            {"idname": "countyrel", "panel": False},
            "idname='countyrel' is not a column in the data. Did you mean 'countyreal'?",
        ),
        (
            {"clustervars": ["county"]},
            "'county' in clustervars is not a column in the data. Did you mean 'countyreal'?",
        ),
        (
            {"tname": "yeer", "xformla": "~ lpopp"},
            "tname='yeer' is not a column in the data. Did you mean 'year'?\n"
            "'lpopp' in xformla is not a column in the data. Did you mean 'lpop'?",
        ),
    ],
)
def test_att_gt_names_misspelled_columns(mpdta_data, changes, message):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"} | changes

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(mpdta_data, **spec)


def test_att_gt_all_treated_notyettreated(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        boot=False,
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
        boot=False,
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
        boot=False,
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
        boot=False,
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
        boot=False,
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
        boot=False,
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
        boot=True,
        biters=50,
        random_state=42,
    )

    result2 = att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=True,
        biters=50,
        random_state=42,
    )

    np.testing.assert_array_equal(result1.se_gt, result2.se_gt)
    assert result1.critical_value == result2.critical_value
    assert result1.estimation_params.get("random_state") == 42


def test_wald_pretest_skipped_with_extra_clustervars(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))
    mpdta_data = mpdta_data.with_columns((pl.col("countyreal") // 10).alias("cluster"))

    with pytest.warns(
        UserWarning,
        match="Wald pre-test is not reported when clustering beyond the unit level",
    ):
        result = att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            clustervars=["cluster"],
            boot=True,
            biters=10,
        )

    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_wald_pretest_returns_valid_stat(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )

    pre_indices = np.where(result.groups > result.times)[0]
    assert len(pre_indices) > 0, "Need pre-treatment periods"

    assert result.wald_stat is not None
    assert result.wald_pvalue is not None
    assert result.wald_stat > 0
    assert 0 <= result.wald_pvalue <= 1


def test_clustervars_strips_idname(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))
    mpdta_data = mpdta_data.with_columns((pl.col("countyreal") // 10).alias("cluster"))

    with pytest.warns(
        UserWarning,
        match="Wald pre-test is not reported when clustering beyond the unit level",
    ):
        result = att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            clustervars=["countyreal", "cluster"],
            boot=True,
            biters=10,
        )

    assert result.estimation_params["clustervars"] == ["cluster"]
    assert result.wald_stat is None


def test_wald_pretest_not_skipped_when_clustering_on_idname(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        clustervars=["countyreal"],
        boot=True,
        biters=10,
    )

    assert result.wald_stat is not None
    assert result.wald_pvalue is not None


def _make_mock_result(mpdta_data, modify_inf_func=None):
    real_result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )
    n_gt = len(real_result.groups)
    attgt_list = [
        ATTgtResult(att=real_result.att_gt[i], group=real_result.groups[i], year=real_result.times[i], post=0)
        for i in range(n_gt)
    ]
    inf_func = real_result.influence_func.copy()
    if modify_inf_func is not None:
        inf_func = modify_inf_func(inf_func, real_result)
    return ComputeATTgtResult(attgt_list=attgt_list, influence_functions=sp.csr_matrix(inf_func))


def test_wald_singular_variance_skips(mpdta_data):
    def make_collinear(inf_func, result):
        pre_indices = np.where(result.groups > result.times)[0]
        if len(pre_indices) >= 2:
            inf_func[:, pre_indices[1]] = inf_func[:, pre_indices[0]]
        return inf_func

    mock_result = _make_mock_result(mpdta_data, modify_inf_func=make_collinear)

    with (
        patch("moderndid.did.att_gt.compute_att_gt", return_value=mock_result),
        pytest.warns(UserWarning, match="singular covariance matrix"),
    ):
        result = att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            boot=False,
        )

    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_wald_linalg_error_skips(mpdta_data):
    mock_result = _make_mock_result(mpdta_data)

    with (
        patch("moderndid.did.att_gt.compute_att_gt", return_value=mock_result),
        patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError("mock")),
        pytest.warns(UserWarning, match="numerical issues"),
    ):
        result = att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            boot=False,
        )

    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_overlap_violation_returns_na():
    rng = np.random.default_rng(42)
    n_treated = 80
    n_control = 20
    n_units = n_treated + n_control
    times = [1, 2, 3, 4]
    rows = []
    for uid in range(1, n_units + 1):
        is_treated = uid <= n_treated
        g = float(2) if is_treated else float("inf")
        x = 10.0 + rng.normal(0, 0.01) if is_treated else -10.0 + rng.normal(0, 0.01)
        for t in times:
            rows.append({"id": uid, "time": t, "group": g, "y": rng.normal(), "x": x})
    data = pl.DataFrame(rows)

    with pytest.warns(UserWarning, match="Overlap condition violated"):
        result = att_gt(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            xformla="~ x",
            est_method="ipw",
            boot=False,
        )

    assert isinstance(result, MPResult)


@pytest.mark.parametrize("mpdta_converted", ["pandas", "pyarrow", "duckdb"], indirect=True)
def test_att_gt_dataframe_interoperability(mpdta_converted, att_gt_baseline_result):
    result = att_gt(
        data=mpdta_converted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )

    np.testing.assert_array_almost_equal(result.att_gt, att_gt_baseline_result.att_gt)
    np.testing.assert_array_almost_equal(result.se_gt, att_gt_baseline_result.se_gt)
    np.testing.assert_array_equal(result.groups, att_gt_baseline_result.groups)
    np.testing.assert_array_equal(result.times, att_gt_baseline_result.times)


@pytest.mark.parametrize("xformla", ["lpop ~ 1", "lemp ~ lpop"])
def test_att_gt_rejects_left_hand_side(mpdta_data, xformla):
    with pytest.raises(ValueError, match="has a left-hand side"):
        att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            xformla=xformla,
            est_method="reg",
        )


@pytest.mark.parametrize(
    ("mpdta_one_nan", "spec"),
    [
        ("lemp", {}),
        ("lpop", {"xformla": "~ lpop"}),
        ("pop", {"weightsname": "pop"}),
        ("year", {}),
        ("countyreal", {}),
        ("cluster", {"panel": False, "clustervars": ["cluster"]}),
    ],
    indirect=["mpdta_one_nan"],
)
def test_att_gt_drops_nan_rows_from_pandas_and_polars_alike(mpdta_one_nan, spec):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        **spec,
    }
    complete = mpdta_one_nan.filter(pl.all_horizontal(pl.col(pl.Float64).is_not_nan()))
    expected = att_gt(complete, **spec)

    for data in (mpdta_one_nan, mpdta_one_nan.to_pandas()):
        with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
            result = att_gt(data, **spec)
        assert result.n_units == expected.n_units
        np.testing.assert_array_equal(result.groups, expected.groups)
        np.testing.assert_array_equal(result.times, expected.times)
        np.testing.assert_array_equal(result.att_gt, expected.att_gt)
        np.testing.assert_array_equal(result.se_gt, expected.se_gt)


def test_att_gt_drops_nan_cohort_instead_of_making_it_never_treated(mpdta_nan_cohort):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    expected = att_gt(mpdta_nan_cohort.filter(pl.col("first.treat").is_not_nan()), **spec)

    for data in (mpdta_nan_cohort, mpdta_nan_cohort.to_pandas()):
        with pytest.warns(UserWarning, match="^Dropped 50 rows from original data due to missing values$"):
            result = att_gt(data, **spec)
        assert result.n_units == 490
        np.testing.assert_array_equal(result.att_gt, expected.att_gt)
        np.testing.assert_array_equal(result.se_gt, expected.se_gt)


def test_att_gt_rejects_negative_cohort(mpdta_negative_never_treated):
    with pytest.raises(ValueError, match="^gname = 'first.treat' holds negative values such as -1\\."):
        att_gt(
            data=mpdta_negative_never_treated,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            est_method="reg",
        )


def test_att_gt_infinite_cohort_marks_never_treated(mpdta_data, mpdta_infinite_never_treated):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    expected = att_gt(mpdta_data, **spec)
    result = att_gt(mpdta_infinite_never_treated, **spec)

    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize(
    ("mpdta_one_infinite", "spec"),
    [
        (("lemp", float("inf")), {}),
        (("lemp", float("-inf")), {"est_method": "dr"}),
        (("lpop", float("inf")), {"xformla": "~ lpop", "est_method": "dr"}),
        (("pop", float("inf")), {"weightsname": "pop"}),
        (("pop", float("-inf")), {"weightsname": "pop"}),
        (("year", float("-inf")), {}),
        (("countyreal", float("inf")), {}),
        (("cluster", float("inf")), {"panel": False, "clustervars": ["cluster"]}),
        (("lemp", float("inf")), {"panel": False}),
        (("lemp", float("-inf")), {"allow_unbalanced_panel": True}),
    ],
    indirect=["mpdta_one_infinite"],
    ids=[
        "outcome",
        "outcome-dr",
        "covariate",
        "weights",
        "weights-negative",
        "time",
        "unit",
        "cluster-rcs",
        "outcome-rcs",
        "outcome-unbalanced",
    ],
)
def test_att_gt_drops_infinite_rows_like_missing_ones(mpdta_one_infinite, spec):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        **spec,
    }
    expected = att_gt(mpdta_one_infinite.filter(pl.all_horizontal(pl.col(pl.Float64).is_finite())), **spec)

    for data in (mpdta_one_infinite, mpdta_one_infinite.to_pandas()):
        with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
            result = att_gt(data, **spec)
        assert result.n_units == expected.n_units
        np.testing.assert_array_equal(result.groups, expected.groups)
        np.testing.assert_array_equal(result.times, expected.times)
        np.testing.assert_array_equal(result.att_gt, expected.att_gt)
        np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize("mpdta_one_nan", ["first.treat", "cluster"], indirect=True)
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_att_gt_checks_cohorts_and_clusters_on_the_rows_that_remain(mpdta_one_nan, allow_unbalanced_panel):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "clustervars": ["cluster"],
        "boot": True,
        "biters": 49,
        "random_state": 7,
        "allow_unbalanced_panel": allow_unbalanced_panel,
        "est_method": "reg",
    }
    missing = pl.any_horizontal(pl.col("first.treat", "cluster").cast(pl.Float64).is_nan())
    county = mpdta_one_nan.filter(missing)["countyreal"].item()
    keep = ~missing if allow_unbalanced_panel else pl.col("countyreal") != county
    expected = att_gt(mpdta_one_nan.filter(keep), **spec)

    for data in (mpdta_one_nan, mpdta_one_nan.to_pandas()):
        result = att_gt(data, **spec)
        assert result.n_units == (500 if allow_unbalanced_panel else 499)
        np.testing.assert_array_equal(result.att_gt, expected.att_gt)
        np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize(("anticipation", "n_dropped"), [(0, 10), (1, 43)])
def test_att_gt_warns_once_about_units_treated_in_the_first_period(mpdta_early_cohorts, anticipation, n_dropped):
    with pytest.warns(UserWarning) as record:
        result = att_gt(
            mpdta_early_cohorts,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            anticipation=anticipation,
            est_method="reg",
        )

    messages = [str(warning.message) for warning in record if "first period" in str(warning.message)]
    assert messages == [f"Dropped {n_dropped} units that were already treated in the first period"]
    assert result.n_units == 500 - n_dropped


def test_att_gt_warns_once_about_units_dropped_to_balance_the_panel(mpdta_unbalanced):
    with pytest.warns(UserWarning) as record:
        result = att_gt(
            mpdta_unbalanced, yname="lemp", tname="year", idname="countyreal", gname="first.treat", est_method="reg"
        )

    assert [str(warning.message) for warning in record] == ["Dropped 74 units while converting to balanced panel"]
    assert result.n_units == 426


@pytest.mark.parametrize("mpdta_one_infinite", [("year", float("-inf"))], indirect=True)
def test_att_gt_warns_only_about_the_rows_and_units_it_drops(mpdta_one_infinite):
    with pytest.warns(UserWarning) as record:
        result = att_gt(
            mpdta_one_infinite, yname="lemp", tname="year", idname="countyreal", gname="first.treat", est_method="reg"
        )

    assert [str(warning.message) for warning in record] == [
        "Dropped 1 rows from original data due to missing values",
        "Dropped 1 units while converting to balanced panel",
    ]
    assert result.n_units == 499


@pytest.mark.parametrize("mpdta_bad_weights", ["zero", "zero outside an infinite row", "one negative"], indirect=True)
@pytest.mark.parametrize("panel", [True, False])
def test_att_gt_rejects_weights_without_positive_mean(mpdta_bad_weights, panel):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        att_gt(
            data=mpdta_bad_weights,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            weightsname="w",
            panel=panel,
            est_method="reg",
        )


def test_att_gt_checks_weights_after_dropping_missing_rows(mpdta_negative_weight_missing_outcome):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "weightsname": "pop",
        "est_method": "reg",
    }
    expected = att_gt(mpdta_negative_weight_missing_outcome.filter(pl.col("lemp").is_not_null()), **spec)

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = att_gt(mpdta_negative_weight_missing_outcome, **spec)

    assert result.n_units == 499
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_without_never_treated_leaves_latest_cohort_out(mpdta_without_never_treated, base_period):
    result = att_gt(
        data=mpdta_without_never_treated,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        base_period=base_period,
        est_method="reg",
    )
    post = result.groups <= result.times

    assert set(result.groups.tolist()) == {2004.0, 2006.0}
    np.testing.assert_allclose(
        result.att_gt[post], [-0.0353990145, -0.0925872029, -0.1339523822, 0.0264925124], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        result.se_gt[post], [0.0233767705, 0.0325760704, 0.0387084579, 0.0193805130], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(result.wald_stat, 0.997741952504902, rtol=1e-10)
    assert result.wald_pvalue == 0.60722


def test_att_gt_without_never_treated_event_study(mpdta_without_never_treated):
    result = att_gt(
        data=mpdta_without_never_treated,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        est_method="reg",
    )
    dynamic = aggte(result, type="dynamic")

    np.testing.assert_array_equal(result.groups, [2004, 2004, 2004, 2006, 2006, 2006])
    np.testing.assert_array_equal(result.times, [2004, 2005, 2006, 2004, 2005, 2006])
    np.testing.assert_array_equal(dynamic.event_times, [-2, -1, 0, 1, 2])
    np.testing.assert_allclose(
        dynamic.att_by_event,
        [-0.0239865432, -0.0000249259, 0.0058620035, -0.0925872029, -0.1339523822],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dynamic.se_by_event,
        [0.0240558316, 0.0224579722, 0.0157330074, 0.0325760704, 0.0387084579],
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("control_group", "expected_att"),
    [
        (
            "nevertreated",
            [
                0.0065201124,
                -0.0027508188,
                -0.0073454257,
                -0.0439752903,
                0.0305066556,
                -0.0027258929,
                -0.0310871194,
                -0.0260544107,
            ],
        ),
        (
            "notyettreated",
            [
                -0.0025625509,
                -0.0019392461,
                0.0027216302,
                -0.0439752903,
                0.0297593648,
                -0.0027258929,
                -0.0310871194,
                -0.0260544107,
            ],
        ),
    ],
)
def test_att_gt_anticipation_keeps_cohort_that_starts_after_panel(
    mpdta_cohort_after_panel, control_group, expected_att
):
    with pytest.warns(UserWarning, match="^Dropped 20 units that were already treated in the first period$"):
        result = att_gt(
            data=mpdta_cohort_after_panel,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            control_group=control_group,
            anticipation=1,
            est_method="reg",
        )

    np.testing.assert_array_equal(result.groups, [2006] * 4 + [2008] * 4)
    np.testing.assert_array_equal(result.times, [2004, 2005, 2006, 2007] * 2)
    np.testing.assert_allclose(result.att_gt, expected_att, rtol=0, atol=1e-9)


@pytest.mark.filterwarnings("error:.*unbalanced:UserWarning")
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_att_gt_rejects_repeated_unit_periods(mpdta_duplicated, allow_unbalanced_panel):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (countyreal, year) pair (17005, 2005)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        att_gt(
            data=mpdta_duplicated,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            est_method="reg",
            allow_unbalanced_panel=allow_unbalanced_panel,
        )
