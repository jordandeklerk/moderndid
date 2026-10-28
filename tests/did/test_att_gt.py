"""Tests for group-time average treatment effects."""

import re
import warnings
from collections import Counter
from importlib import import_module

import numpy as np
import pytest
from scipy.stats import chi2, norm

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import aggte, att_gt, drdid_panel, drdid_rc, reg_did_panel, reg_did_rc, std_ipw_did_panel, std_ipw_did_rc
from moderndid.core.preprocess import preprocess_did
from moderndid.did.att_gt import _finish_att_gt
from moderndid.did.mboot import _mboot


def test_att_gt_reg_without_covariates_is_the_difference_in_mean_outcome_changes(mpdta_data, mean_change_cells):
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ 1",
        est_method="reg",
    )
    att, influence = mean_change_cells(mpdta_data, cells)

    assert result.n_units == 500
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / 500), rtol=1e-12)


def test_att_gt_with_covariates(mpdta_data):
    cells = [(g, t, g - 1 if t >= g else t - 1) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]
    expected = []
    for g, t, base in cells:
        later = mpdta_data.filter(pl.col("year") == t).sort("countyreal")
        earlier = mpdta_data.filter(pl.col("year") == base).sort("countyreal")
        change = later["lemp"].to_numpy() - earlier["lemp"].to_numpy()
        covariates = np.column_stack([np.ones(earlier.height), earlier["lpop"].to_numpy()])
        treated = (earlier["first.treat"] == g).to_numpy()
        never = (earlier["first.treat"] == 0).to_numpy()
        slope = np.linalg.lstsq(covariates[never], change[never], rcond=None)[0]
        expected.append(np.mean(change[treated] - covariates[treated] @ slope))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        est_method="reg",
    )

    np.testing.assert_array_equal(result.groups, [g for g, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _ in cells])
    np.testing.assert_allclose(result.att_gt, expected, rtol=1e-12, atol=1e-13)


def test_att_gt_notyettreated_control(mpdta_data, mean_change_cells):
    cells = [
        (2004, 2004, 2003, [0, 2006, 2007]),
        (2004, 2005, 2003, [0, 2006, 2007]),
        (2004, 2006, 2003, [0, 2007]),
        (2004, 2007, 2003, [0]),
        (2006, 2004, 2003, [0, 2007]),
        (2006, 2005, 2004, [0, 2007]),
        (2006, 2006, 2005, [0, 2007]),
        (2006, 2007, 2005, [0]),
        (2007, 2004, 2003, [0, 2006]),
        (2007, 2005, 2004, [0, 2006]),
        (2007, 2006, 2005, [0]),
        (2007, 2007, 2006, [0]),
    ]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        boot=False,
    )
    att, influence = mean_change_cells(mpdta_data, cells)

    assert result.estimation_params["control_group"] == "notyettreated"
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / 500), rtol=1e-12)


@pytest.mark.filterwarnings("ignore:Dropped.*units that were already treated:UserWarning")
def test_att_gt_with_anticipation(mpdta_data, mean_change_cells):
    cells = [
        (2006, 2004, 2003, [0]),
        (2006, 2005, 2004, [0]),
        (2006, 2006, 2004, [0]),
        (2006, 2007, 2004, [0]),
        (2007, 2004, 2003, [0]),
        (2007, 2005, 2004, [0]),
        (2007, 2006, 2005, [0]),
        (2007, 2007, 2005, [0]),
    ]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        anticipation=1,
        boot=False,
    )
    att, influence = mean_change_cells(mpdta_data, cells)

    assert result.estimation_params["anticipation_periods"] == 1
    assert result.n_units == 480
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / len(influence)), rtol=1e-12)


def test_att_gt_bootstrap_inference(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}

    result = att_gt(data=mpdta_data, boot=True, biters=999, cband=True, random_state=7, **spec)
    analytic = att_gt(data=mpdta_data, **spec)

    assert result.estimation_params["bootstrap"] is True
    assert result.estimation_params["uniform_bands"] is True
    assert result.se_gt.size == 12
    assert np.all(result.se_gt != analytic.se_gt)
    np.testing.assert_allclose(result.se_gt, analytic.se_gt, rtol=0.2)
    assert norm.ppf(0.975) < result.critical_value < norm.ppf(1 - 0.025 / 12)


def test_att_gt_bootstrap_without_uniform_bands_keeps_the_pointwise_critical_value(mpdta_data):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        "boot": True,
        "biters": 999,
        "random_state": 7,
    }

    pointwise = att_gt(data=mpdta_data, cband=False, **spec)
    uniform = att_gt(data=mpdta_data, cband=True, **spec)

    assert pointwise.estimation_params["uniform_bands"] is False
    np.testing.assert_allclose(pointwise.critical_value, norm.ppf(0.975), rtol=1e-12)
    assert uniform.critical_value > pointwise.critical_value + 0.5
    np.testing.assert_array_equal(pointwise.se_gt, uniform.se_gt)


def test_att_gt_alpha_sets_the_simultaneous_critical_value_of_the_bootstrap(mpdta_data):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        "boot": True,
        "biters": 999,
        "random_state": 7,
    }

    tenth = att_gt(data=mpdta_data, alpha=0.10, **spec)
    twentieth = att_gt(data=mpdta_data, alpha=0.05, **spec)

    np.testing.assert_array_equal(tenth.se_gt, twentieth.se_gt)
    assert norm.ppf(0.95) < tenth.critical_value < norm.ppf(1 - 0.05 / 12)
    assert tenth.critical_value < twentieth.critical_value - 0.1


@pytest.mark.parametrize(
    ("est_method", "estimator"), [("dr", drdid_panel), ("ipw", std_ipw_did_panel), ("reg", reg_did_panel)]
)
def test_att_gt_estimation_methods(est_method, estimator, mpdta_data):
    cells = [(g, t, g - 1 if t >= g else t - 1) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]
    direct = []
    for g, t, base in cells:
        later = mpdta_data.filter(pl.col("year") == t).sort("countyreal")
        earlier = mpdta_data.filter(pl.col("year") == base).sort("countyreal")
        cell = earlier["first.treat"].is_in([0, g]).to_numpy()
        covariates = np.column_stack([np.ones(cell.sum()), earlier["lpop"].to_numpy()[cell]])
        treated = (earlier["first.treat"] == g).to_numpy()[cell].astype(float)
        y1, y0 = later["lemp"].to_numpy()[cell], earlier["lemp"].to_numpy()[cell]
        direct.append(estimator(y1, y0, treated, covariates, influence_func=True))

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        est_method=est_method,
        boot=False,
    )

    assert result.estimation_params["estimation_method"] == est_method
    np.testing.assert_array_equal(result.groups, [g for g, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _ in cells])
    np.testing.assert_allclose(result.att_gt, [fit.att for fit in direct], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        result.se_gt, [np.sqrt(np.sum(fit.att_inf_func**2)) / fit.att_inf_func.size for fit in direct], rtol=1e-12
    )


@pytest.mark.parametrize(("est_method", "estimator"), [("dr", drdid_rc), ("ipw", std_ipw_did_rc), ("reg", reg_did_rc)])
def test_att_gt_estimation_methods_on_repeated_cross_sections(est_method, estimator, mpdta_pop_weighted):
    cells = [(g, t, g - 1 if t >= g else t - 1) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]
    direct = []
    for g, t, base in cells:
        rows = mpdta_pop_weighted.filter(pl.col("year").is_in([t, base]) & pl.col("first.treat").is_in([0, g]))
        covariates = np.column_stack([np.ones(rows.height), rows["lpop"].to_numpy()])
        post = (rows["year"] == t).to_numpy().astype(float)
        treated = (rows["first.treat"] == g).to_numpy().astype(float)
        y = rows["lemp"].to_numpy()
        direct.append(estimator(y, post, treated, covariates, i_weights=rows["pop"].to_numpy(), influence_func=True))

    result = att_gt(
        data=mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        gname="first.treat",
        xformla="~ lpop",
        weightsname="pop",
        est_method=est_method,
        panel=False,
        boot=False,
    )

    np.testing.assert_array_equal(result.groups, [g for g, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _ in cells])
    np.testing.assert_allclose(result.att_gt, [fit.att for fit in direct], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        result.se_gt, [np.sqrt(np.sum(fit.att_inf_func**2)) / fit.att_inf_func.size for fit in direct], rtol=1e-12
    )


def test_att_gt_universal_base_period(mpdta_data, mean_change_cells):
    cells = [(g, t, g - 1, [0]) for g in (2004, 2006, 2007) for t in (2003, 2004, 2005, 2006, 2007)]
    reference = np.array([t == base for _, t, base, _ in cells])

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        base_period="universal",
        boot=False,
    )
    att, influence = mean_change_cells(mpdta_data, cells)

    assert result.estimation_params["base_period"] == "universal"
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        result.se_gt, np.where(reference, np.nan, np.sqrt(np.mean(influence**2, axis=0) / 500)), rtol=1e-12
    )


def test_att_gt_with_weights(mpdta_data, mean_change_cells):
    counties = mpdta_data["countyreal"].unique().sort()
    draws = np.random.default_rng(42).uniform(0.2, 3.0, counties.len())
    data = mpdta_data.join(pl.DataFrame({"countyreal": counties, "weights": draws}), on="countyreal")
    units = data.filter(pl.col("year") == 2003)
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    result = att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="weights",
        boot=False,
    )
    att, influence = mean_change_cells(data, cells, weightsname="weights")
    pairs = np.column_stack([np.asarray(result.G), np.asarray(result.weights_ind)])
    cohorts = np.where(units["first.treat"] == 0, np.inf, units["first.treat"])
    expected_pairs = np.column_stack([cohorts, units["weights"] / units["weights"].mean()])

    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / 500), rtol=1e-12)
    np.testing.assert_allclose(
        pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))],
        expected_pairs[np.lexsort((expected_pairs[:, 1], expected_pairs[:, 0]))],
        rtol=1e-12,
    )


def test_att_gt_weights_any_column_name(mpdta_pop_weighted):
    spec = dict(yname="lemp", tname="year", idname="countyreal", gname="first.treat", boot=False, cband=False)
    named = att_gt(data=mpdta_pop_weighted, weightsname="pop", **spec)
    renamed = att_gt(data=mpdta_pop_weighted.rename({"pop": "weights"}), weightsname="weights", **spec)
    by_name = [aggte(named, type=agg_type) for agg_type in ("simple", "group", "dynamic", "calendar")]
    by_rename = [aggte(renamed, type=agg_type) for agg_type in ("simple", "group", "dynamic", "calendar")]

    assert named.weights_ind is not None
    np.testing.assert_array_equal(np.asarray(named.weights_ind), np.asarray(renamed.weights_ind))
    np.testing.assert_array_equal([agg.overall_att for agg in by_name], [agg.overall_att for agg in by_rename])
    np.testing.assert_array_equal([agg.overall_se for agg in by_name], [agg.overall_se for agg in by_rename])


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
    bases = [g - 1 if t >= g or base_period == "universal" else t - 1 for g, t in zip(result.groups, result.times)]
    earlier = [fixed[int(min(base, t))] for base, t in zip(bases, result.times)]

    assert result.att_gt.size == (15 if base_period == "universal" else 12)
    np.testing.assert_array_equal([fit.groups[i] for i, fit in enumerate(earlier)], result.groups)
    np.testing.assert_array_equal([fit.times[i] for i, fit in enumerate(earlier)], result.times)
    np.testing.assert_allclose(result.att_gt, [fit.att_gt[i] for i, fit in enumerate(earlier)], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.se_gt, [fit.se_gt[i] for i, fit in enumerate(earlier)], rtol=1e-12, atol=1e-14)


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


def test_att_gt_repeated_cross_section(mpdta_data, period_mean_cells):
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        panel=False,
        boot=False,
    )
    att, influence = period_mean_cells(mpdta_data, cells)

    assert result.estimation_params["panel"] is False
    assert result.n_units == 2500
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / 2500), rtol=1e-12)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_att_gt_repeated_cross_section_ignores_idname(mpdta_data, period_mean_cells, allow_unbalanced_panel):
    spec = dict(yname="lemp", tname="year", gname="first.treat", est_method="reg", panel=False, boot=False, cband=False)
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    plain = att_gt(data=mpdta_data, **spec)
    with_id = att_gt(data=mpdta_data, idname="countyreal", allow_unbalanced_panel=allow_unbalanced_panel, **spec)
    _, influence = period_mean_cells(mpdta_data, cells)

    assert with_id.n_units == plain.n_units == mpdta_data.height
    np.testing.assert_array_equal(with_id.att_gt, plain.att_gt)
    np.testing.assert_array_equal(with_id.se_gt, plain.se_gt)
    np.testing.assert_allclose(with_id.se_gt, np.sqrt(np.mean(influence**2, axis=0) / 2500), rtol=1e-12)


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

    np.testing.assert_array_equal(with_id.att_gt, plain.att_gt)
    np.testing.assert_array_equal(with_id.se_gt, plain.se_gt)


def test_att_gt_rotating_cross_sections_count_rows(mpdta_rotating):
    spec = dict(yname="lemp", tname="year", gname="first.treat", est_method="reg", panel=False, boot=False, cband=False)

    plain = att_gt(data=mpdta_rotating, **spec)
    with_id = att_gt(data=mpdta_rotating, idname="countyreal", **spec)

    assert with_id.n_units == mpdta_rotating.height == 1500
    np.testing.assert_array_equal(with_id.att_gt, plain.att_gt)
    np.testing.assert_array_equal(with_id.se_gt, plain.se_gt)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
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
    plain_simple = aggte(plain, type="simple", random_state=7)
    with_id_simple = aggte(with_id, type="simple", random_state=7)

    np.testing.assert_array_equal(with_id.estimation_params["cluster"], plain.estimation_params["cluster"])
    np.testing.assert_array_equal(with_id.se_gt, plain.se_gt)
    assert with_id_simple.overall_se == plain_simple.overall_se


def test_att_gt_unbalanced_panel(mpdta_unbalanced, period_mean_cells):
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    result = att_gt(
        data=mpdta_unbalanced,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
        boot=False,
    )
    att, influence = period_mean_cells(mpdta_unbalanced, cells, idname="countyreal")

    assert result.n_units == 500
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / 500), rtol=1e-12)


@pytest.mark.parametrize(
    ("cluster", "message"),
    [
        (
            pl.col("countyreal") // 1000,
            "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
            "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give confidence "
            "intervals and bands that are too narrow.",
        ),
        (
            pl.when(pl.col("first.treat") == 2004).then(pl.col("countyreal") % 3).otherwise(pl.col("countyreal")),
            "The treated units of cohort 2004 fall in 3 clusters. Cluster-robust inference for this cohort rests on "
            "very few clusters and can give confidence intervals and bands that are too narrow.",
        ),
        (
            pl.col("countyreal") % 25,
            "The units fall in only 25 clusters. Cluster-robust inference rests on very few clusters and can give "
            "confidence intervals and bands that are too narrow.",
        ),
    ],
    ids=["states", "one_cohort", "total"],
)
@pytest.mark.parametrize("panel", [True, False], ids=["panel", "repeated_cross_sections"])
@pytest.mark.parametrize("boot", [True, False], ids=["bootstrap", "analytic"])
def test_att_gt_warns_once_when_its_clusters_are_few(mpdta_data, cluster, message, panel, boot):
    data = mpdta_data.with_columns(cluster.alias("cluster"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(data=data, panel=panel, clustervars=["cluster"], boot=boot, biters=99, random_state=7, **spec)

    few = [w for w in caught if "very few clusters" in str(w.message)]
    kind = "Bootstrap" if boot else "Analytical"
    assert [str(w.message) for w in few] == [message]
    assert [w.filename for w in few] == [__file__]
    assert f" {kind} standard errors clustered by cluster" in str(result).splitlines()
    assert result.__maketables_stat__("se_type") == f"{kind} clustered by cluster"
    assert result.__maketables_vcov_info__ == {"vcov_type": kind.lower(), "clustervar": "cluster"}


def test_att_gt_few_clusters_warning_points_at_the_call_through_the_backend_argument(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        att_gt(data=data, clustervars=["state"], backend="numpy", **spec)

    assert [w.filename for w in caught if "very few clusters" in str(w.message)] == [__file__]


@pytest.mark.parametrize(
    ("data_fixture", "options"),
    [
        (
            "mpdta_unbalanced",
            {
                "idname": "countyreal",
                "xformla": "~ lpop",
                "weightsname": "pop",
                "alpha": 0.1,
                "cband": False,
                "boot": True,
                "biters": 99,
                "clustervars": ["cluster"],
                "est_method": "ipw",
                "allow_unbalanced_panel": True,
                "control_group": "notyettreated",
                "anticipation": 1,
                "base_period": "universal",
                "random_state": 7,
            },
        ),
        ("mpdta_data", {"panel": False, "est_method": "reg"}),
    ],
    ids=["unbalanced_panel", "repeated_cross_sections"],
)
def test_att_gt_backend_argument_passes_every_setting_through(request, data_fixture, options):
    data = request.getfixturevalue(data_fixture).with_columns(
        pl.col("lpop").exp().alias("pop"), (pl.col("countyreal") % 50).alias("cluster")
    )
    spec = {"yname": "lemp", "tname": "year", "gname": "first.treat", **options}

    with warnings.catch_warnings(record=True) as default_warnings:
        warnings.simplefilter("always")
        default = att_gt(data, **spec)
    with warnings.catch_warnings(record=True) as numpy_warnings:
        warnings.simplefilter("always")
        on_numpy = att_gt(data, backend="numpy", **spec)

    assert [str(w.message) for w in numpy_warnings] == [str(w.message) for w in default_warnings]
    assert on_numpy.n_units == default.n_units
    assert on_numpy.critical_value == default.critical_value
    np.testing.assert_array_equal(on_numpy.groups, default.groups)
    np.testing.assert_array_equal(on_numpy.times, default.times)
    np.testing.assert_array_equal(on_numpy.att_gt, default.att_gt)
    np.testing.assert_array_equal(on_numpy.se_gt, default.se_gt)


@pytest.mark.parametrize("boot", [True, False], ids=["bootstrap", "analytic"])
def test_att_gt_stays_silent_with_plentiful_or_unit_level_clusters(mpdta_data, boot):
    data = mpdta_data.with_columns((pl.col("countyreal") % 50).alias("cluster"), pl.col("countyreal").alias("copy"))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        "boot": boot,
        "biters": 99,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plentiful = att_gt(data=data, clustervars=["cluster"], **spec)
        unit_level = att_gt(data=data, clustervars=["copy"], **spec)
    plain = att_gt(data=data, **spec)

    kind = "Bootstrap" if boot else "Analytical"
    assert not [w for w in caught if "very few clusters" in str(w.message)]
    assert f" {kind} standard errors clustered by cluster" in str(plentiful).splitlines()
    assert f" {kind} standard errors" in str(unit_level).splitlines()
    assert unit_level.__maketables_stat__("se_type") == kind
    assert unit_level.__maketables_vcov_info__ == {"vcov_type": kind.lower(), "clustervar": None}
    np.testing.assert_array_equal(unit_level.vcov_analytical, plain.vcov_analytical)
    assert unit_level.wald_stat == plain.wald_stat


@pytest.mark.parametrize(
    ("data_fixture", "options"),
    [("mpdta_data", {}), ("mpdta_unbalanced", {"allow_unbalanced_panel": True}), ("mpdta_data", {"panel": False})],
    ids=["panel", "unbalanced_panel", "repeated_cross_sections"],
)
def test_att_gt_without_the_bootstrap_sums_the_influence_function_within_clusters(request, data_fixture, options):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}

    clustered = att_gt(data, clustervars=["state"], **spec, **options)
    plain = att_gt(data, **spec, **options)
    cluster = clustered.estimation_params["cluster"]
    _, cluster_idx = np.unique(cluster, return_inverse=True)
    sums = np.stack([np.bincount(cluster_idx, weights=column) for column in clustered.influence_func.T], axis=1)
    covariance = sums.T @ sums / clustered.n_units
    expected = np.sqrt(np.diag(covariance) / clustered.n_units)
    expected[expected < 1e-7] = np.nan
    rows = data if options.get("panel") is False else data.unique("countyreal")
    cohorts = np.where(rows["first.treat"] == 0, np.inf, rows["first.treat"])

    np.testing.assert_allclose(clustered.se_gt, expected, rtol=1e-12)
    np.testing.assert_allclose(clustered.vcov_analytical, covariance, rtol=1e-12)
    np.testing.assert_array_equal(clustered.att_gt, plain.att_gt)
    assert np.nanmax(np.abs(clustered.se_gt - plain.se_gt)) > 1e-3
    assert len(cluster) == clustered.influence_func.shape[0]
    assert Counter(zip(np.asarray(clustered.G).tolist(), cluster.tolist())) == Counter(
        zip(cohorts.tolist(), rows["state"].to_list())
    )
    assert " Analytical standard errors clustered by state" in str(clustered).splitlines()


@pytest.mark.parametrize("boot", [True, False], ids=["bootstrap", "analytic"])
def test_att_gt_reports_no_wald_pretest_when_it_clusters(mpdta_data, boot):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    explanation = [
        " Since a Wald test on a cluster-robust covariance rejects too often at",
        " realistic cluster counts, no pre-test of parallel trends is reported.",
        " Use the bootstrap's uniform bands over the pre-treatment cells instead.",
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clustered = att_gt(data, clustervars=["state"], boot=boot, biters=99, random_state=7, **spec)
    plain = att_gt(data, boot=boot, biters=99, random_state=7, **spec)
    _, cluster_idx = np.unique(clustered.estimation_params["cluster"], return_inverse=True)
    sums = np.stack([np.bincount(cluster_idx, weights=column) for column in clustered.influence_func.T], axis=1)
    report = str(clustered).splitlines()

    assert clustered.wald_stat is None
    assert clustered.wald_pvalue is None
    assert not [w for w in caught if "Wald" in str(w.message)]
    assert report[report.index(explanation[0]) : report.index(explanation[0]) + 3] == explanation
    assert not [line for line in report if "P-value for pre-test" in line]
    np.testing.assert_allclose(clustered.vcov_analytical, sums.T @ sums / clustered.n_units, rtol=1e-12)
    assert plain.wald_pvalue is not None
    assert explanation[0] not in str(plain).splitlines()


def test_att_gt_wald_pretest(mpdta_data, mean_change_cells):
    cells = [(g, t, t - 1, [0]) for g, t in ((2006, 2004), (2006, 2005), (2007, 2004), (2007, 2005), (2007, 2006))]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )
    att, influence = mean_change_cells(mpdta_data, cells)
    wald = 500 * att @ np.linalg.solve(influence.T @ influence / 500, att)

    np.testing.assert_allclose(result.wald_stat, wald, rtol=1e-12)
    np.testing.assert_allclose(result.wald_pvalue, chi2.sf(wald, 5), rtol=0, atol=5e-6)


def test_att_gt_invalid_control_group(mpdta_data):
    message = "control_group='invalid' is not valid. Must be 'nevertreated' or 'notyettreated'."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            control_group="invalid",
        )


def test_att_gt_missing_column(mpdta_data):
    message = "yname='missing_column' is not a column in the data."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
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


def test_att_gt_summary_output(mpdta_data, mean_change_cells):
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )
    att, influence = mean_change_cells(mpdta_data, cells)
    se = np.sqrt(np.mean(influence**2, axis=0) / 500)
    lower, upper = att - norm.ppf(0.975) * se, att + norm.ppf(0.975) * se
    lines = str(result).splitlines()
    table = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]
    printed = np.array([[float(value) for value in re.findall(r"-?\d+\.?\d*", " ".join(row))] for row in table[1:]])

    assert lines[1] == " Group-Time Average Treatment Effects"
    assert table[0] == ["Group", "Time", "ATT(g,t)", "Std. Error", "[95% Pointwise Conf. Band]"]
    np.testing.assert_array_equal(printed[:, :2], [(g, t) for g, t, _, _ in cells])
    np.testing.assert_allclose(printed[:, 2:], np.column_stack([att, se, lower, upper]), rtol=0, atol=5e-5)
    assert [row[-1].endswith("*") for row in table[1:]] == list((lower > 0) | (upper < 0))


def test_att_gt_influence_function_matches_weight_derivative(mpdta_data, weight_derivative):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~ lpop",
        "weightsname": "w",
    }
    counties = mpdta_data["countyreal"].unique().sort().to_numpy()
    picks = [mpdta_data.filter(pl.col("first.treat") == g)["countyreal"].min() for g in (2004, 2006, 2007, 0)]
    data = mpdta_data.with_columns(pl.lit(1.0).alias("w"))

    def estimate(weights):
        weighted = mpdta_data.join(pl.DataFrame({"countyreal": counties, "w": weights}), on="countyreal")
        return att_gt(weighted, **spec).att_gt

    result = att_gt(data, **spec)
    order = preprocess_did(data, **spec).time_invariant_data["countyreal"].to_numpy()
    derivative = weight_derivative(estimate, counties.size, [int(np.flatnonzero(counties == c)[0]) for c in picks])

    np.testing.assert_allclose(
        result.influence_func[[int(np.flatnonzero(order == county)[0]) for county in picks]],
        derivative,
        rtol=0,
        atol=1e-6,
    )


def test_att_gt_variance_matrix(mpdta_data, mean_change_cells):
    cells = [(g, t, g - 1 if t >= g else t - 1, [0]) for g in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)]

    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )
    _, influence = mean_change_cells(mpdta_data, cells)

    np.testing.assert_allclose(result.vcov_analytical, influence.T @ influence / 500, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.diag(result.vcov_analytical) / 500), rtol=1e-12)


def test_att_gt_custom_alpha(mpdta_data):
    result = att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        alpha=0.10,
        boot=False,
    )

    assert result.alpha == 0.10
    np.testing.assert_allclose(result.critical_value, norm.ppf(0.95), rtol=1e-12)


def test_att_gt_rejects_the_removed_alp_argument(mpdta_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'alp'"):
        att_gt(data=mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", alp=0.10)


def test_att_gt_bootstrap_reproducibility(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties)).sort(["countyreal", "year"])
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "boot": True,
        "biters": 50,
    }

    result1 = att_gt(data=data, random_state=42, **spec)
    result2 = att_gt(data=data, random_state=42, **spec)
    other = att_gt(data=data, random_state=43, **spec)

    np.testing.assert_array_equal(result1.se_gt, result2.se_gt)
    assert result1.critical_value == result2.critical_value
    assert result1.estimation_params.get("random_state") == 42
    assert np.all(other.se_gt != result1.se_gt)
    assert other.critical_value != result1.critical_value


def test_att_gt_biters_sets_the_draws_of_its_bootstrap_and_of_aggte(mpdta_data):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        "boot": True,
        "random_state": 7,
    }

    few = att_gt(data=mpdta_data, biters=99, **spec)
    many = att_gt(data=mpdta_data, biters=999, **spec)
    few_dynamic = aggte(few, type="dynamic")
    many_with_few_draws = aggte(many, type="dynamic", biters=99)

    assert few.estimation_params["biters"] == 99
    assert abs(few.critical_value - many.critical_value) > 1e-2
    np.testing.assert_array_equal(few_dynamic.se_by_event, many_with_few_draws.se_by_event)
    np.testing.assert_array_equal(few_dynamic.critical_values, many_with_few_draws.critical_values)


def test_clustervars_strips_idname(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))
    mpdta_data = mpdta_data.with_columns((pl.col("countyreal") // 10).alias("cluster"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}

    with_id = att_gt(data=mpdta_data, clustervars=["countyreal", "cluster"], **spec)
    without_id = att_gt(data=mpdta_data, clustervars=["cluster"], **spec)

    assert with_id.estimation_params["clustervars"] == ["cluster"]
    np.testing.assert_array_equal(with_id.se_gt, without_id.se_gt)
    np.testing.assert_array_equal(with_id.vcov_analytical, without_id.vcov_analytical)
    assert with_id.wald_stat is None


def test_wald_pretest_not_skipped_when_clustering_on_idname(mpdta_data):
    unique_counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    mpdta_data = mpdta_data.filter(pl.col("countyreal").is_in(unique_counties))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "boot": True,
        "biters": 10,
        "random_state": 7,
    }

    result = att_gt(data=mpdta_data, clustervars=["countyreal"], **spec)
    plain = att_gt(data=mpdta_data, **spec)

    assert result.wald_stat is not None
    assert result.wald_stat == plain.wald_stat
    assert result.wald_pvalue == plain.wald_pvalue
    np.testing.assert_array_equal(result.se_gt, plain.se_gt)


def test_att_gt_records_idname_and_no_cluster_vector_for_idname_alone(mpdta_data):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        "clustervars": ["countyreal"],
        "boot": True,
        "biters": 99,
        "random_state": 7,
    }

    panel = att_gt(mpdta_data, **spec)
    with pytest.warns(UserWarning, match="^panel=False was specified, but units appear across all time periods\\. "):
        cross_section = att_gt(mpdta_data, panel=False, **spec)

    assert [result.estimation_params["idname"] for result in (panel, cross_section)] == ["countyreal"] * 2
    assert [result.estimation_params["clustervars"] for result in (panel, cross_section)] == [[], []]
    assert [result.estimation_params["cluster"] for result in (panel, cross_section)] == [None, None]


@pytest.mark.parametrize(
    ("data_fixture", "options"),
    [
        ("mpdta_data", {}),
        ("mpdta_unbalanced", {"allow_unbalanced_panel": True}),
        ("mpdta_data", {"panel": False, "idname": None}),
    ],
    ids=["panel", "unbalanced_panel", "repeated_cross_sections"],
)
def test_att_gt_records_its_observations_and_units(request, data_fixture, options):
    data = request.getfixturevalue(data_fixture)
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        "random_state": 7,
    }
    n_units = data.height if options.get("panel") is False else data["countyreal"].n_unique()

    result = att_gt(data, **(spec | options))
    simple = aggte(result, type="simple")

    assert result.estimation_params["n_obs"] == data.height
    assert result.estimation_params["n_units"] == result.n_units == n_units
    assert simple.__maketables_stat__("N") == data.height


def test_wald_singular_variance_skips(collinear_pretrend_panel):
    with pytest.warns(UserWarning, match="^Not returning pre-test Wald statistic due to singular covariance matrix$"):
        result = att_gt(collinear_pretrend_panel, yname="y", tname="t", idname="id", gname="g", est_method="reg")

    assert list(zip(result.groups.tolist(), result.times.tolist(), strict=True)) == [
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    np.testing.assert_allclose(result.influence_func[:, 0], result.influence_func[:, 3], rtol=1e-12, atol=1e-14)
    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_wald_linalg_error_skips(monkeypatch, att_gt_baseline_result):
    fit = att_gt_baseline_result

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("solve failed")

    monkeypatch.setattr(np.linalg, "solve", fail)

    with pytest.warns(UserWarning, match="^Not returning pre-test Wald statistic due to numerical issues$"):
        result = _finish_att_gt(
            groups=fit.groups,
            times=fit.times,
            att_values=fit.att_gt,
            variance_matrix=fit.vcov_analytical,
            clustered=False,
            bootstrap_results=None,
            n_units=fit.n_units,
            alpha=0.05,
            cband=False,
            anticipation=0,
            estimation_params=fit.estimation_params,
        )

    assert fit.wald_stat is not None
    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_att_gt_skips_the_wald_pretest_when_an_influence_function_is_infinite(mpdta_data):
    def infinite_estimator(y1, y0, d, covariates, i_weights=None, influence_func=True):
        result = drdid_panel(y1, y0, d, covariates, i_weights=i_weights, influence_func=True)
        return result._replace(att_inf_func=np.r_[np.inf, result.att_inf_func[1:]])

    with pytest.warns(UserWarning, match="^Not returning pre-test Wald statistic due to NaN pre-treatment values$"):
        result = att_gt(
            mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            est_method=infinite_estimator,
        )

    assert np.all(np.isinf(result.se_gt))
    assert result.wald_stat is None
    assert result.wald_pvalue is None


def test_att_gt_wald_pretest_warning_names_the_anticipation_that_leaves_no_pre_treatment_cell(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    late = mpdta_data.filter(pl.col("year") != 2004).with_columns(
        pl.when(pl.col("first.treat") > 0).then(2005).otherwise(0).alias("first.treat")
    )
    early = mpdta_data.filter(pl.col("first.treat").is_in([0, 2004]))
    message = (
        "No pre-treatment periods available for the Wald pre-test of parallel trends. This can happen when all "
        "groups are first treated early in the panel, such as in the second time period, so that no pre-treatment "
        "ATT(g,t) estimates exist."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anticipating = att_gt(late, anticipation=1, **spec)
        immediate = att_gt(early, **spec)

    assert [str(w.message) for w in caught if str(w.message).startswith("No pre-treatment")] == [
        f"{message} With anticipation=1, fewer pre-treatment periods remain.",
        message,
    ]
    np.testing.assert_array_equal(anticipating.times, [2005, 2006, 2007])
    np.testing.assert_array_equal(immediate.times, [2004, 2005, 2006, 2007])
    assert anticipating.wald_stat is None
    assert immediate.wald_stat is None


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

    with pytest.warns(UserWarning, match="^Overlap condition violated for 2 in time period 2$") as record:
        result = att_gt(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            xformla="~ x",
            est_method="ipw",
            base_period="universal",
            boot=False,
        )

    assert [str(w.message) for w in record if str(w.message).startswith("Overlap")] == [
        f"Overlap condition violated for 2 in time period {t}" for t in (2, 3, 4)
    ]
    np.testing.assert_array_equal(result.times, [1, 2, 3, 4])
    np.testing.assert_array_equal(result.att_gt, [0.0, np.nan, np.nan, np.nan])
    assert np.all(np.isnan(result.se_gt))


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

    np.testing.assert_array_equal(result.att_gt, att_gt_baseline_result.att_gt)
    np.testing.assert_array_equal(result.se_gt, att_gt_baseline_result.se_gt)
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
@pytest.mark.parametrize("convert", [pl.DataFrame.clone, pl.DataFrame.to_pandas], ids=["polars", "pandas"])
def test_att_gt_drops_nan_rows_from_pandas_and_polars_alike(mpdta_one_nan, spec, convert):
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

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = att_gt(convert(mpdta_one_nan), **spec)

    assert result.n_units == expected.n_units
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize("convert", [pl.DataFrame.clone, pl.DataFrame.to_pandas], ids=["polars", "pandas"])
def test_att_gt_drops_nan_cohort_instead_of_making_it_never_treated(mpdta_nan_cohort, convert):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}
    expected = att_gt(mpdta_nan_cohort.filter(pl.col("first.treat").is_not_nan()), **spec)

    with pytest.warns(UserWarning, match="^Dropped 50 rows from original data due to missing values$"):
        result = att_gt(convert(mpdta_nan_cohort), **spec)

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
@pytest.mark.parametrize("convert", [pl.DataFrame.clone, pl.DataFrame.to_pandas], ids=["polars", "pandas"])
def test_att_gt_drops_infinite_rows_like_missing_ones(mpdta_one_infinite, spec, convert):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "reg",
        **spec,
    }
    expected = att_gt(mpdta_one_infinite.filter(pl.all_horizontal(pl.col(pl.Float64).is_finite())), **spec)

    with pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"):
        result = att_gt(convert(mpdta_one_infinite), **spec)

    assert result.n_units == expected.n_units
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize("mpdta_one_nan", ["first.treat", "cluster"], indirect=True)
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
@pytest.mark.parametrize("convert", [pl.DataFrame.clone, pl.DataFrame.to_pandas], ids=["polars", "pandas"])
def test_att_gt_checks_cohorts_and_clusters_on_the_rows_that_remain(mpdta_one_nan, allow_unbalanced_panel, convert):
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

    result = att_gt(convert(mpdta_one_nan), **spec)

    assert result.n_units == (500 if allow_unbalanced_panel else 499)
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize(("anticipation", "n_dropped"), [(0, 10), (1, 43)])
def test_att_gt_warns_once_about_units_treated_in_the_first_period(mpdta_early_cohorts, anticipation, n_dropped):
    message = f"Dropped {n_dropped} units that were already treated in the first period"

    with pytest.warns(UserWarning, match=f"^{message}$") as record:
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
    assert messages == [message]
    assert result.n_units == 500 - n_dropped


def test_att_gt_warns_once_about_units_dropped_to_balance_the_panel(mpdta_unbalanced):
    with pytest.warns(UserWarning, match="^Dropped 74 units while converting to balanced panel$") as record:
        result = att_gt(
            mpdta_unbalanced, yname="lemp", tname="year", idname="countyreal", gname="first.treat", est_method="reg"
        )

    assert [str(warning.message) for warning in record] == ["Dropped 74 units while converting to balanced panel"]
    assert result.n_units == 426


@pytest.mark.parametrize("mpdta_one_infinite", [("year", float("-inf"))], indirect=True)
def test_att_gt_warns_only_about_the_rows_and_units_it_drops(mpdta_one_infinite):
    with pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$") as record:
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


@pytest.mark.parametrize("convert", [pl.DataFrame.clone, pl.DataFrame.to_pandas], ids=["polars", "pandas"])
def test_att_gt_reads_decimal_outcome_covariate_and_weights_as_doubles(mpdta_decimal_values, convert):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "pop",
        "random_state": 7,
    }
    floats = mpdta_decimal_values.with_columns(pl.col("lemp", "lpop", "pop").cast(pl.Float64))

    with warnings.catch_warnings(record=True) as expected_record:
        warnings.simplefilter("always")
        expected = att_gt(convert(floats), **spec)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = att_gt(convert(mpdta_decimal_values), **spec)

    messages = [str(warning.message) for warning in record]
    assert messages == [str(warning.message) for warning in expected_record]
    assert "Dropped 1 rows from original data due to missing values" in messages
    assert result.n_units == 499
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


def test_att_gt_keeps_decimal_unit_ids_above_two_to_the_53_apart(mpdta_large_ids):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "random_state": 7}
    decimal_ids = mpdta_large_ids.with_columns(pl.col("countyreal").cast(pl.Decimal(38, 0)))

    result = att_gt(decimal_ids, **spec)
    expected = att_gt(mpdta_large_ids, **spec)

    assert mpdta_large_ids["countyreal"].cast(pl.Float64).n_unique() < 500
    assert result.n_units == expected.n_units == 500
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att_gt, expected.att_gt)
    np.testing.assert_array_equal(result.se_gt, expected.se_gt)


@pytest.mark.parametrize("mpdta_bad_weights", ["one negative"], indirect=True)
@pytest.mark.parametrize("dtype", [pl.Float64, pl.Decimal(18, 6)], ids=["float", "decimal"])
def test_att_gt_rejects_negative_decimal_weights_like_float_weights(mpdta_bad_weights, dtype):
    data = mpdta_bad_weights.with_columns(pl.col("w").cast(dtype))
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="w")


@pytest.mark.parametrize(
    ("base_period", "cells"),
    [
        (
            "varying",
            [
                (2004, 2004, 2003, [2006, 2007]),
                (2004, 2005, 2003, [2006, 2007]),
                (2004, 2006, 2003, [2007]),
                (2006, 2004, 2003, [2007]),
                (2006, 2005, 2004, [2007]),
                (2006, 2006, 2005, [2007]),
            ],
        ),
        (
            "universal",
            [
                (2004, 2003, 2003, [2006, 2007]),
                (2004, 2004, 2003, [2006, 2007]),
                (2004, 2005, 2003, [2006, 2007]),
                (2004, 2006, 2003, [2007]),
                (2006, 2003, 2005, [2007]),
                (2006, 2004, 2005, [2007]),
                (2006, 2005, 2005, [2007]),
                (2006, 2006, 2005, [2007]),
            ],
        ),
    ],
)
def test_att_gt_without_never_treated_leaves_latest_cohort_out(
    mpdta_without_never_treated, mean_change_cells, base_period, cells
):
    pre = np.array([g > t and t != base for g, t, base, _ in cells])

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
    att, influence = mean_change_cells(mpdta_without_never_treated, cells)
    n_units = len(influence)
    se = np.sqrt(np.mean(influence**2, axis=0) / n_units)
    wald = n_units * att[pre] @ np.linalg.solve(influence[:, pre].T @ influence[:, pre] / n_units, att[pre])

    assert pre.sum() == 2
    np.testing.assert_array_equal(result.groups, [g for g, _, _, _ in cells])
    np.testing.assert_array_equal(result.times, [t for _, t, _, _ in cells])
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.where(se == 0, np.nan, se), rtol=1e-12)
    np.testing.assert_allclose(result.wald_stat, wald, rtol=1e-12)
    np.testing.assert_allclose(result.wald_pvalue, np.exp(-wald / 2), rtol=0, atol=5e-6)


def test_att_gt_without_never_treated_event_study(mpdta_without_never_treated, mean_change_cells):
    cells = [
        (2004, 2004, 2003, [2006, 2007]),
        (2004, 2005, 2003, [2006, 2007]),
        (2004, 2006, 2003, [2007]),
        (2006, 2004, 2003, [2007]),
        (2006, 2005, 2004, [2007]),
        (2006, 2006, 2005, [2007]),
    ]
    cohorts = mpdta_without_never_treated.unique("countyreal").sort("countyreal")["first.treat"].to_numpy()

    result = att_gt(
        data=mpdta_without_never_treated,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
        est_method="reg",
        random_state=7,
    )
    dynamic = aggte(result, type="dynamic")
    att, influence = mean_change_cells(mpdta_without_never_treated, cells)
    shares = np.array([np.mean(cohorts == 2004), np.mean(cohorts == 2006)])
    deviations = np.column_stack([cohorts == 2004, cohorts == 2006]) - shares
    share_influence = deviations / shares.sum() - np.outer(deviations.sum(axis=1), shares / shares.sum() ** 2)
    on_impact = influence[:, [0, 5]] @ (shares / shares.sum()) + share_influence @ att[[0, 5]]
    by_event = np.column_stack([influence[:, 3], influence[:, 4], on_impact, influence[:, 1], influence[:, 2]])

    np.testing.assert_array_equal(result.groups, [2004, 2004, 2004, 2006, 2006, 2006])
    np.testing.assert_array_equal(result.times, [2004, 2005, 2006, 2004, 2005, 2006])
    np.testing.assert_array_equal(dynamic.event_times, [-2, -1, 0, 1, 2])
    np.testing.assert_allclose(
        dynamic.att_by_event,
        [att[3], att[4], shares @ att[[0, 5]] / shares.sum(), att[1], att[2]],
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(dynamic.se_by_event, np.sqrt(np.mean(by_event**2, axis=0) / len(cohorts)), rtol=1e-12)


@pytest.mark.parametrize(
    ("control_group", "comparisons"),
    [
        ("nevertreated", [[0]] * 8),
        ("notyettreated", [[0, 2008], [0, 2008], [0, 2008], [0], [0, 2006], [0], [0], [0]]),
    ],
)
def test_att_gt_anticipation_keeps_cohort_that_starts_after_panel(
    mpdta_cohort_after_panel, mean_change_cells, control_group, comparisons
):
    groups = [2006] * 4 + [2008] * 4
    times = [2004, 2005, 2006, 2007] * 2
    cells = list(zip(groups, times, [2003, 2004, 2004, 2004, 2003, 2004, 2005, 2006], comparisons, strict=True))

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
    att, influence = mean_change_cells(mpdta_cohort_after_panel, cells)

    np.testing.assert_array_equal(result.groups, groups)
    np.testing.assert_array_equal(result.times, times)
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / len(influence)), rtol=1e-12)


def test_att_gt_leaves_anticipating_cohorts_out_of_the_comparison_on_repeated_cross_sections(
    mpdta_cohort_after_panel, period_mean_cells
):
    groups = [2006] * 4 + [2008] * 4
    times = [2004, 2005, 2006, 2007] * 2
    comparisons = [[0, 2008], [0, 2008], [0, 2008], [0], [0, 2006], [0], [0], [0]]
    cells = list(zip(groups, times, [2003, 2004, 2004, 2004, 2003, 2004, 2005, 2006], comparisons, strict=True))

    with pytest.warns(UserWarning, match="^Dropped 100 observations that were already treated in the first period$"):
        result = att_gt(
            data=mpdta_cohort_after_panel,
            yname="lemp",
            tname="year",
            gname="first.treat",
            control_group="notyettreated",
            anticipation=1,
            est_method="reg",
            panel=False,
            boot=False,
        )
    att, influence = period_mean_cells(mpdta_cohort_after_panel, cells)

    np.testing.assert_array_equal(result.groups, groups)
    np.testing.assert_array_equal(result.times, times)
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.se_gt, np.sqrt(np.mean(influence**2, axis=0) / len(influence)), rtol=1e-12)


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


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_att_gt_critical_value_skips_cells_with_a_small_bootstrap_scale(
    monkeypatch, draws_from_weights, offsetting_spike_weights, did_offsetting_cohort_data
):
    kwargs = {
        "data": did_offsetting_cohort_data,
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~1",
    }
    cells = att_gt(**kwargs)
    weights = offsetting_spike_weights(cells.n_units, *np.flatnonzero(np.abs(cells.influence_func[:, 0]) > 1e-8))
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = att_gt(**kwargs, boot=True, biters=999, cband=True)

    bres = np.sqrt(cells.n_units) * (weights @ cells.influence_func / cells.n_units)
    bres = bres[:, np.sum(bres**2, axis=0) > np.sqrt(np.finfo(float).eps) * 10]
    scale = (
        np.percentile(bres, 75, axis=0, method="inverted_cdf") - np.percentile(bres, 25, axis=0, method="inverted_cdf")
    ) / 1.3489795
    usable = scale > np.sqrt(np.finfo(float).eps) * 10
    expected = np.percentile(np.max(np.abs(bres[:, usable]) / scale[usable], axis=1), 95, method="inverted_cdf")

    assert usable.sum() == 15
    assert 2 < result.critical_value < 7
    np.testing.assert_allclose(result.critical_value, expected, rtol=1e-12)


def test_att_gt_bootstrap_gives_no_standard_error_where_its_draws_do_not_spread(
    monkeypatch, draws_from_weights, offsetting_spike_weights, did_offsetting_cohort_data
):
    kwargs = {
        "data": did_offsetting_cohort_data,
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~1",
    }
    cells = att_gt(**kwargs)
    weights = offsetting_spike_weights(cells.n_units, *np.flatnonzero(np.abs(cells.influence_func[:, 0]) > 1e-8))
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(weights))

    result = att_gt(**kwargs, boot=True, biters=999, cband=True)
    q75, q25 = np.percentile(weights @ cells.influence_func / cells.n_units, [75, 25], axis=0, method="inverted_cdf")
    scale = (q75 - q25) / 1.3489795

    assert np.all(cells.se_gt > 0.02)
    np.testing.assert_array_equal(result.groups[scale < 1.5e-7], [2, 2, 2, 2, 2])
    np.testing.assert_allclose(result.se_gt, np.where(scale < 1.5e-7, np.nan, scale), rtol=1e-12)


def test_att_gt_warns_once_when_the_critical_value_is_large(monkeypatch, mpdta_data):
    counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    monkeypatch.setattr(
        import_module("moderndid.did.att_gt"), "_mboot", lambda **kwargs: {**_mboot(**kwargs), "crit_val": 8.0}
    )

    with pytest.warns(UserWarning, match="too large") as record:
        att_gt(
            data=mpdta_data.filter(pl.col("countyreal").is_in(counties)),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            boot=True,
            biters=20,
            cband=True,
            random_state=7,
        )

    assert sum("too large" in str(warning.message) for warning in record) == 1


def test_att_gt_stays_silent_about_a_critical_value_below_seven(monkeypatch, mpdta_data):
    counties = mpdta_data["countyreal"].unique().sort()[:100].to_list()
    monkeypatch.setattr(
        import_module("moderndid.did.att_gt"),
        "_mboot",
        lambda **kwargs: {**_mboot(**kwargs), "crit_val": np.nextafter(7.0, 0.0)},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            data=mpdta_data.filter(pl.col("countyreal").is_in(counties)),
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            boot=True,
            biters=20,
            cband=True,
            random_state=7,
        )

    assert result.critical_value == np.nextafter(7.0, 0.0)
    assert not [w for w in caught if "too large" in str(w.message)]


@pytest.mark.filterwarnings("error:Overlap condition violated:UserWarning")
@pytest.mark.parametrize(
    ("est_method", "estimator", "expected"),
    [("dr", drdid_panel, [-0.878511, 1.920720]), ("ipw", std_ipw_did_panel, [-0.808225, 1.862603])],
)
def test_att_gt_overlap_screen_uses_sampling_weights(overlap_flip_panel, est_method, estimator, expected):
    result = att_gt(
        overlap_flip_panel,
        yname="y",
        tname="t",
        idname="id",
        gname="g",
        xformla="~x",
        weightsname="w",
        est_method=est_method,
    )

    direct = []
    for t in (2, 3):
        post = overlap_flip_panel.filter(pl.col("t") == t)
        pre = overlap_flip_panel.filter(pl.col("t") == t - 1)
        covariates = np.column_stack([np.ones(post.height), pre["x"].to_numpy()])
        treated = (post["g"] == 3).to_numpy().astype(float)
        direct.append(
            estimator(post["y"].to_numpy(), pre["y"].to_numpy(), treated, covariates, pre["w"].to_numpy()).att
        )

    np.testing.assert_array_equal(result.groups, [3, 3])
    np.testing.assert_array_equal(result.times, [2, 3])
    np.testing.assert_allclose(result.att_gt, direct, rtol=1e-10)
    np.testing.assert_allclose(result.att_gt, expected, atol=1e-6)


def test_att_gt_overlap_screen_without_weights_skips(overlap_flip_panel):
    with pytest.warns(UserWarning, match="^Overlap condition violated for 3 in time period 3$"):
        result = att_gt(overlap_flip_panel, yname="y", tname="t", idname="id", gname="g", xformla="~x")

    assert result.att_gt.size == 0


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
def test_att_gt_overlap_screen_skips_cells_the_weights_leave_without_overlap(thin_weighted_controls_panel, est_method):
    with pytest.warns(UserWarning, match="^Overlap condition violated for 3 in time period 3$"):
        result = att_gt(
            thin_weighted_controls_panel,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            weightsname="w",
            est_method=est_method,
        )

    assert result.att_gt.size == 0


def test_att_gt_overlap_screen_rescales_tiny_weights(tiny_weight_overlap_panel):
    with pytest.warns(UserWarning, match="^Overlap condition violated for 3 in time period 4$") as record:
        result = att_gt(
            tiny_weight_overlap_panel,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            xformla="~x",
            weightsname="w",
            est_method="ipw",
        )

    flagged = {str(warning.message) for warning in record if str(warning.message).startswith("Overlap")}
    assert flagged == {f"Overlap condition violated for {g} in time period {t}" for g in (3, 4) for t in (2, 3, 4)}
    assert result.att_gt.size == 0


def test_att_gt_regression_screen_uses_sampling_weights(zero_weight_controls_panel):
    with pytest.warns(
        UserWarning, match="^Not enough control units for group 3 in time period 3 to run specified regression$"
    ):
        result = att_gt(
            zero_weight_controls_panel,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            xformla="~x",
            weightsname="w",
            est_method="reg",
        )

    assert result.att_gt.size == 0


@pytest.mark.parametrize("panel", [True, False])
def test_att_gt_drops_comparison_without_estimate(tiny_late_cohort_panel, panel):
    with pytest.warns(
        UserWarning, match="^The 2x2 comparison for group 3 in time period 4 failed\\. The estimator returned NaN"
    ):
        result = att_gt(
            tiny_late_cohort_panel,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            control_group="notyettreated",
            est_method="ipw",
            panel=panel,
        )

    assert list(zip(result.groups.tolist(), result.times.tolist(), strict=True)) == [
        (3, 2),
        (3, 3),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert np.all(np.isfinite(result.att_gt))


@pytest.mark.parametrize("est_method", ["reg", "ipw", "dr"])
def test_att_gt_drops_cohort_without_weight(zero_weight_cohort_panel, est_method):
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "weightsname": "w", "est_method": est_method}

    with pytest.warns(
        UserWarning,
        match="^The 2x2 comparison for group 3 in time period 2 failed\\. The treated units all have zero weight\\.$",
    ):
        result = att_gt(zero_weight_cohort_panel, **spec)
    reference = att_gt(zero_weight_cohort_panel.filter(pl.col("g") != 3), **spec)

    assert 3 not in result.groups
    np.testing.assert_array_equal(result.groups, reference.groups)
    np.testing.assert_array_equal(result.times, reference.times)
    np.testing.assert_allclose(result.att_gt, reference.att_gt, rtol=1e-12)
    np.testing.assert_allclose(result.se_gt, reference.se_gt, rtol=1e-12)


def test_att_gt_drops_cells_of_custom_estimator_returning_nan(zero_weight_cohort_panel):
    def nan_estimator(y1, y0, d, covariates, i_weights=None, influence_func=True):
        return drdid_panel(y1, y0, d, covariates, i_weights=i_weights, influence_func=True)._replace(att=np.nan)

    with pytest.warns(
        UserWarning, match="^The 2x2 comparison for group 4 in time period 5 failed\\. The estimator returned NaN"
    ):
        result = att_gt(
            zero_weight_cohort_panel.filter(pl.col("g") != 3),
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            est_method=nan_estimator,
        )

    assert result.att_gt.size == 0


def test_att_gt_drops_cells_of_custom_estimator_without_influence_function(zero_weight_cohort_panel):
    def estimator_without_influence_function(y1, y0, d, covariates, i_weights=None, influence_func=True):
        result = drdid_panel(y1, y0, d, covariates, i_weights=i_weights, influence_func=True)
        return result._replace(att_inf_func=None)

    with pytest.warns(
        UserWarning,
        match=(
            "^The 2x2 comparison for group 4 in time period 5 failed\\. "
            "The estimator returned no influence function\\.$"
        ),
    ):
        result = att_gt(
            zero_weight_cohort_panel.filter(pl.col("g") != 3),
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            est_method=estimator_without_influence_function,
        )

    assert result.att_gt.size == 0


def test_att_gt_records_a_custom_estimator_as_custom(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "boot": False}

    custom = att_gt(mpdta_data, est_method=drdid_panel, **spec)
    built_in = att_gt(mpdta_data, est_method="dr", **spec)

    assert custom.estimation_params["estimation_method"] == "custom"
    assert " Estimation Method:  custom" in str(custom).splitlines()
    np.testing.assert_array_equal(custom.att_gt, built_in.att_gt)
    np.testing.assert_array_equal(custom.se_gt, built_in.se_gt)


def test_att_gt_drops_cells_whose_comparison_units_have_no_weight(zero_weight_never_treated_panel, mean_change_cells):
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "weightsname": "w",
        "control_group": "notyettreated",
    }
    cells = [(3, 2, 1, [0, 4]), (3, 3, 2, [0, 4]), (4, 2, 1, [0, 3])]

    with pytest.warns(
        UserWarning,
        match=(
            "^The 2x2 comparison for group 4 in time period 3 failed\\. The comparison units all have zero weight\\.$"
        ),
    ):
        result = att_gt(zero_weight_never_treated_panel, **spec, est_method=drdid_panel)
    with pytest.warns(UserWarning, match="^Overlap condition violated for 4 in time period 3$"):
        screened = att_gt(zero_weight_never_treated_panel, **spec, est_method="dr")
    att, _ = mean_change_cells(
        zero_weight_never_treated_panel, cells, weightsname="w", yname="y", tname="t", idname="id", gname="g"
    )

    assert list(zip(result.groups.tolist(), result.times.tolist(), strict=True)) == [(3, 2), (3, 3), (4, 2)]
    np.testing.assert_allclose(result.att_gt, att, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(result.groups, screened.groups)
    np.testing.assert_array_equal(result.times, screened.times)
    np.testing.assert_array_equal(result.att_gt, screened.att_gt)
    np.testing.assert_array_equal(result.se_gt, screened.se_gt)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_names_the_period_whose_treated_rows_have_no_weight(zero_weight_treated_period_rcs, est_method):
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "xformla": "~x", "panel": False}

    with pytest.warns(
        UserWarning,
        match=(
            "^The 2x2 comparison for group 3 in time period 3 failed\\. "
            "The treated units in time period 3 all have zero weight\\.$"
        ),
    ):
        result = att_gt(zero_weight_treated_period_rcs, **spec, weightsname="w", est_method=est_method)
    unweighted = att_gt(zero_weight_treated_period_rcs, **spec, est_method=est_method)

    assert result.groups.tolist() == [3]
    assert result.times.tolist() == [2]
    assert unweighted.times.tolist() == [2, 3]
    np.testing.assert_allclose(result.att_gt, unweighted.att_gt[:1], rtol=1e-12)
    np.testing.assert_allclose(result.se_gt, unweighted.se_gt[:1], rtol=1e-12)


def test_att_gt_failure_warning_names_the_cell(tiny_late_cohort_panel):
    with pytest.warns(
        UserWarning,
        match=(
            "^The 2x2 comparison for group 3 in time period 4 failed\\. No effectively control units after trimming\\."
        ),
    ):
        result = att_gt(
            tiny_late_cohort_panel,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            control_group="notyettreated",
            est_method="dr",
        )

    cells = set(zip(result.groups.tolist(), result.times.tolist(), strict=True))
    assert (3, 3) in cells
    assert (3, 4) not in cells


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("est_method", "est_method=None is not valid. Must be 'dr', 'ipw', or 'reg'."),
        ("control_group", "control_group=None is not valid. Must be 'nevertreated' or 'notyettreated'."),
        ("base_period", "base_period=None is not valid. Must be 'varying' or 'universal'."),
    ],
)
def test_att_gt_rejects_options_that_are_not_strings(mpdta_data, option, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **{option: None})


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"alpha": 1}, "alpha=1 is not valid. Must be between 0 and 1 (exclusive)."),
        ({"biters": 0}, "biters=0 is not valid. Must be a positive integer."),
        ({"biters": 99.5}, "biters=99.5 is not valid. Must be a positive integer."),
        ({"anticipation": -1}, "anticipation=-1 is not valid. Must be a non-negative number."),
        ({"n_jobs": 0}, "n_jobs=0 is not valid. Must be a positive integer or -1 for all cores."),
        ({"n_jobs": 0, "backend": "numpy"}, "n_jobs=0 is not valid. Must be a positive integer or -1 for all cores."),
        ({"gname": None}, "gname is required. Please specify the treatment group column."),
        ({"idname": None}, "idname must be provided when panel=True."),
    ],
)
def test_att_gt_rejects_invalid_arguments(mpdta_data, option, message):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"} | option

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(mpdta_data, **spec)


def test_att_gt_with_all_cores_matches_the_sequential_fit(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "est_method": "reg"}

    parallel = att_gt(mpdta_data, n_jobs=-1, **spec)
    sequential = att_gt(mpdta_data, **spec)

    np.testing.assert_array_equal(parallel.groups, sequential.groups)
    np.testing.assert_array_equal(parallel.times, sequential.times)
    np.testing.assert_array_equal(parallel.att_gt, sequential.att_gt)
    np.testing.assert_array_equal(parallel.se_gt, sequential.se_gt)


def test_att_gt_rejects_a_string_of_cluster_variables(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}
    message = "clustervars must be a list of strings, not a string. Use clustervars=['countyreal']."

    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        att_gt(mpdta_data, clustervars="countyreal", **spec)


@pytest.mark.parametrize("est_method", ["reg", "ipw", "dr"])
def test_att_gt_matches_independent_fits_on_twenty_weighted_covariates(weighted_wide_panel, logit_scores, est_method):
    covariates = [f"cov{j}" for j in range(1, 21)]
    units = weighted_wide_panel.filter(pl.col("time") == 1).sort("id")
    outcomes = weighted_wide_panel.pivot(on="time", index="id", values="y").sort("id")
    design = np.column_stack([np.ones(units.height), units.select(covariates).to_numpy()])
    cohorts = units["group"].to_numpy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            weighted_wide_panel,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            xformla="~" + " + ".join(covariates),
            weightsname="w",
            control_group="notyettreated",
            est_method=est_method,
        )
    expected = []
    scores = []
    for group, time in zip(result.groups, result.times, strict=True):
        base = group - 1 if time >= group else time - 1
        cell = (cohorts == group) | (cohorts == 0) | (cohorts > max(time, base))
        treated = cohorts[cell] == group
        change = (outcomes[str(int(time))] - outcomes[str(int(base))]).to_numpy()[cell]
        weights = units["w"].to_numpy()[cell]
        x = design[cell]
        root = np.sqrt(weights[~treated])
        coefficients = np.linalg.lstsq(x[~treated] * root[:, None], change[~treated] * root, rcond=None)[0]
        residual = change if est_method == "ipw" else change - x @ coefficients
        scores.append(logit_scores(x, treated, weights))
        odds = scores[-1] / (1 - scores[-1])
        control = 0.0 if est_method == "reg" else np.average(residual[~treated], weights=(weights * odds)[~treated])
        expected.append(np.average(residual[treated], weights=weights[treated]) - control)

    assert list(zip(result.groups.tolist(), result.times.tolist(), strict=True)) == [(2, 2), (2, 3), (3, 2), (3, 3)]
    assert [str(warning.message) for warning in caught] == []
    assert 1e-6 < np.concatenate(scores).min() < np.concatenate(scores).max() < 0.995
    np.testing.assert_allclose(result.att_gt, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("est_method", ["reg", "ipw"])
@pytest.mark.parametrize(
    "layout",
    [{"panel": False}, {"idname": "id", "allow_unbalanced_panel": True}],
    ids=["repeated_cross_sections", "unbalanced_panel"],
)
def test_att_gt_rows_match_independent_fits_on_twenty_weighted_covariates(
    weighted_wide_panel, logit_scores, est_method, layout
):
    data = weighted_wide_panel.filter(~((pl.col("id") % 7 == 0) & (pl.col("time") == 2)))
    covariates = [f"cov{j}" for j in range(1, 21)]
    design = np.column_stack([np.ones(data.height), data.select(covariates).to_numpy()])
    cohorts = data["group"].to_numpy()
    periods = data["time"].to_numpy()
    outcome = data["y"].to_numpy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            data,
            yname="y",
            tname="time",
            gname="group",
            xformla="~" + " + ".join(covariates),
            weightsname="w",
            control_group="notyettreated",
            est_method=est_method,
            **layout,
        )
    expected = []
    scores = []
    for group, time in zip(result.groups, result.times, strict=True):
        base = group - 1 if time >= group else time - 1
        rows = np.isin(periods, [time, base]) & ((cohorts == group) | (cohorts == 0) | (cohorts > max(time, base)))
        treated = cohorts[rows] == group
        post = periods[rows] == time
        x, y, weights = design[rows], outcome[rows], data["w"].to_numpy()[rows]
        scores.append(logit_scores(x, treated, weights))
        if est_method == "reg":
            fitted = []
            for period in (post, ~post):
                comparison = ~treated & period
                root = np.sqrt(weights[comparison])
                fitted.append(x @ np.linalg.lstsq(x[comparison] * root[:, None], y[comparison] * root, rcond=None)[0])
            control = np.average((fitted[0] - fitted[1])[treated], weights=weights[treated])
        else:
            odds = weights * scores[-1] / (1 - scores[-1])
            after = np.average(y[~treated & post], weights=odds[~treated & post])
            control = after - np.average(y[~treated & ~post], weights=odds[~treated & ~post])
        expected.append(
            np.average(y[treated & post], weights=weights[treated & post])
            - np.average(y[treated & ~post], weights=weights[treated & ~post])
            - control
        )

    assert list(zip(result.groups.tolist(), result.times.tolist(), strict=True)) == [(2, 2), (2, 3), (3, 2), (3, 3)]
    assert [str(warning.message) for warning in caught] == []
    assert 1e-6 < np.concatenate(scores).min() < np.concatenate(scores).max() < 0.995
    np.testing.assert_allclose(result.att_gt, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize(
    ("est_method", "change"),
    [
        ("reg", pl.col("cov5") * 1e4),
        ("ipw", pl.col("cov5") * 1e4),
        ("dr", pl.col("cov5") * 1e4),
        ("reg", pl.col("cov5") + 1e3),
        ("ipw", pl.col("cov5") + 1e3),
        ("dr", pl.col("cov5") + 1e3),
        ("ipw", pl.col("cov5") * 1e12),
    ],
    ids=["reg-x1e4", "ipw-x1e4", "dr-x1e4", "reg-plus1e3", "ipw-plus1e3", "dr-plus1e3", "ipw-x1e12"],
)
def test_att_gt_estimates_stay_put_when_one_of_twenty_covariates_is_rescaled_or_shifted(
    weighted_wide_panel, est_method, change
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(f"cov{j}" for j in range(1, 21)),
        "weightsname": "w",
        "est_method": est_method,
    }

    original = att_gt(weighted_wide_panel, **spec)
    changed = att_gt(weighted_wide_panel.with_columns(change), **spec)

    assert original.att_gt.size == 4
    assert np.isfinite(original.att_gt).all()
    assert np.isfinite(original.se_gt).all()
    np.testing.assert_array_equal(changed.groups, original.groups)
    np.testing.assert_array_equal(changed.times, original.times)
    np.testing.assert_allclose(changed.att_gt, original.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(changed.se_gt, original.se_gt, rtol=1e-9, atol=0)


@pytest.mark.parametrize("est_method", ["reg", "dr"])
def test_att_gt_regression_screen_skips_each_cell_when_a_covariate_repeats_another(weighted_wide_panel, est_method):
    covariates = " + ".join(f"cov{j}" for j in range(1, 21))
    cells = [(2, 2), (2, 3), (3, 2), (3, 3)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            weighted_wide_panel.with_columns(pl.col("cov7").alias("copy")),
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            xformla=f"~ {covariates} + copy",
            weightsname="w",
            est_method=est_method,
        )

    messages = [str(warning.message) for warning in caught]
    assert messages[:-1] == [
        f"Not enough control units for group {group} in time period {time} to run specified regression"
        for group, time in cells
    ]
    assert messages[-1].startswith("No pre-treatment periods available for the Wald pre-test")
    assert result.att_gt.size == 0


@pytest.mark.parametrize(
    ("est_method", "copy"),
    [
        ("reg", pl.col("cov3") - 2 * pl.col("cov11")),
        ("reg", pl.lit(2.5)),
        ("dr", pl.col("cov3") - 2 * pl.col("cov11")),
    ],
    ids=["reg-combination", "reg-constant", "dr-combination"],
)
def test_att_gt_skips_each_cell_with_a_warning_that_names_it_when_a_covariate_combines_others(
    weighted_wide_panel, est_method, copy
):
    covariates = " + ".join(f"cov{j}" for j in range(1, 21))
    cells = [(2, 2), (2, 3), (3, 2), (3, 3)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            weighted_wide_panel.with_columns(copy.alias("copy")),
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            xformla=f"~ {covariates} + copy",
            weightsname="w",
            est_method=est_method,
        )

    messages = [str(warning.message) for warning in caught]
    named = [
        (group, time)
        for group, time in cells
        if any(f"group {group} in time period {time}" in message for message in messages)
    ]
    assert result.att_gt.size == 0
    assert named == cells


def test_att_gt_rejects_never_treated_group_smaller_than_covariate_count_plus_five(weighted_wide_panel):
    never_treated = weighted_wide_panel.filter(pl.col("group") == 0)["id"].unique().sort()
    accepted = weighted_wide_panel.filter((pl.col("group") != 0) | pl.col("id").is_in(never_treated.head(25).implode()))
    rejected = weighted_wide_panel.filter((pl.col("group") != 0) | pl.col("id").is_in(never_treated.head(24).implode()))
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(f"cov{j}" for j in range(1, 21)),
        "weightsname": "w",
    }
    message = (
        "The never-treated group is too small to serve as the comparison group. "
        "Pass control_group='notyettreated' to compare with units not yet treated."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        att_gt(accepted, **spec)
    with (
        pytest.warns(UserWarning, match="^Be aware that there are some small groups in your dataset\\."),
        pytest.raises(ValueError, match=f"^{re.escape(message)}$"),
    ):
        att_gt(rejected, **spec)

    messages = [str(warning.message) for warning in caught]
    assert [text for text in messages if text.startswith("Be aware that there are some small groups")] == []


@pytest.mark.parametrize("est_method", ["ipw", "dr"])
def test_att_gt_trims_comparison_units_whose_scores_pass_0995_with_twenty_covariates(
    wide_trimming_panel, logit_scores, est_method
):
    covariates = [f"x{j}" for j in range(20)]
    units = wide_trimming_panel.filter(pl.col("t") == 1).sort("id")
    outcomes = wide_trimming_panel.pivot(on="t", index="id", values="y").sort("id")
    design = np.column_stack([np.ones(units.height), units.select(covariates).to_numpy()])
    treated = units["g"].to_numpy() == 3
    weights = units["w"].to_numpy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            wide_trimming_panel,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            xformla="~" + " + ".join(covariates),
            weightsname="w",
            est_method=est_method,
        )
    scores = logit_scores(design, treated, weights)
    kept = np.where(treated | (scores < 0.995), 1.0, 0.0)
    expected = []
    for time in (2, 3):
        change = (outcomes[str(time)] - outcomes[str(time - 1)]).to_numpy()
        root = np.sqrt(weights[~treated])
        coefficients = np.linalg.lstsq(design[~treated] * root[:, None], change[~treated] * root, rcond=None)[0]
        residual = change if est_method == "ipw" else change - design @ coefficients
        control_weights = (kept * weights * scores / (1 - scores))[~treated]
        expected.append(
            np.average(residual[treated], weights=weights[treated])
            - np.average(residual[~treated], weights=control_weights)
        )

    assert np.flatnonzero(~treated & (scores > 0.99)).tolist() == [6, 13]
    assert scores[13] < 0.995
    assert 0.9988 < scores[6] == scores.max() < 0.999
    assert [str(warning.message) for warning in caught] == []
    np.testing.assert_array_equal(result.times, [2, 3])
    np.testing.assert_allclose(result.att_gt, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("est_method", ["ipw", "dr"])
def test_att_gt_skips_cells_whose_scores_reach_0999_with_twenty_covariates(
    wide_trimming_panel, logit_scores, est_method
):
    data = wide_trimming_panel.with_columns(pl.when(pl.col("id") < 6).then(250.0).otherwise(pl.col("w")).alias("w"))
    covariates = [f"x{j}" for j in range(20)]
    units = data.filter(pl.col("t") == 1).sort("id")
    design = np.column_stack([np.ones(units.height), units.select(covariates).to_numpy()])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = att_gt(
            data,
            yname="y",
            tname="t",
            idname="id",
            gname="g",
            xformla="~" + " + ".join(covariates),
            weightsname="w",
            est_method=est_method,
        )
    scores = logit_scores(design, units["g"].to_numpy() == 3, units["w"].to_numpy())

    messages = [str(warning.message) for warning in caught]
    assert 0.999 < scores.max() < 0.9995
    assert messages[:-1] == [
        "Overlap condition violated for 3 in time period 2",
        "Overlap condition violated for 3 in time period 3",
    ]
    assert messages[-1].startswith("No pre-treatment periods available for the Wald pre-test")
    assert result.att_gt.size == 0
