"""Tests for the ETWFE estimator."""

import re
import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")
importorskip("pyfixest")

from moderndid import emfx, etwfe
from moderndid.etwfe.container import EtwfeResult


def test_etwfe_returns_etwfe_result(etwfe_baseline):
    assert isinstance(etwfe_baseline, EtwfeResult)


def test_etwfe_baseline_gt_pairs(etwfe_baseline):
    expected_pairs = [
        (2004.0, 2004.0),
        (2004.0, 2005.0),
        (2004.0, 2006.0),
        (2006.0, 2006.0),
        (2004.0, 2007.0),
        (2006.0, 2007.0),
        (2007.0, 2007.0),
    ]
    assert etwfe_baseline.gt_pairs == expected_pairs


def test_etwfe_baseline_coefficients(etwfe_baseline):
    expected = np.array([-0.019372, -0.078319, -0.136078, 0.002514, -0.104707, -0.039193, -0.043106])
    np.testing.assert_allclose(etwfe_baseline.coefficients, expected, atol=1e-4)


def test_etwfe_baseline_standard_errors(etwfe_baseline):
    expected = np.array([0.022382, 0.030488, 0.035455, 0.019933, 0.033874, 0.024009, 0.018431])
    np.testing.assert_allclose(etwfe_baseline.std_errors, expected, atol=1e-6)


def test_etwfe_default_vcov_clusters_by_idname(etwfe_baseline, mpdta_data):
    explicit = etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        vcov={"CRV1": "countyreal"},
    )
    np.testing.assert_array_equal(etwfe_baseline.std_errors, explicit.std_errors)
    assert etwfe_baseline.estimation_params["vcov_type"] == "CRV1"
    assert etwfe_baseline.estimation_params["clustervar"] == "countyreal"


def test_etwfe_default_vcov_hetero_without_idname(mpdta_data):
    data = mpdta_data.rename({"first.treat": "first_treat"})
    mod = etwfe(data=data, yname="lemp", tname="year", gname="first_treat")
    assert mod.estimation_params["vcov_type"] == "hetero"
    assert mod.estimation_params["clustervar"] is None


def test_etwfe_baseline_obs_counts(etwfe_baseline):
    assert etwfe_baseline.n_obs == 2500
    assert etwfe_baseline.n_units == 500


def test_etwfe_r_squared(etwfe_baseline):
    assert etwfe_baseline.r_squared is not None
    assert 0 < etwfe_baseline.r_squared <= 1
    np.testing.assert_allclose(etwfe_baseline.r_squared, 0.9933, atol=1e-3)


def test_etwfe_vcov_symmetric(etwfe_baseline):
    assert np.allclose(etwfe_baseline.vcov, etwfe_baseline.vcov.T)


def test_etwfe_vcov_diagonal_nonnegative(etwfe_baseline):
    assert np.all(np.diag(etwfe_baseline.vcov) >= 0)


def test_etwfe_se_equals_sqrt_diag_vcov(etwfe_baseline):
    se_from_vcov = np.sqrt(np.diag(etwfe_baseline.vcov))
    np.testing.assert_allclose(etwfe_baseline.std_errors, se_from_vcov, rtol=1e-6)


def test_etwfe_vcov_shape(etwfe_baseline):
    n = len(etwfe_baseline.coefficients)
    assert etwfe_baseline.vcov.shape == (n, n)


@pytest.mark.parametrize("cgroup", ["notyet", "never"])
def test_etwfe_estimation_params_cgroup(mpdta_data, cgroup):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", cgroup=cgroup)
    assert mod.estimation_params["cgroup"] == cgroup


def test_etwfe_never_has_more_gt_pairs(etwfe_baseline, etwfe_never):
    assert len(etwfe_never.gt_pairs) > len(etwfe_baseline.gt_pairs)


def test_etwfe_never_includes_pretreatment(etwfe_never):
    pre_pairs = [(g, t) for g, t in etwfe_never.gt_pairs if t < g]
    assert len(pre_pairs) > 0


def test_etwfe_never_coefficients(etwfe_never):
    expected = np.array(
        [
            -0.003769,
            0.003306,
            -0.010503,
            0.002751,
            0.033813,
            -0.070423,
            0.031087,
            -0.137259,
            -0.004595,
            -0.100811,
            -0.041224,
            -0.026054,
        ]
    )
    np.testing.assert_allclose(etwfe_never.coefficients, expected, atol=1e-4)


def test_etwfe_feo_matches_vs_without_covariates(mpdta_data):
    mod_vs = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", fe="vs")
    mod_feo = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", fe="feo")
    np.testing.assert_allclose(mod_vs.coefficients, mod_feo.coefficients, atol=1e-10)


@pytest.mark.parametrize("fe", ["vs", "feo", "none"])
def test_etwfe_fe_param_stored(mpdta_data, fe):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", fe=fe)
    assert mod.estimation_params["fe"] == fe


def test_etwfe_fe_none_no_fe_spec(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", fe="none")
    assert mod.estimation_params["fe_spec"] is None
    assert len(mod.coefficients) == 7
    assert len(mod.coef_names) > 7


def test_etwfe_fe_spec_in_params(etwfe_baseline):
    assert etwfe_baseline.estimation_params["fe_spec"] == "countyreal + year"


def test_etwfe_without_idname(mpdta_data):
    data = mpdta_data.rename({"first.treat": "first_treat"})
    mod = etwfe(data=data, yname="lemp", tname="year", gname="first_treat")
    assert isinstance(mod, EtwfeResult)
    assert mod.estimation_params["idname"] is None
    assert mod.n_units == mod.n_obs


def test_etwfe_covariates_same_gt_atts(etwfe_baseline, etwfe_with_covariates):
    assert len(etwfe_baseline.gt_pairs) == len(etwfe_with_covariates.gt_pairs)
    for i, (g, t) in enumerate(etwfe_baseline.gt_pairs):
        for j, (g2, t2) in enumerate(etwfe_with_covariates.gt_pairs):
            if abs(g - g2) < 1e-6 and abs(t - t2) < 1e-6:
                np.testing.assert_allclose(
                    etwfe_with_covariates.coefficients[j],
                    etwfe_baseline.coefficients[i],
                    atol=0.1,
                )
                break


def test_etwfe_covariates_more_coefficients(etwfe_baseline, etwfe_with_covariates):
    assert len(etwfe_with_covariates.coef_names) > len(etwfe_baseline.coef_names)
    assert len(etwfe_with_covariates.coefficients) == len(etwfe_baseline.coefficients)


def test_etwfe_covariates_different_se(etwfe_baseline, etwfe_with_covariates):
    simple_no_cov = emfx(etwfe_baseline, type="simple")
    simple_cov = emfx(etwfe_with_covariates, type="simple")
    np.testing.assert_allclose(simple_cov.overall_att, simple_no_cov.overall_att, atol=0.01)
    assert simple_cov.overall_se != simple_no_cov.overall_se


def test_etwfe_explicit_tref_gref(mpdta_data):
    mod = etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        tref=2003,
        gref=0,
    )
    assert isinstance(mod, EtwfeResult)
    assert len(mod.gt_pairs) == 7


def test_etwfe_with_weights(mpdta_data):
    rng = np.random.default_rng(42)
    mpdta_data = mpdta_data.with_columns(pl.Series("w", rng.uniform(0.5, 1.5, len(mpdta_data))))
    mod = etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        weightsname="w",
    )
    assert isinstance(mod, EtwfeResult)
    assert mod.n_obs == 2500


@pytest.mark.parametrize("family", [None, "gaussian"])
def test_etwfe_gaussian_family(mpdta_data, family):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", family=family)
    assert isinstance(mod, EtwfeResult)
    assert mod.estimation_params["family"] == family


def test_etwfe_deterministic(mpdta_data):
    kwargs = dict(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal")
    mod1 = etwfe(**kwargs)
    mod2 = etwfe(**kwargs)
    np.testing.assert_array_equal(mod1.coefficients, mod2.coefficients)
    np.testing.assert_array_equal(mod1.std_errors, mod2.std_errors)


def test_etwfe_custom_alpha(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", alp=0.10)
    assert mod.estimation_params["alpha"] == 0.10


@pytest.mark.parametrize("vcov,expected_type", [("iid", "iid"), ("hetero", "hetero")])
def test_etwfe_vcov_type(mpdta_data, vcov, expected_type):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", vcov=vcov)
    assert mod.estimation_params["vcov_type"] == expected_type


def test_etwfe_cluster_vcov(mpdta_data):
    mod = etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        vcov={"CRV1": "first.treat"},
    )
    assert mod.estimation_params["vcov_type"] == "CRV1"
    assert mod.estimation_params["clustervar"] == "first.treat"


def test_etwfe_different_vcov_different_se(mpdta_data):
    kwargs = dict(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal")
    mod_hetero = etwfe(**kwargs, vcov="hetero")
    mod_iid = etwfe(**kwargs, vcov="iid")
    assert not np.allclose(mod_hetero.std_errors, mod_iid.std_errors)


def test_etwfe_nonlinear_keeps_idname_for_clustering(etwfe_poisson_id):
    assert etwfe_poisson_id.estimation_params["family"] == "poisson"
    assert etwfe_poisson_id.estimation_params["fe"] == "none"
    assert etwfe_poisson_id.estimation_params["idname"] == "countyreal"
    assert etwfe_poisson_id.estimation_params["vcov_type"] == "CRV1"
    assert etwfe_poisson_id.estimation_params["clustervar"] == "countyreal"
    assert etwfe_poisson_id.n_units == 500


def test_etwfe_nonlinear_with_idname_no_warning(mpdta_data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", family="poisson")
    assert [str(w.message) for w in caught] == []


def test_etwfe_poisson_default_matches_explicit_cluster(etwfe_poisson_id, mpdta_data):
    explicit = etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        family="poisson",
        vcov={"CRV1": "countyreal"},
    )
    np.testing.assert_allclose(etwfe_poisson_id.coefficients, explicit.coefficients, rtol=1e-12)
    np.testing.assert_allclose(etwfe_poisson_id.std_errors, explicit.std_errors, rtol=1e-12)


def test_etwfe_poisson_without_idname_no_warning(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", family="poisson")
    assert mod.estimation_params["family"] == "poisson"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"family": "invalid"}, "family must be"),
        ({"cgroup": "invalid"}, "cgroup must be"),
        ({"fe": "invalid"}, "fe must be"),
    ],
)
def test_etwfe_invalid_param(mpdta_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"yname": "nonexistent", "tname": "year", "gname": "first.treat"}, "yname"),
        ({"yname": "lemp", "tname": "nonexistent", "gname": "first.treat"}, "tname"),
        ({"yname": "lemp", "tname": "year", "gname": "nonexistent"}, "gname"),
        ({"yname": "lemp", "tname": "year", "gname": "first.treat", "idname": "nonexistent"}, "idname"),
        ({"yname": "lemp", "tname": "year", "gname": "first.treat", "weightsname": "nonexistent"}, "weightsname"),
        ({"yname": "lemp", "tname": "year", "gname": "first.treat", "xvar": "nonexistent"}, "xvar"),
        ({"yname": "lemp", "tname": "year", "gname": "first.treat", "xformla": "~ nonexistent"}, "xformla"),
    ],
)
def test_etwfe_missing_column(mpdta_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        etwfe(data=mpdta_data, **kwargs)


def test_etwfe_xvar_heterogeneous_effects(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", xvar="lpop")
    assert isinstance(mod, EtwfeResult)
    assert len(mod.coefficients) == 7
    assert len(mod.gt_pairs) == 7
    assert len(mod.coef_names) > 7
    s = emfx(mod, type="simple")
    np.testing.assert_allclose(s.overall_att, -0.0477099182784533, atol=1e-10)
    np.testing.assert_allclose(s.overall_se, 0.012734, atol=1e-6)


@pytest.mark.parametrize("xvar,recoded", [("gls", "notgls"), ("gls", "gls01"), ("lpop", "lpop_aff")])
def test_etwfe_xvar_invariant_to_coding(mpdta_moderators, etwfe_baseline, xvar, recoded):
    kwargs = dict(data=mpdta_moderators, yname="lemp", tname="year", gname="first.treat", idname="countyreal")
    mod = etwfe(**kwargs, xvar=xvar)
    mod_recoded = etwfe(**kwargs, xvar=recoded)
    np.testing.assert_allclose(mod.coefficients, mod_recoded.coefficients, atol=1e-10)
    np.testing.assert_allclose(mod.coefficients, etwfe_baseline.coefficients, atol=1e-10)


@pytest.mark.parametrize("control,xvar", [("lpop", "lpop"), ("lpop", "lpop_aff"), ("x_tv", "x_tv")])
def test_etwfe_xvar_spanned_by_controls_warns_and_adds_no_terms(mpdta_unbalanced_moderators, control, xvar):
    kwargs = dict(
        data=mpdta_unbalanced_moderators,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla=f"~ {control}",
    )
    controls_only = etwfe(**kwargs)
    with pytest.warns(UserWarning, match="adds no terms"):
        mod = etwfe(**kwargs, xvar=xvar)
    np.testing.assert_allclose(mod.coefficients, controls_only.coefficients, atol=1e-12)
    np.testing.assert_allclose(mod.std_errors, controls_only.std_errors, atol=1e-12)
    assert mod.coef_names == controls_only.coef_names


def test_etwfe_string_xvar_invariant_to_labels(mpdta_moderators):
    kwargs = dict(
        data=mpdta_moderators,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ lpop",
    )
    mod = etwfe(**kwargs, xvar="popcat")
    relabeled = etwfe(**kwargs, xvar="popcat2")
    assert len([name for name in mod.coef_names if name.endswith("_xdm")]) > 0
    assert len(mod.coef_names) == len(relabeled.coef_names)
    np.testing.assert_allclose(mod.coefficients, relabeled.coefficients, atol=1e-10)
    np.testing.assert_allclose(mod.std_errors, relabeled.std_errors, atol=1e-10)


@pytest.mark.parametrize(
    "family,expected",
    [
        (
            "logit",
            {
                (2004.0, 2004.0): 0.0257090351884944,
                (2004.0, 2005.0): 1.92886265668132e-15,
                (2004.0, 2006.0): -0.000589575972504749,
                (2004.0, 2007.0): -0.0458502926257386,
                (2006.0, 2006.0): -0.138407679153082,
                (2006.0, 2007.0): -0.0544566643268029,
                (2007.0, 2007.0): -0.151495324092323,
            },
        ),
        (
            "probit",
            {
                (2004.0, 2004.0): 0.0162719285321463,
                (2004.0, 2005.0): -0.000315367898437441,
                (2004.0, 2006.0): -0.000422339775140922,
                (2004.0, 2007.0): -0.0287365131500903,
                (2006.0, 2006.0): -0.0825555947480002,
                (2006.0, 2007.0): -0.0341401439693618,
                (2007.0, 2007.0): -0.09487013416585,
            },
        ),
    ],
)
def test_etwfe_binary_family_cells(mpdta_moderators, family, expected):
    mod = etwfe(data=mpdta_moderators, yname="ybin", tname="year", gname="first.treat", family=family, vcov="hetero")
    assert sorted(mod.gt_pairs) == sorted(expected)
    for cell, coef in zip(mod.gt_pairs, mod.coefficients, strict=True):
        np.testing.assert_allclose(coef, expected[cell], atol=1e-7)


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_etwfe_binary_family_runs_with_cohort_constant_moderator(mpdta_moderators, family):
    mod = etwfe(
        data=mpdta_moderators,
        yname="ybin",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        family=family,
        xvar="gls01",
    )
    assert np.all(np.isfinite(mod.coefficients))
    assert not [name for name in mod.coef_names if "cell_2004" in name and name.endswith("gls01_xdm")]


@pytest.mark.parametrize(
    "xformla,match",
    [
        ("~ gls01", "Since gls01 is constant within cohort 2004, its interaction with that cohort repeats the cohort"),
        ("~ year", "Since year is constant within periods 2003, 2004, 2005, 2006, 2007, its interactions with them"),
        ("~ lpop + lpop_aff", r"lpop_aff is a linear combination of the controls before it\. Drop lpop_aff from"),
    ],
)
def test_etwfe_binary_family_collinear_control_raises(mpdta_moderators, xformla, match):
    with pytest.raises(ValueError, match=f"collinear columns that the binary families cannot drop. {match}"):
        etwfe(data=mpdta_moderators, yname="ybin", tname="year", gname="first.treat", family="logit", xformla=xformla)


def test_etwfe_no_collinearity_warning(mpdta_data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal")
        emfx(mod, type="event")
    assert [str(w.message) for w in caught] == []


def test_etwfe_scipy_backend_matches_default(mpdta_data, etwfe_baseline):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", backend="scipy")
    np.testing.assert_allclose(mod.coefficients, etwfe_baseline.coefficients, atol=1e-10)


def test_etwfe_dropped_cells_are_nan_with_warning(mpdta_state_dummies):
    data, xformla = mpdta_state_dummies
    with pytest.warns(UserWarning, match="dropped the treatment cells"):
        mod = etwfe(data=data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", xformla=xformla)
    assert len(mod.gt_pairs) == 7
    assert np.all(np.isnan(mod.coefficients))
    assert np.all(np.isnan(mod.std_errors))


def test_etwfe_model_coefficients_align_with_coef_names(etwfe_with_covariates):
    mod = etwfe_with_covariates
    assert len(mod.model_coefficients) == len(mod.coef_names) == mod.vcov.shape[0]
    pos = [mod.coef_names.index(f"_Dtreat:__etwfe_cell_{int(g)}_{int(t)}") for g, t in mod.gt_pairs]
    np.testing.assert_array_equal(mod.model_coefficients[pos], mod.coefficients)


def test_etwfe_poisson_emfx_simple(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", family="poisson")
    s = emfx(mod, type="simple")
    np.testing.assert_allclose(s.overall_att, -0.049194, atol=1e-3)
    assert s.overall_se > 0


def test_etwfe_poisson_emfx_event(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", family="poisson")
    e = emfx(mod, type="event")
    np.testing.assert_array_equal(e.event_times, [0.0, 1.0, 2.0, 3.0])
    expected = np.array([-0.032106, -0.055866, -0.135119, -0.106439])
    np.testing.assert_allclose(e.att_by_event, expected, atol=1e-3)
    assert np.all(e.se_by_event > 0)


def test_etwfe_poisson_emfx_group(mpdta_data):
    mod = etwfe(data=mpdta_data, yname="lemp", tname="year", gname="first.treat", family="poisson")
    g = emfx(mod, type="group")
    np.testing.assert_array_equal(g.event_times, [2004.0, 2006.0, 2007.0])
    expected = np.array([-0.08317, -0.022918, -0.044491])
    np.testing.assert_allclose(g.att_by_event, expected, atol=1e-3)


def test_etwfe_xvar_categorical(mpdta_data):
    data = mpdta_data.with_columns(
        pl.when(pl.col("lpop") > pl.col("lpop").median()).then(pl.lit("high")).otherwise(pl.lit("low")).alias("pop_cat")
    )
    mod = etwfe(data=data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", xvar="pop_cat")
    assert isinstance(mod, EtwfeResult)
    assert len(mod.gt_pairs) >= 6
    assert len(mod.coef_names) > 7


def test_etwfe_xvar_single_category_warns_and_adds_no_terms(mpdta_data, etwfe_baseline):
    data = mpdta_data.with_columns(pl.lit("all").alias("one_cat"))
    with pytest.warns(UserWarning, match="adds no terms"):
        mod = etwfe(data=data, yname="lemp", tname="year", gname="first.treat", idname="countyreal", xvar="one_cat")
    np.testing.assert_allclose(mod.coefficients, etwfe_baseline.coefficients, atol=1e-12)


@pytest.mark.parametrize("mpdta_converted", ["pandas", "pyarrow", "duckdb"], indirect=True)
def test_etwfe_dataframe_interoperability(mpdta_converted, etwfe_baseline):
    result = etwfe(
        data=mpdta_converted,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
    )
    np.testing.assert_allclose(result.coefficients, etwfe_baseline.coefficients, atol=1e-10)
    np.testing.assert_allclose(result.std_errors, etwfe_baseline.std_errors, atol=1e-10)
    assert result.gt_pairs == etwfe_baseline.gt_pairs


def _fit(data, **kwargs):
    spec = {"yname": "lemp", "tname": "year", "gname": "first.treat", "idname": "countyreal", **kwargs}
    return etwfe(data=data, **spec)


def _assert_same_fit(mod, ref):
    assert mod.gt_pairs == ref.gt_pairs
    np.testing.assert_allclose(mod.coefficients, ref.coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mod.std_errors, ref.std_errors, rtol=1e-10, atol=0)
    np.testing.assert_allclose(emfx(mod).overall_att, emfx(ref).overall_att, rtol=0, atol=1e-12)
    np.testing.assert_allclose(emfx(mod).overall_se, emfx(ref).overall_se, rtol=1e-10, atol=0)
    assert (mod.n_obs, mod.n_units) == (ref.n_obs, ref.n_units)


@pytest.mark.parametrize(
    "kwargs,plain",
    [
        (
            {"gname": "first.treat", "idname": None, "vcov": "hetero"},
            {"gname": "first_treat", "idname": None, "vcov": "hetero"},
        ),
        (
            {"gname": "first treat", "idname": None, "vcov": "hetero"},
            {"gname": "first_treat", "idname": None, "vcov": "hetero"},
        ),
        ({"idname": "county.id"}, {}),
        ({"idname": "county id"}, {}),
        ({"tname": "year.t"}, {}),
        ({"tname": "year t"}, {}),
        ({"yname": "l.emp"}, {}),
        ({"yname": "l emp"}, {}),
        ({"yname": "l-emp"}, {}),
        ({"xformla": "~ log.pop"}, {"xformla": "~ lpop"}),
        ({"xformla": "~ `log pop`"}, {"xformla": "~ lpop"}),
        ({"weightsname": "w t"}, {"weightsname": "w"}),
        ({"vcov": {"CRV1": "county id"}}, {"vcov": {"CRV1": "countyreal"}}),
        ({"xvar": "popcat spaced"}, {"xvar": "popcat"}),
        ({"yname": "l.emp", "tname": "year.t", "idname": "county.id", "xformla": "~ log.pop"}, {"xformla": "~ lpop"}),
    ],
)
def test_etwfe_column_names_with_dots_or_spaces_match_plain_names(mpdta_renamed, kwargs, plain):
    _assert_same_fit(_fit(mpdta_renamed, **kwargs), _fit(mpdta_renamed, **plain))


@pytest.mark.parametrize("xformla", ["~ I(lpop**2)", "~ np.log(lpop)", "~ C(treat)", "~ lpop*treat", "~ lpop:treat"])
def test_etwfe_transformed_controls_raise(mpdta_data, xformla):
    with pytest.raises(ValueError, match="is not a column name"):
        _fit(mpdta_data, xformla=xformla)


def test_etwfe_xformla_with_outcome_raises(mpdta_data):
    with pytest.raises(ValueError, match="has a left-hand side"):
        _fit(mpdta_data, xformla="lemp ~ lpop")


@pytest.mark.parametrize("label", ["inf", "9999"])
@pytest.mark.parametrize("kwargs", [{}, {"cgroup": "never"}, {"idname": None, "vcov": "hetero"}])
def test_etwfe_never_treated_codes_match_zero_coding(mpdta_data, mpdta_never_codes, label, kwargs):
    _assert_same_fit(_fit(mpdta_never_codes[label], **kwargs), _fit(mpdta_data, **kwargs))


@pytest.mark.parametrize("label", ["null", "nan"])
@pytest.mark.parametrize("kwargs", [{}, {"idname": None, "vcov": "hetero"}])
def test_etwfe_drops_rows_with_missing_cohorts(mpdta_never_codes, mpdta_no_never, label, kwargs):
    with pytest.warns(UserWarning, match="Dropped 1545 rows with missing values in first.treat"):
        mod = _fit(mpdta_never_codes[label], **kwargs)
    _assert_same_fit(mod, _fit(mpdta_no_never, **kwargs))
    assert mod.config.gref == 2007


@pytest.mark.parametrize("label", ["null", "nan"])
def test_etwfe_never_design_without_never_treated_after_missing_cohorts_raises(mpdta_never_codes, label):
    with (
        pytest.warns(UserWarning, match="Dropped 1545 rows with missing values in first.treat"),
        pytest.raises(ValueError, match="Could not identify 'never' control group"),
    ):
        _fit(mpdta_never_codes[label], cgroup="never")


def test_etwfe_drops_nan_cohorts_from_pandas(mpdta_never_codes, mpdta_no_never):
    with pytest.warns(UserWarning, match="Dropped 1545 rows with missing values in first.treat"):
        mod = _fit(mpdta_never_codes["nan"].to_pandas())
    _assert_same_fit(mod, _fit(mpdta_no_never))


@pytest.mark.parametrize("cgroup", ["notyet", "never"])
def test_etwfe_cohort_after_last_period_is_never_treated(mpdta_late_cohort, cgroup):
    late, recoded = mpdta_late_cohort
    mod = _fit(late, cgroup=cgroup)
    _assert_same_fit(mod, _fit(recoded, cgroup=cgroup))
    assert mod.config.gref == 0
    assert all(g != 0 for g, _ in mod.gt_pairs)


@pytest.mark.parametrize("kwargs", [{}, {"cgroup": "never"}, {"idname": None, "vcov": "hetero"}])
def test_etwfe_drops_cohort_treated_in_first_period(mpdta_always_treated, kwargs):
    data, without = mpdta_always_treated
    with pytest.warns(UserWarning, match="already treated in the first period"):
        mod = _fit(data, **kwargs)
    _assert_same_fit(mod, _fit(without, **kwargs))
    assert all(g != 2003 for g, _ in mod.gt_pairs)


@pytest.mark.parametrize("kwargs", [{}, {"idname": None, "vcov": "hetero"}])
def test_etwfe_every_unit_treated_in_first_period_raises(mpdta_data, kwargs):
    data = mpdta_data.with_columns(pl.lit(2003).alias("first.treat"))
    with (
        pytest.warns(UserWarning, match="already treated in the first period"),
        pytest.raises(ValueError, match="No rows are left to estimate from"),
    ):
        _fit(data, **kwargs)


@pytest.mark.parametrize(
    "data_fixture,cgroup,match",
    [
        ("mpdta_calendar_gap", "never", r"no row in period g - 1 \(2006 lacks 2005\)"),
        ("mpdta_late_entry", "never", r"no row in period g - 1 \(2006 lacks 2005\)"),
        ("mpdta_late_entry", "notyet", r"no row before their first treated period \(2006\)"),
    ],
)
@pytest.mark.parametrize("kwargs", [{}, {"idname": None, "vcov": "hetero"}])
def test_etwfe_drops_cohorts_without_untreated_rows(request, data_fixture, cgroup, match, kwargs):
    data = request.getfixturevalue(data_fixture)
    with pytest.warns(UserWarning, match=match):
        mod = _fit(data, cgroup=cgroup, **kwargs)
    _assert_same_fit(mod, _fit(data.filter(pl.col("first.treat") != 2006), cgroup=cgroup, **kwargs))
    assert all(g != 2006 for g, _ in mod.gt_pairs)
    assert np.all(np.isfinite(mod.coefficients))


@pytest.mark.parametrize(
    "column,kwargs",
    [
        ("lpop", {"xformla": "~ lpop"}),
        ("lpop", {"xformla": "~ lpop", "cgroup": "never"}),
        ("gls", {"xvar": "gls"}),
        ("w", {"weightsname": "w"}),
        ("countyreal", {}),
        ("first.treat", {}),
    ],
)
def test_etwfe_drops_rows_with_missing_values(mpdta_missing, column, kwargs):
    data, missing = mpdta_missing
    holed = data.with_columns(pl.when(missing).then(None).otherwise(pl.col(column)).alias(column))
    with pytest.warns(UserWarning, match=f"Dropped 158 rows with missing values in {column}"):
        mod = _fit(holed, **kwargs)
    ref = _fit(data.filter(~missing), **kwargs)
    _assert_same_fit(mod, ref)
    assert mod.data.height == mod.n_obs == 2342
    for agg in ["group", "calendar", "event"]:
        np.testing.assert_allclose(emfx(mod, type=agg).att_by_event, emfx(ref, type=agg).att_by_event, atol=1e-12)


@pytest.mark.parametrize("code", [None, float("nan")])
def test_etwfe_drops_rows_with_missing_cluster_values(mpdta_states, code):
    data, missing = mpdta_states
    holed = data.with_columns(pl.when(missing).then(code).otherwise(pl.col("st")).alias("st"))
    with pytest.warns(UserWarning, match="Dropped 15 rows with missing values in st"):
        mod = _fit(holed, vcov={"CRV1": "st"})
    _assert_same_fit(mod, _fit(data.filter(~missing), vcov={"CRV1": "st"}))
    assert (mod.n_obs, mod.n_units) == (2485, 497)


def test_etwfe_n_units_counts_units_in_estimation_sample(mpdta_data):
    first25 = mpdta_data["countyreal"].unique().sort().head(25)
    holed = mpdta_data.with_columns(
        pl.when(pl.col("countyreal").is_in(first25.implode())).then(None).otherwise(pl.col("lpop")).alias("lpop")
    )
    with pytest.warns(UserWarning, match="Dropped 125 rows"):
        mod = _fit(holed, xformla="~ lpop")
    assert (mod.n_obs, mod.n_units) == (2375, 475)
    assert mod.estimation_params["n_units"] == 475


def test_etwfe_fit_data_excludes_singleton_units(mpdta_singletons):
    mod = _fit(mpdta_singletons, vcov={"CRV1": "countyreal"})
    assert (mod.n_obs, mod.n_units, mod.data.height) == (2375, 475, 2375)
    cell_rows = mod.data.filter(pl.col("_Dtreat") == 1.0).group_by("_g", "_t").len()
    counts = {(g, t): n for g, t, n in cell_rows.iter_rows()}
    weights = np.array([counts[cell] for cell in mod.gt_pairs], dtype=float)
    np.testing.assert_allclose(emfx(mod).overall_att, weights @ mod.coefficients / weights.sum(), atol=1e-14)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"gref": 1999}, "gref=1999 is not a cohort first treated after the first period"),
        ({"gref": 2005}, "gref=2005 is not a cohort in 'first.treat'"),
        ({"tref": 1990}, "tref=1990 is not a period in 'year'"),
    ],
)
def test_etwfe_rejects_references_outside_data(mpdta_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _fit(mpdta_data, **kwargs)


@pytest.mark.parametrize("label,gref", [(None, float("inf")), (None, 2008), (None, 9999), ("9999", 0), ("inf", 0)])
def test_etwfe_never_treated_code_as_gref_matches_default(mpdta_data, mpdta_never_codes, etwfe_baseline, label, gref):
    data = mpdta_data if label is None else mpdta_never_codes[label]
    mod = _fit(data, gref=gref)
    _assert_same_fit(mod, etwfe_baseline)
    assert mod.config.gref == 0


def test_etwfe_treated_reference_keeps_never_treated_as_controls(mpdta_data, mpdta_never_codes):
    mod = _fit(mpdta_data, gref=2007)
    assert mod.gt_pairs == [(2004.0, 2004.0), (2004.0, 2005.0), (2004.0, 2006.0), (2006.0, 2006.0)]
    assert mod.n_obs == 2000
    _assert_same_fit(mod, _fit(mpdta_never_codes["9999"], gref=2007))


def test_etwfe_rejects_repeated_unit_periods(mpdta_duplicated):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (countyreal, year) pair (17005, 2005)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        etwfe(data=mpdta_duplicated, yname="lemp", tname="year", gname="first.treat", idname="countyreal")


def test_etwfe_without_idname_keeps_every_row(mpdta_duplicated):
    result = etwfe(data=mpdta_duplicated, yname="lemp", tname="year", gname="first.treat")

    assert result.n_obs == mpdta_duplicated.height


def test_etwfe_drops_rows_without_a_year_before_the_unit_check(mpdta_without_years, etwfe_baseline):
    with pytest.warns(UserWarning, match=r"^Dropped 2 rows with missing values in year\.$"):
        result = etwfe(data=mpdta_without_years, yname="lemp", tname="year", gname="first.treat", idname="countyreal")

    np.testing.assert_array_equal(result.coefficients, etwfe_baseline.coefficients)
    np.testing.assert_array_equal(result.std_errors, etwfe_baseline.std_errors)
