"""Tests for the ETWFE estimator."""

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
    expected = np.array([0.030820, 0.027551, 0.030408, 0.018148, 0.032949, 0.021658, 0.017891])
    np.testing.assert_allclose(etwfe_baseline.std_errors, expected, atol=1e-4)


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
    assert len(mod.coefficients) > 7


def test_etwfe_fe_spec_in_params(etwfe_baseline):
    assert etwfe_baseline.estimation_params["fe_spec"] == "countyreal + year"


def test_etwfe_without_idname(mpdta_data):
    data = mpdta_data.rename({"first.treat": "first_treat"})
    mod = etwfe(data=data, yname="lemp", tname="year", gname="first_treat")
    assert isinstance(mod, EtwfeResult)
    assert mod.estimation_params["idname"] is None
    assert mod.n_units == mod.n_obs


def test_etwfe_covariates_same_gt_atts(etwfe_baseline, etwfe_with_covariates):
    np.testing.assert_allclose(
        etwfe_with_covariates.coefficients[:7],
        etwfe_baseline.coefficients,
        atol=1e-4,
    )


def test_etwfe_covariates_more_coefficients(etwfe_baseline, etwfe_with_covariates):
    assert len(etwfe_with_covariates.coefficients) > len(etwfe_baseline.coefficients)


def test_etwfe_covariates_different_se(etwfe_baseline, etwfe_with_covariates):
    simple_no_cov = emfx(etwfe_baseline, type="simple")
    simple_cov = emfx(etwfe_with_covariates, type="simple")
    np.testing.assert_allclose(simple_cov.overall_att, simple_no_cov.overall_att, atol=1e-4)
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


def test_etwfe_nonlinear_forces_no_fe(mpdta_data):
    with pytest.warns(UserWarning, match="does not support unit FE absorption"):
        mod = etwfe(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            family="poisson",
        )
    assert mod.estimation_params["family"] == "poisson"
    assert mod.estimation_params["fe"] == "none"


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
    ],
)
def test_etwfe_missing_column(mpdta_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        etwfe(data=mpdta_data, **kwargs)


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
