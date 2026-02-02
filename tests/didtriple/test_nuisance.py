"""Tests for DDD nuisance parameter estimation."""

import pytest

from moderndid.core.preprocessing import preprocess_ddd_2periods
from moderndid.didtriple.dgp import gen_dgp_2periods
from moderndid.didtriple.nuisance import (
    DIDResult,
    OutcomeRegResult,
    PScoreResult,
    _compute_pscore,
    _compute_pscore_null,
    compute_all_did,
    compute_all_nuisances,
)

from ..helpers import importorskip

np = importorskip("numpy")


class TestComputeAllNuisances:
    def test_dr_structure(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        assert len(pscores) == 3
        assert len(or_results) == 3

        for ps_result in pscores:
            assert isinstance(ps_result, PScoreResult)
            assert ps_result.hessian_matrix is not None

        for or_result in or_results:
            assert isinstance(or_result, OutcomeRegResult)
            assert or_result.reg_coeff is not None

    def test_reg_null_pscores(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="reg",
        )

        for ps_result in pscores:
            assert np.all(ps_result.propensity_scores == 1.0)
            assert ps_result.hessian_matrix is None

        for or_result in or_results:
            assert or_result.reg_coeff is not None

    def test_ipw_null_or(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="ipw",
        )

        for ps_result in pscores:
            assert ps_result.hessian_matrix is not None

        for or_result in or_results:
            assert np.all(or_result.or_delta == 0.0)
            assert or_result.reg_coeff is None

    def test_invalid_method_raises(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        with pytest.raises(ValueError, match="est_method must be"):
            compute_all_nuisances(
                y1=ddd_data.y1,
                y0=ddd_data.y0,
                subgroup=ddd_data.subgroup,
                covariates=covariates,
                weights=ddd_data.weights,
                est_method="invalid",
            )

    def test_pscore_range(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, _ = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        for ps_result in pscores:
            assert np.all(ps_result.propensity_scores > 0)
            assert np.all(ps_result.propensity_scores < 1)


class TestComputeAllDid:
    def test_structure(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        did_results, ddd_att, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        assert len(did_results) == 3
        for did_result in did_results:
            assert isinstance(did_result, DIDResult)
            assert isinstance(did_result.dr_att, float)
            assert len(did_result.inf_func) == ddd_data.n_units

        assert isinstance(ddd_att, float)
        assert len(inf_func) == ddd_data.n_units

    def test_ddd_formula(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        did_results, ddd_att, _ = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        expected_ddd = did_results[0].dr_att + did_results[1].dr_att - did_results[2].dr_att

        assert np.isclose(ddd_att, expected_ddd, rtol=1e-10)

    def test_inf_func_mean(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, _, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        mean_inf = np.mean(inf_func)
        assert np.abs(mean_inf) < 1.0

    def test_att_null_dgp(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, ddd_att, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        se = np.std(inf_func, ddof=1) / np.sqrt(ddd_data.n_units)

        assert np.abs(ddd_att) < 3 * se + 5


class TestEstimationMethods:
    @pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
    def test_methods(self, ddd_data_with_covariates, est_method):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method=est_method,
        )

        did_results, ddd_att, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method=est_method,
            n_total=ddd_data.n_units,
        )

        assert len(did_results) == 3
        for did_result in did_results:
            assert isinstance(did_result, DIDResult)
            assert np.isfinite(did_result.dr_att)

        assert np.isfinite(ddd_att)
        assert np.all(np.isfinite(inf_func))
        assert len(inf_func) == ddd_data.n_units

    def test_intercept_only(self, ddd_data_no_covariates):
        ddd_data, covariates = ddd_data_no_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        did_results, ddd_att, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        assert len(did_results) == 3
        for did_result in did_results:
            assert isinstance(did_result, DIDResult)

        assert np.isfinite(ddd_att)
        assert np.all(np.isfinite(inf_func))
        assert len(inf_func) == ddd_data.n_units


class TestReproducibility:
    def test_deterministic(self):
        result = gen_dgp_2periods(n=500, dgp_type=1, random_state=123)
        data = result["data"]

        ddd_data = preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla="~ cov1 + cov2",
        )

        covariates = np.column_stack([np.ones(ddd_data.n_units), ddd_data.covariates])

        pscores1, or_results1 = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, ddd_att1, _ = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores1,
            or_results=or_results1,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        pscores2, or_results2 = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, ddd_att2, _ = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores2,
            or_results=or_results2,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        assert ddd_att1 == ddd_att2


class TestValueRanges:
    def test_se_positive(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        for est_method in ["dr", "reg", "ipw"]:
            pscores, or_results = compute_all_nuisances(
                y1=ddd_data.y1,
                y0=ddd_data.y0,
                subgroup=ddd_data.subgroup,
                covariates=covariates,
                weights=ddd_data.weights,
                est_method=est_method,
            )

            _, _, inf_func = compute_all_did(
                subgroup=ddd_data.subgroup,
                covariates=covariates,
                weights=ddd_data.weights,
                pscores=pscores,
                or_results=or_results,
                est_method=est_method,
                n_total=ddd_data.n_units,
            )

            se = np.std(inf_func, ddof=1) / np.sqrt(ddd_data.n_units)
            assert se > 0, f"SE must be positive for {est_method}"

    def test_se_magnitude(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, _, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        se = np.std(inf_func, ddof=1) / np.sqrt(ddd_data.n_units)
        assert 0.01 < se < 10

    def test_att_bounded(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, ddd_att, _ = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        delta_y = ddd_data.y1 - ddd_data.y0
        outcome_range = np.max(delta_y) - np.min(delta_y)
        assert np.abs(ddd_att) < outcome_range

    def test_inf_func_variance(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        _, _, inf_func = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        assert np.var(inf_func) > 0

    def test_subgroup_atts(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        did_results, _, _ = compute_all_did(
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            pscores=pscores,
            or_results=or_results,
            est_method="dr",
            n_total=ddd_data.n_units,
        )

        for i, did_result in enumerate(did_results):
            assert np.isfinite(did_result.dr_att), f"DiD ATT {i} is not finite"
            assert np.abs(did_result.dr_att) < 100, f"DiD ATT {i} is too large"

    def test_pscore_values(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        pscores, _ = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        for ps_result in pscores:
            ps = ps_result.propensity_scores
            assert np.mean(ps) > 0.1
            assert np.mean(ps) < 0.9
            assert np.std(ps) > 0

    def test_or_coefficients(self, ddd_data_with_covariates):
        ddd_data, covariates = ddd_data_with_covariates

        _, or_results = compute_all_nuisances(
            y1=ddd_data.y1,
            y0=ddd_data.y0,
            subgroup=ddd_data.subgroup,
            covariates=covariates,
            weights=ddd_data.weights,
            est_method="dr",
        )

        for or_result in or_results:
            assert np.all(np.isfinite(or_result.reg_coeff))
            assert np.all(np.abs(or_result.reg_coeff) < 1000)


def test_pscore_result_has_keep_ps(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    pscores, _ = compute_all_nuisances(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        weights=ddd_data.weights,
        est_method="dr",
    )

    for ps_result in pscores:
        assert hasattr(ps_result, "keep_ps")
        assert ps_result.keep_ps is not None
        assert len(ps_result.keep_ps) == len(ps_result.propensity_scores)
        assert ps_result.keep_ps.dtype == bool


def test_pscore_null_keep_ps_all_true():
    subgroup = np.array([4, 4, 4, 3, 3, 3, 3])
    result = _compute_pscore_null(subgroup, comparison_subgroup=3)
    assert np.all(result.keep_ps)


def test_trim_level_parameter(ddd_data_with_covariates):
    ddd_data, covariates = ddd_data_with_covariates

    pscores_default, _ = compute_all_nuisances(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        weights=ddd_data.weights,
        est_method="dr",
        trim_level=0.995,
    )

    pscores_strict, _ = compute_all_nuisances(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates,
        weights=ddd_data.weights,
        est_method="dr",
        trim_level=0.5,
    )

    for ps_default, ps_strict in zip(pscores_default, pscores_strict):
        n_trimmed_default = np.sum(~ps_default.keep_ps)
        n_trimmed_strict = np.sum(~ps_strict.keep_ps)
        assert n_trimmed_strict >= n_trimmed_default


def test_compute_pscore_trimming():
    rng = np.random.default_rng(42)
    n = 200
    subgroup = np.array([4] * 50 + [3] * 150)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), x1, x2])
    weights = np.ones(n)

    result = _compute_pscore(subgroup, covariates, weights, comparison_subgroup=3, trim_level=0.995)

    assert isinstance(result, PScoreResult)
    assert len(result.propensity_scores) == n
    assert len(result.keep_ps) == n
    assert result.hessian_matrix is not None

    pa4 = subgroup == 4
    mask = (subgroup == 4) | (subgroup == 3)
    pa4_sub = pa4[mask]
    assert np.all(result.keep_ps[pa4_sub == 1])

    control_ps = result.propensity_scores[pa4_sub == 0]
    control_keep = result.keep_ps[pa4_sub == 0]

    for ps, keep in zip(control_ps, control_keep):
        if ps >= 0.995:
            assert not keep
        else:
            assert keep
