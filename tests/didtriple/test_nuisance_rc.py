import numpy as np
import pytest

from moderndid.didtriple.nuisance_rc import (
    DIDRCResult,
    OutcomeRegRCResult,
    PScoreRCResult,
    compute_all_did_rc,
    compute_all_nuisances_rc,
)


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_nuisances_rc_structure(rcs_nuisance_data, est_method):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    pscores, or_results = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method=est_method,
    )

    assert len(pscores) == 3
    assert len(or_results) == 3
    for ps in pscores:
        assert len(ps.propensity_scores) > 0
        assert np.all(np.isfinite(ps.propensity_scores))
    for or_r in or_results:
        assert len(or_r.y) > 0


def test_nuisances_rc_dr_hessian_invertible(rcs_nuisance_data):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    pscores, _ = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method="dr",
    )
    for ps in pscores:
        assert ps.hessian_matrix is not None
        det = np.linalg.det(ps.hessian_matrix)
        assert det != 0, "Hessian matrix should be invertible"


def test_nuisances_rc_reg_bypasses_pscore(rcs_nuisance_data):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    pscores, _ = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method="reg",
    )
    for ps in pscores:
        assert np.all(ps.propensity_scores == 1.0), "REG method should set all pscores to 1"
        assert ps.hessian_matrix is None, "REG method should not compute Hessian"


def test_nuisances_rc_invalid_method(rcs_nuisance_data):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    with pytest.raises(ValueError, match="est_method must be"):
        compute_all_nuisances_rc(
            y=y,
            post=post,
            subgroup=subgroup,
            covariates=covariates,
            weights=weights,
            est_method="invalid",
        )


def test_nuisances_rc_pscore_valid_probabilities(rcs_nuisance_data):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    pscores, _ = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method="dr",
    )
    for ps in pscores:
        assert np.all(ps.propensity_scores >= 0), "Propensity scores should be non-negative"
        assert np.all(ps.propensity_scores <= 1), "Propensity scores should be <= 1"
        assert np.all(np.isfinite(ps.propensity_scores)), "Propensity scores should be finite"


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_did_rc_valid_att(rcs_nuisance_data, est_method):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    n_total = len(y)

    pscores, or_results = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method=est_method,
    )
    did_results, ddd_att, _ = compute_all_did_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        pscores=pscores,
        or_results=or_results,
        est_method=est_method,
        n_total=n_total,
    )

    assert len(did_results) == 3
    assert np.isfinite(ddd_att)
    y_range = np.max(y) - np.min(y)
    assert np.abs(ddd_att) < 2 * y_range


def test_did_rc_ddd_formula(rcs_nuisance_data):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    n_total = len(y)

    pscores, or_results = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method="dr",
    )
    did_results, ddd_att, _ = compute_all_did_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        pscores=pscores,
        or_results=or_results,
        est_method="dr",
        n_total=n_total,
    )

    component_sum = did_results[0].dr_att + did_results[1].dr_att - did_results[2].dr_att
    assert np.isclose(ddd_att, component_sum, rtol=1e-10)


def test_did_rc_inf_func_mean(rcs_nuisance_data):
    y, post, subgroup, covariates, weights = rcs_nuisance_data
    n_total = len(y)

    pscores, or_results = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        est_method="dr",
    )
    _, _, inf_func = compute_all_did_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights,
        pscores=pscores,
        or_results=or_results,
        est_method="dr",
        n_total=n_total,
    )

    assert np.abs(np.mean(inf_func)) < 0.5
    assert np.var(inf_func) > 0


def test_outcome_reg_rc_result():
    result = OutcomeRegRCResult(
        y=np.array([1.0, 2.0]),
        out_y_cont=np.array([1.0, 2.0]),
        out_y_cont_pre=np.array([1.0]),
        out_y_cont_post=np.array([2.0]),
        out_y_treat_pre=np.array([1.0]),
        out_y_treat_post=np.array([2.0]),
    )
    assert len(result.y) == 2


def test_pscore_rc_result():
    result = PScoreRCResult(
        propensity_scores=np.array([0.5, 0.6]),
        hessian_matrix=np.eye(2),
    )
    assert len(result.propensity_scores) == 2
    assert result.hessian_matrix.shape == (2, 2)


def test_did_rc_result():
    result = DIDRCResult(dr_att=2.0, inf_func=np.array([0.1, -0.1, 0.05]))
    assert result.dr_att == 2.0
    assert len(result.inf_func) == 3
