"""Tests for distributed influence function computation."""

import numpy as np
import pytest

from moderndid.dask._inf_func import _compute_did, _compute_inf_func
from moderndid.dask._nuisance import DistOutcomeRegResult, DistPScoreResult


def _make_nuisance(subgroup, comp_subgroup, est_method, covariates, y1, y0):
    mask = (subgroup == 4) | (subgroup == comp_subgroup)
    n_sub = int(np.sum(mask))
    k = covariates.shape[1]

    if est_method == "reg":
        ps = np.ones(n_sub)
        hessian = None
    else:
        ps = np.full(n_sub, 0.5)
        hessian = np.eye(k) * n_sub

    keep_ps = np.ones(n_sub, dtype=bool)
    pscore_result = DistPScoreResult(propensity_scores=ps, hessian_matrix=hessian, keep_ps=keep_ps)

    delta_y = (y1 - y0)[mask]
    or_delta = np.zeros(n_sub)
    or_result = DistOutcomeRegResult(delta_y=delta_y, or_delta=or_delta, reg_coeff=None)

    return pscore_result, or_result


def test_compute_did_returns_finite(ddd_subgroup_arrays):
    sg = ddd_subgroup_arrays["subgroup"]
    cov = ddd_subgroup_arrays["covariates"]
    w = ddd_subgroup_arrays["weights"]
    y1 = ddd_subgroup_arrays["y1"]
    y0 = ddd_subgroup_arrays["y0"]
    n = ddd_subgroup_arrays["n"]

    ps_res, or_res = _make_nuisance(sg, 3, "reg", cov, y1, y0)
    att, inf = _compute_did(sg, cov, w, 3, ps_res, or_res, "reg", n)
    assert np.isfinite(att)
    assert np.all(np.isfinite(inf))


def test_compute_did_zeros_outside_mask(ddd_subgroup_arrays):
    sg = ddd_subgroup_arrays["subgroup"]
    cov = ddd_subgroup_arrays["covariates"]
    w = ddd_subgroup_arrays["weights"]
    y1 = ddd_subgroup_arrays["y1"]
    y0 = ddd_subgroup_arrays["y0"]
    n = ddd_subgroup_arrays["n"]

    ps_res, or_res = _make_nuisance(sg, 2, "reg", cov, y1, y0)
    _, inf = _compute_did(sg, cov, w, 2, ps_res, or_res, "reg", n)

    outside_mask = ~((sg == 4) | (sg == 2))
    np.testing.assert_array_equal(inf[outside_mask], 0.0)


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_compute_inf_func_shape(ddd_subgroup_arrays, est_method):
    sg = ddd_subgroup_arrays["subgroup"]
    cov = ddd_subgroup_arrays["covariates"]
    w = ddd_subgroup_arrays["weights"]
    y1 = ddd_subgroup_arrays["y1"]
    y0 = ddd_subgroup_arrays["y0"]

    ps_res, or_res = _make_nuisance(sg, 3, est_method, cov, y1, y0)
    mask = (sg == 4) | (sg == 3)
    sub_sg = sg[mask]
    sub_cov = cov[mask]
    sub_w = w[mask]
    n_sub = int(np.sum(mask))

    pa4 = (sub_sg == 4).astype(float)
    pa_comp = (sub_sg == 3).astype(float)

    pscore = ps_res.propensity_scores
    keep_ps = ps_res.keep_ps.astype(float)
    delta_y = or_res.delta_y
    or_delta = or_res.or_delta

    w_treat = keep_ps * sub_w * pa4
    if est_method == "reg":
        w_control = keep_ps * sub_w * pa_comp
    else:
        w_control = keep_ps * sub_w * pscore * pa_comp / (1 - pscore)

    riesz_treat = w_treat * (delta_y - or_delta)
    riesz_control = w_control * (delta_y - or_delta)
    mean_w_treat = np.mean(w_treat)
    mean_w_control = np.mean(w_control)
    att_treat = np.mean(riesz_treat) / mean_w_treat
    att_control = np.mean(riesz_control) / mean_w_control

    inf = _compute_inf_func(
        sub_covariates=sub_cov,
        sub_weights=sub_w,
        pa4=pa4,
        pa_comp=pa_comp,
        pscore=pscore,
        hessian=ps_res.hessian_matrix,
        delta_y=delta_y,
        or_delta=or_delta,
        w_treat=w_treat,
        w_control=w_control,
        riesz_treat=riesz_treat,
        riesz_control=riesz_control,
        att_treat=att_treat,
        att_control=att_control,
        mean_w_treat=mean_w_treat,
        mean_w_control=mean_w_control,
        est_method=est_method,
    )
    assert inf.shape == (n_sub,)
    assert np.all(np.isfinite(inf))
