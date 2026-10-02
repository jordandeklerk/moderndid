"""Prepare complete estimator calls outside timing."""

from dataclasses import replace
from functools import partial

import numpy as np

from benchmarks.cases import WORKLOADS, make_panel
from moderndid import agg_ddd, aggte, att_gt, cont_did, ddd, did_multiplegt, drdid, ipwdid, ordid

# Since an estimator that is not listed accepts every workload in cases.py, the table holds only the exceptions.
PROFILES = {
    "att_gt": (*WORKLOADS, "ipw", "reg"),
    "ddd": (*WORKLOADS, "ipw", "reg"),
    "did_multiplegt": (*WORKLOADS, "normalized"),
    "drdid": ("small", "baseline", "rows", "covariates"),
    "ipwdid": ("small", "baseline", "rows", "covariates"),
    "ordid": ("small", "baseline", "rows", "covariates"),
}

# Since agg_ddd calls the dynamic aggregation eventstudy, each estimator's function keeps its own name for it.
DYNAMIC_TYPES = {"att_gt": "dynamic", "ddd": "eventstudy"}


def resolve_workload(estimator, profile):
    """Return the workload behind a profile name for one estimator."""
    if profile not in PROFILES.get(estimator, tuple(WORKLOADS)):
        raise ValueError(f"Unknown workload {profile!r} for {estimator!r}")
    workload = WORKLOADS.get(profile, WORKLOADS["baseline"])
    if estimator in {"drdid", "ipwdid", "ordid"}:
        workload = replace(workload, n_periods=2, n_cohorts=1)
    if estimator == "cont_did" and workload.n_covariates:
        raise ValueError("Continuous-treatment workloads cannot include covariates")
    return workload


def make_estimator(estimator, profile):
    """Prepare one seeded estimator call."""
    workload = resolve_workload(estimator, profile)
    data = make_panel(workload, estimator)
    covariate_names = [f"x{index + 1}" for index in range(workload.n_covariates)]
    formula = "~ " + " + ".join(covariate_names) if covariate_names else "~1"
    kwargs = {
        "data": data,
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "xformla": formula,
        "boot": workload.boot,
    }
    if estimator in {"drdid", "ipwdid", "ordid"}:
        # Since these functions have no biters or random_state arguments, they return before those settings are added.
        function = {"drdid": drdid, "ipwdid": ipwdid, "ordid": ordid}[estimator]
        kwargs.update(treatname="treated", panel=True)
        if estimator == "drdid":
            kwargs.update(est_method="imp")
        return partial(function, **kwargs)
    kwargs.update(biters=workload.biters, random_state=42)
    if estimator == "att_gt":
        function = att_gt
        kwargs.update(gname="group", est_method=profile if profile in {"ipw", "reg"} else "dr", n_jobs=1)
    elif estimator == "ddd":
        function = ddd
        kwargs.update(
            gname="group", pname="partition", est_method=profile if profile in {"ipw", "reg"} else "dr", n_jobs=1
        )
    elif estimator == "cont_did":
        function = cont_did
        kwargs.update(
            gname="group", dname="dose", dvals=np.linspace(0.6, 1.9, 20), dose_est_method="parametric", cband=False
        )
    elif estimator == "did_multiplegt":
        function = did_multiplegt
        kwargs.update(
            dname="treatment",
            effects=2,
            placebo=1,
            normalized=profile == "normalized",
            biters=199 if profile == "bootstrap" else workload.biters,
        )
    else:
        raise ValueError(f"Unknown estimator {estimator!r}")
    return partial(function, **kwargs)


def check_estimate(estimator, result):
    """Check the warmed estimate before recording timings."""
    if estimator == "att_gt":
        estimates, errors = result.att_gt, result.se_gt
        assert np.shape(estimates) == np.shape(result.groups) == np.shape(result.times)
    elif estimator == "ddd":
        estimates, errors = result.att, result.se
        assert np.shape(estimates) == np.shape(result.groups) == np.shape(result.times)
        assert np.shape(estimates) == np.shape(errors)
        reference = np.asarray(result.times) == np.asarray(result.groups) - 1
        assert np.all(np.asarray(estimates)[reference] == 0)
        assert np.all(np.isnan(np.asarray(errors)[reference]) | (np.asarray(errors)[reference] == 0))
        estimates = np.asarray(estimates)[~reference]
        errors = np.asarray(errors)[~reference]
    elif estimator == "cont_did":
        estimates, errors = result.att_d, result.att_d_se
        assert np.shape(estimates) == np.shape(result.dose) == (20,)
    elif estimator in {"drdid", "ipwdid", "ordid"}:
        estimates, errors = result.att, result.se
    else:
        estimates, errors = result.effects.estimates, result.effects.std_errors
        assert np.shape(estimates) == (2,)
        assert np.shape(result.placebos.estimates) == (1,)
        assert np.shape(result.placebos.std_errors) == (1,)
        assert np.all(np.isfinite(result.placebos.estimates))
        assert np.all(np.isfinite(result.placebos.std_errors))
        assert np.all(np.asarray(result.placebos.std_errors) >= 0)
    assert np.size(estimates) > 0
    assert np.shape(estimates) == np.shape(errors)
    assert np.all(np.isfinite(estimates))
    assert np.all(np.isfinite(errors))
    assert np.all(np.asarray(errors) >= 0)


def aggregation_type(estimator, aggregation):
    """Translate a shared aggregation name into the type an aggregation function expects."""
    return DYNAMIC_TYPES[estimator] if aggregation == "dynamic" else aggregation


def make_aggregation(estimator, profile, aggregation):
    """Prepare aggregation from an already fitted estimator."""
    result = make_estimator(estimator, profile)()
    check_estimate(estimator, result)
    function = aggte if estimator == "att_gt" else agg_ddd
    kind = aggregation_type(estimator, aggregation)
    aggregate = partial(function, result, type=kind, boot=False, cband=False, biters=199, random_state=42)
    check_aggregation(aggregate())
    return aggregate


def check_aggregation(result):
    """Check the warmed aggregate before recording timings."""
    assert np.isfinite(result.overall_att)
    assert np.isfinite(result.overall_se)
    assert result.overall_se >= 0


class EstimatorBenchmark:
    """Prepare a warmed estimator for ASV."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    version = "1"
    timeout = 120
    estimator = ""

    def setup(self, profile):
        """Build data and warm the estimator."""
        self.estimate = make_estimator(self.estimator, profile)
        check_estimate(self.estimator, self.estimate())
