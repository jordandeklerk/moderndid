"""Test source package validation."""

import pytest
from numpy.testing import assert_allclose

import moderndid as md

try:
    from did_multiplegt_dyn import DidMultiplegtDyn

    HAS_SOURCE_PACKAGE = True
except ImportError:
    HAS_SOURCE_PACKAGE = False


def run_source_package(df, **kwargs):
    model = DidMultiplegtDyn(df, **kwargs)
    model.fit()
    return model.result["did_multiplegt_dyn"]


pytestmark = pytest.mark.skipif(not HAS_SOURCE_PACKAGE, reason="py_did_multiplegt_dyn not installed")


@pytest.mark.parametrize("effects", [1, 4])
def test_effects_count(favara_imbs_data, effects):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=effects,
        placebo=0,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=effects,
        placebo=0,
    )

    assert_allclose(
        our_result.effects.estimates,
        source_res["Effects"]["Estimate"].to_numpy(),
        rtol=1e-5,
    )
    assert_allclose(
        our_result.effects.std_errors,
        source_res["Effects"]["SE"].to_numpy(),
        rtol=1e-3,
    )


@pytest.mark.parametrize("ci_level", [90, 95])
def test_confidence_intervals(favara_imbs_data, ci_level):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=2,
        placebo=0,
        ci_level=ci_level,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        placebo=0,
        ci_level=float(ci_level),
    )

    assert_allclose(
        our_result.effects.ci_lower,
        source_res["Effects"]["LB CI"].to_numpy(),
        rtol=1e-3,
    )
    assert_allclose(
        our_result.effects.ci_upper,
        source_res["Effects"]["UB CI"].to_numpy(),
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "source_kwargs,our_kwargs",
    [
        pytest.param({"normalized": True}, {"normalized": True}, id="normalized"),
        pytest.param({"switchers": "in"}, {"switchers": "in"}, id="switchers_in"),
        pytest.param({"weight": "w1"}, {"weightsname": "w1"}, id="weighted"),
        pytest.param({"same_switchers": True}, {"same_switchers": True}, id="same_switchers"),
        pytest.param(
            {"only_never_switchers": True, "effects": 2},
            {"only_never_switchers": True, "effects": 2},
            id="only_never_switchers",
        ),
    ],
)
def test_estimation_options(favara_imbs_data, source_kwargs, our_kwargs):
    base_source = {
        "outcome": "Dl_vloans_b",
        "group": "county",
        "time": "year",
        "treatment": "inter_bra",
        "effects": 3,
        "placebo": 0,
    }
    base_ours = {
        "yname": "Dl_vloans_b",
        "idname": "county",
        "tname": "year",
        "dname": "inter_bra",
        "effects": 3,
        "placebo": 0,
    }

    source_res = run_source_package(favara_imbs_data, **{**base_source, **source_kwargs})
    our_result = md.did_multiplegt(favara_imbs_data, **{**base_ours, **our_kwargs})

    assert_allclose(
        our_result.effects.estimates,
        source_res["Effects"]["Estimate"].to_numpy(),
        rtol=1e-5,
    )


def test_effects_and_placebos(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=3,
        placebo=2,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=2,
    )

    assert_allclose(
        our_result.effects.estimates,
        source_res["Effects"]["Estimate"].to_numpy(),
        rtol=1e-5,
    )
    assert_allclose(
        our_result.placebos.estimates,
        source_res["Placebos"]["Estimate"].to_numpy(),
        rtol=1e-5,
    )


def test_ate_estimate(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=3,
        placebo=0,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=0,
    )

    source_ate = source_res["ATE"]["Estimate"].item()
    source_ate_se = source_res["ATE"]["SE"].item()

    assert_allclose(our_result.ate.estimate, source_ate, rtol=5e-3)
    assert_allclose(our_result.ate.std_error, source_ate_se, rtol=1e-2)


def test_clustered_se(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=3,
        placebo=0,
        cluster="state_n",
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=0,
        cluster="state_n",
    )

    assert_allclose(
        our_result.effects.estimates,
        source_res["Effects"]["Estimate"].to_numpy(),
        rtol=1e-5,
    )
    assert_allclose(
        our_result.effects.std_errors,
        source_res["Effects"]["SE"].to_numpy(),
        rtol=1e-2,
    )


def test_less_conservative_se(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=2,
        placebo=0,
        less_conservative_se=True,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        placebo=0,
        less_conservative_se=True,
    )

    assert_allclose(
        our_result.effects.std_errors,
        source_res["Effects"]["SE"].to_numpy(),
        rtol=1e-2,
    )


def test_effects_equal(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=3,
        placebo=0,
        effects_equal=True,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=3,
        placebo=0,
        effects_equal=True,
    )

    source_pval = source_res["p_equality_effects"]
    assert_allclose(our_result.effects_equal_test["p_value"], source_pval, rtol=1e-2)


def test_trends_lin(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=2,
        placebo=0,
        trends_lin=True,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        placebo=0,
        trends_lin=True,
    )

    assert_allclose(
        our_result.effects.estimates,
        source_res["Effects"]["Estimate"].to_numpy(),
        rtol=1e-5,
    )


def test_n_switchers(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=4,
        placebo=0,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=4,
        placebo=0,
    )

    assert_allclose(
        our_result.effects.n_switchers,
        source_res["Effects"]["Switchers"].to_numpy(),
        rtol=1e-5,
    )


def test_placebo_joint_test(favara_imbs_data):
    source_res = run_source_package(
        favara_imbs_data,
        outcome="Dl_vloans_b",
        group="county",
        time="year",
        treatment="inter_bra",
        effects=2,
        placebo=3,
    )

    our_result = md.did_multiplegt(
        favara_imbs_data,
        yname="Dl_vloans_b",
        idname="county",
        tname="year",
        dname="inter_bra",
        effects=2,
        placebo=3,
    )

    if "Placebos_joint_test" in source_res:
        source_pval = source_res["Placebos_joint_test"]["p-value"].item()
        assert_allclose(our_result.placebo_joint_test["p_value"], source_pval, rtol=1e-4)


@pytest.mark.parametrize(
    "param,value,match",
    [
        ("effects", 0, "effects=0 is not valid"),
        ("effects", -1, "effects=-1 is not valid"),
        ("placebo", -1, "placebo=-1 is not valid"),
        ("switchers", "bad", "switchers='bad' is not valid"),
        ("ci_level", 0, "ci_level=0 is not valid"),
        ("ci_level", 100, "ci_level=100 is not valid"),
        ("ci_level", -5, "ci_level=-5 is not valid"),
    ],
)
def test_did_multiplegt_invalid_params(favara_imbs_data, param, value, match):
    with pytest.raises(ValueError, match=match):
        md.did_multiplegt(
            favara_imbs_data,
            yname="Dl_vloans_b",
            idname="county",
            tname="year",
            dname="inter_bra",
            **{param: value},
        )
