"""Tests for the multi-period DDD repeated cross-section estimator."""

import re
import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import ddd
from moderndid.didtriple.estimators.ddd_mp_rc import ddd_mp_rc


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_ddd_mp_rc_basic(mp_rcs_data, est_method):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method=est_method,
    )

    assert len(result.att) > 0
    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n == len(mp_rcs_data)


def test_ddd_mp_rc_confidence_intervals(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    valid_mask = ~np.isnan(result.se)
    assert np.all(result.lci[valid_mask] < result.att[valid_mask])
    assert np.all(result.att[valid_mask] < result.uci[valid_mask])


def test_ddd_mp_rc_influence_functions(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert result.inf_func_mat is not None
    assert result.inf_func_mat.shape[0] == result.n
    assert result.inf_func_mat.shape[1] == len(result.att)


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_ddd_mp_rc_control_group(mp_rcs_data, control_group):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        control_group=control_group,
    )

    assert len(result.att) > 0
    assert result.args["control_group"] == control_group


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_rc_base_period(mp_rcs_data, base_period):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
    )

    assert len(result.att) > 0
    assert result.args["base_period"] == base_period


def test_ddd_mp_rc_glist_tlist(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert 3 in result.glist
    assert 4 in result.glist
    assert set(result.tlist) == {1, 2, 3, 4, 5}


def test_ddd_mp_rc_post_treatment_effects(mp_rcs_data):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    post_atts = [
        result.att[i]
        for i, (g, t) in enumerate(zip(result.groups, result.times))
        if t >= g and np.isfinite(result.att[i])
    ]
    assert len(post_atts) > 0
    assert all(0.0 < att < 5.0 for att in post_atts)


def test_ddd_mp_rc_never_treated_as_inf():
    rng = np.random.default_rng(42)
    n_per_period = 100
    time_periods = [1, 2, 3]

    records = []
    for t in time_periods:
        groups = rng.choice([np.inf, 3], size=n_per_period, p=[0.5, 0.5])
        partition = rng.choice([0, 1], size=n_per_period)

        for i in range(n_per_period):
            g = groups[i]
            p = partition[i]
            y = rng.normal(0, 1)
            if np.isfinite(g) and t >= g and p == 1:
                y += 1.5
            records.append({"id": len(records), "time": t, "y": y, "group": g, "partition": p})

    data = pl.DataFrame(records)

    result = ddd_mp_rc(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
    )

    assert len(result.att) > 0
    assert 3 in result.glist


def test_ddd_mp_rc_reproducibility(mp_rcs_data):
    result1 = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        biters=20,
        random_state=123,
    )

    result2 = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        biters=20,
        random_state=123,
    )

    assert np.allclose(result1.att, result2.att)
    valid_mask = ~np.isnan(result1.se) & ~np.isnan(result2.se)
    assert np.allclose(result1.se[valid_mask], result2.se[valid_mask])


@pytest.mark.parametrize("cband", [False, True])
def test_ddd_mp_rc_bootstrap(mp_rcs_data, cband):
    result = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        boot=True,
        biters=50,
        cband=cband,
        random_state=42,
    )

    valid_mask = ~np.isnan(result.se)
    assert np.sum(valid_mask) > 0
    assert np.all(result.lci[valid_mask] < result.att[valid_mask])
    assert np.all(result.att[valid_mask] < result.uci[valid_mask])


def _ddd_mp_rc_kwargs(data, n_jobs):
    return {
        "data": data,
        "y_col": "y",
        "time_col": "time",
        "id_col": "id",
        "group_col": "group",
        "partition_col": "partition",
        "n_jobs": n_jobs,
    }


def _check_ddd_mp_rc_result(result, data):
    assert len(result.att) > 0
    assert len(result.att) == len(result.se)
    assert len(result.att) == len(result.groups)
    assert len(result.att) == len(result.times)
    assert result.n == len(data)
    assert result.inf_func_mat.shape == (result.n, len(result.att))
    valid = ~np.isnan(result.se)
    assert np.all(result.lci[valid] < result.att[valid])
    assert np.all(result.att[valid] < result.uci[valid])


@pytest.mark.benchmark
def test_benchmark_ddd_mp_rc_sequential(benchmark, mp_rcs_data):
    result = benchmark.pedantic(ddd_mp_rc, kwargs=_ddd_mp_rc_kwargs(mp_rcs_data, n_jobs=1), rounds=3, warmup_rounds=1)
    _check_ddd_mp_rc_result(result, mp_rcs_data)


@pytest.mark.benchmark
def test_benchmark_ddd_mp_rc_parallel(benchmark, mp_rcs_data):
    result = benchmark.pedantic(ddd_mp_rc, kwargs=_ddd_mp_rc_kwargs(mp_rcs_data, n_jobs=-1), rounds=3, warmup_rounds=1)
    _check_ddd_mp_rc_result(result, mp_rcs_data)


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_rc_parallel_matches_sequential(mp_rcs_data, base_period):
    result_seq = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
        n_jobs=1,
    )

    result_par = ddd_mp_rc(
        data=mp_rcs_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        base_period=base_period,
        n_jobs=2,
    )

    np.testing.assert_allclose(result_seq.att, result_par.att, rtol=1e-10)
    np.testing.assert_array_equal(result_seq.groups, result_par.groups)
    np.testing.assert_array_equal(result_seq.times, result_par.times)
    np.testing.assert_allclose(result_seq.se, result_par.se, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(result_seq.inf_func_mat, result_par.inf_func_mat, rtol=1e-10)


@pytest.mark.parametrize(
    ("data_fixture", "options"),
    [
        ("mp_rcs_data", {"alpha": 0.2}),
        ("mp_rcs_missing_outcome_df", {}),
        ("mp_rcs_no_never_treated_df", {"control_group": "notyettreated"}),
        ("mp_rcs_no_never_treated_df", {"control_group": "notyettreated", "base_period": "varying"}),
    ],
)
def test_ddd_mp_rc_matches_ddd_on_the_same_data(request, data_fixture, options):
    data = request.getfixturevalue(data_fixture)

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = ddd(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            panel=False,
            est_method="reg",
            **options,
        )
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        result = ddd_mp_rc(
            data=data,
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
            est_method="reg",
            **options,
        )

    assert result.n == expected.n
    np.testing.assert_array_equal(result.tlist, expected.tlist)
    np.testing.assert_array_equal(result.groups, expected.groups)
    np.testing.assert_array_equal(result.times, expected.times)
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    np.testing.assert_array_equal(result.lci, expected.lci)
    np.testing.assert_array_equal(result.inf_func_mat, expected.inf_func_mat)
    np.testing.assert_array_equal(result.unit_groups, expected.unit_groups)
    assert [str(w.message) for w in direct_warnings] == [str(w.message) for w in expected_warnings]


def test_ddd_mp_rc_without_an_observation_column_matches_ddd(mp_first_period_cohort_df):
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = ddd(
            data=mp_first_period_cohort_df,
            yname="y",
            tname="time",
            gname="group",
            pname="partition",
            panel=False,
            est_method="reg",
        )
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        result = ddd_mp_rc(
            data=mp_first_period_cohort_df,
            y_col="y",
            time_col="time",
            id_col=None,
            group_col="group",
            partition_col="partition",
            est_method="reg",
        )

    assert result.n == expected.n
    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    np.testing.assert_array_equal(result.inf_func_mat, expected.inf_func_mat)
    assert [str(w.message) for w in direct_warnings] == [str(w.message) for w in expected_warnings]
    assert any("observations that were already treated in the first period" in str(w.message) for w in direct_warnings)


def test_ddd_mp_rc_normalizes_weights_like_ddd(mp_rcs_weighted_df):
    expected = ddd(
        data=mp_rcs_weighted_df,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        weightsname="w",
        panel=False,
    )
    result = ddd_mp_rc(
        data=mp_rcs_weighted_df,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        weights_col="w",
    )

    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)
    np.testing.assert_array_equal(result.unit_weights, expected.unit_weights)
    np.testing.assert_allclose(result.unit_weights.mean(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("base_period", ["universal", "varying"])
def test_ddd_mp_rc_without_never_treated_units_estimates_no_cell_for_the_latest_cohort(
    mp_rcs_no_never_treated_df, base_period
):
    spec = {
        "y_col": "y",
        "time_col": "time",
        "id_col": "id",
        "group_col": "group",
        "partition_col": "partition",
        "control_group": "notyettreated",
        "base_period": base_period,
        "est_method": "reg",
    }
    result = ddd_mp_rc(data=mp_rcs_no_never_treated_df, **spec)
    trimmed = ddd_mp_rc(data=mp_rcs_no_never_treated_df.filter(pl.col("time") < 4), **spec)

    assert result.n == trimmed.n
    np.testing.assert_array_equal(result.glist, [2, 3])
    np.testing.assert_array_equal(result.groups, trimmed.groups)
    np.testing.assert_array_equal(result.times, trimmed.times)
    np.testing.assert_allclose(result.att, trimmed.att, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.se, trimmed.se, rtol=1e-12, atol=1e-12)


def test_ddd_mp_rc_without_never_treated_units_needs_a_cohort_besides_the_latest(mp_rcs_no_never_treated_df):
    with pytest.raises(ValueError, match=re.escape("No cohort is left to estimate.")):
        ddd_mp_rc(
            data=mp_rcs_no_never_treated_df.filter(pl.col("group") == 4),
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
            control_group="notyettreated",
            est_method="reg",
        )


def test_ddd_mp_rc_warns_about_panel_data_with_its_own_argument_names(multi_period_df):
    with pytest.warns(UserWarning, match=re.escape("panel=False was specified, but units appear across all time")):
        expected = ddd(
            data=multi_period_df,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            panel=False,
            est_method="reg",
        )
    with pytest.warns(UserWarning, match=re.escape("Units in id_col='id' appear in every period. For panel data, use")):
        result = ddd_mp_rc(
            data=multi_period_df,
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
            est_method="reg",
        )

    np.testing.assert_array_equal(result.att, expected.att)
    np.testing.assert_array_equal(result.se, expected.se)


@pytest.mark.parametrize(
    ("change", "ddd_options", "direct_options", "ddd_message", "direct_message"),
    [
        (
            pl.col("y"),
            {"yname": "yy"},
            {"y_col": "yy"},
            "yname='yy' is not a column in the data. Did you mean 'y'?",
            "y_col='yy' is not a column in the data. Did you mean 'y'?",
        ),
        (
            pl.col("id").cast(pl.String),
            {},
            {},
            "idname='id' is not numeric. Please convert it.",
            "id_col='id' is not numeric. Please convert it.",
        ),
        (
            pl.col("partition") + 1,
            {},
            {},
            "pname='partition' must be 1 for eligible units and 0 for ineligible units",
            "partition_col='partition' must be 1 for eligible units and 0 for ineligible units",
        ),
        (
            pl.when(pl.col("id") == 3).then(-1).otherwise(pl.col("group")).alias("group"),
            {},
            {},
            "gname = 'group' holds negative values such as -1.",
            "group_col = 'group' holds negative values such as -1.",
        ),
        (
            pl.when(pl.col("group") == 0).then(3).otherwise(pl.col("group")).alias("group"),
            {},
            {},
            "There is no available never-treated group.",
            "There is no available never-treated group.",
        ),
        (
            pl.col("y"),
            {"est_method": "foo"},
            {"est_method": "foo"},
            "est_method='foo' is not valid.",
            "est_method='foo' is not valid.",
        ),
    ],
)
def test_ddd_mp_rc_raises_the_errors_of_ddd_with_its_own_argument_names(
    mp_rcs_data, change, ddd_options, direct_options, ddd_message, direct_message
):
    data = mp_rcs_data.with_columns(change)
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "pname": "partition",
        "panel": False,
        "est_method": "reg",
    }
    direct = {
        "y_col": "y",
        "time_col": "time",
        "id_col": "id",
        "group_col": "group",
        "partition_col": "partition",
        "est_method": "reg",
    }

    with pytest.raises(ValueError, match=re.escape(ddd_message)):
        ddd(data=data, **(spec | ddd_options))
    with pytest.raises(ValueError, match=re.escape(direct_message)):
        ddd_mp_rc(data=data, **(direct | direct_options))
