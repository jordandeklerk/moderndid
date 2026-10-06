"""Tests for ETWFE internal computation functions."""

import numpy as np
import polars as pl
import pytest

from tests.helpers import importorskip

importorskip("pyfixest")

from moderndid.etwfe.compute import (
    _invlink_and_deriv,
    _match_rows_to_cells,
    _weighted_agg,
    build_etwfe_formula,
    clean_etwfe_data,
    prepare_etwfe_data,
    set_references,
    treatment_cells,
)


def test_set_references_auto_tref(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    assert config.tref == 2003


@pytest.mark.parametrize("cgroup,expected_gref", [("notyet", 0), ("never", 0)])
def test_set_references_auto_gref(mpdta_data, base_config, cgroup, expected_gref):
    base_config.cgroup = cgroup
    config = set_references(base_config, mpdta_data)
    assert config.gref == expected_gref


def test_set_references_respects_explicit_refs(mpdta_data, base_config):
    base_config.tref = 2004
    base_config.gref = 2006
    config = set_references(base_config, mpdta_data)
    assert config.tref == 2004
    assert config.gref == 2006


@pytest.mark.parametrize("gref,expected_flag", [(None, True), (2007, False)])
def test_set_references_gref_min_flag(mpdta_data, base_config, gref, expected_flag):
    if gref is None:
        base_config.cgroup = "never"
    else:
        base_config.gref = gref
    config = set_references(base_config, mpdta_data)
    assert config._gref_min_flag is expected_flag


def test_set_references_no_control_group_raises(base_config):
    df = pl.DataFrame({"g": [1, 2], "t": [1, 2], "y": [1.0, 2.0]})
    base_config.gname = "g"
    base_config.tname = "t"
    base_config.cgroup = "never"
    with pytest.raises(ValueError, match="Could not identify"):
        set_references(base_config, df)


def test_set_references_notyet_fallback_to_max_group(base_config):
    df = pl.DataFrame({"g": [2, 3, 3], "t": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    base_config.gname = "g"
    base_config.tname = "t"
    base_config.cgroup = "notyet"
    config = set_references(base_config, df)
    assert config.gref == 3
    assert config._gref_min_flag is False


@pytest.mark.parametrize("code", [0.0, float("inf"), 4.0, 9999.0])
def test_set_references_never_treated_codes_become_zero(base_config, code):
    df = pl.DataFrame({"g": [2.0, 3.0, code], "t": [1, 2, 3], "y": [1.0, 2.0, 3.0]}, strict=False)
    base_config.gname = "g"
    base_config.tname = "t"
    config = set_references(base_config, df)
    assert config.gref == 0
    assert config._gref_min_flag is True


def test_set_references_explicit_never_treated_code_becomes_zero(mpdta_never_codes, base_config):
    base_config.gref = 9999
    config = set_references(base_config, mpdta_never_codes["9999"])
    assert config.gref == 0


@pytest.mark.parametrize("gref", [0, 2008, 9999, float("inf")])
@pytest.mark.parametrize("label", ["inf", "9999"])
def test_set_references_accepts_any_never_treated_code_as_gref(mpdta_never_codes, base_config, label, gref):
    base_config.gref = gref
    config = set_references(base_config, mpdta_never_codes[label])
    assert config.gref == 0
    assert config._gref_min_flag is True


@pytest.mark.parametrize("gref", [0, 9999])
def test_set_references_never_treated_gref_without_never_treated_raises(mpdta_no_never, base_config, gref):
    base_config.gref = gref
    with pytest.raises(ValueError, match="refers to the never-treated units. No unit in 'first.treat'"):
        set_references(base_config, mpdta_no_never)


@pytest.mark.parametrize("tref", [1990, 2003.5])
def test_set_references_rejects_tref_outside_periods(mpdta_data, base_config, tref):
    base_config.tref = tref
    with pytest.raises(ValueError, match="is not a period in 'year'"):
        set_references(base_config, mpdta_data)


@pytest.mark.parametrize("gref", [2005, 2006.5])
def test_set_references_rejects_gref_outside_cohorts(mpdta_data, base_config, gref):
    base_config.gref = gref
    with pytest.raises(ValueError, match="is not a cohort in 'first.treat'"):
        set_references(base_config, mpdta_data)


@pytest.mark.parametrize("gref", [1999, 2003])
def test_set_references_rejects_gref_treated_by_first_period(mpdta_data, base_config, gref):
    base_config.gref = gref
    with pytest.raises(ValueError, match="not a cohort first treated after the first period"):
        set_references(base_config, mpdta_data)


@pytest.mark.parametrize(
    "no_never,cgroup,match",
    [
        (False, "notyet", r"Pass a later cohort or leave gref unset to use the never-treated units\.$"),
        (True, "notyet", r"Pass a later cohort or leave gref unset to use the latest cohort, 2007\.$"),
        (True, "never", r"2003\. Pass a later cohort\.$"),
    ],
)
def test_set_references_early_gref_message_names_the_default(
    mpdta_data, mpdta_no_never, base_config, no_never, cgroup, match
):
    base_config.gref = 2003
    base_config.cgroup = cgroup
    with pytest.raises(ValueError, match=match):
        set_references(base_config, mpdta_no_never if no_never else mpdta_data)


@pytest.mark.parametrize("label", ["inf", "9999"])
def test_prepare_codes_never_treated_units_as_zero(mpdta_never_codes, base_config, label):
    data = mpdta_never_codes[label]
    config = set_references(base_config, data)
    df = prepare_etwfe_data(data, config)
    never = df.filter(pl.col("_g") == 0)
    assert never.height == 1545
    assert (never["_Dtreat"] == 0.0).all()


@pytest.mark.parametrize("cgroup,gref", [("notyet", 2007), ("never", 2004)])
def test_prepare_never_treated_stay_untreated_with_treated_reference(mpdta_data, base_config, cgroup, gref):
    base_config.cgroup = cgroup
    base_config.gref = gref
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    never = df.filter(pl.col("_g") == 0)["_Dtreat"].drop_nulls()
    assert never.len() > 0
    assert (never == 0.0).all()
    assert all(g != 0 for g, _ in treatment_cells(df, config))


@pytest.mark.parametrize(
    "data_fixture,cgroup,match",
    [
        ("mpdta_calendar_gap", "never", r"Dropped 40 units of cohorts with no row in period g - 1 \(2006 lacks 2005\)"),
        ("mpdta_late_entry", "never", r"Dropped 40 units of cohorts with no row in period g - 1 \(2006 lacks 2005\)"),
        ("mpdta_late_entry", "notyet", r"Dropped 40 units of cohorts with no row before their first treated period"),
    ],
)
def test_prepare_drops_cohorts_without_untreated_rows(request, base_config, data_fixture, cgroup, match):
    data = request.getfixturevalue(data_fixture)
    base_config.cgroup = cgroup
    config = set_references(base_config, data)
    with pytest.warns(UserWarning, match=match):
        df = prepare_etwfe_data(data, config)
    assert 2006.0 not in df["_g"].to_list()
    assert df.height == data.filter(pl.col("first.treat") != 2006).height
    assert all(g != 2006.0 for g, _ in treatment_cells(df, config))


def test_prepare_keeps_cohorts_with_untreated_rows_after_a_gap(mpdta_calendar_gap, base_config):
    config = set_references(base_config, mpdta_calendar_gap)
    df = prepare_etwfe_data(mpdta_calendar_gap, config)
    assert df.height == mpdta_calendar_gap.height
    assert (2006.0, 2006.0) in treatment_cells(df, config)


@pytest.mark.parametrize(
    "column,kwargs",
    [
        ("lpop", {"xformla": "~ lpop"}),
        ("gls", {"xvar": "gls"}),
        ("w", {"weightsname": "w"}),
        ("year", {}),
        ("countyreal", {}),
        ("first.treat", {}),
    ],
)
def test_clean_drops_rows_with_missing_values(mpdta_missing, base_config, column, kwargs):
    data, missing = mpdta_missing
    for key, value in kwargs.items():
        setattr(base_config, key, value)
    holed = data.with_columns(pl.when(missing).then(None).otherwise(pl.col(column)).alias(column))
    with pytest.warns(UserWarning, match=f"Dropped 158 rows with missing values in {column}"):
        cleaned = clean_etwfe_data(holed, base_config)
    assert cleaned.equals(holed.filter(~missing))


def test_clean_drops_rows_with_nan_controls(mpdta_missing, base_config):
    data, missing = mpdta_missing
    base_config.xformla = "~ lpop"
    holed = data.with_columns(pl.when(missing).then(float("nan")).otherwise(pl.col("lpop")).alias("lpop"))
    with pytest.warns(UserWarning, match="Dropped 158 rows with missing values in lpop"):
        cleaned = clean_etwfe_data(holed, base_config)
    assert cleaned.height == data.height - 158


def test_clean_drops_rows_with_nan_cohorts(mpdta_missing, base_config):
    data, missing = mpdta_missing
    holed = data.with_columns(pl.when(missing).then(float("nan")).otherwise(pl.col("first.treat")).alias("first.treat"))
    with pytest.warns(UserWarning, match="Dropped 158 rows with missing values in first.treat"):
        cleaned = clean_etwfe_data(holed, base_config)
    assert cleaned.height == data.height - 158
    assert not cleaned["first.treat"].is_nan().any()


@pytest.mark.parametrize("vcov", [{"CRV1": "st"}, {"CRV1": "st+year"}, {"CRV3": "st"}])
def test_clean_drops_rows_with_missing_cluster_values(mpdta_states, base_config, vcov):
    data, missing = mpdta_states
    holed = data.with_columns(pl.when(missing).then(None).otherwise(pl.col("st")).alias("st"))
    with pytest.warns(UserWarning, match="Dropped 15 rows with missing values in st"):
        cleaned = clean_etwfe_data(holed, base_config, vcov)
    assert cleaned.equals(holed.filter(~missing))


@pytest.mark.parametrize("vcov", [None, "hetero", {"CRV1": "countyreal"}])
def test_clean_keeps_rows_missing_only_an_unused_cluster(mpdta_states, base_config, vcov):
    data, missing = mpdta_states
    holed = data.with_columns(pl.when(missing).then(None).otherwise(pl.col("st")).alias("st"))
    assert clean_etwfe_data(holed, base_config, vcov).equals(holed)


@pytest.mark.parametrize("idname", ["countyreal", None])
def test_clean_rejects_data_with_only_early_cohorts(mpdta_data, base_config, idname):
    base_config.idname = idname
    data = mpdta_data.with_columns(pl.lit(2003).alias("first.treat"))
    with (
        pytest.warns(UserWarning, match="already treated in the first period"),
        pytest.raises(ValueError, match="No rows are left"),
    ):
        clean_etwfe_data(data, base_config)


@pytest.mark.parametrize("idname,dropped", [("countyreal", "30 units"), (None, "150 rows")])
def test_clean_drops_cohorts_treated_in_first_period(mpdta_always_treated, base_config, idname, dropped):
    data, expected = mpdta_always_treated
    base_config.idname = idname
    with pytest.warns(UserWarning, match=f"Dropped {dropped} of cohorts already treated in the first period"):
        cleaned = clean_etwfe_data(data, base_config)
    assert cleaned.equals(expected)


def test_clean_rejects_data_without_complete_rows(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    holed = mpdta_data.with_columns(pl.lit(None, dtype=pl.Float64).alias("lpop"))
    with pytest.warns(UserWarning, match="Dropped 2500 rows"), pytest.raises(ValueError, match="No rows are left"):
        clean_etwfe_data(holed, base_config)


def test_clean_keeps_complete_data(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    assert clean_etwfe_data(mpdta_data, base_config).equals(mpdta_data)


def test_clean_rejects_controls_missing_from_data(mpdta_data, base_config):
    base_config.xformla = "~ lpop + not_a_column"
    with pytest.raises(ValueError, match="not_a_column"):
        clean_etwfe_data(mpdta_data, base_config)


def test_clean_rejects_xformla_with_outcome(mpdta_data, base_config):
    base_config.xformla = "lemp ~ lpop"
    with pytest.raises(ValueError, match="has a left-hand side"):
        clean_etwfe_data(mpdta_data, base_config)


@pytest.mark.parametrize("xformla", ["~ I(lpop**2)", "~ np.log(lpop)", "~ C(treat)", "~ lpop*treat", "~ lpop:treat"])
def test_clean_rejects_transformed_controls(mpdta_data, base_config, xformla):
    base_config.xformla = xformla
    with pytest.raises(ValueError, match="is not a column name"):
        clean_etwfe_data(mpdta_data, base_config)


@pytest.mark.parametrize("xformla", ["~ log.pop", "~ `log pop`"])
def test_prepare_reads_unparseable_controls_through_copies(mpdta_renamed, base_config, xformla):
    base_config.xformla = xformla
    config = set_references(base_config, mpdta_renamed)
    df = prepare_etwfe_data(mpdta_renamed, config)
    formula = build_etwfe_formula(config, df)
    assert config._ctrls == ["__etwfe_x0"]
    np.testing.assert_array_equal(df["__etwfe_x0"].to_numpy(), df["lpop"].to_numpy())
    assert "C(__etwfe_tcat):__etwfe_x0" in formula
    assert "log" not in formula


def test_prepare_reads_unparseable_moderator_labels_through_copies(mpdta_renamed, base_config):
    base_config.xvar = "popcat spaced"
    config = set_references(base_config, mpdta_renamed)
    prepare_etwfe_data(mpdta_renamed, config)
    assert config._xvar_dm_cols == ["__etwfe_m0_xdm", "__etwfe_m1_xdm"]


def test_formula_reads_unparseable_outcome_and_unit_through_copies(mpdta_renamed, base_config):
    base_config.yname = "l-emp"
    base_config.idname = "county id"
    config = set_references(base_config, mpdta_renamed)
    df = prepare_etwfe_data(mpdta_renamed, config)
    formula = build_etwfe_formula(config, df)
    assert formula.startswith("__etwfe_y ~ ")
    assert formula.endswith("| __etwfe_id + _t")


def test_prepare_notyet_treatment_indicator(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    treated = df.filter((pl.col("_g") == 2004) & (pl.col("_t") == 2004))["_Dtreat"]
    assert (treated == 1.0).all()

    untreated = df.filter((pl.col("_g") == 2006) & (pl.col("_t") == 2004))["_Dtreat"]
    assert (untreated == 0.0).all()


def test_prepare_notyet_drops_ref_cohort_periods(mpdta_data, base_config):
    base_config.gref = 2007
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    ref_at_treat = df.filter(pl.col("_t") >= 2007)["_Dtreat"]
    assert ref_at_treat.is_null().all()


def test_prepare_notyet_control_group_untreated(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    control = df.filter((pl.col("_g") == 2007) & (pl.col("_t") < 2007))["_Dtreat"]
    assert (control == 0.0).all()


def test_prepare_never_treatment_indicator(mpdta_data, base_config):
    base_config.cgroup = "never"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    never_treated = df.filter(pl.col("_g") == 0)["_Dtreat"]
    assert (never_treated == 0.0).all()

    pre_period = df.filter((pl.col("_g") == 2004) & (pl.col("_t") == 2003))["_Dtreat"]
    assert (pre_period == 0.0).all()

    post_period = df.filter((pl.col("_g") == 2004) & (pl.col("_t") == 2004))["_Dtreat"]
    assert (post_period == 1.0).all()


def test_prepare_creates_internal_columns(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    for col in ("_g", "_t", "_Dtreat"):
        assert col in df.columns


def test_prepare_internal_columns_are_float64(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert df["_g"].dtype == pl.Float64
    assert df["_t"].dtype == pl.Float64


def test_prepare_demeans_controls(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)

    assert "lpop_dm" in df.columns
    cohort_means = df.group_by("first.treat").agg(pl.col("lpop_dm").mean().alias("mean_dm"))
    assert np.allclose(cohort_means["mean_dm"].to_numpy(), 0.0, atol=1e-10)


def test_prepare_no_controls_returns_empty_ctrls(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    prepare_etwfe_data(mpdta_data, config)
    assert config._ctrls == []


def test_formula_lists_only_treated_cells(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert "C(__etwfe_gcat):C(__etwfe_tcat)" not in formula
    assert formula.count("__etwfe_cell_") == 7
    assert "_Dtreat:(__etwfe_cell_2004_2004 + " in formula


def test_treatment_cells_notyet_ordered_by_time(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert treatment_cells(df, config) == [
        (2004.0, 2004.0),
        (2004.0, 2005.0),
        (2004.0, 2006.0),
        (2006.0, 2006.0),
        (2004.0, 2007.0),
        (2006.0, 2007.0),
        (2007.0, 2007.0),
    ]


def test_treatment_cells_never_skip_reference_period(mpdta_data, base_config):
    base_config.cgroup = "never"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    cells = treatment_cells(df, config)
    assert len(cells) == 12
    assert all(t != g - 1 for g, t in cells)
    assert all(g != 0 for g, _ in cells)


def test_prepare_adds_cell_indicators(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert df["__etwfe_cell_2004_2004"].sum() == 20
    assert df["__etwfe_cell_2007_2007"].sum() == 131
    assert df.filter(pl.col("__etwfe_cell_2006_2007") == 1)["first.treat"].unique().to_list() == [2006]


def test_build_formula_without_treated_cells_raises(mpdta_data, base_config):
    data = mpdta_data.filter(pl.col("first.treat") == 0)
    config = set_references(base_config, data)
    df = prepare_etwfe_data(data, config)
    with pytest.raises(ValueError, match="No treated cohort-time cells"):
        build_etwfe_formula(config, df)


def test_formula_writes_controls_before_cells(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert formula.index("C(__etwfe_tcat):lpop") < formula.index("_Dtreat:")


@pytest.mark.parametrize(
    "fe,idname,has_pipe,has_explicit_cats",
    [
        ("vs", "countyreal", True, False),
        ("feo", "countyreal", True, False),
        ("none", None, False, True),
    ],
)
def test_formula_fe_modes(mpdta_data, base_config, fe, idname, has_pipe, has_explicit_cats):
    base_config.fe = fe
    base_config.idname = idname
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert ("|" in formula) == has_pipe
    assert ("C(__etwfe_gcat) +" in formula or "C(__etwfe_gcat)" in formula.split("+")[-1]) == has_explicit_cats


def test_formula_vs_absorbs_idname(mpdta_data, base_config):
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert formula.endswith("| __etwfe_id + _t")
    np.testing.assert_array_equal(df["__etwfe_id"].to_numpy(), df["countyreal"].to_numpy())


def test_formula_absorbs_cohort_without_idname(mpdta_data, base_config):
    base_config.idname = None
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert formula.endswith("| _g + _t")
    assert "__etwfe_id" not in df.columns


def test_formula_with_controls(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert "lpop_dm" in formula


def test_formula_feo_with_controls_explicit(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    base_config.fe = "feo"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert "C(__etwfe_gcat):lpop" in formula
    assert "C(__etwfe_tcat):lpop" in formula


def test_formula_vs_includes_control_interactions(mpdta_data, base_config):
    base_config.xformla = "~ lpop"
    base_config.fe = "vs"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert "C(__etwfe_gcat):lpop" in formula
    assert "C(__etwfe_tcat):lpop" in formula


def test_prepare_xvar_creates_dm_columns(mpdta_data, base_config):
    base_config.xvar = "lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert "lpop_xdm" in df.columns
    assert len(config._xvar_dm_cols) > 0


def test_prepare_xvar_creates_time_dummies(mpdta_data, base_config):
    base_config.xvar = "lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    assert len(config._xvar_time_dummies) > 0
    for col in config._xvar_time_dummies:
        assert col in df.columns


def test_formula_with_xvar(mpdta_data, base_config):
    base_config.xvar = "lpop"
    config = set_references(base_config, mpdta_data)
    df = prepare_etwfe_data(mpdta_data, config)
    formula = build_etwfe_formula(config, df)
    assert "lpop_xdm" in formula
    for td in config._xvar_time_dummies:
        assert td in formula


def test_prepare_xvar_demeans_within_cohort_time_cells(mpdta_moderators, base_config):
    base_config.xvar = "gls"
    config = set_references(base_config, mpdta_moderators)
    df = prepare_etwfe_data(mpdta_moderators, config)
    cell_means = df.group_by("first.treat", "year").agg(pl.col("gls_xdm").mean())
    np.testing.assert_allclose(cell_means["gls_xdm"].to_numpy(), 0.0, atol=1e-12)


@pytest.mark.parametrize("xvar", ["lpop", "lpop_aff"])
def test_prepare_xvar_spanned_by_controls_adds_no_terms(mpdta_moderators, base_config, xvar):
    base_config.xvar = xvar
    base_config.xformla = "~ lpop"
    config = set_references(base_config, mpdta_moderators)
    prepare_etwfe_data(mpdta_moderators, config)
    assert config._xvar_dm_cols == []
    assert config._xvar_time_dummies == []


@pytest.mark.parametrize("control", ["lpop", "x_tv"])
def test_prepare_xvar_equal_to_control_keeps_cohort_demeaned_control(mpdta_unbalanced_moderators, base_config, control):
    base_config.xvar = control
    base_config.xformla = f"~ {control}"
    config = set_references(base_config, mpdta_unbalanced_moderators)
    df = prepare_etwfe_data(mpdta_unbalanced_moderators, config)
    expected = df.select(pl.col(control) - pl.col(control).mean().over("first.treat")).to_series()
    np.testing.assert_allclose(df[f"{control}_dm"].to_numpy(), expected.to_numpy(), atol=1e-12)
    cell_means = df.group_by("first.treat", "year").agg(pl.col(f"{control}_xdm").mean())
    np.testing.assert_allclose(cell_means[f"{control}_xdm"].to_numpy(), 0.0, atol=1e-12)


def test_formula_skips_interactions_constant_within_cell(mpdta_moderators, base_config):
    base_config.xvar = "gls01"
    config = set_references(base_config, mpdta_moderators)
    df = prepare_etwfe_data(mpdta_moderators, config)
    formula = build_etwfe_formula(config, df)
    xvar_term = formula.split(" | ")[0].split(" + _Dtreat:")[-1]
    assert xvar_term.endswith(":gls01_xdm")
    assert "2004" not in xvar_term.split(":gls01_xdm")[0]


@pytest.mark.parametrize(
    "family,eta,expected_mu",
    [
        ("gaussian", np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        ("poisson", np.array([0.0, 1.0]), np.exp([0.0, 1.0])),
        ("logit", np.array([0.0]), np.array([0.5])),
        ("probit", np.array([0.0]), np.array([0.5])),
    ],
)
def test_invlink_and_deriv_mu(family, eta, expected_mu):
    mu, _ = _invlink_and_deriv(eta, family)
    np.testing.assert_allclose(mu, expected_mu, atol=1e-6)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "logit", "probit"])
def test_invlink_and_deriv_positive_derivative(family):
    eta = np.array([-1.0, 0.0, 1.0])
    _, deriv = _invlink_and_deriv(eta, family)
    assert np.all(deriv > 0)


def test_invlink_and_deriv_poisson_mu_equals_deriv():
    mu, deriv = _invlink_and_deriv(np.array([0.0, 1.0, -1.0]), "poisson")
    np.testing.assert_array_equal(mu, deriv)


def test_invlink_and_deriv_unsupported_family():
    with pytest.raises(ValueError, match="Unsupported family"):
        _invlink_and_deriv(np.array([0.0]), "invalid")


def test_weighted_agg_zero_weights():
    slopes = np.array([1.0, 2.0])
    jac = np.eye(2)
    att, se = _weighted_agg(slopes, jac, np.array([0.0, 0.0]), np.eye(2))
    assert att == 0.0
    assert np.isnan(se)


def test_weighted_agg_none_vcov():
    slopes = np.array([1.0, 2.0])
    jac = np.eye(2)
    att, se = _weighted_agg(slopes, jac, np.ones(2), None)
    np.testing.assert_allclose(att, 1.5)
    assert np.isnan(se)


def test_weighted_agg_nan_slope_gives_nan():
    att, se = _weighted_agg(np.array([1.0, np.nan]), np.eye(2), np.ones(2), np.eye(2))
    assert np.isnan(att)
    assert np.isnan(se)


def test_match_rows_to_cells_keeps_row_order():
    df = pl.DataFrame({"_g": [2006.0, 0.0, 2004.0, 2006.0, 2004.0], "_t": [2007.0, 2005.0, 2004.0, 2006.0, 2004.0]})
    cells = [(2004.0, 2004.0), (2006.0, 2006.0), (2006.0, 2007.0)]
    np.testing.assert_array_equal(_match_rows_to_cells(df, cells), [2, -1, 0, 1, 0])
