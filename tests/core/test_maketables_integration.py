"""Integration tests verifying maketables tables match estimator output."""

import numpy as np
import pytest

from tests.helpers import importorskip

importorskip("formulaic")


def _assert_table_matches(table, estimates, se, *, atol=1e-10):
    np.testing.assert_allclose(table["b"].values, estimates, atol=atol)
    np.testing.assert_allclose(table["se"].values, se, atol=atol)


def _assert_ci(table, ci95l, ci95u, *, atol=1e-10):
    np.testing.assert_allclose(table["ci95l"].values, ci95l, atol=atol)
    np.testing.assert_allclose(table["ci95u"].values, ci95u, atol=atol)


class TestAttGt:
    def test_estimates(self, att_gt_analytical):
        table = att_gt_analytical.__maketables_coef_table__
        _assert_table_matches(table, att_gt_analytical.att_gt, att_gt_analytical.se_gt)

    def test_ci_uses_critical_value(self, att_gt_analytical):
        table = att_gt_analytical.__maketables_coef_table__
        cv = att_gt_analytical.critical_value
        _assert_ci(
            table,
            att_gt_analytical.att_gt - cv * att_gt_analytical.se_gt,
            att_gt_analytical.att_gt + cv * att_gt_analytical.se_gt,
        )

    def test_n_units(self, att_gt_analytical):
        assert att_gt_analytical.__maketables_stat__("N") == att_gt_analytical.n_units

    def test_bootstrap_critical_value_differs_from_normal(self, att_gt_bootstrap):
        table = att_gt_bootstrap.__maketables_coef_table__
        cv = att_gt_bootstrap.critical_value
        assert cv != pytest.approx(1.96, abs=0.01)
        _assert_ci(
            table,
            att_gt_bootstrap.att_gt - cv * att_gt_bootstrap.se_gt,
            att_gt_bootstrap.att_gt + cv * att_gt_bootstrap.se_gt,
        )


class TestAggte:
    def test_overall_att(self, aggte_result):
        table = aggte_result.__maketables_coef_table__
        assert table.loc["Overall ATT", "b"] == pytest.approx(aggte_result.overall_att)
        assert table.loc["Overall ATT", "se"] == pytest.approx(aggte_result.overall_se)

    def test_overall_ci_uses_normal_z(self, aggte_result):
        from scipy import stats

        table = aggte_result.__maketables_coef_table__
        z = stats.norm.ppf(0.975)
        assert table.loc["Overall ATT", "ci95l"] == pytest.approx(
            aggte_result.overall_att - z * aggte_result.overall_se, abs=1e-4
        )
        assert table.loc["Overall ATT", "ci95u"] == pytest.approx(
            aggte_result.overall_att + z * aggte_result.overall_se, abs=1e-4
        )

    def test_event_ci_uses_critical_values(self, aggte_result):
        if aggte_result.event_times is None:
            pytest.skip("no event-level effects")
        from scipy import stats

        table = aggte_result.__maketables_coef_table__
        event_rows = table.iloc[1:]
        if aggte_result.critical_values is not None:
            crit = np.asarray(aggte_result.critical_values)
        else:
            crit = np.full(len(aggte_result.event_times), stats.norm.ppf(0.975))
        _assert_ci(
            event_rows,
            np.asarray(aggte_result.att_by_event) - crit * np.asarray(aggte_result.se_by_event),
            np.asarray(aggte_result.att_by_event) + crit * np.asarray(aggte_result.se_by_event),
        )

    def test_metadata(self, aggte_result):
        assert aggte_result.__maketables_stat__("aggregation") == aggte_result.aggregation_type
        assert aggte_result.__maketables_stat__("se_type") in ("Analytical", "Bootstrap")
        assert aggte_result.__maketables_stat__("control_group") == "Never Treated"


class TestDRDID:
    def test_estimates(self, drdid_result):
        table = drdid_result.__maketables_coef_table__
        assert table.loc["ATT", "b"] == pytest.approx(drdid_result.att)
        assert table.loc["ATT", "se"] == pytest.approx(drdid_result.se)

    def test_ci_matches_precomputed(self, drdid_result):
        table = drdid_result.__maketables_coef_table__
        assert table.loc["ATT", "ci95l"] == pytest.approx(drdid_result.lci)
        assert table.loc["ATT", "ci95u"] == pytest.approx(drdid_result.uci)

    def test_n(self, drdid_result):
        assert drdid_result.__maketables_stat__("N") is not None


class TestDDDMP:
    def test_estimates(self, ddd_mp_result):
        table = ddd_mp_result.__maketables_coef_table__
        _assert_table_matches(table, ddd_mp_result.att, ddd_mp_result.se)

    def test_ci_matches_precomputed(self, ddd_mp_result):
        table = ddd_mp_result.__maketables_coef_table__
        _assert_ci(table, ddd_mp_result.lci, ddd_mp_result.uci)

    def test_n(self, ddd_mp_result):
        assert ddd_mp_result.__maketables_stat__("N") == ddd_mp_result.n


class TestDDDAgg:
    def test_overall_att(self, ddd_agg_result):
        table = ddd_agg_result.__maketables_coef_table__
        assert table.loc["Overall ATT", "b"] == pytest.approx(ddd_agg_result.overall_att)
        assert table.loc["Overall ATT", "se"] == pytest.approx(ddd_agg_result.overall_se)

    def test_event_ci_uses_crit_val(self, ddd_agg_result):
        table = ddd_agg_result.__maketables_coef_table__
        event_rows = table.iloc[1:]
        cv = ddd_agg_result.crit_val
        _assert_ci(
            event_rows,
            np.asarray(ddd_agg_result.att_egt) - cv * np.asarray(ddd_agg_result.se_egt),
            np.asarray(ddd_agg_result.att_egt) + cv * np.asarray(ddd_agg_result.se_egt),
        )


class TestDIDInter:
    def test_effect_estimates(self, didinter_result):
        table = didinter_result.__maketables_coef_table__
        for i, h in enumerate(didinter_result.effects.horizons):
            row = table.loc[f"Effect h={int(h)}"]
            assert row["b"] == pytest.approx(didinter_result.effects.estimates[i])
            assert row["se"] == pytest.approx(didinter_result.effects.std_errors[i])

    def test_effect_ci_matches_precomputed(self, didinter_result):
        table = didinter_result.__maketables_coef_table__
        for i, h in enumerate(didinter_result.effects.horizons):
            row = table.loc[f"Effect h={int(h)}"]
            assert row["ci95l"] == pytest.approx(didinter_result.effects.ci_lower[i])
            assert row["ci95u"] == pytest.approx(didinter_result.effects.ci_upper[i])

    def test_placebo_ci_matches_precomputed(self, didinter_result):
        table = didinter_result.__maketables_coef_table__
        for i, h in enumerate(didinter_result.placebos.horizons):
            row = table.loc[f"Placebo h={int(h)}"]
            assert row["ci95l"] == pytest.approx(didinter_result.placebos.ci_lower[i])
            assert row["ci95u"] == pytest.approx(didinter_result.placebos.ci_upper[i])


class TestContDID:
    def test_has_coef_table(self, cont_did_result):
        table = cont_did_result.__maketables_coef_table__
        assert {"b", "se", "ci95l", "ci95u"}.issubset(set(table.columns))

    def test_depvar(self, cont_did_result):
        assert cont_did_result.__maketables_depvar__ is not None
