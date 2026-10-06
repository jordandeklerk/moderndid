"""Tests for the ML doubly robust DiD estimator."""

import re

import numpy as np
import pytest

from tests.helpers import importorskip

importorskip("moderndid.didml")

from moderndid import att_gt, didml


def test_didml_balanced_panel(mpdta_data, mpdta_spec, didml_options):
    result = didml(mpdta_data, **mpdta_spec, **didml_options)
    reference = att_gt(
        mpdta_data, est_method="dr", control_group="notyettreated", boot=False, cband=False, **mpdta_spec
    )

    assert result.n_units == 500
    assert result.estimation_params["n_obs"] == 2500
    assert result.estimation_params["cohort_counts"][2007.0] == 131
    np.testing.assert_array_equal(result.groups, reference.groups)
    np.testing.assert_array_equal(result.times, reference.times)
    np.testing.assert_allclose(result.drdid_benchmark, reference.att_gt, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.drdid_benchmark_se, reference.se_gt, rtol=1e-10)
    assert np.isfinite(result.att_gt).all()
    assert np.isfinite(result.se_gt).all()


def test_didml_unbalanced_panel_matches_complete_units(
    mpdta_unbalanced, mpdta_without_county_8001, mpdta_spec, didml_options
):
    with (
        pytest.warns(UserWarning, match="^1 units have unbalanced observations and will be dropped$"),
        pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"),
    ):
        result = didml(mpdta_unbalanced, **mpdta_spec, **didml_options)
    expected = didml(mpdta_without_county_8001, **mpdta_spec, **didml_options)

    assert result.n_units == 499
    assert result.estimation_params["n_obs"] == 2495
    assert result.estimation_params["cohort_counts"][2007.0] == 130
    np.testing.assert_array_equal(result.unit_ids, expected.unit_ids)
    np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_func, expected.influence_func, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.drdid_benchmark, expected.drdid_benchmark, rtol=1e-10, atol=0)


def test_didml_unbalanced_panel_benchmark_values(mpdta_unbalanced, mpdta_spec, didml_options):
    with (
        pytest.warns(UserWarning, match="^1 units have unbalanced observations and will be dropped$"),
        pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"),
    ):
        result = didml(mpdta_unbalanced, **mpdta_spec, **didml_options)

    np.testing.assert_allclose(
        result.drdid_benchmark,
        [
            -0.0214476431,
            -0.0819451635,
            -0.1384672740,
            -0.1069038981,
            -0.0078795766,
            -0.0046391636,
            0.0088056750,
            -0.0412938656,
            0.0278802898,
            -0.0039929236,
            -0.0288987794,
            -0.0294653715,
        ],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.drdid_benchmark_se,
        [
            0.0216370386,
            0.0283087981,
            0.0342008245,
            0.0328864930,
            0.0218314053,
            0.0182913483,
            0.0168826750,
            0.0197211441,
            0.0140059112,
            0.0156736865,
            0.0182171165,
            0.0163382674,
        ],
        rtol=0,
        atol=1e-9,
    )


def test_didml_without_never_treated_leaves_latest_cohort_out(mpdta_without_never_treated, mpdta_spec, didml_options):
    result = didml(mpdta_without_never_treated, **mpdta_spec, **didml_options)
    reference = att_gt(
        mpdta_without_never_treated,
        est_method="dr",
        control_group="notyettreated",
        boot=False,
        cband=False,
        **mpdta_spec,
    )

    assert set(result.groups.tolist()) == {2004.0, 2006.0}
    assert result.n_units == 191
    np.testing.assert_array_equal(result.groups, reference.groups)
    np.testing.assert_array_equal(result.times, reference.times)
    np.testing.assert_allclose(result.drdid_benchmark, reference.att_gt, rtol=0, atol=1e-12)
    assert np.isfinite(result.att_gt).all()


@pytest.mark.filterwarnings("error:.*unbalanced:UserWarning")
def test_didml_rejects_repeated_unit_periods(mpdta_repeated_row, mpdta_spec, didml_options):
    message = (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        "Rows repeat for the (countyreal, year) pair (17005, 2005)."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        didml(mpdta_repeated_row, **mpdta_spec, **didml_options)


def test_didml_drops_infinite_outcome_like_a_missing_one(
    mpdta_infinite_outcome, mpdta_without_county_8001, mpdta_spec, didml_options
):
    with (
        pytest.warns(UserWarning, match="^Dropped 1 rows from original data due to missing values$"),
        pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"),
    ):
        result = didml(mpdta_infinite_outcome, **mpdta_spec, **didml_options)
    expected = didml(mpdta_without_county_8001, **mpdta_spec, **didml_options)

    assert result.n_units == 499
    np.testing.assert_array_equal(result.unit_ids, expected.unit_ids)
    np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.drdid_benchmark, expected.drdid_benchmark, rtol=1e-10, atol=0)


def test_didml_rejects_weights_without_positive_mean(mpdta_zero_weights, mpdta_spec, didml_options):
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        didml(mpdta_zero_weights, weightsname="w", **mpdta_spec, **didml_options)
