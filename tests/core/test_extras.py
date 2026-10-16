"""Tests for the errors raised when the packages of an extra are missing."""

from importlib.util import find_spec

import pytest

import moderndid


@pytest.mark.skipif(find_spec("plotnine") is not None, reason="plotnine is installed")
def test_plot_gt_without_plotnine_names_the_missing_module(att_gt_analytical):
    with pytest.raises(ImportError, match=r"'plotnine' module that 'plot_gt' needs.*moderndid\[plots\]"):
        moderndid.plot_gt(att_gt_analytical)


@pytest.mark.skipif(find_spec("cvxpy") is not None, reason="cvxpy is installed")
def test_honest_did_without_cvxpy_names_the_missing_module(aggte_dynamic):
    with pytest.raises(ImportError, match=r"'cvxpy' module that 'honest_did' needs.*moderndid\[didhonest\]"):
        moderndid.honest_did(aggte_dynamic, event_time=0)


@pytest.mark.skipif(find_spec("pyfixest") is not None, reason="pyfixest is installed")
def test_etwfe_without_pyfixest_names_the_etwfe_extra(mpdta):
    with pytest.raises(ImportError, match=r"'pyfixest' module that 'etwfe' needs.*moderndid\[etwfe\]"):
        moderndid.etwfe(mpdta, yname="lemp", tname="year", gname="first.treat", idname="countyreal")


@pytest.mark.skipif(find_spec("sklearn") is not None, reason="scikit-learn is installed")
def test_dyn_balancing_without_scikit_learn_names_the_diddynamic_extra():
    with pytest.raises(ImportError, match=r"'sklearn' module that 'dyn_balancing' needs.*moderndid\[diddynamic\]"):
        moderndid.dyn_balancing(
            moderndid.load_acemoglu(),
            yname="Y",
            tname="Time",
            idname="Unit",
            treatment_name="D",
            ds1=[1, 1],
            ds2=[0, 0],
        )
