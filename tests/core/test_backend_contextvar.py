"""Tests for ContextVar-based backend dispatch, use_backend, and context propagation."""

from __future__ import annotations

import numpy as np
import pytest

from moderndid.core.parallel import parallel_map
from moderndid.cupy.backend import get_backend, set_backend, use_backend


def _dask_import_ok() -> bool:
    try:
        import dask.dataframe  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


_requires_dask_compat = pytest.mark.skipif(
    not _dask_import_ok(),
    reason="Dask/CuPy compatibility issue in this environment",
)


class TestUseBackend:
    def setup_method(self):
        set_backend("numpy")

    def test_sets_and_reverts(self):
        assert get_backend() is np
        with use_backend("numpy"):
            assert get_backend() is np
        assert get_backend() is np

    def test_reverts_on_exception(self):
        assert get_backend() is np
        with pytest.raises(ZeroDivisionError), use_backend("numpy"):
            assert get_backend() is np
            1 / 0  # noqa: B018
        assert get_backend() is np

    def test_nested_reverts(self):
        assert get_backend() is np
        with use_backend("numpy"):
            assert get_backend() is np
            with use_backend("numpy"):
                assert get_backend() is np
            assert get_backend() is np
        assert get_backend() is np

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"), use_backend("tensorflow"):
            pass

    def test_does_not_mutate_outer_backend(self):
        set_backend("numpy")
        with use_backend("numpy"):
            pass
        assert get_backend() is np


class TestParallelContextPropagation:
    def setup_method(self):
        set_backend("numpy")

    def test_propagates_numpy_backend(self):
        def get_backend_name(_):
            return get_backend().__name__

        results = parallel_map(get_backend_name, [(i,) for i in range(4)], n_jobs=2)
        assert all(r == "numpy" for r in results)

    def test_sequential_uses_current_backend(self):
        def get_backend_name(_):
            return get_backend().__name__

        set_backend("numpy")
        results = parallel_map(get_backend_name, [(i,) for i in range(3)], n_jobs=1)
        assert all(r == "numpy" for r in results)


@_requires_dask_compat
class TestAttGtBackendParam:
    def setup_method(self):
        set_backend("numpy")

    def test_backend_numpy_no_mutation(self):
        from moderndid import att_gt, load_mpdta

        df = load_mpdta()
        assert get_backend() is np

        result = att_gt(
            data=df,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            est_method="reg",
            boot=False,
            backend="numpy",
        )

        # Backend must still be numpy after the call
        assert get_backend() is np
        assert result is not None

    def test_backend_none_uses_current(self):
        from moderndid import att_gt, load_mpdta

        df = load_mpdta()
        set_backend("numpy")

        result = att_gt(
            data=df,
            yname="lemp",
            tname="year",
            gname="first.treat",
            idname="countyreal",
            est_method="reg",
            boot=False,
            backend=None,
        )

        assert get_backend() is np
        assert result is not None


@_requires_dask_compat
class TestDddBackendParam:
    def setup_method(self):
        set_backend("numpy")

    def test_backend_numpy_no_mutation(self):
        from moderndid import ddd, gen_dgp_2periods

        dgp = gen_dgp_2periods(n=500, dgp_type=1, random_state=42)
        assert get_backend() is np

        result = ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla="~ cov1 + cov2 + cov3 + cov4",
            est_method="dr",
            backend="numpy",
        )

        assert get_backend() is np
        assert result is not None

    def test_backend_none_uses_current(self):
        from moderndid import ddd, gen_dgp_2periods

        dgp = gen_dgp_2periods(n=500, dgp_type=1, random_state=42)
        set_backend("numpy")

        result = ddd(
            data=dgp["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla="~ cov1 + cov2 + cov3 + cov4",
            est_method="dr",
            backend=None,
        )

        assert get_backend() is np
        assert result is not None
