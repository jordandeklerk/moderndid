"""Shared pure-numpy functions for distributed backends (Dask, Spark).

This package contains all framework-agnostic computation functions used by
both the ``dask`` and ``spark`` backends.  Every function here operates on
NumPy (or CuPy) arrays and has **zero** Dask / Spark dependencies.
"""
