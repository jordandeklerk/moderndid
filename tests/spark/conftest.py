"""Shared fixtures for the tests of att_gt on Spark DataFrames."""

import os
import shutil
import sys
import warnings
from importlib.util import find_spec

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from moderndid import att_gt, gen_did_scalable, load_mpdta
from moderndid.core.preprocess import DIDConfig

# Since pytest imports this file before it can skip anything when the folder is named on the command line, the test
# modules skip themselves without pyspark and this file imports it only when it is installed.
if find_spec("pyspark") is not None:
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema

    from moderndid.spark.preprocess import build_unit_table


@pytest.fixture(scope="session")
def spark():
    """Spark session on a local cluster or on the Spark Connect server that MODERNDID_SPARK_REMOTE names."""
    remote = os.environ.get("MODERNDID_SPARK_REMOTE")
    if not (remote or "").startswith("sc://") and not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        if "MODERNDID_REQUIRE_ALL_DEPS" in os.environ:
            pytest.fail("Spark needs a Java runtime. Set JAVA_HOME or put java on PATH.")
        pytest.skip("Spark needs a Java runtime on JAVA_HOME or PATH")
    # Since a persisted shuffle keeps every shuffle partition, eight partitions in place of 200 make a fit of a small
    # panel about three times faster without changing its results.
    builder = SparkSession.builder.config("spark.sql.shuffle.partitions", "8")
    with pytest.MonkeyPatch.context() as patch:
        # Since a Spark Connect server starts the Python workers that PYSPARK_PYTHON names, a local server runs the
        # interpreter of the tests on its executors.
        patch.setenv("PYSPARK_PYTHON", sys.executable)
        if remote:
            if remote.startswith("local") and int(pyspark.__version__.split(".")[0]) < 4:
                # Since PySpark 3.5 doesn't bundle the Spark Connect server, a local server loads it from Maven Central.
                version = pyspark.__version__
                builder = builder.config("spark.jars.packages", f"org.apache.spark:spark-connect_2.12:{version}")
                # Because the tasks of a Connect session in Spark 3.5 load classes only from the executor's classpath
                # and the session's own jars, the executors of a local cluster read the server jar from the Ivy folder.
                jar = os.path.join(
                    os.path.expanduser("~"), ".ivy2", "jars", f"org.apache.spark_spark-connect_2.12-{version}.jar"
                )
                builder = builder.config("spark.executor.extraClassPath", jar)
            session = builder.remote(remote).getOrCreate()
        else:
            session = (
                builder.master("local-cluster[2,1,1024]")
                .config("spark.pyspark.python", sys.executable)
                .config("spark.pyspark.driver.python", sys.executable)
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
        yield session
        session.stop()


@pytest.fixture(scope="session")
def to_spark(spark):
    """Function that turns a polars DataFrame into a Spark DataFrame through Arrow."""

    def convert(data):
        table = data.to_arrow()
        if int(pyspark.__version__.split(".")[0]) >= 4:
            return spark.createDataFrame(table)
        # Since PySpark 3.5 builds no DataFrame from a pyarrow Table, the rows go in with the table's Spark types.
        # Going through pandas instead would turn integer columns with nulls into floats.
        return spark.createDataFrame(table.to_pylist(), schema=from_arrow_schema(table.schema))

    return convert


@pytest.fixture(scope="session")
def to_pandas():
    """Function that collects a Spark DataFrame as a pandas DataFrame."""

    def collect(frame):
        return frame.toPandas()

    return collect


@pytest.fixture(scope="session")
def to_arrow():
    """Function that collects a Spark DataFrame as a pyarrow Table."""

    def collect(frame):
        if hasattr(frame, "toArrow"):
            return frame.toArrow()
        rows = [row.asDict(recursive=True) for row in frame.collect()]
        return pa.Table.from_pylist(rows, schema=to_arrow_schema(frame.schema))

    return collect


@pytest.fixture(scope="session")
def mpdta_data():
    """The minimum wage panel of 500 counties over five years."""
    return load_mpdta()


@pytest.fixture(scope="session")
def mpdta_with_states(mpdta_data):
    """mpdta with the state of each county as a cluster variable."""
    return mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))


@pytest.fixture(scope="session")
def mpdta_with_state_names(mpdta_with_states):
    """mpdta with the state of each county as a string."""
    return mpdta_with_states.with_columns(pl.col("state").cast(pl.String))


@pytest.fixture(scope="session")
def mpdta_pop_weighted(mpdta_data):
    """mpdta with the county population as a sampling weight."""
    return mpdta_data.with_columns(pl.col("lpop").exp().alias("pop"))


@pytest.fixture(scope="session")
def mpdta_with_weights_and_states(mpdta_with_state_names):
    """mpdta with sampling weights that vary over time and the state of each county as a string."""
    return mpdta_with_state_names.with_columns(
        (pl.col("lpop").exp() * (1 + (pl.col("countyreal") + pl.col("year")) % 4)).alias("w")
    )


@pytest.fixture(scope="session")
def mpdta_with_text_outcome(mpdta_data):
    """mpdta with the outcome stored as text."""
    return mpdta_data.with_columns(pl.col("lemp").cast(pl.String))


@pytest.fixture(scope="session")
def mpdta_with_repeated_rows(mpdta_data):
    """mpdta with the rows of two counties in the last two years repeated."""
    repeated = mpdta_data.filter(pl.col("countyreal").is_in([8001, 8019]) & (pl.col("year") >= 2006))
    return pl.concat([mpdta_data, repeated])


@pytest.fixture(scope="session")
def mpdta_with_varying_cohort(mpdta_data):
    """mpdta in which one county changes its treatment cohort in the last year."""
    changed = (pl.col("countyreal") == 8001) & (pl.col("year") == 2007)
    return mpdta_data.with_columns(pl.when(changed).then(2006).otherwise(pl.col("first.treat")).alias("first.treat"))


@pytest.fixture(scope="session")
def mpdta_missing_a_period(mpdta_data):
    """mpdta without the 2005 rows of the counties whose code is a multiple of nine."""
    return mpdta_data.filter(~((pl.col("countyreal") % 9 == 0) & (pl.col("year") == 2005)))


@pytest.fixture(scope="session")
def mpdta_treated_in_the_first_period(mpdta_data):
    """mpdta in which the never-treated counties whose code ends in three are treated from 2003."""
    early = (pl.col("countyreal") % 10 == 3) & (pl.col("first.treat") == 0)
    return mpdta_data.with_columns(pl.when(early).then(2003).otherwise(pl.col("first.treat")).alias("first.treat"))


@pytest.fixture(scope="session")
def mpdta_without_never_treated(mpdta_data):
    """mpdta without the counties that are never treated."""
    return mpdta_data.filter(pl.col("first.treat") > 0)


@pytest.fixture(scope="session")
def mpdta_with_float_periods(mpdta_data):
    """mpdta with the years and the county codes stored as floats."""
    return mpdta_data.with_columns(pl.col("year").cast(pl.Float64), pl.col("countyreal").cast(pl.Float64))


@pytest.fixture(scope="session")
def mpdta_with_decimal_columns(mpdta_data):
    """mpdta with the years and the county codes and cohorts stored as decimals of ten digits."""
    return mpdta_data.with_columns(pl.col("year", "countyreal", "first.treat").cast(pl.Decimal(10, 0)))


@pytest.fixture(scope="session")
def mpdta_with_large_ids(mpdta_data):
    """mpdta with consecutive county codes above 2**53."""
    return mpdta_data.with_columns((pl.col("countyreal").rank("dense").cast(pl.Int64) + 10**17).alias("countyreal"))


@pytest.fixture(scope="session")
def mpdta_with_large_decimal_ids(mpdta_with_large_ids):
    """mpdta with consecutive county codes above 2**53 stored as decimals of 18 digits."""
    return mpdta_with_large_ids.with_columns(pl.col("countyreal").cast(pl.Decimal(18, 0)))


@pytest.fixture(scope="session")
def mpdta_with_decimal_ids_of_19_digits(mpdta_data):
    """mpdta with the county codes stored as decimals of 19 digits."""
    return mpdta_data.with_columns(pl.col("countyreal").cast(pl.Decimal(19, 0)))


@pytest.fixture(scope="session")
def mpdta_with_fractional_decimal_ids(mpdta_data):
    """mpdta with the county codes stored as decimals with two fraction digits."""
    return mpdta_data.with_columns(pl.col("countyreal").cast(pl.Decimal(12, 2)))


@pytest.fixture(scope="session")
def mpdta_with_fractional_decimal_states(mpdta_with_states):
    """mpdta with the state of each county stored as a decimal with two fraction digits."""
    return mpdta_with_states.with_columns(pl.col("state").cast(pl.Decimal(10, 2)))


@pytest.fixture(scope="session")
def mpdta_with_decimal_values(mpdta_pop_weighted):
    """mpdta with the outcome and the covariate and the population weights stored as decimals."""
    return mpdta_pop_weighted.with_columns(pl.col("lemp", "lpop", "pop").cast(pl.Decimal(18, 6)))


@pytest.fixture(scope="session")
def scalable_panel():
    """Generated panel of 400 units over five periods with four covariates and a standardized outcome."""
    data = gen_did_scalable(400, n_periods=5, n_cohorts=3, n_covariates=4, random_state=42)["data"]
    return data.with_columns((pl.col("y") - pl.col("y").mean()) / pl.col("y").std())


@pytest.fixture(scope="session")
def scalable_panel_with_cluster_names(scalable_panel):
    """Generated panel with the cluster of each unit as a string."""
    return scalable_panel.with_columns(pl.col("cluster").cast(pl.String))


@pytest.fixture(scope="session")
def scalable_panel_with_decimal_clusters(scalable_panel):
    """Generated panel with the cluster of each unit as a decimal of 18 digits."""
    return scalable_panel.with_columns(pl.col("cluster").cast(pl.Decimal(18, 0)))


@pytest.fixture(scope="session")
def large_clustered_panel():
    """Generated panel of 12000 units over three periods with 50 clusters and a standardized outcome."""
    data = gen_did_scalable(12000, n_periods=3, n_cohorts=2, n_covariates=4, random_state=42)["data"]
    return data.with_columns((pl.col("y") - pl.col("y").mean()) / pl.col("y").std())


@pytest.fixture(scope="session")
def small_scalable_panel():
    """Generated panel of 50 units over three periods with a standardized outcome."""
    data = gen_did_scalable(50, n_periods=3, n_cohorts=2, n_covariates=4, random_state=42)["data"]
    return data.with_columns((pl.col("y") - pl.col("y").mean()) / pl.col("y").std())


@pytest.fixture(scope="session")
def universal_base_fits(mpdta_data, to_spark):
    """att_gt fits of mpdta with a universal base period on Spark and in memory."""
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": "universal",
        "biters": 499,
        "random_state": 7,
    }
    return att_gt(to_spark(mpdta_data), **spec), att_gt(mpdta_data, **spec)


@pytest.fixture
def unit_table_of(to_spark):
    """Function that builds the unit table of a panel on Spark and releases its cache after the test."""
    cached = []

    def build(data, **spec):
        config = DIDConfig(panel=True, allow_unbalanced_panel=False, **spec)
        *built, persisted = build_unit_table(to_spark(data), config)
        cached.append(persisted)
        return built

    yield build
    for table in cached:
        if table is not None:
            table.unpersist()


@pytest.fixture
def record_warnings():
    """Function that returns a call's result with its warnings except unclosed sockets and distutils deprecations."""

    def call(function, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Since classic PySpark 3.5 through 4.1 leave a socket open on each collect and PySpark 3.5 checks versions
            # with the deprecated distutils, those warnings come from PySpark rather than from the call.
            warnings.filterwarnings("ignore", message=r"unclosed <socket\.socket", category=ResourceWarning)
            warnings.filterwarnings(
                "ignore", message="distutils Version classes are deprecated", category=DeprecationWarning
            )
            result = function(*args, **kwargs)
        return result, [(warning.category, str(warning.message)) for warning in caught]

    return call


@pytest.fixture
def assert_same_fit():
    """Function that asserts that two att_gt results hold the same cells and estimates up to a tolerance."""

    def check(result, expected, tolerance):
        np.testing.assert_array_equal(result.groups, expected.groups)
        np.testing.assert_array_equal(result.times, expected.times)
        assert (result.groups.dtype, result.times.dtype) == (expected.groups.dtype, expected.times.dtype)
        assert result.n_units == expected.n_units
        np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.vcov_analytical, expected.vcov_analytical, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.critical_value, expected.critical_value, rtol=tolerance)
        wald = [np.nan if fit.wald_stat is None else fit.wald_stat for fit in (result, expected)]
        np.testing.assert_allclose(wald[0], wald[1], rtol=tolerance)

    return check
