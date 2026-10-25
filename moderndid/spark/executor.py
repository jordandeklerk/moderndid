"""Execution of the passes of a sharded att_gt fit on a Spark unit table."""

import pickle
import sys

import pyarrow as pa
from pyspark import TaskContext
from pyspark.sql import functions as F
from pyspark.sql import types as T

from moderndid import _version
from moderndid.did.shards.partials import combine_partials, empty_partial, finalize_keyed


class SparkExecutor:
    """Run each pass of a sharded fit as one Spark action over the unit table.

    Every task applies the kernel to its partition of the unit table and
    returns one row of statistics unless its partition is empty. With
    ``whole_clusters``, each task finalizes its keyed sums first. Otherwise the
    driver finalizes them after combining every row. On the first pass and
    whenever the tasks of the previous pass returned more than 16 rows, a
    second stage combines the rows in groups before the driver collects them.
    The executor counts its passes in ``n_passes``.

    Parameters
    ----------
    unit_table : pyspark.sql.DataFrame
        Table with one row for each unit in the layout that
        :func:`~moderndid.did.shards.layout.shard_from_arrow` reads.
    n_periods : int
        Number of time periods T.
    n_covariates : int
        Number of covariates p, not counting the intercept.
    whole_clusters : bool, default=False
        Whether each partition holds all the units of its clusters.
    """

    def __init__(self, unit_table, n_periods, n_covariates, whole_clusters=False):
        self.unit_table = unit_table
        self.n_periods = n_periods
        self.n_covariates = n_covariates
        self.whole_clusters = whole_clusters
        self.n_passes = 0
        self._n_tasks = None

    def run(self, kernel, state):
        """Apply a kernel to every shard of the unit table and combine the statistics.

        When an executor can't import moderndid or runs a moderndid version or
        Python minor version that differs from the driver's, the pass raises a
        RuntimeError that says what to install.

        Parameters
        ----------
        kernel : callable
            Module-level function ``kernel(shard, state)`` that returns the
            statistics of a shard as a Partial.
        state : object
            Picklable state that the kernel reads.

        Returns
        -------
        Partial
            Statistics of all the units of the table with the keyed sums finalized.
        """
        # Since moderndid shipped to the executors as a zip has no package metadata, both sides compare the version
        # string that the package itself holds.
        version = _version.__version__
        python = "{}.{}".format(*sys.version_info[:2])
        arrow_schema = _arrow_schema()
        task = _shard_task(
            pickle.dumps((kernel, state), protocol=pickle.HIGHEST_PROTOCOL),
            version,
            python,
            self.n_periods,
            self.n_covariates,
            arrow_schema,
            self.whole_clusters,
        )
        frame = self.unit_table.mapInArrow(task, _spark_schema())
        if self._n_tasks is None or self._n_tasks > 16:
            # Since grouping task numbers by their integer square root gives about the root of their count in groups
            # of at most twice that size, the second stage stays balanced without knowing the count in advance.
            grouped = frame.groupBy(F.floor(F.sqrt(F.col("part"))))
            combine = _combine_task(version, python, arrow_schema)
            # Since PySpark 3.5 has no grouped applyInArrow, its groups reach the same combine function through pandas.
            if hasattr(grouped, "applyInArrow"):
                frame = grouped.applyInArrow(combine, _spark_schema())
            else:
                frame = grouped.applyInPandas(_pandas_combine(combine, arrow_schema), _spark_schema())
        rows = frame.collect()
        self.n_passes += 1

        for row in rows:
            if (row["version"], row["python"]) != (version, python):
                raise RuntimeError(_version_message(row["version"], row["python"], row["error"], version, python))
        self._n_tasks = sum(row["n"] for row in rows)

        partial = None
        for row in sorted(rows, key=lambda row: row["part"]):
            statistics = pickle.loads(row["payload"])
            partial = statistics if partial is None else combine_partials(partial, statistics)
        return empty_partial() if partial is None else finalize_keyed(partial)


def _arrow_schema():
    """Return the Arrow schema of the rows that the tasks of a pass return."""
    return pa.schema(
        [
            ("part", pa.int32()),
            ("n", pa.int64()),
            ("version", pa.string()),
            ("python", pa.string()),
            ("error", pa.string()),
            ("payload", pa.binary()),
        ]
    )


def _spark_schema():
    """Return the Spark schema of the rows that the tasks of a pass return."""
    return T.StructType(
        [
            T.StructField("part", T.IntegerType()),
            T.StructField("n", T.LongType()),
            T.StructField("version", T.StringType()),
            T.StructField("python", T.StringType()),
            T.StructField("error", T.StringType()),
            T.StructField("payload", T.BinaryType()),
        ]
    )


def _shard_task(payload, version, python, n_periods, n_covariates, arrow_schema, whole_clusters):
    """Return the ``mapInArrow`` function that one task of a pass runs on its partition.

    The function unpickles the kernel and its state from ``payload``, applies
    the kernel to chunks of the partition, and returns one row of statistics.
    It finalizes the keyed sums of the row only when ``whole_clusters`` says
    that the partition holds all the units of its clusters. An empty partition
    returns no row. On an executor whose moderndid or Python version differs
    from the driver's, it applies no kernel and returns only a report of the
    versions and any import error.
    """

    def task(batches):
        part = TaskContext.get().partitionId()
        # Since an executor without moderndid or with another version must report itself rather than fail, the task
        # imports moderndid only here and loads its modules only once the versions match.
        try:
            from moderndid._version import __version__ as found
        except ImportError as error:
            found, failure = None, f"{type(error).__name__}: {error}"
        else:
            failure = None
        found_python = "{}.{}".format(*sys.version_info[:2])
        if (found, found_python) != (version, python):
            report = {"part": part, "n": 1, "version": found, "python": found_python, "error": failure, "payload": None}
            yield pa.RecordBatch.from_pylist([report], schema=arrow_schema)
            return

        from moderndid.did.shards.layout import shard_from_arrow
        from moderndid.did.shards.partials import combine_partials, finalize_keyed

        kernel, state = pickle.loads(payload)
        partial = None
        for batch in batches:
            # Since a session may ignore the batch size and cap a UDF's memory at 1 GB, as Databricks documents for
            # serverless compute, the kernel reads bounded chunks so its working memory doesn't grow with the partition.
            for start in range(0, batch.num_rows, 4096):
                statistics = kernel(shard_from_arrow(batch.slice(start, 4096), n_periods, n_covariates), state)
                partial = statistics if partial is None else combine_partials(partial, statistics)
        if partial is not None:
            if whole_clusters:
                partial = finalize_keyed(partial)
            blob = pickle.dumps(partial, protocol=pickle.HIGHEST_PROTOCOL)
            row = {"part": part, "n": 1, "version": found, "python": found_python, "error": None, "payload": blob}
            yield pa.RecordBatch.from_pylist([row], schema=arrow_schema)

    return task


def _combine_task(version, python, arrow_schema):
    """Return the function that combines the rows of one group of tasks.

    The function takes the rows of the group as a pyarrow Table and returns a
    pyarrow Table of the combined rows. It folds the statistics in the order of
    the task numbers. It unpickles one payload at a time so that the others
    stay in their Arrow buffers. Rows whose versions differ from the driver's
    pass through unchanged. An executor whose own versions differ returns only
    its report.
    """

    def combine(table):
        parts = table.column("part").to_pylist()
        counts = table.column("n").to_pylist()
        try:
            from moderndid._version import __version__ as found
        except ImportError as error:
            found, failure = None, f"{type(error).__name__}: {error}"
        else:
            failure = None
        found_python = "{}.{}".format(*sys.version_info[:2])
        if (found, found_python) != (version, python):
            report = {
                "part": min(parts),
                "n": sum(counts),
                "version": found,
                "python": found_python,
                "error": failure,
                "payload": None,
            }
            return pa.Table.from_pylist([report], schema=arrow_schema)

        from moderndid.did.shards.partials import combine_partials

        versions = table.column("version").to_pylist()
        pythons = table.column("python").to_pylist()
        errors = table.column("error").to_pylist()
        payloads = table.column("payload")
        result = []
        partial, first, n_tasks = None, None, 0
        for i in sorted(range(table.num_rows), key=parts.__getitem__):
            if (versions[i], pythons[i]) != (version, python):
                result.append(
                    {
                        "part": parts[i],
                        "n": counts[i],
                        "version": versions[i],
                        "python": pythons[i],
                        "error": errors[i],
                        "payload": None,
                    }
                )
                continue
            statistics = pickle.loads(payloads[i].as_py())
            partial = statistics if partial is None else combine_partials(partial, statistics)
            first = parts[i] if first is None else first
            n_tasks += counts[i]
        if partial is not None:
            combined = {
                "part": first,
                "n": n_tasks,
                "version": version,
                "python": python,
                "error": None,
                "payload": pickle.dumps(partial, protocol=pickle.HIGHEST_PROTOCOL),
            }
            result.append(combined)
        return pa.Table.from_pylist(result, schema=arrow_schema)

    return combine


def _pandas_combine(combine, arrow_schema):
    """Return the ``applyInPandas`` function that runs a combine function on the rows of one group.

    The function turns the pandas DataFrame of the group into a pyarrow Table
    of the Arrow schema, combines the rows, and returns the result as a pandas
    DataFrame.
    """

    def combine_frame(frame):
        table = pa.Table.from_pandas(frame, schema=arrow_schema, preserve_index=False)
        return combine(table).to_pandas()

    return combine_frame


def _version_message(found, found_python, error, version, python):
    """Describe why the executors can't run the pass and what to install."""
    if found is None:
        return (
            f'The Spark executors failed to import moderndid with the error "{error}". Install moderndid {version} '
            "and its dependencies on every executor."
        )
    return (
        f"The Spark executors run moderndid {found} on Python {found_python}. The driver runs moderndid {version} on "
        f"Python {python}. Install the same moderndid version and Python minor version on the driver and on every "
        "executor."
    )
