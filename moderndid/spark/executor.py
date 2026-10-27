"""Execution of the passes of a sharded att_gt fit on a Spark table."""

import pickle
import sys

import pyarrow as pa
from pyspark import TaskContext
from pyspark.sql import functions as F
from pyspark.sql import types as T

from moderndid import _version
from moderndid.did.shards.partials import combine_partials, empty_partial, finalize_keyed


class SparkExecutor:
    """Run each pass of a sharded fit as one Spark action over a table.

    Like :class:`~moderndid.did.shards.local.LocalExecutor`, the executor has a
    ``run`` method and counts its passes in ``n_passes``. With
    ``whole_clusters``, each task finalizes the keyed sums of its partition.
    Otherwise the driver finalizes them once it has combined every row. See the
    :ref:`guide to fits on Spark <dev-spark-tasks>` for how the tasks of a pass
    combine their rows.

    Parameters
    ----------
    table : pyspark.sql.DataFrame
        Unit table or row table of the fit.
    read_shard : callable
        Picklable function that reads a shard from a pyarrow RecordBatch of
        rows of the table, such as
        ``functools.partial(shard_from_arrow, n_periods=T, n_covariates=p)``.
    whole_clusters : bool, default=False
        Whether each partition holds all the units of its clusters.
    """

    def __init__(self, table, read_shard, whole_clusters=False):
        self.table = table
        self.read_shard = read_shard
        self.whole_clusters = whole_clusters
        self.n_passes = 0
        self._n_tasks = None

    def run(self, kernel, state):
        """Apply a kernel to every shard of the table and combine the statistics.

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
            Statistics of the whole table with the keyed sums finalized.
        """
        # Both sides compare the version string that the package itself holds because moderndid shipped to the executors
        # as a zip has no package metadata.
        version = _version.__version__
        python = "{}.{}".format(*sys.version_info[:2])
        arrow_schema = _arrow_schema()
        task = _shard_task(
            pickle.dumps((kernel, state, self.read_shard), protocol=pickle.HIGHEST_PROTOCOL),
            version,
            python,
            arrow_schema,
            self.whole_clusters,
        )
        frame = self.table.mapInArrow(task, _spark_schema())
        if self._n_tasks is None or self._n_tasks > 16:
            # Grouping task numbers by their integer square root gives about the root of their count in groups of at
            # most twice that size. The second stage then stays balanced without knowing the count in advance.
            grouped = frame.groupBy(F.floor(F.sqrt(F.col("part"))))
            combine = _combine_task(version, python, arrow_schema)
            # PySpark 3.5 has no grouped applyInArrow. There the groups reach the same combine function through pandas.
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


def _shard_task(payload, version, python, arrow_schema, whole_clusters):
    """Return the ``mapInArrow`` function that one task of a pass runs on its partition.

    On an executor with another moderndid or Python version, the function
    returns a report of the versions in place of statistics.
    """

    def task(batches):
        part = TaskContext.get().partitionId()
        # An executor without moderndid or with another version must report itself rather than fail. The task therefore
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

        from moderndid.did.shards.partials import combine_partials, finalize_keyed

        # The reader waits for the version check too because unpickling it imports its module.
        kernel, state, read_shard = pickle.loads(payload)
        partial = None
        for batch in batches:
            # A session may ignore the batch size and cap a UDF's memory at 1 GB, as Databricks documents for serverless
            # compute. The kernel reads bounded chunks so its working memory doesn't grow with the partition.
            for start in range(0, batch.num_rows, 4096):
                statistics = kernel(read_shard(batch.slice(start, 4096)), state)
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

    Unless the combining executor runs another moderndid or Python version
    itself, the group's version reports pass through to the driver.
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
        # A group's payloads can be large. Each stays in its Arrow buffer until the loop reaches it.
        payloads = table.column("payload")
        result = []
        partial, first, n_tasks = None, None, 0
        # Folding in task order makes the combined statistics independent of the order Spark delivers the rows in.
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
    """Return the ``applyInPandas`` function that runs a combine function on the rows of one group."""

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
