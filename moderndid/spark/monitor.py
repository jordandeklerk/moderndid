"""Cluster monitoring utilities for distributed Spark estimation."""

from __future__ import annotations

import threading


def monitor_spark(spark, interval=15, emit=print, per_executor=False):
    """Periodically log Spark executor memory and task statistics.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    interval : float, default 15
        Seconds between status reports.
    emit : callable, default print
        Function to call with each status line.
    per_executor : bool, default False
        If True, include per-executor memory breakdown.

    Returns
    -------
    callable
        A ``stop()`` function that terminates the monitoring thread.
    """
    stop_event = threading.Event()

    def _loop():
        while not stop_event.is_set():
            try:
                sc = spark.sparkContext
                status = sc.statusTracker()
                active_jobs = status.getActiveJobIds()
                active_stages = status.getActiveStageIds()

                # Get executor info via JVM bridge
                jsc = sc._jsc
                executors = jsc.sc().getExecutorMemoryStatus()
                n_executors = executors.size()

                total_mem = 0
                total_used = 0
                executor_lines = []

                it = executors.iterator()
                while it.hasNext():
                    entry = it.next()
                    addr = entry._1()
                    mem_info = entry._2()
                    mem_limit = mem_info._1()
                    mem_remaining = mem_info._2()
                    mem_used = mem_limit - mem_remaining
                    total_mem += mem_limit
                    total_used += mem_used

                    if per_executor:
                        pct = (mem_used / mem_limit * 100) if mem_limit > 0 else 0
                        executor_lines.append(
                            f"    {addr}: {mem_used / 1e9:.1f} / {mem_limit / 1e9:.1f} GB ({pct:.0f}%)"
                        )

                pct_total = (total_used / total_mem * 100) if total_mem > 0 else 0
                msg = (
                    f"[monitor] {n_executors} executors | "
                    f"Jobs: {len(active_jobs)} | Stages: {len(active_stages)} | "
                    f"Memory: {total_used / 1e9:.1f} / {total_mem / 1e9:.1f} GB "
                    f"({pct_total:.0f}%)"
                )
                emit(msg)

                if per_executor:
                    for line in executor_lines:
                        emit(line)

            except (OSError, KeyError):
                pass

            stop_event.wait(interval)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        thread.join(timeout=2)

    return stop
