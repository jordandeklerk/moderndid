"""Cluster monitoring utilities for Dask distributed execution."""

from __future__ import annotations

import logging
from threading import Event, Thread

logger = logging.getLogger(__name__)


def _format_status(client):
    """Build a one-line cluster health string. Returns None on failure."""
    try:
        info = client.scheduler_info()
    except OSError:
        return None

    workers = info.get("workers", {})
    n_workers = len(workers)

    if n_workers == 0:
        return "[monitor] No workers connected"

    total_mem = 0
    total_limit = 0
    spilling = 0

    for w in workers.values():
        mem = w.get("metrics", {}).get("memory", 0)
        limit = w.get("memory_limit", 0)
        total_mem += mem
        total_limit += limit
        if limit > 0 and mem / limit > 0.80:
            spilling += 1

    mem_gb = total_mem / 1e9
    limit_gb = total_limit / 1e9
    pct = (total_mem / total_limit * 100) if total_limit > 0 else 0

    task_state = {}
    for ts in info.get("tasks", {}).values():
        s = ts if isinstance(ts, str) else ts.get("state", "unknown")
        task_state[s] = task_state.get(s, 0) + 1

    # Some schedulers do not expose task details in scheduler_info(); fall back
    # to worker metrics so task counts remain informative.
    if not task_state:
        for w in workers.values():
            for s, count in w.get("metrics", {}).get("task_counts", {}).items():
                task_state[s] = task_state.get(s, 0) + int(count)

    processing = task_state.get("processing", 0) + task_state.get("executing", 0)
    mem_tasks = task_state.get("memory", 0)
    waiting = task_state.get("waiting", 0) + task_state.get("ready", 0) + task_state.get("queued", 0)
    erred = task_state.get("erred", 0) + task_state.get("error", 0)

    status = "OK"
    if spilling > 0:
        status = f"SPILLING ({spilling}/{n_workers} workers >80% mem)"
    if erred > 0:
        status = f"ERRORS ({erred} erred tasks)"

    return (
        f"[monitor] workers={n_workers}  "
        f"mem={mem_gb:.1f}/{limit_gb:.1f} GB ({pct:.0f}%)  "
        f"tasks: {processing} running, {mem_tasks} done, {waiting} waiting  "
        f"status={status}"
    )


def log_cluster_status(client, emit=None):
    """Log a one-line cluster health summary.

    Parameters
    ----------
    client : distributed.Client
        Active Dask client.
    emit : callable, optional
        Function to call with the status string. Defaults to ``logger.info``.
        Pass ``print`` for visible output in Databricks notebooks.
    """
    if emit is None:
        emit = logger.info
    msg = _format_status(client)
    if msg is not None:
        emit(msg)


def monitor_cluster(client, interval=10, emit=None):
    """Start background monitoring that reports cluster health every ``interval`` seconds.

    Parameters
    ----------
    client : distributed.Client
        Active Dask client.
    interval : int
        Seconds between status reports.
    emit : callable, optional
        Function to call with each status string. Defaults to ``logger.info``.
        Pass ``print`` for visible output in Databricks notebooks.

    Returns
    -------
    stop : callable
        Call ``stop()`` to terminate the monitor thread.
    """
    if emit is None:
        emit = logger.info
    event = Event()

    def _loop():
        while not event.is_set():
            msg = _format_status(client)
            if msg is not None:
                emit(msg)
            event.wait(interval)

    t = Thread(target=_loop, daemon=True)
    t.start()

    def stop():
        event.set()
        t.join(timeout=2)
        emit("[monitor] stopped")

    return stop
