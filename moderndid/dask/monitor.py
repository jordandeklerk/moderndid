"""Cluster monitoring utilities for Dask distributed execution."""

from __future__ import annotations

import logging
from threading import Event, Thread

logger = logging.getLogger(__name__)


def log_cluster_status(client):
    """Log a one-line cluster health summary."""
    try:
        info = client.scheduler_info()
    except OSError:
        logger.warning("Cannot reach scheduler")
        return

    workers = info.get("workers", {})
    n_workers = len(workers)

    if n_workers == 0:
        logger.warning("No workers connected")
        return

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

    processing = task_state.get("processing", 0)
    mem_tasks = task_state.get("memory", 0)
    waiting = task_state.get("waiting", 0)
    erred = task_state.get("erred", 0)

    status = "OK"
    if spilling > 0:
        status = f"SPILLING ({spilling}/{n_workers} workers >80% mem)"
    if erred > 0:
        status = f"ERRORS ({erred} erred tasks)"

    logger.info(
        "workers=%d  mem=%.1f/%.1f GB (%d%%)  tasks: %d running, %d done, %d waiting  status=%s",
        n_workers,
        mem_gb,
        limit_gb,
        pct,
        processing,
        mem_tasks,
        waiting,
        status,
    )


def monitor_cluster(client, interval=10):
    """Start background monitoring that logs cluster health every ``interval`` seconds.

    Returns a callable ``stop()`` to terminate the monitor thread.
    """
    event = Event()

    def _loop():
        while not event.is_set():
            log_cluster_status(client)
            event.wait(interval)

    t = Thread(target=_loop, daemon=True)
    t.start()

    def stop():
        event.set()
        t.join(timeout=2)
        logger.info("monitor stopped")

    return stop
