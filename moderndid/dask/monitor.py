"""Cluster monitoring utilities for distributed DDD estimation."""

from __future__ import annotations

import threading


def monitor_cluster(client, interval=15, emit=print, per_worker=False):
    """Periodically log cluster memory and task statistics.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    interval : float, default 15
        Seconds between status reports.
    emit : callable, default print
        Function to call with each status line.
    per_worker : bool, default False
        If True, include per-worker memory breakdown.

    Returns
    -------
    callable
        A ``stop()`` function that terminates the monitoring thread.
    """
    stop_event = threading.Event()

    def _loop():
        while not stop_event.is_set():
            try:
                info = client.scheduler_info()
                workers = info.get("workers", {})
                n_workers = len(workers)

                total_mem = 0
                total_used = 0
                worker_lines = []

                for addr, w in workers.items():
                    mem_limit = w.get("memory_limit", 0)
                    metrics = w.get("metrics", {})
                    mem_used = metrics.get("memory", 0)
                    total_mem += mem_limit
                    total_used += mem_used

                    if per_worker:
                        pct = (mem_used / mem_limit * 100) if mem_limit > 0 else 0
                        short_addr = addr.split("//")[-1]
                        worker_lines.append(
                            f"    {short_addr}: {mem_used / 1e9:.1f} / {mem_limit / 1e9:.1f} GB ({pct:.0f}%)"
                        )

                pct_total = (total_used / total_mem * 100) if total_mem > 0 else 0
                msg = (
                    f"[monitor] {n_workers} workers | "
                    f"Memory: {total_used / 1e9:.1f} / {total_mem / 1e9:.1f} GB "
                    f"({pct_total:.0f}%)"
                )
                emit(msg)

                if per_worker:
                    for line in worker_lines:
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
