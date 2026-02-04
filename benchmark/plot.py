"""Generate benchmark scaling comparison plot."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labeller,
    labs,
    scale_color_manual,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ESTIMATOR_CONFIG = {
    "attgt": {
        "name": "Staggered DiD",
        "order": 1,
        "patterns": ["benchmark_attgt_custom_*.csv"],
        "filter": {"est_method": "dr", "boot": False},
    },
    "ddd": {
        "name": "Triple DiD",
        "order": 2,
        "patterns": ["benchmark_ddd_scaling_units_*.csv", "benchmark_ddd_custom_*.csv"],
        "filter": {"est_method": "dr", "boot": False, "panel": True},
    },
    "didinter": {
        "name": "Intertemporal DiD",
        "order": 3,
        "patterns": ["benchmark_didinter_custom_*.csv"],
        "filter": {"boot": False, "n_periods": 8, "effects": 3},
    },
    "contdid": {
        "name": "Continuous DiD",
        "order": 4,
        "patterns": ["benchmark_contdid_custom_*.csv", "benchmark_contdid_quick_*.csv"],
        "filter": {"boot": False},
    },
}


def main():
    """Generate benchmark plot from CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate benchmark scaling plot")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/output",
        help="Directory containing benchmark CSV files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_scaling_hires.png",
        help="Output filename for the plot",
    )
    parser.add_argument(
        "--estimators",
        type=str,
        nargs="+",
        choices=list(ESTIMATOR_CONFIG.keys()),
        help="Estimators to include (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=16,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=5,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--max-obs",
        type=int,
        default=None,
        help="Maximum observations to include (filter out larger datasets)",
    )

    args = parser.parse_args()

    generate_plot(
        output_dir=args.output_dir,
        output_file=args.output_file,
        estimators=args.estimators,
        dpi=args.dpi,
        figsize=(args.width, args.height),
        max_obs=args.max_obs,
    )


def generate_plot(
    output_dir: str = "benchmark/output",
    output_file: str = "benchmark_scaling_hires.png",
    estimators: list[str] | None = None,
    dpi: int = 150,
    figsize: tuple[int, int] = (16, 5),
    max_obs: int | None = None,
):
    """Generate the benchmark scaling comparison plot."""
    if estimators is None:
        estimators = list(ESTIMATOR_CONFIG.keys())

    all_dfs = []
    speedup_rows = []

    for key in estimators:
        if key not in ESTIMATOR_CONFIG:
            logger.warning(f"Unknown estimator: {key}")
            continue

        config = ESTIMATOR_CONFIG[key]
        label = f"{config['order']}_{config['name']}"

        data = _load_benchmark_data(output_dir, key, max_obs=max_obs)
        if not data:
            logger.warning(f"No data found for {key}")
            continue

        logger.info(f"{config['name']}: {len(data)} data points")

        df = _create_plot_df(data, label)
        all_dfs.append(df)

        speedup_info = _get_max_speedup(data)
        if speedup_info:
            speedup_rows.append(
                {
                    "n_obs": speedup_info["n_obs"],
                    "time": speedup_info["time"],
                    "impl": "R",
                    "estimator": label,
                    "speedup": f"{speedup_info['speedup']:.0f}x",
                }
            )
            logger.info(f"  Max speedup: {speedup_info['speedup']:.0f}x at {speedup_info['n_obs']:,} obs")

    if not all_dfs:
        logger.error("No benchmark data found")
        return

    df_all = pl.concat(all_dfs)
    df_speedup = pl.DataFrame(speedup_rows)

    ncol = min(4, len(estimators))

    # Convert to pandas for plotnine compatibility
    p = (
        ggplot(df_all.to_pandas(), aes(x="n_obs", y="time", color="impl"))
        + geom_line(size=1)
        + geom_point(size=3)
        + geom_text(
            data=df_speedup.to_pandas(),
            mapping=aes(label="speedup"),
            nudge_y=0.3,
            size=9,
            color="black",
        )
        + facet_wrap("~estimator", ncol=ncol, scales="free", labeller=labeller(estimator=_strip_prefix))
        + scale_x_log10(
            labels=lambda x: [
                f"{int(v / 1000000)}M" if v >= 1000000 else f"{int(v / 1000)}K" if v >= 1000 else str(int(v)) for v in x
            ]
        )
        + scale_y_log10()
        + scale_color_manual(values={"Python": "#1f77b4", "R": "#d62728"})
        + labs(
            x="Observations",
            y="Time (seconds)",
            color="Implementation",
            title="Python vs R Performance: Observation Scaling",
        )
        + theme_minimal()
        + theme(
            figure_size=figsize,
            plot_title=element_text(size=14, weight="bold"),
            strip_text=element_text(size=11, weight="bold"),
            legend_position="bottom",
        )
    )

    output_path = Path(output_dir) / output_file
    p.save(output_path, dpi=dpi)
    logger.info(f"Plot saved to {output_path}")


def _load_benchmark_data(output_dir: str, estimator_key: str, max_obs: int | None = None) -> list[dict]:
    """Load benchmark data from CSV files for a given estimator."""
    config = ESTIMATOR_CONFIG[estimator_key]
    patterns = config.get("patterns", [config.get("pattern", "")])
    filter_config = config.get("filter", {})

    seen_obs = {}

    for pattern in patterns:
        for f in glob.glob(str(Path(output_dir) / pattern)):
            try:
                df = pl.read_csv(f)
                for row in df.iter_rows(named=True):
                    if not row.get("python_success", False):
                        continue

                    if not _matches_filter(row, filter_config):
                        continue

                    n_obs = int(row["n_observations"])

                    if max_obs is not None and n_obs > max_obs:
                        continue
                    python_time = float(row["python_mean_time"])
                    r_time = None

                    if row.get("r_success"):
                        r_time = float(row["r_mean_time"])

                    if n_obs not in seen_obs or (r_time and not seen_obs[n_obs].get("r_time")):
                        seen_obs[n_obs] = {
                            "n_obs": n_obs,
                            "python_time": python_time,
                            "r_time": r_time,
                        }
            except (OSError, pl.exceptions.ComputeError, KeyError, ValueError) as e:
                logger.warning(f"Error reading {f}: {e}")

    return sorted(seen_obs.values(), key=lambda x: x["n_obs"])


def _matches_filter(row: dict, filter_config: dict) -> bool:
    """Check if a row matches the filter criteria."""
    return all(key not in row or row[key] == value for key, value in filter_config.items())


def _create_plot_df(data: list[dict], estimator_label: str) -> pl.DataFrame:
    """Create DataFrame for plotting from benchmark data.

    Only includes observation points where both Python and R results exist.
    """
    rows = []
    for d in data:
        # Only include points where both Python and R succeeded
        if d["r_time"] is None:
            continue
        rows.append(
            {
                "n_obs": d["n_obs"],
                "time": d["python_time"],
                "impl": "Python",
                "estimator": estimator_label,
            }
        )
        rows.append(
            {
                "n_obs": d["n_obs"],
                "time": d["r_time"],
                "impl": "R",
                "estimator": estimator_label,
            }
        )
    return pl.DataFrame(rows)


def _get_max_speedup(data: list[dict]) -> dict | None:
    """Get the maximum R observation point with speedup info."""
    r_points = [d for d in data if d["r_time"] is not None]
    if not r_points:
        return None

    max_r_point = max(r_points, key=lambda x: x["n_obs"])
    speedup = max_r_point["r_time"] / max_r_point["python_time"]

    return {
        "n_obs": max_r_point["n_obs"],
        "time": max_r_point["r_time"],
        "speedup": speedup,
    }


def _strip_prefix(label: str) -> str:
    """Remove ordering prefix from label."""
    if len(label) > 2 and label[1] == "_":
        return label[2:]
    return label


if __name__ == "__main__":
    main()
