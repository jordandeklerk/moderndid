"""Provide project-specific development commands."""

import math
import subprocess
import sys
from pathlib import Path

import click
from spin import util


@click.command()
@click.option(
    "--tests",
    "-t",
    multiple=True,
    metavar="BENCHMARK",
    help="Run benchmarks matching this ASV expression. Repeat to add filters.",
)
@click.option(
    "--compare",
    "-c",
    is_flag=True,
    help="Compare two committed revisions. Defaults to main and HEAD.",
)
@click.option(
    "--quick",
    "-q",
    is_flag=True,
    help="Run each selected benchmark once to check that it executes.",
)
@click.option("--verbose", "-v", is_flag=True, help="Show additional ASV output.")
@click.option(
    "--factor",
    "-f",
    type=float,
    default=1.05,
    show_default=True,
    help="Report comparison changes larger than this factor when comparing revisions.",
)
@click.argument("revisions", nargs=-1)
def bench(tests, compare, quick, verbose, factor, revisions):
    """Run or compare ASV benchmarks."""
    repository = Path(__file__).resolve().parents[1]
    benchmark_arguments = [argument for test in tests for argument in ("--bench", test)]

    if not math.isfinite(factor) or factor <= 1:
        raise click.UsageError("--factor must be a finite number greater than 1")
    if quick:
        benchmark_arguments.append("--quick")
    if verbose:
        benchmark_arguments.append("--verbose")
    if quick or verbose:
        benchmark_arguments.append("--show-stderr")

    if not compare:
        if revisions:
            raise click.UsageError("benchmark revisions require --compare")
        command = [sys.executable, "-m", "asv", "run", "--python=same", "--dry-run"]
        command.extend(benchmark_arguments)
        util.run(command, cwd=str(repository / "benchmarks"))
        return

    if len(revisions) > 2:
        raise click.UsageError("--compare accepts at most two revisions")

    revisions = revisions or ("main", "HEAD")
    if len(revisions) == 1:
        revisions = (*revisions, "HEAD")

    resolved_revisions = []
    for revision in revisions:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"could not resolve benchmark revision {revision!r}")
        resolved_revisions.append(result.stdout.strip())

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        click.secho(
            "The working tree is dirty. Revision comparisons use committed package source. "
            "The current benchmark suite defines both runs.",
            fg="yellow",
        )

    command = [sys.executable, "-m", "asv", "continuous", "--factor", str(factor)]
    command.extend(benchmark_arguments)
    command.extend(resolved_revisions)
    util.run(command, cwd=str(repository / "benchmarks"))


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True, "help_option_names": []},
    add_help_option=False,
)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def compare(arguments):
    """Compare moderndid with the R implementations of its estimators."""
    repository = Path(__file__).resolve().parents[1]
    util.run([sys.executable, "-m", "benchmarks.compare", *arguments], cwd=str(repository))


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True, "help_option_names": []},
    add_help_option=False,
)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def asv(arguments):
    """Run ASV from the benchmark directory."""
    repository = Path(__file__).resolve().parents[1]
    util.run([sys.executable, "-m", "asv", *arguments], cwd=str(repository / "benchmarks"))
