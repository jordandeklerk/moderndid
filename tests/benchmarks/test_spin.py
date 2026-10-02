import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from click.testing import CliRunner

pytest.importorskip("spin")

_REPOSITORY = Path(__file__).resolve().parents[2]
_COMMANDS = runpy.run_path(str(_REPOSITORY / ".spin" / "cmds.py"))


@pytest.mark.parametrize(
    "arguments, options",
    [
        ([], []),
        (["--quick"], ["--quick", "--show-stderr"]),
        (["-v"], ["--verbose", "--show-stderr"]),
        (["-q", "--verbose"], ["--quick", "--verbose", "--show-stderr"]),
    ],
)
def test_bench_runs_current_checkout(monkeypatch, arguments, options):
    run = Mock()
    git = Mock()
    monkeypatch.setattr(_COMMANDS["util"], "run", run)
    monkeypatch.setattr(_COMMANDS["subprocess"], "run", git)

    result = CliRunner().invoke(_COMMANDS["bench"], ["-t", "Panel", "--tests", "Agg", *arguments])

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "asv",
            "run",
            "--python=same",
            "--dry-run",
            "--bench",
            "Panel",
            "--bench",
            "Agg",
            *options,
        ],
        cwd=str(_REPOSITORY / "benchmarks"),
    )
    git.assert_not_called()


@pytest.mark.parametrize("factor", ["-2", "0", "1", "nan", "inf", "-inf"])
def test_bench_rejects_invalid_factor(monkeypatch, factor):
    run = Mock()
    monkeypatch.setattr(_COMMANDS["util"], "run", run)

    result = CliRunner().invoke(_COMMANDS["bench"], ["--factor", factor])

    assert result.exit_code == 2
    assert "--factor must be a finite number greater than 1" in result.output
    run.assert_not_called()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["HEAD"], "benchmark revisions require --compare"),
        (["--compare", "main", "HEAD", "extra"], "--compare accepts at most two revisions"),
    ],
)
def test_bench_rejects_invalid_revision_count(monkeypatch, arguments, message):
    run = Mock()
    git = Mock()
    monkeypatch.setattr(_COMMANDS["util"], "run", run)
    monkeypatch.setattr(_COMMANDS["subprocess"], "run", git)

    result = CliRunner().invoke(_COMMANDS["bench"], arguments)

    assert result.exit_code == 2
    assert message in result.output
    run.assert_not_called()
    git.assert_not_called()


@pytest.mark.parametrize(
    "revisions, resolved",
    [([], ["main", "HEAD"]), (["base"], ["base", "HEAD"]), (["base", "head"], ["base", "head"])],
)
@pytest.mark.parametrize("dirty", [False, True])
def test_bench_compares_resolved_commits(monkeypatch, revisions, resolved, dirty):
    run = Mock()
    git = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n"),
            subprocess.CompletedProcess([], 0, stdout="b" * 40 + "\n"),
            subprocess.CompletedProcess([], 0, stdout=" M .spin/cmds.py\n" if dirty else ""),
        ]
    )
    monkeypatch.setattr(_COMMANDS["util"], "run", run)
    monkeypatch.setattr(_COMMANDS["subprocess"], "run", git)

    result = CliRunner().invoke(_COMMANDS["bench"], ["--compare", "-t", "Panel", "--quick", *revisions])

    assert result.exit_code == 0, result.output
    assert git.call_args_list == [
        *[
            call(
                ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
                cwd=_REPOSITORY,
                check=False,
                capture_output=True,
                text=True,
            )
            for revision in resolved
        ],
        call(["git", "status", "--porcelain"], cwd=_REPOSITORY, check=True, capture_output=True, text=True),
    ]
    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "asv",
            "continuous",
            "--factor",
            "1.05",
            "--bench",
            "Panel",
            "--quick",
            "--show-stderr",
            "a" * 40,
            "b" * 40,
        ],
        cwd=str(_REPOSITORY / "benchmarks"),
    )
    assert ("The working tree is dirty." in result.output) is dirty


def test_bench_stops_on_unresolved_revision(monkeypatch):
    run = Mock()
    git = Mock(return_value=subprocess.CompletedProcess([], 128, stdout="", stderr="unknown revision"))
    monkeypatch.setattr(_COMMANDS["util"], "run", run)
    monkeypatch.setattr(_COMMANDS["subprocess"], "run", git)

    result = CliRunner().invoke(_COMMANDS["bench"], ["--compare", "--", "--untrusted"])

    assert result.exit_code == 1
    assert "could not resolve benchmark revision '--untrusted'" in result.output
    git.assert_called_once_with(
        ["git", "rev-parse", "--verify", "--end-of-options", "--untrusted^{commit}"],
        cwd=_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    run.assert_not_called()


@pytest.mark.parametrize("arguments", [["--help"], ["--estimators", "att_gt", "--workloads", "short"]])
def test_compare_passes_arguments_through(monkeypatch, arguments):
    run = Mock()
    monkeypatch.setattr(_COMMANDS["util"], "run", run)

    result = CliRunner().invoke(_COMMANDS["compare"], arguments)

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        [sys.executable, "-m", "benchmarks.compare", *arguments],
        cwd=str(_REPOSITORY),
    )


@pytest.mark.parametrize("arguments", [["--help"], ["check", "-E", "existing"], ["run", "--bench", "Panel.*"]])
def test_asv_passes_arguments_through(monkeypatch, arguments):
    run = Mock()
    monkeypatch.setattr(_COMMANDS["util"], "run", run)

    result = CliRunner().invoke(_COMMANDS["asv"], arguments)

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        [sys.executable, "-m", "asv", *arguments],
        cwd=str(_REPOSITORY / "benchmarks"),
    )
