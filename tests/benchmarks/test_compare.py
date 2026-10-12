import csv
import inspect
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks import common, compare, reporting
from benchmarks.cases import WORKLOADS
from benchmarks.references import reference_code


def reference(keys=("ATT",), estimates=(1.0,), errors=(0.2,)):
    return {"keys": list(keys), "estimates": list(estimates), "std_errors": list(errors)}


def test_agreement_requires_all_output_keys():
    result = compare.comparison_details(
        (["a", "b"], [1.0, 2.0], [0.2, 0.3]),
        reference(["b", "c"], [2.0, 3.0], [0.3, 0.4]),
        "att_gt",
        None,
    )
    assert result["status"] == "partial"
    assert result["cells"] == result["se_cells"] == 1
    assert result["expected_cells"] == result["expected_se_cells"] == 3
    assert result["missing_python_keys"] == ["c"]
    assert result["missing_r_keys"] == ["a"]


@pytest.mark.parametrize("python_error,r_error", [(np.nan, None), (0.2, None), (np.inf, 0.2)])
def test_unavailable_required_standard_errors_cannot_match(python_error, r_error):
    result = compare.comparison_details((["ATT"], [1.0], [python_error]), reference(errors=[r_error]), "drdid", None)
    assert result["status"] == "partial"
    assert result["se_cells"] == 0
    assert result["expected_se_cells"] == 1
    assert result["nonfinite_se_keys"] == ["ATT"]


def test_nonfinite_estimates_cannot_match():
    result = compare.comparison_details((["ATT"], [np.nan], [0.2]), reference(), "drdid", None)
    assert result["status"] == "no shared cells"
    assert result["expected_cells"] == 1
    assert result["nonfinite_estimate_keys"] == ["ATT"]


@pytest.mark.parametrize(
    "python_error,r_error,field", [(np.nan, None, "unavailable_se_keys"), (0.0, None, "normalized_se_keys")]
)
@pytest.mark.parametrize(
    "estimator,aggregation,keys", [("ddd", None, ["3 2", "3 3"]), ("agg_ddd", "dynamic", ["-1", "0"])]
)
def test_normalized_reference_standard_errors_are_accounted_for(
    python_error, r_error, field, estimator, aggregation, keys
):
    result = compare.comparison_details(
        (keys, [0.0, 1.0], [python_error, 0.2]),
        reference(keys, [0.0, 1.0], [r_error, 0.2]),
        estimator,
        aggregation,
    )
    assert result["status"] == "match"
    assert result["cells"] == result["expected_cells"] == 2
    assert result["se_cells"] == result["expected_se_cells"] == 1
    assert result[field] == [keys[0]]
    assert result["nonfinite_se_keys"] == []


def test_normalized_reference_exception_requires_zero_estimates():
    result = compare.comparison_details(
        (["3 2", "3 3"], [0.1, 1.0], [np.nan, 0.2]),
        reference(["3 2", "3 3"], [0.1, 1.0], [None, 0.2]),
        "ddd",
        None,
    )
    assert result["status"] == "partial"
    assert result["nonfinite_se_keys"] == ["3 2"]


def test_continuous_curves_and_standard_errors_are_known_differences_from_the_grid_basis_reference():
    keys = ["ATT", "ACRT", "ATT dose 1", "ACRT dose 1"]
    result = compare.comparison_details(
        (keys, [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]),
        reference(keys, [1.0, 2.0, 3.5, 4.25], [0.2, 0.2, 0.3, 0.9]),
        "cont_did",
        None,
    )
    assert result["status"] == "match"
    assert result["cells"] == result["expected_cells"] == 2
    assert result["se_cells"] == result["expected_se_cells"] == 0
    assert result["excluded_estimate_keys"] == ["ACRT dose 1", "ATT dose 1"]
    assert result["excluded_se_keys"] == ["ACRT", "ACRT dose 1", "ATT", "ATT dose 1"]
    assert result["excluded_estimate_gap"] == pytest.approx(0.5)
    assert result["excluded_error_gap"] == pytest.approx(0.5)
    assert result["estimate"] == 0.0
    assert result["error"] is None
    assert "boundary knots are the grid's endpoints" in result["exclusion_reason"]


@pytest.mark.parametrize("r_overall", [[1.1, 2.0], [1.0, 2.1]])
def test_continuous_overall_estimates_stay_held_to_the_tolerance(r_overall):
    keys = ["ATT", "ACRT", "ATT dose 1"]
    result = compare.comparison_details(
        (keys, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        reference(keys, [*r_overall, 3.0], [0.1, 0.2, 0.3]),
        "cont_did",
        None,
    )
    assert result["status"] == "differs"
    assert result["estimate"] == pytest.approx(0.1)


def test_continuous_curve_key_missing_in_r_leaves_coverage_partial():
    result = compare.comparison_details(
        (["ATT", "ACRT", "ATT dose 1"], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        reference(["ATT", "ACRT"], [1.0, 2.0], [0.1, 0.2]),
        "cont_did",
        None,
    )
    assert result["status"] == "partial"
    assert result["missing_r_keys"] == ["ATT dose 1"]
    assert result["excluded_estimate_keys"] == ["ATT dose 1"]
    assert result["cells"] == result["expected_cells"] == 2


@pytest.mark.parametrize(
    "python_estimates,r_estimates", [([1.0, 2.0, np.nan], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [1.0, 2.0, None])]
)
def test_nonfinite_known_estimate_difference_leaves_coverage_partial(python_estimates, r_estimates):
    keys = ["ATT", "ACRT", "ATT dose 1"]
    result = compare.comparison_details(
        (keys, python_estimates, [0.1, 0.2, 0.3]), reference(keys, r_estimates, [0.1, 0.2, 0.3]), "cont_did", None
    )
    assert result["status"] == "partial"
    assert result["nonfinite_estimate_keys"] == ["ATT dose 1"]


@pytest.mark.parametrize("python_errors,key", [([0.1, 0.2, np.nan], "ATT dose 1"), ([np.nan, 0.2, 0.3], "ATT")])
def test_nonfinite_known_standard_error_difference_leaves_coverage_partial(python_errors, key):
    keys = ["ATT", "ACRT", "ATT dose 1"]
    result = compare.comparison_details(
        (keys, [1.0, 2.0, 3.0], python_errors), reference(keys, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]), "cont_did", None
    )
    assert result["status"] == "partial"
    assert result["nonfinite_se_keys"] == [key]


def test_triple_difference_group_overall_standard_error_is_known_difference_from_shares_and_labels():
    result = compare.comparison_details(
        (["overall", "3", "5"], [1.0, 1.0, 2.0], [0.2, 0.2, 0.3]),
        reference(["overall", "3", "5"], [1.0, 1.0, 2.0], [0.8, 0.2, 0.3]),
        "agg_ddd",
        "group",
        {"overall": 2, "3": 1, "5": 1},
    )
    assert result["status"] == "match"
    assert result["cells"] == result["expected_cells"] == 3
    assert result["se_cells"] == result["expected_se_cells"] == 2
    assert result["excluded_estimate_keys"] == []
    assert result["excluded_se_keys"] == ["overall"]
    assert result["excluded_error_gap"] == pytest.approx(0.6)
    assert "eligible units" in result["exclusion_reason"]
    assert "group-time cohort labels" in result["exclusion_reason"]


@pytest.mark.parametrize("r_estimates,r_errors", [([1.0, 1.0], [0.2, 0.202]), ([1.001, 1.0], [0.2, 0.2])])
def test_triple_difference_group_estimates_and_cohort_standard_errors_stay_strict(r_estimates, r_errors):
    result = compare.comparison_details(
        (["overall", "3"], [1.0, 1.0], [0.2, 0.2]),
        reference(["overall", "3"], r_estimates, r_errors),
        "agg_ddd",
        "group",
        {"overall": 2, "3": 1},
    )
    assert result["status"] == "differs"


@pytest.mark.parametrize("aggregation", ["simple", "dynamic"])
def test_triple_difference_eligible_shares_make_overall_standard_error_of_several_cohorts_known(aggregation):
    result = compare.comparison_details(
        (["overall"], [1.0], [0.2]), reference(["overall"], [1.0], [0.8]), "agg_ddd", aggregation, {"overall": 2}
    )
    assert result["status"] == "match"
    assert result["se_cells"] == result["expected_se_cells"] == 0
    assert result["excluded_se_keys"] == ["overall"]
    assert result["excluded_error_gap"] == pytest.approx(0.6)
    assert result["error"] is None
    assert "eligible units" in result["exclusion_reason"]
    assert "group-time cohort labels" not in result["exclusion_reason"]


@pytest.mark.parametrize("cohorts", [{"overall": 1}, None])
@pytest.mark.parametrize("aggregation", ["simple", "group", "dynamic"])
def test_triple_difference_standard_errors_of_one_cohort_stay_strict(aggregation, cohorts):
    result = compare.comparison_details(
        (["overall"], [1.0], [0.2]), reference(["overall"], [1.0], [0.8]), "agg_ddd", aggregation, cohorts
    )
    assert result["status"] == "differs"
    assert result["excluded_se_keys"] == []
    assert result["exclusion_reason"] is None


def test_triple_difference_event_study_records_event_times_of_several_cohorts_as_known_differences():
    keys = ["overall", "-2", "-1", "0", "2"]
    result = compare.comparison_details(
        (keys, [1.0, 0.1, 0.0, 1.0, 2.0], [0.2, 0.3, np.nan, 0.25, 0.4]),
        reference(keys, [1.0, 0.1, 0.0, 1.0, 2.0], [0.21, 0.31, None, 0.26, 0.4]),
        "agg_ddd",
        "dynamic",
        {"overall": 2, "-2": 2, "-1": 0, "0": 2, "2": 1},
    )
    assert result["status"] == "match"
    assert result["cells"] == result["expected_cells"] == 5
    assert result["se_cells"] == result["expected_se_cells"] == 1
    assert result["excluded_estimate_keys"] == []
    assert result["excluded_se_keys"] == ["-2", "0", "overall"]
    assert result["unavailable_se_keys"] == ["-1"]
    assert result["excluded_error_gap"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "r_estimates,r_errors",
    [
        ([1.0, 0.1, 0.0, 1.0, 2.0], [0.2, 0.3, None, 0.25, 0.41]),
        ([1.0, 0.1, 0.0, 1.001, 2.0], [0.2, 0.3, None, 0.25, 0.4]),
    ],
)
def test_triple_difference_event_study_holds_estimates_and_event_times_of_one_cohort_to_the_tolerance(
    r_estimates, r_errors
):
    keys = ["overall", "-2", "-1", "0", "2"]
    result = compare.comparison_details(
        (keys, [1.0, 0.1, 0.0, 1.0, 2.0], [0.2, 0.3, np.nan, 0.25, 0.4]),
        reference(keys, r_estimates, r_errors),
        "agg_ddd",
        "dynamic",
        {"overall": 2, "-2": 2, "-1": 0, "0": 2, "2": 1},
    )
    assert result["status"] == "differs"


@pytest.mark.parametrize(
    "aggregation,counts",
    [
        ("simple", {"overall": 2, "3": 1, "5": 1}),
        ("group", {"overall": 2, "3": 1, "5": 1}),
        ("dynamic", {"-4": 1, "-3": 1, "-2": 2, "-1": 0, "0": 2, "1": 2, "2": 1, "3": 1, "overall": 2}),
    ],
)
def test_averaged_cohorts_count_the_estimated_cells_behind_each_output(aggregation, counts):
    fitted = SimpleNamespace(groups=np.repeat([3.0, 5.0], 6), times=np.tile(np.arange(1.0, 7.0), 2))
    assert compare.averaged_cohorts(fitted, aggregation) == counts


def test_averaged_cohorts_of_a_single_cohort_are_one():
    fitted = SimpleNamespace(groups=np.repeat(3.0, 4), times=np.arange(1.0, 5.0))
    assert compare.averaged_cohorts(fitted, "dynamic") == {"-2": 1, "-1": 0, "0": 1, "1": 1, "overall": 1}
    assert compare.averaged_cohorts(fitted, "group") == {"overall": 1, "3": 1}


def test_triple_difference_aggregation_call_holds_the_fitted_result_whose_cohorts_are_counted():
    call, _ = compare.prepare_python("agg_ddd", "baseline", "dynamic")
    counts = compare.averaged_cohorts(call.args[0], "dynamic")
    assert counts == {"-4": 1, "-3": 1, "-2": 2, "-1": 0, "0": 2, "1": 2, "2": 1, "3": 1, "overall": 2}
    assert set(compare.python_rows("agg_ddd", call())[0]) == set(counts)


def test_triple_difference_case_counts_cohorts_in_the_fitted_result(monkeypatch):
    fitted = SimpleNamespace(groups=np.repeat([3.0, 5.0], 6), times=np.tile(np.arange(1.0, 7.0), 2))
    aggregated = SimpleNamespace(overall_att=1.0, overall_se=0.2, egt=None, att_egt=None, se_egt=None)
    monkeypatch.setattr(compare, "prepare_python", lambda *args: (partial(lambda result: aggregated, fitted), [1, 2]))
    monkeypatch.setattr(compare, "package_version", lambda package: "1.0")
    monkeypatch.setattr(
        compare, "run_reference", lambda *args: {**reference(["overall"], [1.0], [0.3]), "seconds": [0.1]}
    )
    row = compare.run_case("agg_ddd", "baseline", "simple", 1)
    assert row["status"] == "match"
    assert row["excluded_se_keys"] == ["overall"]
    assert row["excluded_error_gap"] == pytest.approx(0.1)


def test_printed_report_explains_known_differences(capsys):
    keys = ["ATT", "ACRT", "ATT dose 1", "ACRT dose 1"]
    details = compare.comparison_details(
        (keys, [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]),
        reference(keys, [1.0, 2.0, 3.5, 4.0], [0.1, 0.2, 0.3, 0.4]),
        "cont_did",
        None,
    )
    compare.print_report([{"case": "cont_did", "workload": "baseline", **details}], 1)
    printed = capsys.readouterr().out
    assert (
        "cont_did on the baseline workload has known differences in 2 estimates (largest gap 5.00e-01) "
        "and 4 standard errors (largest gap 0.00e+00). The reference evaluates the dose curves"
    ) in printed
    assert "known |" in printed


def test_continuous_rows_include_all_curves_and_deterministic_inference():
    fitted = SimpleNamespace(
        dose=np.array([0.5, 1.0]),
        overall_att=1.0,
        overall_acrt=2.0,
        att_d=np.array([3.0, 4.0]),
        acrt_d=np.array([5.0, 6.0]),
        overall_att_inf_func=np.array([1.0, -1.0]),
        overall_acrt_inf_func=np.array([2.0, -2.0]),
        att_d_inf_func=np.array([[3.0, 4.0], [-3.0, -4.0]]),
        acrt_d_inf_func=np.array([[5.0, 6.0], [-5.0, -6.0]]),
    )
    keys, estimates, errors = compare.python_rows("cont_did", fitted)
    assert keys == ["ATT", "ACRT", "ATT dose 1", "ATT dose 2", "ACRT dose 1", "ACRT dose 2"]
    np.testing.assert_array_equal(estimates, np.arange(1.0, 7.0))
    np.testing.assert_allclose(errors, np.arange(1.0, 7.0) / np.sqrt(2))


def test_case_records_raw_timings_and_actual_workload(monkeypatch):
    monkeypatch.setattr(compare, "prepare_python", lambda *args: (lambda: SimpleNamespace(att=1.0, se=0.2), [1, 2]))
    monkeypatch.setattr(compare, "package_version", lambda package: "1.0")
    monkeypatch.setattr(compare, "run_reference", lambda *args: {**reference(), "seconds": [0.1, 0.2, 0.3]})
    result = compare.run_case("drdid", "baseline", None, 3)
    assert result["status"] == "match"
    assert len(result["python_seconds"]) == 3
    assert result["r_seconds"] == [0.1, 0.2, 0.3]
    assert result["R"] == 0.2
    assert result["r_iqr"] == pytest.approx(0.1)
    assert result["workload_spec"]["n_periods"] == 2
    assert result["workload_spec"]["n_cohorts"] == 1
    assert result["data_rows"] == 2


def test_python_failure_is_recorded_instead_of_aborting(monkeypatch):
    def failed(*args):
        raise RuntimeError("estimator failed")

    monkeypatch.setattr(compare, "prepare_python", failed)
    result = compare.run_case("drdid", "baseline", None, 1)
    assert result["status"] == "Python failed, RuntimeError: estimator failed"
    assert result["workload_spec"]["n_periods"] == 2


def test_aggregation_has_one_warmup_outside_timing(monkeypatch):
    result = SimpleNamespace(overall_att=1.0, overall_se=0.2, event_times=None, att_by_event=None, se_by_event=None)
    aggregate = Mock(return_value=result)
    prepare = Mock(return_value=aggregate)
    monkeypatch.setattr(compare, "make_estimator", lambda *args: SimpleNamespace(keywords={"data": [1, 2]}))
    monkeypatch.setattr(compare, "make_aggregation", prepare)
    monkeypatch.setattr(compare, "package_version", lambda package: "1.0")
    monkeypatch.setattr(
        compare, "run_reference", lambda *args: {**reference(keys=["overall"]), "seconds": [0.1, 0.2, 0.3]}
    )
    row = compare.run_case("aggte", "baseline", "simple", 3)
    assert row["status"] == "match"
    prepare.assert_called_once_with("att_gt", "baseline", "simple", warmup=False)
    assert aggregate.call_count == 4


def test_continuous_reference_matches_python_dose_grid_and_inference():
    python = compare.make_estimator("cont_did", "baseline")
    _, r_call, _ = reference_code("cont_did", compare.reference_options("cont_did", "baseline", None))
    np.testing.assert_allclose(python.keywords["dvals"], np.linspace(0.6, 1.9, 20))
    assert python.keywords["cband"] is False
    assert python.keywords["biters"] == 199
    assert "dvals = seq(0.6, 1.9, length.out = 20)" in r_call
    assert "cband = FALSE" in r_call
    assert "bstrap = TRUE, biters = 199" in r_call


@pytest.mark.parametrize("estimator", [name for name in compare.ESTIMATORS if name not in compare.AGGREGATES])
def test_estimator_factory_passes_only_arguments_the_estimator_takes(estimator):
    profiles = [
        profile
        for profile in common.PROFILES.get(estimator, tuple(WORKLOADS))
        if estimator != "cont_did" or not WORKLOADS[profile].n_covariates
    ]

    calls = [common.make_estimator(estimator, profile) for profile in profiles]

    for call in calls:
        inspect.signature(call.func).bind(*call.args, **call.keywords)


@pytest.mark.parametrize("estimator", sorted(compare.AGGREGATES.values()))
def test_aggregation_factory_passes_only_arguments_the_aggregation_takes(monkeypatch, estimator):
    fitted = SimpleNamespace()
    monkeypatch.setattr(common, "make_estimator", lambda *args: lambda: fitted)
    monkeypatch.setattr(common, "check_estimate", lambda *args: None)

    call = common.make_aggregation(estimator, "small", "dynamic", warmup=False)

    assert call.args == (fitted,)
    inspect.signature(call.func).bind(*call.args, **call.keywords)


def test_json_and_csv_replace_nonfinite_values_with_missing_values(tmp_path):
    output = tmp_path / "reports" / "comparison.json"
    csv_path = tmp_path / "reports" / "comparison.csv"
    report = {"results": [{"case": "x", "estimate": np.nan, "timings": [1.0, np.inf], "nested": {"x": -np.inf}}]}
    reporting.write_report(report, output, csv_path)

    def invalid_constant(value):
        pytest.fail(f"Nonstandard JSON constant {value}")

    saved = json.loads(output.read_text(), parse_constant=invalid_constant)
    assert saved["results"][0]["estimate"] is None
    assert saved["results"][0]["timings"] == [1.0, None]
    assert saved["results"][0]["nested"] == {"x": None}
    with csv_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["estimate"] == ""
    assert json.loads(row["timings"]) == [1.0, None]
    assert json.loads(row["nested"]) == {"x": None}


@pytest.mark.parametrize(
    "status,strict,exit_code",
    [("differs", False, 1), ("partial", True, 1), ("partial", False, 0), ("skipped, unsupported", True, 0)],
)
def test_reports_are_saved_before_exit(monkeypatch, tmp_path, status, strict, exit_code):
    arguments = SimpleNamespace(
        estimators=["drdid"],
        workloads=["baseline"],
        aggregations=["simple"],
        repeats=1,
        output=tmp_path / "results.json",
        csv=tmp_path / "results.csv",
        strict=strict,
    )
    monkeypatch.setattr(compare, "parse_args", lambda: arguments)
    monkeypatch.setattr(compare, "capture_environment", lambda estimators: {"cpu": "test CPU"})
    monkeypatch.setattr(compare, "print_environment", lambda estimators: None)
    monkeypatch.setattr(compare, "print_report", lambda rows, repeats: None)
    monkeypatch.setattr(compare, "run_case", lambda *args: {"case": "drdid", "workload": "baseline", "status": status})
    if exit_code:
        with pytest.raises(SystemExit, match="1"):
            compare.main()
    else:
        compare.main()
    report = json.loads(arguments.output.read_text())
    assert report["summary"] == {status: 1}
    assert report["environment"]["cpu"] == "test CPU"
    assert report["configuration"]["seed"] == 42
    assert report["generated_at"].endswith("Z")
    assert arguments.csv.exists()


def test_source_fingerprint_covers_added_source_and_ignores_documentation(monkeypatch, tmp_path):
    (tmp_path / "moderndid").mkdir()
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "moderndid" / "core.py").write_text("original\n")
    added = tmp_path / "benchmarks" / "reporting.py"
    added.write_text("first\n")

    def git_files(root, *arguments):
        return "moderndid/core.py\0" if "--cached" in arguments else "benchmarks/reporting.py\0"

    monkeypatch.setattr(reporting, "git_output", git_files)
    original = reporting.source_fingerprint(tmp_path)
    (tmp_path / "documentation.rst").write_text("Documentation change\n")
    assert reporting.source_fingerprint(tmp_path) == original
    added.write_text("second\n")
    updated = reporting.source_fingerprint(tmp_path)
    assert updated["sha256"] != original["sha256"]
    assert updated["tracked_sha256"] == original["tracked_sha256"]
    assert updated["files"] == ["benchmarks/reporting.py", "moderndid/core.py"]


def cpu_probe(monkeypatch, sysctl_locations):
    calls = []

    def which(command, path=None):
        return sysctl_locations.get(path)

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Apple M2 Pro\n")

    monkeypatch.setattr(reporting, "platform", SimpleNamespace(system=lambda: "Darwin", processor=lambda: "arm"))
    monkeypatch.setattr(reporting, "shutil", SimpleNamespace(which=which))
    monkeypatch.setattr(reporting, "subprocess", SimpleNamespace(run=run))
    monkeypatch.setattr(reporting, "Path", lambda path: SimpleNamespace(is_file=lambda: False))
    return calls


def test_cpu_name_reads_sysctl_in_usr_sbin_when_path_omits_it(monkeypatch):
    calls = cpu_probe(monkeypatch, {"/usr/sbin": "/usr/sbin/sysctl"})
    assert reporting.cpu_name() == "Apple M2 Pro"
    assert calls == [["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]]


def test_cpu_name_falls_back_to_sysctl_on_path(monkeypatch):
    calls = cpu_probe(monkeypatch, {None: "/opt/tools/sysctl"})
    assert reporting.cpu_name() == "Apple M2 Pro"
    assert calls == [["/opt/tools/sysctl", "-n", "machdep.cpu.brand_string"]]


def test_cpu_name_falls_back_to_platform_metadata_without_sysctl(monkeypatch):
    calls = cpu_probe(monkeypatch, {})
    assert reporting.cpu_name() == "arm"
    assert calls == []
