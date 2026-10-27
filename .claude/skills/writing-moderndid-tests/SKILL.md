---
name: writing-moderndid-tests
description: Use when writing, changing, reviewing, or deleting a test in tests/, when fixing a bug or adding or changing behavior in moderndid/, when porting an estimator, when asked for tests or more coverage of existing code, or when a change makes an existing test fail and its expected value, tolerance, skip, or warning check seems to need changing. Holds the project's test-first workflow and its standards for expected values, test data, tolerances, randomness, errors and warnings, fixtures, layout, and GPU tests. R validation tests in tests/validation belong to comparing-with-r. Library code style belongs to writing-moderndid-code.
---

# Writing moderndid tests

A test earns its place only by failing on wrong code. One that passes against the bug it was written for, or against a plausible wrong version of the change, checks nothing and still costs every later change a maintenance step. No test counts until you have seen it fail for the reason it exists, either on the code before your change or under a deliberate break of the code after it. The Code section of `.claude/CLAUDE.md` sets the base rules. For contributors, `docs/source/contributing/testing.rst` explains the reasoning behind them.

## The test-first loop

Every change to what moderndid computes or prints goes through this loop, whether it fixes a bug, adds an argument, or ports an estimator. A port starts from the paper, with a known truth recovered on simulated data, a small case computed by hand, or the influence-function identity, each written before the computation exists. Refactors and speedups follow their own section below.

1. Name the break. Before writing the test, write down the wrong code it has to catch. For a bug fix the break is the bug itself. For new behavior it is the missing feature and its likeliest wrong version, such as a term added instead of subtracted, a weight left out, the wrong comparison group, or an argument that never reaches the computation. A fix with several parts names one break per part. If no realistic wrong code would fail the test, it checks nothing and needs a behavior that can go wrong.
2. Take the expected value from a source that doesn't run the code under test, as Expected values describes.
3. Before writing the assertion, work out what the wrong code returns on your data. If it returns the right answer there, change the data rather than the assertion, using Data that separates.
4. Run the new test before changing library code and read the failure. It has to fail at its assertion, on a wrong number or on an error or warning that never came. An import error, a fixture error, a collection error, or a typo is not a red run. Fix it and run again. A new argument goes into the signature first and stays unused, so that the red run fails at the assertion on the likeliest wrong version of any new argument. A new function can only fail on its missing name. Its proof comes from step 6.
5. Change the library code until the test passes. Then run every test file that covers the change. Leave the test alone between the red run and the green run. If the test turns out wrong, say so in the report and go back to step 1 for a new red run.
6. Put each named break back into the library code, run the new tests, and see a named test fail at its assertion. Undo the edit before anything else runs, as Mutating safely describes. Mutate each part of a fix on its own, since two errors can cancel. Reverting both signs of the fix to the repeated cross-section influence function moves the test's entries by 0.18. Reverting either sign alone moves them by more than 4. A guard, check, or warning needs two mutations. Deleting it shows that it fires when it should. Widening its condition until it always fires shows that it stays quiet when it should. Only a control test, whose data must not trip the guard, catches that second mutation. Every guard gets one.
7. Report the evidence in the form under Report.

A break that no test catches is a gap or an equivalent mutant. A gap gets a test. An equivalent mutant is an edit no input can observe, such as lowering the period threshold in `create_bounds_second_difference_matrix` from 3 to 2, where two periods give zero rows either way. It gets a sentence in the report instead of a test named as if it guarded the line.

## Expected values

Each expected value comes from outside the code path it checks.

- A closed form or a small case computed by hand, such as a two-by-two difference of means on a dozen rows.
- The same quantity computed another way, such as `np.linalg.lstsq` or statsmodels WLS for a regression, or `npiv` called directly for the CCK curve that `cont_did` reports.
- An identity the method implies. The `weight_derivative` fixture in `tests/conftest.py` checks an influence function as n times the derivative of the estimate in each observation's weight. A standard error can be rebuilt from the stored influence function, summed within clusters with `np.bincount` when the fit clusters. An aggregation can be rebuilt as the paper's weighted average of its cells.
- An invariance, such as shuffled rows, relabeled units, integer weights against replicated units, parallel against sequential runs, shards against an in-memory fit, and GPU against CPU.
- A truth planted by a `gen_*` simulator, with the tolerance taken from the Monte Carlo standard error as the auditing-estimators skill describes.

Values from R belong in `tests/validation/` under the comparing-with-r skill. An expected value never comes from moderndid's own output, from a helper the code under test calls, from a copy of the code's own formula, or from running the changed code, since each of those shares the code's mistakes. A number pinned from an earlier run of moderndid detects change without checking correctness. New tests don't pin one.

Invariances run on fixed datasets and a few seeds rather than on generated inputs. Generated inputs find failures at floating-point extremes that real data never reaches. Those failures don't call for new guards.

## Data that separates

Wrong code often returns the right number on convenient data. These choices hide moderndid bugs most often.

- Equal weights hide a weight that the code drops. Draw unequal weights.
- A balanced panel hides bugs in balancing and in the unbalanced-panel path.
- One cohort, or cohorts of equal size and effect, hide the aggregation weights.
- One unit per cluster hides clustering.
- Without not-yet-treated units, the two `control_group` options give the same answer.
- An intercept-only fit hides covariate terms. The two pre-period regression terms of the repeated cross-section influence function cancel there.
- An argument left at its default, such as `biters=1000`, hides an argument that never reaches the computation. Run two settings and assert that the result moves, as in `assert abs(few.att_d_crit_val - many.att_d_crit_val) > 1e-2`.
- A check on slices of an array hides wrong entries outside them. Compare the whole array.
- An exact boundary can take a different path than a value one rounding step away. Without its guard, the zero-variance branch of the ARP test still reaches the right decision at a correlation of exactly -1 and goes wrong only at `np.nextafter(-1.0, -2.0)`.
- Hand-built data shares your assumptions. When a bug came from real data, check the fix on a real dataset such as `load_mpdta()` or `load_nsw()` too.

## Assertions that can't fail

- An assert inside an `if`, a `for`, or a `try`. A condition that skips it, a loop over an empty collection, or a caught exception passes in silence. Assert the precondition first, such as the number of cells.
- `assert a or b`, or a substring check such as `"ATT" in report` where the count or the values matter.
- A type, shape, `is not None`, or sign check in place of a value.
- A stub that replaces the code under test. Run the real code and stub only a random or external boundary, such as bootstrap draws through `fixed_draws`. Assert on moderndid's result, never on the stub's calls.
- A test that skips in your environment. An `importorskip` module or a `gpu` or `spark` mark can skip a whole file. Read the `-ra` summary before counting a run as evidence.

## Tolerances

- Measure the tolerance. After the fix, compute the largest absolute and relative error against the expected value, and the gap under each named break. The tolerance sits just above the first and far below the second. The report gives both numbers.
- A closed-form path reproduces itself to about `1e-12` after its operations are reordered, an iterative fit such as a logit propensity score to about `1e-8`, and a finite difference to about `1e-6`. Find the cause of any larger difference before touching a tolerance.
- `np.testing.assert_allclose` counts NaN as equal to NaN. Assert finiteness wherever a value must be finite.
- An expected value near zero needs `atol=0` and a relative tolerance, since an absolute tolerance would accept a result that collapsed to zero.
- A simulation check takes its tolerance from the Monte Carlo standard error. When the gap under the named break isn't several times that tolerance, the test can't see the break and the data has to change.
- Never widen a tolerance to make a test pass.

## When an existing test fails

Read what the failing test asserts before touching it.

- If it checks only a type, a shape, `is not None`, or a sign, it never covered the behavior. Add a real test beside it.
- If it pins the right value, your change is the regression. Fix the change.
- If it pins the wrong value because it matched the bug, update the number only from an independent source as in step 2. Grep `tests/` for every other place the old value appears. Report each old and new value with its source.

Nothing else makes a failing test pass. Widening its tolerance, rewriting its expected value or the formula behind it to match the new output, adding a skip or an xfail, and deleting it all hide the failure instead of explaining it. When a request can only be met that way, such as getting the tests green without touching the change that broke them, stop and report the cause with its numbers. The maintainer decides what happens next.

## Tests for existing code

Since a coverage push, a characterization test, or a test written after its fix can't be seen failing first, the mutations in step 6 are its whole proof. Derive each expected value from the docstring, the paper, or an independent computation, and never from printing the function's output. A test built from today's output pins today's bugs. Then delete or break the line each test claims to cover and see it fail. A test that still passes never reaches that line. Coverage shows which lines ran and never which ones a test checked. A coverage goal relaxes no step of the loop.

When the docstring or the paper says one thing and the code does another, the test asserts the documented behavior and fails. A new test that exposes a bug outside your task stays as written. Report the bug and let the maintainer choose between fixing it now and a strict xfail that names it.

## Refactors and speedups

A refactor or speedup leaves every number where it was. Its existing tests pass unchanged before and after it. If no test with an independent expected value covers the code you're changing, write one first, see it pass on the old code, and prove it with a mutation before refactoring. Treat a refactor that moves a result past the reordering limits under Tolerances as a change in behavior under When an existing test fails.

## Red flags

Each of these thoughts has come before a test that passes no matter what. Stop and go back to the loop.

- "The rules forbid widening the tolerance, so I'll rewrite the expected value instead." Rewriting an expected value or its formula to fit new output is the same move, since both make the test agree with whatever the code now does.
- "The new expected value is still independent, since I wrote the formula out by hand." A formula picked to reproduce what the changed code returns is the changed code restated.
- "No public path reaches this code, so the change is harmless." The test pins the function's contract. Whether that contract changes is the maintainer's decision.
- "I'll make it green and ask the maintainer to confirm." Changing the test before the answer arrives has already made the decision. Report first and leave the test red.
- "The rewritten test can still fail, since it rejects the other variants." It checks the wrong thing however sharply it checks it.
- "The tests cover everything except the part the bug touches, so they won't need to change once the maintainer decides." Then they pass on the bug and on its fix alike and can't catch it. Assert the documented behavior and let it fail.
- "The test passed on its first run." Until it has failed on wrong code, it has proven nothing.
- "The expected value is what the function returns today." Built that way, the test pins today's bugs.
- "Deleting the guard made the test fail, so the guard is covered." That covers one direction. Widen the guard until it always fires and see the control test fail too.
- "It's a rounding difference." Measure it against the limits under Tolerances before believing it.

## Mutating safely

- Make each mutation as a small edit with the Edit tool and undo it the same way. Then check with `git diff` that the file holds only your intended change.
- Never undo with `git checkout`, `git restore`, or `git stash` on a file that holds other uncommitted work, yours or the maintainer's, since each of them discards that work too.
- Since moderndid is an editable install, every session on the checkout imports a mutated file. Run only the named tests while a mutation is in place. Undo it before anything else runs.
- A mutation that breaks the import or the collection tests nothing. Keep the module importable.
- In a git worktree, a script run by path imports moderndid from the main checkout through the editable install. Run tests and scratch checks from the worktree's root with `python -m pytest` or `python -c`, or set `PYTHONPATH` to the worktree.

## Report

Give each new or changed test in this form.

```
test: tests/drdid/test_drdid_rc.py::test_drdid_rc_influence_function_matches_weight_derivative
catches: the pre-period regression terms added instead of subtracted
expected from: n times the central difference of the ATT in each row's weight (weight_derivative)
red before the fix: 3 of 12 entries mismatched, largest absolute difference 0.179
mutations: treated term alone reverted -> failed (4.18); control term alone reverted -> failed (4.36)
tolerance: rtol=1e-6, atol=1e-6; error after the fix 2.2e-7; gap under the break 0.179
```

Below the forms, list each pytest command with its pass, fail, and skip counts. Run each with `-ra` so that every skip shows its reason. Give each existing test you changed with its old and new values and their source. Give each surviving mutation with the reason it is equivalent.

## Reviewing a test

A test written by another agent counts only with its evidence. Check that its report fills every line of the form above, rerun at least one named break yourself, and trace every literal number in the test to its source. A guard needs its control test. In `git diff tests/`, send back any assertion removed, tolerance widened, skip or xfail added, or expected value changed without a source.

## Conventions

The later half of `tests/didtriple/test_ddd.py`, from `test_ddd_alpha_above_tenth_uses_005_on_every_route` on, shows most of these rules in one file. The cleanest folders are `tests/etwfe/` and `tests/didinter/`.

- `moderndid/<family>/<module>.py` is tested in `tests/<family>/test_<module>.py`, flat even when the module sits in a subfolder. `didtriple/estimators/ddd_mp.py`, for example, is tested in `tests/didtriple/test_ddd_mp.py`. Plots are tested in `tests/plotting/` and the GPU backend in `tests/core/test_gpu*.py`. Every test folder has an `__init__.py` and a `conftest.py`.
- A test module opens with `"""Tests for <subject>."""` and its imports. Import the API under test from `moderndid`. Import a submodule only to reach a private helper. An optional dependency such as pyfixest or formulaic loads at the top through `from tests.helpers import importorskip`. That helper raises instead of skipping when `MODERNDID_REQUIRE_ALL_DEPS` is set. A slow module sets `pytestmark = pytest.mark.slow` after its imports to keep it out of `pixi run -e dev tests-core`.
- Tests are flat functions with no docstrings and no comments, named as behavior sentences such as `test_ddd_mp_never_treated_units_lost_to_balancing_leave_no_comparison`. There are no test classes and no module-level constants.
- Fixtures live in the folder's `conftest.py`, each with a one-line docstring that has no comma, and never in a test file. Reuse one before adding another, such as `mpdta_data` in `tests/did/conftest.py`. Parametrize over fixtures by name with `request.getfixturevalue(data_fixture)`. Arguments shared by several calls go in a `spec` dict passed as `**spec`.
- Lay a test out as its inputs, a blank line, the calls, a blank line, and the asserts.
- Compare numbers with `np.testing.assert_allclose(result, expected, rtol=..., atol=...)` and explicit tolerances. Use `np.testing.assert_array_equal` for labels and for runs that must be identical.
- Build data with the `gen_*` simulators in `moderndid/core/data.py` at `random_state=42`, or with `load_mpdta()`. Draw anything else from `np.random.default_rng(<seed>)`, never `np.random.seed`, `np.random.normal`, or `RandomState`. Pass `random_state` to every call that draws a bootstrap. That includes `att_gt` under its default `cband=True`, since `aggte` then draws one for the simultaneous bands.
- Test every error with `pytest.raises(ValueError, match=...)` on a distinctive part of the message. Wrap the match in `re.escape` when it holds regex characters. Never write `pytest.raises(Exception)`, since it also passes on an `AttributeError` from a missing export.
- Test an expected warning with `pytest.warns(UserWarning, match=...)`. To show that no warning is raised, record with `warnings.catch_warnings(record=True)` and `warnings.simplefilter("always")` and assert on the recorded messages. Since `tests/conftest.py` ignores numerical RuntimeWarnings such as divide by zero and invalid value across the suite, a test sees one only by recording it this way. A `filterwarnings` mark names only the warning the test is about. The `forbid_errstate` fixture in `tests/conftest.py` makes `np.errstate` raise. A test that takes it proves that a guarded path needs no suppression. Since SciPy's `linprog` calls `np.errstate` itself, a path through it records warnings instead.
- For bootstrap results that must match exactly, hand the code fixed draws through the `fixed_draws` and `draws_from_weights` fixtures in `tests/conftest.py`.
- `tests/conftest.py` skips every test marked `gpu` unless CuPy reaches a CUDA GPU. A test with an independent reference takes the `xp` fixture, runs under NumPy and CuPy, and asserts that promised device arrays come back as `xp.ndarray`. A comparison of a GPU run with the CPU run of the same call carries the marker alone and takes CuPy from the `cp` fixture. Test modules never import CuPy and switch backends only through `use_backend` or a `backend` argument. Assertions on GPU results allow for rounding in the last bits and judge CuPy's random draws by their distribution alone. Spark tests follow the Testing on Spark section of `docs/source/contributing/testing.rst`.
- Older files break these rules and are no model for new tests. Don't copy the test classes in `tests/core/` and `tests/didtriple/test_nuisance.py`, the fixtures inside `tests/didhonest/` test files, the `np.random.seed` calls, the smoke-only tests at the top of `tests/didtriple/test_ddd.py`, or the hundreds of type, shape, and `is not None` asserts across the older suite.
