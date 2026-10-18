# Batch runner and lower-bound review

Validated in the cloud CPU environment on October 18, 2026. These checks cover
batch execution and numerical error handling; full-budget scientific sweeps and
GPU execution were not performed.

Subsequent [proof and ResNet review](LOWER_BOUND_AUDIT.md) found that the supplied
conditional bound requires exact gradient matching and smoothness assumptions
that these experiments do not establish. Execution checks below should not be
read as validation of those mathematical assumptions.

## Corrections

- Individual run errors previously produced exit status 1 after the schedule
  finished, which could stop an enclosing shell script. The default now exits
  normally after processing the schedule. `--strict-errors` restores that status
  for automation, and `--fail-fast` stops immediately for debugging.
- Console output identifies completed runs and continuation after errors.
  Full tracebacks remain in `results/run-errors.jsonl`; `--tracebacks` also prints
  them. Database initialization errors and keyboard interrupts still stop work.
- Reference-noise scaling used float32 norms that could underflow or overflow
  for finite directions. Normalization now uses float64, checks the result's
  representability, and records the actual applied norm.
- A zero reference target returns zero noise without a needless SVD solve.
  Reference and lower-bound diagnostics preserve the numerical stage that failed.
- Maximum-SVD could stop early on a float32 power-iteration proxy, then fail
  its final residual check with unused iterations remaining. The internal stop
  now leaves a margin for rounding differences. The final tolerance and declared
  iteration caps are unchanged; natural float32 regression cases cover both
  further refinement and acceptance at an explicit hard cap.

## Verification

The final regression suite passed **95 tests**, including 702 generated
description/YAML round trips. Dependency checks and `git diff --check` passed.

An additional fresh-database execution matrix used the actual cached CIFAR-10
test split and LeNet: all nine attacks without a defense, DLG across the other
25 defenses, and a final independent healthy sentinel. Each case used one
restart, at most two outer iterations and LBFGS `max_iter=1` where applicable.
The MSE stopping script configured success at 0.02 and failure on stalling;
numerical solver budgets remained at their preset defaults.

All **35 runs were recorded**: 31 normal reconstruction failures and four
minimum-SVD numerical errors. There were no unexpected application errors. The
healthy sentinel executed after the errors, and the process returned status 0.
The three maximum-SVD cases that initially stopped prematurely completed after
the stopping fix. Reference lower-bound cases warned about infeasibility and
executed their attacks. These short runs test execution, not reconstruction
success rates. The existing experiment database's SHA-256 was unchanged.

Fresh-process regressions execute both script and `python -m run` entry points, real
Jacobian/SVD failures in both ranking and reference scaling, then an actual LeNet
attack. They check process status, saved numerical diagnostics, subsequent attack
iterations, error retries and completed-run skipping. They replace only external
data with deterministic CIFAR-shaped tensors and use temporary databases.

Independent affine-gradient calculations check the first feasible lower-bound
root when both budget endpoints are infeasible, and the interior best-effort
optimum when the target is infeasible throughout. Extreme finite reference
directions and output-dtype overflow are covered separately.

Detailed local evidence is saved under
`/workspace/prepush-tests/recheck/runner_broad_final_13a4196/` (`broad_smoke.py`,
`results.json`, `timings.json`, `runs.db` and `run-errors.jsonl`). The full-suite
log is `/tmp/playground-recheck-final-tests.log`. These are validation artifacts,
not application dependencies.

## Interpretation and limits

`INFEASIBLE` is a calibration result under the supplied constants and perturbation
budget. The attack receives a bounded best-effort perturbation and continues;
the saved diagnostic is `infeasible_best_effort` with `certified=False`. This is
not evidence that MSE 0.02 is guaranteed. The numerical accuracy thresholds and
scientific constants were preserved.

Genuine nonconvergence remains possible. On actual CIFAR-10, one checked LeNet
maximum solve exhausted its default 500 steps at residual 0.0004383. Repeating
with the identical initial vector and an explicitly larger 1000-step cap reached
residual 0.0000998 after 704 steps. Such rejected solves are recorded as errors;
they do not prevent later scheduled runs from executing.

A representative ResNet18 maximum solve did not finish within a separate
90-second CPU validation limit. ResNet numerical convergence, full attack
budgets, trained-model utility and GPU execution are not established by these
checks. Minimum-SVD may still reject unresolved or null directions. Numerical
errors are distinct from normal attacks that stall or miss the target.

## Run and inspect

```bash
python -m run --schedule schedual
# Optional debugging and automation:
python run.py --schedule schedual --tracebacks
python run.py --schedule schedual --strict-errors
```

The final summary reports completed, skipped and errored runs. Restarting retries
errors and skips completed successes/failures. Existing YAML solver caps remain
in force; changing Python preset defaults does not rewrite saved YAML files.
