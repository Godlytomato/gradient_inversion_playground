# Pre-push validation

Validated on October 18, 2026, in the cloud CPU environment. No commit or push was
performed. Existing experiment data was preserved.

## Results

| Check | Result |
| --- | --- |
| Clean virtual environment | Installed all pins from `requirements-cpu.txt`; no system site packages required. |
| Repeatable installation | Re-executed the reusable install script against the new environment successfully. |
| Dependencies | `pip check`: no broken requirements. Python 3.12.14, torch 2.8.0+cpu, torchvision 0.23.0+cpu, Streamlit 1.66.0. |
| Full regression suite | **76 tests passed** in about 9 seconds using the clean environment. Includes 702 description/YAML round trips. |
| Compilation/whitespace | All 322 Python source files compile; `git diff --check` passes. |
| CIFAR-10 | Downloaded from the official Toronto host, retained TLS verification, and matched torchvision's archive MD5 `c58f30108f718f92721af3b95e74349a`. Standard torchvision loading/extraction succeeded. |
| Real-data runner | Four paired LeNet/DLG experiments completed on the actual CIFAR-10 test split: none, GradDefense, Soteria, max-SVD noise. |
| Persistence | Eight iteration logs and twelve images saved; practical/oracle selections, metadata and diagnostics round-tripped. Re-execution skipped completed configurations. |
| Applied noise | Real-data GradDefense requested relative L2 0.01 and measured 0.01000000695. |
| Checked max-SVD | Real-data construction converged in 99 steps, relative residual 9.80e-5 at tolerance 1e-4. A separate synthetic runner case required 278 steps. |
| Streamlit | AppTest: no exceptions/errors, five tabs, filters and comparisons exercised. Live server health and page requests returned HTTP 200; validation server stopped cleanly. |
| Reporting commands | Database initialization/migration, viewer, image export and stratified heatmap commands completed; black/white pixel rendering was preserved. |
| Historical data | Existing `database/playground.db` SHA-256 was identical before and after all checks. Tests used separate databases. |

The real-data smoke runs used **two outer iterations, one restart and LBFGS
max_iter=1**, preserving the actual line-search behavior. They exhausted that
small budget and were recorded as reconstruction failures against target MSE
0.02. The execution checks passed; these runs do not establish reconstruction
quality, attack success rates, or privacy guarantees.

## Failure found and corrected during this validation

The broad runner smoke exposed max-SVD public presets still passing the old hard
ten-step budget to the checked solver. The numerical helper had adaptive support,
but the presets prevented its use. Their defaults now request adaptive refinement
while preserving explicit hard iteration limits.

The synthetic runner also showed that the earlier 200-step adaptive cap was
insufficient for a slower spectral gap. The default cap is now **500**, with
unchanged residual tolerance and early stopping. Added regressions cover public
preset forwarding and slower-gap convergence. The full suite and both synthetic
and real-data runner checks passed after these corrections.

## Scope and remaining limits

- GPU execution was unavailable and untested.
- Full-budget scientific sweeps, trained-model utility and external-paper
  replication were not run.
- Large-model minimum-SVD may still reject unconverged or numerical-null
  directions; this is an explicit unresolved numerical case, not a successful
  defense result.
- Lower-bound calculations retain `certified=False`; no formal guarantee was
  established by these tests.
- Cloud install/start instructions were refreshed in the environment draft.
  Draft saving does not publish the environment or push the Git repository.

## Reproduce

Follow the clean CPU installation in `README.md`, then run:

```bash
MPLCONFIGDIR=/tmp/playground-matplotlib XDG_CACHE_HOME=/tmp/playground-cache \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
git diff --check
```

Additional local validation scripts/logs are in `/workspace/prepush-tests`:
`install.log`, `install-repeat.log`, `regression-final.log`,
`realdata/runner_smoke.py`, `realdata/real/results.json`, and
`viewer/validate.py` / `viewer/results.json`. They are local evidence artifacts,
not tracked application dependencies. The verified CIFAR cache is in the
repository's ignored `data/` paths.
