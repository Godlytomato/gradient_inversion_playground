# Gradient Inversion Playground

Python tools for gradient reconstruction experiments and a Streamlit results viewer.
The current implementation uses a **version 2 experimental protocol**. Regenerate
YAML descriptions and rerun affected experiments; old results do not become
corrected results when the source changes.

## Development

The cloud CPU environment was tested with Python 3.12 and PyTorch 2.8.0. Install
its pinned package snapshot in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt
.venv/bin/python -m unittest discover -s tests -v
```

The regression suite uses synthetic images and temporary databases. It does not
need CIFAR-10, GPUs, credentials, or access to an experiment database. The CPU
requirements file is specific to this workflow; CUDA installations require a
matching PyTorch/torchvision distribution and have not been validated here.

## Sparse gradient disguise experiment

For the seeded model/dataset gradient decomposition experiment, see
[the experiment guide](experiments/coalition_disguise/README.md) and
[the baseline failure review](experiments/coalition_disguise/REVIEW.md).
Select a model and dataset, draw a seeded sample, decompose its gradient, and
check full-share reconstruction before attacking all proper subsets. Each subset
is evaluated against its fixed gradient-nearest disguise and the ground truth.

## Experiments

Run commands from the checkout. Dataset loading is lazy: configuration generation
works offline, while the first real experiment downloads CIFAR-10 into `data/`.
The dataset source (`www.cs.toronto.edu` and its download redirect) must be allowed
by the cloud network policy. TLS and torchvision artifact checks remain enabled.

Generate a selected schedule rather than an expensive complete matrix:

```python
from validation.generator import generate_yamls

generate_yamls(
    attacks=["dlg"],
    defenses=["none", "graddefense", "soteria"],
    models=["lenet5"],
    datasets=["cifar10"],
    target=0.02,
)
```

```bash
.venv/bin/python run.py --schedule schedual
.venv/bin/python -m streamlit run interface.py --browser.gatherUsageStats false
```

The runner initializes/migrates the database non-destructively. Standalone code
using database writers should first call `database.setup.setup_database()`.
`python -m database.setup` is also non-destructive. Its explicit `--reset` flag
deletes the database and results. Do not reset an open/in-use database.

### Stop at the MSE target or when an attack stalls

After generating your run YAMLs, apply this explicit MSE-stopping configuration:

```bash
.venv/bin/python scripts/set_mse_stopping.py schedual --target 0.02 --patience 200 --minimum-change 0.0001
```

It stops successfully when MSE is at most 0.02 and fails an attempt after 200
iterations without a cumulative MSE improvement of at least 0.0001. Existing
restart counts and iteration safety caps are retained. The saved/returned image
is selected by best MSE so a successful reconstruction is not replaced by an
earlier image with lower gradient-matching loss. This mode explicitly uses true
image MSE for stopping and selection.

The script validates every changed description before writing, backs up the
original YAMLs as `*.before-mse-stop.bak`, and accepts `--dry-run` to preview.
It updates generated YAML files, not the Python preset builders: rerun it after
regenerating a schedule. Stop active experiments before applying new schedules;
already loaded runs will not change. Stored database results are not modified.

## What version 2 measures

- Victim and auxiliary tensors are raw `[0,1]` pixels. Models are randomly
  initialized and run in evaluation mode; this does not simulate trained victim
  weights or calibrated batch-normalization statistics.
- Generated runs use the selected attack preset's iteration cap. Ground truth
  does not stop the practical attack. The returned practical reconstruction is
  selected by observed attack loss, while a separate oracle image is selected
  by true-image MSE for evaluation. Both images and their metrics are saved.
- Generated target sweeps also forward the target to defenses that explicitly
  calibrate to `target_error`. Custom YAML can set defense and evaluation targets
  independently; report both when doing so.
- The target MSE is assessed after the budget against the practical selection.
  Oracle recovery is recorded separately. Explicit legacy ground-truth stopping
  policies remain available, but should be labeled oracle assisted.
- Batched evaluation uses minimum-cost one-to-one image assignment. Stored raw
  reconstructions retain attacker order; assignment maps are saved for display.
  The current iDLG and HF label policies require batch size one and reject larger
  batches rather than silently inferring unsupported labels.
- `experiment_seed` controls independent model, sampling, defense, and auxiliary
  streams. The runner derives and records the attack stream using that seed and
  the attack's `base_seed`. Configurations sharing the experiment seed/data/model
  use the same victim model and sampled examples. Use distinct experiment seeds
  for independent replicates.
- Auxiliary examples exclude private sample identities, including nested
  `Subset` indices. An external public dataset can instead be explicitly declared
  disjoint; the provider refuses to guess sample identities for opaque batches.
- Configuration hashes have a `v2:` prefix. Historical unprefixed results remain
  separate. Error/interrupted attempts can be retried as new rows; previous rows,
  logs, images and diagnostics are preserved. Completed configurations are skipped;
  change the experiment seed for another independent replicate.

See [defense protocol changes](docs/DEFENSE_PROTOCOL.md) for the precise Soteria
feature-column and GradDefense Frobenius-estimator definitions. They are declared
adaptations, not verified reproductions of external papers. Historical Soteria
YAML using a global draft is rejected with regeneration guidance.

See [pre-push validation](docs/PREPUSH_VALIDATION.md) for clean-install,
real-CIFAR runner and results-viewer checks and their limits.
See the [batch runner and lower-bound review](docs/RUNNER_READINESS.md) for the
subsequent continuation fixes, numerical checks and remaining limits.
The [proof and ResNet audit](docs/LOWER_BOUND_AUDIT.md) checks the supplied
derivation and explains why its exact-gradient-match and smoothness assumptions
are not established for the current noisy ReLU ResNet experiments.
The corrected [residual-aware bound](docs/RESIDUAL_BOUND.md) removes the exact-fit
requirement and reports conditional results using each saved image's actual
gradient residual. ReLU smoothness and supplied constants remain uncertified.
The [SVD/reference scientific-readiness diagnosis](docs/SCIENTIFIC_READINESS.md)
reproduces the reported ResNet failure and identifies remaining solver,
direction-consistency, masking and noise-budget issues in those comparisons.
Minimum-SVD presets now use the [minimum numerically nonnull protocol](docs/MINIMUM_POSITIVE_SVD.md)
with `solver_method: positive_svd_v1`, `rank_policy: smallest_nonzero`, and a
relative rank cutoff of `1e-5`. This handles CIFAR LeNet's proven input nullspace
by selecting its smallest resolved positive mode. Bounded dense QR/SVD covers
the supplied LeNet case; larger models retain checked iterative estimates and
explicit numerical limits. Unconfigured schedules adopt this policy before
hashing; historical results remain separate. To update the YAMLs themselves,
run `python scripts/set_min_svd_solver.py schedual` after stopping the runner.
No database reset is needed. The earlier [minimum-SVD solver review](docs/MINIMUM_SVD_SOLVER.md)
records the original full-minimum diagnosis and convergence repair.

## Numerical limits and reporting

The batch runner continues after individual run errors by default:

```bash
.venv/bin/python run.py --schedule schedual
```

Attacks that miss the target or stall remain completed `failure` outcomes.
Numerical/setup exceptions are saved as `error` attempts, with tracebacks appended
to `results/run-errors.jsonl`; subsequent YAML files still execute. Restarting the
same schedule skips completed successes/failures and retries error attempts.
The process exits normally after finishing the schedule even if individual runs
errored, so expected numerical failures do not stop a surrounding shell script.
Use `--strict-errors` to return status 1 after completing a batch with errors,
`--fail-fast` to stop immediately when debugging, or `--error-log PATH` to change
the log. Full tracebacks are always stored in that log; `--tracebacks` also prints
them to the console. Progress messages show when each run completes or the batch
continues after an error.
Keyboard interrupts still stop the batch. Database initialization errors stop
before the batch starts.

Singular directions carry convergence/rank diagnostics. Minimum-positive cases
within declared dense caps use CPU QR/SVD; larger cases use checked matrix-free
estimates. Maximum-SVD
has bounded adaptive defaults; explicit iteration caps are honored. Minimum-SVD
can reject numerical nullspaces or unconverged solves. Larger budgets may help,
but the implementation will not replace an unresolved minimum direction with
normalized maximum-direction leakage. Do not count a rejected experiment as a
successful defense.

Maximum-SVD YAML files keep their explicit solver limits. For a reference-scaling run,
`transform.parameters.scaling.parameters.reference_max_iterations` increases the
reference-direction cap. `svd_max_iterations` increases the spectral-norm cap
only for `legacy_proof_v1`; the corrected calibration needs no maximum SVD.
Maximum-SVD ranking and construction use their own
`max_iterations` parameters. New minimum-SVD presets use `positive_svd_v1`,
with dense resource limits and `krylov_max_iterations` for larger cases.
Unconfigured minimum schedules upgrade before hashing; the script linked above
also rewrites their YAMLs. Raising a cap keeps the residual tolerance unchanged
and still permits a checked numerical error if the solve does not converge.

Lower-bound calibration uses `directional_residual_v1` and explicitly covers
raw gradient residuals up to `residual_l2 + residual_fraction * noise_l2`
(defaults 0 and 0.1). Measured residuals and conditional bounds for the saved
images appear in the final summary's `experiment_metadata.reconstruction_bounds`.
Old YAML is upgraded in memory before hashing so historical bound results cannot
suppress corrected runs; the files and existing database rows are preserved.
Saved diagnostics include target, budget, residual condition and
`certified=False`; empirical constants
and attack failure do **not** prove a guaranteed privacy bound. An `INFEASIBLE`
warning means the requested target cannot be established under those constants
and perturbation budget. The attack continues with a bounded best-effort
perturbation, recorded as `infeasible_best_effort`; this does not certify MSE 0.02.
The reported ResNet/CIFAR minimum-direction case now converges. Broader case
replicates, full attack-success sweeps, GPU execution, training utility and
external paper replication remain separate validation work.

The heatmap aggregates completed attempts within comparable model/data/evaluation
strata and separates attack budgets. Review sample counts and error/incomplete
attempts alongside success rates. Image viewers share raw-pixel display conversion
instead of silently rescaling each reconstruction's contrast.
