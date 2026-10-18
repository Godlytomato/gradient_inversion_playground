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

## Numerical limits and reporting

Singular directions carry convergence/rank diagnostics. Small problems use a
bounded exact SVD; larger ones use checked matrix-free estimates. Maximum-SVD
has bounded adaptive defaults; explicit iteration caps are honored. Minimum-SVD
can reject numerical nullspaces or unconverged solves. Larger budgets may help,
but the implementation will not replace an unresolved minimum direction with
normalized maximum-direction leakage. Do not count a rejected experiment as a
successful defense.

Lower-bound feasibility uses supplied/estimated constants. Its saved diagnostics
include the target, budget, residual and `certified=False`; empirical constants
and attack failure do **not** prove a guaranteed privacy bound. Minimum-direction
convergence on full CIFAR-shaped models, full attack-success sweeps, GPU execution,
training utility and external paper replication remain separate validation work.

The heatmap aggregates completed attempts within comparable model/data/evaluation
strata and separates attack budgets. Review sample counts and error/incomplete
attempts alongside success rates. Image viewers share raw-pixel display conversion
instead of silently rescaling each reconstruction's contrast.
