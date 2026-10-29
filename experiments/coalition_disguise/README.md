# Secondary experiment: sparse gradient disguise

This isolated experiment tests two distinct claims for one CIFAR-10 test image
(default index 193), 100 distinct public training images, and three servers:

1. The actual saved float32 shares sum to the true gradient within tolerance;
   each of the six proper coalition sums is closer in squared gradient distance
   to some public dummy gradient than to the private gradient.
2. An ordinary gradient inversion attack on a coalition reconstructs an image
   closer to a public dummy image than to the private image. This is measured,
   not guaranteed by claim 1.

No shared reconstruction code or SQLite database is changed. The historical
`coalition_single_image_experiment_v3` runner, `disguise_search.py`, trained
checkpoint, and image-193 decomposition from the supplied handoff are absent
from this checkout. This implementation is a new isolated implementation of
that described construction, not a verification of the historical artifacts.

## Run

From the repository root, supply a trained RGB CIFAR LeNet5 **state_dict** matching
`models.lenet.LeNet5` (3 input channels, 32x32 input):

```bash
.venv/bin/python -m experiments.coalition_disguise.run \
  --checkpoint /path/to/lenet5_fixed_state_dict.pt \
  --output results/secondary_disguise_193 --prepare-only

.venv/bin/python -m experiments.coalition_disguise.run \
  --checkpoint /path/to/lenet5_fixed_state_dict.pt \
  --output results/secondary_disguise_193 --presets idlg dlg geiping
```

CIFAR must already be available under `data/`; there is no implicit download.
The checkpoint must use raw [0,1] RGB tensors without normalization, and must be
trained externally. The script validates tensor compatibility but cannot prove
training provenance or the preprocessing used during training.

The first command computes gradients and audits the decomposition. The second
reuses the exact decomposition. Each attack configuration gets its own hashed
subdirectory; `--lr`, `--restarts`, `--patience`, and `--minimum-change` allow
baseline calibration while preserving the shares. Seeds are explicit and the
same attack configuration/seeds apply to the full gradient and coalitions.

Default stopping uses the registered `no_improvement` policy on **attack_loss**
only. Playground validation requires a maximum-iteration safety policy; this
experiment validates with that policy and then removes it from its local
resolved observer and recorded description. The shared validator is unchanged.
Stall-only execution can run for a long time. Optional `--max-iterations 2`
provides a quick engineering smoke test; exhausting this cap fails calibration.

The full-set control uses the actual float32 share sum. Each preset must achieve
private-image MSE < 0.01 and stop all restarts by `no_improvement` before that
preset's six coalition attacks run. A failed calibration skips its coalitions;
try a new learning rate/restart budget in the same output directory. The MSE
criterion is an evaluation gate, never an optimizer stopping/selection rule.
Reconstruction selection across steps/restarts uses the lowest attack objective.
No oracle-selected image is used. Private images and original gradients are
absent from the attack information dictionary. Geiping's known-label setting
is recorded; iDLG infers labels and DLG optimizes them.

## Construction and limits

Coordinates are randomly assigned to servers. When necessary, the solver adds
`t * alpha[i]` at one coordinate; `sum(alpha) == 0`. It searches public gradients
bracketing that coordinate and active server sets with one negative coefficient.
For every coalition/public pair the squared-distance margin is affine in `t`.
The solver enumerates nonnegative boundaries and uses the first feasible boundary
for each candidate. Among audited candidates it minimizes extra nonzeros, then
worst/total nearest-public squared distance. This restricted search does not
establish global minimum storage or globally optimal repair magnitudes.

All coalition observations are raw sums: there is no N/|S| rescaling. Float64
construction precision and serialized float32 errors are independently audited.
A small extra numerical search margin protects serialization. Search exhaustion
is reported as search failure, not mathematical infeasibility. Repair coefficients,
coordinate ownership, exact and serialized shares, and original/public gradients
are saved in `decomposition.pt`.

This is an ordinary-attack experiment. Adaptive partial-gradient and robust
matching attacks remain future work. Dense saved arrays do not realize sparse
wire/storage costs. Public gradients constrain feasibility rather than directly
mixing dummy images into shares. A positive gradient margin alone proves no
image-space privacy guarantee.

## Outputs and resumption

- `manifest.json`: checkpoint/source/data hashes, parameter ordering, seeds,
  public indices and preprocessing; changed inputs refuse resume.
- `audit.json`, `coalition_geometry.csv`: six float32 coalition checks, norms,
  nonzeros, overlap, amplification, aggregation error.
- Per configuration/coalition: effective attack config, objective trajectory,
  selected reconstruction tensor/PNG, comparison PNG (private, reconstruction,
  best-SSIM public image), and JSON summary with effort/stopping reasons.
- `report.json` / `report.csv`: the configurations requested by the current
  invocation; prior per-configuration artifacts remain intact.

Completed jobs with intact reconstruction hashes are reused; interrupted jobs
without a summary restart from the beginning. Evaluation occurs after the attack
and reports private MSE/SSIM, best public MSE/SSIM and dataset indices, their
advantages, attack objective, and direct squared gradient L2 mismatch. The latter
uses the true label as an **evaluation-only** diagnostic, including for DLG/iDLG;
it is not their optimized label-dependent objective. The private-image oracle
residual is likewise evaluation-only. Display PNGs clamp to [0,1]; metrics use
the actual selected tensors. Original/public data are not passed to optimizers.

```bash
.venv/bin/python -m unittest discover -s tests -p test_coalition_disguise.py -v
```

Tests cover enumeration, distance algebra, a feasible zero-repair partition,
one-extra-nonzero repair, aggregation, serialization loss, raw sums, manifest
resume protection, attack information separation, local stopping adaptation,
and a real registered Geiping smoke attack with completed-job reuse.
