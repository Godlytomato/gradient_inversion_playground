# Secondary experiment: sparse gradient disguise

This experiment separates two claims for CIFAR-10 test image 193, 100 public
training images, and three servers:

1. All shares recover the private gradient, while every proper coalition sum is
   closer in squared gradient distance to a public dummy gradient.
2. Gradient inversion of those coalition observations reconstructs images closer
   to public dummy images than to the private image. This must be measured; the
   first claim does not imply it.

No production SQLite database is opened. The model, registered attacks, losses,
optimization engine, and observer/stopping policies come from the Playground.
See [the implementation review](REVIEW.md) for the observed baseline failure,
its limits, and the corrections. Trained weights and historical result tensors
from the user's computer are not included in this repository.

## Start with the existing checkpoint and shares

Activate your Python environment and run from the repository root. If the
checkpoint is at the path reported in the user's terminal:

```bash
python -m experiments.coalition_disguise.run \
  --checkpoint results/cifar10_disguise/lenet5_fixed_state_dict.pt \
  --output results/secondary_disguise_193 --diagnose-only
```

This inspects saved shares and old reconstructions without rerunning attacks or
changing old artifacts. It writes `diagnosis_v2.json`, including raw versus
clamped image MSE, pixel range, out-of-range fraction, identical final objective
rows, and the true-label L2 objective's derivative with respect to the image.
These are explicitly evaluation-only diagnostics. If there are no saved shares,
use `--prepare-only` instead to generate and audit them first.

Run the corrected paired baseline controls:

```bash
python -m experiments.coalition_disguise.run \
  --checkpoint results/cifar10_disguise/lenet5_fixed_state_dict.pt \
  --output results/secondary_disguise_193 \
  --presets idlg --baseline-only
```

The default `--attack-profile calibrated` applies recorded modifications:

| Attack | Initialization and domain | Restarts | Optimizer changes |
| --- | --- | ---: | --- |
| iDLG / DLG | Uniform raw pixels in (0.05,0.95); optimize sigmoid coordinates, so every trial image remains in [0,1] | 4 | LBFGS `tolerance_grad=1e-12`, `tolerance_change=1e-15` |
| Geiping | Uniform raw pixels; existing clamp, cosine loss, sign update, and TV penalty | 8 | Adam learning rate 0.01 |

Sigmoid coordinates avoid projecting LBFGS steps after its line search and
invalidating its curvature history. This changes optimization conditioning;
it does not guarantee recovery. Image-coordinate and variable-coordinate
gradient norms are logged so sigmoid saturation can also be investigated.
Original registered presets are unchanged: `--attack-profile preset` uses their
Gaussian initialization, optimizer defaults, and restart counts. CLI overrides
`--lr` and `--restarts` apply to either profile. Effective settings, including
optimizer defaults, code hashes, versions, observation/model hashes, and attack
label information are saved per job.

Both controls use **identical settings and seeds**:

1. `full`: sum of the actual saved float32 shares (always runs first).
2. `original_control`: undecomposed private gradient, explicitly identified as a
   diagnostic positive control. It is never substituted for the full-share sum.

`calibration.json` reports which stage failed and the difference between the
control reconstructions. If both controls fail, the baseline inversion remains
uncalibrated. If only the share-sum control fails, aggregation sensitivity or a
different optimization basin needs investigation before interpreting privacy.

To continue to all six coalitions when the controls pass:

```bash
python -m experiments.coalition_disguise.run \
  --checkpoint results/cifar10_disguise/lenet5_fixed_state_dict.pt \
  --output results/secondary_disguise_193 --presets idlg
```

Completed controls are reused. Add `dlg geiping` to `--presets` to evaluate those
configurations independently. Each must pass its own paired calibration gate.

## Calibration, stopping, and selection

Both selected control reconstructions must have private-image MSE < 0.01, valid
pixel range, and all configured restarts stopped by objective stagnation. No
image similarity criterion stops an attempt or selects its returned image.
The existing `no_improvement` policy watches `attack_loss`, with patience 200
and absolute `--minimum-change 1e-10` by default. Internal LBFGS tolerances are
separate from this observer policy. Optional `--max-iterations` provides a safety
cap; exhausting it fails calibration. Stall-only runs may be long.

The shared run builder requires a safety cap. This experiment validates all
components with that cap, then removes only its own resolved and recorded cap
when stall-only execution was requested. Shared validation is unchanged.

Private/public images and the original gradient are absent from coalition
attack information. The original gradient appears only as the explicitly
identified undecomposed control's observation. Geiping receives a known label;
iDLG infers one; DLG optimizes soft labels. Selection across steps and restarts
always minimizes the attacker-visible objective. Evaluation occurs afterward.

The new generic initializers and parameterized-image support are opt-in additions
to the shared engine. Existing preset descriptions and hashes are unchanged.
Pixel gradient/update mods are rejected for parameterized images; the calibrated
sigmoid path is used only for iDLG/DLG, whose presets have none.

## Inputs, decomposition, and safe reuse

Supply a trained RGB CIFAR LeNet5 **state_dict** compatible with
`models.lenet.LeNet5` adapted to 3 channels and 32x32 inputs. CIFAR must already
be under `data/`; this command does not train or download anything. Preprocessing
is `ToTensor`, raw [0,1], with cross-entropy loss. Tensor compatibility is checked;
checkpoint training provenance must be established separately.

Coordinates are randomly assigned to servers, followed if needed by cancelling
single-coordinate perturbations `t * alpha[i]`, with `sum(alpha) == 0`.
The solver searches public gradients bracketing a coordinate and active server
sets with one negative coefficient. Squared-distance margins are affine in `t`;
it searches nonnegative boundaries and keeps the first feasible one per candidate.
Among audited candidates, extra nonzeros are minimized before nearest-public
distances. This restricted heuristic does not establish global optimality or
prove infeasibility when its budget is exhausted.

Observations are **raw sums**, never N/|S| rescaled. The full set is excluded
from disguise verification. Audits independently check float64 construction,
float32 serialization, every proper coalition, overlaps, nonzeros, norms, and
amplification. Dense artifacts do not implement sparse wire/storage costs.

Existing decomposition artifacts, original manifest, and historical attack
folders are preserved. Decomposition compatibility uses checkpoint/data hashes,
parameter ordering, public indices, preprocessing, and generation arguments.
A reconstruction code update no longer invalidates shares. Every reuse checks
the saved artifact hash and recomputes the current private gradient. Execution
code/configuration changes create new hashed attack folders, so old failures
are never mistaken for corrected runs. Different decomposition inputs require a
fresh output directory. If you originally changed generation options, pass those
same options again.

## Outputs

- `input_checks_v2.json`, `audit_v2.json`, `coalition_geometry_v2.csv`: verified
  input routing, label inference, construction/serialization, and six coalitions.
- `decomposition.pt`: original/public gradients, exact/serialized shares,
  ownership, repair coordinate, coefficients, magnitude, and solver metadata.
- Per job: `job.json`, `attack_config.json`, `trajectory.csv`, `summary.json`,
  `reconstruction.pt`, reconstruction PNG, and comparison PNG (private,
  reconstruction, best-SSIM public image).
- Per configuration: `calibration.json`, with a stage-specific diagnosis.
- `report.json` / `report.csv`: all evaluated configurations, including historical
  ones. Configurations and profiles are separate rows, never silently replaced.

Traces record objective, pixel range, out-of-range fraction, update size, inferred
or optimized label, and periodically the objective derivatives in image and
optimization coordinates (`--diagnostics-interval`, default 20). Summaries include
restart effort and stopping reasons. Zero image movement plus flat loss indicates
an ineffective search, not proof of privacy or successful image recovery.

Evaluation records private/public MSE and SSIM, best public indices, advantages,
actual-label gradient L2 mismatch, and a separately named true-label diagnostic.
DLG's selected soft labels are retained, so its gradient mismatch is evaluated
with its actual selected labels. Cosine/regularized objectives remain distinct
from squared L2 gradient residuals. PNGs clamp for display; metrics use actual
selected tensors, and clamped-image MSE is only an additional diagnostic.

A completed reconstruction is committed before image evaluation and reused on
resume. Changed model/observation/configuration identities or modified saved
reconstructions reject resume. Interrupted optimizations restart that job, not
other completed jobs. `--diagnose-only` reads old artifacts without replacing them.

Adaptive robust/partial-gradient attacks, trained-model baseline calibration on
the user's checkpoint, and a successful full coalition comparison remain pending.
No current result establishes a privacy guarantee.

## Checks

```bash
python -m unittest discover -s tests -p test_coalition_disguise.py -v
python -m unittest discover -s tests -v
```

Tests include sparse aggregation/rounding, margin algebra, zero/one-extra-nonzero
cases, unchanged original presets, bounded trial images through LBFGS closures,
recovery on an identifiable linear fixture, paired original/aggregate controls,
physical versus sigmoid gradients, calibration gates, old manifest migration,
resume identity protection, report preservation, and evaluation label consistency.
