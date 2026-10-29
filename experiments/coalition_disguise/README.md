# Seeded gradient decomposition experiment

The workflow is:

1. Select a model and dataset.
2. Draw one private test sample using the seed; select a distinct pool of public
   training disguises using an independent seeded stream.
3. Compute gradients with that model and construct the proposed sparse cancelling
   decomposition. Verify aggregation and every nonempty proper subset.
4. For each subset, mark its closest public disguise **in gradient space**, using
   the actual raw float32 subset sum. Save those assignments before any attack.
5. Reconstruct the full-share gradient and measure image MSE to ground truth.
6. If the full reconstruction passes the configured MSE target, reconstruct every
   proper subset. Measure its reconstruction against **that subset's preassigned
   disguise image** and against the ground-truth image.

There is one full-share baseline per attack configuration. There is no mandatory
second original-gradient control or additional calibration stage. The original
Playground attack engine is reused. No research SQLite database is opened.

## Run

Activate your existing Python environment and run from the repository root:

```bash
python -m experiments.coalition_disguise.run \
  --model lenet5 --dataset cifar10 --seed 2026 \
  --presets idlg \
  --output results/disguise_lenet5_cifar10_seed2026
```

Models: `lenet5`, `resnet18`, `resnet50`.
Datasets: `cifar10`, `mnist` (raw [0,1] tensors, cross-entropy, batch size one).
The model is adapted to the dataset's channels, dimensions and class count.

By default, the model uses seeded random weights, which are saved with the run.
A checkpoint is optional:

```bash
python -m experiments.coalition_disguise.run \
  --model lenet5 --dataset cifar10 --seed 2026 \
  --checkpoint results/cifar10_disguise/lenet5_fixed_state_dict.pt \
  --output results/disguise_trained_seed2026
```

Use `--data` to select the dataset directory (default `data/`). If a dataset is
missing, explicitly allow torchvision to fetch it with `--download`:

```bash
python -m experiments.coalition_disguise.run \
  --model lenet5 --dataset mnist --seed 2026 --download \
  --output results/disguise_lenet5_mnist_seed2026
```

Use `--private-index 193` to override random sample selection. The chosen index,
label, model state, sample hashes, public indices and generation settings are
saved. Independent streams prevent model construction from changing which sample
is selected. Defaults are three servers and 100 distinct public disguises.
`--servers N` enumerates all `2**N - 2` nonempty proper subsets (N from 2 to 8).
Public gradient storage scales with model size and `--public-count`.

## What counts as a successful baseline

The selected full-share reconstruction must have image MSE below
`--baseline-mse-target` (default `0.01`). This is an explicit post-attack gate.
If it fails, the run records `baseline_failed` and launches no subset attacks.
The runner never silently changes the target or claims the baseline passed.

Attack optimization and image selection use only attacker-visible objectives.
The existing `no_improvement` policy watches attack loss, with patience 200 and
minimum change 1e-10 by default. `--max-iterations` optionally caps each restart.
After the attack ends, the selected image's MSE decides whether subsets run;
there is no extra gate based on which stopping policy fired. Ground-truth MSE
never terminates an optimizer or selects its returned image.

`--presets idlg dlg geiping` runs the same workflow separately for each attack.
Each has its own baseline and subset jobs. `--attack-profile calibrated` is the
default: bounded sigmoid-coordinate LBFGS for iDLG/DLG with four restarts and
smooth seeded initialization; Geiping uses uniform initialization, its existing clamp,
cosine/TV/sign configuration, eight restarts, and learning rate .01. The original
registered settings remain available with `--attack-profile preset`.
`--lr`, `--restarts`, and stopping parameters can be set explicitly. All settings
are saved; the same configuration and seeds apply to baseline and subsets.

The calibrated iDLG/DLG initialization starts from a random 4x4 grid per channel,
resized smoothly to the image dimensions before creating full-resolution sigmoid
coordinates. The optimizer still controls every pixel. This avoids starting with
pixel noise that weakly observed gradient directions may preserve; no ground-truth
or disguise pixels are used in initialization. The previous pixel-noise start is
available with `--initialization uniform_sigmoid`. The original registered presets
remain unchanged. Initialization choice is recorded in each job's attack config.

On the locally reproduced seed-2026 CIFAR-10 image-564 case, changing only the
initialization reduced MSE from 0.04188 to 0.00289 at the same diagnostic budget.
The standard four-restart run reached MSE 0.00274 and SSIM 0.944. This validates
this seeded baseline case; other model/sample combinations retain the MSE gate.

The [completed reference run](reference_runs/lenet5_cifar10_seed2026.json) records
all six subset reconstructions and their fixed-disguise/ground-truth distances.
Only one of the six was closer to its marked disguise by MSE; this run therefore
does not show consistent attraction to the marked disguise.

## Per-subset measurements

For a subset observation H, the preassigned disguise is:

```text
j = argmin_j squared_L2(H, public_gradient[j])
```

Its identity is frozen before reconstruction. Even if the reconstructed image
looks closer to another public image, evaluation still uses the marked disguise.
This is not a post-reconstruction search over the image pool.

The report includes:

| Column | Meaning |
| --- | --- |
| `model`, `dataset`, `seed`, `private_index` | Selected experimental inputs |
| `coalition` / `subset` | Server bit mask; `full` is the baseline |
| `disguise_index`, `disguise_slot`, `disguise_label` | Fixed public training image and its pool slot |
| `gradient_distance_to_disguise` | Squared L2 from subset sum to its marked disguise gradient |
| `gradient_distance_to_ground_truth` | Squared L2 from subset sum to the true gradient |
| `ground_truth_mse` | Reconstructed image versus private image |
| `disguise_mse` | Reconstructed image versus marked disguise image |
| `disguise_mse_advantage` | Ground-truth MSE minus disguise MSE; positive means closer to the disguise |
| `ground_truth_ssim`, `disguise_ssim` | Corresponding image similarities |
| `attack_loss`, `gradient_l2_squared` | Attacker objective and separately evaluated gradient residual |
| `iterations`, `restarts`, `stopping_reason` | Reconstruction effort and stopping |

Image MSE is mean squared raw pixel distance; gradient distances are squared
Euclidean sums. Both images have the same shape, so their MSEs are directly
comparable. DLG gradient residuals use its selected soft labels; a separate
true-label residual remains an evaluation-only diagnostic. The baseline has no
assigned disguise, so its disguise columns are empty.

## Outputs and resume

- `manifest.json`, `workflow_manifest_v3.json`: model/dataset/sample identities.
- `model_state_v3.pt`: fixed model state, including seeded random initialization.
- `decomposition.pt`, `decomposition_hash.json`: exact and float32 shares, original
  and public gradients, coordinate ownership, repair coefficients and metadata.
- `input_checks_v3.json`: aggregation and all coalition geometry checks.
- `subsets_v3.json`: every subset and its fixed disguise assignment, saved before
  attacks. For three servers: 001, 010, 011, 100, 101, 110.
- `ground_truth.png`, `disguises.png`: selected inputs.
- Per configuration: `baseline.json` with gate result and completed-subset count.
- Per reconstruction: config, identity hashes, objective/optimizer trajectory,
  selected tensor, summary, and comparison PNG. Subset comparison images show
  **ground truth | reconstruction | marked disguise**.
- `report.csv` and `report.json`: baseline first, then subset rows per
  configuration. `workflow_status_v3.json` records overall completion.

Repeat the command to resume. Completed jobs are reused; model state, shares,
assignments and job input identities are checked. Different model/data/sample
settings require a new output directory. Execution-code changes create fresh
attack configurations while retaining the decomposition. Old result folders
remain intact; historical report rows lack the new assigned-disguise fields.

Existing LeNet5/CIFAR10 output directories retain their selected image (for
example 193) and public pool. Supply the original checkpoint the first time you
reuse a historical directory; the model state is then saved for subsequent runs.
The original manifest and shares are not overwritten.

Optional stage commands:

- `--prepare-only`: prepare shares and mark every subset's disguise; no attacks.
- `--baseline-only`: run the full-share baseline and stop.
- `--diagnose-only`: inspect existing decomposition/reconstruction artifacts.

## Checks

```bash
python -m unittest discover -s tests -p 'test_coalition*.py' -v
python -m unittest discover -s tests -v
```

The workflow tests execute one successful baseline followed by all six subsets
on a small identifiable fixture, enforce baseline failure gating, verify that a
gradient-selected disguise is not replaced by an image-selected one, check seeded
sampling/model creation and full resume, and verify model adaptation for all
three model choices with both dataset shapes. Geometry and reconstruction tests
also cover aggregation, rounding, margins, sparse repair, optimizer diagnostics,
label consistency and preservation of old artifacts.

The sparse solver's guarantee concerns relative gradient distances. The image
comparisons above are the empirical test of the proposed disguise behavior.
