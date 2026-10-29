# Historical review of the v2 baseline correction

This describes the earlier diagnostic workflow. The current [experiment guide](README.md)
uses one full-share baseline followed by every nonempty proper subset and
a fixed gradient-nearest disguise for each subset.

## Goal and present status

The intended single-image experiment must establish both exact all-share
recovery and whether unauthorized subsets steer reconstructed images toward
public dummy images. Its positive control must first reconstruct the private
image reliably from the actual all-share sum.

The user's trained-checkpoint run establishes the first requirement numerically.
It has not established the second. Baseline inversion failed, and the gate
correctly withheld all subset attacks. No privacy conclusion follows yet.

## Evidence supplied by the user

For image 193 and three servers:

| Check | Observed result |
| --- | --- |
| All six proper coalition inequalities | Passed |
| Extra stored nonzeros | 1 |
| Shared coordinate | 453 |
| Float64 aggregation error | 0 |
| Float32 relative aggregation error | 7.715e-9 |
| True-image/full-gradient squared residual | 5.684e-14 |
| iDLG selected attack objective | 5.724951916e-5 |
| Independently recomputed squared gradient residual | 5.724951954e-5 |
| Private-image MSE | 0.6326395 |
| Private-image SSIM | 0.1409496 |
| Pixel range | [-2.408885, 3.022656] |
| Fraction outside [0,1] | 0.5406901 |
| MSE after post-hoc clamping | 0.1778034 |
| Stopping | no_improvement after 604 outer steps |
| Last ten objective values | Identical |

The true image remains a near-exact feasible solution. The agreement between
reported and independently recomputed losses argues against a mismatched target
or loss report. The recovered image is invalid for the raw-pixel domain, and
post-hoc clamping still misses the 0.01 target substantially. This locates the
observed failure at the full-gradient inversion/calibration stage.

These facts do not prove that floating-point aggregation never affects a
nonconvex optimizer, nor do they prove exactly which LBFGS internal stopping
condition fired. Both possibilities require additional diagnostics or controls.

## What worked

- Sparse shares preserved aggregation while passing every coalition inequality.
- Gradient parameter ordering, model, input coordinates and cross-entropy were
  consistently used in generation and attack dispatch in the inspected code.
- Attacks selected reconstructions by attacker objective, not private-image MSE.
- The full-share baseline ran before coalition attacks.
- The calibration gate prevented an uncalibrated baseline from being reported as
  successful defense evidence.

## What went wrong in the initial implementation

1. **Baseline configuration was insufficient.** The runner reused the faithful
   iDLG preset (unbounded Gaussian pixels, one restart) without a separate bounded
   calibration profile. More than half of the selected pixels left the valid
   domain. Simply clamping an output or running longer is not a demonstrated fix.
2. **Stagnation was underdiagnosed.** Only scalar objective trajectories were
   recorded. Default LBFGS gradient/change tolerances and a 1e-7 observer improvement
   threshold were not separated in the diagnostic output. Exact flat loss cannot
   distinguish useful convergence from optimizer steps that do nothing.
3. **The control comparison was incomplete.** No same-seed attack on the raw
   original gradient accompanied the actual-share-sum control. Numerical
   aggregation sensitivity could not be measured directly.
4. **Updates could block share reuse.** The decomposition manifest compared all
   experiment source hashes, so changing reconstruction code would reject the
   user's otherwise valid saved shares.
5. **Reporting/resume had gaps.** Reports only retained the current invocation;
   completion was not committed until evaluation finished; resume checked the
   reconstruction hash but not explicit model/observation identities.
6. **DLG residual labels were ambiguous.** Its independent residual used the true
   label rather than the selected optimized soft labels, without a separate
   actual-label residual. This did not explain the iDLG failure but needed correction.

Earlier guidance to try another learning rate was a calibration suggestion, not
an established diagnosis. A synthetic launch test and passing tests alone do not
validate the user's trained-model reconstruction. The historical disappearance of
the user's results directory is unexplained; inspected runner code does not
remove output directories.

## Corrections implemented

- Add an explicit calibrated profile. iDLG/DLG optimize sigmoid image coordinates
  from uniform raw pixels, with four restarts and tighter LBFGS tolerances.
  Every trial image is bounded; no post-step projection corrupts LBFGS history.
  Existing registered presets remain selectable and unchanged.
- Keep the registered optimizer/loss/observer engine. Add only opt-in initializer
  support for parameterized images and an optional diagnostics callback.
- Record pixel domain, update size, selected labels, image/variable derivatives,
  actual optimizer defaults, versions, code hashes and restart stopping reasons.
- Run paired actual-share-sum and undecomposed-gradient controls using identical
  configurations/seeds. Both must pass calibration before coalition attacks.
- Add a non-optimizing diagnosis command for old saved artifacts.
- Decouple immutable decomposition input identity from execution source identity.
  Verify the tensor hash, current gradient, float64/float32 sums and all margins
  before reuse. Preserve old shares, manifest and prior run folders.
- Preserve all evaluated runs in the aggregate report; commit reconstruction
  completion before evaluation; reject changed job inputs on resume.
- Evaluate DLG gradient mismatch with its selected soft labels; report true-label
  mismatch separately. Keep image metrics evaluation-only.

## Validation and remaining work

All **207 tests passed**, including **20 coalition experiment tests**. An
identifiable linear fixture recovered its image with MSE below 1e-8 using the
bounded registered iDLG path; all LBFGS trial inputs stayed in [0,1]. Paired exact
and cancelling-share observations agreed to the test tolerance. Existing attack
regression tests passed with the optional extension disabled.

A real-CIFAR engineering smoke reused a pre-update decomposition and ran both
controls with an explicitly **untrained** checkpoint and a three-iteration cap.
Full/original MSEs were approximately 0.048716 / 0.048714; calibration failed and
coalitions were skipped as required. Decomposition, original manifest, artifact
hash file and historical reconstruction files remained byte-identical. Old and
new report rows survived. Diagnosis and completed-job reuse were also checked.
This smoke checks mechanics only; it is not a trained-model privacy result.

The user's trained checkpoint and selected reconstruction tensors are absent
from the execution workspace. They must be run with the corrected code to see
whether image 193 reaches MSE < 0.01. The new profile is a correction to experimental
setup and observability, not a guarantee of inversion success. If both paired
controls still fail, investigate conditioning, parameterization saturation and
additional optimizer configurations before proceeding to coalition claims.
Adaptive attacks remain separate future work.


## Seeded random LeNet, CIFAR-10 sample 564: initialization correction

The later workflow's seed-2026 run uses seeded random weights and test image 564,
which is a different case from the trained-checkpoint/image-193 results above.
The user reported MSE 0.03958 and objective 4.517e-8 with a flat tail. Local
reproduction showed MSE 0.04188, objective 5.406e-8, and zero pixel movement.

Controlled probes held the local full-share observation, model, loss, seed, and
150-outer-step budget fixed while changing initialization:

| Initialization | Image MSE | Squared gradient residual |
| --- | ---: | ---: |
| Independent random pixels, bounded sigmoid coordinates | 0.041876 | 5.406e-8 |
| Random 4x4 grid resized to a smooth image, bounded sigmoid coordinates | 0.002886 | 3.018e-8 |

Separate loss-rescaling probes improved gradient matching substantially while
leaving pixel MSE near 0.039. That is insufficient evidence to attribute the
failure solely to LBFGS tolerances. The initialization comparison demonstrates
that injected pixel noise was a consequential experimental choice in this case.
It does not prove the precise internal stopping condition or unique inversion.

The calibrated iDLG/DLG profile now uses the smooth seeded initializer. Only the
starting image is smoothed; all pixels remain trainable. Original presets and the
previous initializer remain explicitly selectable. The method uses no private or
public reference image content, and applies identically to baseline and subsets.
No loss scaling, decomposition change, MSE-target relaxation or oracle selection
was added to the default.

The standard four-restart full-share control reached MSE 0.00274067 and SSIM
0.943994 with the unchanged MSE target 0.01, allowing the subset stage to start.
This evidence concerns this seeded random-model case; trained checkpoint results
and other samples must pass their own baseline gates.

All six subset attacks subsequently completed. Their recorded image MSEs were
independently recomputed against ground truth and the preassigned disguise, and
the saved decomposition hash remained unchanged. Only one of six had lower MSE
to its assigned disguise than to ground truth. The complete machine-readable
[reference report](reference_runs/lenet5_cifar10_seed2026.json) preserves these
results; the initialization fix establishes a usable baseline for this case,
not general evidence that the decomposition attracts reconstructions to disguises.

## Residual simplicity correction: fewer nonzero coordinates

The clarified requirement concerns the residual vectors themselves: for every
proper subset H and its gradient-nearest disguise D, H-D must be shorter and have
no more nonzero entries than H-G. New residual-mode runs default to strictly
fewer entries; a configurable gap of zero permits equality. Share nonzeros are
a different quantity from these residual counts.

The old saved seed-2026 sparse construction passed the distance inequality but
failed residual simplicity. For subset 001, the disguise residual had length
22.1205 versus 22.1359 toward truth, yet contained 83,126 nonzeros versus 55,381.
The same reversal of residual sparsity occurred in all six proper subsets.

The revised construction projects initializer shares into the reference-gradient
span, then fits selected disguise coordinates with cancelling adjustments
orthogonal to that span. Projection can make the true-gradient residual denser;
coordinate fitting reduces the disguise residual's nonzeros. Both absolute counts
are reported. Exact aggregation and all numerical constraints are rechecked on
the actual float32 subset observations; no near-zero threshold inflates the
number of exact matches. The search ranks absolute distances first and retains
only fully audited candidates. Global optimality is not guaranteed, and exhausted
search does not prove infeasibility.

A completed experiment used the same saved model identity, CIFAR-10 image 564,
100 public images, and identical iDLG/initialization/stopping settings as the
previous reference. It requested a residual nonzero gap of 1,000 and retained the
relative margin and baseline MSE target unchanged.

| Quantity | Previous sparse construction | Residual construction |
| --- | ---: | ---: |
| Worst squared distance to nearest disguise | 1,943.6391 | 32.2670 |
| Full-share reconstruction MSE | 0.00274067 | 0.00291235 |
| Full-share baseline passed at target 0.01 | Yes | Yes |
| Proper subset reconstructions completed | 6 | 6 |
| Reconstructions closer to fixed disguise by image MSE | 1 / 6 | 3 / 6 |

Every revised subset passed both gradient conditions:

| Subset | Disguise residual nonzeros | True residual nonzeros | True-image MSE | Fixed-disguise MSE |
| --- | ---: | ---: | ---: | ---: |
| 001 | 79,126 | 83,126 | 0.210452 | 0.138873 |
| 010 | 79,126 | 83,126 | 0.213349 | 0.221453 |
| 011 | 80,037 | 83,126 | 0.238861 | 0.254243 |
| 100 | 79,126 | 83,126 | 0.299427 | 0.328240 |
| 101 | 80,824 | 83,126 | 0.242996 | 0.211030 |
| 110 | 80,493 | 83,126 | 0.246329 | 0.210137 |

The minimum residual nonzero advantage was 2,302. The float64 complement-pair
lower bound on the best possible worst squared disguise distance was about
0.8091; this heuristic remains well above that bound. Passing the gradient
constraints has not established consistent image reconstruction toward disguises.

All 232 regression tests passed. Independent verification recomputed the true
gradient and all 100 public gradients, checked both aggregation precisions,
nearest-disguise identities, exact float32 residual counts, and image MSE/SSIM
from saved reconstructions. The original saved model, manifests, decomposition
and subset plan remained byte-identical after a legacy resume check. The
[machine-readable reference](reference_runs/lenet5_cifar10_seed2026_residual_gap1000.json)
records inputs, settings, geometry, code fingerprints, and all seven attack rows.
