# Progress and correction review

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
