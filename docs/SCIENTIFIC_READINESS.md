# SVD, reference and ResNet threshold diagnosis

Reviewed against commit `2542703` and the reported E2EGI/min-SVD/ResNet18 failure.
The SVD-based comparisons are not yet validated for final extremal-direction
claims on ResNet. Passing execution tests does not establish that those solves
converge on the intended experimental cases.

## The user's exact failure is reproducible

The reproduction matched the model SHA-256
`7c33f3aef9de71cda2f28bdfb2dfbaad9af085fc0e6f77916fd5259c96410180`,
experiment seed 0 and CIFAR-10 test image 5872.

Five inverse updates, each limited to eight CG iterations, produced:

| Inner solve | Actual relative linear residual | Requested tolerance |
| --- | ---: | ---: |
| 1 | 1.12418 | 0.000001 |
| 2 | 1.02376 | 0.000001 |
| 3 | 1.24454 | 0.000001 |
| 4 | 1.82261 | 0.000001 |
| 5 | 1.52173 | 0.000001 |

The final singular residual was 0.0615891 versus tolerance 0.0001, reproducing
the reported 0.0615892 to float32 rounding. The candidate sigma 0.108572 must
not be interpreted as an established minimum singular value. The rejection is
substantial nonconvergence, not a threshold-rounding issue or a demonstrated
nullspace. CG minimizes an energy norm, so Euclidean residuals greater than the
initial residual do not alone establish a CG implementation bug; eight steps
are nevertheless plainly inadequate for these requested solves.

This happens before the attack starts. `min_svd_noise` uses fixed noise scaling,
not lower-bound calibration, so its failure cannot be caused by the lower bound.

## Why one maximum solve affects several references

The reference scaler first constructs a maximum-SVD direction, then calls the
residual-aware calibration, then copies the reference's noise norm to the tested
noise. A construction error therefore prevents that reference budget from
existing. Gaussian/Laplace/uniform/GradDefense reference variants share this
dependency. Their errors are correlated failures of a common numerical control,
not independent evidence that each defense or the scalar bound failed.

The corrected scalar calibration uses a VJP and needs no maximum SVD. Its
formula is a=(epsilon_abs+kappa*r+mu_J*r^2/2)/(1-rho). A finite calibration can
still be conditionally inapplicable to an attack or exceed a declared noise
budget. Those statuses must remain distinct from an upstream singular solve.

For maximum-SVD, power iteration can require many steps when leading singular
values cluster. Earlier validation reproduced a 500-step failure that needed
704 steps under a larger explicit cap. This does not identify the exact cause
of an unprovided current maximum error; its diagnostics must be inspected.

## Repeated solves and masking change the experiment

Ranking and noise construction currently solve separately with successive
random starting vectors. With clustered singular values, both can pass the
residual check while returning different directions. A controlled diagonal map
diag(1,0.99999,0.5) gave:

| Quantity | Ranking solve | Construction solve |
| --- | --- | --- |
| Absolute direction | (0.18359,0.98300,0.00000226) | (0.55037,0.83492,0.00000084) |
| Relative singular residual | 0.00000495 | 0.00000928 |

Thus selection can use one valid approximate mode while construction uses
another. Clustered eigenspaces inherently have unstable individual bases;
repeatability and subspace stability need explicit treatment. A small singular
residual does not by itself prove global extremality or a unique direction.

Alignment functions also solve the full Jacobian, truncate its left singular
vector to the selected coordinates and renormalize. This is a masked full-SVD
direction, not generally an extremum for the restricted observation operator.
For A=[[1,0],[0,2],[1,1]] and selected rows 0 and 2:

- Full minimum coupled singular value: 1.30278.
- Gain ||A^T*d|| after truncation/normalization: 1.50513.
- Actual minimum gain within those two selected gradient coordinates: 0.618034.

The comparison needs an explicit definition: full coupled singular modes,
masked full modes, restricted-operator modes, or minimum gain in gradient space.
For a tall Jacobian, gradient-space left-null directions also exist; these are
different from the smallest coupled input singular mode currently estimated.

## Why ResNet thresholds look unusual

For a 3x32x32 input, MSE target 0.02 means r=7.83837. With the preset mu_J=0.1,
the curvature contribution mu_J*r^2/2 is already 3.072 gradient-L2 units, before
adding directional gain or the residual margin. That constant dominates weak
directions. It is an assumed quantity in model-dependent units, not a universal
ResNet constant. ReLU gate crossings and local-probe limitations remain as
documented in the [proof audit](LOWER_BOUND_AUDIT.md).

Noise budgets are also currently different across presets:

- Gaussian/uniform/Laplace defaults use a global relative L2 budget of 0.01.
- Minimum/maximum-SVD noise defaults use absolute L2 0.01 after normalizing the
  selected direction.
- Bound/reference variants use model- and direction-dependent calibration.

For the matched ResNet case, ||g0||=40.0360. Thus relative 0.01 means noise L2
0.40036, about 40 times the SVD preset's 0.01. A nominal scale of 0.01 therefore
does not mean equal perturbation across these methods. Masks also differ.

E2EGI uses a cosine objective with regularizers. Convergence of that objective
does not establish the raw-gradient residual condition used by the conditional
bound. Measured MSE and measured raw residual must be reported separately.

## Changes needed for reliable scientific comparisons

1. **Repair and validate the singular solver path.** Use a rank-aware
   matrix-free Krylov/SVD implementation with bounded resources, checked inner
   solves where applicable, and final triplet residuals. Test against explicit
   small SVDs, clustered spectra, ill-conditioned maps and nullspaces. Validate
   on the exact case above and independent cases. Increasing a cap is an
   experiment, not a guarantee; loosening the accuracy check is not a repair.

2. **Define which operator and direction the experiment means.** Either label
   current modes as masked full-SVD modes, or solve the operator corresponding
   to the selected observations. Report actual directional gain after masking.
   Minimum-positive and numerical-null experiments must remain distinct.

3. **Reuse verified numerical controls.** Rank and construct from one checked
   triplet. Give numerical solvers random streams independent of noise draws.
   Precompute one reference artifact per model state, private sample, operator/
   mask, settings and seed, and reuse it across attacks. If it fails, mark that
   shared reference case unavailable; never silently substitute another method.

4. **Make fixed-budget empirical comparisons the primary experiment.** Sweep
   the same global relative noise L2 across directions/distributions, with paired
   victim weights, samples, seeds, masks where appropriate, and attack budgets.
   Use `relative_gradient_scale` with `mode: global` for matched additive-noise
   controls. Keep conditional mu_J calibrations as a separate sensitivity study,
   with assumptions and actual fit-condition coverage visible.

5. **Report numerical coverage independently from privacy outcomes.** Give
   requested cases, converged transforms, numerical errors, stalled attacks and
   MSE successes. Do not count a solver error as a failed attack. Report which
   matched cases support each comparison; silently dropping difficult cases can
   introduce selection bias. Use independent case replicates and paired
   uncertainty estimates rather than treating attack restarts as independent
   victim samples.

If a full extremal solve is too expensive, an explicitly named approximate
fixed-budget direction or declared subspace method can still support a valid
empirical experiment. Its measured gain and computational budget should be
reported, without silently calling it a converged full minimum/maximum SVD.

The existing Gaussian/uniform/Laplace relative-budget presets provide controls
that do not depend on an SVD or reference threshold. They can establish the
baseline while the SVD path is validated. They do not replace the missing
minimum/maximum comparisons.

Local reproduction:
`/workspace/prepush-tests/scientific-readiness/reproduce_user_min.py` and
`user_min_results.json`. This review does not claim that the remaining numerical
solver changes have already been implemented or validated.
