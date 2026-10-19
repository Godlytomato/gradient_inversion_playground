# SVD, reference and ResNet threshold diagnosis

The minimum-SVD diagnosis below was reproduced against commit `2542703`.
The subsequently supplied GradDefense-reference maximum-SVD failure was also
reproduced exactly and its precision/stopping defect has now been repaired.
The minimum solver has since been replaced in new presets by a bounded checked
Krylov estimator; the [minimum-solver report](MINIMUM_SVD_SOLVER.md) documents its
exact-case validation and explicit migration of old inverse-iteration schedules.
The later CIFAR LeNet report exposed a structural distinction: this architecture
has rank(D_x g) <=1992 for 3072 input coordinates, so its full minimum is exactly
zero. The current coupled-left construction is unavailable there; a structural
check now records that reason before solving. This is a mathematical obstruction
separate from the resolved ResNet underconvergence.
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
The new `lanczos_v1` method passes this exact case at step 400 with sigma
0.03287184 and residual 0.0000882293, keeping tolerance 0.0001. The historical
five/eight settings are retained only when explicitly selecting the old method;
they are not silently reinterpreted. This repairs the demonstrated default
underconvergence, without establishing a certified global minimum.

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

## Maximum-SVD precision defect: reproduced and repaired

The supplied GradDefense-reference failure used the same model fingerprint,
image and experiment seed as above. Its maximum solver stopped at iteration
29 of the 500-step budget, with sigma 2.9320445 and singular residual 0.00079207
against tolerance 0.0001. This was not an exhausted iteration budget.

The JVP has 11,173,962 gradient coordinates. Accumulating its norm in float32
underestimated it: the accurate norm of the same float32 vector is 2.9332040.
Using that norm reduces the same iterate's singular residual to 0.000039807.
The old normal-equations stopping proxy was 0.000039821. It passed while the
biased normalization made final validation fail. A fixed safety margin on the
proxy cannot repair that precision loss.

The corrected solver accumulates norms in float64 while retaining the model and
derivative products in their original dtype. It normalizes between JVP/VJP
products and stops on the actual singular residual used by final validation.
If that residual fails, refinement continues from the current iterate within
the declared cap. The tolerance remains 0.0001. Minimum-SVD norms, masked
alignment normalization and reported perturbation norms use accurate
accumulation too.

On the exact supplied maximum case, the corrected default solver converges at
iteration 27 with sigma 2.9332039 and residual 0.000082112. A compact regression
with 65,536 gradient coordinates reproduces the old false failure: its old
proxy is zero but biased normalization yields residual 0.000422. Accurate
normalization gives 0.000000223. At an explicitly stricter tolerance 0.00000001,
the solver correctly rejects after consuming the full test cap, rather than
stopping early on the zero proxy. Tests also check minimum-SVD normalization,
masked max/min directions and reported perturbation norms.

The complete default GradDefense-reference transformation was then run on the
same actual CIFAR-10/ResNet18 case, including all 16 sensitivity probes per layer,
slice selection, Gaussian construction, the maximum solve, conditional
calibration and final noise application. It completed in 82.4 seconds on one CPU
thread. The masked reference gain was 2.15458; reference L2 was 22.1782165868 and
actual applied L2 was 22.1782165862. Calibration recorded MSE 0.01999999987 under
its supplied assumptions and `matched_reference_norm` for the test direction.
This validates construction/calibration, not an actual attack outcome or a
guaranteed MSE for the Gaussian direction. All 126 unit/regression tests passed;
the experiment database checksum remained unchanged. Full-case artifacts are
`verify_user_reference.py` and `user_reference_fixed.json` in the local artifact
directory identified below. CUDA execution was not tested.

This is an implementation precision defect upstream of the bound. It does not
invalidate the conditional scalar formula. The earlier minimum-SVD failure is
different: its residual near 0.0616 and unconverged inner solves are not repaired
by accurate normalization alone. Power iteration can also genuinely need more
than 500 steps when leading singular values cluster; earlier validation needed
704 steps in a separate case. A small triplet residual remains an estimate,
not a certificate of global extremality or an established spectral upper bound.

## Repeated solves and masking change the experiment

Maximum-SVD ranking and noise construction still solve separately with successive
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

New minimum-SVD presets now use an independent solver stream and reuse one
checked triplet for matching settings/operator within a cacheable evaluation
context. That fixes ranking/construction drift on this minimum path. Shared
controls across separate attack contexts and the maximum path still need the
broader reuse policy below.

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

1. **Finish validating the singular solver path.** The demonstrated maximum
   precision/early-stop defect is repaired, and the new bounded minimum Krylov
   estimator passes the supplied case and checks final triplet residuals without
   inner inverse solves. Rank-aware large-scale minimum-positive/null-space
   certification remains a separate capability. Controlled regressions cover
   small SVDs, clustered spectra, ill-conditioned maps and nullspaces. Extend
   real-model validation beyond the supplied case with independent cases. Increasing a cap is an
   experiment, not a guarantee; loosening the accuracy check is not a repair.

2. **Define which operator and direction the experiment means.** Either label
   current modes as masked full-SVD modes, or solve the operator corresponding
   to the selected observations. Report actual directional gain after masking.
   Minimum-positive and numerical-null experiments must remain distinct.

3. **Reuse verified numerical controls.** The new minimum path already reuses
   one checked triplet per matching evaluation context/settings and has an
   independent solver stream. Extend that policy to maximum solves and shared
   controls across attacks.
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
`user_min_results.json`. Maximum-case before/after evidence is in
`reproduce_user_max.py`, `user_max_baseline.json` and `user_max_fixed.json` in the
same local artifact directory. The compact precision and cap regressions are
committed in `tests/test_svd_stopping.py`. Minimum-solver tests and migration are
described in [MINIMUM_SVD_SOLVER.md](MINIMUM_SVD_SOLVER.md). The comparison-
protocol changes listed above remain outstanding.
