# Residual-aware conditional bound

The default bound is now `directional_residual_v1`. It accounts for nonzero
gradient-matching residual, keeps the useful dependence on noise direction, and
does not require a maximum-SVD solve inside calibration. The victim models,
MSE success condition and numerical SVD accuracy checks are unchanged.

The prior formula remains available as `legacy_proof_v1` for explicit historical
comparisons. See [the proof audit](LOWER_BOUND_AUDIT.md) for its assumptions.

## Derivation

Let g(x) be the same fixed-model, fixed-label parameter-gradient map used for
the private batch. Put A=Dg(x0), eta=a*d with ||d||=1, and

\[
\epsilon=\|g(\hat x)-(g(x_0)+\eta)\|_2,\qquad
\kappa=\|A^T d\|_2,\qquad E=\|\hat x-x_0\|_2.
\]

If Dg is mu_J-Lipschitz along the segment to the reconstruction, Taylor's
remainder has norm at most mu_J*E^2/2. Projecting
eta=A*(xhat-x0)+remainder-residual onto d gives

\[
a-\epsilon\leq\kappa E+\tfrac12\mu_J E^2.
\]

This needs neither an exact fit nor a stationary/global-optimal reconstruction.
For b=max(0,a-epsilon), the stable positive inverse is

\[
B=\frac{2b}{\kappa+\sqrt{\kappa^2+2\mu_J b}},\qquad
\operatorname{MSE}(\hat x,x_0)\geq B^2/N.
\]

The implementation handles mu_J=0, kappa=0 and uninformative residuals separately.
If both constants are zero but b is positive, the residual condition is
incompatible with those assumptions; it does not report an infinite privacy
guarantee. Binary scaling avoids intermediate overflow in the scalar root.

For an affine map diag(10,1) and noise norm 2, maximum-direction noise gives
B=0.2 and minimum-direction noise gives B=2, both exact. The previous inequality
gave a much weaker minimum-direction value. No sign reversal is needed.

## Calibration and measured evaluation

Before an attack, its eventual residual is unknown. Calibration explicitly
declares the class of reconstructions it covers:

\[
\epsilon\leq\epsilon_{\rm abs}+\rho a,\quad 0\leq\rho<1.
\]

For target radius r=sqrt(N*target_MSE), sufficient amplitude is

\[
a_{\rm required}=\frac{\epsilon_{\rm abs}+\kappa r+
\tfrac12\mu_Jr^2}{1-\rho}.
\]

The generated defaults explicitly store `residual_l2: 0.0` and
`residual_fraction: 0.1`: the calibration covers raw gradient residuals at most
10% of the applied noise L2. This is a declared condition, not a promise that an
attack will achieve that fit. It is unrelated to the pixel MSE success threshold.
Use fraction 0 for an explicit exact-fit condition; fraction 1 is vacuous and
rejected. Infeasible perturbation budgets return bounded best effort.

After each completed attack, evaluation computes the **actual raw gradient L2
residual** for the practical and oracle saved images. It uses true labels and
batch alignment only for evaluation. These values are never passed to the attack.
The evaluation recomputes kappa using the total actual released perturbation,
so matching a maximum reference's noise norm does not transfer its directional
bound to the tested noise.

Results are stored in the final summary under:

```text
experiment_metadata.reconstruction_bounds.practical
experiment_metadata.reconstruction_bounds.oracle
```

Fields include `gradient_residual_l2`, `calibration_residual_condition_met`,
`conditional_lower_bound_mse` and `observed_mse`. A stalled reconstruction with
residual at least the noise norm correctly has an uninformative bound of zero.
If the computed candidate exceeds observed error beyond rounding tolerance,
status becomes `assumptions_contradicted` and the reported conditional bound is
null. Its candidate and an observed minimum-curvature estimate remain as
diagnostics. This detects contradictions; passing the check does not validate
the assumptions elsewhere.

## ResNet interpretation

These remain **conditional, uncertified** quantities for the current ReLU
ResNets. Manual mu_J=0.1 is a declared assumption, not a validated constant.
Empirical probes are local directional observations, not certified uniform upper
bounds. ReLU gate crossings can invalidate the differentiability/smoothness
conditions entirely. `certified` therefore remains false, and nonsmooth module
types are recorded. Zero observed variation is allowed for affine maps but is
not interpreted as proof of zero variation over an entire region.

Maximum/minimum-SVD constructions still require their own converged directions.
This change removes the additional maximum solve formerly inside the bound; it
does not repair every singular solve or relax its accuracy check. Conditional
calibration also must precede other gradient modifications; unsupported changed
baselines are rejected when the original gradients are available.

## Existing schedules and results

Reading/building old YAML resolves the new method and residual defaults in
memory before hashing. Stored descriptions include these settings. Only bound
and reference configurations change identity; completed unrelated experiments
continue to skip. Historical rows and schedule files are preserved. Corrected
bound configurations run as distinct experiments, rather than reusing old
results. Explicit legacy settings remain possible.

In a scaler's YAML `parameters` mapping, the important settings are:

```yaml
bound_method: directional_residual_v1
target_error: 0.02
mu_j_method: manual
mu_j: 0.1  # assumed, not certified for ResNet
residual_l2: 0.0
residual_fraction: 0.1
```

`reference_max_iterations` still controls reference-direction construction.
`svd_iterations` and `svd_max_iterations` are used only by the legacy bound;
the corrected bound does not estimate sigma_max. Unsupported forwarded settings
and invalid residual conditions are rejected before expensive execution.

## Validation

- **121 regression tests passed**, including analytic quadratic equality with
  nonzero residual, exact affine minimum/maximum directions, known-constant
  vector quadratic maps, degenerate/overflow cases, baseline checks, database
  identity migration, actual saved-residual persistence and batch continuation.
- A fresh **35-run actual-CIFAR/LeNet matrix** recorded 31 normal failures and
  four minimum-SVD numerical errors. All subsequent runs, including a healthy
  final sentinel, executed; no unexpected application error occurred. The tiny
  budgets were two outer iterations, one restart and LBFGS max_iter=1.
- The actual registered **ResNet18 on cached CIFAR-10** completed direct Gaussian
  directional calibration, one DLG step and measured-bound evaluation in about
  5.7 seconds on this CPU instance, with no SVD call. Applied noise L2 was 3.48354,
  the declared residual budget 0.348354, and the measured attack residual 55.4040.
  The attack was a normal reconstruction failure and the evaluated conditional
  bound was correctly zero/uninformative. This is an execution check, not an
  attack-success or privacy-guarantee result.
- The original experiment database's SHA-256 stayed unchanged. Dependencies and
  whitespace checks passed. Full-budget sweeps and GPU execution were not run;
  minimum/maximum-SVD constructions retain their existing convergence limits.

Local evidence: `/tmp/residual-bound-final.log`,
`/workspace/prepush-tests/recheck/residual_bound_matrix/`, and
`/workspace/prepush-tests/recheck/residual_bound_resnet/results.json`.
