# Lower-bound mathematics and ResNet audit

Reviewed October 18, 2026, against commit `fb1b676` and the subsequently supplied
`proofs.docx`. This review separates execution readiness from mathematical
validity. The earlier smoke tests verified batch continuation and numerical
checks, not a reconstruction-error theorem.

**The code matches the proof's final inequality, and that inequality is valid
under its smoothness and exact-gradient-match assumptions. Those assumptions
are not established for these ResNet/noise experiments.** The preset constants
and empirical probes do not supply rigorous bounds, and the minimum-SVD defaults
can fail with correctly calculated Jacobian products.

## Findings

| Finding | Evidence | Implication |
| --- | --- | --- |
| Exact-fit assumption is not established by calibration | Exact CE example outside that assumption: requested MSE 0.02, reported feasible, actual global optimum MSE 0.01 | This illustrates a conditional theorem applied outside its hypotheses; it does not refute the supplied proof. |
| Presets do not estimate mu_J | All seven bound/reference builders select manual mode and default to 0.1 | Their SVD failures cannot be caused by the estimator miscalculating mu_J. |
| Manual 0.1 is not a general ResNet bound | One controlled ResNet probe gives directional variation ratio 2.5177 in float64 | Default constants are unsupported for that sample/model. |
| Empirical probes can drastically underestimate mu_J | Smooth rank-one example: exact mu=1, default estimate=0.00165056 in 3072 dimensions | A fixed safety multiplier of 1.5 is not a reliable upper-bound construction. |
| ReLU violates needed smoothness on regions crossing gates | Explicit CE example has a gradient jump approaching 1 as input distance tends to 0 | A finite uniform Jacobian-Lipschitz constant is generally unavailable on such a region. |
| Minimum-SVD defaults are inadequate in a tested ResNet | Relative residual 0.0617629 versus 0.0001 | Solver rejection is genuine; it does not imply an incorrect Jacobian. |

## The supplied proof and its scope

The document explicitly assumes that the optimal reconstruction x_delta attains
zero gradient-matching loss: g(x_delta)=g0+delta. An optimal attacker is not
necessarily an exact-fit attacker; the minimum loss can be positive.

The proof also assumes differentiability and uniform Lipschitz bounds for both
the parameter-gradient map g and its Jacobian throughout the relevant region.
The following direct argument confirms its lower-bound direction without relying
on the cited I2F result. Use the code's Jacobian orientation A=partial g/partial x,
h=x_delta-x0, E=||h||, sigma=||A(x0)||, and eta=delta. Exact matching gives

\[
\eta=\int_0^1 A(x_0+th)h\,dt,\qquad
\|A_0^T\eta\|\leq\sigma^2E+\tfrac12\sigma\mu_J E^2.
\]

Adding nonnegative terms gives exactly the document's weaker inequality

\[
\|A_0^T\eta\|\leq
E\left[\sigma^2+\sigma\mu_J E+2\mu_J\|g_0+\eta\|\right].
\]

The polynomial on the right increases with E for sigma>0, so its positive root
is a conditional lower bound. The threshold search in the code inverts this
inequality correctly. The document's J*delta and the code's A^T*delta use opposite
Jacobian orientations; their singular values agree. This is not a transposition
bug.

Proof presentation needs several corrections:

- The triangle-norm equality in paragraph 61 must be an inequality; the
  Lipschitz substitution in paragraph 64 must also use <=.
- The estimate mu_L <= sigma+mu_J*E applies to a suitable local segment constant,
  not an arbitrary supplied global mu_L. Define the local constant explicitly.
- Positive mu_J does not imply sigma>0; handle a zero Jacobian separately.
- The derivative section must differentiate the bound's root B, not actual E.
  An inequality for E cannot be changed into an equality and differentiated as
  though it described the true reconstruction error.

Generic noise in millions of parameter-gradient coordinates need not be the
gradient of any possible input. For a smooth P-dimensional gradient map of an
N-dimensional image with P much larger than N, the attainable set has at most
N local dimensions. Even tangent SVD noise need not remain on this set at finite
amplitude, and masking changes its direction further. The implementation does
not establish attainability. A finite attack with nonzero residual cannot by
itself determine the global minimum residual either.

## What the implemented search solves

Let g(x) be the vector of parameter gradients, J its input Jacobian, g0=g(x0),
d a unit perturbation direction, a its nonnegative amplitude, and
r=sqrt(N*target_MSE). The implementation searches for

\[
F(a)=a\|J_0^T d\|-r\left[\sigma^2+\sigma\mu r+
2\mu\|g_0+ad\|\right]\geq0.
\]

See [the residual calculation](../transformations/transformation_implimentations/threasholds/lower_bound_threashold.py#L192).
The golden-section maximum search and first-root bisection consistently solve
this concave scalar problem. Independent analytic cases cover interior feasible
roots and infeasible interior maxima. The main issue is what this inequality
means, rather than floating-point instability in that scalar search.

The MSE-to-L2 conversion is correct: one 3x32x32 image with target MSE 0.02 has
r=7.838367. The estimator probes global input-L2 radii 0.001 and 0.01. These local
probes do not establish constants throughout a radius-7.84 neighborhood.

Even for the implemented equation, a feasible scale need not exist. For large a,
F's slope approaches ||J0^T d||-2*r*mu. Small minimum-singular gains can make the
criterion infeasible regardless of additional scale-search iterations. A
negative asymptotic slope alone does not exclude an interior feasible interval;
the existing search correctly checks for one.

## Example showing why the exact-fit assumption matters

Use input x in [0,1], original x0=0.5, and the smooth gradient map

\[
g(x)=10+0.1(x-0.5)+0.5(x-0.5)^2
    =9.995+0.5(x-0.4)^2.
\]

This is an actual cross-entropy parameter gradient: choose one scalar parameter
theta=0, logits [2*theta*g(x),0], and class label 1. At theta=0 the parameter
gradient is exactly g(x). The local singular value is exactly 0.1, and the
Jacobian's global Lipschitz constant is exactly 1.

The current function requests target MSE 0.02, chooses perturbation
eta=-7.396879308487042, and reports `feasible_under_supplied_constants`. The
observed gradient is y=2.6031206915129577. Since g(x) is always at least 9.995,
the unique global minimizer of squared gradient distance to y is x=0.4, inside
the allowed input domain. Its true MSE is (0.5-0.4)^2=0.01, below the requested
0.02. Computation uses float64, exact SVD with residual zero, and exact mu_J.
The minimum gradient-matching residual is about 7.391879: zero-residual matching
is impossible. Thus this is outside the proof's explicit assumptions.

Reproduce without touching experiment data:

```bash
python scripts/reproduce_bound_counterexample.py
```

The output field `reported_feasible_but_target_violated` is true in the reviewed
implementation, and `proof_exact_fit_assumption_satisfied` is false. This
demonstrates that a feasible numerical calibration does not establish the
theorem's applicability. The uploaded proof resolves the apparent contradiction.

For comparison, assuming a genuine Jacobian-Lipschitz constant throughout the
relevant neighborhood, an interior stationary point at distance rho of
0.5*||g(x)-(g0+eta)||^2 necessarily obeys

\[
\|J_0^T\eta\|\leq\rho\left[\sigma^2+
\tfrac32\sigma\mu\rho+\tfrac12\mu^2\rho^2+\mu\|\eta\|\right].
\]

This follows from ||g(x)-g0|| <= sigma*rho+0.5*mu*rho^2 and stationarity.
It illustrates a different result one could use when exact fitting is absent;
it is not needed to validate the exact-fit theorem and is not implemented here.
Its hypotheses also need checking for constrained solutions, labels,
regularizers and the different gradient-matching losses in this repository.

## mu_J: configuration and mathematical assumptions

The [reference preset](../validation/implementations/defenses/gaussian_noise_reference.py#L54)
and the six other bound/reference builders explicitly use `mu_j_method: manual`
with default `mu_j: 0.1`. Existing YAML can override these defaults.

For the repository's CIFAR-style ResNet18 in eval mode, seed 123, a controlled
random raw [0,1] input and one normalized probe at displacement L2 0.01, measured
||[J(x+delta)-J(x)]v||/||delta|| was 2.51638 in float32 and 2.51772 in float64
with the identical weights/input/probes. Thus 0.1 is smaller than even this
single-direction lower estimate of the required variation bound. Matching
float64 results rule out float32 cancellation as the explanation.

The [empirical estimator](../transformations/ranking_implimentations/jacobian_variation.py#L29)
samples both random displacement and random JVP directions. Neither maximizes
the operator norm or covers the whole neighborhood. A smooth 3072-dimensional
rank-one map with analytically exact mu_J=1 returns 0.00165056 using all default
probes, including the 1.5 multiplier. Its diagnostics correctly label the result
uncertified. That diagnostic cannot justify inserting the estimate as a proven
upper bound.

ReLU presents a stronger limitation: the parameter-gradient map itself can jump
when a gate changes. An explicit 2-input, 1-ReLU, 2-output cross-entropy network
gives a parameter-gradient difference approaching 1 for inputs separated by
2*epsilon as epsilon tends to zero. A region containing that gate is not a
continuously differentiable gradient map with a finite Lipschitz Jacobian.
Autograd computes the derivative of the active branch; repeated accurate JVPs
cannot repair the missing smoothness assumption.

This was also reproduced directly on the repository's ResNet18 in float64.
A first-ReLU boundary was only 1.456921e-5 input L2 from a controlled raw image.
On its two sides, both still in [0,1], the parameter-gradient difference stayed
about 0.002417928 while input separation decreased from 2e-5 to 2e-9. The ratio
grew from about 121 to 1.21 million. This supports a jump in g on the actual
architecture, violating the document's uniform Lipschitz-g assumption there.

An underestimated mu makes the current feasibility condition easier to satisfy.
It can create optimistic calibration, but does not explain an SVD convergence
exception. Raising an honestly justified mu can make a target infeasible.

## What is correct locally, and what failed numerically

On a repository CIFAR-style ResNet18, seed 34, eval mode, random raw [0,1] input,
and float64 derivatives:

- Two adjoint-identity checks had relative errors 2.11e-15 and 8.75e-16.
- Finite-difference checks at steps 1e-5 and 1e-6 matched JVPs to relative errors
  3.46e-9 and 3.48e-8, with no ReLU gate crossings.
- Disabling in-place ReLUs changed the JVP by zero; BatchNorm buffers stayed
  unchanged.
- Steps 1e-3 and 1e-4 crossed 11 and 2 ReLU decisions and no longer matched the
  single-branch derivative. This is evidence of nonsmoothness, not an adjoint bug.

These checks support the local Jacobian implementation in that configuration.
The models used controlled generated inputs, not the user's particular images.

The actual float32 default minimum solve failed at residual 0.0617629 after
about 125 seconds in this CPU environment. Its five inverse iterations and eight
inner CG steps work with J^T*J, squaring conditioning; inner solve residuals are
not validated before normalization. The final residual check correctly rejected
the output. A following maximum solve hit a shared wall-time limit; its numerical
convergence was not assessed. Neither result establishes every ResNet's rank.

The user's posted reference traceback occurs in maximum-SVD construction before
`lower_bound_threshold` is called. Maximum/minimum noise and pruning variants
can also fail without using a lower bound or mu_J at all.

## Minimum versus maximum directions

The document differentiates the positive bound root with respect to sigma_max,
holding ||J*delta|| fixed. Its negative derivative does not imply that choosing
minimum-singular noise improves the bound: with a fixed victim and original
input, sigma_max of the full Jacobian stays fixed, while choosing a minimum
direction reduces the numerator. The observed-gradient norm in the denominator
can change too.

For example, for the affine map A=diag(10,1) and perturbation norm a, a maximum
left-singular perturbation has true reconstruction displacement a/10 and the
mu=0 limit of this bound is exactly a/10. A minimum-direction perturbation has
true displacement a, but that bound is only a/100. A weaker lower bound can
coexist with larger actual error. Calibration failure is not proof of weak
privacy or an unstable mathematical search.

Maximum-reference scaling also only matches perturbation norms. Its directional
bound does not transfer automatically to Gaussian, Laplace or other directions.

## Recommended action

1. Document the supplied theorem's exact-fit, differentiability and uniform
   Lipschitz assumptions alongside calibration results. A feasible scalar search
   does not validate them. Retain the distinction between a conditional theorem
   and measured attack error.
2. Treat existing lower-bound calibrations as heuristic perturbation settings,
   including those labeled feasible. Attack measurements remain separate from
   whether a bound was established.
3. For a smooth-model theorem, justify region-wide constants and attainability,
   or derive an explicit approximate-fit version that controls residual error.
   Changing ReLU to a smooth activation would be a new victim model/protocol,
   not a silent numerical fix.
4. Improve and validate matrix-free singular solvers with inner residual and
   extremality/rank diagnostics. Preserve accuracy checks and record unresolved
   directions as errors; increasing budgets alone is not a general cure.
5. Keep reference comparisons labeled as matched noise norms: matching a
   maximum-direction reference's norm does not transfer its directional bound
   to a different perturbation direction.

Local supporting scripts/results are in `/workspace/prepush-tests/bound-audit/`:
`mu_j_probe.py` / `.json`, `resnet_jacobian_probe.py` / `.jsonl`,
`resnet_default_solvers.jsonl`, `resnet_relu_boundary.py` / `.json`, and
`lower_bound_math_counterexample.py` / `.json`. Equations from the uploaded
document were inspected as well as its text.
No experiment database or numerical formula was changed by this review.
