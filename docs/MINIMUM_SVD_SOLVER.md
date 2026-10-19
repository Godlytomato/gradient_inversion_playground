# Minimum-SVD convergence repair

The reported ResNet error concerns numerical accuracy of a singular triplet,
not MSE success, a reconstruction threshold, or lower-bound feasibility. Writing
J for the gradient-map Jacobian, the solver must find vectors satisfying
Jv=sigma*u and J.T*u=sigma*v. A small sigma is permitted. A large residual means
the returned vectors do not satisfy those equations accurately enough.

## Reproduced cause

The reported case has experiment seed 0, CIFAR-10 test image 5872, and model hash
`7c33f3aef9de71cda2f28bdfb2dfbaad9af085fc0e6f77916fd5259c96410180`.
After the float64 norm-accumulation fix, the old matrix-free inverse method
still returned sigma 0.108704 and relative singular residual 0.0614898 against
tolerance 0.0001. Five inverse updates with eight CG steps each leave their
inner systems substantially unconverged. This is an inadequate numerical
approximation for this case. It does not establish that the minimum is zero or
that a reconstruction bound is infeasible. `min_svd_noise` uses fixed scaling
and has not called any reconstruction-bound calibration at this point.
The resolved maximum/minimum estimates differ by about a factor of 89; the
corresponding normal-equations spectral ratio is about 8,000. Eight inner CG
steps cannot be assumed sufficient for this spectrum. Krylov search still faces
the conditioning of J.T*J, but preserves an orthogonal search space and checks
the resulting triplet instead of relying on those inaccurate inverse updates.

## New checked method

New minimum-SVD presets select `solver_method: lanczos_v1`. The solver searches
an input-space Krylov basis for the lowest Rayleigh--Ritz mode of normalized
J.T*J. It retains all searched basis vectors and products within its declared
caps, with two-pass float64 reorthogonalization and projected eigendecomposition.
It does not perform the old inaccurate inner inverse solves.

The model and derivative products retain their original dtype. Forward-over-
reverse `torch.func` derivatives reduce search cost; unsupported operations use
ordinary autograd, with the backend/fallback reason recorded. Every candidate
is independently checked with the original public autograd JVP/VJP operators.
The final acceptance criterion remains the singular residual normalized by the
recorded spectral-scale estimate, at tolerance 0.0001. Scale estimation is not a
certified spectral upper bound. Numerical-null directions are rejected before
normalizing them into noise. At an invariant subspace, the method explores an
independent orthogonal direction when dimensions and budget remain, instead of
accepting an isolated interior mode simply because its residual is zero.

Default limits are 512 normal-operator products for the search, checking every
16 steps, and 8,000,000 total basis/product elements (64 MB in float64). There
are five preliminary power updates for scale estimation and separate candidate/
final derivative checks. Thus the search cap is not a total derivative-call
cap. The independent solver seed defaults to 0; noise draws do not change it.
Runtime depends on the model and case, and exhausting a cap still produces an
explicit numerical error. No noise method or approximate direction is silently
substituted after failure.

For matching operator/settings within one evaluation context, ranking and
construction reuse one successful checked triplet. Cached values are detached
and isolated; changed inputs, model/loss state or settings invalidate them.
Unsupported mutable custom losses disable reuse. This avoids a second expensive
solve and different directions from successive random starts. The cache does
not change which coordinates are selected: construction still masks the full
left direction, as discussed in the [scientific review](SCIENTIFIC_READINESS.md).

## Existing schedules

Old YAML with explicit five/eight inverse budgets keeps `inverse_v2`; reading it
does not silently grant a larger budget or switch algorithms. New Python
presets choose Lanczos unless the caller explicitly supplies legacy inverse
settings. To update existing schedules, stop the runner and execute:

```bash
python scripts/set_min_svd_solver.py schedual --dry-run
python scripts/set_min_svd_solver.py schedual
python -m run
```

The migration validates the whole directory before writing, preserves initial
backups as `*.before-min-svd-solver.bak`, and is idempotent. It replaces inverse
solver parameters on min-SVD ranking/construction with matching Lanczos settings.
Selection masks, noise scaling, attacks and observer settings remain unchanged.
Custom tolerances and exact-SVD caps are retained. CLI options expose the Krylov
work/memory/check limits and solver seed. The algorithm and effective settings
are serialized and normalized before configuration hashing; new-method runs
therefore have a distinct scientific identity. Historical database rows are
preserved, including their errors. No database reset is necessary.

## Validation and limits

The integrated solver reproduces the supplied model/image identity and passes
at step 400 with sigma 0.03287184 and relative singular residual 0.0000882293.
This took 165.8 seconds on one CPU thread. A separate prototype using the
historical random start passed at step 336 with sigma 0.03287186 and residual
0.0000867371. These independent starts agree closely on the estimated singular
value. Regression cases compare against small dense SVDs, including rotated,
clustered, rank-deficient and rescaled maps; enforce work/memory caps; explore
an initially missed minimum; and verify backend fallback and triplet reuse.
All 155 unit/regression tests passed, including configuration migration,
historical-result preservation and cache invalidation for changing model/loss
state. Tests use synthetic data and temporary databases.

The full Beyond Gradients/min-SVD-noise/ResNet18 pipeline also completed on the
same actual CIFAR case, using one restart and one attack iteration in an isolated
database. It took 169.8 seconds, recorded a normal attack failure after its
declared budget, and preserved the original database checksum. This checks
construction, attack startup and result storage; it is not a full attack-success
sweep. The min-SVD triplet was reused for construction after ranking.

All three additive minimum-SVD presets were also exercised on the same case,
with the unchanged cached triplet reused across control checks:

| Control | Applied noise L2 | Result |
| --- | ---: | --- |
| `min_svd_noise` | 0.00999999952 | finite, matched fixed budget |
| `min_svd_lower_bound` | 3.79664240 | conditionally calibrated |
| `min_svd_noise_reference` | 23.34385461 | matched maximum-reference norm |

The combined check took 198.3 seconds. The reference maximum solve also passed
at residual 0.0000922597. These control checks validate construction/calibration,
not a guaranteed MSE for the applied direction or verified smoothness constants.

A converged residual supports an estimated singular mode. It does not certify
the global smallest value or a unique basis in a clustered eigenspace. Very weak
or null modes may remain unresolved in the model's precision; the minimum-
positive rank policy still requires an exact SVD within its caps. Larger cases
can require a different declared budget or a separate rank-aware method.
ReLU smoothness, supplied mu_J values, attack residual assumptions, masking and
matched-noise-budget concerns remain separate from numerical convergence. This
repair does not turn conditional ResNet calibration into a certified bound.

Local exact-case artifacts are under `/workspace/prepush-tests/min-solver/`:
`production_case.py`, `production_case.json`, prototype logs/states, and
`beyond-gradients/results.json`. The three-control check is preserved in
`min_controls.py` and `min_controls.json`. CUDA execution was not tested.
