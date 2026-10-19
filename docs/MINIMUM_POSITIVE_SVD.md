# Minimum numerically nonnull singular directions

Minimum-SVD experiments now select the smallest singular value that is resolved
above an explicit numerical rank cutoff. This is the requested common convention
for every model, including architectures whose gradient Jacobian has an exact
input nullspace. It changes the experimental definition from the full minimum
singular value, which can be zero, to the minimum numerically nonnull coupled
mode.

The defaults are:

```yaml
solver_method: positive_svd_v1
rank_policy: smallest_nonzero
relative_rank_tolerance: 1.0e-5
```

These settings apply to minimum-SVD ranking and, where present, noise construction
in `min_svd_noise`, `min_svd_lower_bound`, `min_svd_noise_reference`, and
`min_svd_pruning`. They do not change maximum-SVD references, noise budgets,
selection masks, reconstruction targets, or attack stopping rules.

## Meaning of the cutoff

For the fixed gradient-map Jacobian J, retain singular values satisfying

```text
sigma > relative_rank_tolerance * spectral_scale
```

The dense method obtains the spectral scale from its largest singular value.
Matrix-free estimates record their scale estimate and its lack of certification.
The smallest retained value defines the requested numerical mode. Values below
the cutoff are treated as numerically null, even when they are mathematically
positive. Changing the cutoff changes the scientific protocol and configuration
hash; it must be reported with comparisons.

The rank cutoff `1e-5` and the singular residual tolerance `1e-4` serve different
purposes. The cutoff defines which values count as nonnull. The residual tests
whether a returned triplet satisfies Jv=sigma*u and J.T*u=sigma*v accurately
enough. Neither is the reconstruction MSE target. A zero Jacobian has no positive
mode and remains an explicit numerical error, rather than producing arbitrary
noise.

## Bounded dense calculation and the supplied LeNet case

When the Jacobian fits both declared dense caps, the method builds its columns
and performs a CPU float64 QR/SVD calculation. The default limits are
`dense_max_elements: 300000000` Jacobian entries and
`dense_max_input_size: 4096` input coordinates. These bounds permit the reported
CIFAR LeNet case. They do not imply a small allocation: the Jacobian, float64
factorization, and workspaces can require several GB of RAM. A zero dense cap
disables this route. Explicit smaller exact-SVD limits also constrain the dense
route; increasing a limit is an explicit resource decision.

The independently reproduced LeNet case uses CIFAR-10 test image 5872, experiment
seed 0, and model hash
`38df24d9424830dd70b9403ba4a7145ccf9403211bfa9e6e47cec6e43635bf5e`.
Its Jacobian has shape 83,126 by 3,072. At the default relative cutoff:

| Quantity | Value |
| --- | ---: |
| Numerical rank | 1,992 |
| Input nullity | 1,080 |
| Largest singular value | 0.0915130274 |
| Absolute rank cutoff | 0.000000915130274 |
| Smallest retained singular value | 0.000003090901344 |
| Independently checked relative singular residual | 0.000000029308 |

The retained rank agrees with the architecture's proven rank upper bound. The
full minimum is still zero; the selected positive mode is a different, now
explicit protocol. The standalone calculation took approximately 30 seconds on
four CPU threads, and approximately 76 seconds under concurrent load. Runtime
and memory use depend on hardware and other work.

The implementation preserves the left singular vector returned by the dense
factorization. Replacing it with a newly computed float32 Jv/sigma amplifies
cancellation for this weak mode: that reconstruction gave residual approximately
0.00028. The preserved triplet instead passes independent checks against the
original derivative operators at the residual reported above.

## Larger matrix-free cases

Cases outside the dense caps use the bounded input-space Krylov path with default
search limits of 512 steps, checks every 16 steps, and 8,000,000 stored basis/
product elements. The local solver seed remains 0. Every accepted triplet is
checked with the original JVP/VJP operators, and matching ranking/construction
calls reuse the same checked triplet.

A partial matrix-free subspace does not establish numerical rank or the global
smallest positive value. In particular, when an unresolved null direction is
present, selecting the next positive projected eigenvalue can return an interior
mode. The solver must report that case as unresolved rather than silently
skipping it. A full-basis calculation can also remain ambiguous when float32
normal-equations errors obscure the requested cutoff; that case is rejected.
The diagnostics retain the lack of global-minimum certification for iterative
estimates. This convention removes the undefined division by an exact zero;
it does not guarantee convergence for every model or precision.

The supplied ResNet18/CIFAR case (same image and seed, model hash
`7c33f3aef9de71cda2f28bdfb2dfbaad9af085fc0e6f77916fd5259c96410180`)
also passes under the new protocol: sigma 0.03287183895675 after 400 search
steps, with relative singular residual 0.0000881678598. It took approximately
183 seconds under concurrent CPU load. Its diagnostics explicitly record
`rank_complete: false` and no global-minimum certification.

A Beyond Gradients/LeNet integration run using the registered positive-mode
preset completed one restart and one attack iteration in approximately
32.4 seconds. It recorded an ordinary attack `failure` after that budget,
with no numerical error, and preserved the original database checksum. This
checks transformation, attack startup and storage. It is not validation of all
468 scheduled runs, a full attack-success sweep, or a certified privacy result.

All four minimum presets were also applied to fresh copies of that LeNet's
original gradients, reusing its checked triplet. The fixed-noise L2 was 0.01,
the conditional lower-bound L2 was 3.42894198, the reference-matched L2 was
3.90001728, and pruning marked 9.9993% of coordinates missing. Applied norms
matched the scaling diagnostics. The masked minimum direction's measured gain
was 0.00179218; masking changes it from the original full singular mode.

An isolated generation/migration check validated all 468 schedule configurations.
Its 72 minimum schedules (126 ranking/construction packages) adopted the shared
positive policy. Migration preserved attacks, observers, masks, budgets and
explicit residual/resource controls, wrote 72 backups, and left the 396 unrelated
files byte-identical. Repeating migration made no changes. These are configuration
checks; they do not execute the complete experiment matrix.

## Existing schedules and results

Stop the running process and load the updated code before restarting. Existing
minimum-SVD schedule entries without an explicit rank policy adopt the new
positive protocol in memory, including entries naming the older inverse or
Lanczos methods. Their obsolete inverse budgets are removed from the effective
configuration. Resolved method, rank policy, cutoff and resource settings are
included before validation and hashing. The source YAML is not rewritten during
this normalization.

Historical database rows, errors, logs and images remain unchanged. The new
protocol has a distinct configuration identity, so a completed old full-minimum
attempt does not suppress its positive replacement. Unrelated completed runs
still skip normally. No database reset is needed.

An explicitly recorded strict `rank_policy: error` with an older method remains
available for historical reproduction. To migrate all minimum-SVD YAMLs,
including those explicit strict entries, and make the new settings visible in
the files:

```bash
python scripts/set_min_svd_solver.py schedual --dry-run
python scripts/set_min_svd_solver.py schedual
python -m run
```

The script validates the entire schedule first, preserves initial
`*.before-min-svd-solver.bak` backups, and is idempotent. It updates both minimum
ranking and construction, preserving attacks, observers, masks, scaling and
custom singular residual tolerances. The CLI exposes
`--relative-rank-tolerance`, `--dense-max-elements`, `--dense-max-input-size`,
Krylov work/memory/check caps, and the solver seed. Apply the same declared rank
cutoff across models for a matched protocol.

## Scientific interpretation

A minimum-positive coupled mode, a right-null input direction, and a left-null
gradient-noise direction are distinct objects. This change deliberately selects
the first. It does not establish a privacy bound for either nullspace.

Construction still masks and renormalizes the full left singular vector. That
masked direction is generally not the minimum mode of the restricted observation
operator. Report the actual directional gain after masking. Fixed and relative
noise budgets also remain distinct. ReLU smoothness and supplied mu_J values
remain unverified assumptions for conditional bound calibration. See the
[scientific-readiness review](SCIENTIFIC_READINESS.md) and
[residual-aware bound](RESIDUAL_BOUND.md) for those separate limitations.
