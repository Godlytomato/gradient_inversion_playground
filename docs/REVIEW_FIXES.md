# Review fixes and validation

The review of the initial commit found implementation and protocol defects. The
working tree now contains the corrections below. Historical experiment rows are
preserved; their outcomes are not retroactively corrected. Regenerate schedules
and rerun affected comparisons using the version 2 protocol in the README.

## Implemented corrections

| Area | Corrected behavior |
| --- | --- |
| Soteria adaptation | Complete feature columns of the selected linear layer, eligible-feature counts, automatic final-classifier mapping; historical global drafts rejected. |
| GradDefense adaptation | Declared Frobenius Rademacher sensitivity, explicit legacy-sum option, nonempty slices, applied-mask noise normalization. |
| Noise/reference settings | Effective relative L2 budgets, forwarded target errors, discoverable reference preset, boolean all-selection, odd DGP counts, consistent quantizer validation. |
| Jacobian/SVD | Zero derivatives materialized, bounded exact fallback, relative residual/rank checks, forwarded solver settings, bounded adaptive maximum-SVD. |
| Matching/labels | Observed zero constraints retained in DLG/iDLG, layer weights follow hidden-layer filtering, modifier contract fixed, invalid labels rejected. |
| Image/data protocol | Raw-pixel bounds, finite sampling, lazy downloads, private-target exclusion from auxiliary references, device/dtype alignment. |
| Experiment protocol | Declared preset budgets; separate practical attack-loss and oracle-MSE selections; final target evaluation without oracle stopping; paired, recorded random streams. |
| Batch evaluation | One-to-one image assignment with saved mappings; raw attacker-order tensors preserved. |
| Persistence | Complete summaries and numerical/effort metadata, additive migrations, preserved retry attempts, distinct canonical v2 hashes, non-destructive initialization. |
| Reporting | Correct PSNR infinity, shared raw-pixel display, comparable-stratum success-rate aggregation with counts and exclusions. |
| Validation | Reject unknown components/parameters, missing policy metrics, unbounded runs, zero restarts, incompatible label batch sizes and historical Soteria drafts. |

## Explicitly guarded or unresolved cases

- **Minimum-SVD on large models:** the implementation rejects unresolved or
  numerical-null directions. A CIFAR-shaped synthetic LeNet check still failed
  its relative residual tolerance at 20 inverse iterations and 64 CG iterations.
  This is a reported numerical limitation, not a completed minimum-SVD result or
  evidence of privacy. Larger/tuned budgets need separate validation.
- **Maximum-SVD:** checked adaptive defaults converged on the tested LeNet in
  76 power steps (about 1.6 seconds with two CPU threads). A hard insufficient
  budget raises rather than silently accepting a poor direction. Pre-push
  validation additionally found a case needing 278 steps: public presets now
  select the adaptive solver and its default cap is 500, without relaxing the
  residual tolerance. Explicit iteration limits remain hard limits.
- **HF/iDLG batch inference:** batch sizes above one are rejected until a
  justified inference method exists. The known class-exclusion/batch-shape bugs
  are prevented; a new multi-image label algorithm has not been invented.
- **Partial matching regularizer:** the incompatible registration is rejected
  with guidance to use the matching loss. A separate new gradient regularizer
  has not been claimed or silently substituted.
- **Theory/paper fidelity:** lower-bound diagnostics explicitly say
  `certified=False`. No formal bound, external-paper replication, trained-model
  utility result, full real-data sweep or GPU correctness claim follows from
  these implementation tests.

## Regression checks

Run from the repository:

```bash
MPLCONFIGDIR=/tmp/playground-matplotlib XDG_CACHE_HOME=/tmp/playground-cache \
  .venv/bin/python -m unittest discover -s tests -v
```

The suite uses synthetic data and temporary databases. It includes all nine
attack presets under bounded smoke settings, all 26 discoverable defense presets
on tiny data, all 702 description/YAML round trips, explicit Jacobian/SVD and
finite-difference checks, real LeNet/ResNet victim-gradient controls, paired-run
reproducibility, database migration/retry preservation, and practical/oracle
end-to-end persistence. The tests intentionally assert rejection for unsupported
or unconverged cases; passing them does not mean every defense converges on every
model. Bounded attack smoke tests reduce expensive inner/search budgets.

The latest pre-push run passed **76 tests** (about 9 seconds in a clean virtual
environment), including public maximum-SVD preset/adaptive-budget regressions,
generator-level target propagation and class-population/split/domain grouping
checks added after independent integration review. All 322 Python source files
in the current tracked/unignored source inventory compile. The results viewer also rendered a
populated temporary database through Streamlit AppTest with zero exceptions.
The existing experiment database was not used as a test fixture or reset.
