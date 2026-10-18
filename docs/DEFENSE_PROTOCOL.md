# Defense protocol changes

These corrections change the experimental protocol. Regenerate descriptions and
rerun affected comparisons; old saved YAML files keep their explicitly selected
ranking and draft policies. Historical Soteria global drafts now fail with a
migration instruction rather than silently selecting outside the defended layer.
Historical results should retain their original
configuration and be labelled as historical implementations.

## Soteria-inspired feature pruning

New presets select complete input-feature columns of one fully connected weight.
`layer_name="auto"` chooses the last registered `torch.nn.Linear` module, which
matches the supplied LeNet and ResNet classifier layouts. An explicit layer name
is respected. `percent` means `floor(percent * input_feature_count)` columns of
that layer; biases and other parameters are excluded. The feature score is the
summed input derivative norm divided by `abs(feature) + stabilizer`, with the
highest scores pruned. This defines the implemented adaptation; it is not a claim
of paper-faithful Soteria replication. The old global coordinate selection is
incompatible with this protocol.

## GradDefense-inspired sensitivity and slices

The default `sensitivity_method="frobenius_rademacher"` estimates each parameter
tensor's loss-gradient Jacobian Frobenius norm. Independent random Rademacher
VJPs are squared and averaged before taking a square root. Saved descriptions
record the method, sample count (default 16), and a local estimator seed; these
probes do not advance the global random stream. The Jacobian is with respect to
all batch input coordinates and uses the configured loss reduction. It is not a
worst-case sensitivity bound or a paper-faithfulness claim.

`sensitivity_method="legacy_sum"` retains the old coordinate-sum derivative only
for explicitly labelled historical comparisons. Opposing derivatives can cancel
under that method, especially in softmax classifiers.

Slicing keeps each gradient atom intact and reserves at least one atom per slice.
Every configured slice is nonempty. Equal total sensitivity can be impossible
when an indivisible atom dominates; slice sizes can therefore differ. The noise
budget is normalized after selection. Global mode measures the requested
relative L2 perturbation against the whole original gradient; layer mode uses
the whole gradient tensor for each selected layer. Layers with no selected
coordinates and empty selections receive zero noise.

## Other corrected settings

- Gaussian and GradDefense reference presets forward their requested error
  target; the GradDefense reference preset is now discoverable.
- Solver settings are forwarded both to singular-direction ranking and noise
  construction. Convergence/rank checks can reject unsupported cases explicitly.
  Maximum-SVD presets use a checked adaptive default (10 initial steps, at most
  500). An explicit `iterations` value is a hard limit, rather than an instruction
  to keep refining silently beyond that budget.
- The `all` draft returns boolean masks. DGP keeps odd requested coordinate
  counts, allocating the extra coordinate to the high end.
- The supported signed symmetric quantizer requires at least two bits; one-bit
  settings fail during configuration instead of partway through an experiment.

Passing these implementation checks does not establish a guaranteed privacy
bound, reconstruction success rate, or training utility. Report measured applied
perturbation and the chosen method alongside attack effort and evaluation policy.
