# Code review and validation scope

The active path stores raw feature-space centroids and associates one centroid with each successfully trained learner, including multiple learners per cluster. Constant feature columns do not create NaNs during clustering. ANN/CNN outputs preserve their numeric training labels; an all-tied valid selection retains a nonempty pool. The main driver writes one row per repeat; unmeasured deviation columns contain NaN, not a claim of zero uncertainty. The alternate driver aggregates repeats. The separate `runTraining_SVM.m` variant no longer z-scores labels or pre-scales only the training fold: SVM models handle predictor standardization internally. Both fold drivers now have a synthetic MATLAB smoke test.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Executed MATLAB validation

Actual SVM/tree fitting, ANN/CNN training, utilities, compilation of root functions, and both fold drivers on a synthetic two-fold dataset.

Nested loop indexing is compatible with parfor. Both fold-driver smoke tests set params.maxWorkers=0 to execute the parfor body locally; multi-worker scaling, timing and reproducibility remain untested. Default maxWorkers remains Inf. Validation splitting preserves both training and validation rows per class; singleton classes fail explicitly. The SVM variant is covered by this smoke test.

See [validation.md](validation.md) for the exact run, release and boundaries.

## Remaining validation

- Compatibility beyond the tested MATLAB R2024b/toolbox release.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The regression/smoke suite passed in MATLAB R2024b using MathWorks GitHub Actions. Synthetic fixtures check implementation behavior; full research experiments, historical accuracy and performance benchmarks have not been reproduced.
