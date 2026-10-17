# Code review and validation scope

The active path stores raw feature-space centroids and associates one centroid with each successfully trained learner, including multiple learners per cluster. Constant feature columns do not create NaNs during clustering. ANN/CNN outputs preserve their numeric training labels; an all-tied valid selection retains a nonempty pool. The main driver writes one row per repeat; unmeasured deviation columns contain NaN, not a claim of zero uncertainty. The alternate driver aggregates repeats. The separate `runTraining_SVM.m` variant no longer z-scores labels or pre-scales only the training fold: SVM models handle predictor standardization internally. That variant still needs runtime verification.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Still requires target-runtime validation

- MATLAB/toolbox compatibility and neural-network training/prediction APIs.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The review environment has no MATLAB installation. Static lint passed, but the MATLAB unit tests were not run. No reproduced accuracy or benchmark claim is made.
