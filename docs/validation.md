# Executed validation

Date: 17 October 2026. [Successful GitHub Actions run 39000999554](https://github.com/zohaibjan/IEEE_TNNLS/actions/runs/39000999554) at commit `0f895ec6d3bf7b432e240378ceb6c9c2d4a922f4`.

- MATLAB R2024b on Ubuntu 22.04.
- 22 tests passed; zero failed or incomplete.
- 31 root function files loaded/compiled by MATLAB.
- Installed toolboxes: Statistics and Machine Learning, Deep Learning, Global Optimization, Optimization (dependency), and Parallel Computing.

## What was executed

Actual SVM/tree fitting, ANN/CNN training, utilities, compilation of root functions, and both fold drivers on a synthetic two-fold dataset.

Both fold drivers ran on a synthetic 160-row dataset with two outer folds. The tests use params.maxWorkers=0 to execute the parfor body locally, decision trees for the general driver, and fitted radial/linear SVMs for the SVM driver. Voting, selection and centroid routing all return finite accuracies. Separate tests exercise real ANN/CNN training and preserve the original numeric label vocabulary.

## Corrected behavior

Nested loop indexing is compatible with parfor. Both fold-driver smoke tests set params.maxWorkers=0 to execute the parfor body locally; multi-worker scaling, timing and reproducibility remain untested. Default maxWorkers remains Inf. The SVM variant is covered by this smoke test. The validation splitter guarantees at least one training and validation row per class, preserves the requested ratio when feasible, and explicitly rejects singleton classes.

The splitter selects max(1,round(fraction*classCount)) validation rows per class, capped to leave training rows. This changes the historical 1% holdout only where rounding would produce an empty class subset. A class with only one training-fold row fails explicitly instead of leaking the row between subsets.

## Reproduce

Use a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The read-only test workflow runs on MATLAB changes/pull requests and supports manual dispatch. It does not deploy anything.

## Limits

Root function loading is a compilation check, not full branch coverage. Full research datasets, published accuracy, multi-worker performance, memory and historical results have not been reproduced. Default training and experiment budgets remain available. Deterministic model fixtures isolate utility behavior; toolbox and fold smoke tests train actual models. Historical CSVs were not regenerated or relabelled.
