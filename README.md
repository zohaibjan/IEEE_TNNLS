# Multicluster Class-Balanced Ensemble

MATLAB research code for multicluster class-balanced ensembles.

## Research reference and attribution

Jan, Z. and Verma, B. (2020). **Multicluster Class-Balanced Ensemble.** IEEE Transactions on Neural Networks and Learning Systems.

Please cite the relevant publication and retain source author notices. Existing dataset and third-party terms still apply; this cleanup does not grant a new licence.

## Data and environment

Supply `P-Data/<name>.mat` files with `X` (samples × features) and label column `y`. No dataset folder is committed in this repository. The current driver requests `students`; change `Problem` to a dataset you can supply. Original notes remain in [readMe.txt](readMe.txt).

**MATLAB dependencies:** Statistics and Machine Learning Toolbox; Deep Learning Toolbox for ANN/CNN variants; Parallel Computing Toolbox for fold-level `parfor`.

No exact original MATLAB/toolbox version is recorded. Use MATLAB rather than assuming GNU Octave compatibility for the learner APIs.

## Start here

1. Open this repository as the MATLAB current folder. Use a separate session/path for each research repository: helper filenames overlap across projects.
2. Supply or extract the data described above. Check feature/label row counts and class labels.
3. Edit `Problem` in `mainProgram.m` (or `dataSetNames.m` where used), and start with one dataset and one repeat.
4. Run the synthetic utility checks, then inspect `mainProgram.m` before starting the full experiment:

```matlab
results = runtests('tests');
assertSuccess(results);
rng(42, 'twister');
mainProgram;
```

The seed is a starting point for recording runs; nested seed resets, parallel workers and toolbox versions can still affect reproducibility. `runMultiple.m` performs 50 full driver repetitions and is not the quick-start command.

## Workflow and output

`mainProgram.m` uses SVMs, five folds, a 1–10 cluster-setting sweep and five repeats. `runTraining.m` compares balanced/unbalanced pools, ensemble selection and centroid fusion. `runTraining_SVM.m` is a separate legacy variant. `mainProgramSingleRun.m` is the renamed historical alternate driver.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

The active path stores raw feature-space centroids and associates one centroid with each successfully trained learner, including multiple learners per cluster. Constant feature columns do not create NaNs during clustering. ANN/CNN outputs preserve their numeric training labels; an all-tied valid selection retains a nonempty pool. The main driver writes one row per repeat; unmeasured deviation columns contain NaN, not a claim of zero uncertainty. The alternate driver aggregates repeats. The separate `runTraining_SVM.m` variant no longer z-scores labels or pre-scales only the training fold: SVM models handle predictor standardization internally. That variant still needs runtime verification.

## Correctness review — October 2026

This pass fixes implementation defects; it does not validate the associated research results. Changes can alter model pools, predictions and measured scores. Re-run the experiments before comparing corrected code with historical CSVs or published results.

- Label conversion uses numeric category names, not categorical indices.
- Voting rejects empty/nonfinite inputs; a prediction failure is reported rather than counted as a fabricated class-one vote.
- Neural models store their training labels in `net.userdata.classLabels`. Retrain old models or restore that exact training vocabulary before prediction; inferring it from a validation subset is unsafe.
- Synthetic checks cover normalization, CSV append behavior and the new prediction/selection boundaries. Additional cluster checks run where applicable.

See [the code review](docs/code-review.md) for repository-specific changes, tests and remaining limitations.

**Validation:** all MATLAB sources were statically parsed/linted with MISS_HIT. MATLAB and its learner toolboxes are unavailable in the review environment. The MATLAB regression suites and full training runs have **not been executed** here. Run `runtests('tests')` and `assertSuccess(results)` in MATLAB before using the corrected experiments.

[Profile and project directory](https://github.com/zohaibjan)
