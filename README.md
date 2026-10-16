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

The main driver writes one row per repeat and deliberately writes zero in its deviation columns; those zeros are placeholders, not measured uncertainty. The alternate driver aggregates repeated runs. `runTraining_SVM.m` historically z-scores labels with features and leaves test features unscaled; it needs a separate methodological review before use. Cluster centroids are computed in a scaled space while the fusion comparison uses raw features; validate the intended preprocessing against the paper.

The cleanup aligns function/file names, restores visible warnings, makes CSV handles close on errors, and tests constant-feature normalisation plus CSV append behaviour. MATLAB experiments and the new utility tests have **not** been executed in the cleanup environment, which has no MATLAB installation. Static parsing is not a reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
