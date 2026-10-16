# Multicluster Class-Balanced Ensemble

MATLAB research code associated with:

Jan, Z. and Verma, B. (2020). **Multicluster Class-Balanced Ensemble.** IEEE Transactions on Neural Networks and Learning Systems.

Please cite the associated paper when using this work.

## Getting started

1. Review the `Problem` variable in `mainProgram.m`.
2. Supply the required datasets in the `P-Data` layout expected by the scripts.
3. Install the MATLAB toolboxes required by the selected learners.
4. Run `mainProgram` from the repository root.

The original instructions are retained in [readMe.txt](readMe.txt). They refer to a `P-Data` folder; verify dataset availability locally before running.

## Experiment variants

- `runTraining.m`: multiple-classifier workflow.
- `runTraining_SVM.m`: SVM variant; select it in the entry point when appropriate.
- `cnnEnsembleSelection.m` and `cnnFusion.m`: CNN ensemble helpers.
- `saveResults.m`: CSV output.

## Status

Historical research implementation. Source, dataset and toolbox compatibility must be checked before reproducing experiments; the documentation update does not claim a fresh benchmark run.

[Profile and project directory](https://github.com/zohaibjan)
