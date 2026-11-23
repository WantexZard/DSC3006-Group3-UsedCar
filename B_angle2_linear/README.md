# B — Angle 2 linear side

Code:

- `code/run_linear_models.py`: OLS/Ridge/Lasso on the fixed outer ten folds; Ridge/Lasso alpha selected by inner 5-fold GroupKFold; preprocessing fitted inside the inner pipeline.
- `code/check_acceptance.py`: row counts, row uniqueness, fold coverage, group isolation, pooled-metric consistency.
- `code/build_manifest.py`, `code/make_figures.py`: packaging and figures. `make_figures.py` is optional; install `matplotlib` separately before using the notebook's figure cell.
- `results/`: current model config, pooled metrics, Lasso selected features.

Recorded development pooled OOF (million BDT): OLS 2.8882, Ridge 2.9762, Lasso 2.9332; Lasso keeps 60/214 columns. These are the delivered results from the pinned scikit-learn 1.4.0 environment; a different version may select different alphas/features. Precomputed result files are authoritative. No test predictions for the linear models; Angle 2 test reporting is restricted to the frozen RF.
