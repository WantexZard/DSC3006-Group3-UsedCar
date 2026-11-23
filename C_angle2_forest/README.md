# C — Angle 2 Random Forest side

Code:

- `code/tune_rf.py`: 12 configurations over `max_features` and `min_samples_leaf`; selected by pooled development OOF RMSE.
- `code/evaluate_rf.py`: OOF predictions, grouped permutation importance, MDI comparison.
- `code/evaluate_conditional.py`: permutation within the same `brand_model` to measure information beyond vehicle identity.
- `code/finalize_rf.py`: frozen configuration; one holdout evaluation.
- `code/merge_angle2.py`, `code/figures.py`, `code/selection_check.py`: comparison and figures. `selection_check.py` is optional and needs `outputs/oof_all_configs_main.csv` from a full retrain; that file is not in this archive.

Recorded frozen model (pinned environment; see requirements.txt): RF/price, `max_features=0.3`, `min_samples_leaf=1`, 500 trees, seed 42. Development pooled OOF RMSE 2.5391; holdout 1.5377 (million BDT). `outputs/` contains the tuning table, predictions, test predictions, importance tables and model config.
