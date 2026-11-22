# E — Angle 4 price vs log-price

Code:

- `code/assemble_results.py`: joins the frozen C RF/price predictions to the existing RF/log, Lasso/price and Lasso/log results; validates `row_id`, `group_id`, `cv_fold` and `y_true`; writes the four-model tables/figures.
- `code/update_rf_log_03.py`: RF/log 0.3 run (previous round).
- `code/make_figures.py`: figures.
- `code/requirements.txt`: RF/log revision environment.

Inputs for C are `../C_angle2_forest/outputs/predictions.csv`, `test_predictions.csv`, `model_config.json`. The script reuses predictions and does not retrain C's RF, Lasso, or RF/log. Frozen inputs are under `E_angle4/results/`; `exp()` is applied without a smearing correction; the holdout was previously viewed.
