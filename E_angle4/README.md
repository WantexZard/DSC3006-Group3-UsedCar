# E — Angle 4 price vs log-price

Code:

Path-adapted copies of the original scripts that generated the reused Lasso predictions are in [archived_training/](archived_training/README.md). They document the historical run; they are not needed to assemble the published results.

- `code/assemble_results.py`: joins the frozen C RF/price predictions to the existing RF/log, Lasso/price and Lasso/log results; validates `row_id`, `group_id`, `cv_fold` and `y_true`; writes the four-model tables/figures.
- `code/update_rf_log_03.py`: RF/log 0.3 run (previous round).
- `code/make_figures.py`: figures.
- `code/requirements.txt`: RF/log revision environment.

C inputs are `../C_angle2_forest/outputs/predictions.csv`, `test_predictions.csv`, and the wrapper `C_RF_price_0.3_model_config.json` (it embeds the original C model config; do not pass C's plain `outputs/model_config.json`). Pass the three files from the same frozen run; do not pair newly retrained C predictions with the archived wrapper. The script reuses predictions and does not retrain C's RF, Lasso, or RF/log. Frozen inputs are under `E_angle4/results/`; `exp()` is applied without a smearing correction; the holdout was previously viewed. Running the script rewrites `E_angle4/results/` and `E_angle4/figures/`.
