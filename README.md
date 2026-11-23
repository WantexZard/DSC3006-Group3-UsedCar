# DSC3006 Group 3 — Used-Car Price Modelling

**Start here:** [`START_HERE.md`](START_HERE.md) — environment setup and notebook map. Run one part per virtual environment; precomputed results are included.

**Project:** Predicting Used-Car Asking Prices: What Do the Models Tell Us?  
**Course:** Fundamentals of Machine Learning I  
**Group:** Group 3

This repository contains the current analysis code for a grouped-validation study of used-car **asking prices** in Bangladeshi Taka (BDT). It is not a transaction-price or causal-pricing study.

## Data

- Source: Md Saikot Hossain (2026), *Used Car Price Prediction Dataset*, Version 2, Mendeley Data.
- DOI: [`10.17632/8d38h82fyt.2`](https://doi.org/10.17632/8d38h82fyt.2)
- License: CC BY 4.0.
- `member-a-v1`: 5,997 listings; **4,789 development / 1,208 holdout**; **5,608 attribute groups**; grouped ten-fold development CV.
- Matching attribute groups do not cross the development/holdout boundary or validation folds. Attribute groups are not verified vehicle IDs.
- `data/cleaned.csv` and `data/split_manifest.csv` are included for reproduction. See `DATA.md` for SHA-256.

## Code map

| Directory | Role | Main content |
|---|---|---|
| `A_data/` | A | data build, grouped split, shared preprocessing, EDA, built-in validation |
| `B_angle2_linear/` | B | OLS/Ridge/Lasso with grouped inner alpha selection |
| `C_angle2_forest/` | C | Random Forest tuning, grouped/conditional permutation, comparison |
| `D_angle5/` | D | five-model Lasso ablation: M0, Age², Mileage², M1, M2 |
| `E_angle4/` | E | price/log comparison, C RF/price integration, figures |
| `data/` | shared | `cleaned.csv`, `split_manifest.csv` |

`ENVIRONMENT.md` records per-part environments. The project did not use one identical environment across all members.

## Reproduction

Precomputed result tables are included so the repository can be inspected without retraining. Re-running requires the per-part environment; B, C and E record different scikit-learn versions.

- **Inspect**: read the included tables; no environment and no writes.
- **Summarise**: E's `assemble_results.py` / `make_figures.py` reuse frozen predictions and rewrite `E_angle4/results/` and `E_angle4/figures/`; they do not retrain models.
- **Retrain**: A rebuild, B, C or D fits new models. Retraining overwrites the included outputs (C: `C_angle2_forest/outputs/`; B: the command below uses `results_notebook/`; D: its ignored results directory). Use a fresh clone/copy if you need to preserve the recorded results.

The parts use different scikit-learn versions. Use a **separate virtual environment per part**; do not install all five requirement files into one environment.

```bash
# Example: B
python -m venv .venv-b
source .venv-b/bin/activate
pip install -r B_angle2_linear/requirements.txt
```

Run from the repository root.

### A — data build / EDA (optional; data are already included)

The EDA script needs matplotlib. Install `A_data/code/requirements.txt` plus `A_data/code/requirements-eda.txt`, or the pinned `A_data/code/requirements-validated.txt`.

```bash
# Optional rebuild from the official CSV (download it separately; it is not redistributed).
python A_data/code/build_dataset.py --source /path/to/UsedCarInfo.csv --out A_data/rebuilt

# EDA from the included member-a-v1 data
python A_data/code/eda.py --data-dir data --code-dir A_data/code --out A_data/eda
```

### B — linear models

Write to `results_notebook/` so the included `results/` stays authoritative.

```bash
cd B_angle2_linear
python code/run_linear_models.py --data-dir ../data --out results_notebook
python code/check_acceptance.py --results results_notebook
```

Optional figures additionally need matplotlib.

### C — Random Forest

The full chain overwrites `C_angle2_forest/outputs/`; use a fresh copy if you need the included outputs unchanged.

```bash
cd C_angle2_forest
python -B code/tune_rf.py
python -B code/evaluate_rf.py
python -B code/evaluate_conditional.py
python -B code/finalize_rf.py
```

### D — Angle 5

```bash
cd D_angle5
# Reads cleaned.csv and split_manifest.csv from ../data by default.
python angle5_member_a_v1.py
```

### E — Angle 4

```bash
cd E_angle4
python -B code/assemble_results.py --data-dir ../data \
  --c-oof ../C_angle2_forest/outputs/predictions.csv \
  --c-holdout ../C_angle2_forest/outputs/test_predictions.csv \
  --c-config ../C_angle2_forest/outputs/C_RF_price_0.3_model_config.json
python -B code/make_figures.py
```

`--c-config` is the **wrapper** `C_RF_price_0.3_model_config.json` (it embeds the original C model config), not C's `outputs/model_config.json`. Pass the three C files from the **same frozen run**; do not pair freshly retrained C predictions with the archived wrapper. This step rewrites `E_angle4/results/` and `E_angle4/figures/`. The frozen E inputs needed by `assemble_results.py` are under `E_angle4/results/` (legacy Lasso predictions and the RF/log 0.3 files). They are archive inputs, not new model fits.

E's integration reuses stored predictions and does not retrain C's RF, Lasso, or RF/log.

## Scope and exclusions

This is a completed coursework analysis archive: holdout labels and test predictions are included for post-hoc inspection. It is not a blind benchmark; do not use the included holdout as an untouched evaluation set for a new model.

- The repository includes result tables and row-level predictions for selected models, not predictions for every intermediate run.
- E's historical run and its original source are retained for provenance and Lasso prediction reuse. Its legacy RF predictions do not enter the final comparison. Fixed-alpha intermediate results, the obsolete 5,700-row analysis and V1–V5 slides are excluded.
- Results are model associations, not causal effects. The holdout was previously viewed for RQ3; it is a revised evaluation, not a new blind test.
