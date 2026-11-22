# D — Angle 5 nonlinearity and interaction

Code: `angle5_member_a_v1.py`.

Five Lasso specifications on the same seven feature groups, fixed outer ten folds and fold-local preprocessing:

- M0: base seven groups
- Age²: M0 + age squared
- Mileage²: M0 + mileage squared
- M1: M0 + both squared terms
- M2: M1 + age × mileage

Primary metric: pooled development OOF RMSE. Precomputed result tables are included; retraining may vary with dependency versions. M1 is the selected development model; the interaction has no consistent incremental benefit. The script reads the shared `data/cleaned.csv` and `data/split_manifest.csv` by default; `results/` contains the compact result tables.
