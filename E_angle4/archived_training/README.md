# RQ3 — Archived training and diagnostics

This source archive documents the historical run that produced the Lasso predictions reused in the final Angle 4 comparison. The final assembly in [`../code/assemble_results.py`](../code/assemble_results.py) reads existing predictions and fits no models. From the historical run it reuses only Lasso/price and Lasso/log predictions. The historical Random Forest used `max_features = 1 / 3`; final RQ3 uses `max_features = 0.3` for both RF targets, taking RF/price from Angle C and RF/log from E's separate RF/log run. **The archived RF configuration is therefore not the published RQ3 model**, and the archived RF metrics must not be read as final results.

## What this copy is

These three Python files are a **path port**, not byte-for-byte originals: only the input and output path constants were changed so the scripts can run inside this repository. Model selection, preprocessing calls, fitting, evaluation, metric definitions, and the files/schemas they write are unchanged. The ported copies have **not been re-run**; every number already under `results/` comes from the original historical run.

The original source files had these SHA-256 values:

| Archived file | Historical role | Original SHA-256 (byte-for-byte original) |
|---|---|---|
| `angle4_development_cv.py` | Produced the development out-of-fold (OOF) predictions. Uses the ten fixed grouped outer folds and selects Lasso's alpha within each outer training fold by five-fold `GroupKFold` with negative mean squared error. | `fc5484cd099fc43010103a5a2cbdabc0d5f3c888de9f0988b7d1b761d3c72ccb` |
| `angle4_final_holdout.py` | Produced the historical holdout predictions. Selects Lasso's alpha by five-fold `GroupKFold` on all development rows, refits on those rows, and loads the holdout only after the models and settings are fixed. | `f94624eae1c99c47e4c00f331f455a45a306eef654d3b5da76698669c92ea80c` |
| `angle4_diagnostics.py` | Produced the historical development diagnostics. Reads the saved development fold metrics and OOF predictions, then refits models within folds for Lasso group retention and RF grouped permutation importance; it also summarizes the saved OOF residuals. It supplies neither the reused predictions nor the final RQ3 figures. | `832da25e315fdcfedb570b7cb4fccb7f3c7f033488934a6440c82edf8f0bbcf9` |

The hashes are of the originals; they also match the original historical submission and the audit manifest. The ported files in this directory have different hashes because the path lines differ, and only those path constants changed.

## Path changes in this port

| Constant | Original | This copy |
|---|---|---|
| `CODE_DIR` | `Path(__file__).resolve().parent` (scripts sat beside `shared_preprocessing.py`) | `PACKAGE_ROOT / "code"` (now provides `shared_preprocessing.py` and `build_dataset.py`) |
| `DATA_DIR` | `CODE_DIR / "data"` | `PACKAGE_ROOT.parent / "data"` (repository `data/` with `cleaned.csv`, `split_manifest.csv`) |
| Output roots | Results subdirectories inside the old presentation workspace | `PACKAGE_ROOT / "archived_training_outputs" / {development,final,diagnostics}` |

`PACKAGE_ROOT = Path(__file__).resolve().parents[1]` is unchanged, so it resolves to `E_angle4/`. `E_angle4/archived_training_outputs/` is ignored by Git.

## How to run

Requirements are the historical ones listed below; the later RF/log revision's environment is in [`../code/requirements.txt`](../code/requirements.txt). Paths are derived from `__file__`, so the working directory does not matter.

```bash
python E_angle4/archived_training/angle4_development_cv.py   # writes archived_training_outputs/development/
python E_angle4/archived_training/angle4_diagnostics.py     # reads archived_training_outputs/development/, writes archived_training_outputs/diagnostics/
python E_angle4/archived_training/angle4_final_holdout.py   # writes archived_training_outputs/final/
```

`angle4_diagnostics.py` needs `development/fold_metrics.csv` and `development/predictions.csv`, so run the development script first. The holdout script reads the holdout partition; the holdout has already been viewed, so it is not a blind evaluation. These scripts are executable training sources, not entry points for the final assembly: work starts at module level and there is no safe `--help` mode. Generated files stay under `archived_training_outputs/`, apart from the released `results/`.

No path, model, or evaluation code beyond the constants above was ported. Lasso uses fold-local median imputation, full one-hot/multi-hot encoding, and numeric scaling; `log_price` predictions are converted back to BDT with `exp()` without smearing or clipping. The two historical prediction CSVs under [`../results/audit/legacy_1over3/`](../results/audit/legacy_1over3/) were checked against the final results: for each Lasso target, the 4,789 OOF and 1,208 holdout predictions match exactly after numeric parsing, with row IDs, groups, true prices, and OOF folds checked against the frozen data.

## Historical dependencies

`shared_preprocessing.py` loads the data and uses `build_dataset.py` for schema and split validation; both are in [`../code/`](../code/), and the frozen `cleaned.csv` and `split_manifest.csv` are in [`../../data/`](../../data/), byte-identical to the files in the original run package. That package declared Python ≥3.10, NumPy ≥1.24 and <3, pandas ≥2.2.2 and <4, and scikit-learn ≥1.4.2 and <2. The holdout and diagnostics scripts also import Matplotlib, which was not listed in that requirements file. Exact package versions for the historical run were not verified.

The historical development configuration is omitted here because the script writes the resolved absolute data path to `config_used.json`; the archived Python files contain no hard-coded machine path. Porting was verified by static checks (source diff limited to the path constants, AST comparison, and path-resolution checks) and by the existing output files; none of the ported scripts was executed.
