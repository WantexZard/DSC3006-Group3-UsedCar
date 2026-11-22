# Start here — before running anything

This repository is a **code + results archive** for a completed coursework analysis. It is not a blind benchmark: holdout labels and test predictions are included for post-hoc inspection.

## Read this first

1. Use **Python 3.12, 64-bit**.
2. **Do not install all parts into one virtual environment.** B, C and E record different scikit-learn versions:
   - B: scikit-learn 1.4.0
   - C: scikit-learn 1.9.1
   - D: scikit-learn >= 1.2 (unpinned)
   - E: scikit-learn 1.8.0
   - A: scikit-learn 1.9.1
3. Use one virtual environment per part you want to run.
4. Precomputed result tables are included. If you only want to inspect results, you do not need to run the notebooks.
5. Open a notebook from `notebooks/` in VS Code/Jupyter, select the venv interpreter for that part, and run cells from top to bottom.

## Create an environment

### macOS / Linux

```bash
python3.12 -m venv .venv-b
source .venv-b/bin/activate
pip install -r B_angle2_linear/requirements.txt
# optional notebook UI
pip install -r requirements-jupyter.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv-b
.\.venv-b\Scripts\Activate.ps1
pip install -r B_angle2_linear\requirements.txt
# optional notebook UI
pip install -r requirements-jupyter.txt
```

Replace `B_angle2_linear` with `A_data`, `C_angle2_forest`, `D_angle5` or `E_angle4` for another part. A's requirements are in `A_data/code/requirements.txt`.

## Cross-platform notes

- Use `/` in paths; Python's `pathlib` handles it on Windows too.
- On Windows PowerShell, set UTF-8 if your terminal shows mojibake: `$env:PYTHONUTF8 = "1"`.
- On headless Linux, set `MPLBACKEND=Agg` before figure cells.
- For B, the notebook uses `--n-jobs 1` for reliable Windows/macOS/Linux execution.
- Exact numbers can differ if you install a different scikit-learn version; the included results are the recorded ones.
- C's full retrain writes into `C_angle2_forest/outputs/`. If you want to preserve the included outputs, run it from a fresh clone/copy.
- E's `assemble_results.py` reuses frozen predictions; it does not retrain C's RF, Lasso or RF/log.

## Notebook map

| Notebook | Part | Runtime | Writes |
|---|---|---|---|
| `notebooks/A_data.ipynb` | data build, EDA | seconds | `A_data/rebuilt_notebook/`, `A_data/eda_notebook/` |
| `notebooks/B_linear.ipynb` | OLS/Ridge/Lasso | ~3 min | `B_angle2_linear/results_notebook/` |
| `notebooks/C_forest.ipynb` | RF tuning/eval/merge | ~10 min if full retrain | `C_angle2_forest/outputs/` |
| `notebooks/D_angle5.ipynb` | five-model Lasso | minutes | ignored `D_angle5/Angle5_member_a_v1_final_results/` |
| `notebooks/E_angle4.ipynb` | price/log integration | seconds | `E_angle4/results/` and `figures/` |

## If you only want one step

Open the matching notebook and run only the setup cell plus the cell for that step. The notebooks print the exact command before running it, so you can copy the command into a terminal instead.
