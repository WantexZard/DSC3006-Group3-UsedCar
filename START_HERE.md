# Start here — before running anything

This repository is a **code + results archive** for a completed coursework analysis. It is not a blind benchmark: holdout labels and test predictions are included for post-hoc inspection.

## Read this first

1. Use **Python 3.12, 64-bit**.
2. **Do not install all parts into one virtual environment.** B, C and E record different scikit-learn versions:
   - B: scikit-learn 1.4.0
   - C: scikit-learn 1.9.1
   - D: scikit-learn 1.9.1 in `D_angle5/requirements.txt` (the code itself only needs >= 1.2 for `sparse_output`)
   - E: scikit-learn 1.8.0
   - A: scikit-learn 1.9.1
3. Use one virtual environment per part you want to run.
4. Precomputed result tables are included. If you only want to inspect results, you do not need to run the notebooks.
5. Open a notebook from `notebooks/` in VS Code/Jupyter, select the environment for that part, and run the setup cells before the steps you need. Do not use Run All on the only copy of the archive.

## Choose how far to go

| Mode | What you do | Writes / risk |
|---|---|---|
| Inspect only | Read the included tables and figures; install nothing | Nothing is written |
| Summarise | Run E (`assemble_results.py`, `make_figures.py`) or the small comparison scripts; they reuse frozen predictions | Rewrites `E_angle4/results/` and `E_angle4/figures/`; it does not retrain models |
| Retrain | Run A build, B, C or D end to end | Writes new predictions/models. C overwrites `C_angle2_forest/outputs/`, B's example below writes `results_notebook/`, D writes its ignored results directory |

Running a retrain changes the numbers only for that environment; the committed tables remain the authoritative recorded results. To keep the included outputs, run a retrain from a fresh clone/copy, or point the command at a new output directory where the script supports it.

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

### Requirements file for each part

| Part | Install from |
|---|---|
| A — data build | `A_data/code/requirements.txt` |
| A — EDA (plots) | `A_data/code/requirements.txt` plus `A_data/code/requirements-eda.txt` (adds matplotlib), or the pinned `A_data/code/requirements-validated.txt` |
| B — linear | `B_angle2_linear/requirements.txt`; add `matplotlib` only for optional figures |
| C — Random Forest | `C_angle2_forest/requirements.txt` |
| D — Angle 5 | `D_angle5/requirements.txt` (pinned; code needs scikit-learn >= 1.2) |
| E — Angle 4 | `E_angle4/code/requirements.txt` |

## Cross-platform notes

- Use `/` in paths; Python's `pathlib` handles it on Windows too.
- On Windows PowerShell, set UTF-8 if your terminal shows mojibake: `$env:PYTHONUTF8 = "1"`.
- On headless Linux, set `MPLBACKEND=Agg` before figure cells.
- B's notebook uses `--n-jobs 1` to avoid multiprocessing differences in notebook environments.
- Exact numbers can differ if you install a different scikit-learn version; the included results are the recorded ones.
- C's full retrain writes into `C_angle2_forest/outputs/`. If you want to preserve the included outputs, run it from a fresh clone/copy.
- E's `assemble_results.py` reuses frozen predictions; it does not retrain C's RF, Lasso or RF/log.

## Notebook map

| Notebook | Part | Runtime | Writes |
|---|---|---|---|
| `notebooks/A_data.ipynb` | data build, EDA | seconds (EDA needs matplotlib) | `A_data/rebuilt_notebook/`, `A_data/eda_notebook/` |
| `notebooks/B_linear.ipynb` | OLS/Ridge/Lasso | ~3 min | `B_angle2_linear/results_notebook/` and optional `figures_notebook/` |
| `notebooks/C_forest.ipynb` | RF tuning/eval/merge | minutes; hardware dependent | `C_angle2_forest/outputs/` and `figures/` |
| `notebooks/D_angle5.ipynb` | five-model Lasso | minutes | ignored `D_angle5/Angle5_member_a_v1_final_results/` |
| `notebooks/E_angle4.ipynb` | price/log integration | seconds | `E_angle4/results/` and `figures/` |

## If you only want one step

Open the matching notebook, run its setup cells and any prerequisite step, then run the selected cell. For example, B's optional figure cell reads the results from B's training cell, and C's selection audit requires its full tuning output. Each command is printed with its working directory.
