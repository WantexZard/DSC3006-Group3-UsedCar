"""Merge B_angle2_linear's linear results with the forest results (Angle 2, shared table).

B_angle2_linear supplies `predictions.csv` in the interface A_data specified, so the
two members can be compared on exactly the same rows and folds. This script:

  1. verifies the row sets and fold assignments agree (a mismatch would make the
     comparison meaningless, so it stops instead of guessing);
  2. recomputes pooled out-of-fold metrics for OLS / Ridge / Lasso / Random Forest
     on the same 4,789 development records;
  3. reports paired differences against the forest with a bootstrap interval --
     the folds are not independent experiments, so a per-fold t-test would
     overstate precision;
  4. maps Lasso's selected encoded columns back to the six interpretation groups,
     so the answer to "do Lasso and the forest agree?" is at group level rather
     than a rank of 60 encoded columns.

Inputs are read-only: B_angle2_linear's folder and `outputs/predictions.csv`.
Outputs go to `outputs/`:

    angle2_models.csv         one row per model: pooled metrics, paired difference
                              against the forest with its interval, fold spread
    angle2_lasso_groups.csv   Lasso selection aggregated to interpretation groups
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import MEMBER_C_DIR, OUTPUT_DIR, SEED

B_DIR = Path(os.environ.get("MEMBER_B_DIR", MEMBER_C_DIR.parent / "B_angle2_linear" / "results"))
FOREST_LABEL = "Random Forest"
M = 1_000_000

GROUP_OF_PREFIX = (
    ("numeric__car_age", "age"),
    ("numeric__kilometers_run", "mileage"),
    ("numeric__energy_capacity", "capacity"),
    ("categorical__brand_model_", "vehicle_identity"),
    ("categorical__brand_", "vehicle_identity"),
    ("categorical__transmission", "transmission"),
    ("fuel__", "fuel"),
)


def group_for(variable: str) -> str:
    """Map an encoded column name to A_data's interpretation group."""
    for prefix, group in GROUP_OF_PREFIX:
        if variable.startswith(prefix):
            return group
    raise ValueError(f"Unmapped encoded column: {variable}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    return {
        "n": int(y_true.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual**2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def bootstrap_delta(
    y_true: np.ndarray,
    pred_baseline: np.ndarray,
    pred_model: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict:
    """Bootstrap the difference in pooled RMSE (model - baseline), resampling rows.

    Rows, not folds: the ten folds were trained on overlapping data, so treating
    fold-level scores as independent replicates would understate uncertainty.
    """
    rng = np.random.default_rng(seed)
    n = y_true.size
    base_sq = (y_true - pred_baseline) ** 2
    model_sq = (y_true - pred_model) ** 2
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = np.sqrt(model_sq[idx].mean()) - np.sqrt(base_sq[idx].mean())
    point = float(np.sqrt(model_sq.mean()) - np.sqrt(base_sq.mean()))
    return {
        "delta_rmse": point,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "prob_model_better": float((deltas < 0).mean()),
    }


def main() -> None:
    forest = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    linear = pd.read_csv(B_DIR / "predictions.csv")
    folds = pd.read_csv(OUTPUT_DIR / "fold_metrics.csv")[["model_id", "cv_fold"]]

    forest_rows = forest[["row_id", "group_id", "cv_fold", "y_true"]].copy()
    forest_rows["row_id"] = forest_rows["row_id"].astype(str)
    linear["row_id"] = linear["row_id"].astype(str)

    models = sorted(linear["model_id"].unique())
    print("linear models in B's file:", models, "rows:", len(linear))

    for model in models:
        block = linear[linear["model_id"] == model]
        if len(block) != len(forest_rows):
            raise SystemExit(f"{model}: {len(block)} rows vs forest {len(forest_rows)}")
        merged = block[["row_id", "cv_fold", "y_true"]].merge(
            forest_rows, on="row_id", suffixes=("_b", "_c"), how="outer", indicator=True
        )
        if (merged["_merge"] != "both").any():
            raise SystemExit(f"{model}: row_id sets differ")
        if not np.allclose(merged["cv_fold_b"], merged["cv_fold_c"]):
            raise SystemExit(f"{model}: fold assignment differs")
        if not np.allclose(merged["y_true_b"], merged["y_true_c"]):
            raise SystemExit(f"{model}: y_true differs on shared rows")
    print("row alignment verified for", models)

    # ---- pooled metrics on identical rows ----------------------------------
    reference = forest.sort_values("row_id")
    rows = []
    predictions = {}
    for model in models:
        block = linear[linear["model_id"] == model].sort_values("row_id")
        predictions[model] = block["y_pred_price"].to_numpy()
        block_metrics = metrics(reference["y_true"].to_numpy(), predictions[model])
        rows.append({"model_id": model, **block_metrics})
    predictions[FOREST_LABEL] = reference["y_pred_price"].to_numpy()
    rows.append({"model_id": FOREST_LABEL, **metrics(reference["y_true"].to_numpy(), predictions[FOREST_LABEL])})
    combined = pd.DataFrame(rows).sort_values("rmse")

    # per-fold spread from B's own file, for the "mean ± sd" line
    fold_metrics = pd.read_csv(B_DIR / "fold_metrics.csv")
    fold_spread = (
        fold_metrics.groupby("model_id")["RMSE_BDT"].agg(["mean", "std"]).rename(
            columns={"mean": "mean_fold_rmse", "std": "fold_rmse_sd"}
        )
    )
    combined = combined.merge(fold_spread.reset_index(), on="model_id", how="left")

    # ---- paired comparison against the forest ------------------------------
    y = reference["y_true"].to_numpy()
    paired = []
    for model in models:
        diff = bootstrap_delta(y, predictions[FOREST_LABEL], predictions[model])
        paired.append(
            {
                "baseline": FOREST_LABEL,
                "compared": model,
                "delta_rmse_bdt": diff["delta_rmse"],
                "ci95_low": diff["ci_low"],
                "ci95_high": diff["ci_high"],
                "prob_compared_better": diff["prob_model_better"],
            }
        )
    paired_frame = pd.DataFrame(paired)
    combined = combined.merge(
        paired_frame[["compared", "delta_rmse_bdt", "ci95_low", "ci95_high", "prob_compared_better"]].rename(
            columns={"compared": "model_id"}
        ),
        on="model_id",
        how="left",
    )
    combined.to_csv(OUTPUT_DIR / "angle2_models.csv", index=False)

    # ---- Lasso selection aggregated to interpretation groups ---------------
    lasso = pd.read_csv(B_DIR / "lasso_selected_features.csv")
    lasso["group"] = lasso["variable"].map(group_for)
    lasso["abs_coef"] = lasso["coefficient"].abs()
    by_group = (
        lasso.groupby("group")
        .agg(n_selected=("variable", "size"), abs_coef_sum=("abs_coef", "sum"), max_abs_coef=("abs_coef", "max"))
        .reset_index()
    )
    by_group["share_of_selected"] = by_group["n_selected"] / by_group["n_selected"].sum()
    by_group["share_of_abs_coef"] = by_group["abs_coef_sum"] / by_group["abs_coef_sum"].sum()
    by_group = by_group.sort_values("abs_coef_sum", ascending=False)
    by_group.to_csv(OUTPUT_DIR / "angle2_lasso_groups.csv", index=False)

    # ---- summary for the console (nothing extra is written) -----------------
    importance = pd.read_csv(OUTPUT_DIR / "importance.csv")
    perm = importance[importance["measure"] == "permutation_raw"].set_index("group")["value"]
    cond = importance[importance["measure"] == "permutation_conditional"].set_index("group")["value"]
    group_table = by_group.set_index("group").join(perm.rename("permutation_drmse")).join(
        cond.rename("conditional_drmse")
    )

    forest_best = combined[combined["model_id"] == FOREST_LABEL].iloc[0]
    linear_best = combined[combined["model_id"] != FOREST_LABEL].iloc[0]
    gain = (linear_best["rmse"] - forest_best["rmse"]) / linear_best["rmse"] * 100

    print("\npooled OOF on identical rows")
    for row in combined.itertuples():
        print(f"  {row.model_id:14s} RMSE {row.rmse / M:>7.3f}M  MAE {row.mae / M:>6.3f}M  R2 {row.r2:.4f}")
    print("\npaired vs forest (bootstrap over rows, 2,000 resamples)")
    for row in paired_frame.itertuples():
        print(
            f"  {row.compared:6s} delta {row.delta_rmse_bdt / M:+.3f}M "
            f"[{row.ci95_low / M:+.3f}, {row.ci95_high / M:+.3f}] "
            f"P(model better)={row.prob_compared_better:.3f}"
        )
    print("\nLasso selection by group")
    for row in by_group.itertuples():
        print(
            f"  {row.group:17s} n={row.n_selected:3d}  "
            f"share of selected {row.share_of_selected:.1%}  share of |coef| {row.share_of_abs_coef:.1%}"
        )
    print(f"\nforest gain vs best linear: {gain:+.2f}%")
    print(f"written {OUTPUT_DIR / 'angle2_models.csv'} and angle2_lasso_groups.csv")


if __name__ == "__main__":
    main()
