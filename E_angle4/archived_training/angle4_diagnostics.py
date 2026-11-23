"""Angle 4 residual and interpretation diagnostics on development folds."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
from shared_preprocessing import (
    INTERPRETATION_GROUPS,
    feature_mapping,
    load_development,
    make_preprocessor,
)


DATA_DIR = str(PACKAGE_ROOT.parent / "data")
DEV_OUTPUT = PACKAGE_ROOT / "archived_training_outputs" / "development"
OUT_DIR = PACKAGE_ROOT / "archived_training_outputs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X, y, meta, folds = load_development(DATA_DIR)
y_price = y.to_numpy(dtype=float)
y_log = np.log(y_price)
fold_metrics = pd.read_csv(DEV_OUTPUT / "fold_metrics.csv")
predictions = pd.read_csv(DEV_OUTPUT / "predictions.csv")


def group_columns(group_name):
    originals = INTERPRETATION_GROUPS[group_name]
    return list(originals)


groups = list(INTERPRETATION_GROUPS)
lasso_rows = []
rf_rows = []
warning_rows = []

for fold, (train_pos, valid_pos) in enumerate(folds):
    X_train = X.iloc[train_pos]
    X_valid = X.iloc[valid_pos]
    y_train_price = y_price[train_pos]
    y_valid_price = y_price[valid_pos]
    y_train_log = y_log[train_pos]
    y_valid_log = y_log[valid_pos]

    for target_name, y_train, y_valid, inverse, target_index in (
        ("price", y_train_price, y_valid_price, lambda z: z, 0),
        ("log_price", y_train_log, y_valid_log, np.exp, 1),
    ):
        alpha_row = fold_metrics[
            (fold_metrics.fold == fold)
            & (fold_metrics.model == "Lasso")
            & (fold_metrics.target == target_name)
        ].iloc[0]
        alpha = float(alpha_row.alpha)
        lasso_pipe = Pipeline([
            ("pre", make_preprocessor(scale_numeric=True)),
            ("model", Lasso(alpha=alpha, max_iter=100000, tol=1e-4)),
        ])
        with warnings.catch_warnings(record=True) as caught_lasso:
            warnings.simplefilter("always")
            lasso_pipe.fit(X_train, y_train)
        warning_rows.extend({"fold": fold, "model": "Lasso", "target": target_name,
                             "message": str(w.message)} for w in caught_lasso)
        mapping = feature_mapping(lasso_pipe["pre"])
        coefficients = lasso_pipe["model"].coef_
        mapping["coefficient"] = coefficients
        mapping["abs_coefficient"] = np.abs(coefficients)
        for group_name in groups:
            selected = mapping.loc[
                mapping.interpretation_group == group_name, "abs_coefficient"
            ] > 1e-8
            coeffs = mapping.loc[
                mapping.interpretation_group == group_name, "coefficient"
            ]
            lasso_rows.append({
                "fold": fold,
                "target": target_name,
                "group": group_name,
                "n_encoded_features": int(selected.shape[0]),
                "n_nonzero": int(selected.sum()),
                "retained": bool(selected.any()),
                "positive_count": int((coeffs > 1e-8).sum()),
                "negative_count": int((coeffs < -1e-8).sum()),
                "alpha": alpha,
            })

        rf_pipe = Pipeline([
            ("pre", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestRegressor(
                n_estimators=500,
                max_features=1 / 3,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        with warnings.catch_warnings(record=True) as caught_rf:
            warnings.simplefilter("always")
            rf_pipe.fit(X_train, y_train)
        warning_rows.extend({"fold": fold, "model": "RandomForest", "target": target_name,
                             "message": str(w.message)} for w in caught_rf)
        baseline_target = rf_pipe.predict(X_valid)
        baseline_price = np.asarray(inverse(baseline_target), dtype=float)
        base_mse_target = mean_squared_error(y_valid, baseline_target)
        base_mse_price = mean_squared_error(y_valid_price, baseline_price)

        for group_index, group_name in enumerate(groups):
            target_increases = []
            price_increases = []
            columns = group_columns(group_name)
            for repeat in range(10):
                rng = np.random.default_rng(42 + fold * 1000 + target_index * 100 + group_index * 10 + repeat)
                permutation = rng.permutation(len(X_valid))
                X_permuted = X_valid.copy()
                X_permuted.loc[:, columns] = X_valid.iloc[permutation][columns].to_numpy()
                perm_target = rf_pipe.predict(X_permuted)
                perm_price = np.asarray(inverse(perm_target), dtype=float)
                target_increases.append(
                    mean_squared_error(y_valid, perm_target) - base_mse_target
                )
                price_increases.append(
                    mean_squared_error(y_valid_price, perm_price) - base_mse_price
                )
            rf_rows.append({
                "fold": fold,
                "target": target_name,
                "group": group_name,
                "mse_increase_target_mean": float(np.mean(target_increases)),
                "mse_increase_target_sd": float(np.std(target_increases, ddof=1)),
                "mse_increase_price_BDT2_mean": float(np.mean(price_increases)),
                "mse_increase_price_BDT2_sd": float(np.std(price_increases, ddof=1)),
                "n_repeats": 10,
            })

lasso_df = pd.DataFrame(lasso_rows)
rf_df = pd.DataFrame(rf_rows)
lasso_summary = lasso_df.groupby(["target", "group"], as_index=False).agg(
    selection_frequency=("retained", "mean"),
    mean_nonzero=("n_nonzero", "mean"),
    folds=("retained", "size"),
    positive_count_total=("positive_count", "sum"),
    negative_count_total=("negative_count", "sum"),
)
rf_summary = rf_df.groupby(["target", "group"], as_index=False).agg(
    mean_mse_increase_target=("mse_increase_target_mean", "mean"),
    sd_fold_mse_increase_target=("mse_increase_target_mean", "std"),
    mean_mse_increase_price_BDT2=("mse_increase_price_BDT2_mean", "mean"),
    sd_fold_mse_increase_price_BDT2=("mse_increase_price_BDT2_mean", "std"),
    folds=("group", "size"),
)

# Residual summaries using the already saved OOF predictions.
dev_true = predictions["y_true_price_BDT"]
price_bins = pd.qcut(dev_true, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
predictions = predictions.copy()
predictions["price_bin"] = price_bins.to_numpy()
residual_rows = []
for (model, target), group in predictions.groupby(["model", "target"], sort=True):
    residual_rows.append({
        "model": model,
        "target": target,
        "n": len(group),
        "rmse_BDT": float(np.sqrt(np.mean(group.residual_price_BDT ** 2))),
        "mae_BDT": float(np.mean(np.abs(group.residual_price_BDT))),
        "mean_residual_BDT": float(group.residual_price_BDT.mean()),
        "sd_residual_BDT": float(group.residual_price_BDT.std(ddof=1)),
        "negative_prediction_rate": float((group.y_pred_price_BDT < 0).mean()),
    })
residual_summary = pd.DataFrame(residual_rows)
bin_rows = []
for (model, target, price_bin), group in predictions.groupby(["model", "target", "price_bin"], observed=True):
    bin_rows.append({
        "model": model,
        "target": target,
        "price_bin": str(price_bin),
        "n": len(group),
        "rmse_BDT": float(np.sqrt(np.mean(group.residual_price_BDT ** 2))),
        "mae_BDT": float(np.mean(np.abs(group.residual_price_BDT))),
        "mean_residual_BDT": float(group.residual_price_BDT.mean()),
    })
bin_summary = pd.DataFrame(bin_rows)

lasso_df.to_csv(OUT_DIR / "lasso_group_selection_by_fold.csv", index=False)
lasso_summary.to_csv(OUT_DIR / "lasso_group_selection_summary.csv", index=False)
rf_df.to_csv(OUT_DIR / "rf_group_permutation_by_fold.csv", index=False)
rf_summary.to_csv(OUT_DIR / "rf_group_permutation_summary.csv", index=False)
residual_summary.to_csv(OUT_DIR / "residual_summary.csv", index=False)
bin_summary.to_csv(OUT_DIR / "residual_by_price_bin.csv", index=False)
(OUT_DIR / "warnings.json").write_text(json.dumps(warning_rows, ensure_ascii=False, indent=2) + "\n")

# Residual plots.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
for ax, (model, target) in zip(axes.ravel(), [("Lasso", "price"), ("Lasso", "log_price"),
                                               ("RandomForest", "price"), ("RandomForest", "log_price")]):
    group = predictions[(predictions.model == model) & (predictions.target == target)]
    ax.scatter(group.y_pred_price_BDT / 1e6, group.residual_price_BDT / 1e6,
               s=8, alpha=.25, color="#23658A", edgecolors="none")
    ax.axhline(0, color="#B65E28", linewidth=1)
    ax.set_title(f"{model} — {target}")
    ax.set_xlabel("Predicted price (million BDT)")
    ax.set_ylabel("Residual (million BDT)")
    ax.grid(alpha=.18)
fig.suptitle("Development OOF residuals on the original price scale", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(OUT_DIR / "residual_vs_fitted.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# Error by true-price quartile.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
order = ["Q1_low", "Q2", "Q3", "Q4_high"]
labels = []
for model, target in [("Lasso", "price"), ("Lasso", "log_price"),
                      ("RandomForest", "price"), ("RandomForest", "log_price")]:
    sub = bin_summary[(bin_summary.model == model) & (bin_summary.target == target)].set_index("price_bin").reindex(order)
    labels.append(f"{model}\n{target}")
    axes[0].plot(order, sub.rmse_BDT / 1e6, marker="o", label=f"{model} / {target}")
    axes[1].plot(order, sub.mae_BDT / 1e6, marker="o", label=f"{model} / {target}")
axes[0].set_title("RMSE by true-price quartile")
axes[1].set_title("MAE by true-price quartile")
for ax in axes:
    ax.set_xlabel("True price quartile")
    ax.set_ylabel("Error (million BDT)")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
fig.suptitle("Error concentration across the price distribution", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(OUT_DIR / "error_by_price_quartile.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# Lasso retention frequency.
pivot = lasso_summary.pivot(index="group", columns="target", values="selection_frequency").reindex(groups)
fig, ax = plt.subplots(figsize=(8, 4.8))
im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center",
                color="white" if pivot.iloc[i, j] > .55 else "black")
fig.colorbar(im, ax=ax, label="Selection frequency across 10 folds")
ax.set_title("Lasso group-retention stability")
fig.tight_layout()
fig.savefig(OUT_DIR / "lasso_group_stability.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# RF grouped permutation importance.
rf_pivot = rf_summary.pivot(index="group", columns="target", values="mean_mse_increase_target").reindex(groups)
rf_pivot = rf_pivot / rf_pivot.max(axis=0).replace(0, np.nan)
fig, ax = plt.subplots(figsize=(8, 4.8))
im = ax.imshow(rf_pivot.to_numpy(), vmin=0, vmax=1, cmap="Oranges", aspect="auto")
ax.set_xticks(range(len(rf_pivot.columns)), rf_pivot.columns)
ax.set_yticks(range(len(rf_pivot.index)), rf_pivot.index)
for i in range(rf_pivot.shape[0]):
    for j in range(rf_pivot.shape[1]):
        value = rf_pivot.iloc[i, j]
        ax.text(j, i, "NA" if pd.isna(value) else f"{value:.2f}", ha="center", va="center",
                color="white" if value > .55 else "black")
fig.colorbar(im, ax=ax, label="Relative mean MSE increase within target scale")
ax.set_title("Random Forest grouped permutation importance")
fig.tight_layout()
fig.savefig(OUT_DIR / "rf_group_importance.png", dpi=250, bbox_inches="tight")
plt.close(fig)

metadata = {
    "status": "development_diagnostics_only",
    "n_outer_folds": 10,
    "permutation_repeats_per_group_fold": 10,
    "groups": groups,
    "warnings": len(warning_rows),
    "note": "Permutation importance is evaluated on validation folds; target-scale MSE increases are used for ranking, while original-price MSE increases are saved as a secondary view.",
}
(OUT_DIR / "diagnostic_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "PASS", "lasso_rows": len(lasso_df), "rf_rows": len(rf_df), "warnings": len(warning_rows), "out": str(OUT_DIR)}, ensure_ascii=False, indent=2))
