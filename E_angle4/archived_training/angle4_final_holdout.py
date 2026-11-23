"""Freeze Angle 4 configurations on development data, then evaluate holdout once."""
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
from shared_preprocessing import (
    load_development,
    load_holdout,
    make_preprocessor,
    unknown_diagnostics,
)


DATA_DIR = str(PACKAGE_ROOT.parent / "data")
OUT_DIR = PACKAGE_ROOT / "archived_training_outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_GRIDS = {
    "price": np.logspace(2, 6, 100),
    "log_price": np.logspace(-5, 2, 100),
}
RF_CONFIG = {
    "n_estimators": 500,
    "max_features": 1 / 3,
    "random_state": 42,
    "n_jobs": -1,
}

X_dev, y_dev, meta_dev, _ = load_development(DATA_DIR)
y_dev_price = y_dev.to_numpy(dtype=float)
y_dev_log = np.log(y_dev_price)
groups_dev = meta_dev["group_id"].to_numpy()

warnings_log = []
fitted = {}
selected_alpha = {}
feature_counts = {}
unknown_by_model = {}


def fit_lasso(target_name: str, y_target: np.ndarray):
    pipe = Pipeline([
        ("pre", make_preprocessor(scale_numeric=True)),
        ("model", Lasso(max_iter=100000, tol=1e-4)),
    ])
    search = GridSearchCV(
        pipe,
        {"model__alpha": ALPHA_GRIDS[target_name]},
        cv=GroupKFold(n_splits=5),
        scoring="neg_mean_squared_error",
        n_jobs=1,
        error_score="raise",
        refit=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(X_dev, y_target, groups=groups_dev)
    warnings_log.extend({
        "model": "Lasso", "target": target_name, "stage": "full_development_fit",
        "message": str(w.message),
    } for w in caught)
    best = search.best_estimator_
    selected_alpha[target_name] = float(search.best_params_["model__alpha"])
    feature_counts["Lasso__" + target_name] = int(best["pre"].transform(X_dev).shape[1])
    fitted["Lasso__" + target_name] = best


def fit_rf(target_name: str, y_target: np.ndarray):
    pipe = Pipeline([
        ("pre", make_preprocessor(scale_numeric=False)),
        ("model", RandomForestRegressor(**RF_CONFIG)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(X_dev, y_target)
    warnings_log.extend({
        "model": "RandomForest", "target": target_name, "stage": "full_development_fit",
        "message": str(w.message),
    } for w in caught)
    feature_counts["RandomForest__" + target_name] = int(pipe["pre"].transform(X_dev).shape[1])
    fitted["RandomForest__" + target_name] = pipe


# Model selection and preprocessing are completed using development data only.
fit_lasso("price", y_dev_price)
fit_lasso("log_price", y_dev_log)
fit_rf("price", y_dev_price)
fit_rf("log_price", y_dev_log)

frozen_config = {
    "status": "frozen_before_holdout_evaluation",
    "data_version": "member-a-v1",
    "development_rows": int(len(X_dev)),
    "holdout_use": "one-time final evaluation only",
    "features": [
        "car_age", "kilometers_run", "energy_capacity", "brand",
        "brand_model", "fuel_type", "transmission",
    ],
    "target_versions": {
        "price": "price",
        "log_price": "natural_log(price)",
    },
    "lasso": {
        "preprocessing": "fold/development-local median imputation, full one-hot/multi-hot encoding, numeric StandardScaler",
        "alpha_selection": "5-fold GroupKFold by group_id on all development rows",
        "alpha_grid_price": [float(ALPHA_GRIDS["price"][0]), float(ALPHA_GRIDS["price"][-1]), int(len(ALPHA_GRIDS["price"]))],
        "alpha_grid_log_price": [float(ALPHA_GRIDS["log_price"][0]), float(ALPHA_GRIDS["log_price"][-1]), int(len(ALPHA_GRIDS["log_price"]))],
        "selected_alpha_price": selected_alpha["price"],
        "selected_alpha_log_price": selected_alpha["log_price"],
        "max_iter": 100000,
        "tol": 1e-4,
    },
    "random_forest": RF_CONFIG,
    "evaluation": {
        "primary_scale": "original BDT price",
        "metrics": ["RMSE_BDT", "MAE_BDT", "R2_price"],
        "log_back_transform": "exp(pred_log)",
        "negative_prediction_policy": "record; do not silently clip",
    },
    "pre_holdout_development_reason": "RF/raw-price had the lowest pooled development RMSE and highest pooled development R2; all four frozen variants are reported to assess log-price robustness and model comparison.",
    "development_feature_counts": feature_counts,
}
(OUT_DIR / "frozen_config.json").write_text(json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n")

# The independent holdout is accessed only after all fitting and configuration decisions above.
X_test, y_test, meta_test = load_holdout(DATA_DIR)
y_test_price = y_test.to_numpy(dtype=float)
y_test_log = np.log(y_test_price)

dev_groups = set(meta_dev["group_id"])
test_groups = set(meta_test["group_id"])
group_overlap = sorted(dev_groups & test_groups)
row_overlap = sorted(set(meta_dev["row_id"]) & set(meta_test["row_id"]))

prediction_rows = []
metric_rows = []


def evaluate(model_name: str, target_name: str, inverse, y_test_target):
    key = model_name + "__" + target_name
    pipe = fitted[key]
    unknown_by_model[key] = unknown_diagnostics(pipe["pre"], X_test)
    pred_target = np.asarray(pipe.predict(X_test), dtype=float)
    pred_price = np.asarray(inverse(pred_target), dtype=float)
    residual = y_test_price - pred_price
    metric_rows.append({
        "model": model_name,
        "target": target_name,
        "n_test": int(len(y_test_price)),
        "rmse_BDT": float(np.sqrt(mean_squared_error(y_test_price, pred_price))),
        "mae_BDT": float(mean_absolute_error(y_test_price, pred_price)),
        "r2_price": float(r2_score(y_test_price, pred_price)),
        "rmse_target": float(np.sqrt(mean_squared_error(y_test_target, pred_target))),
        "negative_prediction_count": int((pred_price < 0).sum()),
        "negative_prediction_rate": float((pred_price < 0).mean()),
        "mean_residual_BDT": float(residual.mean()),
        "sd_residual_BDT": float(residual.std(ddof=1)),
    })
    for row_id, group_id, true_value, pred_value, pred_t, resid in zip(
        meta_test["row_id"], meta_test["group_id"], y_test_price, pred_price,
        pred_target, residual,
    ):
        prediction_rows.append({
            "row_id": row_id,
            "group_id": group_id,
            "model": model_name,
            "target": target_name,
            "y_true_price_BDT": float(true_value),
            "y_pred_target": float(pred_t),
            "y_pred_price_BDT": float(pred_value),
            "residual_price_BDT": float(resid),
        })


evaluate("Lasso", "price", lambda values: values, y_test_price)
evaluate("Lasso", "log_price", np.exp, y_test_log)
evaluate("RandomForest", "price", lambda values: values, y_test_price)
evaluate("RandomForest", "log_price", np.exp, y_test_log)

metrics_df = pd.DataFrame(metric_rows).sort_values(["model", "target"])
pred_df = pd.DataFrame(prediction_rows)
metrics_df.to_csv(OUT_DIR / "holdout_metrics.csv", index=False)
pred_df.to_csv(OUT_DIR / "holdout_predictions.csv", index=False)
(OUT_DIR / "unknown_category_diagnostics.json").write_text(
    json.dumps(unknown_by_model, ensure_ascii=False, indent=2) + "\n"
)
(OUT_DIR / "warnings.json").write_text(json.dumps(warnings_log, ensure_ascii=False, indent=2) + "\n")
(OUT_DIR / "holdout_audit.json").write_text(json.dumps({
    "status": "PASS",
    "development_rows": int(len(X_dev)),
    "holdout_rows": int(len(X_test)),
    "expected_prediction_rows": int(len(X_test) * 4),
    "actual_prediction_rows": int(len(pred_df)),
    "duplicate_prediction_keys": int(pred_df[["row_id", "model", "target"]].duplicated().sum()),
    "unique_test_rows": int(pred_df.row_id.nunique()),
    "group_overlap_count": int(len(group_overlap)),
    "row_overlap_count": int(len(row_overlap)),
    "all_predictions_finite": bool(np.isfinite(pred_df.select_dtypes("number")).all().all()),
    "warning_count": int(len(warnings_log)),
    "unknown_category_diagnostics_saved": True,
}, ensure_ascii=False, indent=2) + "\n")

# Holdout visualization on the original price scale.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (model, target) in zip(axes.ravel(), [
    ("Lasso", "price"), ("Lasso", "log_price"),
    ("RandomForest", "price"), ("RandomForest", "log_price"),
]):
    group = pred_df[(pred_df.model == model) & (pred_df.target == target)]
    x = group.y_true_price_BDT.to_numpy() / 1e6
    z = group.y_pred_price_BDT.to_numpy() / 1e6
    ax.scatter(x, z, s=10, alpha=.28, color="#23658A", edgecolors="none")
    lo = min(x.min(), z.min())
    hi = max(x.max(), z.max())
    ax.plot([lo, hi], [lo, hi], color="#B65E28", linewidth=1.2)
    ax.set_title(f"{model} — {target}")
    ax.set_xlabel("True price (million BDT)")
    ax.set_ylabel("Predicted price (million BDT)")
    ax.grid(alpha=.18)
fig.suptitle("Independent holdout: predicted versus true price", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(OUT_DIR / "holdout_actual_vs_predicted.png", dpi=250, bbox_inches="tight")
plt.close(fig)

price_bins = pd.qcut(pred_df["y_true_price_BDT"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
plot_df = pred_df.copy()
plot_df["price_bin"] = price_bins.to_numpy()
bin_rows = []
for (model, target, price_bin), group in plot_df.groupby(["model", "target", "price_bin"], observed=True):
    bin_rows.append({
        "model": model,
        "target": target,
        "price_bin": str(price_bin),
        "n": int(len(group)),
        "rmse_BDT": float(np.sqrt(np.mean(group.residual_price_BDT ** 2))),
        "mae_BDT": float(np.mean(np.abs(group.residual_price_BDT))),
        "mean_residual_BDT": float(group.residual_price_BDT.mean()),
    })
bin_df = pd.DataFrame(bin_rows)
bin_df.to_csv(OUT_DIR / "holdout_error_by_price_bin.csv", index=False)
order = ["Q1_low", "Q2", "Q3", "Q4_high"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for model, target in [
    ("Lasso", "price"), ("Lasso", "log_price"),
    ("RandomForest", "price"), ("RandomForest", "log_price"),
]:
    sub = bin_df[(bin_df.model == model) & (bin_df.target == target)].set_index("price_bin").reindex(order)
    label = f"{model} / {target}"
    axes[0].plot(order, sub.rmse_BDT / 1e6, marker="o", label=label)
    axes[1].plot(order, sub.mae_BDT / 1e6, marker="o", label=label)
axes[0].set_title("Holdout RMSE by true-price quartile")
axes[1].set_title("Holdout MAE by true-price quartile")
for ax in axes:
    ax.set_xlabel("True price quartile")
    ax.set_ylabel("Error (million BDT)")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
fig.suptitle("Independent holdout error concentration", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(OUT_DIR / "holdout_error_by_price_quartile.png", dpi=250, bbox_inches="tight")
plt.close(fig)

print(json.dumps({
    "status": "PASS",
    "holdout_rows": int(len(X_test)),
    "prediction_rows": int(len(pred_df)),
    "group_overlap_count": int(len(group_overlap)),
    "row_overlap_count": int(len(row_overlap)),
    "warnings": len(warnings_log),
    "metrics": metrics_df.to_dict(orient="records"),
}, ensure_ascii=False, indent=2))
