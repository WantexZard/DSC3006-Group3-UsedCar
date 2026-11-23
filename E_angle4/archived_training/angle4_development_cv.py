"""Provisional Angle 4 development experiment.

Runs grouped outer 10-fold evaluation for Lasso and Random Forest on the raw
and natural-log price targets. This is a development result, not the final
held-out test evaluation.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

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
from shared_preprocessing import load_development, make_preprocessor


DATA_DIR = str(PACKAGE_ROOT.parent / "data")
OUT_DIR = PACKAGE_ROOT / "archived_training_outputs" / "development"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X, y, meta, folds = load_development(DATA_DIR)
y_price = y.to_numpy(dtype=float)
y_log = np.log(y_price)

alpha_grids = {
    "price": np.logspace(2, 6, 100),
    "log_price": np.logspace(-5, 2, 100),
}

rows = []
predictions = []
warnings_log = []


def metrics(y_true_price, pred_price, y_true_target, pred_target):
    return {
        "rmse_price_BDT": float(mean_squared_error(y_true_price, pred_price) ** 0.5),
        "mae_price_BDT": float(mean_absolute_error(y_true_price, pred_price)),
        "r2_price": float(r2_score(y_true_price, pred_price)),
        "rmse_target": float(mean_squared_error(y_true_target, pred_target) ** 0.5),
        "r2_target": float(r2_score(y_true_target, pred_target)),
        "negative_prediction_count": int((pred_price < 0).sum()),
        "prediction_count": int(len(pred_price)),
    }


for fold, (train_pos, valid_pos) in enumerate(folds):
    X_train = X.iloc[train_pos]
    X_valid = X.iloc[valid_pos]
    groups_train = meta.iloc[train_pos]["group_id"].to_numpy()
    y_train_price = y_price[train_pos]
    y_valid_price = y_price[valid_pos]
    y_train_log = y_log[train_pos]
    y_valid_log = y_log[valid_pos]

    inner_cv = GroupKFold(n_splits=5)

    for target_name, y_train, y_valid, inverse in (
        ("price", y_train_price, y_valid_price, lambda values: values),
        ("log_price", y_train_log, y_valid_log, np.exp),
    ):
        lasso_pipe = Pipeline([
            ("pre", make_preprocessor(scale_numeric=True)),
            ("model", Lasso(max_iter=100000, tol=1e-4)),
        ])
        search = GridSearchCV(
            lasso_pipe,
            {"model__alpha": alpha_grids[target_name]},
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            error_score="raise",
            refit=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X_train, y_train, groups=groups_train)
        pred_target = search.predict(X_valid)
        pred_price = np.asarray(inverse(pred_target), dtype=float)
        m = metrics(y_valid_price, pred_price, y_valid, pred_target)
        rows.append({
            "fold": fold,
            "model": "Lasso",
            "target": target_name,
            "alpha": float(search.best_params_["model__alpha"]),
            "n_features_after_preprocessing": int(search.best_estimator_["pre"].transform(X_valid).shape[1]),
            "warnings": len(caught),
            **m,
        })
        warnings_log.extend({"fold": fold, "model": "Lasso", "target": target_name,
                             "message": str(w.message)} for w in caught)
        for row_id, group_id, true_value, pred_value, pred_t in zip(
            meta.iloc[valid_pos]["row_id"], meta.iloc[valid_pos]["group_id"],
            y_valid_price, pred_price, pred_target,
        ):
            predictions.append({
                "row_id": row_id,
                "group_id": group_id,
                "cv_fold": fold,
                "model": "Lasso",
                "target": target_name,
                "y_true_price_BDT": float(true_value),
                "y_pred_target": float(pred_t),
                "y_pred_price_BDT": float(pred_value),
                "residual_price_BDT": float(true_value - pred_value),
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
        pred_target_rf = rf_pipe.predict(X_valid)
        pred_price_rf = np.asarray(inverse(pred_target_rf), dtype=float)
        m_rf = metrics(y_valid_price, pred_price_rf, y_valid, pred_target_rf)
        rows.append({
            "fold": fold,
            "model": "RandomForest",
            "target": target_name,
            "alpha": np.nan,
            "n_features_after_preprocessing": int(rf_pipe["pre"].transform(X_valid).shape[1]),
            "warnings": len(caught_rf),
            **m_rf,
        })
        warnings_log.extend({"fold": fold, "model": "RandomForest", "target": target_name,
                             "message": str(w.message)} for w in caught_rf)
        for row_id, group_id, true_value, pred_value, pred_t in zip(
            meta.iloc[valid_pos]["row_id"], meta.iloc[valid_pos]["group_id"],
            y_valid_price, pred_price_rf, pred_target_rf,
        ):
            predictions.append({
                "row_id": row_id,
                "group_id": group_id,
                "cv_fold": fold,
                "model": "RandomForest",
                "target": target_name,
                "y_true_price_BDT": float(true_value),
                "y_pred_target": float(pred_t),
                "y_pred_price_BDT": float(pred_value),
                "residual_price_BDT": float(true_value - pred_value),
            })

    print(f"completed fold {fold + 1}/10", flush=True)

fold_metrics = pd.DataFrame(rows).sort_values(["model", "target", "fold"])
pred_df = pd.DataFrame(predictions)

summary_rows = []
for (model, target), group in fold_metrics.groupby(["model", "target"], sort=True):
    pred_group = pred_df[(pred_df["model"] == model) & (pred_df["target"] == target)]
    summary_rows.append({
        "model": model,
        "target": target,
        "folds": int(len(group)),
        "mean_fold_rmse_price_BDT": float(group["rmse_price_BDT"].mean()),
        "sd_fold_rmse_price_BDT": float(group["rmse_price_BDT"].std(ddof=1)),
        "mean_fold_mae_price_BDT": float(group["mae_price_BDT"].mean()),
        "sd_fold_mae_price_BDT": float(group["mae_price_BDT"].std(ddof=1)),
        "mean_fold_r2_price": float(group["r2_price"].mean()),
        "sd_fold_r2_price": float(group["r2_price"].std(ddof=1)),
        "pooled_oof_rmse_price_BDT": float(mean_squared_error(pred_group["y_true_price_BDT"], pred_group["y_pred_price_BDT"]) ** 0.5),
        "pooled_oof_mae_price_BDT": float(mean_absolute_error(pred_group["y_true_price_BDT"], pred_group["y_pred_price_BDT"])),
        "pooled_oof_r2_price": float(r2_score(pred_group["y_true_price_BDT"], pred_group["y_pred_price_BDT"])),
        "total_negative_predictions": int((pred_group["y_pred_price_BDT"] < 0).sum()),
        "negative_prediction_rate": float((pred_group["y_pred_price_BDT"] < 0).mean()),
    })

summary = pd.DataFrame(summary_rows).sort_values(["model", "target"])
fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)
(OUT_DIR / "warnings.json").write_text(json.dumps(warnings_log, ensure_ascii=False, indent=2) + "\n")
(OUT_DIR / "config_used.json").write_text(json.dumps({
    "data_dir": DATA_DIR,
    "development_rows": int(len(X)),
    "outer_folds": 10,
    "inner_cv": "5-fold GroupKFold by group_id for Lasso alpha",
    "lasso_alpha_grid": {key: [float(values[0]), float(values[-1]), int(len(values))] for key, values in alpha_grids.items()},
    "lasso_max_iter": 100000,
    "lasso_tol": 1e-4,
    "rf_n_estimators": 500,
    "rf_max_features": 1 / 3,
    "rf_random_state": 42,
    "rf_n_jobs": -1,
    "primary_metrics": ["RMSE_BDT", "MAE_BDT", "R2_price"],
    "log_back_transform": "exp(pred_log)",
    "status": "provisional_development_experiment_not_final_holdout",
}, ensure_ascii=False, indent=2) + "\n")
print(summary.to_string(index=False))
print(f"saved outputs in {OUT_DIR}")
