"""Fit the frozen configuration on the whole development split, then evaluate the
held-out test split exactly once.

Run this only after tune_rf.py and evaluate_rf.py have finished: the test split is
the one resource that cannot be re-used, so everything inspectable is decided
before this script touches it.

Outputs (C_angle2_forest/outputs/)
---------------------------
model_config.json     full provenance: data hashes, features, preprocessing,
                      hyperparameters, search space, selection rule, seed, versions
test_predictions.csv  row-level test predictions (row_id, y_true, y_pred_price)
metrics.csv           appended with the in-sample and the one-shot test rows
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version

from common import (
    OUTPUT_DIR,
    SEED,
    coerce_max_features,
    data_fingerprint,
    ensure_dirs,
    load_dev,
    load_test_split,
    make_preprocessor,
    make_rf,
    metric_block,
    resolved_max_features,
    write_json,
)

SELECTION_PATH = OUTPUT_DIR / "rf_selection_main.json"


def software_versions() -> dict[str, str]:
    import matplotlib
    import scipy

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main() -> None:
    ensure_dirs()
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    config = payload["best_config"]
    config["max_depth"] = None if config["max_depth"] in (None, "None") else int(config["max_depth"])
    config["max_features"] = coerce_max_features(config["max_features"])
    model_id = payload["best_model_id"]

    started = time.time()
    X_dev, y_dev, dev_meta, folds = load_dev()
    pre = make_preprocessor(scale_numeric=False)
    X_dev_encoded = pre.fit_transform(X_dev)
    rf = make_rf(max_features=config["max_features"], min_samples_leaf=config["min_samples_leaf"])
    rf.fit(X_dev_encoded, y_dev)

    dev_pred = rf.predict(X_dev_encoded)
    dev_block = metric_block(y_dev, dev_pred)

    # The single test evaluation.
    X_test, y_test, test_meta = load_test_split()
    X_test_encoded = pre.transform(X_test)
    test_pred = rf.predict(X_test_encoded)
    test_block = metric_block(y_test, test_pred)

    pd.DataFrame(
        {
            "model_id": model_id,
            "target_scale": "price",
            "row_id": test_meta["row_id"].to_numpy(),
            "group_id": test_meta["group_id"].to_numpy(),
            "y_true": np.asarray(y_test, dtype=float),
            "y_pred_price": test_pred,
        }
    ).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # One metrics file for every split: development pooled/mean-of-fold from
    # evaluate_rf.py, plus the in-sample fit and the one-shot test rows here.
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[~metrics["evaluation_split"].isin(["test", "development_in_sample"])]
    extra = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "evaluation_split": split,
                "aggregation": "single_fit",
                "n": block["n"],
                "RMSE_BDT": block["rmse"],
                "MAE_BDT": block["mae"],
                "R2": block["r2"],
            }
            for split, block in (("development_in_sample", dev_block), ("test", test_block))
        ]
    )
    pd.concat([metrics, extra], ignore_index=True).to_csv(metrics_path, index=False)

    model_config = {
        "model_id": model_id,
        "owner": "C_angle2_forest",
        "angle": "Angle 2 - Random Forest side",
        "data": {
            "package": "member-a-v1",
            "path": "data",
            "sha256": data_fingerprint(),
            "features": [
                "car_age",
                "kilometers_run",
                "energy_capacity",
                "brand",
                "brand_model",
                "fuel_type",
                "transmission",
            ],
            "n_development": int(len(X_dev)),
            "n_test": int(len(X_test)),
            "cv_folds": len(folds),
        },
        "preprocessing": {
            "numeric": "median imputation; no scaling (RF is scale invariant)",
            "categorical": 'one-hot, drop=None, handle_unknown="ignore"',
            "fuel": "multi-label one-hot over the sorted label set",
            "fitted_in": "training fold only; test transformed by the development fit",
            "n_encoded_features": int(X_dev_encoded.shape[1]),
        },
        "hyperparameters": {
            "n_estimators": int(config["n_estimators"]),
            "max_features": config["max_features"],
            "max_features_resolved": resolved_max_features(
                config["max_features"], X_dev_encoded.shape[1]
            ),
            "min_samples_leaf": int(config["min_samples_leaf"]),
            "max_depth": config["max_depth"],
            "bootstrap": bool(config["bootstrap"]),
            "random_state": SEED,
            "n_jobs": 1,
        },
        "search_space": {
            "n_estimators": [500],
            "max_features": ["sqrt", 0.3, 0.5, 1.0],
            "min_samples_leaf": [1, 2, 5],
            "n_configs": payload["n_configs"],
        },
        "selection": {
            "criterion": "pooled out-of-fold RMSE on the 10 development folds",
            "not_used": "mean of per-fold RMSE, and never the test split",
            "selected_config_is": "selected on CV, so its OOF score is optimistic",
            "tuning_table": "outputs/rf_tuning_main.csv",
        },
        "evaluation": {
            "dev_pooled_oof": payload["best_pooled_oof"],
            "dev_in_sample": dev_block,
            "test": test_block,
        },
        "software": software_versions(),
        "timestamps": {"frozen_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_seconds": None},
        "reproduce": "python -B code/tune_rf.py && python -B code/evaluate_rf.py && python -B code/finalize_rf.py",
    }
    model_config["timestamps"]["wall_seconds"] = round(time.time() - started, 1)
    write_json(OUTPUT_DIR / "model_config.json", model_config)

    print(f"development in-sample RMSE {dev_block['rmse']:,.0f} BDT (context only)")
    print(f"pooled OOF RMSE           {payload['best_pooled_oof']['rmse']:,.0f} BDT")
    print(f"TEST RMSE                 {test_block['rmse']:,.0f} BDT")
    print(f"TEST MAE                  {test_block['mae']:,.0f} BDT | R2 {test_block['r2']:.4f} | n={test_block['n']}")
    print("model_config.json written")


if __name__ == "__main__":
    main()
