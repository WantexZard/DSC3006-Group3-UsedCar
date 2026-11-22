"""Fit ONLY RF/log 0.3: frozen ten-fold OOF, then one revised holdout evaluation.

Public member-a-v1 files are read-only and are never copied into this package.
Existing outputs prevent accidental repeat holdout evaluation. This script does
not train Lasso or RF/price and does not select any parameter using holdout data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from shared_preprocessing import load_development, load_holdout, make_preprocessor

HASHES = {
    "cleaned.csv": "6f4791ce8763b6c73aa3af82be9b3d3cf5065e4cd901737211476ec5b4c5097f",
    "split_manifest.csv": "eae5f790e40e356e2f17c7df4956cf85e02e6a5a7f049a2e40d8ab8ab468a179",
}
RF_CONFIG = dict(n_estimators=500, max_features=0.3, min_samples_leaf=1,
                 bootstrap=True, random_state=42, n_jobs=-1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def make_model():
    return Pipeline([("pre", make_preprocessor(scale_numeric=False)),
                     ("model", RandomForestRegressor(**RF_CONFIG))])


def prediction_frame(meta, y, pred_log, split):
    result = meta[["row_id", "group_id", "cv_fold"]].copy()
    result.insert(0, "evaluation_split", split)
    result.insert(0, "target_scale", "log_price")
    result.insert(0, "model_id", "RF_log_mf03")
    result["y_true"] = np.asarray(y, dtype=float)
    result["y_pred_price"] = np.exp(pred_log)
    result["y_pred_log"] = pred_log
    result["residual_BDT"] = result.y_true - result.y_pred_price
    assert np.isfinite(result[["y_true", "y_pred_price", "y_pred_log"]]).all().all()
    assert result.row_id.is_unique and (result.y_pred_price > 0).all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--package-dir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    data = Path(args.data_dir).resolve()
    root = Path(args.package_dir).resolve()
    output = root / "results" / "rf_log_03"
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        assert sha(data / name) == expected, f"Data hash mismatch: {name}"
    if (output / "completed.json").exists():
        print("Completed predictions already exist; no model was refitted.")
        return
    if any(output.iterdir()):
        raise RuntimeError("Partial run exists. Inspect it before explicitly resuming; do not overwrite holdout results.")
    config = {
        "group": "Group 3", "data_version": "member-a-v1", "data_sha256": HASHES,
        "random_forest": RF_CONFIG, "target": "natural_log(price)",
        "inverse_transform": "exp(pred_log); no smearing or bias correction",
        "preprocessing": "original shared_preprocessing.make_preprocessor(scale_numeric=False)",
        "preprocessing_sha256": sha(Path(__file__).with_name("shared_preprocessing.py")),
        "selection": "max_features=0.3 fixed by group handoff before this run; no tuning",
        "holdout_status": "Revised evaluation on the SAME previously viewed holdout; not a new blind test",
        "run_scope": "Only RF/log 0.3; Lasso and RF/price are not fitted",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    save_json(output / "model_config.json", config)
    X, y, meta, folds = load_development(data)
    assert len(X) == 4789 and len(folds) == 10
    assert (y > 0).all()
    oof_parts, run_log = [], []
    for fold, (train, valid) in enumerate(folds):
        assert set(meta.iloc[train].group_id).isdisjoint(set(meta.iloc[valid].group_id))
        assert (meta.iloc[valid].cv_fold == fold).all()
        pipe = make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit(X.iloc[train], np.log(y.iloc[train].to_numpy()))
        part = prediction_frame(meta.iloc[valid], y.iloc[valid], pipe.predict(X.iloc[valid]), "development_oof")
        part.to_csv(output / f"oof_fold_{fold}.csv", index=False)
        oof_parts.append(part)
        run_log.append({"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                        "encoded_features": int(pipe["pre"].transform(X.iloc[valid]).shape[1]),
                        "warnings": [str(w.message) for w in caught]})
        print(f"RF/log 0.3 completed OOF fold {fold + 1}/10", flush=True)
    oof = pd.concat(oof_parts).sort_values("row_id")
    assert len(oof) == 4789 and oof.row_id.is_unique
    oof.to_csv(output / "oof_predictions.csv", index=False)
    # Full development fit is completed before the single revised holdout access.
    pipe = make_model()
    pipe.fit(X, np.log(y.to_numpy()))
    save_json(output / "frozen_before_holdout.json", {
        **config, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_development_rows": len(X), "encoded_features": int(pipe["pre"].transform(X).shape[1]),
    })
    Xt, yt, mt = load_holdout(data)
    assert len(Xt) == 1208
    assert set(meta.row_id).isdisjoint(set(mt.row_id))
    assert set(meta.group_id).isdisjoint(set(mt.group_id))
    holdout = prediction_frame(mt, yt, pipe.predict(Xt), "holdout_revision")
    holdout.to_csv(output / "holdout_predictions.csv", index=False)
    save_json(output / "run_log.json", run_log)
    save_json(output / "completed.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "oof_rows": len(oof), "holdout_rows": len(holdout),
        "outer_fits": 10, "full_development_fits": 1, "holdout_prediction_calls": 1,
        "lasso_fits": 0, "rf_price_fits": 0,
        "oof_sha256": sha(output / "oof_predictions.csv"),
        "holdout_sha256": sha(output / "holdout_predictions.csv"),
    })
    print("RF/log 0.3 completed. No Lasso or RF/price models were trained.", flush=True)


if __name__ == "__main__":
    main()
