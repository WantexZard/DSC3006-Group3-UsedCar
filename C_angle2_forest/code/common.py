"""Shared utilities for C_angle2_forest's Angle 2 experiments (Random Forest side).

All data access goes through A_data's frozen interface (`shared_preprocessing`),
so the split, folds and preprocessing rules cannot drift from the published
`member-a-v1` package. Nothing here may touch the held-out test split except
`load_test_split`, which only `finalize_rf.py` calls after the configuration is
frozen.

Conventions
-----------
- Positions in `folds` are positional row indices into X and y (as returned by
  A_data's loader); we always index positionally, never by label.
- Metrics are pooled: one RMSE over all out-of-fold predictions, not the mean of
  per-fold RMSEs.
- Permutation is applied to *raw* columns and the preprocessor is re-applied, so
  multi-hot encodings stay internally consistent.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

CODE_DIR = Path(__file__).resolve().parent
MEMBER_C_DIR = CODE_DIR.parent
PROJECT_DIR = MEMBER_C_DIR.parent
# Data is read from the shared member-a-v1 directory at the archive root.
# Set MEMBER_A_DATA to point elsewhere if needed.
DATA_DIR = Path(os.environ.get("MEMBER_A_DATA", PROJECT_DIR / "data"))
OUTPUT_DIR = MEMBER_C_DIR / "outputs"
FIGURE_DIR = MEMBER_C_DIR / "figures"

SEED = 42

# Raw input column(s) belonging to each interpretation group. Derived from the
# interpretation groups declared in A_data's technical note, section 4.
INTERPRETATION_COLUMNS: dict[str, tuple[str, ...]] = {
    "age": ("car_age",),
    "mileage": ("kilometers_run",),
    "capacity": ("energy_capacity",),
    "vehicle_identity": ("brand", "brand_model"),
    "fuel": ("fuel_type",),
    "transmission": ("transmission",),
}

sys.path.insert(0, str(CODE_DIR))

from shared_preprocessing import (  # noqa: E402
    feature_mapping,
    load_development,
    load_holdout,
    make_preprocessor,
    unknown_diagnostics,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dev() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Return (X, y, metadata, folds) for the 4,789 development records only."""
    X, y, metadata, folds = load_development(DATA_DIR)
    return X, np.asarray(y, dtype=float), metadata, folds


def load_test_split() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return the held-out test split. Call once, after the model is frozen."""
    return load_holdout(DATA_DIR)


def data_fingerprint() -> dict[str, str]:
    """SHA-256 of the two published files, for model_config.json provenance."""
    out = {}
    for name in ("cleaned.csv", "split_manifest.csv"):
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        out[name] = digest
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / R2 for one set of predictions, in original BDT units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "rmse": _rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - ss_res / ss_tot,
    }


def fold_metric_rows(
    model_id: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """Per-fold metrics using the same positional fold definition as training."""
    rows = []
    for fold_id, (_, valid_pos) in enumerate(folds):
        block = metric_block(y_true[valid_pos], y_pred[valid_pos])
        rows.append({"model_id": model_id, "evaluation_split": split, "cv_fold": fold_id, **block})
    return rows


def summary_rows(
    model_id: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """Two aggregation rows: pooled (the headline number) and mean over folds."""
    pooled = metric_block(y_true, y_pred)
    fold_rows = fold_metric_rows(model_id, split, y_true, y_pred, folds)
    per_fold = np.array([r["rmse"] for r in fold_rows], dtype=float)
    return [
        {"model_id": model_id, "evaluation_split": split, "aggregation": "pooled_oof", **pooled},
        {
            "model_id": model_id,
            "evaluation_split": split,
            "aggregation": "mean_of_fold_rmse",
            "n": int(per_fold.size),
            "rmse": float(per_fold.mean()),
            "mae": float(np.mean([r["mae"] for r in fold_rows])),
            "r2": float(np.mean([r["r2"] for r in fold_rows])),
            "rmse_fold_sd": float(per_fold.std(ddof=1)),
        },
    ]


# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
N_ESTIMATORS = 500

DEFAULT_RF: dict = {
    "n_estimators": N_ESTIMATORS,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "max_depth": None,
    "bootstrap": True,
    "random_state": SEED,
}


def make_rf(**overrides) -> RandomForestRegressor:
    """Deterministic single-threaded RF.

    n_jobs stays at 1: joblib worker pools need named pipes, which this machine's
    sandbox blocks (PermissionError [WinError 5]). Single-threaded 500-tree fits
    take a few seconds on 4,310 x ~203, which is acceptable for 120 fits.
    """
    params = {**DEFAULT_RF, **overrides}
    params["n_jobs"] = 1
    return RandomForestRegressor(**params)


def model_id_for(params: dict) -> str:
    mf = params["max_features"]
    mf_tag = f"{mf:g}" if isinstance(mf, float) else str(mf)
    return f"rf__mf-{mf_tag}__msl-{params['min_samples_leaf']}"


def coerce_max_features(value):
    """Accept the string form that older JSON outputs used ("0.3") as a typed value."""
    if isinstance(value, (int, float)):
        return value
    if value in ("sqrt", "log2", "None", None):
        return value
    number = float(value)
    return int(number) if number.is_integer() and number >= 1 else number


def resolved_max_features(value, n_features: int) -> int:
    """How many columns max_features actually means for a given encoded matrix.

    sklearn 1.9 no longer exposes a fitted `max_features_` attribute, so the
    conversion from a fraction to a column count has to be done explicitly,
    otherwise a fraction silently gets recorded as "all columns".
    """
    if value == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if value == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(value, float):
        return max(1, int(value * n_features))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return n_features


# --------------------------------------------------------------------------- #
# Grouped permutation importance
# --------------------------------------------------------------------------- #
def group_columns(X: pd.DataFrame) -> dict[str, list[str]]:
    """Map interpretation group -> raw column names present in X."""
    groups: dict[str, list[str]] = {}
    for group, cols in INTERPRETATION_COLUMNS.items():
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Group {group!r} expects columns missing from X: {missing}")
        groups[group] = list(cols)
    covered = {c for cols in groups.values() for c in cols}
    if covered != set(X.columns):
        raise ValueError(f"Unassigned columns: {sorted(set(X.columns) - covered)}")
    return groups


def grouped_permutation(
    estimator,
    preprocessor,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    groups: dict[str, list[str]],
    n_repeats: int = 30,
    seed: int = SEED,
    baseline_rmse: float | None = None,
) -> dict[str, dict[str, float]]:
    """Permutation RMSE increase for each raw-variable group.

    One row permutation is shared by every column of a group (so brand and
    brand_model stay consistent), applied to the raw frame, then pushed through
    the already-fitted preprocessor. This keeps the multi-hot fuel encoding
    valid: shuffling encoded columns would invent fuel-label combinations.
    """
    if baseline_rmse is None:
        baseline_rmse = _rmse(y_valid, estimator.predict(preprocessor.transform(X_valid)))

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, float]] = {}
    n_rows = len(X_valid)
    for group, cols in groups.items():
        deltas = np.empty(n_repeats, dtype=float)
        for rep in range(n_repeats):
            order = rng.permutation(n_rows)
            X_perm = X_valid.copy()
            values = X_valid[cols].to_numpy()
            X_perm[cols] = values[order]
            pred = estimator.predict(preprocessor.transform(X_perm))
            deltas[rep] = _rmse(y_valid, pred) - baseline_rmse
        results[group] = {
            "delta_rmse_mean": float(deltas.mean()),
            "delta_rmse_sd": float(deltas.std(ddof=1)),
            "delta_rmse_min": float(deltas.min()),
            "delta_rmse_max": float(deltas.max()),
            "n_repeats": int(n_repeats),
            "baseline_rmse": float(baseline_rmse),
        }
    return results


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
