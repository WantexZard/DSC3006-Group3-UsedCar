"""Conditional permutation for the three numeric groups.

Raw permutation shuffles a column across the whole validation fold, which creates
vehicles that cannot exist (a Toyota with 4,500 cc). If the forest leans on
capacity partly because capacity identifies the model, raw permutation will
overstate how much *independent* information the group carries.

This script repeats the permutation within `brand_model` strata: capacity is only
reshuffled among cars that share the same brand and model name. Strata of size 1
cannot move, so the conditional number is a lower bound and is always smaller for
mechanical reasons. Read the two numbers together:

  raw          = total information in the group, including redundancy with the
                 rest of the vehicle description;
  conditional  = what the group adds once the model name is held fixed.

Outputs: appends `permutation_conditional` rows to outputs/importance.csv and
outputs/importance_by_fold.csv (any earlier conditional rows are replaced first).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    OUTPUT_DIR,
    SEED,
    coerce_max_features,
    data_fingerprint,
    ensure_dirs,
    load_dev,
    make_preprocessor,
    make_rf,
    metric_block,
)

N_REPEATS = 30
NUMERIC_GROUPS = {"age": "car_age", "mileage": "kilometers_run", "capacity": "energy_capacity"}


def stratified_permutation(
    estimator,
    preprocessor,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    column: str,
    stratum: str = "brand_model",
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> dict[str, float]:
    baseline = metric_block(y_valid, estimator.predict(preprocessor.transform(X_valid)))["rmse"]
    strata = {key: np.asarray(pos) for key, pos in X_valid.groupby(stratum, sort=False).indices.items()}
    rng = np.random.default_rng(seed)
    values = X_valid[column].to_numpy()
    deltas = np.empty(n_repeats, dtype=float)
    for rep in range(n_repeats):
        permuted = values.copy()
        for positions in strata.values():
            if positions.size > 1:
                permuted[positions] = values[positions][rng.permutation(positions.size)]
        X_perm = X_valid.copy()
        X_perm[column] = permuted
        pred = estimator.predict(preprocessor.transform(X_perm))
        deltas[rep] = metric_block(y_valid, pred)["rmse"] - baseline
    return {
        "delta_rmse_mean": float(deltas.mean()),
        "delta_rmse_sd": float(deltas.std(ddof=1)),
        "baseline_rmse": float(baseline),
    }


def main() -> None:
    ensure_dirs()
    payload = json.loads((OUTPUT_DIR / "rf_selection_main.json").read_text(encoding="utf-8"))
    config = payload["best_config"]
    config["max_features"] = coerce_max_features(config["max_features"])
    model_id = payload["best_model_id"]

    X, y, metadata, folds = load_dev()
    rows = []
    for fold_id, (train_pos, valid_pos) in enumerate(folds):
        pre = make_preprocessor(scale_numeric=False)
        X_train = pre.fit_transform(X.iloc[train_pos])
        rf = make_rf(max_features=config["max_features"], min_samples_leaf=config["min_samples_leaf"])
        rf.fit(X_train, y[train_pos])
        X_valid = X.iloc[valid_pos]
        for group, column in NUMERIC_GROUPS.items():
            result = stratified_permutation(
                rf, pre, X_valid, y[valid_pos], column, seed=SEED + fold_id
            )
            rows.append({"model_id": model_id, "cv_fold": fold_id, "group": group, **result})
        print(f"fold {fold_id} done", flush=True)

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["model_id", "group"])
        .agg(
            conditional_delta_rmse_mean=("delta_rmse_mean", "mean"),
            conditional_delta_rmse_between_fold_sd=("delta_rmse_mean", "std"),
            conditional_delta_rmse_within_fold_sd=("delta_rmse_sd", "mean"),
            n_folds=("cv_fold", "nunique"),
        )
        .reset_index()
    )

    # Raw permutation of the same groups, for the "share retained" column.
    importance = pd.read_csv(OUTPUT_DIR / "importance.csv")
    raw = importance[importance["measure"] == "permutation_raw"][["group", "value"]].rename(
        columns={"value": "raw_delta_rmse_mean"}
    )
    merged = summary.merge(raw, on=["group"], how="left")
    merged["conditional_share_of_raw"] = merged["conditional_delta_rmse_mean"] / merged["raw_delta_rmse_mean"]
    merged = merged.sort_values("raw_delta_rmse_mean", ascending=False)

    # Append to the long tables, replacing any earlier conditional rows so that a
    # rerun is idempotent instead of duplicating them.
    importance_out = importance[importance["measure"] != "permutation_conditional"]
    importance_out = pd.concat(
        [
            importance_out,
            pd.DataFrame(
                {
                    "group": merged["group"],
                    "measure": "permutation_conditional",
                    "unit": "delta_rmse_bdt",
                    "value": merged["conditional_delta_rmse_mean"],
                    "sd_between_fold": merged["conditional_delta_rmse_between_fold_sd"],
                    "sd_within_fold": merged["conditional_delta_rmse_within_fold_sd"],
                    "n_folds": merged["n_folds"],
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["measure", "value"], ascending=[True, False])
    importance_out.to_csv(OUTPUT_DIR / "importance.csv", index=False)

    by_fold = pd.read_csv(OUTPUT_DIR / "importance_by_fold.csv")
    by_fold = by_fold[by_fold["measure"] != "permutation_conditional"]
    by_fold = pd.concat(
        [
            by_fold,
            pd.DataFrame(
                {
                    "group": frame["group"],
                    "measure": "permutation_conditional",
                    "cv_fold": frame["cv_fold"],
                    "value": frame["delta_rmse_mean"],
                    "sd": frame["delta_rmse_sd"],
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["measure", "group", "cv_fold"])
    by_fold.to_csv(OUTPUT_DIR / "importance_by_fold.csv", index=False)

    print("\ngroup      raw (M BDT)   conditional within brand_model (M BDT)   share retained")
    for row in merged.itertuples():
        print(
            f"{row.group:10s} {row.raw_delta_rmse_mean / 1e6:>10.2f} "
            f"{row.conditional_delta_rmse_mean / 1e6:>20.2f} "
            f"{row.conditional_share_of_raw:>22.0%}"
        )
    print("\nappended permutation_conditional rows to outputs/importance.csv")


if __name__ == "__main__":
    main()
