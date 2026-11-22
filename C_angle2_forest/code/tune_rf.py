"""Ten-fold Random Forest grid search on the development split.

Selection criterion is the *pooled* out-of-fold RMSE: the ten validation
predictions are concatenated into one vector and scored once. This is what
A_data's technical note prescribes, and it is not the same as the mean of the
ten per-fold RMSEs (which is what GridSearchCV reports by default), so the loop
is written by hand.

Outputs (in C_angle2_forest/outputs/)
------------------------------
rf_tuning_{tag}.csv          one row per configuration: pooled OOF metrics and fold spread
oof_all_configs_{tag}.csv    out-of-fold prediction of every configuration
unknown_by_fold_{tag}.csv    unknown-category load of each validation fold
rf_selection_{tag}.json      chosen configuration + leave-one-fold-out audit
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import (
    DEFAULT_RF,
    OUTPUT_DIR,
    SEED,
    data_fingerprint,
    ensure_dirs,
    load_dev,
    make_preprocessor,
    make_rf,
    metric_block,
    model_id_for,
    resolved_max_features,
    unknown_diagnostics,
    write_json,
)

GRID = [
    {"max_features": mf, "min_samples_leaf": msl}
    for mf in ("sqrt", 0.3, 0.5, 1.0)
    for msl in (1, 2, 5)
]


def main() -> None:
    parser = argparse.ArgumentParser(description="RF grid search on development folds")
    parser.add_argument("--limit-folds", type=int, default=None, help="use only the first N folds (self-test)")
    parser.add_argument("--limit-configs", type=int, default=None, help="use only the first N grid points (self-test)")
    parser.add_argument("--tag", default="main", help="suffix for output files")
    args = parser.parse_args()

    ensure_dirs()
    X, y, metadata, folds = load_dev()
    full_folds = folds
    if args.limit_folds:
        folds = folds[: args.limit_folds]
    grid = GRID[: args.limit_configs] if args.limit_configs else GRID
    full_run = len(folds) == len(full_folds) and not args.limit_configs

    n_rows = len(X)
    config_rows: list[dict] = []
    oof_matrix: dict[str, np.ndarray] = {}
    unknown_rows: list[dict] = []
    resolved_features: dict[int, int] = {}
    run_started = time.time()

    for index, overrides in enumerate(grid, start=1):
        params = {**DEFAULT_RF, **overrides}
        model_id = model_id_for(params)
        oof = np.full(n_rows, np.nan)
        fold_rmse: list[float] = []
        config_started = time.time()

        for fold_id, (train_pos, valid_pos) in enumerate(folds):
            pre = make_preprocessor(scale_numeric=False)
            X_train = pre.fit_transform(X.iloc[train_pos])
            X_valid = pre.transform(X.iloc[valid_pos])
            rf = make_rf(**overrides)
            rf.fit(X_train, y[train_pos])
            pred = rf.predict(X_valid)
            oof[valid_pos] = pred

            block = metric_block(y[valid_pos], pred)
            fold_rmse.append(block["rmse"])
            resolved_features[fold_id] = resolved_max_features(
                overrides["max_features"], X_train.shape[1]
            )

            if full_run and index == 1:
                diag = unknown_diagnostics(pre, X.iloc[valid_pos])
                unknown_rows.append(
                    {
                        "cv_fold": fold_id,
                        "valid_rows": len(valid_pos),
                        "brand_unknown_rows": diag["categorical"]["brand"]["rows_with_unknown"],
                        "brand_model_unknown_rows": diag["categorical"]["brand_model"]["rows_with_unknown"],
                        "transmission_unknown_rows": diag["categorical"]["transmission"]["rows_with_unknown"],
                        "fuel_unknown_rows": diag["fuel"]["rows_with_unknown"],
                        "fuel_unknown_tokens": diag["fuel"]["unknown_token_occurrences"],
                    }
                )

        pooled = metric_block(y, oof)
        elapsed = time.time() - config_started
        config_rows.append(
            {
                "model_id": model_id,
                "max_features": str(overrides["max_features"]),
                "min_samples_leaf": overrides["min_samples_leaf"],
                "n_estimators": params["n_estimators"],
                "max_depth": "None" if params["max_depth"] is None else params["max_depth"],
                "pooled_oof_rmse": pooled["rmse"],
                "pooled_oof_mae": pooled["mae"],
                "pooled_oof_r2": pooled["r2"],
                "mean_fold_rmse": float(np.mean(fold_rmse)),
                "fold_rmse_sd": float(np.std(fold_rmse, ddof=1)),
                "n_folds": len(folds),
                "fit_seconds": round(elapsed, 1),
            }
        )
        oof_matrix[model_id] = oof
        best_so_far = min(row["pooled_oof_rmse"] for row in config_rows)
        marker = "  <- best so far" if pooled["rmse"] == best_so_far else ""
        print(
            f"[{index:2d}/{len(grid)}] {model_id:24s} pooled RMSE {pooled['rmse']:>14,.0f} BDT "
            f"({elapsed:5.1f}s){marker}",
            flush=True,
        )

        # Persist incrementally so an interruption does not lose finished configurations.
        pd.DataFrame(config_rows).to_csv(OUTPUT_DIR / f"rf_tuning_{args.tag}.csv", index=False)

    config_frame = pd.DataFrame(config_rows).sort_values("pooled_oof_rmse").reset_index(drop=True)
    best = config_frame.iloc[0]
    best_id = str(best["model_id"])
    best_overrides = next(item for item in grid if model_id_for({**DEFAULT_RF, **item}) == best_id)

    pd.DataFrame(
        {
            "row_id": metadata["row_id"].to_numpy(),
            "cv_fold": metadata["cv_fold"].to_numpy(),
            "y_true": y,
            **{mid: oof_matrix[mid] for mid in oof_matrix},
        }
    ).to_csv(OUTPUT_DIR / f"oof_all_configs_{args.tag}.csv", index=False)
    if unknown_rows:
        pd.DataFrame(unknown_rows).to_csv(OUTPUT_DIR / f"unknown_by_fold_{args.tag}.csv", index=False)

    # Leave-one-fold-out selection audit: does the winner depend on which fold is scored?
    per_fold_rmse = {
        mid: [metric_block(y[valid_pos], oof[valid_pos])["rmse"] for _, valid_pos in folds]
        for mid, oof in oof_matrix.items()
    }
    audit = []
    for fold_id in range(len(folds)):
        others = [f for f in range(len(folds)) if f != fold_id]
        scores = {mid: float(np.sqrt(np.mean([per_fold_rmse[mid][f] ** 2 for f in others]))) for mid in oof_matrix}
        chosen = min(scores, key=scores.get)
        audit.append(
            {
                "held_out_fold": fold_id,
                "chosen_config": chosen,
                "chosen_config_rmse_on_held_out_fold": per_fold_rmse[chosen][fold_id],
                "global_best_rmse_on_held_out_fold": per_fold_rmse[best_id][fold_id],
            }
        )

    payload = {
        "selection_criterion": "pooled_oof_rmse",
        "seed": SEED,
        "n_estimators": DEFAULT_RF["n_estimators"],
        "n_folds": len(folds),
        "n_configs": len(grid),
        "full_run": full_run,
        "best_model_id": best_id,
        "best_config": {
            "max_features": best_overrides["max_features"],
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "n_estimators": int(best["n_estimators"]),
            "max_depth": None,
            "bootstrap": True,
            "random_state": SEED,
        },
        "best_pooled_oof": {
            "rmse": float(best["pooled_oof_rmse"]),
            "mae": float(best["pooled_oof_mae"]),
            "r2": float(best["pooled_oof_r2"]),
            "mean_fold_rmse": float(best["mean_fold_rmse"]),
            "fold_rmse_sd": float(best["fold_rmse_sd"]),
        },
        "resolved_max_features_per_fold": resolved_features,
        "selection_audit_leave_one_fold_out": audit,
        "selection_audit_note": (
            "Not nested CV. Models for the other folds were trained on data that includes "
            "the held-out fold; this only checks whether the ranking is stable when one "
            "fold is removed."
        ),
        "data": data_fingerprint(),
        "wall_seconds": round(time.time() - run_started, 1),
    }
    write_json(OUTPUT_DIR / f"rf_selection_{args.tag}.json", payload)

    print("\nBest configuration:", best_id)
    print(
        f"  pooled OOF RMSE {best['pooled_oof_rmse']:,.0f} BDT | "
        f"MAE {best['pooled_oof_mae']:,.0f} | R2 {best['pooled_oof_r2']:.4f}"
    )
    print(
        f"  mean fold RMSE {best['mean_fold_rmse']:,.0f} "
        f"(sd {best['fold_rmse_sd']:,.0f}) -- reported separately, never used for selection"
    )
    winners = sorted({row["chosen_config"] for row in audit})
    print(f"  leave-one-fold-out selection: {len(winners)} distinct winner(s) -> {winners}")
    print(f"  total {payload['wall_seconds']}s")


if __name__ == "__main__":
    main()
