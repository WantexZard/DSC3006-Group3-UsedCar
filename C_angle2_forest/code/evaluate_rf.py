"""Refit the frozen RF configuration fold by fold and explain it.

Produces, for the chosen configuration only:
  - out-of-fold predictions, so any other member can reproduce the metrics;
  - grouped permutation importance on each validation fold (30 repeats, seed 42);
  - impurity (MDI) importance aggregated to the same six interpretation groups;
  - the canonical interface files (metrics.csv, fold_metrics.csv, predictions.csv).

Nothing here is allowed to look at the test split; that happens once, in
finalize_rf.py, after this script has frozen everything.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    OUTPUT_DIR,
    SEED,
    coerce_max_features,
    ensure_dirs,
    feature_mapping,
    fold_metric_rows,
    group_columns,
    grouped_permutation,
    load_dev,
    make_preprocessor,
    make_rf,
    metric_block,
    summary_rows,
)

N_REPEATS = 30
SELECTION_PATH = OUTPUT_DIR / "rf_selection_main.json"


def load_best_config() -> dict:
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    config = payload["best_config"]
    config["max_depth"] = None if config["max_depth"] in (None, "None") else int(config["max_depth"])
    config["max_features"] = coerce_max_features(config["max_features"])
    return {"payload": payload, "config": config, "model_id": payload["best_model_id"]}


def rename_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical column names required by A_data's interface."""
    return frame.rename(
        columns={
            "rmse": "RMSE_BDT",
            "mae": "MAE_BDT",
            "r2": "R2",
            "rmse_fold_sd": "RMSE_BDT_fold_sd",
        }
    )


def main() -> None:
    ensure_dirs()
    best = load_best_config()
    config, model_id = best["config"], best["model_id"]
    X, y, metadata, folds = load_dev()
    groups = group_columns(X)

    oof = np.full(len(X), np.nan)
    perm_rows: list[dict] = []
    mdi_rows: list[dict] = []
    mapping_rows: list[dict] = []
    feature_counts: list[int] = []

    for fold_id, (train_pos, valid_pos) in enumerate(folds):
        pre = make_preprocessor(scale_numeric=False)
        X_train = pre.fit_transform(X.iloc[train_pos])
        X_valid = pre.transform(X.iloc[valid_pos])
        rf = make_rf(max_features=config["max_features"], min_samples_leaf=config["min_samples_leaf"])
        rf.fit(X_train, y[train_pos])
        pred = rf.predict(X_valid)
        oof[valid_pos] = pred
        feature_counts.append(X_train.shape[1])

        baseline = metric_block(y[valid_pos], pred)["rmse"]
        results = grouped_permutation(
            rf,
            pre,
            X.iloc[valid_pos],
            y[valid_pos],
            groups,
            n_repeats=N_REPEATS,
            seed=SEED + fold_id,
            baseline_rmse=baseline,
        )
        for group, values in results.items():
            perm_rows.append({"model_id": model_id, "cv_fold": fold_id, "group": group, **values})

        mapping = feature_mapping(pre)
        importance = pd.Series(rf.feature_importances_, index=mapping["feature_index"].to_numpy())
        grouped_mdi = importance.groupby(mapping["interpretation_group"].to_numpy()).sum()
        for group, value in grouped_mdi.items():
            mdi_rows.append({"model_id": model_id, "cv_fold": fold_id, "group": group, "mdi_sum": float(value)})
        mapping_rows.append({"cv_fold": fold_id, "n_features": X_train.shape[1], "n_groups": mapping["interpretation_group"].nunique()})

        print(
            f"fold {fold_id}: RMSE {baseline:>14,.0f} BDT "
            f"| top group {max(results.items(), key=lambda kv: kv[1]['delta_rmse_mean'])[0]}",
            flush=True,
        )

    # ---- canonical interface: predictions / metrics / fold metrics ----------
    pred_frame = pd.DataFrame(
        {
            "model_id": model_id,
            "target_scale": "price",
            "evaluation_split": "development_oof",
            "row_id": metadata["row_id"].to_numpy(),
            "group_id": metadata["group_id"].to_numpy(),
            "cv_fold": metadata["cv_fold"].to_numpy(),
            "y_true": y,
            "y_pred_price": oof,
        }
    )
    pred_frame.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    metrics = rename_metrics(pd.DataFrame(summary_rows(model_id, "development_oof", y, oof, folds)))
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    fold_metrics = rename_metrics(pd.DataFrame(fold_metric_rows(model_id, "development_oof", y, oof, folds)))
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    # ---- importance, one long table per measure ----------------------------
    perm_frame = pd.DataFrame(perm_rows)
    perm_summary = (
        perm_frame.groupby(["model_id", "group"])
        .agg(
            delta_rmse_mean=("delta_rmse_mean", "mean"),
            delta_rmse_between_fold_sd=("delta_rmse_mean", "std"),
            delta_rmse_within_fold_sd=("delta_rmse_sd", "mean"),
            baseline_rmse_mean=("baseline_rmse", "mean"),
            n_folds=("cv_fold", "nunique"),
        )
        .reset_index()
        .sort_values("delta_rmse_mean", ascending=False)
    )

    mdi_frame = pd.DataFrame(mdi_rows)
    mdi_summary = (
        mdi_frame.groupby(["model_id", "group"])
        .agg(mdi_mean=("mdi_sum", "mean"), mdi_sd=("mdi_sum", "std"), n_folds=("cv_fold", "nunique"))
        .reset_index()
    )

    importance = pd.concat(
        [
            pd.DataFrame(
                {
                    "group": perm_summary["group"],
                    "measure": "permutation_raw",
                    "unit": "delta_rmse_bdt",
                    "value": perm_summary["delta_rmse_mean"],
                    "sd_between_fold": perm_summary["delta_rmse_between_fold_sd"],
                    "sd_within_fold": perm_summary["delta_rmse_within_fold_sd"],
                    "n_folds": perm_summary["n_folds"],
                }
            ),
            pd.DataFrame(
                {
                    "group": mdi_summary["group"],
                    "measure": "impurity_mdi_share",
                    "unit": "share_of_total",
                    "value": mdi_summary["mdi_mean"],
                    "sd_between_fold": mdi_summary["mdi_sd"],
                    "sd_within_fold": float("nan"),
                    "n_folds": mdi_summary["n_folds"],
                }
            ),
        ],
        ignore_index=True,
    )
    importance.to_csv(OUTPUT_DIR / "importance.csv", index=False)

    importance_by_fold = pd.concat(
        [
            pd.DataFrame(
                {
                    "group": perm_frame["group"],
                    "measure": "permutation_raw",
                    "cv_fold": perm_frame["cv_fold"],
                    "value": perm_frame["delta_rmse_mean"],
                    "sd": perm_frame["delta_rmse_sd"],
                }
            ),
            pd.DataFrame(
                {
                    "group": mdi_frame["group"],
                    "measure": "impurity_mdi_share",
                    "cv_fold": mdi_frame["cv_fold"],
                    "value": mdi_frame["mdi_sum"],
                    "sd": float("nan"),
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["measure", "group", "cv_fold"])
    importance_by_fold.to_csv(OUTPUT_DIR / "importance_by_fold.csv", index=False)

    rank_corr = (
        perm_summary.set_index("group")["delta_rmse_mean"]
        .rank(ascending=False, method="min")
        .corr(mdi_summary.set_index("group")["mdi_mean"].rank(ascending=False, method="min"), method="spearman")
    )

    print("\nPermutation importance (mean delta RMSE over folds, BDT):")
    for row in perm_summary.itertuples():
        print(
            f"  {row.group:18s} {row.delta_rmse_mean:>14,.0f} "
            f"(between-fold sd {row.delta_rmse_between_fold_sd:,.0f}, "
            f"within-fold sd {row.delta_rmse_within_fold_sd:,.0f})"
        )
    print(f"\nMDI vs permutation Spearman rank correlation: {rank_corr:.3f}")
    print(f"pooled OOF RMSE {metric_block(y, oof)['rmse']:,.0f} BDT")


if __name__ == "__main__":
    main()
