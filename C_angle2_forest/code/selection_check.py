"""Quantify how much the ten-fold selection rule could be flattering us.

Single-level ten-fold selection (choose the best pooled OOF score, then report
that score) is optimistic: the winning configuration was picked partly because
it happened to fit these folds well. Full nested CV would price that optimism
exactly, at roughly ten times the cost. This script instead re-prices selection
with a leave-one-fold-out rule using the predictions already saved by tune_rf.py:

  for each fold k: choose the configuration with the best pooled RMSE on the
  other nine folds, then keep that configuration's predictions for fold k.

The result is *not* unbiased -- those models were still trained on data that
includes fold k -- but it shows whether the reported headline number would move
materially if selection had been done without fold k. If the answer is "barely",
single-level selection is defensible for this report; if it moves a lot, the
limitation has to be stated next to the headline number.

Reads: outputs/oof_all_configs_main.csv (written by a full tune_rf.py run; not included
in this archive).
Outputs: outputs/selection_stability.json (per-fold audit rows included) and a printed verdict.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, metric_block


def main() -> None:
    oof_path = OUTPUT_DIR / "oof_all_configs_main.csv"
    if not oof_path.exists():
        raise SystemExit(
            "Missing outputs/oof_all_configs_main.csv. selection_check.py needs the "
            "full tune_rf.py output (12 configs x 10 folds); this file is not "
            "included in the archive."
        )
    oof = pd.read_csv(oof_path)
    selection = json.loads((OUTPUT_DIR / "rf_selection_main.json").read_text(encoding="utf-8"))
    best_id = selection["best_model_id"]
    config_cols = [c for c in oof.columns if c.startswith("rf__")]
    y = oof["y_true"].to_numpy()
    folds = oof["cv_fold"].to_numpy()

    pooled = {mid: metric_block(y, oof[mid].to_numpy())["rmse"] for mid in config_cols}
    table = []
    chosen_predictions = np.empty(len(oof), dtype=float)
    for fold_id in np.unique(folds):
        held = folds == fold_id
        others = ~held
        scores = {mid: metric_block(y[others], oof.loc[others, mid].to_numpy())["rmse"] for mid in config_cols}
        chosen = min(scores, key=scores.get)
        chosen_predictions[held] = oof.loc[held, chosen].to_numpy()
        table.append(
            {
                "held_out_fold": int(fold_id),
                "n_held_out": int(held.sum()),
                "chosen_config": chosen,
                "chosen_rmse_on_held_out": metric_block(y[held], oof.loc[held, chosen].to_numpy())["rmse"],
                "global_best_rmse_on_held_out": metric_block(y[held], oof.loc[held, best_id].to_numpy())["rmse"],
            }
        )

    frame = pd.DataFrame(table)

    lofo_rmse = metric_block(y, chosen_predictions)["rmse"]
    headline = pooled[best_id]
    gap = lofo_rmse - headline
    relative = gap / headline * 100
    winners = sorted(frame["chosen_config"].unique())
    runner_up = sorted(pooled.items(), key=lambda kv: kv[1])[1]

    verdict = (
        "single-level selection is defensible: re-selecting without the scored fold "
        f"moves the headline by {relative:+.2f}%"
        if abs(relative) < 1.0
        else "selection is unstable: the headline must be reported with this caveat"
    )

    print(f"headline (best pooled OOF)        {headline:>14,.0f} BDT  [{best_id}]")
    print(f"leave-one-fold-out selection      {lofo_rmse:>14,.0f} BDT  ({relative:+.2f}%)")
    print(f"runner-up ({runner_up[0]})  {runner_up[1]:>14,.0f} BDT  ({(runner_up[1] - headline) / headline * 100:+.2f}%)")
    print(f"distinct winners across 10 folds: {len(winners)} -> {winners}")
    print("verdict:", verdict)

    (OUTPUT_DIR / "selection_stability.json").write_text(
        json.dumps(
            {
                "headline_pooled_oof_rmse": headline,
                "headline_config": best_id,
                "leave_one_fold_out_selection_rmse": lofo_rmse,
                "relative_change_percent": relative,
                "runner_up": {"config": runner_up[0], "pooled_oof_rmse": runner_up[1]},
                "audit_by_fold": frame.to_dict(orient="records"),
                "distinct_winners": winners,
                "verdict": verdict,
                "caveat": "LOFO selection is not nested CV; models still saw the held-out fold in training.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
