"""Figures for C_angle2_forest's Angle 2 deliverable.

All labels are in English (the final slides are English) and all values are in
millions of BDT, matching A_data's EDA convention. Each figure is regenerated
from the CSV outputs, never hand-edited.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURE_DIR, OUTPUT_DIR, ensure_dirs

MILLION = 1_000_000


def importance_measure(name: str) -> pd.DataFrame:
    """One slice of outputs/importance.csv, sorted by value."""
    frame = pd.read_csv(OUTPUT_DIR / "importance.csv")
    return frame[frame["measure"] == name].sort_values("value", ascending=False).reset_index(drop=True)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def tuning_heatmap() -> None:
    frame = pd.read_csv(OUTPUT_DIR / "rf_tuning_main.csv")
    frame["mf"] = frame["max_features"].astype(str)
    grid = frame.pivot(index="min_samples_leaf", columns="mf", values="pooled_oof_rmse") / MILLION
    order = [c for c in ["sqrt", "0.3", "0.5", "1.0"] if c in grid.columns]
    grid = grid[order]
    grid = grid.sort_index()

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    image = ax.imshow(grid.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("max_features (share of the ~203 encoded columns; 'sqrt' = √p)")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"min_samples_leaf={v}" for v in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid.iloc[i, j]):
                ax.text(j, i, f"{grid.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Random Forest: pooled out-of-fold RMSE (million BDT)\nlower is better; 10 development folds")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="pooled OOF RMSE (million BDT)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig01_tuning_pooled_rmse.png")
    plt.close(fig)


def permutation_bars() -> None:
    frame = importance_measure("permutation_raw").sort_values("value")
    values = frame["value"] / MILLION
    errors = frame["sd_between_fold"] / MILLION
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(frame))
    ax.barh(y, values, xerr=errors, color="#4c72b0", error_kw={"elinewidth": 1, "capsize": 3})
    ax.set_yticks(y)
    ax.set_yticklabels(frame["group"])
    ax.set_xlabel("increase in RMSE when the group is permuted (million BDT)")
    ax.set_title("Grouped permutation importance on validation folds\n(30 repeats per fold; error bars = SD across 10 folds)")
    ax.set_xlim(0, max(values + errors) * 1.18)
    for yi, value, error in zip(y, values, errors):
        ax.text(value + error + max(values) * 0.015, yi, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig02_grouped_permutation.png")
    plt.close(fig)


def importance_scale_comparison() -> None:
    perm = importance_measure("permutation_raw").set_index("group")["value"]
    mdi = importance_measure("impurity_mdi_share").set_index("group")["value"]
    frame = pd.DataFrame({"perm_share": perm / perm.sum(), "mdi_share": mdi / mdi.sum()}).reset_index()
    frame = frame.sort_values("perm_share", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    y = np.arange(len(frame))
    axes[0].barh(y, frame["perm_share"], color="#4c72b0")
    axes[0].set_title("Permutation (share of total)")
    axes[1].barh(y, frame["mdi_share"], color="#dd8452")
    axes[1].set_title("Impurity / MDI (share of total)")
    for ax in axes:
        ax.set_xlabel("share of total importance")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame["group"])
    fig.suptitle("Same model, two importance scales: the ranking is not interchangeable", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGURE_DIR / "fig03_importance_scales.png", bbox_inches="tight")
    plt.close(fig)


def fold_dispersion() -> None:
    folds = pd.read_csv(OUTPUT_DIR / "fold_metrics.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    pooled = metrics.loc[metrics["aggregation"] == "pooled_oof", "RMSE_BDT"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(folds["cv_fold"].astype(str), folds["RMSE_BDT"] / MILLION, color="#4c72b0")
    ax.axhline(pooled / MILLION, color="#c44e52", linestyle="--", linewidth=1.2,
               label=f"pooled OOF = {pooled / MILLION:.2f}M")
    ax.set_xlabel("development fold")
    ax.set_ylabel("RMSE (million BDT)")
    ax.set_title("Per-fold RMSE is dominated by a few high-price listings\npooled OOF is the reported aggregate, not the fold mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig04_fold_rmse_dispersion.png")
    plt.close(fig)


def conditional_vs_raw() -> None:
    perm = importance_measure("permutation_raw").set_index("group")["value"].rename("raw_delta_rmse_mean")
    cond = importance_measure("permutation_conditional").set_index("group")["value"].rename(
        "conditional_delta_rmse_mean"
    )
    frame = pd.concat([perm, cond], axis=1).dropna().reset_index().sort_values("raw_delta_rmse_mean")
    y = np.arange(len(frame))
    height = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.barh(y + height / 2, frame["raw_delta_rmse_mean"] / MILLION, height=height,
            color="#4c72b0", label="permute across the fold (raw)")
    ax.barh(y - height / 2, frame["conditional_delta_rmse_mean"] / MILLION, height=height,
            color="#dd8452", label="permute within brand_model (conditional)")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["group"])
    ax.set_xlabel("increase in RMSE (million BDT)")
    ax.set_title("Much of 'capacity matters' is really 'the model badge matters'\n(raw vs within-model permutation; 30 repeats x 10 folds)")
    ax.legend(loc="lower right", fontsize=8)
    for yi, raw, cond in zip(y, frame["raw_delta_rmse_mean"] / MILLION, frame["conditional_delta_rmse_mean"] / MILLION):
        ax.text(raw, yi + height / 2, f" {raw:.2f}", va="center", fontsize=8)
        ax.text(cond, yi - height / 2, f" {cond:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, frame["raw_delta_rmse_mean"].max() / MILLION * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig05_raw_vs_conditional.png")
    plt.close(fig)


def pairwise_delta() -> None:
    """Paired RMSE difference against the forest, with the bootstrap interval.

    Read as "what each linear model costs compared with the forest": positive
    means worse. The interval is bootstrapped over rows because the ten folds
    were trained on overlapping data.
    """
    path = OUTPUT_DIR / "angle2_models.csv"
    if not path.exists():
        print("skip fig06: run merge_angle2.py first")
        return
    frame = pd.read_csv(path)
    frame = frame[frame["delta_rmse_bdt"].notna()].sort_values("delta_rmse_bdt")
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    y = np.arange(len(frame))
    ax.hlines(y, frame["ci95_low"] / MILLION, frame["ci95_high"] / MILLION, color="#4c72b0", linewidth=2)
    ax.plot(frame["delta_rmse_bdt"] / MILLION, y, "o", color="#c1461b", markersize=6)
    ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["model_id"])
    ax.set_xlabel("ΔRMSE vs Random Forest (million BDT); positive = forest better")
    ax.set_title("Paired difference on identical rows\n95% interval from resampling rows (not groups), 2,000 draws")
    for yi, row in zip(y, frame.itertuples()):
        ax.text(row.delta_rmse_bdt / MILLION, yi + 0.16, f"{row.delta_rmse_bdt / MILLION:+.2f}", ha="center", fontsize=8)
    ax.set_xlim(-0.15, frame["ci95_high"].max() / MILLION * 1.18)
    ax.set_ylim(-0.6, len(frame) - 0.4)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig06_paired_delta.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    tuning_heatmap()
    permutation_bars()
    importance_scale_comparison()
    fold_dispersion()
    conditional_vs_raw()
    pairwise_delta()
    perm = importance_measure("permutation_raw").set_index("group")["value"]
    mdi = importance_measure("impurity_mdi_share").set_index("group")["value"]
    spearman = perm.rank(ascending=False).corr(mdi.rank(ascending=False), method="spearman")
    print("figures written to", FIGURE_DIR)
    print("permutation vs MDI rank correlation:", round(float(spearman), 3))


if __name__ == "__main__":
    main()
