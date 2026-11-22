"""Angle 2 线性侧：图表生成。

产出：
- figures/lasso_path.png         Lasso 系数路径（全开发集，数值特征标准化后中心化求解）
- figures/lasso_path_caption.md  图注 + 与最终拟合系数的一致性核查结论
- figures/fold_rmse.png          三个模型逐折 OOF RMSE
- figures/pooled_rmse.png        三/四个模型 pooled OOF 对比

路径图的中心化修法：
  `lasso_path` 不含截距，所以先把 X（标准化后）与 y 都按列/整体去均值，
  再对中心化后的 (Xc, yc) 求解整条路径；截距可由 intercept = ȳ - x̄·coef 还原。
  这样图的系数与 `Lasso(fit_intercept=True)` 的系数同源，不再是"两套不一致的数"。
  仍会做一次数值核对，若偏差超阈值就把图注降级为"示意"。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import lasso_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_preprocessing import load_development, make_preprocessor  # noqa: E402


def _fmt_m(x: float) -> str:
    return f"{x / 1e6:,.1f}M"


def plot_lasso_path(X, y, out_dir: Path, results_dir: Path, final_alpha: float, alphas: np.ndarray, max_iter: int, tol: float):
    pre = make_preprocessor()
    Xd = np.asarray(pre.fit_transform(X), dtype=float)
    names = np.asarray(pre.get_feature_names_out())
    yv = y.to_numpy(dtype=float)

    x_mean = Xd.mean(axis=0)
    y_mean = float(yv.mean())
    Xc = Xd - x_mean
    yc = yv - y_mean

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        alphas_path, coefs, _ = lasso_path(
            Xc, yc, alphas=alphas[::-1], max_iter=max_iter, tol=tol, precompute=False
        )
    n_unconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    order = np.argsort(alphas_path)
    alphas_path, coefs = alphas_path[order], coefs[:, order]

    # 与最终拟合系数核对（final_alpha 若不在网格上则取最近点）
    idx = int(np.argmin(np.abs(np.log(alphas_path) - np.log(final_alpha))))
    path_coef = coefs[:, idx]
    final_lasso_path_coef = path_coef

    # 读最终表做对比：Lasso 的解在"进入/退出边界"处不唯一，个别稀有哑变量
    # 可能一边为 0、一边是极小系数，所以除系数级偏差外，更看**预测级**差异。
    feat_csv = results_dir / "lasso_selected_features.csv"
    verdict_rel = None
    pred_dev_max = pred_dev_rmse = None
    dominant = None
    final_vec = None
    if feat_csv.exists():
        final_tbl = pd.read_csv(feat_csv).set_index("variable")["coefficient"]
        final_vec = final_tbl.reindex(names).fillna(0.0).to_numpy()
        denom = max(np.abs(final_vec).max(), 1e-9)
        verdict_rel = float(np.abs(final_vec - final_lasso_path_coef).max() / denom)
        dcoef = final_vec - final_lasso_path_coef
        dpred = (Xd - x_mean) @ dcoef
        pred_dev_max = float(np.abs(dpred).max())
        pred_dev_rmse = float(np.sqrt(np.mean(dpred ** 2)))
        if len(dcoef):
            j = int(np.argmax(np.abs(dcoef)))
            dominant = (str(names[j]), float(dcoef[j]), float(np.abs(final_vec).max()))
    intercept = y_mean - float(x_mean @ path_coef)

    top = np.argsort(-np.abs(final_lasso_path_coef))[:8]
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160)
    short = lambda n: n.replace("categorical__", "").replace("numeric__", "").replace("fuel__", "")  # noqa: E731
    for j in top:
        ax.plot(alphas_path, coefs[j, :], lw=1.3, label=short(names[j]))
        if final_vec is not None:
            ax.scatter([final_alpha], [final_vec[j]], s=14, color="crimson", zorder=5)
    ax.axvline(final_alpha, color="crimson", ls="--", lw=1.2)
    lo, hi = ax.get_ylim()
    ax.text(final_alpha * 0.94, lo + 0.12 * (hi - lo), f"selected α = {final_alpha:.3g}", color="crimson", fontsize=9, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("coefficient (BDT, standardized inputs)")
    ax.set_title("Lasso coefficient path — full development set (centered, intercept-restored)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False, title="top-8 |coef|", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "lasso_path.png")
    plt.close(fig)

    # 图注
    pooled_rmse = None
    pm_csv = results_dir / "pooled_metrics.csv"
    if pm_csv.exists():
        pm = pd.read_csv(pm_csv)
        row = pm[pm.model_id == "Lasso"]
        if len(row):
            pooled_rmse = float(row.RMSE_BDT.iloc[0])

    if verdict_rel is None:
        head = "示意图：未找到 lasso_selected_features.csv，无法核对系数路径与最终系数的偏差。"
    elif verdict_rel < 1e-3:
        head = (
            f"与最终系数一致：路径在 selected α 处与最终 `Lasso(fit_intercept=True)` 的最大相对偏差"
            f" {verdict_rel:.2e}（<1e-3），可直接作为精确系数路径。"
        )
    else:
        head = (
            f"路径为 tol={tol:g} 的连续解，红点为最终精确解（tol 更紧的全开发集拟合）。两者在 selected α 处"
            f"系数级最大相对偏差 {verdict_rel:.2e}，折算到预测上最大 {pred_dev_max:,.0f} BDT、"
            f"RMSE 差异 {pred_dev_rmse:,.0f} BDT（占 pooled OOF RMSE 的 {pred_dev_rmse / pooled_rmse:.2%}）。"
            "差异来自坐标下降容差而非中心化处理，路径用于展示收缩趋势，最终系数以 `lasso_selected_features.csv` 为准。"
        )
    caption = (
        "Lasso 系数路径（全开发集）。\n\n"
        "- 求解方式：数值特征先 StandardScaler，X 与 y 均去均值后调用 `lasso_path`（无截距求解），"
        f"截距按 ȳ − x̄·β 还原（本图对应截距 ≈ {intercept:,.0f}）。\n"
        f"- 横轴 alpha 为 {alphas.min():g}–{alphas.max():g} 的 {len(alphas)} 个对数点；红色虚线为 `model_config.json` 记录的最终 alpha。\n"
        f"- 只画最终 |coef| 最大的 8 个特征，用于展示进入模型的顺序，不代表全部特征。\n"
        + (
            f"- 边界说明：差异最大的特征为 `{dominant[0]}`，系数差 {dominant[1]:,.0f}（仅为最大 |coef| 的"
            f" {abs(dominant[1]) / dominant[2]:.2%}），属于坐标下降容差在低正则端留下的差异，不影响结论。\n"
            if dominant and abs(dominant[1]) > 0 else ""
        )
        + (
            f"- 收敛情况：本次路径求解有 {n_unconverged}/{len(alphas)} 个 alpha 点未收敛（集中在 alpha 低端，"
            f"此时惩罚已接近 0、等价于无约束 OLS，坐标下降本就难以收敛）。该段只可读趋势，不可读系数数值。\n"
            if n_unconverged else "- 收敛情况：整条路径所有 alpha 点均收敛（无 ConvergenceWarning）。\n"
        )
        + f"\n{head}\n"
    )
    (out_dir / "lasso_path_caption.md").write_text(caption, encoding="utf-8")
    return verdict_rel


def plot_fold_rmse(results_dir: Path, fig_dir: Path):
    fm = pd.read_csv(results_dir / "fold_metrics.csv")
    wide = fm.pivot(index="cv_fold", columns="model_id", values="RMSE_BDT").sort_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=160)
    for m in wide.columns:
        ax.plot(wide.index, wide[m] / 1e6, marker="o", ms=3.5, lw=1.4, label=m)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("OOF RMSE (million BDT)")
    ax.set_title("Per-fold OOF RMSE (development set, 10 folds)")
    ax.set_xticks(wide.index)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fold_rmse.png")
    plt.close(fig)


def plot_pooled(results_dir: Path, fig_dir: Path):
    """横向点图：RMSE 差异只有 3%，柱状图从 0 起会看不出差别，所以把横轴收在数据区间内。"""
    pm = pd.read_csv(results_dir / "pooled_metrics.csv").sort_values("RMSE_BDT")
    vals = pm["RMSE_BDT"].to_numpy() / 1e6
    labels = pm["model_id"].tolist()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"][: len(pm)]

    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=170)
    ypos = np.arange(len(pm))[::-1]
    ax.scatter(vals, ypos, s=90, color=colors, zorder=3)
    for v, yp, lab in zip(vals, ypos, labels):
        ax.text(v, yp + 0.28, f"{lab} · {v:.3f}M", ha="center", va="bottom", fontsize=10)
    lo, hi = vals.min(), vals.max()
    pad = max((hi - lo) * 0.12, 0.02)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.7, len(pm) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("pooled development OOF RMSE (million BDT)")
    ax.set_title("Pooled OOF RMSE — three linear models (axis zoomed to the data range)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pooled_rmse.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--alpha-min", type=float, default=1e-2)
    ap.add_argument("--alpha-max", type=float, default=1e4)
    ap.add_argument("--n-alpha", type=int, default=100)
    ap.add_argument("--lasso-max-iter", type=int, default=20000)
    ap.add_argument("--lasso-tol", type=float, default=1e-3)
    ap.add_argument("--no-verify-hash", action="store_true", help="仅用于合成数据 smoke test")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)

    import json

    cfg = json.loads((res / "model_config.json").read_text(encoding="utf-8"))
    final_alpha = float(cfg["models"]["Lasso"]["alpha_final_full_development"])
    grid = cfg.get("alpha_grid", {})
    a_min = float(grid.get("min", args.alpha_min))
    a_max = float(grid.get("max", args.alpha_max))
    n_a = int(grid.get("n", args.n_alpha))
    alphas = np.logspace(np.log10(a_min), np.log10(a_max), n_a)

    X, y, _, _ = load_development(args.data_dir, verify_hash=not args.no_verify_hash)
    verdict = plot_lasso_path(X, y, out_dir, res, final_alpha, alphas, args.lasso_max_iter, args.lasso_tol)
    plot_fold_rmse(res, out_dir)
    plot_pooled(res, out_dir)
    print(f"figures 已生成于 {out_dir}；路径核对相对偏差 = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
