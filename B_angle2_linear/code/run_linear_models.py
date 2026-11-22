"""Angle 2 线性侧重跑：嵌套（外层固定十折 + 内层 group 隔离 5-fold）评估。

改动相对旧脚本（逐条对应交接说明）：
1. Ridge / Lasso 的 alpha 不再固定为 100 / 5000，改为在**外层训练折内部**用
   5-fold GroupKFold（按 group_id 分组）逐折选参；预处理在每个内层训练折内
   `fit`，不再先对整个外层训练集 `fit_transform`。
2. Ridge 不再依赖 RidgeCV 的默认高效留一法（等价于随机 KFold，不按组），
   与 Lasso 用同一套 GroupKFold 内层流程，同网格、同准则。
3. 外层十折从旧 predictions.csv 的 cv_fold 读回，完全不变；OLS 不调参。
4. 只用开发集；不生成任何测试集预测。
5. 输出 model_config.json（网格、准则、预处理、最终 alpha、种子、版本）。
6. 重出 lasso_selected_features.csv，并附系数质量与跨折稳定性。

用法（在 B_angle2_linear 目录下）：
    python code/run_linear_models.py --data-dir data --out results
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, ridge_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    FUEL_FEATURES,
    NUMERIC_FEATURES,
    load_development,
    make_preprocessor,
    sha256_of,
)

SEED = 42
OLS_RCOND = 1e-10
DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_N_ALPHA = 1e-2, 1e4, 100


# ---------------------------------------------------------------- 指标
def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "RMSE_BDT": _rmse(y_true, y_pred),
        "MAE_BDT": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------- 内层折
def inner_group_splits(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 group_id 分组的 GroupKFold（无 shuffle，结果可复现）。"""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros((len(groups), 1)), np.zeros(len(groups)), groups))


def _pick_alpha(alphas: np.ndarray, mean_rmse: np.ndarray) -> tuple[int, float, float]:
    """取平均验证 RMSE 最小的 alpha；并列时取 alpha 更大者（更强正则）。

    返回 (index, best_rmse, 与次优之间的相对差)。
    """
    best = float(np.nanmin(mean_rmse))
    tol = 1e-9 * max(abs(best), 1.0)
    candidates = np.flatnonzero(mean_rmse <= best + tol)
    idx = int(candidates.max())
    ordered = np.sort(mean_rmse)
    gap = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
    return idx, best, gap


def _fit_ols(Xt, yt) -> tuple[np.ndarray, float, int, int]:
    """SVD 最小二乘（等价于无惩罚 OLS），显式截断近零奇异值。

    为什么要显式 rcond：brand 哑变量与 brand_model 哑变量之间存在**精确线性依赖**
    （每个品牌的 brand_model 求和恒等于该品牌的 brand 指示），加上稀有车型的单行哑变量，
    设计阵 208 列只有 178 秩。sklearn `LinearRegression` 默认 rcond 会把 1e-17 量级的
    奇异值留在解里，把系数放大到 1e18 量级、个别预测跑到 1e18 BDT。
    取 rcond=1e-10 只截断数值噪声方向，OLS 的拟合意义不变。
    """
    Xt = np.asarray(Xt, dtype=float)
    yt = np.asarray(yt, dtype=float).ravel()
    x_mean, y_mean = Xt.mean(axis=0), float(yt.mean())
    Xc, yc = Xt - x_mean, yt - y_mean
    U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = d > (d.max() * OLS_RCOND) if d.size else np.zeros(0, dtype=bool)
    coef = Vt[keep].T @ ((U[:, keep].T @ yc) / d[keep])
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept, int(keep.sum()), int(d.size)


def _ridge_inner_rmse(Xi, yi, Xv, yv, alphas) -> np.ndarray:
    """一条 SVD 解出整条 alpha 路径（内层每个折只分解一次）。

    中心化后用 SVD 手算 ridge 解：coef(alpha) = V diag(d/(d^2+alpha)) U^T y_c，
    等价于 Ridge(fit_intercept=True)，且一次分解覆盖全部 alpha。
    """
    Xi = np.asarray(Xi, dtype=float)
    Xv = np.asarray(Xv, dtype=float)
    yi = np.asarray(yi, dtype=float).ravel()
    yv = np.asarray(yv, dtype=float).ravel()
    x_mean, y_mean = Xi.mean(axis=0), float(yi.mean())
    Xc = Xi - x_mean
    yc = yi - y_mean

    U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    uty = U.T @ yc
    coefs = np.empty((len(alphas), Xc.shape[1]), dtype=float)
    for k, a in enumerate(alphas):
        coefs[k] = Vt.T @ (d / (d * d + float(a)) * uty)

    preds = (Xv - x_mean) @ coefs.T + y_mean
    return np.sqrt(np.mean((preds - yv.reshape(-1, 1)) ** 2, axis=0))


def _lasso_inner_rmse(Xi, yi, Xv, yv, alphas, max_iter, tol) -> tuple[np.ndarray, int]:
    """从大 alpha 向小 alpha warm start 走一遍网格，返回逐 alpha 的验证 RMSE。"""
    est = Lasso(warm_start=True, fit_intercept=True, max_iter=max_iter, tol=tol)
    curve = np.empty(len(alphas), dtype=float)
    n_unconverged = 0
    for k in range(len(alphas) - 1, -1, -1):
        est.set_params(alpha=float(alphas[k]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            est.fit(Xi, yi)
        n_unconverged += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        curve[k] = _rmse(yv, est.predict(Xv))
    return curve, n_unconverged


def _fit_lasso(Xt, yt, alpha, alphas, max_iter, tol) -> tuple[Lasso, int]:
    """直接拟合；万一在目标 alpha 不收敛，就从大 alpha warm start 走到目标值。"""
    est = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=max_iter, tol=tol)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        est.fit(Xt, yt)
    n_unconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    if n_unconverged:
        est = Lasso(warm_start=True, fit_intercept=True, max_iter=max_iter, tol=tol)
        for a in [a for a in alphas if a >= alpha]:
            est.set_params(alpha=float(a))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                est.fit(Xt, yt)
            n_unconverged += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        if float(est.alpha) != float(alpha):
            est.set_params(alpha=float(alpha))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                est.fit(Xt, yt)
            n_unconverged += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return est, n_unconverged


# ---------------------------------------------------------------- 单个外层折
def run_outer_fold(
    kind: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    train_pos: np.ndarray,
    valid_pos: np.ndarray,
    fold_id: int,
    alphas: np.ndarray,
    n_inner: int,
    max_iter: int,
    tol: float,
) -> dict:
    X_tr, y_tr = X.iloc[train_pos], y.iloc[train_pos]
    g_tr = groups[train_pos]

    if kind == "OLS":
        pre = make_preprocessor()
        Xt = np.asarray(pre.fit_transform(X_tr), dtype=float)
        Xv = np.asarray(pre.transform(X.iloc[valid_pos]), dtype=float)
        coef, intercept, rank, n_cols = _fit_ols(Xt, y_tr)
        return {
            "model_id": kind,
            "cv_fold": fold_id,
            "pred": Xv @ coef + intercept,
            "ols_rank": rank,
            "ols_ncols": n_cols,
            "alpha": np.nan,
            "inner_rmse": np.nan,
            "second_best_gap": np.nan,
            "coef": coef,
            "intercept": intercept,
            "names": np.asarray(pre.get_feature_names_out()),
            "n_unconverged": 0,
        }

    # ---- 内层：每个内层训练折各自 fit 预处理
    curves = []
    n_unconverged = 0
    for itr, iva in inner_group_splits(g_tr, n_inner):
        pre = make_preprocessor()
        Xi = pre.fit_transform(X_tr.iloc[itr])
        Xv_i = pre.transform(X_tr.iloc[iva])
        yi, yv_i = y_tr.iloc[itr], y_tr.iloc[iva]
        if kind == "Ridge":
            curves.append(_ridge_inner_rmse(Xi, yi, Xv_i, yv_i, alphas))
        else:
            curve, n_unc = _lasso_inner_rmse(Xi, yi, Xv_i, yv_i, alphas, max_iter, tol)
            curves.append(curve)
            n_unconverged += n_unc

    mean_curve = np.mean(np.vstack(curves), axis=0)
    idx, best_rmse, gap = _pick_alpha(alphas, mean_curve)
    alpha = float(alphas[idx])

    # ---- 用内层选定的 alpha 在外层训练折上整体重新 fit
    pre = make_preprocessor()
    Xt = pre.fit_transform(X_tr)
    Xv = pre.transform(X.iloc[valid_pos])
    if kind == "Ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha, fit_intercept=True).fit(Xt, y_tr)
        coef, intercept = model.coef_, float(model.intercept_)
    else:
        model, n_unc = _fit_lasso(Xt, y_tr, alpha, alphas, max_iter, tol)
        n_unconverged += n_unc
        coef, intercept = model.coef_, float(model.intercept_)

    return {
        "model_id": kind,
        "cv_fold": fold_id,
        "pred": model.predict(Xv),
        "alpha": alpha,
        "inner_rmse": best_rmse,
        "second_best_gap": gap,
        "coef": coef,
        "intercept": intercept,
        "names": np.asarray(pre.get_feature_names_out()),
        "n_unconverged": n_unconverged,
    }


# ---------------------------------------------------------------- 系数质量
def build_lasso_quality(
    final_names: np.ndarray,
    final_coef: np.ndarray,
    final_intercept: float,
    fold_coefs: list[tuple[np.ndarray, np.ndarray]],
    fold_alphas: list[float],
) -> tuple[pd.DataFrame, str]:
    final = pd.Series(final_coef, index=final_names)
    selected = final[final != 0].sort_values(key=np.abs, ascending=False)

    # 跨折稳定性：同一特征在外层 10 折 refit 中非零的折数、符号一致性
    stats = {}
    for names, coef in fold_coefs:
        for n, c in zip(names, coef):
            d = stats.setdefault(n, {"nonzero": 0, "pos": 0, "neg": 0, "coefs": []})
            if c != 0:
                d["nonzero"] += 1
                d["pos" if c > 0 else "neg"] += 1
            d["coefs"].append(float(c))
    rows = []
    for n in final_names:
        d = stats.get(n, {"nonzero": 0, "pos": 0, "neg": 0, "coefs": [0.0]})
        nz = d["nonzero"]
        rows.append(
            {
                "variable": n,
                "coef_final": float(final[n]),
                "nonzero_folds": nz,
                "selection_freq": nz / max(len(fold_coefs), 1),
                "sign_consistency": (max(d["pos"], d["neg"]) / nz) if nz else 0.0,
                "coef_median_across_folds": float(np.median(d["coefs"])) if d["coefs"] else 0.0,
            }
        )
    stability = (
        pd.DataFrame(rows)
        .sort_values(["selection_freq", "coef_final"], key=lambda s: s.abs() if s.name == "coef_final" else s, ascending=False)
        .reset_index(drop=True)
    )

    n_total = len(final)
    n_sel = int((final != 0).sum())
    abs_sum = float(final.abs().sum())
    top10 = selected.head(10)
    top10_share = float(top10.abs().sum() / abs_sum) if abs_sum else 0.0
    n_stable = int(((stability.selection_freq == 1.0) & (stability.coef_final != 0)).sum())
    n_selected_but_unstable = int(((stability.coef_final != 0) & (stability.selection_freq < 0.6)).sum())

    lines = [
        "# Lasso 系数质量说明（重跑后）",
        "",
        f"- 全开发集最终拟合：截距 {final_intercept:,.0f}（price 原始量纲）。",
        f"- 候选特征 {n_total} 个，非零 {n_sel} 个（稀疏度 {n_sel / n_total:.1%}）。",
        f"- 非零系数绝对值合计 {abs_sum:,.0f}；前 10 个特征占该合计 {top10_share:.1%}。",
        f"- 外层 10 折 refit 全部非零（selection_freq = 1.0）且最终非零的特征：{n_stable} 个。",
        f"- 最终非零但跨折选择率 < 0.6 的特征：{n_selected_but_unstable} 个（见 `lasso_stability.csv`）。",
        "",
        "## 量纲与可比性（写进 PPT 图注/脚注）",
        "",
        "有效系数是\"标准化后\"的：数值特征 `car_age / kilometers_run / energy_capacity` 已 StandardScaler，",
        "系数表示\"该变量每变动 1 个标准差带来的价格变动（元）\"；类别特征是 one-hot 哑变量，",
        "系数表示\"该类别出现 vs 基准类别组平均的价格差（元）\"。",
        "**两者量级不可直接互比大小**——品牌哑变量的系数天然大于数值特征的系数，",
        "不能据此说\"品牌比车龄重要\"。比较重要性应看标准化后的 |coef| 排名仅在同类特征内部成立。",
        "",
        "符号解读：正系数 = 抬价，负系数 = 压价。系数绝对值随 alpha 增大而系统性收缩，",
        "不同 alpha 下的系数不可混用（本表只对应 `model_config.json` 里记录的最终 alpha）。",
        "",
        "## 结构性问题（需在 PPT 中如实说明）",
        "",
        "- `brand` 与 `brand_model` 哑变量高度共线（brand_model 隐含 brand），Lasso 会在两者间",
        "  不稳定地挑选（同一信息可能落到 brand 系数、也可能落到 brand_model 系数），",
        "  因此单个哑变量的系数不能当作\"品牌溢价\"的因果估计；`lasso_stability.csv` 里的",
        "  selection_freq 就是这一不稳定性的量化。",
        "- 极端稀有类别（如 rolls royce、个别 2025 年新车型）只有个位数样本，其系数由个别记录驱动，",
        "  方差极大，只能读作\"样本内拟合值\"，不能外推。",
        "- 本文档的系数用于解释方向与量级，不用于因果推断；主表模型比较只看 OOF RMSE/MAE/R2。",
        "",
        "## 复算方式",
        "",
        "```bash",
        "python code/run_linear_models.py --data-dir data --out results",
        "```",
        "脚本用 `ColumnTransformer` 的 `get_feature_names_out()` 取变量名，",
        "与旧表 `lasso_selected_features.csv` 的命名一致（numeric__ / categorical__ / fuel__）。",
        "",
    ]
    return stability, "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds-reference", default=None, help="旧 predictions.csv；默认 data/predictions_reference.csv")
    ap.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    ap.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    ap.add_argument("--n-alpha", type=int, default=DEFAULT_N_ALPHA)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--lasso-max-iter", type=int, default=20000)
    ap.add_argument("--lasso-tol", type=float, default=1e-3, help="内层选参/逐折 refit 的 tol")
    ap.add_argument("--final-tol", type=float, default=1e-8, help="全开发集最终系数的 tol（用于出表，需比选参更紧）")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--rf-oof", default=None, help="可选的冻结 RF OOF 文件（row_id,y_pred_price），用于主表补第四行")
    ap.add_argument("--data-version", default="member-a-v1")
    ap.add_argument("--no-verify-hash", action="store_true", help="仅用于合成数据 smoke test，正式跑不要加")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.n_alpha)
    started = time.time()

    X, y, metadata, folds = load_development(
        args.data_dir, folds_reference=args.folds_reference, verify_hash=not args.no_verify_hash
    )
    groups = metadata["group_id"].to_numpy()
    print(f"[data] 开发集 {len(X):,} 行 / {metadata['group_id'].nunique():,} 组 / {len(folds)} 折")

    models = ["OLS", "Ridge", "Lasso"]
    tasks = [(kind, f, tr, va) for kind in models for f, (tr, va) in enumerate(folds)]
    results = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(run_outer_fold)(
            kind, X, y, groups, tr, va, f, alphas, args.inner_splits, args.lasso_max_iter, args.lasso_tol
        )
        for kind, f, tr, va in tasks
    )

    pred_rows, fold_rows, alpha_rows = [], [], []
    oof = {m: np.full(len(y), np.nan) for m in models}
    fold_coefs, fold_alphas = [], []
    total_unconv = 0
    names_ref = None

    ols_rank_rows = []
    for r in results:
        m, f = r["model_id"], r["cv_fold"]
        if m == "OLS":
            ols_rank_rows.append({"cv_fold": f, "rank": r["ols_rank"], "n_columns": r["ols_ncols"]})
        valid_pos = folds[f][1]
        oof[m][valid_pos] = r["pred"]
        y_va = y.iloc[valid_pos]
        met = _metrics(y_va, r["pred"])
        fold_rows.append({"model_id": m, "cv_fold": f, **met})
        total_unconv += int(r["n_unconverged"])
        if r["alpha"] == r["alpha"]:
            alpha_rows.append(
                {
                    "model_id": m,
                    "scope": f"outer_fold_{f}",
                    "alpha": r["alpha"],
                    "best_inner_rmse": r["inner_rmse"],
                    "second_best_gap_abs": r["second_best_gap"],
                }
            )
        if m == "Lasso":
            fold_coefs.append((r["names"], r["coef"]))
            fold_alphas.append(float(r["alpha"]))
            names_ref = r["names"]

    for m in models:
        if np.isnan(oof[m]).any():
            raise RuntimeError(f"{m} 的 OOF 有 {int(np.isnan(oof[m]).sum())} 条空值，折覆盖不完整")
        for rid, gid, fold, yt, yp in zip(metadata["row_id"], metadata["group_id"], metadata["cv_fold"], y, oof[m]):
            pred_rows.append(
                {
                    "model_id": m,
                    "target_scale": "price",
                    "evaluation_split": "development_oof",
                    "row_id": rid,
                    "group_id": gid,
                    "cv_fold": int(fold),
                    "y_true": float(yt),
                    "y_pred_price": float(yp),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)[["model_id", "cv_fold", "n", "RMSE_BDT", "MAE_BDT", "R2"]]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    pooled = []
    for m in models:
        met = _metrics(y, oof[m])
        pooled.append({"model_id": m, **met, "evaluation_split": "development_oof_pooled"})
    pooled_df = pd.DataFrame(pooled).sort_values("RMSE_BDT").reset_index(drop=True)
    pooled_df.insert(0, "rank", np.arange(1, len(pooled_df) + 1))
    pooled_df.to_csv(out_dir / "pooled_metrics.csv", index=False)

    # ---- 全开发集最终 Lasso（选参与外层同规则，但用全部开发数据）
    print("[final] 全开发集内层选参与最终 Lasso 拟合")
    full_curves, full_unconv = [], 0
    for itr, iva in inner_group_splits(groups, args.inner_splits):
        pre = make_preprocessor()
        Xi = pre.fit_transform(X.iloc[itr])
        Xv_i = pre.transform(X.iloc[iva])
        curve, n_unc = _lasso_inner_rmse(Xi, y.iloc[itr], Xv_i, y.iloc[iva], alphas, args.lasso_max_iter, args.lasso_tol)
        full_curves.append(curve)
        full_unconv += n_unc
    full_mean = np.mean(np.vstack(full_curves), axis=0)
    idx, full_best, full_gap = _pick_alpha(alphas, full_mean)
    final_alpha = float(alphas[idx])
    total_unconv += full_unconv

    pre_full = make_preprocessor()
    X_full = pre_full.fit_transform(X)
    final_name_array = np.asarray(pre_full.get_feature_names_out())
    final_model, n_unc = _fit_lasso(
        X_full, y, final_alpha, alphas, max(args.lasso_max_iter, 200000), args.final_tol
    )
    total_unconv += n_unc
    final_coef = final_model.coef_
    final_intercept = float(final_model.intercept_)

    selected = pd.DataFrame({"variable": final_name_array, "coefficient": final_coef})
    selected = selected[selected["coefficient"] != 0]
    selected["_abs"] = selected["coefficient"].abs()
    selected = selected.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)
    selected.to_csv(out_dir / "lasso_selected_features.csv", index=False)

    stability, quality_md = build_lasso_quality(
        final_name_array, final_coef, final_intercept, fold_coefs, fold_alphas
    )
    stability.to_csv(out_dir / "lasso_stability.csv", index=False)
    (out_dir / "lasso_coefficient_quality.md").write_text(quality_md, encoding="utf-8")

    alpha_rows.append(
        {
            "model_id": "Lasso",
            "scope": "full_development",
            "alpha": final_alpha,
            "best_inner_rmse": full_best,
            "second_best_gap_abs": full_gap,
        }
    )
    pd.DataFrame(alpha_rows).to_csv(out_dir / "cv_alpha_selection.csv", index=False)

    # ---- 主表（第四个模型仅在提供冻结 RF OOF 时补入）
    main_table = pooled_df.copy()
    rf_note = "未提供冻结 RF 的 OOF 文件，主表暂只含三个线性模型；RF 行由 A 侧提供。"
    if args.rf_oof:
        rf = pd.read_csv(args.rf_oof)
        rf = rf.drop_duplicates("row_id").set_index("row_id")["y_pred_price"]
        rf_pred = metadata["row_id"].map(rf)
        if rf_pred.isna().any():
            rf_note = f"RF OOF 缺 {int(rf_pred.isna().sum())} 条 row_id，未并入主表。"
        else:
            met = _metrics(y, rf_pred.to_numpy())
            main_table = pd.concat(
                [main_table, pd.DataFrame([{"model_id": "RF (frozen, Angle 1)", **met, "evaluation_split": "development_oof_pooled", "rank": np.nan}])],
                ignore_index=True,
            )
            rf_note = "RF 行来自冻结模型的开发集 OOF，未重新训练。"
    main_table = main_table[["rank", "model_id", "n", "RMSE_BDT", "MAE_BDT", "R2", "evaluation_split"]]
    main_table.to_csv(out_dir / "angle2_main_table.csv", index=False)

    # ---- 配置
    alpha_by_fold = {m: [] for m in models}
    for m in ("Ridge", "Lasso"):
        alpha_by_fold[m] = [
            float(row["alpha"]) for row in alpha_rows if row["model_id"] == m and row["scope"].startswith("outer_fold")
        ]
    config = {
        "experiment": "Angle 2 线性侧（开发集 OOF 主表）",
        "data": {
            "version": args.data_version,
            "cleaned_csv_sha256": EXPECTED_SHA256["cleaned.csv"],
            "n_rows_all": 5997,
            "n_development": int(len(X)),
            "n_groups_development": int(metadata["group_id"].nunique()),
            "outer_folds": int(len(folds)),
            "test_set_used": False,
        },
        "target": {"column": "price", "unit": "BDT", "transform": "none"},
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical_onehot": list(CATEGORICAL_FEATURES),
            "fuel_multi_hot": list(FUEL_FEATURES),
            "excluded": ["title", "model", "year_of_manufacture", "row_id", "source_line"],
        },
        "preprocessing": {
            "numeric": "SimpleImputer(median) -> StandardScaler",
            "categorical": "SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')",
            "fuel": "SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')  # 燃料多热",
            "placement": "ColumnTransformer 内，每个训练折单独 fit（含内层折），不使用全量 fit_transform",
        },
        "ols_numerics": {
            "solver": f"SVD least squares, rcond={OLS_RCOND:g}（显式截断近零奇异值）",
            "reason": "brand 与 brand_model 哑变量精确共线（每个品牌的 brand_model 哑变量求和恒等于该品牌哑变量），"
                      "叠加稀有车型单行哑变量后，每折设计阵 202–209 列、秩只有 171–177；"
                      "sklearn 默认 rcond 会把数值噪声方向放大，出现 1e18 量级预测",
            "rank_per_fold": ols_rank_rows,
        },
        "outer_cv": {
            "scheme": "固定 10 折（沿用 split_manifest.csv / 旧 predictions.csv 的 cv_fold，逐行读回，未重划）",
            "group_isolation": "同组不跨折（已校验）",
            "ols_tuned": False,
        },
        "inner_cv": {
            "scheme": "5-fold GroupKFold，按 group_id 分组，无 shuffle（seed 固定）",
            "selection_criterion": "内层 5 折平均验证 RMSE 最小",
            "tie_break": "并列时取 alpha 较大者（更强正则）",
            "ridge_alternative": "不使用 RidgeCV 的默认留一法（其等价 KFold 不按组）；Ridge 与 Lasso 共用同一内层流程",
        },
        "alpha_grid": {
            "min": args.alpha_min,
            "max": args.alpha_max,
            "n": args.n_alpha,
            "spacing": "logspace(base 10)",
            "ridge_solver": "svd（一条路径解出全部 alpha）",
            "lasso_solver": f"coordinate descent，warm_start 沿 alpha 递减，max_iter={args.lasso_max_iter}, tol={args.lasso_tol}（内层选参）",
            "lasso_final_tol": args.final_tol, 
        },
        "models": {
            "OLS": {"alpha": None, "note": "不调参，直接在外层训练折 fit"},
            "Ridge": {"alpha_selected_per_fold": alpha_by_fold["Ridge"], "alpha_median": float(np.median(alpha_by_fold["Ridge"]))},
            "Lasso": {
                "alpha_selected_per_fold": alpha_by_fold["Lasso"],
                "alpha_median": float(np.median(alpha_by_fold["Lasso"])),
                "alpha_final_full_development": final_alpha,
                "n_selected_features": int(len(selected)),
                "n_candidate_features": int(len(final_name_array)),
            },
        },
        "final_configuration": {
            "linear_side_deployable_model": "Lasso(alpha=最终 alpha) 在全开发集拟合，仅用于系数解读；不做测试集预测",
            "note": "Angle 2 主表比较四个模型（含冻结 RF）的开发集 pooled OOF；测试集只报冻结 RF 的 1,208 条",
        },
        "seed": SEED,
        "convergence_warnings": int(total_unconv),
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rf_row": rf_note,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out_dir / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    log = [
        f"开发集 {len(X)} 行 / {metadata['group_id'].nunique()} 组 / {len(folds)} 折",
        f"alpha 网格 {args.alpha_min:g}..{args.alpha_max:g} × {args.n_alpha} 点",
        f"Ridge 逐折 alpha: {['%.3g' % a for a in alpha_by_fold['Ridge']]}",
        f"Lasso 逐折 alpha: {['%.3g' % a for a in alpha_by_fold['Lasso']]}",
        f"Lasso 全开发集最终 alpha: {final_alpha:.6g}，非零特征 {len(selected)}/{len(final_name_array)}",
        f"收敛警告累计 {total_unconv}",
        f"pooled: " + "; ".join(f"{r.model_id} RMSE={r.RMSE_BDT:,.0f}" for r in pooled_df.itertuples()),
        f"耗时 {config['runtime_seconds']}s",
    ]
    (out_dir / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
