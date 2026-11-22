"""按交接说明的“验收”逐条自检，输出 results/acceptance_report.md / .json。

检查项：
A. 每个模型 4,789 行 OOF，每行恰好一次（row_id 不重复、无 NaN 预测）
B. 只含 development_oof，无测试集预测
C. 同组不跨折
D. 用 predictions.csv 重算 pooled OOF，与 pooled_metrics.csv 一致
E. 用 predictions.csv 重算 fold_metrics，与 fold_metrics.csv 一致
F. 特征列未使用 title / year_of_manufacture / row_id（配置声明检查）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 4789
EXPECTED_GROUPS = 4486


def _bool_row(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(ok), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = Path(args.results)
    out_md = Path(args.out) if args.out else res / "acceptance_report.md"

    pred = pd.read_csv(res / "predictions.csv")
    fold = pd.read_csv(res / "fold_metrics.csv")
    pooled = pd.read_csv(res / "pooled_metrics.csv")
    cfg = json.loads((res / "model_config.json").read_text(encoding="utf-8"))

    checks = []

    # A
    sizes = pred.groupby("model_id").size()
    dup = pred.duplicated(subset=["model_id", "row_id"]).sum()
    nunique = pred.groupby("model_id")["row_id"].nunique()
    ok_a = bool((sizes == EXPECTED_ROWS).all() and dup == 0 and (nunique == EXPECTED_ROWS).all() and pred.y_pred_price.notna().all())
    checks.append(_bool_row("A 每模型 4,789 行 OOF 且 row_id 唯一", ok_a,
                            f"行数 {sizes.to_dict()}；重复 row_id={int(dup)}；每模型唯一 row_id={nunique.to_dict()}；NaN 预测={int(pred.y_pred_price.isna().sum())}"))

    # B
    splits = sorted(pred.evaluation_split.unique().tolist())
    ok_b = splits == ["development_oof"]
    checks.append(_bool_row("B 只含 development_oof", ok_b, f"evaluation_split={splits}"))

    # C
    n_groups = pred.groupby("model_id")["group_id"].nunique().to_dict()
    cross = pred.groupby("group_id")["cv_fold"].nunique()
    ok_c = bool((cross > 1).sum() == 0 and all(v == EXPECTED_GROUPS for v in n_groups.values()))
    checks.append(_bool_row("C 同组不跨折", ok_c,
                            f"跨折组数={int((cross > 1).sum())}；每模型组数={n_groups}（期望 {EXPECTED_GROUPS}）"))

    # D
    worst_d = 0.0
    for m, g in pred.groupby("model_id"):
        rmse = float(np.sqrt(np.mean((g.y_true - g.y_pred_price) ** 2)))
        ref = float(pooled.loc[pooled.model_id == m, "RMSE_BDT"].iloc[0])
        worst_d = max(worst_d, abs(rmse - ref) / ref)
    ok_d = worst_d < 1e-9
    checks.append(_bool_row("D 重算 pooled OOF RMSE 与 pooled_metrics.csv 一致", ok_d, f"最大相对差={worst_d:.2e}"))

    # E
    worst_e = 0.0
    for (m, f), g in pred.groupby(["model_id", "cv_fold"]):
        rmse = float(np.sqrt(np.mean((g.y_true - g.y_pred_price) ** 2)))
        row = fold[(fold.model_id == m) & (fold.cv_fold == f)]
        worst_e = max(worst_e, abs(rmse - float(row.RMSE_BDT.iloc[0])) / rmse)
    ok_e = worst_e < 1e-9
    checks.append(_bool_row("E 重算 fold_metrics 与 fold_metrics.csv 一致", ok_e, f"最大相对差={worst_e:.2e}"))

    # F
    excluded = cfg["features"]["excluded"]
    used = set(sum([cfg["features"][k] for k in ("numeric", "categorical_onehot", "fuel_multi_hot")], []))
    ok_f = not (used & set(excluded))
    checks.append(_bool_row("F 特征不含 title / model / year_of_manufacture / row_id / source_line", ok_f, f"used={sorted(used)}; excluded={excluded}"))

    # G 折覆盖
    folds_per_model = pred.groupby("model_id")["cv_fold"].nunique().to_dict()
    ok_g = all(v == cfg["data"]["outer_folds"] for v in folds_per_model.values())
    checks.append(_bool_row("G 每模型覆盖全部外层折", ok_g, f"每模型折数={folds_per_model}"))

    all_ok = all(c["passed"] for c in checks)
    lines = ["# 验收自检报告", "", f"结论：{'全部通过' if all_ok else '存在未通过项'}", "", "| 检查 | 结果 | 证据 |", "| --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    lines += ["", "复跑命令：`python code/check_acceptance.py --results results`", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (res / "acceptance_report.json").write_text(
        json.dumps({"all_passed": all_ok, "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']} — {c['detail']}" for c in checks))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
