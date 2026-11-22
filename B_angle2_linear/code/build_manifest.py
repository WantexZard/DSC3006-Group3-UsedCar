"""生成 manifest.json：记录交付文件 hash、数据版本、软件版本、验收结论。

公共数据（cleaned.csv / split_manifest.csv）不复制进包，只登记 SHA-256 与
数据版本 member-a-v1；如果 data/ 目录里有原始文件，会同时校验并写入。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_preprocessing import EXPECTED_SHA256, sha256_of  # noqa: E402

SKIP = {"manifest.json"}
SKIP_DIRS = {"__pycache__", "data"}
SKIP_SUFFIX = (".pyc",)
SKIP_NAMES = {"predictions_reference.csv"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--data-version", default="member-a-v1")
    args = ap.parse_args()

    root = Path(args.package).resolve()
    data_dir = Path(args.data_dir)

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP or path.name in SKIP_NAMES \
            or any(part in SKIP_DIRS for part in path.parts) or path.suffix in SKIP_SUFFIX:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )

    data_hashes = {}
    for name, expected in EXPECTED_SHA256.items():
        p = data_dir / name
        actual = sha256_of(p) if p.exists() else None
        data_hashes[name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": bool(actual == expected),
            "included_in_package": False,
            "note": "公共只读数据，不复制进包，继续使用现有 member-a-v1",
        }

    acceptance_path = root / "results" / "acceptance_report.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.exists() else None

    manifest = {
        "package": "B_angle2_linear / Angle 2 线性侧",
        "data_version": args.data_version,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "target_drop_path": "not applicable (public code archive)",
        "data": data_hashes,
        "acceptance": {
            "all_passed": None if acceptance is None else acceptance.get("all_passed"),
            "report": "results/acceptance_report.md",
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files": files,
        "excluded_from_package": [
            "cleaned.csv / split_manifest.csv（公共数据，沿用 member-a-v1，不复制）",
            "任何测试集预测（Angle 2 测试集只报冻结 RF 的 1,208 条，由 A 侧提供）",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest.json 写出：{len(files)} 个文件，数据版本 {args.data_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
