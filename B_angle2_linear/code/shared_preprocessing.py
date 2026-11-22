"""Angle 2 线性侧：共享预处理与开发集加载。

设计要点（对应交接说明的“执行要求”）：
1. 特征列固定为 car_age / kilometers_run / energy_capacity（数值）、
   brand / brand_model / transmission（类别）、fuel_type（燃料多热）共 7 列；
   title / model / year_of_manufacture / row_id / source_line 一律不进特征。
2. 所有插补、标准化、one-hot、燃料多热都在 `make_preprocessor()` 返回的
   ColumnTransformer 内部完成，**由调用方在训练折内 fit**——数据分段本身
   不预缩放，避免“先在整个外层训练集 fit_transform 再做内层 CV”的泄漏。
3. 列名与既有产物一致（旧 lasso_selected_features.csv 的口径）：
   numeric__car_age、categorical__brand_*、
   categorical__brand_model_["toyota","corolla"]、categorical__transmission_*、
   fuel__fuel_type_hybrid …
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ("car_age", "kilometers_run", "energy_capacity")
CATEGORICAL_FEATURES = ("brand", "brand_model", "transmission")
FUEL_FEATURES = ("fuel_type",)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FUEL_FEATURES

TARGET_COLUMN = "price"

BLOCK_NUMERIC = "numeric"
BLOCK_CATEGORICAL = "categorical"
BLOCK_FUEL = "fuel"

# 交接说明给定的只读数据校验值
EXPECTED_SHA256 = {
    "cleaned.csv": "6f4791ce8763b6c73aa3af82be9b3d3cf5065e4cd901737211476ec5b4c5097f",
    "split_manifest.csv": "eae5f790e40e356e2f17c7df4956cf85e02e6a5a7f049a2e40d8ab8ab468a179",
}


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FuelMultiHot(BaseEstimator, TransformerMixin):
    """把 `cng|octane|petrol` 这类管道分隔的多值燃料拆成多热哑变量。

    `fuel_type` 在数据里是单列多标签（一辆车可以同时是 hybrid + octane + petrol），
    所以不能用 OneHotEncoder（那会把整串当一个类别），必须 MultiLabelBinarizer。
    输出列名 `fuel_type_<标签>`，经 ColumnTransformer 前缀后与旧表
    `fuel__fuel_type_hybrid` 等完全一致。
    """

    @staticmethod
    def _as_lists(values) -> list[list[str]]:
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        out: list[list[str]] = []
        for v in np.asarray(values, dtype=object):
            if v is None or (isinstance(v, float) and np.isnan(v)) or not str(v).strip():
                out.append([])
                continue
            out.append([t.strip() for t in str(v).split("|") if t.strip()])
        return out

    def fit(self, X, y=None):
        self.mlb_ = MultiLabelBinarizer(sparse_output=False)
        self.mlb_.fit(self._as_lists(X))
        return self

    def transform(self, X):
        # 某些折里某个燃料标签整体缺席时 MLB 会警告"unknown class will be ignored"，
        # 这是预期行为（该标签在这些折内不产生信号），不用污染日志。
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="unknown class.*will be ignored")
            return np.asarray(self.mlb_.transform(self._as_lists(X)), dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"fuel_type_{c}" for c in self.mlb_.classes_], dtype=object)


def make_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """返回未 fit 的预处理管线。每次 fit 只应看到当前训练折的数据。"""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (BLOCK_NUMERIC, numeric_pipe, list(NUMERIC_FEATURES)),
            (BLOCK_CATEGORICAL, categorical_pipe, list(CATEGORICAL_FEATURES)),
            (BLOCK_FUEL, FuelMultiHot(), list(FUEL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def load_development(
    data_dir: str | Path = "data",
    *,
    cleaned_name: str = "cleaned.csv",
    manifest_name: str = "split_manifest.csv",
    folds_reference: str | Path | None = None,
    verify_hash: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """载入开发集，返回 (X, y, metadata, folds)。

    - 开发集口径由 `split_manifest.csv` 的 `split == "train"` 决定（4,789 行 / 4,486 组）；
    - 外层十折直接用 manifest 的 `cv_fold`（0–9），**不重划**；
    - 若提供 `folds_reference`（旧版 predictions.csv），会逐行核对 cv_fold 是否完全相同。
    """
    data_dir = Path(data_dir)
    cleaned_path = data_dir / cleaned_name
    manifest_path = data_dir / manifest_name
    for p in (cleaned_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}（需要交接说明里的只读数据文件）。")

    if verify_hash:
        for p in (cleaned_path, manifest_path):
            expected = EXPECTED_SHA256.get(p.name)
            if expected:
                digest = sha256_of(p)
                if digest != expected:
                    raise ValueError(
                        f"{p.name} 的 SHA-256 与交接说明不一致：\n  期望 {expected}\n  实际 {digest}\n"
                        "数据是只读的，请确认没有重新清洗。"
                    )

    cleaned = pd.read_csv(cleaned_path, low_memory=False)
    manifest = pd.read_csv(manifest_path)

    missing = [c for c in FEATURE_COLUMNS + (TARGET_COLUMN,) if c not in cleaned.columns]
    if missing:
        raise ValueError(f"cleaned.csv 缺少列: {missing}")
    for col in ("row_id", "group_id", "split", "cv_fold"):
        if col not in manifest.columns:
            raise ValueError(f"split_manifest.csv 缺少列 {col}")

    dev = manifest[manifest["split"] == "train"].copy()
    if dev.empty:
        raise ValueError("split_manifest.csv 里没有 split == 'train' 的行。")
    if dev["row_id"].duplicated().any():
        raise ValueError("split_manifest.csv 的开发集 row_id 有重复。")

    cross = dev.groupby("group_id")["cv_fold"].nunique()
    if (cross > 1).any():
        raise ValueError(f"{int((cross > 1).sum())} 个组在开发集内跨折，违反同组不跨折。")
    leak = manifest.groupby("group_id")["split"].nunique()
    if (leak > 1).any():
        raise ValueError(f"{int((leak > 1).sum())} 个组同时出现在开发集与测试集，违反同组不跨集合。")

    frame = dev[["row_id", "group_id", "cv_fold"]].merge(
        cleaned[["row_id", *FEATURE_COLUMNS, TARGET_COLUMN]],
        on="row_id",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)

    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{int(frame[TARGET_COLUMN].isna().sum())} 条开发记录的 price 未在 cleaned.csv 匹配上，row_id 口径不一致。"
        )

    if folds_reference is not None:
        ref_path = Path(folds_reference)
        if ref_path.exists():
            ref = pd.read_csv(ref_path).drop_duplicates("row_id").set_index("row_id")["cv_fold"]
            ref_fold = frame["row_id"].map(ref)
            if ref_fold.isna().any():
                raise ValueError(f"折参照缺 {int(ref_fold.isna().sum())} 条 row_id。")
            if (ref_fold.to_numpy() != frame["cv_fold"].to_numpy()).any():
                n = int((ref_fold.to_numpy() != frame["cv_fold"].to_numpy()).sum())
                raise ValueError(f"{n} 条记录的 cv_fold 与旧 predictions.csv 不一致，外层十折被改动过。")
            print(f"[check] 外层十折与旧 predictions.csv 逐行一致（{len(frame):,} 行）")

    X = frame[list(FEATURE_COLUMNS)].copy()
    y = frame[TARGET_COLUMN].astype(float).copy()
    metadata = frame[["row_id", "group_id", "cv_fold"]].copy()
    metadata["cv_fold"] = metadata["cv_fold"].astype(int)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    fold_ids = sorted(metadata["cv_fold"].unique())
    if len(fold_ids) != 10:
        raise ValueError(f"开发集折数应为 10，实际 {len(fold_ids)}：{fold_ids}")
    for fold_id in fold_ids:
        valid_pos = np.flatnonzero(metadata["cv_fold"].to_numpy() == fold_id)
        train_pos = np.flatnonzero(metadata["cv_fold"].to_numpy() != fold_id)
        folds.append((train_pos, valid_pos))

    print(
        f"[data] 开发集 {len(X):,} 行 / {metadata['group_id'].nunique():,} 组 / {len(folds)} 折；"
        f"特征块 {len(make_preprocessor().transformers)} 个"
    )
    return X, y, metadata, folds
