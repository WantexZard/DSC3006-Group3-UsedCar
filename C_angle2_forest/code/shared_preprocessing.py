"""Fold-local preprocessing and validated access to member-a-v1 data.

load_development returns (X, y, metadata, folds). Use X.iloc[train_pos]
and X.iloc[valid_pos] from each fold, and fit a fresh preprocessor inside
that fold. load_holdout returns (X, y, metadata), for final evaluation only.
Neither loader fits anything. Holdout targets are not parsed or retained by
load_development; the CSV itself is public under a procedural embargo.

Full onehot plus an intercept and nested brand/model is rank deficient.
OLS owners must specify a minimum-norm solver convention and report rank.
Training fitted values are unique, but out-of-sample estimates can depend on
the solver convention, especially for unseen categories. Separate brand and
model effects are not identifiable. Ridge/Lasso coefficient allocations are
encoding dependent: report joint selections/paths, not magnitudes compared
with RF importance. For joint permutation, use the same row permutation
for brand and brand_model, and likewise for all fuel bits. No models are
implemented in this module.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

if __package__:
    from .build_dataset import (
        CLEANED_COLUMNS, N_FOLDS, SPLIT_COLUMNS, assign_partitions, canonical_group_key,
        canonical_json, normalize_category, sha256_bytes,
    )
else:
    from build_dataset import (
        CLEANED_COLUMNS, N_FOLDS, SPLIT_COLUMNS, assign_partitions, canonical_group_key,
        canonical_json, normalize_category, sha256_bytes,
    )


NUMERIC_FEATURES = ("car_age", "kilometers_run", "energy_capacity")
CATEGORICAL_FEATURES = ("brand", "brand_model", "transmission")
MODEL_FEATURES = (
    "car_age", "kilometers_run", "energy_capacity", "brand", "brand_model",
    "fuel_type", "transmission",
)
INTERPRETATION_GROUPS = {
    "age": ("car_age",),
    "mileage": ("kilometers_run",),
    "capacity": ("energy_capacity",),
    "vehicle_identity": ("brand", "brand_model"),
    "fuel": ("fuel_type",),
    "transmission": ("transmission",),
}


def _fuel_tokens(value: str) -> tuple[str, ...]:
    """Validate one canonical fuel cell without learning any vocabulary."""
    if not isinstance(value, str) or not value:
        raise ValueError("fuel_type must contain a nonempty canonical token set.")
    tokens = tuple(value.split("|"))
    if (
        tokens != tuple(sorted(set(tokens)))
        or any(not token or "," in token or token != normalize_category(token) for token in tokens)
    ):
        raise ValueError("fuel_type must use sorted unique normalized tokens separated by |.")
    return tokens


class FuelMultiHotEncoder(TransformerMixin, BaseEstimator):
    """Dense multi-hot encoding with a sorted vocabulary learned only in fit.

    Accept a two-dimensional single-column array or DataFrame of canonical
    fuel strings. Unknown tokens produce no new columns and no fitted-state
    mutation. unknown_diagnostics reports token occurrences once per row.
    """

    @staticmethod
    def _rows(X) -> list[tuple[str, ...]]:
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("FuelMultiHotEncoder expects exactly one input column.")
        return [_fuel_tokens(value) for value in values[:, 0]]

    def fit(self, X, y=None):
        rows = self._rows(X)
        if not rows:
            raise ValueError("Cannot fit a fuel vocabulary on zero rows.")
        self.n_features_in_ = 1
        # Clear optional metadata on refit, as the input container may change.
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if isinstance(X, pd.DataFrame) and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.classes_ = np.asarray(sorted({token for row in rows for token in row}), dtype=object)
        self.vocabulary_ = {token: index for index, token in enumerate(self.classes_)}
        return self

    def _checked_rows(self, X) -> list[tuple[str, ...]]:
        check_is_fitted(self, "vocabulary_")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("Fuel input column differs from the fitted column.")
        return self._rows(X)

    def transform(self, X):
        rows = self._checked_rows(X)
        result = np.zeros((len(rows), len(self.classes_)), dtype=np.float64)
        for row_index, tokens in enumerate(rows):
            for token in tokens:
                index = self.vocabulary_.get(token)
                if index is not None:
                    result[row_index, index] = 1.0
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vocabulary_")
        expected = getattr(self, "feature_names_in_", np.asarray(["fuel_type"], dtype=object))
        names = expected if input_features is None else np.asarray(input_features, dtype=object)
        if names.shape != (1,):
            raise ValueError("Exactly one fuel feature name is required.")
        if hasattr(self, "feature_names_in_") and not np.array_equal(names, expected):
            raise ValueError("Fuel feature name differs from the fitted column.")
        return np.asarray([f"{names[0]}_{token}" for token in self.classes_], dtype=object)

    def unknown_diagnostics(self, X) -> dict:
        """Return rows with unknown tokens and per-token row counts; do not fit."""
        rows = self._checked_rows(X)
        counts = Counter()
        rows_with_unknown = 0
        for tokens in rows:
            unknown = [token for token in tokens if token not in self.vocabulary_]
            rows_with_unknown += bool(unknown)
            counts.update(unknown)
        return {
            "rows": len(rows), "rows_with_unknown": rows_with_unknown,
            "unknown_token_occurrences": sum(counts.values()),
            "by_token": dict(sorted(counts.items())),
        }


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Return an unfitted, clone-compatible dense transformer.

    Only numeric columns are scaled. Median imputation retains fully empty
    training columns with sklearn's zero fallback; these are computational
    placeholders, not recovered specifications. No missing indicators are added.
    """
    if not isinstance(scale_numeric, bool):
        raise TypeError("scale_numeric must be a bool.")
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps), list(NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(
                drop=None, handle_unknown="ignore", sparse_output=False,
                dtype=np.float64,
            ), list(CATEGORICAL_FEATURES)),
            ("fuel", FuelMultiHotEncoder(), ["fuel_type"]),
        ],
        remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True,
    )


def feature_mapping(preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Map every fitted output column to its source and interpretation group.

    Columns: feature_index, feature_name, original_feature, interpretation_group.
    Group by interpretation_group for joint importance/permutation. Recompute
    this mapping after each fold fit because vocabularies can differ by fold.
    """
    check_is_fitted(preprocessor, "transformers_")
    names = preprocessor.get_feature_names_out()
    original_by_index = {}
    numeric_slice = preprocessor.output_indices_["numeric"]
    for index, original in zip(range(numeric_slice.start, numeric_slice.stop), NUMERIC_FEATURES):
        original_by_index[index] = original
    encoder = preprocessor.named_transformers_["categorical"]
    position = preprocessor.output_indices_["categorical"].start
    for original, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        for _ in categories:
            original_by_index[position] = original
            position += 1
    fuel_slice = preprocessor.output_indices_["fuel"]
    for index in range(fuel_slice.start, fuel_slice.stop):
        original_by_index[index] = "fuel_type"
    if set(original_by_index) != set(range(len(names))):
        raise ValueError("Feature mapping requires the unmodified make_preprocessor configuration.")
    group_for = {
        original: group for group, originals in INTERPRETATION_GROUPS.items()
        for original in originals
    }
    return pd.DataFrame([
        {"feature_index": index, "feature_name": str(name),
         "original_feature": original_by_index[index],
         "interpretation_group": group_for[original_by_index[index]]}
        for index, name in enumerate(names)
    ])


def unknown_diagnostics(preprocessor: ColumnTransformer, X: pd.DataFrame) -> dict:
    """Count unknown categories/tokens against fitted training vocabularies.

    This does not transform, fit, or mutate the preprocessor. Category counts
    count rows; fuel occurrences count each distinct unknown token in each row.
    X must be the canonical feature DataFrame, never a target-containing report.
    """
    check_is_fitted(preprocessor, "transformers_")
    if not isinstance(X, pd.DataFrame) or not set(MODEL_FEATURES).issubset(X.columns):
        raise ValueError("Unknown diagnostics require the model feature DataFrame.")
    encoder = preprocessor.named_transformers_["categorical"]
    diagnostics = {}
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        unknown = X.loc[~X[column].isin(categories), column]
        counts = Counter(unknown.tolist())
        diagnostics[column] = {
            "rows_with_unknown": len(unknown),
            "by_label": {str(label): count for label, count in sorted(counts.items())},
        }
    return {
        "rows": len(X), "categorical": diagnostics,
        "fuel": preprocessor.named_transformers_["fuel"].unknown_diagnostics(X[["fuel_type"]]),
    }


def _csv_records(path: Path, columns: tuple[str, ...]):
    """Reject duplicate/extra headers, ragged rows and blank records at the file boundary."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, [])) != columns:
            raise ValueError(f"Invalid schema in {path.name}.")
        for cells in reader:
            if len(cells) != len(columns):
                raise ValueError(f"Malformed record in {path.name}.")
            yield dict(zip(columns, cells))


def _unsigned_integer(value: str, column: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Invalid integer in {column}.")
    result = int(value)
    if str(result) != value:
        raise ValueError(f"Noncanonical integer in {column}.")
    return result


def _load_partition(data_dir: str | Path, selected_split: str) -> tuple:
    data_dir = Path(data_dir)
    manifest = {}
    group_partitions = {}
    for row in _csv_records(data_dir / "split_manifest.csv", SPLIT_COLUMNS):
        row_id, gid, split, fold_text = (row[column] for column in SPLIT_COLUMNS)
        if not row_id or row_id in manifest or not gid:
            raise ValueError("Manifest row IDs must be nonempty and unique; group IDs must be nonempty.")
        if split not in {"train", "test"}:
            raise ValueError("Manifest split must be train or test.")
        if (split == "test" and fold_text != "-1") or (
            split == "train" and fold_text not in {str(fold) for fold in range(N_FOLDS)}
        ):
            raise ValueError("Manifest fold must be 0..9 for train and -1 for test.")
        partition = (split, int(fold_text))
        if gid in group_partitions and group_partitions[gid] != partition:
            raise ValueError("A group spans partitions or validation folds.")
        group_partitions[gid] = partition
        manifest[row_id] = (gid, *partition)
    if {value[0] for value in group_partitions.values()} != {"train", "test"}:
        raise ValueError("Both development and holdout partitions must be nonempty.")
    if {value[1] for value in group_partitions.values() if value[0] == "train"} != set(range(N_FOLDS)):
        raise ValueError("All ten development validation folds must be present.")

    features, targets, metadata = [], [], []
    seen_rows, seen_lines = set(), set()
    for row in _csv_records(data_dir / "cleaned.csv", CLEANED_COLUMNS):
        row_id = row["row_id"]
        if row_id in seen_rows or row_id not in manifest:
            raise ValueError("Cleaned row IDs must join one-to-one with the manifest.")
        seen_rows.add(row_id)
        line = _unsigned_integer(row["source_line"], "source_line")
        if line < 2 or line in seen_lines or row_id != f"ucv2-{line - 1:06d}":
            raise ValueError("Invalid row lineage in cleaned.csv.")
        seen_lines.add(line)
        gid, split, fold = manifest[row_id]
        if row["group_id"] != gid:
            raise ValueError("Group ID mismatch between cleaned.csv and manifest.")
        for column in NUMERIC_FEATURES:
            row[column] = (
                None if column == "energy_capacity" and row[column] == ""
                else _unsigned_integer(row[column], column)
            )
        for column in ("brand", "model", "transmission"):
            if not row[column] or row[column] != normalize_category(row[column]):
                raise ValueError(f"Noncanonical category in {column}.")
        if row["brand_model"] != canonical_json([row["brand"], row["model"]]):
            raise ValueError("brand_model is not the canonical brand/model pair.")
        _fuel_tokens(row["fuel_type"])
        if sha256_bytes(canonical_group_key(row).encode("utf-8")) != gid:
            raise ValueError("Group ID does not match the cleaned pre-imputation attributes.")
        # The stream necessarily reads CSV bytes, but unselected target strings
        # are neither parsed nor retained in any returned/accumulated object.
        if split != selected_split:
            continue
        price = _unsigned_integer(row["price"], "price")
        if price <= 0:
            raise ValueError("Selected targets must be positive.")
        features.append({column: row[column] for column in MODEL_FEATURES})
        targets.append(price)
        metadata.append({
            "row_id": row_id, "source_line": line, "group_id": gid,
            "split": split, "cv_fold": fold,
        })
    if seen_rows != set(manifest):
        raise ValueError("Manifest contains rows absent from cleaned.csv.")
    if seen_lines != set(range(2, len(seen_rows) + 2)):
        raise ValueError("Cleaned source lineage is not contiguous.")
    group_sizes = Counter(value[0] for value in manifest.values())
    if group_partitions != assign_partitions(group_sizes):
        raise ValueError("Manifest assignments differ from the frozen hash/balancing rules.")
    X = pd.DataFrame(features, columns=MODEL_FEATURES)
    # Float numeric arrays ensure missing capacity is np.nan even if a whole
    # selected partition has no observed capacity; categories remain strings.
    X[list(NUMERIC_FEATURES)] = X[list(NUMERIC_FEATURES)].astype(np.float64)
    y = pd.Series(targets, name="price", dtype=np.float64)
    meta = pd.DataFrame(metadata)
    return X, y, meta


def load_development(data_dir: str | Path) -> tuple:
    """Return (X, y, metadata, folds), in the order encountered in cleaned.csv.

    data_dir contains cleaned.csv and split_manifest.csv. The manifest may be
    reordered: joining is by row_id, never by position. All returned objects
    have matching zero-based RangeIndex. Each (train_pos, valid_pos) contains
    integer NumPy positions relative to X, suitable for .iloc or sklearn cv.
    Group assignments must match the frozen hash/balancing rules exactly.
    IDs, raw model, titles, years, and audit flags do not enter X.
    """
    X, y, metadata = _load_partition(data_dir, "train")
    assignments = metadata["cv_fold"].to_numpy()
    folds = [
        (np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold))
        for fold in range(N_FOLDS)
    ]
    return X, y, metadata, folds


def load_holdout(data_dir: str | Path) -> tuple:
    """Return (X, y, metadata) for final use only after methods are frozen.

    No preprocessor is fitted here. Use the same fitted development pipeline
    for final holdout transformation. This explicit access is a procedural
    boundary, not an authentication or encryption mechanism.
    """
    return _load_partition(data_dir, "test")
