"""Build the frozen member-a-v1 data and audit artifacts using only the stdlib.

Usage: python build_dataset.py --source UsedCarInfo.csv --out new_directory
The output directory must be absent or empty. No learned preprocessing occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


CONFIG_VERSION = "member-a-v1"
SOURCE_SHA256 = "bc16a88f3eff81aa293feec7efa2aad45e54a32a96ae21f54440fbffc7419006"
EXPECTED_ROWS = 5997
SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 10
REFERENCE_YEAR = 2026
DOCUMENTED_YEAR_RANGE = (1996, 2025)
SOURCE_COLUMNS = (
    "title", "brand", "model", "transmission", "fuel_type", "price",
    "year_of_manufacture", "energy_capacity", "kilometers_run", "car_age",
)
CLEANED_COLUMNS = (
    "row_id", "source_line", "group_id", "price", "car_age", "kilometers_run",
    "energy_capacity", "brand", "model", "brand_model", "fuel_type", "transmission",
)
SPLIT_COLUMNS = ("row_id", "group_id", "split", "cv_fold")
GROUP_KEY_FEATURES = (
    "brand", "model", "transmission", "fuel_type", "car_age",
    "kilometers_run", "energy_capacity",
)
ACTION_COLUMNS = ("row_id", "source_line", "column", "original", "cleaned", "reason")
QUALITY_COLUMNS = (
    "row_id", "source_line", "title_original", "year_of_manufacture_original",
    "energy_capacity_original", "flags",
)
CAPACITY_TEXT_LINE = 5840
CAPACITY_TEXT = "Toyota Passo 2005 reg 2010 1000cc"
SUSPECT_CAPACITY_LINES = frozenset({96, 3029})
FORMATTED_INTEGER = r"(?:[0-9]{1,3}(?:,[0-9]{3})*)"


def canonical_json(value: object) -> str:
    """Canonical UTF-8 JSON: sorted keys, compact separators, no ASCII escaping."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_category(value: str) -> str:
    """NFKC, collapse Unicode whitespace, casefold, then NFKC again."""
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    return unicodedata.normalize("NFKC", normalized)


def canonical_group_key(row: dict) -> str:
    """Serialize only the seven pre-imputation attributes; never inspect price."""
    values = [row[column] for column in GROUP_KEY_FEATURES]
    values[3] = row["fuel_type"].split("|")
    return canonical_json(values)


def assign_partitions(group_sizes: dict[str, int]) -> dict[str, tuple[str, int]]:
    """Return group -> (split, fold) using frozen, target-free hash ordering."""
    test_order = sorted(
        group_sizes,
        key=lambda gid: (sha256_bytes(f"test|{SEED}|{gid}".encode("utf-8")), gid),
    )
    n_test = math.ceil(TEST_FRACTION * len(test_order))
    test_groups = set(test_order[:n_test])
    development = [gid for gid in test_order if gid not in test_groups]
    if not test_groups or len(development) < N_FOLDS:
        raise ValueError("Insufficient groups for a holdout and ten nonempty folds.")
    development.sort(key=lambda gid: (
        -group_sizes[gid], sha256_bytes(f"cv|{SEED}|{gid}".encode("utf-8")), gid,
    ))
    assignments = {gid: ("test", -1) for gid in test_groups}
    fold_sizes = [0] * N_FOLDS
    for gid in development:
        fold = min(range(N_FOLDS), key=lambda index: (fold_sizes[index], index))
        assignments[gid] = ("train", fold)
        fold_sizes[fold] += group_sizes[gid]
    return assignments


def _parse_integer(value: str, pattern: str, column: str, line: int) -> int:
    match = re.fullmatch(pattern, value)
    if match is None:
        raise ValueError(f"Invalid {column} format at source_line {line}.")
    return int(match.group(1).replace(",", ""))


def _read_source(source: Path) -> list[dict[str, str]]:
    # Validate the exact bytes before parsing or applying any exception whitelist.
    payload = source.read_bytes()
    if sha256_bytes(payload) != SOURCE_SHA256:
        raise ValueError("Source SHA256 does not match the frozen official CSV.")
    reader = csv.reader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
    if tuple(next(reader, [])) != SOURCE_COLUMNS:
        raise ValueError("Source columns or column order do not match the specification.")
    records = []
    for line, cells in enumerate(reader, start=2):
        if len(cells) != len(SOURCE_COLUMNS) or reader.line_num != line:
            raise ValueError(f"Malformed or multiline source record at source_line {line}.")
        if any(not cell.strip() for cell in cells):
            raise ValueError(f"Unexpected empty source cell at source_line {line}.")
        records.append(dict(zip(SOURCE_COLUMNS, cells)))
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source records, got {len(records)}.")
    return records


def _clean_records(records: list[dict[str, str]]) -> tuple[list, list, list]:
    cleaned, actions, quality = [], [], []
    group_keys: dict[str, str] = {}
    for line, raw in enumerate(records, start=2):
        row_id = f"ucv2-{line - 1:06d}"
        row = {"row_id": row_id, "source_line": line}
        reasons = {}
        flags = set()
        for column in ("brand", "model", "transmission"):
            row[column] = normalize_category(raw[column])
            reasons[column] = "category_nfkc_whitespace_casefold"
        tokens = [normalize_category(token) for token in raw["fuel_type"].split(",")]
        if any(not token or "|" in token for token in tokens):
            raise ValueError(f"Invalid fuel token at source_line {line}.")
        row["fuel_type"] = "|".join(sorted(set(tokens)))
        reasons["fuel_type"] = "fuel_nfkc_whitespace_casefold_sorted_unique_tokens"
        row["brand_model"] = canonical_json([row["brand"], row["model"]])
        row["price"] = _parse_integer(
            raw["price"], rf"Tk ({FORMATTED_INTEGER})", "price", line,
        )
        row["kilometers_run"] = _parse_integer(
            raw["kilometers_run"], rf"({FORMATTED_INTEGER}) km", "kilometers_run", line,
        )
        row["car_age"] = _parse_integer(raw["car_age"], r"([0-9]+)", "car_age", line)
        year = _parse_integer(
            raw["year_of_manufacture"], r"([0-9]{4})", "year_of_manufacture", line,
        )
        if row["price"] <= 0 or row["kilometers_run"] < 0:
            raise ValueError(f"Invalid target or mileage at source_line {line}.")
        if row["car_age"] != REFERENCE_YEAR - year:
            raise ValueError(f"Age/year inconsistency at source_line {line}.")
        for column in ("price", "kilometers_run", "car_age"):
            reasons[column] = "strict_integer_unit_and_separator_parsing"

        original_cc = raw["energy_capacity"]
        if line == CAPACITY_TEXT_LINE:
            if original_cc != CAPACITY_TEXT:
                raise ValueError("Capacity text whitelist value mismatch.")
            row["energy_capacity"] = int(re.search(r"([0-9]+)cc$", original_cc).group(1))
            reasons["energy_capacity"] = "whitelisted_explicit_1000cc_suffix_recovery"
            flags.add("capacity_text_recovered")
        else:
            row["energy_capacity"] = _parse_integer(
                original_cc, rf"({FORMATTED_INTEGER}) cc", "energy_capacity", line,
            )
            reasons["energy_capacity"] = "strict_integer_unit_and_separator_parsing"
        if line in SUSPECT_CAPACITY_LINES:
            if original_cc != "150 cc":
                raise ValueError("Suspect capacity whitelist value mismatch.")
            row["energy_capacity"] = None
            reasons["energy_capacity"] = "suspect_150cc_set_missing_not_proven_error"
            flags.add("capacity_150cc_suspect")
        if set(tokens) == {"electric"}:
            row["energy_capacity"] = None
            reasons["energy_capacity"] = "pure_electric_capacity_semantics_unclear_set_missing"
            flags.add("pure_electric_capacity_missing")

        km = row["kilometers_run"]
        if km == 0:
            flags.add("zero_km")
        if 0 < km < 100:
            flags.add("positive_km_under_100")
        if km > 300000:
            flags.add("km_over_300000")
        if re.search(r"\bbrand new\b", normalize_category(raw["title"])):
            flags.add("title_brand_new")
        if "electric" in tokens:
            flags.add("electric_token_present")
        if year < DOCUMENTED_YEAR_RANGE[0]:
            flags.update(("year_below_documented_min", "age_above_documented_year_range"))
        if year > DOCUMENTED_YEAR_RANGE[1]:
            flags.update(("year_above_documented_max", "age_below_documented_year_range"))
        if any(
            value == "unknown" or value.startswith("other")
            for value in (row["brand"], row["model"], row["transmission"], *tokens)
        ):
            flags.add("placeholder_category_retained")
        if row["energy_capacity"] is None:
            flags.add("capacity_missing")

        key = canonical_group_key(row)
        gid = sha256_bytes(key.encode("utf-8"))
        if gid in group_keys and group_keys[gid] != key:
            raise ValueError("SHA256 group-key collision detected.")
        group_keys[gid] = key
        row["group_id"] = gid
        cleaned.append(row)
        # One final action per changed source cell, including deterministic formatting.
        # Derived fields have no original source cell and are not cleaning actions.
        for column in SOURCE_COLUMNS:
            if column not in reasons:
                continue
            value = "" if row[column] is None else str(row[column])
            if value != raw[column]:
                actions.append({
                    "row_id": row_id, "source_line": line, "column": column,
                    "original": raw[column], "cleaned": value, "reason": reasons[column],
                })
        quality.append({
            "row_id": row_id, "source_line": line,
            "title_original": raw["title"],
            "year_of_manufacture_original": raw["year_of_manufacture"],
            "energy_capacity_original": original_cc, "flags": "|".join(sorted(flags)),
        })
    return cleaned, actions, quality


def _write_csv(path: Path, columns: tuple[str, ...], records: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        stream.write("\n")


def build_dataset(source: str | Path, out: str | Path) -> dict:
    """Build all artifacts and return the summary; refuse to overwrite any output.

    Inputs are fully parsed before creating artifacts. If I/O fails mid-write,
    partial files are retained for inspection; file_manifest.json is written last
    as the completion marker. A subsequent build refuses that nonempty directory.
    """
    source, out = Path(source), Path(out)
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("Output must be a new or empty nonsymlink directory.")
    records = _read_source(source)
    cleaned, actions, quality = _clean_records(records)
    group_sizes = Counter(row["group_id"] for row in cleaned)
    partitions = assign_partitions(group_sizes)
    splits = [
        {"row_id": row["row_id"], "group_id": row["group_id"],
         "split": partitions[row["group_id"]][0], "cv_fold": partitions[row["group_id"]][1]}
        for row in cleaned
    ]

    out.mkdir(parents=True, exist_ok=True)
    if out.is_symlink() or any(out.iterdir()):
        raise ValueError("Output directory became nonempty before writing.")
    # Exclusive subdirectory and file creation prevents accidental overwrites.
    (out / "data").mkdir()
    (out / "audit").mkdir()
    _write_csv(out / "data/cleaned.csv", CLEANED_COLUMNS, cleaned)
    _write_csv(out / "data/split_manifest.csv", SPLIT_COLUMNS, splits)
    _write_csv(out / "audit/cleaning_actions.csv", ACTION_COLUMNS, actions)
    _write_csv(out / "audit/row_quality.csv", QUALITY_COLUMNS, quality)
    summary = {
        "config_version": CONFIG_VERSION,
        "source": {"artifact": "UsedCarInfo.csv", "sha256": SOURCE_SHA256},
        "cleaned_sha256": sha256_bytes((out / "data/cleaned.csv").read_bytes()),
        "row_flow": {"input": len(records), "output": len(cleaned), "removed": 0},
        "configuration": {
            "seed": SEED, "test_fraction_groups": TEST_FRACTION, "cv_folds": N_FOLDS,
            "reference_year": REFERENCE_YEAR, "documented_year_range": DOCUMENTED_YEAR_RANGE,
            "category_normalization": "NFKC -> whitespace collapse -> casefold -> NFKC",
            "fuel_serialization": "sorted unique normalized tokens joined by |",
            "brand_model_serialization": "canonical JSON array [brand,model]",
            "group_key_fields_in_order": GROUP_KEY_FEATURES,
            "group_key_serialization": (
                "UTF-8 JSON array in declared order; fuel is a sorted token array; "
                "numeric values are integers, missing capacity is null; "
                "ensure_ascii=false, separators=(',',':'), allow_nan=false"
            ),
            "group_id": "lowercase SHA256 hex digest of canonical group-key UTF-8 bytes",
            "holdout_assignment": (
                "sort (SHA256(UTF-8('test|42|'+group_id)), group_id); "
                "first ceil(0.20 * number_of_groups) groups are test"
            ),
            "cv_assignment": (
                "remaining groups sorted by (-size,SHA256(UTF-8('cv|42|'+group_id)),group_id); "
                "assign each to min((current_row_count,fold_number)) for folds 0..9"
            ),
            "capacity_text_whitelist": {
                "source_line": CAPACITY_TEXT_LINE, "original": CAPACITY_TEXT, "cleaned": 1000,
                "source_sha256": SOURCE_SHA256,
            },
            "suspect_150cc_lines_set_missing": sorted(SUSPECT_CAPACITY_LINES),
            "pure_electric_capacity": "set missing only for token set exactly {electric}",
            "imputation_encoding_scaling": "not performed by builder; fit within training folds",
            "cleaning_action_scope": "changed source cells, including format normalization; no derived fields",
        },
        "action_counts": {
            "total": len(actions),
            "by_reason": dict(sorted(Counter(row["reason"] for row in actions).items())),
            "by_column": dict(sorted(Counter(row["column"] for row in actions).items())),
        },
        "quality_flag_counts": dict(sorted(Counter(
            flag for row in quality for flag in row["flags"].split("|") if flag
        ).items())),
        "null_counts": {
            column: sum(row[column] is None or row[column] == "" for row in cleaned)
            for column in CLEANED_COLUMNS
        },
        "groups": {
            "total": len(group_sizes), "max_size": max(group_sizes.values()),
            "size_histogram": {str(size): count for size, count in sorted(Counter(group_sizes.values()).items())},
        },
        "splits": {
            split: {
                "rows": sum(row["split"] == split for row in splits),
                "groups": sum(value[0] == split for value in partitions.values()),
            } for split in ("train", "test")
        },
        "folds": [
            {"cv_fold": fold, "rows": sum(row["cv_fold"] == fold for row in splits),
             "groups": sum(value == ("train", fold) for value in partitions.values())}
            for fold in range(N_FOLDS)
        ],
        "limitations": [
            "The source methodology reports upstream imputation; original missingness is unknown.",
            "1000cc recovery has text evidence, not original platform verification.",
            "150cc is suspect, not proven erroneous; pure-electric capacity semantics are unclear.",
            "Later imputed capacity is a computational placeholder, not a factual engine specification.",
            "Attribute groups are not vehicle IDs and cannot establish physical-vehicle independence.",
            "Held-out targets remain in cleaned.csv and source; the embargo is procedural, not cryptographic.",
        ],
        "artifacts": [
            "data/cleaned.csv", "data/split_manifest.csv", "audit/cleaning_actions.csv",
            "audit/row_quality.csv", "audit/build_summary.json", "audit/file_manifest.json",
        ],
    }
    _write_json(out / "audit/build_summary.json", summary)
    manifest = {}
    for relative in summary["artifacts"]:
        if relative == "audit/file_manifest.json":
            continue
        payload = (out / relative).read_bytes()
        manifest[relative] = {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}
    _write_json(out / "audit/file_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Frozen official source CSV")
    parser.add_argument("--out", required=True, type=Path, help="New or empty output directory")
    args = parser.parse_args()
    try:
        summary = build_dataset(args.source, args.out)
    except (ValueError, csv.Error) as exc:
        parser.exit(2, f"Build failed: {exc}\n")
    except OSError as exc:
        # Avoid including host paths in logs copied into distributed artifacts.
        parser.exit(2, f"Build failed: file operation error ({exc.strerror}).\n")
    print(f"Built {summary['row_flow']['output']} records; {summary['groups']['total']} attribute groups.")


if __name__ == "__main__":
    main()
