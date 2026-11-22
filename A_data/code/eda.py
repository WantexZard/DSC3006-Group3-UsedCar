"""Training-only descriptive plots. No fitting, imputation, trimming or sampling.

Usage: python -B eda.py --data-dir BUILD --code-dir CODE --out OUTPUT
Dependencies: the shared loader's dependencies, pandas, numpy and matplotlib.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.data_dir.is_dir() or not args.code_dir.is_dir():
        parser.error("Data and code directories must exist.")
    if not (args.code_dir / "shared_preprocessing.py").is_file():
        parser.error("Code directory must contain shared_preprocessing.py.")
    out = args.out.resolve()
    if not out.parent.is_dir():
        parser.error("Output parent directory must already exist.")
    out.mkdir(exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    # Keep caches inside the requested output; never create caches in shared code.
    sys.dont_write_bytecode = True
    os.environ["MPLCONFIGDIR"] = str(out / ".mplconfig")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(args.code_dir.resolve()))
    from shared_preprocessing import load_development
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    X, y, meta, _folds = load_development(args.data_dir)
    if len(X) == 0 or not (len(X) == len(y) == len(meta)):
        raise ValueError("Empty or misaligned training data.")
    if not meta["split"].eq("train").all() or not X.index.equals(y.index):
        raise ValueError("Expected aligned training-only observations.")
    n = len(X)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "svg.fonttype": "none",
        "svg.hashsalt": "training-eda", "savefig.facecolor": "white",
    })
    blue, orange = "#23658A", "#B65E28"
    source = "Training only | Source: Mendeley v2; member-a-v1 via load_development"
    figures = []

    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")

    def save(fig, stem, title, description, note):
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        fig.text(0.02, 0.025, source + f" | n = {n:,}\n" + note,
                 fontsize=9, va="bottom", linespacing=1.6)
        fig.tight_layout(rect=(0, 0.115, 1, 0.92), w_pad=3)
        for ext in ("png", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                        metadata={"Creator": "Training EDA"} if ext == "svg" else None)
        figures.append({"id": stem, "title": title, "description": description,
                        "files": [f"{stem}.png", f"{stem}.svg"],
                        "caption": source + f" | n = {n}; " + note})
        plt.close(fig)

    numeric = X[["car_age", "kilometers_run", "energy_capacity"]].copy()
    numeric["price_BDT"] = y
    numeric["ln_price_BDT"] = np.log(y)
    units = {"car_age": "years", "kilometers_run": "km", "energy_capacity": "cc",
             "price_BDT": "BDT", "ln_price_BDT": "natural log of price / 1 BDT"}
    stats = {}
    for col in numeric:
        s = numeric[col]
        q = s.quantile([0, .25, .5, .75, 1]).tolist()
        vals = dict(zip(("min", "q1", "median", "q3", "max"), q))
        vals["mean"] = s.mean()
        stats[col] = {"unit": units[col], "observed": int(s.notna().sum()),
                      "missing": int(s.isna().sum()),
                      **{k: None if pd.isna(v) else float(v) for k, v in vals.items()}}
    pd.DataFrame.from_dict(stats, orient="index").rename_axis("variable").to_csv(
        tables / "numeric_summary.csv")

    def frequencies(s, label):
        counts = s.value_counts(dropna=False).rename_axis(label).reset_index(name="count")
        counts = counts.sort_values(["count", label], ascending=[False, True]).reset_index(drop=True)
        counts["share_of_training_rows"] = counts["count"] / n
        return counts

    brands = frequencies(X["brand"], "brand")
    models = frequencies(X["brand_model"], "canonical_pair")
    pairs = models.pop("canonical_pair").map(json.loads)
    models.insert(0, "brand", pairs.map(lambda pair: pair[0]))
    models.insert(1, "model", pairs.map(lambda pair: pair[1]))
    models.insert(2, "label", pairs.map(lambda pair: " / ".join(pair)))
    fuel_counts = Counter(token for value in X["fuel_type"] for token in value.split("|"))
    fuel = pd.DataFrame(sorted(fuel_counts.items(), key=lambda p: (-p[1], p[0])),
                        columns=["fuel_token", "count"])
    fuel["share_of_training_rows"] = fuel["count"] / n
    transmission = frequencies(X["transmission"], "transmission")
    coverage = []
    for name, frame in (("brand", brands), ("brand_model", models)):
        counts = frame["count"]
        coverage.append({"category": name, "observed_categories": len(frame),
                         "singleton_categories": int(counts.eq(1).sum()),
                         "rare_categories_n_lt_5": int(counts.lt(5).sum()),
                         "rows_in_rare_categories": int(counts[counts.lt(5)].sum()),
                         "top_12_row_share": float(counts.iloc[:12].sum() / n)})
    for name, frame in (("brand_counts", brands), ("brand_model_counts", models),
                        ("fuel_token_counts", fuel), ("transmission_counts", transmission),
                        ("category_coverage", pd.DataFrame(coverage))):
        frame.to_csv(tables / f"{name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, values, label, color in zip(
            axes, (y / 1e6, np.log(y)),
            ("Listing price (million BDT; linear scale)", "ln(price / 1 BDT)"), (blue, orange)):
        ax.hist(values, bins=40, color=color, edgecolor="white", linewidth=.4)
        ax.set(xlabel=label, ylabel="Training records", title="Original price" if ax is axes[0] else "Natural-log price")
        ax.grid(axis="y", alpha=.18)
    save(fig, "01_price_distributions", "Training listing-price distributions",
         "Forty equal-width bins on each displayed scale; all training prices retained.",
         "Full ranges retained. Log display does not imply equivalent errors on the original BDT scale.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
    for ax, frame, label, detail in zip(axes, (brands, models), ("brand", "label"), coverage):
        top = frame.iloc[:12].iloc[::-1]
        labels = [textwrap.fill(value, 27) for value in top[label]]
        bars = ax.barh(range(len(top)), top["count"], color=blue, height=.68)
        ax.set_yticks(range(len(top)), labels, fontsize=9)
        ax.bar_label(bars, padding=4, fontsize=9, fmt="%d")
        ax.set_xlim(0, max(top["count"]) * 1.17)
        ax.set_xlabel("Training records (linear scale)")
        ax.set_title(f"{detail['category'].replace('_', ' ').title()}: top {len(top)} of {len(frame)}\n"
                     f"Rare (n < 5): {detail['rare_categories_n_lt_5']} categories; "
                     f"{detail['rows_in_rare_categories']} records", fontsize=11)
        ax.grid(axis="x", alpha=.18)
    save(fig, "02_category_coverage", "Brand and within-brand model coverage",
         "Most frequent observed training categories; full frequencies in tables. Rare means fewer than five training records.",
         "Rare = 1-4 training records per category; no category merging. Full counts are provided as CSV tables.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, values, label in zip(axes, (X.car_age, np.log1p(X.kilometers_run)),
                                ("Car age (years; linear scale)", "ln(1 + mileage / 1 km)")):
        ax.scatter(values, y / 1e6, s=11, alpha=.28, color=blue, edgecolors="none")
        ax.set(xlabel=label, ylabel="Listing price (million BDT; log scale)", yscale="log")
        ax.grid(alpha=.18)
    save(fig, "03_age_mileage_price", "Age, mileage and listing price",
         "All training records plotted without fitted trends. Price uses a log axis; mileage is explicitly transformed with ln(1 + km).",
         "All extremes retained; no trend fitted. ln(1 + mileage / 1 km) includes zero-mileage records.")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for ax, mileage, label in zip(axes, (X.kilometers_run / 1000, np.log1p(X.kilometers_run)),
                                  ("Mileage (thousand km; linear scale)", "ln(1 + mileage / 1 km)")):
        hb = ax.hexbin(X.car_age, mileage, gridsize=30, mincnt=1,
                       norm=LogNorm(), cmap="viridis", linewidths=.15)
        ax.set(xlabel="Car age (years; linear scale)", ylabel=label)
        cb = fig.colorbar(hb, ax=ax, pad=.025)
        cb.set_label("Records per hexagon (log color scale)", fontsize=9)
    save(fig, "04_age_mileage_support", "Observed age-mileage sample support",
         "Occupied hexagons only, in original and transformed mileage coordinates. Blank areas have no observed bin counts; no extrapolation curves.",
         "Full observed ranges; blank hexagons contain no records. Bins differ across panels; no extrapolation fitted.")

    # Fingerprint only the returned training objects, never reopen the cleaned CSV.
    training_csv = pd.concat([meta, X, y], axis=1).to_csv(index=False, lineterminator="\n")
    summary = {
        "n": n, "split": "train", "source": "Mendeley v2 / member-a-v1",
        "loader": "load_development", "models_trained": False,
        "training_content_sha256": hashlib.sha256(training_csv.encode()).hexdigest(),
        "numeric": stats, "missing_by_feature": X.isna().sum().astype(int).to_dict(),
        "brand_counts": brands.to_dict("records"), "brand_model_counts": models.to_dict("records"),
        "category_coverage": coverage, "fuel_tokens": fuel.to_dict("records"),
        "fuel_count_note": "Each token counted once per record; multi-label totals may exceed n.",
        "transmission_counts": transmission.to_dict("records"),
        "preprocessing": "None fitted; missing values retained; no clipping, trimming or sampling.",
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "code_sha256": {name: hashlib.sha256((args.code_dir / name).read_bytes()).hexdigest()
                        for name in ("shared_preprocessing.py", "build_dataset.py")},
        "eda_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(out / "EDA_summary.json", summary)
    write_json(out / "manifest.json", {
        "split": "train", "n": n, "figures": figures,
        "summary": "EDA_summary.json",
        "tables": sorted(str(p.relative_to(out)) for p in tables.glob("*.csv")),
        "training_content_sha256": summary["training_content_sha256"],
    })
    print(json.dumps({"status": "complete", "n": n, "figures": len(figures),
                      "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
