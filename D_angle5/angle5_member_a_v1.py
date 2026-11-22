from pathlib import Path
import os
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DSC_DATA_DIR", ROOT.parent / "data"))
DATA_PATH = DATA_DIR / "cleaned.csv"
SPLIT_PATH = DATA_DIR / "split_manifest.csv"

OUT = ROOT / "Angle5_member_a_v1_final_results"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_DATA_SHA256 = (
    "6f4791ce8763b6c73aa3af82be9b3d3cf5065e4cd901737211476ec5b4c5097f"
)

EXPECTED_SPLIT_SHA256 = (
    "eae5f790e40e356e2f17c7df4956cf85e02e6a5a7f049a2e40d8ab8ab468a179"
)

RANDOM_STATE = 42
ALPHAS = np.logspace(2, 6, 15)
MIN_SUPPORT = 8

MAIN_MODELS = ["M0", "M1", "M2"]
ALL_MODELS = ["M0", "Age2", "Mileage2", "M1", "M2"]

NUMERIC_BASE = [
    "car_age",
    "kilometers_run",
    "energy_capacity"
]

CATEGORICAL_BASE = [
    "brand",
    "brand_model",
    "fuel_type",
    "transmission"
]

BASE_FEATURES = NUMERIC_BASE + CATEGORICAL_BASE

MODEL_NAMES = {
    "M0": "M0: Linear baseline",
    "Age2": "Age squared only",
    "Mileage2": "Mileage squared only",
    "M1": "M1: Both quadratic terms",
    "M2": "M2: + Interaction"
}

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(SPLIT_PATH)

    if sha256_file(DATA_PATH) != EXPECTED_DATA_SHA256:
        raise ValueError("cleaned.csv SHA-256 mismatch")

    if sha256_file(SPLIT_PATH) != EXPECTED_SPLIT_SHA256:
        raise ValueError("split_manifest.csv SHA-256 mismatch")

    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(SPLIT_PATH)

    if len(data) != 5997 or len(manifest) != 5997:
        raise ValueError("Expected 5997 rows")

    required_data = [
        "row_id",
        "group_id",
        "price"
    ] + BASE_FEATURES

    required_manifest = [
        "row_id",
        "group_id",
        "split",
        "cv_fold"
    ]

    for column in required_data:
        if column not in data.columns:
            raise ValueError(
                f"Missing cleaned.csv column: {column}"
            )

    for column in required_manifest:
        if column not in manifest.columns:
            raise ValueError(
                f"Missing split_manifest.csv column: {column}"
            )

    if not data["row_id"].is_unique:
        raise ValueError("Duplicate row_id in cleaned.csv")

    if not manifest["row_id"].is_unique:
        raise ValueError("Duplicate row_id in split_manifest.csv")

    merged = data.merge(
        manifest,
        on="row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_manifest")
    )

    if len(merged) != 5997:
        raise ValueError("Row ID mismatch")

    if not (
        merged["group_id"]
        == merged["group_id_manifest"]
    ).all():
        raise ValueError("Group ID mismatch")

    merged = merged.drop(
        columns=["group_id_manifest"]
    )

    if merged["group_id"].isna().any():
        raise ValueError("Missing group_id")

    if set(merged["split"].unique()) != {"train", "test"}:
        raise ValueError("Invalid split labels")

    for column in NUMERIC_BASE + ["price"]:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

    if merged["price"].isna().any():
        raise ValueError("Missing or invalid prices")

    if (merged["price"] <= 0).any():
        raise ValueError("Prices must be positive")

    development = (
        merged.loc[merged["split"] == "train"]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        merged.loc[merged["split"] == "test"]
        .copy()
        .reset_index(drop=True)
    )

    if len(development) != 4789:
        raise ValueError("Expected 4789 development records")

    if len(test) != 1208:
        raise ValueError("Expected 1208 test records")

    development["cv_fold"] = pd.to_numeric(
        development["cv_fold"],
        errors="raise"
    ).astype(int)

    if set(development["cv_fold"].unique()) != set(range(10)):
        raise ValueError("Expected folds 0 to 9")

    if not (test["cv_fold"] == -1).all():
        raise ValueError("Invalid test fold identifiers")

    if set(development["group_id"]) & set(test["group_id"]):
        raise ValueError("Group leakage between development and test")

    group_fold_counts = (
        development.groupby("group_id")["cv_fold"]
        .nunique()
    )

    if (group_fold_counts > 1).any():
        raise ValueError("Group leakage across CV folds")

    print("\nDATA VALIDATION")
    print("Data version: member-a-v1")
    print("Total records:", len(merged))
    print("Development records:", len(development))
    print("Test records:", len(test))

    print(
        "CV fold sizes:",
        development["cv_fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("\nBASE FEATURES")
    print("Numeric:", NUMERIC_BASE)
    print("Categorical:", CATEGORICAL_BASE)

    return development, test


def make_features(data, model):
    X = data[BASE_FEATURES].copy()

    age = X["car_age"].astype(float)
    mileage = X["kilometers_run"].astype(float) / 10000.0

    if model in ["Age2", "M1", "M2"]:
        X["age_squared"] = age ** 2

    if model in ["Mileage2", "M1", "M2"]:
        X["mileage_squared"] = mileage ** 2

    if model == "M2":
        X["age_x_mileage"] = age * mileage

    return X


def create_search(X):
    numeric_columns = [
        column
        for column in X.columns
        if column not in CATEGORICAL_BASE
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_BASE
        )
    ])

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            Lasso(
                max_iter=30000,
                tol=1e-4,
                random_state=RANDOM_STATE
            )
        )
    ])

    inner_cv = GroupKFold(n_splits=3)

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "regressor__alpha": ALPHAS
        },
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        n_jobs=1,
        refit=True
    )


def calculate_metrics(y_true, y_pred):
    return {
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        )
    }


def run_oof_cv(development, model):
    X = make_features(development, model)

    y = development["price"].to_numpy(dtype=float)
    folds = development["cv_fold"].to_numpy()
    groups = development["group_id"].to_numpy()

    predictions = np.full(
        len(development),
        np.nan,
        dtype=float
    )

    prediction_counts = np.zeros(
        len(development),
        dtype=int
    )

    fold_records = []

    for fold in range(10):
        train_indices = np.flatnonzero(
            folds != fold
        )

        valid_indices = np.flatnonzero(
            folds == fold
        )

        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]

        y_train = y[train_indices]
        y_valid = y[valid_indices]

        search = create_search(X_train)

        search.fit(
            X_train,
            y_train,
            groups=groups[train_indices]
        )

        predicted = search.predict(X_valid)

        predictions[valid_indices] = predicted
        prediction_counts[valid_indices] += 1

        metrics = calculate_metrics(
            y_valid,
            predicted
        )

        fold_records.append({
            "model": model,
            "fold": fold,
            "n": len(valid_indices),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "best_alpha": float(
                search.best_params_["regressor__alpha"]
            )
        })

        print(
            f"{model:9s} | "
            f"Fold {fold} | "
            f"N={len(valid_indices)} | "
            f"RMSE={metrics['rmse']:,.3f}",
            flush=True
        )

    if not np.all(prediction_counts == 1):
        raise ValueError("Invalid OOF prediction counts")

    if np.isnan(predictions).any():
        raise ValueError("Missing OOF predictions")

    fold_df = pd.DataFrame(fold_records)

    pooled = calculate_metrics(
        y,
        predictions
    )

    summary = {
        "model": model,
        "pooled_oof_rmse": pooled["rmse"],
        "pooled_oof_mae": pooled["mae"],
        "pooled_oof_r2": pooled["r2"],
        "mean_fold_rmse": float(
            fold_df["rmse"].mean()
        ),
        "sd_fold_rmse": float(
            fold_df["rmse"].std(ddof=1)
        )
    }

    predictions_df = pd.DataFrame({
        "row_id": development["row_id"],
        "group_id": development["group_id"],
        "cv_fold": development["cv_fold"],
        "y_true": y,
        "y_pred_price": predictions
    })

    predictions_df.to_csv(
        OUT / f"predictions_{model}.csv",
        index=False
    )

    return summary, fold_df, predictions_df


def fit_and_test(development, test, model):
    X_train = make_features(development, model)
    X_test = make_features(test, model)

    y_train = development["price"].to_numpy(dtype=float)
    y_test = test["price"].to_numpy(dtype=float)

    search = create_search(X_train)

    search.fit(
        X_train,
        y_train,
        groups=development["group_id"].to_numpy()
    )

    predicted = search.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predicted
    )

    predictions_df = pd.DataFrame({
        "row_id": test["row_id"],
        "y_true": y_test,
        "y_pred_price": predicted
    })

    predictions_df.to_csv(
        OUT / f"test_predictions_{model}.csv",
        index=False
    )

    test_result = {
        "model": model,
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "final_alpha": float(
            search.best_params_["regressor__alpha"]
        )
    }

    return (
        test_result,
        predictions_df,
        search.best_estimator_
    )


def improvement_percent(old_rmse, new_rmse):
    return (
        (old_rmse - new_rmse)
        / old_rmse
        * 100
    )


def save_comparison_figure(results, models, filename, title):
    ordered = (
        results.set_index("model")
        .loc[models]
    )

    values = (
        ordered["pooled_oof_rmse"].to_numpy()
        / 1_000_000
    )

    labels = [
        MODEL_NAMES[model]
        for model in models
    ]

    color_map = {
        "M0": "#999999",
        "Age2": "#D5AC8B",
        "Mileage2": "#BD825D",
        "M1": "#AD5F3C",
        "M2": "#763F2D"
    }

    colors = [
        color_map[model]
        for model in models
    ]

    positions = np.arange(len(models))

    fig, ax = plt.subplots(
        figsize=(10, 5.6 if len(models) == 5 else 4.7)
    )

    ax.barh(
        positions,
        values,
        color=colors,
        height=0.55
    )

    ax.set_yticks(positions, labels)
    ax.invert_yaxis()

    ax.set_xlim(
        0,
        max(values) * 1.20
    )

    for pos, val in zip(positions, values):
        ax.text(
            val + max(values) * 0.025,
            pos,
            f"{val:.4f} M",
            va="center",
            fontsize=11
        )

    ax.set_title(title)

    ax.set_xlabel(
        "Pooled OOF RMSE "
        "(million price units; lower is better)"
    )

    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUT / filename,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_interpretation_figures(development, fitted_m2):
    reference_candidates = development

    toyota = development.loc[
        development["brand"]
        .astype(str)
        .str.lower() == "toyota"
    ]

    if not toyota.empty:
        reference_candidates = toyota

    distance = (
        (reference_candidates["car_age"] - 6).abs()
        +
        (
            reference_candidates["kilometers_run"] - 60000
        ).abs() / 10000
    )

    reference = (
        reference_candidates
        .loc[distance.idxmin(), BASE_FEATURES]
        .copy()
        .to_dict()
    )

    observed_age = development["car_age"].to_numpy(
        dtype=float
    )

    observed_mileage = development["kilometers_run"].to_numpy(
        dtype=float
    )

    def support_count(age, mileage):
        nearby = (
            (np.abs(observed_age - age) <= 1.5)
            &
            (np.abs(observed_mileage - mileage) <= 15000)
        )

        return int(nearby.sum())

    def make_grid(ages, mileages):
        rows = []

        for age, mileage in zip(ages, mileages):
            row = reference.copy()

            row["car_age"] = float(age)
            row["kilometers_run"] = float(mileage)

            rows.append(row)

        grid = pd.DataFrame(
            rows,
            columns=BASE_FEATURES
        )

        return make_features(grid, "M2")

    age_start = max(
        0,
        float(np.percentile(observed_age, 5))
    )

    age_end = min(
        25,
        float(np.percentile(observed_age, 95))
    )

    ages = np.linspace(
        age_start,
        age_end,
        120
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for mileage in [20000, 60000, 100000]:
        mileage_grid = np.full_like(
            ages,
            mileage,
            dtype=float
        )

        X_grid = make_grid(
            ages,
            mileage_grid
        )

        predicted = (
            fitted_m2.predict(X_grid)
            / 1_000_000
        )

        supported = np.array([
            support_count(age, mileage) >= MIN_SUPPORT
            for age in ages
        ])

        predicted[~supported] = np.nan

        ax.plot(
            ages,
            predicted,
            linewidth=2.5,
            label=f"{mileage:,} km"
        )

    ax.set_xlabel("Car Age (Years)")
    ax.set_ylabel("Predicted Price (million price units)")

    ax.set_title(
        "Predicted Price by Age at Different Mileage Levels"
    )

    ax.legend(title="Mileage")
    ax.grid(alpha=0.25)

    fig.text(
        0.5,
        0.01,
        "Other vehicle features held constant; "
        "unsupported combinations omitted.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    fig.savefig(
        OUT / "Figure3_Nonlinear_Interaction_Curves.png",
        bbox_inches="tight"
    )

    plt.close(fig)

    age_values = np.arange(
        max(0, int(np.floor(age_start))),
        int(np.ceil(age_end)) + 1
    )

    mileage_values = np.arange(
        0,
        160001,
        10000
    )

    A, M = np.meshgrid(
        age_values,
        mileage_values
    )

    X_grid = make_grid(
        A.ravel(),
        M.ravel()
    )

    prices = (
        fitted_m2.predict(X_grid)
        .reshape(A.shape)
        / 1_000_000
    )

    supports = np.array([
        support_count(age, mileage)
        for age, mileage in zip(
            A.ravel(),
            M.ravel()
        )
    ]).reshape(A.shape)

    prices = np.where(
        supports >= MIN_SUPPORT,
        prices,
        np.nan
    )

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=(10, 6))

    heatmap = ax.imshow(
        np.ma.masked_invalid(prices),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest"
    )

    ax.set_xticks(
        np.arange(len(age_values))[::2],
        age_values[::2]
    )

    ax.set_yticks(
        np.arange(len(mileage_values))[::2],
        mileage_values[::2] // 1000
    )

    ax.set_xlabel("Car Age (Years)")
    ax.set_ylabel("Mileage (thousand km)")

    ax.set_title(
        "Predicted Price by Age and Mileage"
    )

    fig.colorbar(
        heatmap,
        ax=ax
    ).set_label(
        "Predicted Price (million price units)"
    )

    fig.text(
        0.5,
        0.01,
        "Gray cells: fewer than 8 nearby observations.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    fig.savefig(
        OUT / "Figure4_Age_Mileage_Heatmap.png",
        bbox_inches="tight"
    )

    plt.close(fig)


def save_config(selected_model):
    config = {
        "data_version": "member-a-v1",
        "data_sha256": EXPECTED_DATA_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "total_rows": 5997,
        "development_rows": 4789,
        "test_rows": 1208,
        "outer_cv": "Predefined 10 folds",
        "inner_cv": "GroupKFold(n_splits=3)",
        "estimator": "Lasso",
        "random_state": RANDOM_STATE,
        "alpha_grid": ALPHAS.tolist(),
        "numeric_features": NUMERIC_BASE,
        "categorical_features": CATEGORICAL_BASE,
        "base_features": BASE_FEATURES,
        "categorical_encoding": "OneHotEncoder",
        "numeric_imputation": "Median within training fold",
        "numeric_scaling": "StandardScaler within training fold",
        "mileage_polynomial_unit": "10000 km",
        "models": {
            "M0": "Original seven feature groups",
            "Age2": "M0 + age_squared",
            "Mileage2": "M0 + mileage_squared",
            "M1": "M0 + age_squared + mileage_squared",
            "M2": "M1 + age_x_mileage"
        },
        "primary_metric": "Pooled OOF RMSE",
        "selected_main_model": selected_model
    }

    with open(
        OUT / "model_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False
        )


def save_manifest():
    manifest = {
        "data_version": "member-a-v1",
        "input_files": {
            DATA_PATH.name: sha256_file(DATA_PATH),
            SPLIT_PATH.name: sha256_file(SPLIT_PATH)
        },
        "outputs": {}
    }

    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = sha256_file(path)

    script_path = Path(__file__).resolve()

    manifest["code"] = {
        script_path.name: sha256_file(script_path)
    }

    with open(
        OUT / "manifest.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():
    development, test = load_data()

    summaries = []
    fold_tables = []
    oof_tables = []

    for model in ALL_MODELS:
        print(f"\nRUNNING CV: {model}", flush=True)

        summary, fold_df, predictions_df = run_oof_cv(
            development,
            model
        )

        summaries.append(summary)
        fold_tables.append(fold_df)

        oof_tables.append(
            predictions_df.assign(model=model)
        )

    results = pd.DataFrame(summaries)

    fold_results = pd.concat(
        fold_tables,
        ignore_index=True
    )

    combined_oof = pd.concat(
        oof_tables,
        ignore_index=True
    )

    results.to_csv(
        OUT / "results.csv",
        index=False,
        float_format="%.12f"
    )

    fold_results.to_csv(
        OUT / "fold_results.csv",
        index=False,
        float_format="%.12f"
    )

    combined_oof.to_csv(
        OUT / "predictions_all_models.csv",
        index=False
    )

    rmse_table = results.set_index("model")["pooled_oof_rmse"]

    selected_model = rmse_table.loc[
        MAIN_MODELS
    ].idxmin()

    selected_oof = pd.read_csv(
        OUT / f"predictions_{selected_model}.csv"
    )

    selected_oof.to_csv(
        OUT / "predictions.csv",
        index=False
    )

    test_results = []
    test_prediction_tables = []
    fitted_models = {}

    for model in ALL_MODELS:
        print(
            f"\nFINAL FIT AND TEST: {model}",
            flush=True
        )

        test_metrics, predictions_df, estimator = (
            fit_and_test(
                development,
                test,
                model
            )
        )

        test_results.append(test_metrics)

        test_prediction_tables.append(
            predictions_df.assign(model=model)
        )

        fitted_models[model] = estimator

    test_results_df = pd.DataFrame(test_results)

    test_results_df.to_csv(
        OUT / "test_results.csv",
        index=False
    )

    pd.concat(
        test_prediction_tables,
        ignore_index=True
    ).to_csv(
        OUT / "test_predictions_all_models.csv",
        index=False
    )

    selected_test = pd.read_csv(
        OUT / f"test_predictions_{selected_model}.csv"
    )

    selected_test.to_csv(
        OUT / "test_predictions.csv",
        index=False
    )

    improvements = pd.DataFrame([
        {
            "comparison": "M0_to_Age2",
            "rmse_reduction_pct": improvement_percent(
                rmse_table["M0"],
                rmse_table["Age2"]
            )
        },
        {
            "comparison": "M0_to_Mileage2",
            "rmse_reduction_pct": improvement_percent(
                rmse_table["M0"],
                rmse_table["Mileage2"]
            )
        },
        {
            "comparison": "M0_to_M1",
            "rmse_reduction_pct": improvement_percent(
                rmse_table["M0"],
                rmse_table["M1"]
            )
        },
        {
            "comparison": "M1_to_M2",
            "rmse_reduction_pct": improvement_percent(
                rmse_table["M1"],
                rmse_table["M2"]
            )
        }
    ])

    improvements.to_csv(
        OUT / "improvements.csv",
        index=False
    )

    save_comparison_figure(
        results,
        ALL_MODELS,
        "Figure1_Five_Model_Comparison.png",
        "Angle 5: Incremental Predictive Value"
    )

    save_comparison_figure(
        results,
        MAIN_MODELS,
        "Figure2_Main_Model_Comparison.png",
        "M0 vs. M1 vs. M2"
    )

    save_interpretation_figures(
        development,
        fitted_models["M2"]
    )

    save_config(selected_model)
    save_manifest()

    print("\nPOOLED OOF RESULTS")
    print(
        results.round(4).to_string(index=False)
    )

    print("\nINCREMENTAL IMPROVEMENTS")
    print(
        improvements.round(4).to_string(index=False)
    )

    print("\nINDEPENDENT TEST RESULTS")
    print(
        test_results_df.round(4).to_string(index=False)
    )

    print("\nSelected model:", selected_model)

    print("\nALL ANALYSES COMPLETED")
    print("Output directory:", OUT.resolve())

    print("\nGenerated files:")

    for path in sorted(OUT.iterdir()):
        print("-", path.name)


if __name__ == "__main__":
    main()
