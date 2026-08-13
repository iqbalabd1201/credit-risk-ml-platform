import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.features import build_v2_dataset


RANDOM_STATE = 42
FINAL_N_ESTIMATORS = 1000

MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 2500,
    "learning_rate": 0.025,
    "num_leaves": 40,
    "min_child_samples": 40,
    "colsample_bytree": 0.90,
    "subsample": 0.85,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_data(data_dir: Path):
    application = pd.read_csv(data_dir / "application_train.csv")
    bureau = pd.read_csv(data_dir / "bureau.csv")
    bureau_balance = pd.read_csv(data_dir / "bureau_balance.csv")
    previous = pd.read_csv(data_dir / "previous_application.csv")
    installments = pd.read_csv(data_dir / "installments_payments.csv")

    return (
        application,
        bureau,
        bureau_balance,
        previous,
        installments,
    )


def prepare_features(df: pd.DataFrame):
    y = df["TARGET"].copy()
    X = df.drop(columns=["TARGET"])

    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if "SK_ID_CURR" in numeric_cols:
        numeric_cols.remove("SK_ID_CURR")

    feature_cols = numeric_cols + categorical_cols

    X = X[feature_cols].copy()

    for col in categorical_cols:
        X[col] = X[col].astype("category")

    return X, y, feature_cols, categorical_cols


def run_cross_validation(X, y):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_prob = np.zeros(len(X))
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**MODEL_PARAMS)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0),
            ],
        )

        prob = model.predict_proba(X_valid)[:, 1]
        oof_prob[valid_idx] = prob

        roc = roc_auc_score(y_valid, prob)
        pr = average_precision_score(y_valid, prob)

        fold_results.append(
            {
                "fold": fold,
                "roc_auc": float(roc),
                "pr_auc": float(pr),
                "best_iteration": int(model.best_iteration_),
            }
        )

        print(
            f"Fold {fold} | "
            f"ROC-AUC={roc:.5f} | "
            f"PR-AUC={pr:.5f} | "
            f"Best iteration={model.best_iteration_}"
        )

    return oof_prob, fold_results


def select_thresholds(y, probabilities):
    thresholds = np.arange(0.01, 0.51, 0.01)

    rows = []

    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(
                    y, pred, zero_division=0
                ),
                "recall": recall_score(y, pred),
                "f1": f1_score(y, pred),
            }
        )

    threshold_df = pd.DataFrame(rows)

    best_f1_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    high_recall = threshold_df[
        threshold_df["recall"] >= 0.60
    ].copy()

    high_recall_row = (
        high_recall
        .sort_values(
            ["precision", "f1"],
            ascending=False,
        )
        .iloc[0]
    )

    return (
        round(float(best_f1_row["threshold"]), 2),
        round(float(high_recall_row["threshold"]), 2),
    )


def train_final_model(X, y):
    params = MODEL_PARAMS.copy()
    params["n_estimators"] = FINAL_N_ESTIMATORS

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    return model


def save_artifacts(
    artifact_dir,
    model,
    feature_cols,
    categorical_cols,
    threshold_f1,
    threshold_high_recall,
    fold_results,
    oof_roc,
    oof_pr,
):
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        artifact_dir / "credit_risk_lgbm_v2.pkl",
    )

    feature_config = {
        "model_version": "v2_multitable",
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
    }

    threshold_config = {
        "model_version": "v2_multitable",
        "threshold_f1": threshold_f1,
        "threshold_high_recall": threshold_high_recall,
        "high_recall_target": 0.60,
    }

    roc_scores = [
        row["roc_auc"] for row in fold_results
    ]

    pr_scores = [
        row["pr_auc"] for row in fold_results
    ]

    metrics = {
        "model_version": "v2_multitable",
        "cv_folds": 5,
        "cv_roc_auc_mean": float(np.mean(roc_scores)),
        "cv_roc_auc_std": float(np.std(roc_scores)),
        "cv_pr_auc_mean": float(np.mean(pr_scores)),
        "cv_pr_auc_std": float(np.std(pr_scores)),
        "oof_roc_auc": float(oof_roc),
        "oof_pr_auc": float(oof_pr),
        "threshold_f1": threshold_f1,
        "threshold_high_recall": threshold_high_recall,
        "fold_results": fold_results,
    }

    with open(
        artifact_dir / "feature_config.json",
        "w",
    ) as f:
        json.dump(feature_config, f, indent=2)

    with open(
        artifact_dir / "threshold_config.json",
        "w",
    ) as f:
        json.dump(threshold_config, f, indent=2)

    with open(
        artifact_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
    )

    args = parser.parse_args()

    print("Loading data...")

    (
        application,
        bureau,
        bureau_balance,
        previous,
        installments,
    ) = load_data(args.data_dir)

    print("Building V2 multi-table dataset...")

    df = build_v2_dataset(
        application,
        bureau,
        bureau_balance,
        previous,
        installments,
    )

    X, y, feature_cols, categorical_cols = prepare_features(df)

    print(f"Rows: {len(X):,}")
    print(f"Features: {len(feature_cols)}")

    print("\nRunning 5-fold CV...")

    oof_prob, fold_results = run_cross_validation(
        X,
        y,
    )

    oof_roc = roc_auc_score(y, oof_prob)
    oof_pr = average_precision_score(y, oof_prob)

    print(f"\nOOF ROC-AUC: {oof_roc:.5f}")
    print(f"OOF PR-AUC : {oof_pr:.5f}")

    threshold_f1, threshold_high_recall = (
        select_thresholds(
            y,
            oof_prob,
        )
    )

    print(f"F1 threshold: {threshold_f1}")
    print(
        "High-recall threshold:",
        threshold_high_recall,
    )

    print("\nTraining final model...")

    final_model = train_final_model(
        X,
        y,
    )

    save_artifacts(
        args.artifact_dir,
        final_model,
        feature_cols,
        categorical_cols,
        threshold_f1,
        threshold_high_recall,
        fold_results,
        oof_roc,
        oof_pr,
    )

    print("\nTraining complete.")
    print(
        f"Artifacts saved to: "
        f"{args.artifact_dir.resolve()}"
    )


if __name__ == "__main__":
    main()