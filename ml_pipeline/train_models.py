"""
Diabetes Risk Assessment: ML Classifier Comparison
===================================================
Trains five classifiers on the UCI Early Stage Diabetes Risk Prediction
Dataset and exports evaluation data for the Next.js dashboard.
"""

import json
import os
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "diabetes_early_stage.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "public", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")
RANDOM_STATE = 42
TEST_RATIO = 0.2
CV_FOLDS = 5
ROC_POINTS = 100

FEATURE_NAMES = [
    "Age", "Gender", "Polyuria", "Polydipsia", "SuddenWeightLoss",
    "Weakness", "Polyphagia", "GenitalThrush", "VisualBlurring", "Itching",
    "Irritability", "DelayedHealing", "PartialParesis", "MuscleStiffness",
    "Alopecia", "Obesity",
]
TARGET_NAME = "Class"
YES_NO_FEATURES = FEATURE_NAMES[2:]

MODEL_CONFIG = [
    {
        "name": "Logistic Regression",
        "short_name": "LR",
        "color": "#06b6d4",
        "description": "A linear probability model that gives an interpretable baseline for symptom-based risk estimation.",
    },
    {
        "name": "Random Forest",
        "short_name": "RF",
        "color": "#8b5cf6",
        "description": "A bagged tree ensemble that captures non-linear symptom interactions and ranks feature influence.",
    },
    {
        "name": "Support Vector Machine",
        "short_name": "SVM",
        "color": "#f59e0b",
        "description": "A scaled RBF margin classifier suited to finding non-linear boundaries in compact feature spaces.",
    },
    {
        "name": "K-Nearest Neighbors",
        "short_name": "KNN",
        "color": "#10b981",
        "description": "An instance-based model that predicts from the majority pattern among nearby symptom profiles.",
    },
    {
        "name": "Gradient Boosting (XGBoost)",
        "short_name": "XGB",
        "color": "#ec4899",
        "description": "A boosted-tree model that sequentially corrects errors and learns high-capacity tabular patterns.",
    },
]


def download_dataset():
    """Download the UCI file only when the local copy is absent."""
    if os.path.exists(DATASET_PATH):
        print(f"  Dataset already exists at {DATASET_PATH}")
        return

    print(f"  Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print(f"  Saved dataset to {DATASET_PATH}")


def load_and_preprocess():
    """Load the categorical UCI data and encode it as numeric features."""
    print("\n[1/4] Loading and preprocessing data...")
    download_dataset()

    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={
        "sudden weight loss": "SuddenWeightLoss",
        "weakness": "Weakness",
        "Genital thrush": "GenitalThrush",
        "visual blurring": "VisualBlurring",
        "delayed healing": "DelayedHealing",
        "partial paresis": "PartialParesis",
        "muscle stiffness": "MuscleStiffness",
        "class": TARGET_NAME,
    })
    print(f"  Dataset shape: {df.shape}")

    for column in YES_NO_FEATURES:
        df[column] = df[column].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df[TARGET_NAME] = df[TARGET_NAME].map({"Positive": 1, "Negative": 0})

    if df[FEATURE_NAMES + [TARGET_NAME]].isna().any().any():
        raise ValueError("The UCI dataset contains an unexpected category or missing value")

    X = df[FEATURE_NAMES].astype(float).values
    y = df[TARGET_NAME].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_descriptions = [
        "Patient age in years", "Patient gender encoded as a binary feature",
        "Excessive urination", "Excessive thirst", "Sudden weight loss",
        "General weakness", "Excessive hunger", "Genital thrush symptom",
        "Visual blurring symptom", "Itching symptom", "Irritability symptom",
        "Delayed wound healing", "Partial muscle weakness", "Muscle stiffness",
        "Hair loss symptom", "Obesity symptom",
    ]
    positive = int(y.sum())
    dataset_info = {
        "name": "UCI Early Stage Diabetes Risk Prediction Dataset",
        "description": (
            "A 520-record UCI benchmark collected from patients in Sylhet, Bangladesh. "
            "It uses demographic information and reported symptoms to classify early-stage "
            "diabetes risk as Positive or Negative."
        ),
        "total_samples": int(len(df)),
        "features": [
            {"name": name, "description": description}
            for name, description in zip(FEATURE_NAMES, feature_descriptions)
        ],
        "target": TARGET_NAME,
        "class_distribution": {
            "positive": positive,
            "negative": int(len(y) - positive),
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Positive: {positive}, Negative: {int(len(y) - positive)}")
    return X_train, X_test, y_train, y_test, dataset_info


def build_classifiers():
    """Instantiate the five comparison classifiers."""
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
    except ImportError:
        print("  WARNING: XGBoost not installed; using GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier

        xgb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)

    return [
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        KNeighborsClassifier(n_neighbors=5),
        xgb,
    ]


def sample_roc_curve(fpr_raw, tpr_raw, n_points=ROC_POINTS):
    """Sample an ROC curve onto a common grid for the dashboard."""
    fpr_sampled = np.linspace(0, 1, n_points)
    tpr_sampled = np.interp(fpr_sampled, fpr_raw, tpr_raw)
    fpr_sampled[0], tpr_sampled[0] = 0.0, 0.0
    fpr_sampled[-1], tpr_sampled[-1] = 1.0, 1.0
    return [
        {"fpr": round(float(fpr), 4), "tpr": round(float(tpr), 4)}
        for fpr, tpr in zip(fpr_sampled, tpr_sampled)
    ]


def evaluate_model(clf, X_train, X_test, y_train, y_test, config, feature_names, has_feature_importance):
    """Fit one model and return the dashboard-compatible evaluation payload."""
    print(f"\n  Training {config['name']}...")
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time_ms = (time.perf_counter() - start) * 1000

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    specificity = tn / (tn + fp)
    fpr_raw, tpr_raw, _ = roc_curve(y_test, y_prob)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    feature_importance = None
    if has_feature_importance:
        importances = clf.feature_importances_
        feature_importance = [
            {"feature": feature_names[index], "importance": round(float(importances[index]), 4)}
            for index in np.argsort(importances)[::-1]
        ]

    print(f"    Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"    F1: {f1:.4f} | AUC: {auc:.4f} | Specificity: {specificity:.4f}")
    print(f"    CV Mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"    Training Time: {train_time_ms:.1f}ms")

    return {
        **config,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "auc": round(float(auc), 4),
            "specificity": round(float(specificity), 4),
            "training_time_ms": round(float(train_time_ms), 1),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": sample_roc_curve(fpr_raw, tpr_raw),
        "cv_scores": [round(float(score), 4) for score in cv_scores],
        "feature_importance": feature_importance,
    }


def main():
    print("=" * 64)
    print("  Diabetes Risk Assessment - UCI Early Stage Dataset")
    print("=" * 64)
    X_train, X_test, y_train, y_test, dataset_info = load_and_preprocess()

    print("\n[2/4] Building classifiers...")
    classifiers = build_classifiers()
    has_feature_importance = [False, True, False, False, True]

    print("\n[3/4] Training and evaluating models...")
    results = []
    for clf, config, fi_flag in zip(classifiers, MODEL_CONFIG, has_feature_importance):
        results.append(evaluate_model(
            clf, X_train, X_test, y_train, y_test, config,
            FEATURE_NAMES, fi_flag,
        ))

    output = {
        "dataset_info": dataset_info,
        "models": results,
        "preprocessing": {
            "scaling": "StandardScaler",
            "missing_value_strategy": "No missing values; categorical symptoms encoded as binary values",
            "categorical_encoding": "Yes/No and Male/Female mapped to numeric indicators",
            "test_split_ratio": TEST_RATIO,
            "random_state": RANDOM_STATE,
        },
    }

    print("\n[4/4] Exporting results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=True)
    print(f"  Results saved to {os.path.abspath(OUTPUT_PATH)}")

    best = max(results, key=lambda result: result["metrics"]["auc"])
    print("\n" + "=" * 64)
    print(f"  BEST MODEL: {best['name']} (AUC = {best['metrics']['auc']:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
