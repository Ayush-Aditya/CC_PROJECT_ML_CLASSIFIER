"""
Diabetes Risk Assessment: ML Classifier Comparison
====================================================
Trains 5 ML classifiers on the Pima Indians Diabetes Dataset,
evaluates their performance, and exports results to JSON for
the Next.js dashboard.

Classifiers:
  1. Logistic Regression
  2. Random Forest
  3. Support Vector Machine (SVM)
  4. K-Nearest Neighbors (KNN)
  5. Gradient Boosting (XGBoost)
"""

import json
import os
import time
import warnings
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "public", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")
RANDOM_STATE = 42
TEST_RATIO = 0.2
CV_FOLDS = 5
ROC_POINTS = 100  # number of points to sample on ROC curve

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_NAME = "Outcome"

# Columns with biologically impossible zero values
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Model display config
MODEL_CONFIG = [
    {
        "name": "Logistic Regression",
        "short_name": "LR",
        "color": "#06b6d4",
        "description": "A linear model that estimates the probability of diabetes using a logistic function. Highly interpretable and fast to train."
    },
    {
        "name": "Random Forest",
        "short_name": "RF",
        "color": "#8b5cf6",
        "description": "An ensemble of decision trees using bagging and feature randomness for robust, variance-reduced predictions."
    },
    {
        "name": "Support Vector Machine",
        "short_name": "SVM",
        "color": "#f59e0b",
        "description": "A margin-based classifier that finds the optimal hyperplane separating diabetic and non-diabetic cases in high-dimensional space."
    },
    {
        "name": "K-Nearest Neighbors",
        "short_name": "KNN",
        "color": "#10b981",
        "description": "An instance-based learner that classifies based on the majority vote of the k closest training samples."
    },
    {
        "name": "Gradient Boosting (XGBoost)",
        "short_name": "XGB",
        "color": "#ec4899",
        "description": "A powerful gradient boosting framework that sequentially builds trees to correct previous prediction errors."
    },
]


def download_dataset():
    """Download the Pima Indians Diabetes Dataset if not present."""
    if os.path.exists(DATASET_PATH):
        print(f"  Dataset already exists at {DATASET_PATH}")
        return

    print(f"  Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH + ".tmp")

    # The raw file has no headers — add them
    raw = pd.read_csv(DATASET_PATH + ".tmp", header=None)
    raw.columns = FEATURE_NAMES + [TARGET_NAME]
    raw.to_csv(DATASET_PATH, index=False)
    os.remove(DATASET_PATH + ".tmp")
    print(f"  Saved dataset with headers to {DATASET_PATH}")


def load_and_preprocess():
    """Load the dataset and handle missing values."""
    print("\n[1/4] Loading and preprocessing data...")
    download_dataset()

    df = pd.read_csv(DATASET_PATH)
    print(f"  Dataset shape: {df.shape}")

    # Replace biologically impossible zeros with NaN, then impute with median
    for col in ZERO_INVALID_COLS:
        zero_count = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: imputed {zero_count} zeros with median={median_val:.2f}")

    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    dataset_info = {
        "name": "Pima Indians Diabetes Dataset",
        "description": (
            "Originally from the National Institute of Diabetes and Digestive "
            "and Kidney Diseases. The dataset is used to predict whether a patient "
            "has diabetes based on diagnostic measurements. All patients are "
            "females at least 21 years old of Pima Indian heritage."
        ),
        "total_samples": int(len(df)),
        "features": [
            {"name": n, "description": d}
            for n, d in zip(FEATURE_NAMES, [
                "Number of times pregnant",
                "Plasma glucose concentration (2h OGTT)",
                "Diastolic blood pressure (mm Hg)",
                "Triceps skin fold thickness (mm)",
                "2-Hour serum insulin (μU/ml)",
                "Body mass index (kg/m²)",
                "Diabetes pedigree function",
                "Age (years)",
            ])
        ],
        "target": TARGET_NAME,
        "class_distribution": {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Positive: {int(y.sum())}, Negative: {int(len(y) - y.sum())}")

    return X_train, X_test, y_train, y_test, dataset_info


def build_classifiers():
    """Instantiate all 5 classifiers."""
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss',
            verbosity=0
        )
    except ImportError:
        print("  WARNING: XGBoost not installed, using GradientBoostingClassifier instead")
        from sklearn.ensemble import GradientBoostingClassifier
        xgb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)

    return [
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        KNeighborsClassifier(n_neighbors=5),
        xgb,
    ]


def sample_roc_curve(fpr_raw, tpr_raw, n_points=ROC_POINTS):
    """Sample n evenly-spaced points along the ROC curve."""
    # Create evenly spaced FPR values
    fpr_sampled = np.linspace(0, 1, n_points)
    tpr_sampled = np.interp(fpr_sampled, fpr_raw, tpr_raw)
    # Ensure endpoints
    fpr_sampled[0], tpr_sampled[0] = 0.0, 0.0
    fpr_sampled[-1], tpr_sampled[-1] = 1.0, 1.0
    return [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr_sampled, tpr_sampled)
    ]


def evaluate_model(clf, X_train, X_test, y_train, y_test, config, has_feature_importance):
    """Train and evaluate a single model."""
    name = config["name"]
    print(f"\n  Training {name}...")

    # Train with timing
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time_ms = (time.perf_counter() - start) * 1000

    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    spec = tn / (tn + fp)

    # ROC Curve
    fpr_raw, tpr_raw, _ = roc_curve(y_test, y_prob)
    roc_data = sample_roc_curve(fpr_raw, tpr_raw)

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

    # Feature importance
    fi = None
    if has_feature_importance:
        importances = clf.feature_importances_
        fi = [
            {"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]), 4)}
            for i in np.argsort(importances)[::-1]
        ]

    print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"    F1: {f1:.4f} | AUC: {auc:.4f} | Specificity: {spec:.4f}")
    print(f"    CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Training Time: {train_time_ms:.1f}ms")

    return {
        **config,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "auc": round(float(auc), 4),
            "specificity": round(float(spec), 4),
            "training_time_ms": round(float(train_time_ms), 1),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "roc_curve": roc_data,
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "feature_importance": fi,
    }


def main():
    print("=" * 60)
    print("  Diabetes Risk Assessment — ML Classifier Comparison")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, dataset_info = load_and_preprocess()

    # Build classifiers
    print("\n[2/4] Building classifiers...")
    classifiers = build_classifiers()
    has_fi = [False, True, False, False, True]  # LR, RF, SVM, KNN, XGB

    # Evaluate
    print("\n[3/4] Training and evaluating models...")
    results = []
    for clf, config, fi_flag in zip(classifiers, MODEL_CONFIG, has_fi):
        result = evaluate_model(clf, X_train, X_test, y_train, y_test, config, fi_flag)
        results.append(result)

    # Build output
    output = {
        "dataset_info": dataset_info,
        "models": results,
        "preprocessing": {
            "scaling": "StandardScaler",
            "missing_value_strategy": "Median imputation for biologically impossible zero values",
            "test_split_ratio": TEST_RATIO,
            "random_state": RANDOM_STATE,
        },
    }

    # Save JSON
    print("\n[4/4] Exporting results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {os.path.abspath(OUTPUT_PATH)}")

    # Summary
    best = max(results, key=lambda r: r["metrics"]["auc"])
    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best['name']} (AUC = {best['metrics']['auc']:.4f})")
    print("=" * 60)
    print("\nDone! The dashboard can now be started with:")
    print("  cd dashboard && npm run dev")


if __name__ == "__main__":
    main()
