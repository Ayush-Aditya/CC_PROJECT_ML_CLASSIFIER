"""
Diabetes Risk Assessment: ML Classifier Comparison
===================================================
Trains five classifiers on the CDC BRFSS 2015 Diabetes Health Indicators
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
    balanced_accuracy_score,
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

DATASET_URL = "https://raw.githubusercontent.com/ai2ys/CDC-Diabetes-Health-Indicators/main/dataset/data.csv"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "diabetes_brfss2015.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "public", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.10
TEST_RATIO = 0.2
CV_FOLDS = 5
ROC_POINTS = 100

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
TARGET_NAME = "Diabetes_binary"

MODEL_CONFIG = [
    {
        "name": "Logistic Regression",
        "short_name": "LR",
        "color": "#06b6d4",
        "description": "An interpretable linear baseline that estimates diabetes probability from population health indicators.",
    },
    {
        "name": "Random Forest",
        "short_name": "RF",
        "color": "#8b5cf6",
        "description": "A bagged tree ensemble that captures non-linear relationships across health and lifestyle variables.",
    },
    {
        "name": "Support Vector Machine",
        "short_name": "SVM",
        "color": "#f59e0b",
        "description": "An RBF-kernel margin classifier suited to finding non-linear boundaries in a structured feature space.",
    },
    {
        "name": "K-Nearest Neighbors",
        "short_name": "KNN",
        "color": "#10b981",
        "description": "An instance-based model that predicts from the most similar health profiles in the training data.",
    },
    {
        "name": "Gradient Boosting (XGBoost)",
        "short_name": "XGB",
        "color": "#ec4899",
        "description": "A boosted-tree model that sequentially corrects errors and learns complex tabular patterns.",
    },
]

FEATURE_DESCRIPTIONS = {
    "HighBP": "High blood pressure indicator",
    "HighChol": "High cholesterol indicator",
    "CholCheck": "Cholesterol check within the last five years",
    "BMI": "Body mass index",
    "Smoker": "Smoking history indicator",
    "Stroke": "History of stroke indicator",
    "HeartDiseaseorAttack": "History of coronary heart disease or heart attack",
    "PhysActivity": "Physical activity outside work in the past 30 days",
    "Fruits": "Fruit consumption frequency indicator",
    "Veggies": "Vegetable consumption frequency indicator",
    "HvyAlcoholConsump": "Heavy alcohol consumption indicator",
    "AnyHealthcare": "Health care coverage indicator",
    "NoDocbcCost": "Could not see a doctor because of cost",
    "GenHlth": "Self-reported general health rating",
    "MentHlth": "Number of days mental health was not good",
    "PhysHlth": "Number of days physical health was not good",
    "DiffWalk": "Difficulty walking or climbing stairs",
    "Sex": "Respondent sex encoded numerically",
    "Age": "Age category encoded numerically",
    "Education": "Education level category",
    "Income": "Income category",
}


def download_dataset():
    """Download the CDC file only when the local copy is absent."""
    if os.path.exists(DATASET_PATH):
        print(f"  Dataset already exists at {DATASET_PATH}")
        return

    print(f"  Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print(f"  Saved dataset to {DATASET_PATH}")


def load_and_preprocess():
    """Randomly sample 10% of the CDC data and create a stratified split."""
    print("\n[1/4] Loading and preprocessing data...")
    download_dataset()

    df = pd.read_csv(DATASET_PATH)
    print(f"  Dataset shape: {df.shape}")
    required_columns = ["ID", TARGET_NAME] + FEATURE_NAMES
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    df = df[required_columns].dropna()
    original_sample_count = len(df)
    df = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Random sample: {len(df)} of {original_sample_count} rows ({SAMPLE_FRACTION:.0%})")
    X = df[FEATURE_NAMES].astype(float).values
    y = df[TARGET_NAME].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    positive = int(y.sum())
    dataset_info = {
        "name": "CDC BRFSS 2015 Diabetes Health Indicators Dataset",
        "description": (
            "A large population-health benchmark derived from the 2015 Behavioral Risk "
            "Factor Surveillance System. It combines health history, lifestyle, access "
            "to care, demographic, and self-reported health indicators to classify diabetes."
        ),
        "total_samples": int(len(df)),
        "source_total_samples": int(original_sample_count),
        "sample_fraction": SAMPLE_FRACTION,
        "features": [
            {"name": name, "description": FEATURE_DESCRIPTIONS[name]}
            for name in FEATURE_NAMES
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


def build_classifiers(y_train):
    """Instantiate the five comparison classifiers."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    positive_weight = negative_count / positive_count
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            verbosity=0,
        )
    except ImportError:
        print("  WARNING: XGBoost not installed; using GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier

        xgb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)

    return [
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights="distance"),
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
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    specificity = tn / (tn + fp)
    fpr_raw, tpr_raw, _ = roc_curve(y_test, y_prob)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    feature_importance = None
    if has_feature_importance:
        importances = clf.feature_importances_
        feature_importance = [
            {"feature": feature_names[index], "importance": round(float(importances[index]), 4)}
            for index in np.argsort(importances)[::-1]
        ]

    print(f"    Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"    Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"    F1: {f1:.4f} | AUC: {auc:.4f} | Specificity: {specificity:.4f}")
    print(f"    CV Mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"    Training Time: {train_time_ms:.1f}ms")

    return {
        **config,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "balanced_accuracy": round(float(balanced_accuracy), 4),
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
    print("  Diabetes Risk Assessment - CDC BRFSS 2015 Dataset")
    print("=" * 64)
    X_train, X_test, y_train, y_test, dataset_info = load_and_preprocess()

    print("\n[2/4] Building classifiers...")
    classifiers = build_classifiers(y_train)
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
            "class_balance_strategy": "Balanced class weights for LR, RF, and SVM; positive-class weighting for XGBoost; distance weighting for KNN",
            "missing_value_strategy": "Rows with missing values removed; source indicators are numeric",
            "test_split_ratio": TEST_RATIO,
            "sample_fraction": SAMPLE_FRACTION,
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
