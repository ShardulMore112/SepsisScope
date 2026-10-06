"""
Stage 3.2.2 — XGBoost Classical Baseline
=========================================

Purpose:
    Establish a strong nonlinear tree-based baseline before
    evaluating temporal deep-learning models.

Feature representation:
    F0 (80 features)
        ↓
    Patient-level temporal aggregation
        ↓
    560 features per patient

Patient-level aggregation:
    mean
    std
    min
    max
    first
    last
    delta (last - first)

Experimental rules:
    - Frozen Stage-1 patient split
    - Frozen Stage-2 preprocessing
    - Training data only for model fitting
    - Validation only for threshold selection
    - Test only for final evaluation
    - No test-driven hyperparameter tuning
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = (
    PROJECT_ROOT / "data" / "processed"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline_result"
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_SET = "F0"

SEED = 42

SUMMARY_STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "first",
    "last",
    "delta",
]


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

XGB_CONFIG = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
}


# ============================================================
# LOAD PROCESSED SPLIT
# ============================================================

def load_split(split: str):

    root = (
        PROCESSED_ROOT
        / FEATURE_SET
        / split
    )

    X = np.load(
        root / "X.npy",
        mmap_mode="r",
    )

    y = np.load(
        root / "labels.npy",
        mmap_mode="r",
    )

    offsets = np.load(
        root / "patient_offsets.npy",
    )

    patient_keys = np.load(
        root / "patient_keys.npy",
    )

    return (
        X,
        y,
        offsets,
        patient_keys,
    )


# ============================================================
# PATIENT-LEVEL LABELS
# ============================================================

def create_patient_labels(
    y: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:

    labels = np.zeros(
        len(offsets) - 1,
        dtype=np.int64,
    )

    for i in range(len(labels)):

        start = int(offsets[i])
        end = int(offsets[i + 1])

        labels[i] = int(
            np.any(
                y[start:end] == 1
            )
        )

    return labels


# ============================================================
# PATIENT-LEVEL TEMPORAL AGGREGATION
# ============================================================

def create_patient_features(
    X: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:

    n_patients = len(offsets) - 1
    n_features = X.shape[1]

    n_output_features = (
        n_features
        * len(SUMMARY_STATISTICS)
    )

    output = np.zeros(
        (
            n_patients,
            n_output_features,
        ),
        dtype=np.float32,
    )

    for i in range(n_patients):

        start = int(offsets[i])
        end = int(offsets[i + 1])

        sequence = np.array(
            X[start:end],
            dtype=np.float32,
            copy=True,
        )

        mean = np.mean(
            sequence,
            axis=0,
        )

        std = np.std(
            sequence,
            axis=0,
        )

        minimum = np.min(
            sequence,
            axis=0,
        )

        maximum = np.max(
            sequence,
            axis=0,
        )

        first = sequence[0]

        last = sequence[-1]

        delta = last - first

        output[i] = np.concatenate(
            [
                mean,
                std,
                minimum,
                maximum,
                first,
                last,
                delta,
            ]
        )

    return output


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def find_best_f1_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):

    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        _, _, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                predictions,
                average="binary",
                zero_division=0,
            )
        )

        if f1 > best_f1:

            best_f1 = float(f1)

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    auroc = roc_auc_score(
        y_true,
        probabilities,
    )

    auprc = average_precision_score(
        y_true,
        probabilities,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Precision": float(precision),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": float(threshold),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 3.2.2 — XGBOOST BASELINE")
    print("=" * 70)

    print(
        f"\nFeature set: {FEATURE_SET}"
    )

    print(
        "\nXGBoost configuration:"
    )

    for key, value in XGB_CONFIG.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading frozen Step-2 data..."
    )

    (
        X_train_raw,
        y_train_raw,
        train_offsets,
        train_keys,
    ) = load_split("train")

    (
        X_validation_raw,
        y_validation_raw,
        validation_offsets,
        validation_keys,
    ) = load_split("validation")

    (
        X_test_raw,
        y_test_raw,
        test_offsets,
        test_keys,
    ) = load_split("test")

    print(
        f"Train patients: "
        f"{len(train_offsets) - 1:,}"
    )

    print(
        f"Validation patients: "
        f"{len(validation_offsets) - 1:,}"
    )

    print(
        f"Test patients: "
        f"{len(test_offsets) - 1:,}"
    )

    # --------------------------------------------------------
    # Patient labels
    # --------------------------------------------------------

    print(
        "\nCreating patient-level labels..."
    )

    y_train = create_patient_labels(
        y_train_raw,
        train_offsets,
    )

    y_validation = create_patient_labels(
        y_validation_raw,
        validation_offsets,
    )

    y_test = create_patient_labels(
        y_test_raw,
        test_offsets,
    )

    print(
        f"Train positive patients: "
        f"{y_train.sum():,}"
    )

    print(
        f"Validation positive patients: "
        f"{y_validation.sum():,}"
    )

    print(
        f"Test positive patients: "
        f"{y_test.sum():,}"
    )

    # --------------------------------------------------------
    # Temporal aggregation
    # --------------------------------------------------------

    print(
        "\nCreating patient-level temporal features..."
    )

    X_train = create_patient_features(
        X_train_raw,
        train_offsets,
    )

    X_validation = create_patient_features(
        X_validation_raw,
        validation_offsets,
    )

    X_test = create_patient_features(
        X_test_raw,
        test_offsets,
    )

    print(
        f"Aggregated train shape: "
        f"{X_train.shape}"
    )

    print(
        f"Aggregated validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"Aggregated test shape: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Finite-value verification
    # --------------------------------------------------------

    print(
        "\nChecking feature matrices..."
    )

    for name, array in [
        ("X_train", X_train),
        (
            "X_validation",
            X_validation,
        ),
        ("X_test", X_test),
    ]:

        if not np.isfinite(array).all():

            raise ValueError(
                f"Non-finite values found in "
                f"{name}."
            )

    print(
        "Feature matrix check: PASSED"
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    positive = int(
        y_train.sum()
    )

    negative = int(
        len(y_train) - positive
    )

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"\nTraining positive patients: "
        f"{positive:,}"
    )

    print(
        f"Training negative patients: "
        f"{negative:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_params = dict(
        XGB_CONFIG
    )

    model_params[
        "scale_pos_weight"
    ] = scale_pos_weight

    model = XGBClassifier(
        **model_params
    )

    print(
        "\nTraining XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    print(
        "\nGenerating validation predictions..."
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_threshold, validation_f1 = (
        find_best_f1_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        f"Validation-selected threshold: "
        f"{validation_threshold:.4f}"
    )

    print(
        f"Validation F1: "
        f"{validation_f1:.6f}"
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        validation_threshold,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Test is evaluated only after threshold
    # selection on validation.
    # --------------------------------------------------------

    print(
        "\nGenerating final test predictions..."
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        validation_threshold,
    )

    # --------------------------------------------------------
    # Print validation results
    # --------------------------------------------------------

    print(
        "\nValidation metrics:"
    )

    for key, value in validation_metrics.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Print test results
    # --------------------------------------------------------

    print(
        "\nTest metrics:"
    )

    for key, value in test_metrics.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {

        "stage": "3.2.2",

        "model": "XGBoost",

        "feature_set": FEATURE_SET,

        "seed": SEED,

        "summary_statistics": (
            SUMMARY_STATISTICS
        ),

        "train_patients": int(
            len(y_train)
        ),

        "validation_patients": int(
            len(y_validation)
        ),

        "test_patients": int(
            len(y_test)
        ),

        "train_positive_patients": int(
            y_train.sum()
        ),

        "validation_positive_patients": int(
            y_validation.sum()
        ),

        "test_positive_patients": int(
            y_test.sum()
        ),

        "scale_pos_weight": float(
            scale_pos_weight
        ),

        "xgboost_configuration": (
            model_params
        ),

        "validation": validation_metrics,

        "test": test_metrics,

    }

    output_path = (
        RESULTS_ROOT
        / "baseline_xgboost_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nSaved:\n{output_path}"
    )

    print("\n" + "=" * 70)
    print(
        "STAGE 3.2.2 COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()