"""
Stage 3.2.1 — Classical Baselines
==================================

Baselines:
    1. Majority-class predictor
    2. Logistic Regression

Feature set:
    F0

Important:
    - Frozen patient-level train/validation/test split
    - Training-only feature scaling
    - Validation-only threshold selection
    - Test set used only for final evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RESULTS_ROOT = PROJECT_ROOT / "results" / "baseline_result"

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
# LOAD SPLIT
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

    return X, y, offsets, patient_keys


# ============================================================
# PATIENT LABELS
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
# PATIENT-LEVEL TEMPORAL FEATURES
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
# VALIDATION THRESHOLD
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

    return best_threshold, best_f1


# ============================================================
# MAJORITY BASELINE
# ============================================================

def run_majority_baseline(
    y_train,
    y_validation,
    y_test,
):

    print("\n" + "=" * 70)
    print("MAJORITY-CLASS BASELINE")
    print("=" * 70)

    positive_rate = float(
        np.mean(y_train)
    )

    print(
        f"Training positive rate: "
        f"{positive_rate:.6f}"
    )

    validation_probabilities = np.full(
        len(y_validation),
        positive_rate,
        dtype=np.float32,
    )

    test_probabilities = np.full(
        len(y_test),
        positive_rate,
        dtype=np.float32,
    )

    validation_auprc = (
        average_precision_score(
            y_validation,
            validation_probabilities,
        )
    )

    test_auprc = (
        average_precision_score(
            y_test,
            test_probabilities,
        )
    )

    results = {
        "model": "majority_class",
        "training_positive_rate": positive_rate,
        "validation_AUPRC": float(
            validation_auprc
        ),
        "test_AUPRC": float(
            test_auprc
        ),
    }

    print(
        f"Validation AUPRC: "
        f"{validation_auprc:.6f}"
    )

    print(
        f"Test AUPRC: "
        f"{test_auprc:.6f}"
    )

    return results


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def run_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    print(
        f"Train shape:       {X_train.shape}"
    )

    print(
        f"Validation shape:  {X_validation.shape}"
    )

    print(
        f"Test shape:        {X_test.shape}"
    )

    # --------------------------------------------------------
    # TRAINING-ONLY STANDARDIZATION
    # --------------------------------------------------------

    print(
        "\nFitting StandardScaler on training data only..."
    )

    scaler = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True,
    )

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(
        np.float32,
        copy=False,
    )

    X_validation_scaled = (
        scaler.transform(
            X_validation
        ).astype(
            np.float32,
            copy=False,
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        ).astype(
            np.float32,
            copy=False,
        )
    )

    print(
        "Training-only scaling complete."
    )

    # --------------------------------------------------------
    # Verify finite values
    # --------------------------------------------------------

    for name, array in [
        ("X_train_scaled", X_train_scaled),
        (
            "X_validation_scaled",
            X_validation_scaled,
        ),
        ("X_test_scaled", X_test_scaled),
    ]:

        if not np.isfinite(array).all():

            raise ValueError(
                f"Non-finite values found in "
                f"{name}."
            )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        tol=1e-4,
        random_state=SEED,
    )

    print(
        "\nTraining Logistic Regression..."
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    print(
        "Training complete."
    )

    print(
        f"Solver iterations used: "
        f"{model.n_iter_[0]}"
    )

    if model.n_iter_[0] >= 5000:

        raise RuntimeError(
            "Logistic Regression still reached "
            "max_iter=5000. Do not freeze this "
            "baseline until convergence is achieved."
        )

    print(
        "Convergence check: PASSED"
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation_scaled
        )[:, 1]
    )

    validation_threshold, validation_f1 = (
        find_best_f1_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        f"\nValidation-selected threshold: "
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
    # Test
    #
    # IMPORTANT:
    # Threshold is frozen from validation.
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test_scaled
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        validation_threshold,
    )

    print("\nValidation metrics:")

    for key, value in validation_metrics.items():

        print(
            f"  {key}: {value}"
        )

    print("\nTest metrics:")

    for key, value in test_metrics.items():

        print(
            f"  {key}: {value}"
        )

    return (
        model,
        scaler,
        validation_metrics,
        test_metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 3.2.1 — CLASSICAL BASELINES")
    print("=" * 70)

    print(
        f"\nFeature set: {FEATURE_SET}"
    )

    print(
        "Loading frozen Step-2 data..."
    )

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

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
    # Majority baseline
    # --------------------------------------------------------

    majority_results = run_majority_baseline(
        y_train,
        y_validation,
        y_test,
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
    # Logistic Regression
    # --------------------------------------------------------

    (
        model,
        scaler,
        validation_metrics,
        test_metrics,
    ) = run_logistic_regression(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {

        "stage": "3.2.1",

        "feature_set": FEATURE_SET,

        "seed": SEED,

        "summary_statistics": (
            SUMMARY_STATISTICS
        ),

        "scaling": {
            "method": "StandardScaler",
            "fit_on": "training_split_only",
        },

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

        "majority_baseline": (
            majority_results
        ),

        "logistic_regression": {

            "solver": "lbfgs",

            "class_weight": "balanced",

            "max_iter": 5000,

            "convergence": True,

            "validation": (
                validation_metrics
            ),

            "test": (
                test_metrics
            ),
        },
    }

    output_path = (
        RESULTS_ROOT
        / "baseline_classical_summary.json"
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
    print("STAGE 3.2.1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()