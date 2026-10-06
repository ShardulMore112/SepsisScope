"""
Stage 3.7.1
Feature-Combination Ensemble Analysis

Purpose
-------
Evaluate all non-empty subsets of F0/F1/F2/F3 using the already-generated
validation predictions.

For each architecture:
    GRU5
    LSTM4
    TR1

Evaluate:
    - 4 single-feature candidates
    - 6 two-feature combinations
    - 4 three-feature combinations
    - 1 four-feature combination

Total:
    15 combinations / architecture
    45 total candidates

IMPORTANT
---------
- Validation predictions only
- NO training
- NO test predictions
- NO modification of processed data
- Equal-weight probability averaging
- Threshold selected on validation by maximum F1
- AUPRC is the primary ranking metric
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"D:\CAPSTONE\SEPSIS")

PRED_ROOT = (
    ROOT
    / "results"
    / "model_result"
    / "feature_ablation"
)

OUT_DIR = (
    ROOT
    / "results"
    / "model_result"
    / "ensemble_analysis"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


ARCHITECTURES = {
    "GRU5": {
        "F0": "GRU5-F0",
        "F1": "GRU5-F1",
        "F2": "GRU5-F2",
        "F3": "GRU5-F3",
    },
    "LSTM4": {
        "F0": "LSTM4-F0",
        "F1": "LSTM4-F1",
        "F2": "LSTM4-F2",
        "F3": "LSTM4-F3",
    },
    "TR1": {
        "F0": "TR1-F0",
        "F1": "TR1-F1",
        "F2": "TR1-F2",
        "F3": "TR1-F3",
    },
}


FEATURES = ["F0", "F1", "F2", "F3"]


# ============================================================
# HELPERS
# ============================================================

def find_prediction_file(model_dir: Path) -> Path:
    """
    Locate validation_predictions.npz.
    """

    candidates = [
        model_dir / "validation_predictions.npz",
        model_dir / "val_predictions.npz",
        model_dir / "validation_prediction.npz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No validation prediction file found in:\n{model_dir}"
    )


def extract_arrays(npz_path: Path):
    """
    Supports both F0 and F1-F3 naming conventions.

    F0 files:
        probabilities
        labels

    F1-F3 files:
        probs
        targets
    """

    data = np.load(npz_path, allow_pickle=True)

    probability_keys = [
        "probabilities",
        "probs",
        "prediction_probabilities",
        "predictions",
        "y_prob",
    ]

    target_keys = [
        "labels",
        "targets",
        "y_true",
        "true_labels",
        "targets_flat",
    ]

    probs = None
    targets = None

    for key in probability_keys:
        if key in data:
            probs = np.asarray(data[key]).reshape(-1)
            break

    for key in target_keys:
        if key in data:
            targets = np.asarray(data[key]).reshape(-1)
            break

    if probs is None:
        raise KeyError(
            f"Could not find probability array in {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    if targets is None:
        raise KeyError(
            f"Could not find target array in {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    if len(probs) != len(targets):
        raise ValueError(
            f"Length mismatch in {npz_path}: "
            f"probs={len(probs)}, targets={len(targets)}"
        )

    return probs.astype(np.float64), targets.astype(np.int64)


def load_predictions():
    """
    Load all 12 architecture-feature validation predictions.
    """

    predictions = {}
    labels_reference = None

    print("\n" + "=" * 80)
    print("LOADING VALIDATION PREDICTIONS")
    print("=" * 80)

    for architecture, feature_map in ARCHITECTURES.items():

        predictions[architecture] = {}

        for feature, dirname in feature_map.items():

            model_dir = PRED_ROOT / dirname
            npz_path = find_prediction_file(model_dir)

            probs, labels = extract_arrays(npz_path)

            if labels_reference is None:
                labels_reference = labels

            else:
                if not np.array_equal(labels_reference, labels):
                    raise ValueError(
                        f"Target mismatch detected for "
                        f"{architecture}-{feature}"
                    )

            predictions[architecture][feature] = probs

            print(
                f"{architecture}-{feature:<2} | "
                f"n={len(probs):,} | "
                f"positive={labels.sum():,} | "
                f"file={npz_path}"
            )

    return predictions, labels_reference


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_thresholds: int = 1001,
):
    """
    Select threshold using maximum validation F1.

    Threshold is selected ONLY on validation data.
    """

    thresholds = np.linspace(
        0.0,
        1.0,
        n_thresholds,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        pred = (probs >= threshold).astype(np.int8)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            zero_division=0,
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
):
    """
    Calculate validation metrics and select threshold by F1.
    """

    auprc = average_precision_score(
        y_true,
        probs,
    )

    auroc = roc_auc_score(
        y_true,
        probs,
    )

    threshold, _ = find_best_threshold(
        y_true,
        probs,
    )

    pred = (
        probs >= threshold
    ).astype(np.int8)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
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
        "AUPRC": float(auprc),
        "AUROC": float(auroc),
        "F1": float(f1),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "Threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def make_combinations():
    """
    Generate all 15 non-empty feature subsets.
    """

    combinations = []

    for size in range(1, len(FEATURES) + 1):

        for combo in itertools.combinations(
            FEATURES,
            size,
        ):

            combinations.append(combo)

    return combinations


def combination_name(combo):
    return "+".join(combo)


def average_predictions(
    predictions: dict,
    architecture: str,
    combo: tuple[str, ...],
):
    """
    Equal-weight probability averaging.
    """

    arrays = [
        predictions[architecture][feature]
        for feature in combo
    ]

    stacked = np.vstack(arrays)

    return np.mean(
        stacked,
        axis=0,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("STAGE 3.7.1")
    print("FEATURE-COMBINATION ENSEMBLE ANALYSIS")
    print("=" * 80)

    print("\nThis stage is VALIDATION-ONLY.")
    print("No models will be trained.")
    print("No test predictions will be accessed.")
    print("No processed data will be modified.")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    predictions, y_true = load_predictions()

    print("\n" + "-" * 80)
    print("VALIDATION DATA")
    print("-" * 80)

    print(f"Samples       : {len(y_true):,}")
    print(f"Positive      : {int(y_true.sum()):,}")
    print(f"Negative      : {int((y_true == 0).sum()):,}")
    print(
        f"Prevalence    : "
        f"{100 * y_true.mean():.4f}%"
    )

    # --------------------------------------------------------
    # Generate combinations
    # --------------------------------------------------------

    combinations = make_combinations()

    print("\n" + "-" * 80)
    print("FEATURE COMBINATIONS")
    print("-" * 80)

    print(
        f"Total combinations per architecture: "
        f"{len(combinations)}"
    )

    for combo in combinations:
        print(
            f"  {combination_name(combo)}"
        )

    print(
        f"\nTotal candidates: "
        f"{len(combinations) * len(ARCHITECTURES)}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    rows = []

    for architecture in ARCHITECTURES:

        print("\n")
        print("=" * 80)
        print(f"ARCHITECTURE: {architecture}")
        print("=" * 80)

        for combo in combinations:

            name = combination_name(combo)

            ensemble_probs = average_predictions(
                predictions,
                architecture,
                combo,
            )

            metrics = calculate_metrics(
                y_true,
                ensemble_probs,
            )

            row = {
                "Architecture": architecture,
                "Combination": name,
                "NumFeatures": len(combo),
                "Features": ",".join(combo),
                **metrics,
            }

            rows.append(row)

            print(
                f"{name:<14} | "
                f"AUPRC={metrics['AUPRC']:.6f} | "
                f"AUROC={metrics['AUROC']:.6f} | "
                f"F1={metrics['F1']:.6f} | "
                f"Sens={metrics['Sensitivity']:.4f} | "
                f"Spec={metrics['Specificity']:.4f} | "
                f"Thr={metrics['Threshold']:.3f}"
            )

    results = pd.DataFrame(rows)

    # ========================================================
    # SAVE COMPLETE RESULTS
    # ========================================================

    all_results_path = (
        OUT_DIR
        / "validation_feature_combination_ensembles.csv"
    )

    results.to_csv(
        all_results_path,
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)

    print(all_results_path)

    # ========================================================
    # BEST PER ARCHITECTURE
    # ========================================================

    best_rows = []

    for architecture in ARCHITECTURES:

        subset = results[
            results["Architecture"] == architecture
        ].copy()

        subset = subset.sort_values(
            by=[
                "AUPRC",
                "AUROC",
                "F1",
            ],
            ascending=False,
        )

        best = subset.iloc[0]

        best_rows.append(best)

    best_per_architecture = pd.DataFrame(
        best_rows
    )

    best_arch_path = (
        OUT_DIR
        / "validation_best_feature_combination_per_architecture.csv"
    )

    best_per_architecture.to_csv(
        best_arch_path,
        index=False,
    )

    # ========================================================
    # RANK EVERYTHING
    # ========================================================

    ranked = results.sort_values(
        by=[
            "AUPRC",
            "AUROC",
            "F1",
        ],
        ascending=False,
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "Rank",
        np.arange(1, len(ranked) + 1),
    )

    ranked_path = (
        OUT_DIR
        / "validation_feature_combination_ranking.csv"
    )

    ranked.to_csv(
        ranked_path,
        index=False,
    )

    # ========================================================
    # BEST BY COMBINATION SIZE
    # ========================================================

    best_by_size_rows = []

    for size in sorted(
        results["NumFeatures"].unique()
    ):

        subset = results[
            results["NumFeatures"] == size
        ].sort_values(
            by=[
                "AUPRC",
                "AUROC",
                "F1",
            ],
            ascending=False,
        )

        best = subset.iloc[0]

        best_by_size_rows.append(best)

    best_by_size = pd.DataFrame(
        best_by_size_rows
    )

    best_size_path = (
        OUT_DIR
        / "validation_best_by_combination_size.csv"
    )

    best_by_size.to_csv(
        best_size_path,
        index=False,
    )

    # ========================================================
    # F0 BASELINE COMPARISON
    # ========================================================

    baseline_rows = []

    for architecture in ARCHITECTURES:

        subset = results[
            (results["Architecture"] == architecture)
            & (results["Combination"] == "F0")
        ]

        if len(subset) == 0:
            continue

        baseline = subset.iloc[0]

        best = (
            results[
                results["Architecture"] == architecture
            ]
            .sort_values(
                by="AUPRC",
                ascending=False,
            )
            .iloc[0]
        )

        improvement = (
            best["AUPRC"]
            - baseline["AUPRC"]
        )

        relative_improvement = (
            improvement
            / baseline["AUPRC"]
            if baseline["AUPRC"] != 0
            else np.nan
        )

        baseline_rows.append(
            {
                "Architecture": architecture,
                "F0_AUPRC": baseline["AUPRC"],
                "Best_Combination": best["Combination"],
                "Best_AUPRC": best["AUPRC"],
                "Absolute_AUPRC_Improvement": improvement,
                "Relative_AUPRC_Improvement": relative_improvement,
                "F0_AUROC": baseline["AUROC"],
                "Best_AUROC": best["AUROC"],
                "F0_F1": baseline["F1"],
                "Best_F1": best["F1"],
            }
        )

    baseline_comparison = pd.DataFrame(
        baseline_rows
    )

    baseline_path = (
        OUT_DIR
        / "validation_f0_vs_best_combination.csv"
    )

    baseline_comparison.to_csv(
        baseline_path,
        index=False,
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    global_best = ranked.iloc[0]

    summary = {
        "stage": "3.7.1",
        "description": (
            "Validation-only equal-weight feature-combination "
            "ensemble analysis."
        ),
        "architectures": list(
            ARCHITECTURES.keys()
        ),
        "features": FEATURES,
        "combinations_per_architecture": len(
            combinations
        ),
        "total_candidates": len(
            combinations
        ) * len(ARCHITECTURES),
        "validation_samples": int(
            len(y_true)
        ),
        "validation_positive": int(
            y_true.sum()
        ),
        "validation_negative": int(
            (y_true == 0).sum()
        ),
        "selection_metric": "AUPRC",
        "threshold_selection": (
            "Maximum validation F1"
        ),
        "ensemble_method": (
            "Equal-weight probability averaging"
        ),
        "global_best": {
            "rank": int(
                ranked.iloc[0]["Rank"]
            ),
            "architecture": str(
                global_best["Architecture"]
            ),
            "combination": str(
                global_best["Combination"]
            ),
            "AUPRC": float(
                global_best["AUPRC"]
            ),
            "AUROC": float(
                global_best["AUROC"]
            ),
            "F1": float(
                global_best["F1"]
            ),
            "Sensitivity": float(
                global_best["Sensitivity"]
            ),
            "Specificity": float(
                global_best["Specificity"]
            ),
            "Threshold": float(
                global_best["Threshold"]
            ),
        },
    }

    summary_path = (
        OUT_DIR
        / "stage_3_7_1_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT FINAL RANKING
    # ========================================================

    print("\n")
    print("=" * 80)
    print("GLOBAL VALIDATION RANKING")
    print("=" * 80)

    display_cols = [
        "Rank",
        "Architecture",
        "Combination",
        "AUPRC",
        "AUROC",
        "F1",
    ]

    print(
        ranked[display_cols]
        .head(20)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("BEST PER ARCHITECTURE")
    print("=" * 80)

    print(
        best_per_architecture[
            [
                "Architecture",
                "Combination",
                "AUPRC",
                "AUROC",
                "F1",
                "Sensitivity",
                "Specificity",
                "Threshold",
            ]
        ].to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("BEST BY COMBINATION SIZE")
    print("=" * 80)

    print(
        best_by_size[
            [
                "NumFeatures",
                "Architecture",
                "Combination",
                "AUPRC",
                "AUROC",
                "F1",
            ]
        ].to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("F0 VS BEST COMBINATION")
    print("=" * 80)

    print(
        baseline_comparison.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("STAGE 3.7.1 COMPLETE")
    print("=" * 80)

    print(
        f"\nGlobal best validation candidate:\n"
        f"  Architecture : "
        f"{global_best['Architecture']}\n"
        f"  Combination  : "
        f"{global_best['Combination']}\n"
        f"  AUPRC        : "
        f"{global_best['AUPRC']:.6f}\n"
        f"  AUROC        : "
        f"{global_best['AUROC']:.6f}\n"
        f"  F1           : "
        f"{global_best['F1']:.6f}\n"
    )

    print("\nFiles:")
    print(f"  {all_results_path}")
    print(f"  {ranked_path}")
    print(f"  {best_arch_path}")
    print(f"  {best_size_path}")
    print(f"  {baseline_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()