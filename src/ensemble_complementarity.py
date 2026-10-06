"""
X-FedSepsis

STAGE 3.7 — PREDICTION COMPLEMENTARITY ANALYSIS

Validation-only analysis.

No model training.
No processed-data modification.
No test predictions used for ensemble selection.

Models:
    GRU5-F0/F1/F2/F3
    LSTM4-F0/F1/F2/F3
    TR1-F0/F1/F2/F3
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

FEATURE_ABLATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_result"
    / "feature_ablation"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_result"
    / "ensemble_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "GRU5-F0",
    "GRU5-F1",
    "GRU5-F2",
    "GRU5-F3",

    "LSTM4-F0",
    "LSTM4-F1",
    "LSTM4-F2",
    "LSTM4-F3",

    "TR1-F0",
    "TR1-F1",
    "TR1-F2",
    "TR1-F3",
]


# ============================================================
# FIND PREDICTION FILE
# ============================================================

def find_prediction_file(model):

    direct = (
        FEATURE_ABLATION_ROOT
        / model
        / "validation_predictions.npz"
    )

    if direct.exists():
        return direct

    results_root = (
        PROJECT_ROOT
        / "results"
    )

    candidates = []

    for path in results_root.rglob(
        "validation_predictions.npz"
    ):

        if model.upper() in str(path).upper():
            candidates.append(path)

    if candidates:
        return candidates[0]

    architecture = model.split("-")[0]
    feature = model.split("-")[1]

    for path in results_root.rglob(
        "validation_predictions.npz"
    ):

        text = str(path).upper()

        if (
            architecture.upper() in text
            and feature.upper() in text
        ):
            candidates.append(path)

    if candidates:
        return candidates[0]

    return None


# ============================================================
# NPZ INSPECTION
# ============================================================

def inspect_npz(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    print()
    print(f"Inspecting: {path}")
    print("Keys:", list(data.keys()))

    for key in data.keys():

        arr = np.asarray(data[key])

        print(
            f"  {key:<25} "
            f"shape={arr.shape} "
            f"dtype={arr.dtype}"
        )

    return data


# ============================================================
# EXTRACT ARRAYS
# ============================================================

def extract_arrays(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    keys = list(data.keys())

    # --------------------------------------------------------
    # Supported label keys
    # --------------------------------------------------------

    y_candidates = [
        "y_val",
        "y_validation",
        "y_true",
        "labels",
        "targets",
        "val_labels",
        "validation_labels",
        "validation_targets",
        "y",
    ]

    # --------------------------------------------------------
    # Supported prediction keys
    # --------------------------------------------------------

    p_candidates = [
        "val_probs",
        "validation_probs",
        "val_predictions",
        "validation_predictions",
        "probs",
        "predictions",
        "probabilities",
        "y_prob",
    ]

    y_key = None
    p_key = None

    for key in y_candidates:
        if key in keys:
            y_key = key
            break

    for key in p_candidates:
        if key in keys:
            p_key = key
            break

    if y_key is None:

        raise RuntimeError(
            f"No validation label array found in:\n"
            f"{path}\n\n"
            f"Available keys: {keys}"
        )

    if p_key is None:

        raise RuntimeError(
            f"No validation prediction array found in:\n"
            f"{path}\n\n"
            f"Available keys: {keys}"
        )

    y = np.asarray(
        data[y_key]
    ).reshape(-1)

    p = np.asarray(
        data[p_key]
    ).reshape(-1)

    if len(y) != len(p):

        raise RuntimeError(
            f"Length mismatch in {path}\n"
            f"labels ({y_key}) = {len(y)}\n"
            f"predictions ({p_key}) = {len(p)}"
        )

    return (
        y.astype(np.int8),
        p.astype(np.float64),
        y_key,
        p_key,
    )

# ============================================================
# THRESHOLD
# ============================================================

def find_threshold(y, p):

    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.arange(
        0.05,
        0.951,
        0.01,
    ):

        pred = (
            p >= threshold
        ).astype(int)

        score = f1_score(
            y,
            pred,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
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

def metrics(y, p):

    auprc = average_precision_score(
        y,
        p,
    )

    auroc = roc_auc_score(
        y,
        p,
    )

    threshold, f1 = find_threshold(
        y,
        p,
    )

    pred = (
        p >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y,
        pred,
        zero_division=0,
    )

    tn = np.sum(
        (y == 0) &
        (pred == 0)
    )

    fp = np.sum(
        (y == 0) &
        (pred == 1)
    )

    if tn + fp > 0:

        specificity = (
            tn /
            (tn + fp)
        )

    else:

        specificity = np.nan

    return {
        "AUPRC": float(auprc),
        "AUROC": float(auroc),
        "F1": float(f1),
        "Precision": float(precision),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "threshold": float(threshold),
    }


# ============================================================
# ALERT OVERLAP
# ============================================================

def alert_overlap(
    p_a,
    p_b,
    threshold_a,
    threshold_b,
):

    alerts_a = p_a >= threshold_a
    alerts_b = p_b >= threshold_b

    intersection = np.sum(
        alerts_a & alerts_b
    )

    union = np.sum(
        alerts_a | alerts_b
    )

    alerts_a_count = np.sum(
        alerts_a
    )

    alerts_b_count = np.sum(
        alerts_b
    )

    if union > 0:
        jaccard = intersection / union
    else:
        jaccard = np.nan

    if alerts_a_count > 0:
        overlap_a = (
            intersection /
            alerts_a_count
        )
    else:
        overlap_a = np.nan

    if alerts_b_count > 0:
        overlap_b = (
            intersection /
            alerts_b_count
        )
    else:
        overlap_b = np.nan

    return {
        "alerts_a": int(alerts_a_count),
        "alerts_b": int(alerts_b_count),
        "intersection": int(intersection),
        "union": int(union),
        "jaccard_overlap": float(jaccard),
        "overlap_fraction_a": float(overlap_a),
        "overlap_fraction_b": float(overlap_b),
    }


# ============================================================
# MEAN ENSEMBLE
# ============================================================

def mean_ensemble(
    predictions,
    members,
):

    return np.mean(
        np.vstack(
            [
                predictions[m]
                for m in members
            ]
        ),
        axis=0,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STAGE 3.7 — PREDICTION COMPLEMENTARITY ANALYSIS")
    print("=" * 80)

    print()
    print(
        "This stage uses VALIDATION predictions only."
    )

    print(
        "No models will be trained."
    )

    print(
        "No test predictions will be used for selection."
    )

    print()

    # ========================================================
    # LOCATE ALL PREDICTIONS
    # ========================================================

    paths = {}

    print("=" * 80)
    print("LOCATING VALIDATION PREDICTIONS")
    print("=" * 80)
    print()

    for model in MODELS:

        path = find_prediction_file(
            model
        )

        if path is None:

            print(
                f"[MISSING] {model}"
            )

        else:

            paths[model] = path

            print(
                f"[FOUND]   {model:<12} "
                f"{path}"
            )

    print()

    missing = [
        model
        for model in MODELS
        if model not in paths
    ]

    if missing:

        print(
            "Missing validation predictions:"
        )

        for model in missing:
            print(
                f"  - {model}"
            )

        print()

        print(
            "STOPPING — we will not fabricate "
            "or substitute predictions."
        )

        return

    # ========================================================
    # LOAD
    # ========================================================

    predictions = {}

    thresholds = {}

    reference_y = None

    print("=" * 80)
    print("LOADING VALIDATION PREDICTIONS")
    print("=" * 80)
    print()

    metadata = {}

    for model in MODELS:

        path = paths[model]

        y, p, y_key, p_key = (
            extract_arrays(path)
        )

        # ----------------------------------------------------
        # Ensure identical validation labels
        # ----------------------------------------------------

        if reference_y is None:

            reference_y = y.copy()

        else:

            if len(reference_y) != len(y):

                raise RuntimeError(
                    f"Validation length mismatch: {model}"
                )

            if not np.array_equal(
                reference_y,
                y,
            ):

                raise RuntimeError(
                    f"Validation labels differ for {model}."
                )

        predictions[model] = p

        # ----------------------------------------------------
        # Read saved threshold if present
        # ----------------------------------------------------

        data = np.load(
            path,
            allow_pickle=True,
        )

        if "threshold" in data:

            threshold_value = float(
                np.asarray(
                    data["threshold"]
                ).reshape(-1)[0]
            )

        else:

            threshold_value, _ = (
                find_threshold(
                    y,
                    p,
                )
            )

        thresholds[model] = (
            threshold_value
        )

        metadata[model] = {
            "file": str(path),
            "y_key": y_key,
            "prediction_key": p_key,
            "n": int(len(y)),
            "threshold": float(
                threshold_value
            ),
        }

        print(
            f"{model:<12} "
            f"n={len(y):,} "
            f"positives={np.sum(y == 1):,} "
            f"min={p.min():.5f} "
            f"max={p.max():.5f} "
            f"threshold={threshold_value:.5f}"
        )

    y = reference_y

    print()
    print(
        f"Validation samples : {len(y):,}"
    )

    print(
        f"Positive timesteps : "
        f"{np.sum(y == 1):,}"
    )

    print(
        f"Negative timesteps : "
        f"{np.sum(y == 0):,}"
    )

    # ========================================================
    # INDIVIDUAL METRICS
    # ========================================================

    print()
    print("=" * 80)
    print("INDIVIDUAL VALIDATION PERFORMANCE")
    print("=" * 80)

    individual = []

    for model in MODELS:

        result = metrics(
            y,
            predictions[model],
        )

        row = {
            "model": model,
            **result,
        }

        individual.append(row)

        print(
            f"{model:<12} "
            f"AUPRC={result['AUPRC']:.6f} "
            f"AUROC={result['AUROC']:.6f} "
            f"F1={result['F1']:.6f} "
            f"Sens={result['Sensitivity']:.4f} "
            f"Spec={result['Specificity']:.4f}"
        )

    individual_df = pd.DataFrame(
        individual
    )

    individual_df = individual_df.sort_values(
        "AUPRC",
        ascending=False,
    )

    individual_df.to_csv(
        OUTPUT_ROOT
        / "validation_individual_metrics.csv",
        index=False,
    )

    # ========================================================
    # CORRELATION / DISAGREEMENT
    # ========================================================

    print()
    print("=" * 80)
    print("PAIRWISE PREDICTION COMPLEMENTARITY")
    print("=" * 80)

    correlation = pd.DataFrame(
        index=MODELS,
        columns=MODELS,
        dtype=float,
    )

    disagreement = pd.DataFrame(
        index=MODELS,
        columns=MODELS,
        dtype=float,
    )

    for a in MODELS:

        for b in MODELS:

            correlation.loc[a, b] = (
                np.corrcoef(
                    predictions[a],
                    predictions[b],
                )[0, 1]
            )

            disagreement.loc[a, b] = (
                np.mean(
                    np.abs(
                        predictions[a]
                        -
                        predictions[b]
                    )
                )
            )

    correlation.to_csv(
        OUTPUT_ROOT
        / "validation_prediction_correlation.csv"
    )

    disagreement.to_csv(
        OUTPUT_ROOT
        / "validation_prediction_disagreement.csv"
    )

    # ========================================================
    # ALERT OVERLAP
    # ========================================================

    print()
    print("=" * 80)
    print("PAIRWISE ALERT OVERLAP")
    print("=" * 80)

    alert_rows = []

    for i, a in enumerate(MODELS):

        for b in MODELS[i + 1:]:

            overlap = alert_overlap(
                predictions[a],
                predictions[b],
                thresholds[a],
                thresholds[b],
            )

            alert_rows.append({
                "model_a": a,
                "model_b": b,
                "threshold_a": thresholds[a],
                "threshold_b": thresholds[b],
                **overlap,
                "prediction_correlation":
                    correlation.loc[a, b],
                "mean_abs_disagreement":
                    disagreement.loc[a, b],
            })

    alert_df = pd.DataFrame(
        alert_rows
    )

    alert_df = alert_df.sort_values(
        "jaccard_overlap",
        ascending=True,
    )

    alert_df.to_csv(
        OUTPUT_ROOT
        / "validation_alert_overlap.csv",
        index=False,
    )

    print()
    print(
        "Lowest-overlap model pairs:"
    )

    for _, row in alert_df.head(10).iterrows():

        print(
            f"{row['model_a']:<12} + "
            f"{row['model_b']:<12} "
            f"| Jaccard={row['jaccard_overlap']:.4f} "
            f"| corr={row['prediction_correlation']:.4f} "
            f"| diff={row['mean_abs_disagreement']:.5f}"
        )

    # ========================================================
    # PAIRWISE MEAN ENSEMBLES
    # ========================================================

    print()
    print("=" * 80)
    print("PAIRWISE MEAN ENSEMBLES")
    print("=" * 80)

    pair_results = []

    for i, a in enumerate(MODELS):

        for b in MODELS[i + 1:]:

            ensemble = (
                predictions[a]
                +
                predictions[b]
            ) / 2.0

            result = metrics(
                y,
                ensemble,
            )

            pair_results.append({
                "model_a": a,
                "model_b": b,
                "correlation":
                    correlation.loc[a, b],
                "mean_abs_disagreement":
                    disagreement.loc[a, b],
                **result,
            })

    pair_df = pd.DataFrame(
        pair_results
    )

    pair_df = pair_df.sort_values(
        "AUPRC",
        ascending=False,
    )

    pair_df.to_csv(
        OUTPUT_ROOT
        / "validation_pairwise_ensembles.csv",
        index=False,
    )

    print()

    for _, row in pair_df.head(15).iterrows():

        print(
            f"{row['model_a']:<12} + "
            f"{row['model_b']:<12} "
            f"| AUPRC={row['AUPRC']:.6f} "
            f"| AUROC={row['AUROC']:.6f} "
            f"| corr={row['correlation']:.4f} "
            f"| diff={row['mean_abs_disagreement']:.5f}"
        )

    # ========================================================
    # HOMOGENEOUS FAMILY ENSEMBLES
    # ========================================================

    print()
    print("=" * 80)
    print("HOMOGENEOUS FAMILY ENSEMBLES")
    print("=" * 80)

    families = {

        "GRU": [
            "GRU5-F0",
            "GRU5-F1",
            "GRU5-F2",
            "GRU5-F3",
        ],

        "LSTM": [
            "LSTM4-F0",
            "LSTM4-F1",
            "LSTM4-F2",
            "LSTM4-F3",
        ],

        "Transformer": [
            "TR1-F0",
            "TR1-F1",
            "TR1-F2",
            "TR1-F3",
        ],
    }

    family_results = []

    family_predictions = {}

    for family, members in families.items():

        ensemble = mean_ensemble(
            predictions,
            members,
        )

        family_predictions[family] = ensemble

        result = metrics(
            y,
            ensemble,
        )

        family_results.append({
            "ensemble": family,
            "members": ",".join(members),
            **result,
        })

        print(
            f"{family:<15} "
            f"AUPRC={result['AUPRC']:.6f} "
            f"AUROC={result['AUROC']:.6f} "
            f"F1={result['F1']:.6f}"
        )

    family_df = pd.DataFrame(
        family_results
    )

    family_df.to_csv(
        OUTPUT_ROOT
        / "validation_family_ensembles.csv",
        index=False,
    )

    # ========================================================
    # HETEROGENEOUS ARCHITECTURE ENSEMBLE
    # ========================================================

    print()
    print("=" * 80)
    print("HETEROGENEOUS ARCHITECTURE ENSEMBLES")
    print("=" * 80)

    heterogeneous_results = []

    # --------------------------------------------------------
    # H1: Architecture-level ensemble
    # GRU ensemble + LSTM ensemble + Transformer ensemble
    # --------------------------------------------------------

    architecture_ensemble = (
        family_predictions["GRU"]
        +
        family_predictions["LSTM"]
        +
        family_predictions["Transformer"]
    ) / 3.0

    architecture_result = metrics(
        y,
        architecture_ensemble,
    )

    heterogeneous_results.append({
        "ensemble": "HET_ARCHITECTURE",
        "members": (
            "GRU_family,LSTM_family,Transformer_family"
        ),
        **architecture_result,
    })

    print(
        f"{'HET_ARCHITECTURE':<24} "
        f"AUPRC={architecture_result['AUPRC']:.6f} "
        f"AUROC={architecture_result['AUROC']:.6f} "
        f"F1={architecture_result['F1']:.6f}"
    )

    # --------------------------------------------------------
    # H2: Best individual from each architecture
    # --------------------------------------------------------

    best_gru = (
        individual_df[
            individual_df["model"].str.startswith("GRU")
        ]
        .sort_values(
            "AUPRC",
            ascending=False,
        )
        .iloc[0]["model"]
    )

    best_lstm = (
        individual_df[
            individual_df["model"].str.startswith("LSTM")
        ]
        .sort_values(
            "AUPRC",
            ascending=False,
        )
        .iloc[0]["model"]
    )

    best_transformer = (
        individual_df[
            individual_df["model"].str.startswith("TR")
        ]
        .sort_values(
            "AUPRC",
            ascending=False,
        )
        .iloc[0]["model"]
    )

    best_architecture_members = [
        best_gru,
        best_lstm,
        best_transformer,
    ]

    best_architecture_ensemble = mean_ensemble(
        predictions,
        best_architecture_members,
    )

    best_architecture_result = metrics(
        y,
        best_architecture_ensemble,
    )

    heterogeneous_results.append({
        "ensemble": "HET_BEST_ARCHITECTURE_MODELS",
        "members": ",".join(
            best_architecture_members
        ),
        **best_architecture_result,
    })

    print(
        f"{'HET_BEST_MODELS':<24} "
        f"AUPRC={best_architecture_result['AUPRC']:.6f} "
        f"AUROC={best_architecture_result['AUROC']:.6f} "
        f"F1={best_architecture_result['F1']:.6f}"
    )

    print()
    print(
        "Best GRU        :", best_gru
    )

    print(
        "Best LSTM       :", best_lstm
    )

    print(
        "Best Transformer:",
        best_transformer,
    )

    # --------------------------------------------------------
    # H3: All 12 models
    # --------------------------------------------------------

    all_models_ensemble = mean_ensemble(
        predictions,
        MODELS,
    )

    all_models_result = metrics(
        y,
        all_models_ensemble,
    )

    heterogeneous_results.append({
        "ensemble": "ALL_12_MODELS",
        "members": ",".join(MODELS),
        **all_models_result,
    })

    print(
        f"{'ALL_12_MODELS':<24} "
        f"AUPRC={all_models_result['AUPRC']:.6f} "
        f"AUROC={all_models_result['AUROC']:.6f} "
        f"F1={all_models_result['F1']:.6f}"
    )

    heterogeneous_df = pd.DataFrame(
        heterogeneous_results
    )

    heterogeneous_df = heterogeneous_df.sort_values(
        "AUPRC",
        ascending=False,
    )

    heterogeneous_df.to_csv(
        OUTPUT_ROOT
        / "validation_heterogeneous_ensembles.csv",
        index=False,
    )

    # ========================================================
    # COMPLETE ENSEMBLE COMPARISON
    # ========================================================

    print()
    print("=" * 80)
    print("COMPLETE VALIDATION ENSEMBLE COMPARISON")
    print("=" * 80)

    comparison_rows = []

    # Best individual
    best_individual = (
        individual_df.iloc[0]
    )

    comparison_rows.append({
        "type": "individual",
        "name": best_individual["model"],
        "members": best_individual["model"],
        "AUPRC": best_individual["AUPRC"],
        "AUROC": best_individual["AUROC"],
        "F1": best_individual["F1"],
        "Precision": best_individual["Precision"],
        "Sensitivity": best_individual["Sensitivity"],
        "Specificity": best_individual["Specificity"],
        "threshold": best_individual["threshold"],
    })

    # Best pair
    best_pair = pair_df.iloc[0]

    comparison_rows.append({
        "type": "pair",
        "name": (
            f"{best_pair['model_a']}+"
            f"{best_pair['model_b']}"
        ),
        "members": (
            f"{best_pair['model_a']},"
            f"{best_pair['model_b']}"
        ),
        "AUPRC": best_pair["AUPRC"],
        "AUROC": best_pair["AUROC"],
        "F1": best_pair["F1"],
        "Precision": best_pair["Precision"],
        "Sensitivity": best_pair["Sensitivity"],
        "Specificity": best_pair["Specificity"],
        "threshold": best_pair["threshold"],
    })

    # Family ensembles
    for _, row in family_df.iterrows():

        comparison_rows.append({
            "type": "homogeneous",
            "name": row["ensemble"],
            "members": row["members"],
            "AUPRC": row["AUPRC"],
            "AUROC": row["AUROC"],
            "F1": row["F1"],
            "Precision": row["Precision"],
            "Sensitivity": row["Sensitivity"],
            "Specificity": row["Specificity"],
            "threshold": row["threshold"],
        })

    # Heterogeneous ensembles
    for _, row in heterogeneous_df.iterrows():

        comparison_rows.append({
            "type": "heterogeneous",
            "name": row["ensemble"],
            "members": row["members"],
            "AUPRC": row["AUPRC"],
            "AUROC": row["AUROC"],
            "F1": row["F1"],
            "Precision": row["Precision"],
            "Sensitivity": row["Sensitivity"],
            "Specificity": row["Specificity"],
            "threshold": row["threshold"],
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = comparison_df.sort_values(
        "AUPRC",
        ascending=False,
    )

    comparison_df.to_csv(
        OUTPUT_ROOT
        / "validation_ensemble_comparison.csv",
        index=False,
    )

    print()

    for _, row in comparison_df.iterrows():

        print(
            f"{row['type']:<14} "
            f"{row['name']:<28} "
            f"AUPRC={row['AUPRC']:.6f} "
            f"AUROC={row['AUROC']:.6f} "
            f"F1={row['F1']:.6f}"
        )

    # ========================================================
    # VALIDATION LEADERS
    # ========================================================

    best_overall = (
        comparison_df.iloc[0]
    )

    print()
    print("=" * 80)
    print("VALIDATION LEADERS")
    print("=" * 80)

    print()

    print(
        f"Best individual : "
        f"{best_individual['model']}"
    )

    print(
        f"Best individual AUPRC : "
        f"{best_individual['AUPRC']:.6f}"
    )

    print()

    print(
        f"Best pair : "
        f"{best_pair['model_a']} + "
        f"{best_pair['model_b']}"
    )

    print(
        f"Best pair AUPRC : "
        f"{best_pair['AUPRC']:.6f}"
    )

    print(
        f"Pair improvement : "
        f"{best_pair['AUPRC'] - best_individual['AUPRC']:+.6f}"
    )

    print()

    print(
        f"Best overall candidate : "
        f"{best_overall['name']}"
    )

    print(
        f"Best overall AUPRC : "
        f"{best_overall['AUPRC']:.6f}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "stage": "3.7",

        "purpose":
            "Validation-only prediction "
            "complementarity and ensemble analysis",

        "test_used_for_selection": False,

        "models": MODELS,

        "validation_samples": int(
            len(y)
        ),

        "validation_positive_timesteps": int(
            np.sum(y == 1)
        ),

        "prediction_files": {
            model: str(paths[model])
            for model in MODELS
        },

        "thresholds": thresholds,

        "best_individual":
            best_individual.to_dict(),

        "best_pair":
            best_pair.to_dict(),

        "best_overall":
            best_overall.to_dict(),

        "best_architecture_members": {
            "GRU": best_gru,
            "LSTM": best_lstm,
            "Transformer": best_transformer,
        },

        "outputs": {

            "individual_metrics":
                "validation_individual_metrics.csv",

            "correlation":
                "validation_prediction_correlation.csv",

            "disagreement":
                "validation_prediction_disagreement.csv",

            "alert_overlap":
                "validation_alert_overlap.csv",

            "pairwise_ensembles":
                "validation_pairwise_ensembles.csv",

            "family_ensembles":
                "validation_family_ensembles.csv",

            "heterogeneous_ensembles":
                "validation_heterogeneous_ensembles.csv",

            "ensemble_comparison":
                "validation_ensemble_comparison.csv",
        },
    }

    with open(
        OUTPUT_ROOT
        / "stage_3_7_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 3.7 COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Results saved to:\n{OUTPUT_ROOT}"
    )

    print()

    print(
        "No test predictions were used "
        "for ensemble selection."
    )

    print(
        "Next stage: freeze ensemble design "
        "before final evaluation."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()