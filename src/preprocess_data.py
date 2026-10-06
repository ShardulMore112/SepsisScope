"""
X-FedSepsis — Step 2: Causal preprocessing and feature engineering.

Creates F0/F1/F2/F3 datasets using the frozen patient split archive.

Raw files in data/training are NEVER modified.

F0 = raw + missingness
F1 = F0 + causal deltas
F2 = F1 + causal rolling statistics
F3 = F2 + clinical ratios/scores
"""

from __future__ import annotations

import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

SET_A_DIR = PROJECT_ROOT / "data" / "training" / "training_setA"
SET_B_DIR = PROJECT_ROOT / "data" / "training" / "training_setB"

SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "split_indices.npz"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "preprocessing_result"

BASE_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
    "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Fibrinogen", "Platelets", "Age", "Gender",
    "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

DYNAMIC_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "WBC", "Lactate", "Creatinine", "Bilirubin_total", "Platelets",
]

DELTA_LAGS = (1, 3, 6)
ROLLING_WINDOWS = (3, 6)

SPLITS = ("train", "validation", "test")


# ============================================================================
# RAW DATA
# ============================================================================

def get_all_patient_files() -> list[Path]:

    files_a = sorted(SET_A_DIR.glob("*.psv"))
    files_b = sorted(SET_B_DIR.glob("*.psv"))

    if not files_a or not files_b:
        raise FileNotFoundError(
            "Could not find PSV files in both training_setA and training_setB."
        )

    return files_a + files_b


def load_frozen_split_paths(
    all_files: list[Path],
) -> dict[str, list[Path]]:

    archive = np.load(
        SPLIT_FILE,
        allow_pickle=True,
    )

    split_paths = {}
    used_indices = set()

    for split in SPLITS:

        indices = archive[f"{split}_indices"].astype(int)
        expected_ids = archive[f"{split}_patient_ids"].astype(str)

        if len(indices) != len(expected_ids):
            raise ValueError(
                f"{split}: index count != patient ID count."
            )

        paths = [all_files[i] for i in indices]

        actual_ids = np.asarray(
            [p.stem for p in paths],
            dtype=str,
        )

        if not np.array_equal(
            actual_ids,
            expected_ids,
        ):
            raise ValueError(
                f"{split}: frozen split does not match raw-file ordering."
            )

        overlap = used_indices.intersection(
            indices.tolist()
        )

        if overlap:
            raise ValueError(
                f"{split}: patient overlap detected."
            )

        used_indices.update(
            indices.tolist()
        )

        split_paths[split] = paths

    if len(used_indices) != len(all_files):
        raise ValueError(
            "Frozen splits do not cover exactly all patients."
        )

    return split_paths


# ============================================================================
# PATIENT READING
# ============================================================================

def read_patient(path: Path):

    df = pd.read_csv(
        path,
        sep="|",
    )

    values = df[BASE_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # IMPORTANT:
    # This is captured BEFORE forward-fill.
    original_missing = values.isna().astype(
        np.float32
    )

    # Strictly causal.
    causal_values = values.ffill()

    labels = (
        pd.to_numeric(
            df["SepsisLabel"],
            errors="coerce",
        )
        .fillna(0)
        .astype(np.int8)
        .to_numpy()
    )

    return (
        causal_values,
        original_missing,
        labels,
    )


# ============================================================================
# TRAINING-ONLY MEDIANS
# ============================================================================

def calculate_training_medians(
    train_paths: list[Path],
):

    """
    Calculate medians from TRAIN patients only.

    Important optimization:
    we do NOT retain all patient DataFrames in memory.

    We collect numeric observations feature-by-feature and release
    each patient's DataFrame immediately.
    """

    feature_values = {
        feature: []
        for feature in BASE_FEATURES
    }

    total_rows = 0
    missing_counts = np.zeros(
        len(BASE_FEATURES),
        dtype=np.int64,
    )

    for index, path in enumerate(
        train_paths,
        start=1,
    ):

        causal, original_missing, _ = read_patient(
            path
        )

        total_rows += len(causal)

        missing_counts += (
            original_missing
            .sum(axis=0)
            .to_numpy(dtype=np.int64)
        )

        for feature in BASE_FEATURES:

            values = (
                causal[feature]
                .dropna()
                .to_numpy(
                    dtype=np.float64
                )
            )

            if len(values):
                feature_values[feature].append(
                    values
                )

        del causal
        del original_missing

        if index % 2000 == 0:
            print(
                f"Median pass: "
                f"{index:,}/{len(train_paths):,}"
            )

    medians = {}

    for feature in BASE_FEATURES:

        chunks = feature_values[feature]

        if not chunks:
            medians[feature] = 0.0
        else:
            values = np.concatenate(chunks)

            medians[feature] = float(
                np.median(values)
            )

            del values

    medians = pd.Series(
        medians,
        dtype=np.float64,
    )

    statistics = pd.DataFrame({
        "feature": BASE_FEATURES,
        "training_rows": total_rows,
        "original_missing_values": missing_counts,
        "original_missing_percent":
            missing_counts / total_rows * 100.0,
        "training_median_after_causal_ffill":
            medians.to_numpy(),
    })

    return medians, statistics


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

def add_deltas(
    causal_values: pd.DataFrame,
) -> pd.DataFrame:

    feature_frames = []

    for feature in DYNAMIC_FEATURES:

        current = causal_values[feature]

        columns = {}

        for lag in DELTA_LAGS:

            historical = current.shift(lag)

            valid = (
                current.notna()
                & historical.notna()
            )

            delta = current - historical

            columns[
                f"{feature}_delta_{lag}h"
            ] = delta.where(
                valid,
                0.0,
            )

            columns[
                f"{feature}_delta_{lag}h_valid"
            ] = valid.astype(np.float32)

        feature_frames.append(
            pd.DataFrame(
                columns,
                index=causal_values.index,
            )
        )

    return pd.concat(
    feature_frames,
    axis=1,
    )


def add_rolling_statistics(
    causal_values: pd.DataFrame,
) -> pd.DataFrame:

    feature_frames = []

    for feature in DYNAMIC_FEATURES:

        series = causal_values[feature]

        for window in ROLLING_WINDOWS:

            rolling = series.rolling(
                window=window,
                min_periods=1,
            )

            count = rolling.count()

            feature_frames.append(
                pd.DataFrame({
                    f"{feature}_rolling_mean_{window}h":
                        rolling.mean().fillna(0.0),

                    f"{feature}_rolling_std_{window}h":
                        rolling.std(ddof=0).fillna(0.0),

                    f"{feature}_rolling_min_{window}h":
                        rolling.min().fillna(0.0),

                    f"{feature}_rolling_max_{window}h":
                        rolling.max().fillna(0.0),

                    f"{feature}_rolling_{window}h_valid":
                        (count > 0).astype(np.float32),

                    f"{feature}_rolling_{window}h_complete":
                        (count >= window).astype(np.float32),
                }, index=causal_values.index)
            )

    return pd.concat(
    feature_frames,
    axis=1,
    )


# ============================================================================
# CLINICAL FEATURES
# ============================================================================

def safe_ratio(
    numerator,
    denominator,
):

    valid = (
        numerator.notna()
        & denominator.notna()
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    result = pd.Series(
        0.0,
        index=numerator.index,
        dtype=np.float64,
    )

    result.loc[valid] = (
        numerator.loc[valid]
        / denominator.loc[valid]
    )

    return (
        result,
        valid.astype(np.float32),
    )


def calculate_mews(
    causal_values: pd.DataFrame,
):

    hr = causal_values["HR"]
    sbp = causal_values["SBP"]
    resp = causal_values["Resp"]
    temp = causal_values["Temp"]

    valid = (
        hr.notna()
        & sbp.notna()
        & resp.notna()
        & temp.notna()
    )

    score = pd.Series(
        0.0,
        index=causal_values.index,
    )

    score += np.select(
        [
            hr <= 40,
            hr.between(41, 50),
            hr.between(101, 110),
            hr.between(111, 129),
            hr >= 130,
        ],
        [2, 1, 1, 2, 3],
        default=0,
    )

    score += np.select(
        [
            sbp <= 70,
            sbp.between(71, 80),
            sbp.between(81, 100),
            sbp >= 200,
        ],
        [3, 2, 1, 2],
        default=0,
    )

    score += np.select(
        [
            resp <= 8,
            resp.between(15, 20),
            resp.between(21, 29),
            resp >= 30,
        ],
        [2, 1, 2, 3],
        default=0,
    )

    score += np.select(
        [
            temp < 35.0,
            temp >= 38.5,
        ],
        [2, 2],
        default=0,
    )

    score.loc[~valid] = 0.0

    return (
        score,
        valid.astype(np.float32),
    )


def add_clinical_features(
    causal_values: pd.DataFrame,
) -> pd.DataFrame:

    output = pd.DataFrame(
        index=causal_values.index
    )

    shock, shock_valid = safe_ratio(
        causal_values["HR"],
        causal_values["SBP"],
    )

    bun_creatinine, bun_valid = safe_ratio(
        causal_values["BUN"],
        causal_values["Creatinine"],
    )

    o2_fio2, o2_valid = safe_ratio(
        causal_values["O2Sat"],
        causal_values["FiO2"],
    )

    sirs_valid = causal_values[
        ["HR", "Temp", "Resp", "WBC"]
    ].notna().all(axis=1)

    sirs = (
        (causal_values["HR"] > 90).astype(float)
        +
        (
            (causal_values["Temp"] < 36)
            |
            (causal_values["Temp"] > 38)
        ).astype(float)
        +
        (causal_values["Resp"] > 20).astype(float)
        +
        (
            (causal_values["WBC"] < 4)
            |
            (causal_values["WBC"] > 12)
        ).astype(float)
    )

    sirs.loc[~sirs_valid] = 0.0

    mews, mews_valid = calculate_mews(
        causal_values
    )

    output["ShockIndex_HR_SBP"] = shock
    output["ShockIndex_HR_SBP_valid"] = shock_valid

    output["BUN_Creatinine_ratio"] = bun_creatinine
    output["BUN_Creatinine_ratio_valid"] = bun_valid

    output["O2Sat_FiO2_ratio"] = o2_fio2
    output["O2Sat_FiO2_ratio_valid"] = o2_valid

    output["SIRS_score"] = sirs
    output["SIRS_score_valid"] = (
        sirs_valid.astype(np.float32)
    )

    output["MEWS_4_variable"] = mews
    output["MEWS_4_variable_valid"] = mews_valid

    return output


# ============================================================================
# FEATURE SETS
# ============================================================================

def build_feature_sets(
    causal_values: pd.DataFrame,
    original_missing: pd.DataFrame,
    medians: pd.Series,
):

    # Median imputation happens AFTER causal features are created.
    imputed_values = causal_values.fillna(
        medians
    )

    missing_indicators = original_missing.rename(
        columns={
            feature:
                f"{feature}_missing"
            for feature in BASE_FEATURES
        }
    )

    # F0
    f0 = pd.concat(
        [
            imputed_values,
            missing_indicators,
        ],
        axis=1,
    )

    # F1
    deltas = add_deltas(
        causal_values
    )

    f1 = pd.concat(
        [f0, deltas],
        axis=1,
    )

    # F2
    rolling = add_rolling_statistics(
        causal_values
    )

    f2 = pd.concat(
        [f1, rolling],
        axis=1,
    )

    # F3
    clinical = add_clinical_features(
        causal_values
    )

    f3 = pd.concat(
        [f2, clinical],
        axis=1,
    )

    return {
        "F0": f0,
        "F1": f1,
        "F2": f2,
        "F3": f3,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def prepare_split_metadata(
    paths: list[Path],
):

    lengths = []

    for index, path in enumerate(
        paths,
        start=1,
    ):

        n = len(
            pd.read_csv(
                path,
                sep="|",
                usecols=["SepsisLabel"],
            )
        )

        lengths.append(n)

        if (
            index % 2000 == 0
            or index == len(paths)
        ):
            print(
                f"Counting rows: "
                f"{index:,}/{len(paths):,}"
            )

    return lengths, sum(lengths)


def write_split(
    split: str,
    paths: list[Path],
    medians: pd.Series,
):

    lengths, total_rows = (
        prepare_split_metadata(paths)
    )

    # Get feature dimensionality from one patient.
    first_values, first_missing, _ = (
        read_patient(paths[0])
    )

    first_sets = build_feature_sets(
        first_values,
        first_missing,
        medians,
    )

    arrays = {}

    for feature_set, frame in first_sets.items():

        output_dir = (
            PROCESSED_DIR
            / feature_set
            / split
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        arrays[feature_set] = (
            np.lib.format.open_memmap(
                output_dir / "X.npy",
                mode="w+",
                dtype=np.float32,
                shape=(
                    total_rows,
                    frame.shape[1],
                ),
            )
        )

        with open(
            output_dir / "feature_names.json",
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                frame.columns.tolist(),
                handle,
                indent=2,
            )

    del first_values
    del first_missing
    del first_sets

    labels = np.empty(
        total_rows,
        dtype=np.int8,
    )

    offsets = np.zeros(
        len(paths) + 1,
        dtype=np.int64,
    )

    hour_index = np.empty(
        total_rows,
        dtype=np.int16,
    )

    patient_keys = []

    cursor = 0

    for patient_index, (
        path,
        patient_length,
    ) in enumerate(
        zip(paths, lengths),
        start=1,
    ):

        causal_values, original_missing, patient_labels = (
            read_patient(path)
        )

        feature_sets = build_feature_sets(
            causal_values,
            original_missing,
            medians,
        )

        if len(patient_labels) != patient_length:
            raise ValueError(
                f"Row count changed: {path}"
            )

        for feature_set, frame in feature_sets.items():

            matrix = frame.to_numpy(
                dtype=np.float32,
                copy=False,
            )

            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"NaN/inf found in "
                    f"{feature_set}: {path}"
                )

            arrays[feature_set][
                cursor:
                cursor + patient_length
            ] = matrix

        labels[
            cursor:
            cursor + patient_length
        ] = patient_labels

        hour_index[
            cursor:
            cursor + patient_length
        ] = np.arange(
            patient_length,
            dtype=np.int16,
        )

        patient_keys.append(
            f"{path.parent.name}:{path.stem}"
        )

        cursor += patient_length

        offsets[patient_index] = cursor

        # Free per-patient objects immediately.
        del causal_values
        del original_missing
        del feature_sets

        if (
            patient_index % 2000 == 0
            or patient_index == len(paths)
        ):
            print(
                f"{split}: "
                f"{patient_index:,}/"
                f"{len(paths):,}"
            )
            gc.collect()

    if cursor != total_rows:
        raise RuntimeError(
            f"{split}: output row-count mismatch."
        )

    summary = {
        "patients": len(paths),
        "timesteps": total_rows,
        "positive_timesteps": int(
            labels.sum()
        ),
        "files": {},
    }

    for feature_set, matrix in arrays.items():

        matrix.flush()

        output_dir = (
            PROCESSED_DIR
            / feature_set
            / split
        )

        np.save(
            output_dir / "labels.npy",
            labels,
        )

        np.save(
            output_dir / "patient_offsets.npy",
            offsets,
        )

        np.save(
            output_dir / "hour_index.npy",
            hour_index,
        )

        np.save(
            output_dir / "patient_keys.npy",
            np.asarray(
                patient_keys,
                dtype=str,
            ),
        )

        summary["files"][feature_set] = {
            "path": str(output_dir),
            "shape": [
                int(total_rows),
                int(matrix.shape[1]),
            ],
        }

    return summary


# ============================================================================
# VALIDATION
# ============================================================================

def validate_causal_primitives():

    example = pd.DataFrame({
        "HR": [
            80.0,
            np.nan,
            100.0,
        ]
    })

    causal = example.ffill()

    if causal["HR"].tolist() != [
        80.0,
        80.0,
        100.0,
    ]:
        raise AssertionError(
            "Forward-fill causality failed."
        )

    delta = (
        causal["HR"]
        - causal["HR"].shift(1)
    )

    # First timestep has no history.
    if not np.isnan(delta.iloc[0]):
        raise AssertionError(
            "First delta should have no history."
        )

    if delta.iloc[1] != 0:
        raise AssertionError(
            "Second delta incorrect."
        )

    if delta.iloc[2] != 20:
        raise AssertionError(
            "Third delta incorrect."
        )

    return {
        "status": "passed",
        "forward_fill": [
            80.0,
            80.0,
            100.0,
        ],
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print("X-FedSepsis — STEP 2: CAUSAL PREPROCESSING")
    print("=" * 72)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # 1. Verify raw dataset
    # ------------------------------------------------------------

    all_files = get_all_patient_files()

    # ------------------------------------------------------------
    # 2. Verify frozen split
    # ------------------------------------------------------------

    split_paths = load_frozen_split_paths(
        all_files
    )

    print(
        f"Verified frozen split for "
        f"{len(all_files):,} patients."
    )

    print(
        f"Train:      {len(split_paths['train']):,}"
    )

    print(
        f"Validation: {len(split_paths['validation']):,}"
    )

    print(
        f"Test:       {len(split_paths['test']):,}"
    )

    # ------------------------------------------------------------
    # 3. Training-only medians
    # ------------------------------------------------------------

    print(
        "\nCalculating training-only medians..."
    )

    medians, imputation_statistics = (
        calculate_training_medians(
            split_paths["train"]
        )
    )

    imputation_statistics.to_csv(
        RESULTS_DIR
        / "imputation_statistics.csv",
        index=False,
    )

    medians.rename(
        "training_median_after_causal_ffill"
    ).rename_axis(
        "feature"
    ).to_csv(
        RESULTS_DIR
        / "training_medians.csv"
    )

    # ------------------------------------------------------------
    # 4. Causality validation
    # ------------------------------------------------------------

    causal_validation = (
        validate_causal_primitives()
    )

    print(
        "Causality validation: PASSED"
    )

    # ------------------------------------------------------------
    # 5. Generate F0-F3
    # ------------------------------------------------------------

    split_summary = {}

    for split in SPLITS:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Generating {split.upper()}"
        )

        print(
            "=" * 60
        )

        split_summary[split] = write_split(
            split,
            split_paths[split],
            medians,
        )

    # ------------------------------------------------------------
    # 6. Feature manifest
    # ------------------------------------------------------------

    sample_values, sample_missing, _ = (
        read_patient(
            split_paths["train"][0]
        )
    )

    manifest_frames = build_feature_sets(
        sample_values,
        sample_missing,
        medians,
    )

    feature_manifest = {
        feature_set:
            frame.columns.tolist()
        for feature_set, frame
        in manifest_frames.items()
    }

    with open(
        RESULTS_DIR
        / "feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            feature_manifest,
            handle,
            indent=2,
        )

    feature_statistics = pd.DataFrame([
        {
            "feature_set": feature_set,
            "feature_index": index,
            "feature": feature,
        }

        for feature_set, features
        in feature_manifest.items()

        for index, feature
        in enumerate(features)
    ])

    feature_statistics.to_csv(
        RESULTS_DIR
        / "feature_statistics.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------

    summary = {

        "project":
            "X-FedSepsis",

        "step":
            "2",

        "raw_training_data_modified":
            False,

        "split_file":
            str(SPLIT_FILE),

        "feature_counts": {
            feature_set:
                len(features)
            for feature_set, features
            in feature_manifest.items()
        },

        "feature_sets": {

            "F0":
                "Raw clinical variables + missingness indicators",

            "F1":
                "F0 + causal 1h/3h/6h deltas + history validity",

            "F2":
                "F1 + causal 3h/6h rolling statistics + validity",

            "F3":
                "F2 + clinical ratios/scores",
        },

        "clinical_features": [
            "ShockIndex_HR_SBP",
            "BUN_Creatinine_ratio",
            "O2Sat_FiO2_ratio",
            "SIRS_score",
            "MEWS_4_variable",
        ],

        "excluded_scores": [
            "SOFA",
            "qSOFA",
        ],

        "causality": {
            "forward_fill":
                "Within patient only",

            "deltas":
                "Current minus historical value",

            "rolling":
                "Trailing windows only",

            "median_imputation":
                "Training patients only",

            "future_information":
                False,
        },

        "causal_validation":
            causal_validation,

        "splits":
            split_summary,
    }

    with open(
        RESULTS_DIR
        / "preprocessing_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("\n")
    print("=" * 72)
    print("STEP 2 COMPLETE")
    print("=" * 72)

    print(
        "Feature counts:"
    )

    for name, count in summary[
        "feature_counts"
    ].items():

        print(
            f"  {name}: {count}"
        )

    print(
        "\nProcessed data:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        "\nResults:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()