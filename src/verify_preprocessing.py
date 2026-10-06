"""
X-FedSepsis — STEP 2 VERIFICATION AUDIT

Verifies the outputs produced by Step 2 causal preprocessing.

Checks:
1. Required files/directories exist
2. Feature counts and feature-name manifest
3. F0 -> F1 -> F2 -> F3 hierarchy
4. Train/validation/test patient counts
5. Patient alignment between X, labels, offsets and patient keys
6. No NaN / Inf in processed arrays
7. Label integrity
8. No patient overlap between splits
9. Sequence/offset integrity
10. Feature-name consistency across splits
11. Training-only median/imputation metadata
12. Basic causal feature-name sanity checks
13. Preprocessing summary consistency

This script DOES NOT modify raw or processed data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "preprocessing_result"

MANIFEST_FILE = RESULTS_DIR / "feature_manifest.json"
SUMMARY_FILE = RESULTS_DIR / "preprocessing_summary.json"
MEDIANS_FILE = RESULTS_DIR / "training_medians.csv"
IMPUTATION_FILE = RESULTS_DIR / "imputation_statistics.csv"

SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "split_indices.npz"

FEATURE_SETS = ("F0", "F1", "F2", "F3")
SPLITS = ("train", "validation", "test")

EXPECTED_PATIENT_COUNTS = {
    "train": 27427,
    "validation": 4841,
    "test": 8068,
}

EXPECTED_FEATURE_COUNTS = {
    "F0": 80,
    "F1": 152,
    "F2": 296,
    "F3": 306,
}


# ============================================================================
# REPORTING
# ============================================================================

checks = []
failures = []
warnings = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"

    checks.append(
        {
            "check": name,
            "status": status,
            "detail": detail,
        }
    )

    if passed:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        failures.append(
            {
                "check": name,
                "detail": detail,
            }
        )

    if detail:
        print(f"       {detail}")


def warn(name: str, detail: str) -> None:
    print(f"[WARN] {name}")
    print(f"       {detail}")

    warnings.append(
        {
            "check": name,
            "detail": detail,
        }
    )


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================================
# FILE HELPERS
# ============================================================================

def required_path_check() -> None:
    section("1. REQUIRED FILES")

    required = [
        PROJECT_ROOT,
        PROCESSED_DIR,
        RESULTS_DIR,
        MANIFEST_FILE,
        SUMMARY_FILE,
        MEDIANS_FILE,
        IMPUTATION_FILE,
        SPLIT_FILE,
    ]

    for path in required:
        record(
            f"Exists: {path.name}",
            path.exists(),
            str(path),
        )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# MANIFEST
# ============================================================================

def verify_manifest() -> dict:
    section("2. FEATURE MANIFEST")

    if not MANIFEST_FILE.exists():
        record(
            "Feature manifest exists",
            False,
            str(MANIFEST_FILE),
        )
        return {}

    manifest = load_json(MANIFEST_FILE)

    record(
        "Manifest contains F0/F1/F2/F3",
        all(name in manifest for name in FEATURE_SETS),
        f"Found: {list(manifest.keys())}",
    )

    for feature_set in FEATURE_SETS:
        features = manifest.get(feature_set, [])

        expected = EXPECTED_FEATURE_COUNTS[feature_set]

        record(
            f"{feature_set} feature count",
            len(features) == expected,
            f"Expected {expected}, found {len(features)}",
        )

        record(
            f"{feature_set} feature names unique",
            len(features) == len(set(features)),
            f"{len(features)} names",
        )

        duplicate_features = [
            feature
            for feature in set(features)
            if features.count(feature) > 1
        ]

        if duplicate_features:
            warn(
                f"{feature_set} duplicate features",
                str(duplicate_features[:20]),
            )

    return manifest


# ============================================================================
# FEATURE HIERARCHY
# ============================================================================

def verify_feature_hierarchy(manifest: dict) -> None:
    section("3. F0 -> F1 -> F2 -> F3 HIERARCHY")

    if not manifest:
        record(
            "Feature hierarchy",
            False,
            "Manifest unavailable.",
        )
        return

    for previous, current in zip(FEATURE_SETS[:-1], FEATURE_SETS[1:]):
        previous_features = set(manifest[previous])
        current_features = set(manifest[current])

        missing = previous_features - current_features

        record(
            f"{previous} is subset of {current}",
            len(missing) == 0,
            (
                "All previous features retained."
                if not missing
                else f"Missing: {sorted(missing)[:20]}"
            ),
        )


# ============================================================================
# PROCESSED DATA
# ============================================================================

def load_split_arrays(feature_set: str, split: str):
    directory = PROCESSED_DIR / feature_set / split

    x_file = directory / "X.npy"
    labels_file = directory / "labels.npy"
    offsets_file = directory / "patient_offsets.npy"
    keys_file = directory / "patient_keys.npy"
    names_file = directory / "feature_names.json"

    files = {
        "X": x_file,
        "labels": labels_file,
        "offsets": offsets_file,
        "keys": keys_file,
        "feature_names": names_file,
    }

    for name, path in files.items():
        record(
            f"{feature_set}/{split}/{name} exists",
            path.exists(),
            str(path),
        )

    if not all(path.exists() for path in files.values()):
        return None

    X = np.load(x_file, mmap_mode="r")
    labels = np.load(labels_file, mmap_mode="r")
    offsets = np.load(offsets_file)
    keys = np.load(keys_file, allow_pickle=True)

    with open(names_file, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return {
        "X": X,
        "labels": labels,
        "offsets": offsets,
        "keys": keys,
        "feature_names": feature_names,
    }


def verify_processed_data(manifest: dict):
    section("4. PROCESSED DATA INTEGRITY")

    loaded = {}

    for feature_set in FEATURE_SETS:
        loaded[feature_set] = {}

        for split in SPLITS:
            print()
            print(f"--- {feature_set} / {split} ---")

            data = load_split_arrays(feature_set, split)

            if data is None:
                continue

            loaded[feature_set][split] = data

            X = data["X"]
            labels = data["labels"]
            offsets = data["offsets"]
            keys = data["keys"]
            feature_names = data["feature_names"]

            expected_features = EXPECTED_FEATURE_COUNTS[feature_set]
            expected_patients = EXPECTED_PATIENT_COUNTS[split]

            # ------------------------------------------------------------
            # Shape checks
            # ------------------------------------------------------------

            record(
                f"{feature_set}/{split} feature dimension",
                X.ndim == 2 and X.shape[1] == expected_features,
                f"X shape = {X.shape}",
            )

            record(
                f"{feature_set}/{split} labels dimension",
                labels.ndim == 1 and labels.shape[0] == X.shape[0],
                f"labels shape = {labels.shape}",
            )

            record(
                f"{feature_set}/{split} offsets dimension",
                offsets.ndim == 1
                and offsets.shape[0] == expected_patients + 1,
                f"offsets shape = {offsets.shape}",
            )

            record(
                f"{feature_set}/{split} patient count",
                len(keys) == expected_patients,
                f"Expected {expected_patients}, found {len(keys)}",
            )

            # ------------------------------------------------------------
            # Feature names
            # ------------------------------------------------------------

            record(
                f"{feature_set}/{split} feature names count",
                len(feature_names) == expected_features,
                f"Expected {expected_features}, found {len(feature_names)}",
            )

            if feature_set in manifest:
                record(
                    f"{feature_set}/{split} feature names match manifest",
                    feature_names == manifest[feature_set],
                    "Processed feature names match manifest.",
                )

            # ------------------------------------------------------------
            # Numerical integrity
            # ------------------------------------------------------------

            finite = np.isfinite(X).all()

            record(
                f"{feature_set}/{split} no NaN/Inf",
                finite,
                "All processed feature values are finite.",
            )

            # ------------------------------------------------------------
            # Labels
            # ------------------------------------------------------------

            unique_labels = np.unique(labels)

            record(
                f"{feature_set}/{split} labels binary",
                np.all(np.isin(unique_labels, [0, 1])),
                f"Unique labels: {unique_labels.tolist()}",
            )

            # ------------------------------------------------------------
            # Offsets
            # ------------------------------------------------------------

            offsets_valid = (
                offsets[0] == 0
                and offsets[-1] == X.shape[0]
                and np.all(np.diff(offsets) >= 0)
            )

            record(
                f"{feature_set}/{split} offsets valid",
                offsets_valid,
                (
                    f"First={offsets[0]}, "
                    f"last={offsets[-1]}, "
                    f"rows={X.shape[0]}"
                ),
            )

            # ------------------------------------------------------------
            # Patient keys
            # ------------------------------------------------------------

            unique_keys = len(set(map(str, keys)))

            record(
                f"{feature_set}/{split} patient keys unique",
                unique_keys == len(keys),
                f"{unique_keys}/{len(keys)} unique",
            )

    return loaded


# ============================================================================
# CROSS-SPLIT PATIENT CONTAMINATION
# ============================================================================

def verify_patient_separation(loaded) -> None:
    section("5. PATIENT-LEVEL SPLIT INTEGRITY")

    for feature_set in FEATURE_SETS:
        split_keys = {}

        for split in SPLITS:
            if split not in loaded.get(feature_set, {}):
                continue

            keys = loaded[feature_set][split]["keys"]
            split_keys[split] = set(map(str, keys))

        if len(split_keys) != 3:
            warn(
                f"{feature_set} patient-overlap check",
                "One or more splits unavailable.",
            )
            continue

        train_val = split_keys["train"] & split_keys["validation"]
        train_test = split_keys["train"] & split_keys["test"]
        val_test = split_keys["validation"] & split_keys["test"]

        record(
            f"{feature_set} train/validation disjoint",
            len(train_val) == 0,
            f"Overlap: {len(train_val)}",
        )

        record(
            f"{feature_set} train/test disjoint",
            len(train_test) == 0,
            f"Overlap: {len(train_test)}",
        )

        record(
            f"{feature_set} validation/test disjoint",
            len(val_test) == 0,
            f"Overlap: {len(val_test)}",
        )


# ============================================================================
# CROSS-FEATURE-SET ALIGNMENT
# ============================================================================

def verify_feature_set_alignment(loaded) -> None:
    section("6. F0/F1/F2/F3 PATIENT ALIGNMENT")

    for split in SPLITS:
        available = [
            feature_set
            for feature_set in FEATURE_SETS
            if split in loaded.get(feature_set, {})
        ]

        if len(available) != len(FEATURE_SETS):
            warn(
                f"{split} feature-set alignment",
                f"Available: {available}",
            )
            continue

        reference = loaded["F0"][split]

        ref_keys = list(map(str, reference["keys"]))
        ref_offsets = reference["offsets"]

        for feature_set in FEATURE_SETS[1:]:
            current = loaded[feature_set][split]

            current_keys = list(map(str, current["keys"]))
            current_offsets = current["offsets"]

            record(
                f"{split}: F0 vs {feature_set} patient order",
                current_keys == ref_keys,
                "Patient ordering is identical.",
            )

            record(
                f"{split}: F0 vs {feature_set} sequence lengths",
                np.array_equal(
                    np.diff(ref_offsets),
                    np.diff(current_offsets),
                ),
                "Patient sequence lengths are identical.",
            )

            record(
                f"{split}: F0 vs {feature_set} labels",
                np.array_equal(
                    reference["labels"],
                    current["labels"],
                ),
                "Hourly labels are identical.",
            )


# ============================================================================
# CAUSAL FEATURE NAME AUDIT
# ============================================================================

def verify_causal_feature_names(manifest: dict) -> None:
    section("7. CAUSAL FEATURE-NAME AUDIT")

    if not manifest:
        record(
            "Causal feature audit",
            False,
            "Manifest unavailable.",
        )
        return

    f0 = set(manifest["F0"])
    f1 = set(manifest["F1"])
    f2 = set(manifest["F2"])
    f3 = set(manifest["F3"])

    delta_candidates = [
        feature
        for feature in f1 - f0
        if "delta" in feature.lower()
    ]

    rolling_candidates = [
        feature
        for feature in f2 - f1
        if any(
            token in feature.lower()
            for token in ("rolling", "roll", "mean", "std", "min", "max")
        )
    ]

    clinical_candidates = list(f3 - f2)

    record(
        "F1 contains temporal delta features",
        len(delta_candidates) > 0,
        f"Found {len(delta_candidates)} delta-related features.",
    )

    record(
        "F2 adds rolling/statistical features",
        len(rolling_candidates) > 0,
        f"Found {len(rolling_candidates)} rolling/statistical features.",
    )

    expected_clinical = {
        "ShockIndex_HR_SBP",
        "BUN_Creatinine_ratio",
        "O2Sat_FiO2_ratio",
        "SIRS_score",
        "MEWS_4_variable",
    }

    found_clinical = expected_clinical & f3

    record(
        "F3 contains expected clinical features",
        found_clinical == expected_clinical,
        f"Found: {sorted(found_clinical)}",
    )

    print()
    print("Delta features:")
    for feature in delta_candidates:
        print(f"  {feature}")

    print()
    print("F2-added temporal/statistical features:")
    for feature in rolling_candidates[:50]:
        print(f"  {feature}")

    print()
    print("F3-added features:")
    for feature in clinical_candidates:
        print(f"  {feature}")


# ============================================================================
# TRAINING-ONLY STATISTICS
# ============================================================================

def verify_training_statistics(summary: dict) -> None:
    section("8. TRAINING-ONLY STATISTICS")

    if not MEDIANS_FILE.exists():
        record(
            "Training medians file exists",
            False,
            str(MEDIANS_FILE),
        )
    else:
        record(
            "Training medians file exists",
            True,
            str(MEDIANS_FILE),
        )

    if not IMPUTATION_FILE.exists():
        record(
            "Imputation statistics file exists",
            False,
            str(IMPUTATION_FILE),
        )
    else:
        record(
            "Imputation statistics file exists",
            True,
            str(IMPUTATION_FILE),
        )

    causality = summary.get("causality", {})

    median_text = str(
        causality.get("median_imputation", "")
    ).lower()

    record(
        "Summary states train-only median fitting",
        "train" in median_text
        and (
            "only" in median_text
            or "patients" in median_text
        ),
        str(causality.get("median_imputation")),
    )

    record(
        "Summary states causal forward-fill",
        "within patient" in str(
            causality.get("forward_fill", "")
        ).lower(),
        str(causality.get("forward_fill")),
    )


# ============================================================================
# SUMMARY CONSISTENCY
# ============================================================================

def verify_summary(summary: dict, manifest: dict) -> None:
    section("9. PREPROCESSING SUMMARY CONSISTENCY")

    summary_counts = summary.get("feature_counts", {})

    for feature_set in FEATURE_SETS:
        expected = EXPECTED_FEATURE_COUNTS[feature_set]
        actual = summary_counts.get(feature_set)

        record(
            f"Summary {feature_set} count",
            actual == expected,
            f"Expected {expected}, summary reports {actual}",
        )

    record(
        "Raw training data marked unmodified",
        summary.get("raw_training_data_modified") is False,
        str(summary.get("raw_training_data_modified")),
    )

    causality = summary.get("causality", {})

    record(
        "Future information explicitly rejected",
        (
            "future_information" not in causality
            or causality.get("future_information") is False
        ),
        str(causality.get("future_information")),
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 72)
    print("X-FedSepsis — STEP 2 VERIFICATION AUDIT")
    print("=" * 72)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Processed: {PROCESSED_DIR}")
    print(f"Results:   {RESULTS_DIR}")

    required_path_check()

    manifest = verify_manifest()

    verify_feature_hierarchy(manifest)

    loaded = verify_processed_data(manifest)

    verify_patient_separation(loaded)

    verify_feature_set_alignment(loaded)

    verify_causal_feature_names(manifest)

    if SUMMARY_FILE.exists():
        summary = load_json(SUMMARY_FILE)
        verify_training_statistics(summary)
        verify_summary(summary, manifest)
    else:
        summary = {}

    # ------------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------------

    section("FINAL VERIFICATION RESULT")

    total = len(checks)
    passed = sum(
        1
        for check in checks
        if check["status"] == "PASS"
    )
    failed = len(failures)

    print(f"Total checks : {total}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")
    print(f"Warnings     : {len(warnings)}")

    overall_pass = failed == 0

    print()

    if overall_pass:
        print("=" * 72)
        print("STEP 2 VERIFICATION: PASSED")
        print("=" * 72)
        print()
        print("Preprocessing outputs are internally consistent.")
        print("Safe to freeze Step 2 and proceed to Step 3.")
    else:
        print("=" * 72)
        print("STEP 2 VERIFICATION: FAILED")
        print("=" * 72)
        print()
        print("DO NOT proceed to model training.")
        print("Fix the failed checks first.")

        for failure in failures:
            print()
            print(f"[FAIL] {failure['check']}")
            print(f"       {failure['detail']}")

    # ------------------------------------------------------------------------
    # Save machine-readable report
    # ------------------------------------------------------------------------

    report = {
        "project": "X-FedSepsis",
        "step": "2_verification",
        "overall_pass": overall_pass,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "warnings": len(warnings),
        "checks": checks,
        "failures": failures,
        "warnings_detail": warnings,
    }

    output_file = RESULTS_DIR / "preprocessing_verification_report.json"

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(f"Verification report:")
    print(output_file)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())