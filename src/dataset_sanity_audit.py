from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

SET_A_DIR = PROJECT_ROOT / "data" / "training" / "training_setA"
SET_B_DIR = PROJECT_ROOT / "data" / "training" / "training_setB"

RESULTS_DIR = PROJECT_ROOT / "results"

warnings.filterwarnings("ignore")


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
    "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Fibrinogen", "Platelets", "Age", "Gender",
    "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"
]


# Plausible ranges for basic sanity checking.
# These are NOT clinical diagnostic cutoffs.
RANGES = {
    "HR": (0, 300),
    "O2Sat": (0, 100),
    "Temp": (20, 45),
    "SBP": (0, 300),
    "MAP": (0, 250),
    "DBP": (0, 200),
    "Resp": (0, 100),
    "EtCO2": (0, 150),
    "FiO2": (0, 1.5),
    "pH": (6, 9),
    "PaCO2": (0, 200),
    "SaO2": (0, 100),
    "AST": (0, 10000),
    "BUN": (0, 300),
    "Calcium": (0, 20),
    "Chloride": (0, 200),
    "Creatinine": (0, 30),
    "Glucose": (0, 1000),
    "Lactate": (0, 30),
    "Magnesium": (0, 20),
    "Phosphate": (0, 20),
    "Potassium": (0, 20),
    "Hct": (0, 100),
    "Hgb": (0, 30),
    "WBC": (0, 300),
    "Platelets": (0, 2000),
    "Age": (0, 120),
    "Gender": (0, 1),
    "Unit1": (0, 1),
    "Unit2": (0, 1),
    "ICULOS": (1, 1000),
    "SepsisLabel": (0, 1),
}


# ============================================================
# DISCOVER FILES
# ============================================================

print("=" * 75)
print("X-FedSepsis — STEP 1.5: FULL DATASET SANITY AUDIT")
print("=" * 75)

files_a = sorted(SET_A_DIR.glob("*.psv"))
files_b = sorted(SET_B_DIR.glob("*.psv"))

all_files = files_a + files_b

print(f"\nSet A patients: {len(files_a):,}")
print(f"Set B patients: {len(files_b):,}")
print(f"Total patients: {len(all_files):,}")


# ============================================================
# STORAGE
# ============================================================

feature_missing = {
    feature: 0
    for feature in EXPECTED_FEATURES
}

feature_total = {
    feature: 0
    for feature in EXPECTED_FEATURES
}

missing_by_class = {
    "sepsis": {
        feature: 0
        for feature in EXPECTED_FEATURES
    },
    "non_sepsis": {
        feature: 0
        for feature in EXPECTED_FEATURES
    }
}

total_by_class = {
    "sepsis": 0,
    "non_sepsis": 0
}

rows_by_class = {
    "sepsis": 0,
    "non_sepsis": 0
}

patient_lengths = {
    "sepsis": [],
    "non_sepsis": []
}

positive_timesteps = 0
negative_timesteps = 0

patients_with_nonbinary_labels = []
patients_with_label_transition_back = []

duplicate_row_counts = []

range_violations = {
    feature: 0
    for feature in RANGES
}

range_examples = []

schema_errors = []

patient_records = []


# ============================================================
# MAIN AUDIT
# ============================================================

for i, path in enumerate(all_files, start=1):

    try:
        df = pd.read_csv(path, sep="|")

    except Exception as exc:
        schema_errors.append({
            "patient_id": path.stem,
            "error": str(exc)
        })
        continue

    patient_id = path.stem

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    missing_columns = [
        c for c in EXPECTED_FEATURES
        if c not in df.columns
    ]

    if missing_columns:
        schema_errors.append({
            "patient_id": patient_id,
            "error": f"Missing columns: {missing_columns}"
        })
        continue

    # --------------------------------------------------------
    # Sepsis labels
    # --------------------------------------------------------

    labels = pd.to_numeric(
        df["SepsisLabel"],
        errors="coerce"
    )

    nonbinary = labels.dropna()[
        ~labels.dropna().isin([0, 1])
    ]

    if len(nonbinary) > 0:
        patients_with_nonbinary_labels.append({
            "patient_id": patient_id,
            "values": nonbinary.unique().tolist()
        })

    # Treat only exact 1 as positive.
    positive_mask = labels == 1

    patient_has_sepsis = bool(
        positive_mask.any()
    )

    class_name = (
        "sepsis"
        if patient_has_sepsis
        else "non_sepsis"
    )

    # --------------------------------------------------------
    # Label counts
    # --------------------------------------------------------

    pos_count = int(positive_mask.sum())
    neg_count = int((labels == 0).sum())

    positive_timesteps += pos_count
    negative_timesteps += neg_count

    rows_by_class[class_name] += len(df)
    patient_lengths[class_name].append(len(df))

    total_by_class[class_name] += 1

    # --------------------------------------------------------
    # Label transition behavior
    # --------------------------------------------------------

    clean_labels = labels.dropna().to_numpy()

    # Once SepsisLabel becomes 1, check whether it ever returns
    # to 0 later.
    if len(clean_labels) > 0:

        first_positive = np.where(
            clean_labels == 1
        )[0]

        if len(first_positive) > 0:

            first_idx = first_positive[0]

            if np.any(
                clean_labels[first_idx:] == 0
            ):
                patients_with_label_transition_back.append(
                    patient_id
                )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    for feature in EXPECTED_FEATURES:

        missing = int(df[feature].isna().sum())
        total = len(df)

        feature_missing[feature] += missing
        feature_total[feature] += total

        missing_by_class[
            class_name
        ][feature] += missing

    # --------------------------------------------------------
    # Basic range checks
    # --------------------------------------------------------

    for feature, (lower, upper) in RANGES.items():

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        violations = (
            (values < lower) |
            (values > upper)
        )

        count = int(violations.sum())

        range_violations[feature] += count

        if count > 0 and len(range_examples) < 100:

            bad_values = values[violations].head(5)

            range_examples.append({
                "patient_id": patient_id,
                "feature": feature,
                "count": count,
                "examples": bad_values.tolist()
            })

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    duplicate_row_counts.append(
        duplicates
    )

    # --------------------------------------------------------
    # Patient summary
    # --------------------------------------------------------

    patient_records.append({
        "patient_id": patient_id,
        "dataset": (
            "A"
            if path.parent.name == "training_setA"
            else "B"
        ),
        "hours": len(df),
        "sepsis": int(patient_has_sepsis),
        "positive_timesteps": pos_count,
        "negative_timesteps": neg_count,
        "duplicate_rows": duplicates,
    })

    if i % 1000 == 0 or i == len(all_files):
        print(
            f"Processed {i:,}/{len(all_files):,}"
        )


# ============================================================
# MISSINGNESS TABLE
# ============================================================

missingness_rows = []

for feature in EXPECTED_FEATURES:

    total = feature_total[feature]
    missing = feature_missing[feature]

    sepsis_total = total_by_class["sepsis"]
    non_total = total_by_class["non_sepsis"]

    sepsis_missing = missing_by_class[
        "sepsis"
    ][feature]

    non_missing = missing_by_class[
        "non_sepsis"
    ][feature]

    missingness_rows.append({
        "feature": feature,

        "total_values": total,
        "missing_values": missing,
        "missing_percent": (
            missing / total * 100
            if total else 0
        ),

        "sepsis_missing_values": sepsis_missing,
        "sepsis_missing_percent": (
            sepsis_missing /
            rows_by_class["sepsis"] *
            100
            if rows_by_class["sepsis"]
            else 0
        ),

        "non_sepsis_missing_values": non_missing,
        "non_sepsis_missing_percent": (
            non_missing /
            rows_by_class["non_sepsis"] *
            100
            if rows_by_class["non_sepsis"]
            else 0
        ),
    })

missingness_df = pd.DataFrame(
    missingness_rows
).sort_values(
    "missing_percent",
    ascending=False
)


# ============================================================
# PATIENT LENGTH SUMMARY
# ============================================================

length_rows = []

for class_name in ["sepsis", "non_sepsis"]:

    values = np.array(
        patient_lengths[class_name]
    )

    length_rows.append({
        "class": class_name,
        "patients": len(values),
        "min_hours": int(values.min()),
        "p25_hours": float(np.percentile(values, 25)),
        "median_hours": float(np.median(values)),
        "mean_hours": float(np.mean(values)),
        "p75_hours": float(np.percentile(values, 75)),
        "max_hours": int(values.max()),
    })

length_df = pd.DataFrame(length_rows)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

missingness_path = (
    RESULTS_DIR /
    "full_dataset_missingness.csv"
)

length_path = (
    RESULTS_DIR /
    "patient_length_by_class.csv"
)

patients_path = (
    RESULTS_DIR /
    "full_dataset_patient_audit.csv"
)

violations_path = (
    RESULTS_DIR /
    "range_violations.csv"
)

missingness_df.to_csv(
    missingness_path,
    index=False
)

length_df.to_csv(
    length_path,
    index=False
)

pd.DataFrame(patient_records).to_csv(
    patients_path,
    index=False
)

pd.DataFrame([
    {
        "feature": feature,
        "violations": count
    }
    for feature, count
    in range_violations.items()
]).to_csv(
    violations_path,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "patients": len(all_files),

    "rows": int(
        rows_by_class["sepsis"] +
        rows_by_class["non_sepsis"]
    ),

    "patients_by_class": {
        "sepsis": total_by_class["sepsis"],
        "non_sepsis": total_by_class["non_sepsis"],
    },

    "rows_by_class": rows_by_class,

    "timesteps": {
        "positive": positive_timesteps,
        "negative": negative_timesteps,
        "total": (
            positive_timesteps +
            negative_timesteps
        )
    },

    "label_checks": {
        "patients_with_nonbinary_labels":
            len(patients_with_nonbinary_labels),

        "patients_with_label_transition_back":
            len(patients_with_label_transition_back),
    },

    "duplicate_rows": {
        "patients_with_duplicates":
            int(
                sum(
                    x > 0
                    for x in duplicate_row_counts
                )
            ),

        "total_duplicate_rows":
            int(sum(duplicate_row_counts)),
    },

    "range_checks": {
        feature: int(count)
        for feature, count
        in range_violations.items()
    },

    "files": {
        "missingness":
            str(missingness_path),

        "patient_lengths":
            str(length_path),

        "patient_audit":
            str(patients_path),

        "range_violations":
            str(violations_path),
    }
}


summary_path = (
    RESULTS_DIR /
    "full_dataset_sanity_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FULL DATASET SANITY AUDIT — RESULTS")
print("=" * 75)

print(
    f"\nPatients: {len(all_files):,}"
)

print(
    f"Sepsis patients: "
    f"{total_by_class['sepsis']:,}"
)

print(
    f"Non-sepsis patients: "
    f"{total_by_class['non_sepsis']:,}"
)

print(
    f"\nPositive timesteps: "
    f"{positive_timesteps:,}"
)

print(
    f"Negative timesteps: "
    f"{negative_timesteps:,}"
)

print(
    f"Total timesteps: "
    f"{positive_timesteps + negative_timesteps:,}"
)


print("\n" + "-" * 75)
print("LABEL CHECKS")
print("-" * 75)

print(
    "Non-binary label patients:",
    len(patients_with_nonbinary_labels)
)

print(
    "Label transitions 1 -> 0:",
    len(patients_with_label_transition_back)
)


print("\n" + "-" * 75)
print("DUPLICATE ROW CHECK")
print("-" * 75)

print(
    "Patients with duplicate rows:",
    sum(x > 0 for x in duplicate_row_counts)
)

print(
    "Total duplicate rows:",
    sum(duplicate_row_counts)
)


print("\n" + "-" * 75)
print("MISSINGNESS — TOP 20")
print("-" * 75)

print(
    missingness_df[
        [
            "feature",
            "missing_percent",
            "sepsis_missing_percent",
            "non_sepsis_missing_percent"
        ]
    ].head(20).to_string(index=False)
)


print("\n" + "-" * 75)
print("PATIENT LENGTH")
print("-" * 75)

print(
    length_df.to_string(index=False)
)


print("\n" + "-" * 75)
print("RANGE CHECKS")
print("-" * 75)

for feature, count in range_violations.items():

    if count > 0:
        print(
            f"{feature}: {count:,} violations"
        )

print("\nResults saved to:")
print(summary_path)
print(missingness_path)
print(length_path)
print(patients_path)
print(violations_path)

print("\nSTEP 1.5 COMPLETE")