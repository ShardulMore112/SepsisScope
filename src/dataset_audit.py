from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

SET_A_DIR = PROJECT_ROOT / "data" / "training" / "training_setA"
SET_B_DIR = PROJECT_ROOT / "data" / "training" / "training_setB"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42

TEST_SIZE = 8068
VALIDATION_SIZE = 0.15

PILOT_POSITIVE = 300
PILOT_NEGATIVE = 700

EXPECTED_TOTAL_PATIENTS = 40336
EXPECTED_TEST_PATIENTS = 8068
EXPECTED_DEVELOPMENT_PATIENTS = 32268


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================

def get_patient_files(directory):
    """
    Return all .psv patient files in a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.psv"))

    if not files:
        raise RuntimeError(f"No .psv files found in: {directory}")

    return files


def patient_id_from_file(path):
    """
    Patient ID is the filename without extension.
    Example:
        p000001.psv -> p000001
    """
    return path.stem


def audit_patient(path):
    """
    Read one patient PSV and collect basic audit information.
    """
    result = {
        "patient_id": patient_id_from_file(path),
        "file": str(path),
        "rows": 0,
        "sepsis": 0,
        "columns": None,
        "malformed": False,
        "error": None,
        "missing_values": 0,
    }

    try:
        df = pd.read_csv(path, sep="|")

        result["rows"] = len(df)
        result["columns"] = list(df.columns)

        if "SepsisLabel" not in df.columns:
            result["malformed"] = True
            result["error"] = "Missing SepsisLabel column"
            return result

        # A patient is considered sepsis-positive if SepsisLabel
        # is positive at least once.
        labels = pd.to_numeric(
            df["SepsisLabel"],
            errors="coerce"
        ).fillna(0)

        result["sepsis"] = int((labels == 1).any())

        result["missing_values"] = int(df.isna().sum().sum())

    except Exception as exc:
        result["malformed"] = True
        result["error"] = str(exc)

    return result


# ============================================================
# STEP 1A — DISCOVER PATIENT FILES
# ============================================================

print("=" * 70)
print("X-FedSepsis — STEP 1: DATASET AUDIT")
print("=" * 70)

print("\nLocating datasets...")

files_a = get_patient_files(SET_A_DIR)
files_b = get_patient_files(SET_B_DIR)

print(f"Set A files: {len(files_a):,}")
print(f"Set B files: {len(files_b):,}")

all_files = files_a + files_b

print(f"Total files: {len(all_files):,}")


# ============================================================
# STEP 1B — CHECK DUPLICATE PATIENT IDS
# ============================================================

print("\nChecking duplicate patient IDs...")

patient_ids = [patient_id_from_file(p) for p in all_files]

duplicates = pd.Series(patient_ids)[
    pd.Series(patient_ids).duplicated()
].tolist()

if duplicates:
    print(f"WARNING: {len(duplicates)} duplicate patient IDs found.")
    print("Examples:", duplicates[:10])
else:
    print("No duplicate patient IDs found.")


# ============================================================
# STEP 1C — AUDIT EVERY PATIENT
# ============================================================

print("\nAuditing patient files...")
print("This may take a few minutes.\n")

audit_results = []

for i, path in enumerate(all_files, start=1):

    result = audit_patient(path)

    if path.parent.name == "training_setA":
        result["dataset"] = "A"
    else:
        result["dataset"] = "B"

    audit_results.append(result)

    if i % 1000 == 0 or i == len(all_files):
        print(f"Processed {i:,}/{len(all_files):,}")


audit_df = pd.DataFrame(audit_results)


# ============================================================
# STEP 1D — BASIC DATASET STATISTICS
# ============================================================

valid_df = audit_df[~audit_df["malformed"]].copy()

total_patients = len(audit_df)
valid_patients = len(valid_df)

total_rows = int(valid_df["rows"].sum())

positive_patients = int(
    valid_df["sepsis"].sum()
)

negative_patients = valid_patients - positive_patients

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total patients:          {total_patients:,}")
print(f"Valid patients:          {valid_patients:,}")
print(f"Malformed patients:      {total_patients - valid_patients:,}")
print(f"Total hourly rows:       {total_rows:,}")
print(f"Sepsis-positive:         {positive_patients:,}")
print(f"Non-sepsis:              {negative_patients:,}")

if valid_patients > 0:
    print(
        f"Patient-level prevalence: "
        f"{positive_patients / valid_patients * 100:.3f}%"
    )


# ============================================================
# SET A / SET B SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SET A / SET B")
print("=" * 70)

for dataset_name in ["A", "B"]:

    subset = valid_df[valid_df["dataset"] == dataset_name]

    n = len(subset)
    positives = int(subset["sepsis"].sum())
    negatives = n - positives
    rows = int(subset["rows"].sum())

    print(f"\nSet {dataset_name}")
    print(f"  Patients:       {n:,}")
    print(f"  Sepsis:         {positives:,}")
    print(f"  Non-sepsis:     {negatives:,}")
    print(f"  Hourly rows:    {rows:,}")

    if n > 0:
        print(f"  Prevalence:     {positives / n * 100:.3f}%")


# ============================================================
# PATIENT LENGTH STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT LENGTH STATISTICS")
print("=" * 70)

lengths = valid_df["rows"]

print(f"Minimum hours:    {lengths.min():.0f}")
print(f"25th percentile:  {lengths.quantile(0.25):.1f}")
print(f"Median hours:     {lengths.median():.1f}")
print(f"Mean hours:       {lengths.mean():.1f}")
print(f"75th percentile:  {lengths.quantile(0.75):.1f}")
print(f"Maximum hours:    {lengths.max():.0f}")


# ============================================================
# SCHEMA CHECK
# ============================================================

print("\n" + "=" * 70)
print("SCHEMA CHECK")
print("=" * 70)

column_signatures = {}

for _, row in valid_df.iterrows():

    columns = tuple(row["columns"])

    if columns not in column_signatures:
        column_signatures[columns] = 0

    column_signatures[columns] += 1

print(f"Unique column schemas: {len(column_signatures)}")

for i, (columns, count) in enumerate(
    column_signatures.items(),
    start=1
):
    print(f"\nSchema {i}: {count:,} patients")
    print(f"Number of columns: {len(columns)}")
    print("Columns:")
    print(list(columns))


# ============================================================
# MISSINGNESS AUDIT
# ============================================================

print("\n" + "=" * 70)
print("MISSINGNESS AUDIT")
print("=" * 70)

print("Calculating column-level missingness...")

# Sample files for detailed missingness statistics.
# The complete patient-level audit above already records
# total missing values.
#
# We use up to 1,000 patients here to keep the audit fast.

missingness_files = all_files[:min(1000, len(all_files))]

missing_counts = {}
total_values = {}

for i, path in enumerate(missingness_files, start=1):

    try:
        df = pd.read_csv(path, sep="|")

        for column in df.columns:

            missing = int(df[column].isna().sum())
            total = len(df)

            missing_counts[column] = (
                missing_counts.get(column, 0) + missing
            )

            total_values[column] = (
                total_values.get(column, 0) + total
            )

    except Exception:
        continue

missingness_rows = []

for column in missing_counts:

    missing = missing_counts[column]
    total = total_values[column]

    percentage = (
        missing / total * 100
        if total > 0
        else 0
    )

    missingness_rows.append({
        "feature": column,
        "missing_values": missing,
        "total_values": total,
        "missing_percent": percentage,
    })


missingness_df = pd.DataFrame(
    missingness_rows
).sort_values(
    "missing_percent",
    ascending=False
)

print("\nTop missing features:")

if not missingness_df.empty:
    print(
        missingness_df.head(15).to_string(index=False)
    )


# ============================================================
# STEP 1E — REMOVE MALFORMED PATIENTS FROM SPLITTING
# ============================================================

if len(valid_df) != EXPECTED_TOTAL_PATIENTS:

    print("\nWARNING:")
    print(
        f"Expected {EXPECTED_TOTAL_PATIENTS:,} valid patients "
        f"but found {len(valid_df):,}."
    )

    print(
        "The split will use the valid patients that were found."
    )


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING PATIENT-LEVEL SPLITS")
print("=" * 70)

valid_df = valid_df.reset_index(drop=True)

all_indices = np.arange(len(valid_df))

# ------------------------------------------------------------
# TEST SPLIT
# ------------------------------------------------------------

if len(valid_df) == EXPECTED_TOTAL_PATIENTS:

    test_indices, development_indices = train_test_split(
        all_indices,
        test_size=EXPECTED_DEVELOPMENT_PATIENTS,
        random_state=RANDOM_SEED,
        stratify=valid_df["sepsis"].values,
    )

else:

    # Fallback if dataset size differs.
    test_indices, development_indices = train_test_split(
        all_indices,
        test_size=1 - (TEST_SIZE / len(valid_df)),
        random_state=RANDOM_SEED,
        stratify=valid_df["sepsis"].values,
    )


# ------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

development_labels = valid_df.loc[
    development_indices,
    "sepsis"
].values

train_indices, validation_indices = train_test_split(
    development_indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=development_labels,
)


# ============================================================
# PILOT DATASET
# ============================================================

print("\nCreating 1,000-patient pilot dataset...")

# IMPORTANT:
# Pilot is selected ONLY from training patients.
# Validation and test remain untouched.

train_df = valid_df.loc[train_indices].copy()

positive_train = train_df[
    train_df["sepsis"] == 1
]

negative_train = train_df[
    train_df["sepsis"] == 0
]

if len(positive_train) < PILOT_POSITIVE:
    raise RuntimeError(
        f"Not enough positive patients for pilot: "
        f"{len(positive_train)} available."
    )

if len(negative_train) < PILOT_NEGATIVE:
    raise RuntimeError(
        f"Not enough negative patients for pilot: "
        f"{len(negative_train)} available."
    )


pilot_positive = positive_train.sample(
    n=PILOT_POSITIVE,
    random_state=RANDOM_SEED
)

pilot_negative = negative_train.sample(
    n=PILOT_NEGATIVE,
    random_state=RANDOM_SEED
)

pilot_df = pd.concat(
    [pilot_positive, pilot_negative]
)

pilot_indices = pilot_df.index.to_numpy()

# Sort for reproducibility/readability
pilot_indices = np.sort(pilot_indices)


# ============================================================
# CONVERT TO PATIENT IDS
# ============================================================

train_patient_ids = valid_df.loc[
    train_indices, "patient_id"
].to_numpy()

validation_patient_ids = valid_df.loc[
    validation_indices, "patient_id"
].to_numpy()

test_patient_ids = valid_df.loc[
    test_indices, "patient_id"
].to_numpy()

pilot_patient_ids = valid_df.loc[
    pilot_indices, "patient_id"
].to_numpy()


# ============================================================
# SPLIT SUMMARY
# ============================================================

def split_summary(name, indices):

    subset = valid_df.loc[indices]

    n = len(subset)
    positives = int(subset["sepsis"].sum())
    negatives = n - positives

    return {
        "name": name,
        "patients": n,
        "sepsis_positive": positives,
        "non_sepsis": negatives,
        "prevalence_percent": (
            positives / n * 100
            if n > 0 else 0
        ),
        "total_hours": int(subset["rows"].sum()),
    }


split_summaries = [
    split_summary("train", train_indices),
    split_summary("validation", validation_indices),
    split_summary("test", test_indices),
    split_summary("pilot", pilot_indices),
]


print("\nFinal splits:")

for summary in split_summaries:

    print(
        f"\n{summary['name'].upper()}"
    )

    print(
        f"  Patients:      {summary['patients']:,}"
    )

    print(
        f"  Sepsis:        {summary['sepsis_positive']:,}"
    )

    print(
        f"  Non-sepsis:    {summary['non_sepsis']:,}"
    )

    print(
        f"  Prevalence:    "
        f"{summary['prevalence_percent']:.3f}%"
    )

    print(
        f"  Total hours:   "
        f"{summary['total_hours']:,}"
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE SPLITS
# ============================================================

split_file = SPLIT_DIR / "split_indices.npz"

np.savez_compressed(
    split_file,

    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    pilot_indices=pilot_indices,

    train_patient_ids=train_patient_ids,
    validation_patient_ids=validation_patient_ids,
    test_patient_ids=test_patient_ids,
    pilot_patient_ids=pilot_patient_ids,

    random_seed=np.array([RANDOM_SEED]),
)


# ============================================================
# SAVE HUMAN-READABLE PATIENT SPLIT CSV
# ============================================================

split_records = []

for split_name, indices in [
    ("train", train_indices),
    ("validation", validation_indices),
    ("test", test_indices),
    ("pilot", pilot_indices),
]:

    for idx in indices:

        row = valid_df.loc[idx]

        split_records.append({
            "patient_id": row["patient_id"],
            "dataset": row["dataset"],
            "split": split_name,
            "sepsis": int(row["sepsis"]),
            "hours": int(row["rows"]),
        })


split_csv = RESULTS_DIR / "patient_splits.csv"

pd.DataFrame(
    split_records
).sort_values(
    ["split", "patient_id"]
).to_csv(
    split_csv,
    index=False
)


# ============================================================
# SAVE AUDIT RESULTS
# ============================================================

audit_csv = RESULTS_DIR / "dataset_audit_patients.csv"

audit_df.to_csv(
    audit_csv,
    index=False
)


missingness_csv = RESULTS_DIR / "missingness_audit.csv"

missingness_df.to_csv(
    missingness_csv,
    index=False
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

audit_summary = {
    "project": "X-FedSepsis",

    "random_seed": RANDOM_SEED,

    "dataset": {
        "set_a_patients": int(
            (valid_df["dataset"] == "A").sum()
        ),
        "set_b_patients": int(
            (valid_df["dataset"] == "B").sum()
        ),
        "total_patients": int(valid_patients),
        "total_hourly_rows": int(total_rows),
        "sepsis_positive": int(positive_patients),
        "non_sepsis": int(negative_patients),
        "prevalence_percent": (
            positive_patients / valid_patients * 100
        ),
    },

    "splits": split_summaries,

    "files": {
        "split_indices": str(split_file),
        "patient_splits": str(split_csv),
        "patient_audit": str(audit_csv),
        "missingness_audit": str(missingness_csv),
    },
}


summary_json = RESULTS_DIR / "dataset_audit_summary.json"

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit_summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)

print("\nCreated:")

print(f"  {split_file}")
print(f"  {split_csv}")
print(f"  {audit_csv}")
print(f"  {missingness_csv}")
print(f"  {summary_json}")

print("\nRaw dataset was NOT modified.")

print("\nNext step:")
print("Review the audit results before starting preprocessing.")