"""
Migrate existing GRU family experiments to the new
resume-safe experiment format.

Existing files:
    gru_family_results.json
    gru-1.pt ... gru-4.pt
    gru-1_test_predictions.npz ... gru-4_test_predictions.npz

This script:
    1. Reads the existing global result file.
    2. Extracts GRU-1 ... GRU-4 results.
    3. Creates individual *_result.json files.
    4. Creates *_complete.json markers.
    5. Does NOT retrain anything.
    6. Does NOT modify checkpoints or predictions.
"""

from pathlib import Path
import json
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_result"
    / "gru_family"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "gru_family"
)

GLOBAL_RESULT_FILE = (
    RESULTS_ROOT
    / "gru_family_results.json"
)


# ============================================================
# EXPERIMENTS THAT ARE ALREADY KNOWN TO BE COMPLETE
# ============================================================

COMPLETED_EXPERIMENTS = [
    "gru-1",
    "gru-2",
    "gru-3",
    "gru-4",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):

    if not isinstance(value, str):
        return None

    value = value.strip().lower()

    # Accept:
    # GRU-1
    # gru-1
    # GRU1
    # gru1

    if value in {
        "gru-1",
        "gru1",
    }:
        return "gru-1"

    if value in {
        "gru-2",
        "gru2",
    }:
        return "gru-2"

    if value in {
        "gru-3",
        "gru3",
    }:
        return "gru-3"

    if value in {
        "gru-4",
        "gru4",
    }:
        return "gru-4"

    if value in {
        "gru-5",
        "gru5",
    }:
        return "gru-5"

    return None


def identify_experiment(record):

    """
    Try to identify an experiment from a result record
    regardless of the exact old JSON schema.
    """

    if not isinstance(record, dict):
        return None

    # Most likely keys
    possible_keys = [
        "id",
        "experiment",
        "experiment_id",
        "name",
        "model",
    ]

    for key in possible_keys:

        if key in record:

            normalized = normalize_id(
                record[key]
            )

            if normalized is not None:
                return normalized

    return None


def recursively_find_results(obj):

    """
    Recursively search the old JSON structure for
    dictionaries representing GRU experiments.
    """

    found = {}

    if isinstance(obj, dict):

        experiment_id = identify_experiment(
            obj
        )

        if experiment_id is not None:

            found[
                experiment_id
            ] = obj

        for value in obj.values():

            nested = (
                recursively_find_results(
                    value
                )
            )

            found.update(nested)

    elif isinstance(obj, list):

        for item in obj:

            nested = (
                recursively_find_results(
                    item
                )
            )

            found.update(nested)

    return found


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GRU FAMILY — EXISTING RESULT MIGRATION"
    )
    print("=" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nGlobal result file:\n"
        f"{GLOBAL_RESULT_FILE}"
    )

    # --------------------------------------------------------
    # CHECK GLOBAL RESULT FILE
    # --------------------------------------------------------

    if not GLOBAL_RESULT_FILE.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"{GLOBAL_RESULT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD OLD RESULTS
    # --------------------------------------------------------

    with open(
        GLOBAL_RESULT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        old_results = json.load(f)

    print(
        "\nExisting global result file loaded."
    )

    # --------------------------------------------------------
    # EXTRACT EXPERIMENT RESULTS
    # --------------------------------------------------------

    extracted = (
        recursively_find_results(
            old_results
        )
    )

    print(
        "\nExperiments found in old result file:"
    )

    for experiment_id in sorted(
        extracted.keys()
    ):

        print(
            f"  {experiment_id}"
        )

    # --------------------------------------------------------
    # MIGRATE GRU-1 ... GRU-4
    # --------------------------------------------------------

    migrated = []

    for experiment_id in (
        COMPLETED_EXPERIMENTS
    ):

        print("\n" + "-" * 70)

        print(
            f"Checking {experiment_id.upper()}"
        )

        checkpoint_path = (
            CHECKPOINT_ROOT
            / f"{experiment_id}.pt"
        )

        prediction_path = (
            RESULTS_ROOT
            / f"{experiment_id}_test_predictions.npz"
        )

        result_path = (
            RESULTS_ROOT
            / f"{experiment_id}_result.json"
        )

        completion_path = (
            RESULTS_ROOT
            / f"{experiment_id}_complete.json"
        )

        # ----------------------------------------------------
        # VERIFY CHECKPOINT
        # ----------------------------------------------------

        if not checkpoint_path.exists():

            print(
                f"ERROR: Missing checkpoint:\n"
                f"{checkpoint_path}"
            )

            raise FileNotFoundError(
                checkpoint_path
            )

        print(
            f"Checkpoint: FOUND"
        )

        # ----------------------------------------------------
        # VERIFY TEST PREDICTIONS
        # ----------------------------------------------------

        if not prediction_path.exists():

            print(
                f"ERROR: Missing test predictions:\n"
                f"{prediction_path}"
            )

            raise FileNotFoundError(
                prediction_path
            )

        print(
            f"Test predictions: FOUND"
        )

        # ----------------------------------------------------
        # FIND RESULT
        # ----------------------------------------------------

        if experiment_id not in extracted:

            raise RuntimeError(
                f"Could not find result information "
                f"for {experiment_id.upper()} "
                f"in gru_family_results.json."
            )

        result = extracted[
            experiment_id
        ]

        # ----------------------------------------------------
        # ADD MIGRATION METADATA
        # ----------------------------------------------------

        if isinstance(
            result,
            dict
        ):

            result = dict(result)

            result[
                "migration"
            ] = {

                "migrated_from":
                    "gru_family_results.json",

                "migrated_at":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "original_experiment":
                    experiment_id,
            }

            result[
                "checkpoint"
            ] = str(
                checkpoint_path
            )

            result[
                "test_predictions"
            ] = str(
                prediction_path
            )

        # ----------------------------------------------------
        # SAVE INDIVIDUAL RESULT
        # ----------------------------------------------------

        with open(
            result_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )

        print(
            f"Individual result created:\n"
            f"{result_path}"
        )

        # ----------------------------------------------------
        # SAVE COMPLETION MARKER
        # ----------------------------------------------------

        completion_record = {

            "experiment":
                experiment_id.upper(),

            "status":
                "COMPLETE",

            "completed_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "migrated":
                True,

            "migration_source":
                str(
                    GLOBAL_RESULT_FILE
                ),

            "result_file":
                str(result_path),

            "checkpoint":
                str(checkpoint_path),

            "test_predictions":
                str(prediction_path),
        }

        with open(
            completion_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                completion_record,
                f,
                indent=2,
            )

        print(
            f"Completion marker created:\n"
            f"{completion_path}"
        )

        migrated.append(
            experiment_id
        )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "MIGRATION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nMigrated experiments:"
    )

    for experiment_id in migrated:

        print(
            f"  {experiment_id.upper()} "
            f"→ COMPLETE"
        )

    print("\n")
    print(
        "GRU-1 through GRU-4 will now be skipped "
        "by the resume-safe gru_family.py."
    )

    print(
        "\nGRU-5 will be the next experiment to run."
    )

    print("\n")
    print(
        "IMPORTANT:"
    )

    print(
        "No model was trained."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "No test prediction was modified."
    )

    print(
        "No existing result was deleted."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()