"""
Stage 3.7 preparation
Recover validation predictions for the three existing F0 family winners.

NO TRAINING.
NO TEST INFERENCE.
NO PARAMETER UPDATES.

Outputs:
    results/model_result/feature_ablation/GRU5-F0/validation_predictions.npz
    results/model_result/feature_ablation/LSTM4-F0/validation_predictions.npz
    results/model_result/feature_ablation/TR1-F0/validation_predictions.npz
"""

from pathlib import Path
import sys

import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

SRC_ROOT = PROJECT_ROOT / "src"

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_result"
    / "feature_ablation"
)


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 128


# ============================================================
# SAFE CHECKPOINT LOAD
# ============================================================

def load_checkpoint(path):

    try:

        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=DEVICE,
        )


# ============================================================
# SAVE
# ============================================================

def save_predictions(
    output_path,
    probabilities,
    labels,
    threshold=None,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dict = {
        "probabilities":
            np.asarray(
                probabilities,
                dtype=np.float32,
            ),

        "labels":
            np.asarray(
                labels,
                dtype=np.int8,
            ),
    }

    if threshold is not None:

        save_dict["threshold"] = np.array(
            threshold,
            dtype=np.float32,
        )

    np.savez_compressed(
        output_path,
        **save_dict,
    )


# ============================================================
# GRU5-F0
# ============================================================

def extract_gru():

    print()
    print("=" * 70)
    print("GRU5-F0 — VALIDATION INFERENCE")
    print("=" * 70)

    import gru_family as gf

    checkpoint_path = (
        CHECKPOINT_ROOT
        / "gru_family"
        / "gru-5.pt"
    )

    output_path = (
        OUTPUT_ROOT
        / "GRU5-F0"
        / "validation_predictions.npz"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Missing checkpoint:\n{checkpoint_path}"
        )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = gf.load_checkpoint(
        checkpoint_path,
        DEVICE,
    )

    config = (
        gf.get_model_config_from_checkpoint(
            checkpoint,
            "GRU-5",
        )
    )

    print(
        "Configuration:",
        config,
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    (
        X_val,
        y_val,
        val_offsets,
        _,
        _,
    ) = gf.load_split(
        "validation"
    )

    if X_val.shape[1] != 80:

        raise RuntimeError(
            f"Expected F0 = 80 features, "
            f"got {X_val.shape[1]}"
        )

    dataset = gf.SepsisSequenceDataset(
        X_val,
        y_val,
        val_offsets,
    )

    loader = gf.SequenceDataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = gf.GRUModel(
        input_size=80,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        state_dict = checkpoint.get(
            "state_dict"
        )

    if state_dict is None:

        raise RuntimeError(
            "GRU5 checkpoint has no model state."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Checkpoint loaded."
    )

    # --------------------------------------------------------
    # INFERENCE ONLY
    # --------------------------------------------------------

    with torch.no_grad():

        probabilities, labels = (
            gf.predict_dataset(
                model,
                loader,
                DEVICE,
            )
        )

    threshold = checkpoint.get(
        "validation_threshold"
    )

    if threshold is None:

        threshold, _ = (
            gf.find_best_threshold(
                probabilities,
                labels,
            )
        )

    save_predictions(
        output_path,
        probabilities,
        labels,
        threshold,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Samples: {len(labels):,}"
    )

    print(
        f"Positive: {np.sum(labels == 1):,}"
    )


# ============================================================
# LSTM4-F0
# ============================================================

def extract_lstm():

    print()
    print("=" * 70)
    print("LSTM4-F0 — VALIDATION INFERENCE")
    print("=" * 70)

    import lstm_family as lf

    checkpoint_path = (
        CHECKPOINT_ROOT
        / "lstm_family"
        / "lstm-4.pt"
    )

    output_path = (
        OUTPUT_ROOT
        / "LSTM4-F0"
        / "validation_predictions.npz"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Missing checkpoint:\n{checkpoint_path}"
        )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = lf.load_checkpoint(
        checkpoint_path,
        DEVICE,
    )

    config = (
        lf.get_model_config_from_checkpoint(
            checkpoint,
            "LSTM-4",
        )
    )

    print(
        "Configuration:",
        config,
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    (
        X_val,
        y_val,
        val_offsets,
        _,
        _,
    ) = lf.load_split(
        "validation"
    )

    if X_val.shape[1] != 80:

        raise RuntimeError(
            f"Expected F0 = 80 features, "
            f"got {X_val.shape[1]}"
        )

    dataset = lf.SepsisSequenceDataset(
        X_val,
        y_val,
        val_offsets,
    )

    loader = lf.SequenceDataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = lf.LSTMModel(
        input_size=80,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        state_dict = checkpoint.get(
            "state_dict"
        )

    if state_dict is None:

        raise RuntimeError(
            "LSTM4 checkpoint has no model state."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Checkpoint loaded."
    )

    # --------------------------------------------------------
    # INFERENCE ONLY
    # --------------------------------------------------------

    with torch.no_grad():

        probabilities, labels = (
            lf.predict_dataset(
                model,
                loader,
                DEVICE,
            )
        )

    threshold = checkpoint.get(
        "validation_threshold"
    )

    if threshold is None:

        threshold, _ = (
            lf.find_best_threshold(
                probabilities,
                labels,
            )
        )

    save_predictions(
        output_path,
        probabilities,
        labels,
        threshold,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Samples: {len(labels):,}"
    )

    print(
        f"Positive: {np.sum(labels == 1):,}"
    )


# ============================================================
# TRANSFORMER F0
# ============================================================

def extract_transformer():

    print()
    print("=" * 70)
    print("TR1-F0 — VALIDATION INFERENCE")
    print("=" * 70)

    import transformer_family as tf

    checkpoint_path = (
        CHECKPOINT_ROOT
        / "transformer_family"
        / "TR-1.pt"
    )

    output_path = (
        OUTPUT_ROOT
        / "TR1-F0"
        / "validation_predictions.npz"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Missing checkpoint:\n{checkpoint_path}"
        )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    config = checkpoint.get(
        "config"
    )

    if config is None:

        config = tf.EXPERIMENTS[
            "TR-1"
        ]

    print(
        "Configuration:",
        config,
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    val_dataset = tf.SepsisSequenceDataset(
        "validation"
    )

    val_loader = tf.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tf.collate_sequences,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    input_dim = int(
        val_dataset.X.shape[1]
    )

    if input_dim != 80:

        raise RuntimeError(
            f"Expected F0 = 80 features, "
            f"got {input_dim}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = tf.CausalTransformer(
        input_dim=input_dim,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config[
            "dim_feedforward"
        ],
        dropout=config["dropout"],
        max_len=512,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Checkpoint loaded."
    )

    # --------------------------------------------------------
    # INFERENCE ONLY
    # --------------------------------------------------------

    with torch.no_grad():

        (
            labels,
            probabilities,
            _,
        ) = tf.predict_dataset(
            model,
            val_loader,
        )

    threshold = checkpoint.get(
        "validation_threshold"
    )

    if threshold is None:

        threshold, _ = (
            tf.calculate_threshold(
                labels,
                probabilities,
            )
        )

    save_predictions(
        output_path,
        probabilities,
        labels,
        threshold,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Samples: {len(labels):,}"
    )

    print(
        f"Positive: {np.sum(labels == 1):,}"
    )


# ============================================================
# VERIFY
# ============================================================

def verify():

    print()
    print("=" * 70)
    print("VERIFYING F0 PREDICTION FILES")
    print("=" * 70)

    paths = {

        "GRU5-F0":
            OUTPUT_ROOT
            / "GRU5-F0"
            / "validation_predictions.npz",

        "LSTM4-F0":
            OUTPUT_ROOT
            / "LSTM4-F0"
            / "validation_predictions.npz",

        "TR1-F0":
            OUTPUT_ROOT
            / "TR1-F0"
            / "validation_predictions.npz",
    }

    reference_labels = None

    for name, path in paths.items():

        if not path.exists():

            raise RuntimeError(
                f"{name} was not created:\n{path}"
            )

        data = np.load(path)

        print()
        print(name)
        print(
            "  keys:",
            list(data.keys()),
        )

        probabilities = np.asarray(
            data["probabilities"]
        )

        labels = np.asarray(
            data["labels"]
        )

        print(
            "  probabilities:",
            probabilities.shape,
        )

        print(
            "  labels:",
            labels.shape,
        )

        if len(probabilities) != len(labels):

            raise RuntimeError(
                f"{name}: length mismatch."
            )

        if reference_labels is None:

            reference_labels = labels.copy()

        elif not np.array_equal(
            reference_labels,
            labels,
        ):

            raise RuntimeError(
                f"{name}: validation labels "
                f"do not match the other F0 models."
            )

    print()
    print(
        "ALL THREE F0 VALIDATION FILES VERIFIED."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "F0 VALIDATION PREDICTION RECOVERY"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This script performs inference only."
    )
    print(
        "No training will occur."
    )
    print(
        "No test data will be loaded."
    )

    extract_gru()
    extract_lstm()
    extract_transformer()

    verify()

    print()
    print("=" * 70)
    print("F0 RECOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()