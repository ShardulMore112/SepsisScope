"""
Stage 3.5 — Causal Transformer Family

Purpose
-------
Systematically evaluate a controlled family of strictly causal Transformer
models for timestep-level sepsis prediction.

Frozen methodology
------------------
- Feature set: F0
- Patient-level train/validation/test split already frozen
- Variable-length patient sequences
- Dynamic padding
- Padding masks
- Strictly causal self-attention
- Weighted BCE using training-only positive-class weight
- AdamW optimizer
- Gradient clipping
- Best checkpoint selected by validation AUPRC
- Classification threshold selected on validation by maximum F1
- Test evaluated only after model selection
- Resume-safe checkpoints and result files
- Raw and processed datasets are never modified

Transformer family
------------------
TR-1: d_model=64,  nhead=4, layers=2, FFN=128, dropout=0.1, LR=1e-3
TR-2: d_model=128, nhead=4, layers=2, FFN=256, dropout=0.1, LR=1e-3
TR-3: d_model=128, nhead=4, layers=3, FFN=256, dropout=0.1, LR=1e-3
TR-4: d_model=128, nhead=8, layers=3, FFN=256, dropout=0.2, LR=5e-4
TR-5: d_model=256, nhead=8, layers=4, FFN=512, dropout=0.2, LR=5e-4

Selection
---------
Primary metric: Validation AUPRC
Secondary metrics: Validation AUROC, F1

The final Transformer-family winner is selected ONLY using validation AUPRC.
"""

from __future__ import annotations

import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "transformer_family"
RESULT_ROOT = PROJECT_ROOT / "results" / "model_result" / "transformer_family"

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

FEATURE_SET = "F0"

BATCH_SIZE = 128
NUM_EPOCHS = 15

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training-only positive-class weight.
# This is calculated from the frozen training split.
POS_WEIGHT = 54.7340

# Family definitions
EXPERIMENTS = {
    "TR-1": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "lr": 1e-3,
    },
    "TR-2": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "lr": 1e-3,
    },
    "TR-3": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "lr": 1e-3,
    },
    "TR-4": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.2,
        "lr": 5e-4,
    },
    "TR-5": {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "lr": 5e-4,
    },
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior.
    # This may slightly reduce GPU performance but improves reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SepsisSequenceDataset(Dataset):
    """
    Variable-length patient sequence dataset backed by the processed
    F0 memmap arrays.
    """

    def __init__(self, split: str):
        self.split = split

        split_dir = PROCESSED_ROOT / FEATURE_SET / split

        x_path = split_dir / "X.npy"
        y_path = split_dir / "labels.npy"
        offsets_path = split_dir / "patient_offsets.npy"
        keys_path = split_dir / "patient_keys.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"Missing: {x_path}")

        if not y_path.exists():
            raise FileNotFoundError(f"Missing: {y_path}")

        if not offsets_path.exists():
            raise FileNotFoundError(f"Missing: {offsets_path}")

        if not keys_path.exists():
            raise FileNotFoundError(f"Missing: {keys_path}")

        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.offsets = np.load(offsets_path)
        self.patient_keys = np.load(keys_path, allow_pickle=True)

        self.n_patients = len(self.offsets) - 1

        if self.n_patients != len(self.patient_keys):
            raise RuntimeError(
                f"{split}: patient count mismatch: "
                f"offsets={self.n_patients}, keys={len(self.patient_keys)}"
            )

    def __len__(self):
        return self.n_patients

    def __getitem__(self, idx: int):
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])

        # Copy from memmap so PyTorch does not receive a non-writable NumPy view.
        x = np.array(self.X[start:end], dtype=np.float32, copy=True)
        y = np.array(self.y[start:end], dtype=np.float32, copy=True)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "length": end - start,
            "patient_key": str(self.patient_keys[idx]),
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_sequences(batch: List[Dict]):
    """
    Dynamic padding.

    Returns
    -------
    x:
        [B, T, F]

    y:
        [B, T]

    padding_mask:
        [B, T]
        True  = padding
        False = valid timestep

    lengths:
        [B]
    """

    batch_size = len(batch)

    max_len = max(item["length"] for item in batch)

    n_features = batch[0]["x"].shape[1]

    x = torch.zeros(
        batch_size,
        max_len,
        n_features,
        dtype=torch.float32,
    )

    y = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.float32,
    )

    padding_mask = torch.ones(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    lengths = []

    patient_keys = []

    for i, item in enumerate(batch):
        length = item["length"]

        x[i, :length] = item["x"]
        y[i, :length] = item["y"]

        padding_mask[i, :length] = False

        lengths.append(length)
        patient_keys.append(item["patient_key"])

    lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        "x": x,
        "y": y,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "patient_keys": patient_keys,
    }


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloader(
    split: str,
    shuffle: bool,
) -> DataLoader:

    dataset = SepsisSequenceDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sequences,
    )

    return loader


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Shape:
        [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model,
            dtype=torch.float32,
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(
                position * div_term[:-1]
            )

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        return self.dropout(x)


# ============================================================
# CAUSAL TRANSFORMER MODEL
# ============================================================

class CausalTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int = 512,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by "
                f"nhead={nhead}"
            )

        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(
            d_model,
            1,
        )

    @staticmethod
    def causal_mask(
        seq_len: int,
        device: torch.device,
    ):

        # Upper triangular positions are future positions.
        #
        # Example T=4:
        #
        #  0 -inf -inf -inf
        #  0   0  -inf -inf
        #  0   0    0  -inf
        #  0   0    0    0

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
            ),
            diagonal=1,
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf"),
        )

        mask = mask.masked_fill(
            mask == 0,
            0.0,
        )

        return mask

    def forward(
        self,
        x,
        padding_mask,
    ):

        # x: [B, T, F]

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        seq_len = x.size(1)

        causal_mask = self.causal_mask(
            seq_len,
            x.device,
        )

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.output_layer(x).squeeze(-1)

        # logits: [B, T]

        return logits


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def safe_auroc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def safe_auprc(y_true, y_prob):
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def calculate_threshold(
    y_true,
    y_prob,
) -> Tuple[float, float]:

    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            y_prob >= threshold
        ).astype(np.int32)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:

            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(
    y_true,
    y_prob,
    threshold,
):

    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
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

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    return {
        "auroc": safe_auroc(
            y_true,
            y_prob,
        ),
        "auprc": safe_auprc(
            y_true,
            y_prob,
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    loader,
):

    model.eval()

    all_probs = []
    all_labels = []

    total_loss = 0.0
    total_valid = 0

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            POS_WEIGHT,
            device=DEVICE,
        ),
        reduction="sum",
    )

    for batch in loader:

        x = batch["x"].to(
            DEVICE,
            non_blocking=True,
        )

        y = batch["y"].to(
            DEVICE,
            non_blocking=True,
        )

        padding_mask = batch["padding_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            x,
            padding_mask,
        )

        valid_mask = ~padding_mask

        valid_logits = logits[valid_mask]
        valid_y = y[valid_mask]

        loss = criterion(
            valid_logits,
            valid_y,
        )

        total_loss += loss.item()
        total_valid += valid_y.numel()

        probs = torch.sigmoid(
            valid_logits
        )

        all_probs.append(
            probs.detach().cpu().numpy()
        )

        all_labels.append(
            valid_y.detach().cpu().numpy()
        )

    y_prob = np.concatenate(
        all_probs
    )

    y_true = np.concatenate(
        all_labels
    )

    mean_loss = (
        total_loss / total_valid
        if total_valid > 0
        else float("nan")
    )

    return (
        y_true,
        y_prob,
        mean_loss,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    total_valid = 0

    for batch in loader:

        x = batch["x"].to(
            DEVICE,
            non_blocking=True,
        )

        y = batch["y"].to(
            DEVICE,
            non_blocking=True,
        )

        padding_mask = batch["padding_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            x,
            padding_mask,
        )

        valid_mask = ~padding_mask

        valid_logits = logits[valid_mask]
        valid_y = y[valid_mask]

        loss = criterion(
            valid_logits,
            valid_y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        total_valid += (
            valid_y.numel()
        )

    mean_loss = (
        total_loss / total_valid
        if total_valid > 0
        else float("nan")
    )

    return mean_loss


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_paths(exp_id: str):

    checkpoint_path = (
        CHECKPOINT_ROOT
        / f"{exp_id}.pt"
    )

    prediction_path = (
        RESULT_ROOT
        / f"{exp_id}_test_predictions.npz"
    )

    result_path = (
        RESULT_ROOT
        / f"{exp_id}_result.json"
    )

    completion_path = (
        RESULT_ROOT
        / f"{exp_id}_COMPLETE.json"
    )

    return (
        checkpoint_path,
        prediction_path,
        result_path,
        completion_path,
    )


def experiment_complete(exp_id: str):

    (
        checkpoint_path,
        prediction_path,
        result_path,
        completion_path,
    ) = get_paths(exp_id)

    return (
        checkpoint_path.exists()
        and prediction_path.exists()
        and result_path.exists()
        and completion_path.exists()
    )


# ============================================================
# RECONSTRUCT COMPLETED EXPERIMENT
# ============================================================

def reconstruct_completed_result(
    exp_id: str,
):

    (
        checkpoint_path,
        prediction_path,
        result_path,
        completion_path,
    ) = get_paths(exp_id)

    if not result_path.exists():
        raise RuntimeError(
            f"{exp_id} marked complete but result file "
            f"is missing."
        )

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as f:

        result = json.load(f)

    return result


# ============================================================
# RUN SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    exp_id: str,
    config: Dict,
    train_loader,
    val_loader,
    test_loader,
    input_dim: int,
):

    print()
    print("=" * 70)
    print(f"STARTING {exp_id}")
    print("=" * 70)

    print(f"Configuration: {config}")

    (
        checkpoint_path,
        prediction_path,
        result_path,
        completion_path,
    ) = get_paths(exp_id)

    # --------------------------------------------------------
    # Resume-safe behavior
    # --------------------------------------------------------

    if experiment_complete(exp_id):

        print(
            f"{exp_id}: COMPLETE — SKIP"
        )

        return reconstruct_completed_result(
            exp_id
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = CausalTransformer(
        input_dim=input_dim,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=512,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            POS_WEIGHT,
            device=DEVICE,
        )
    )

    best_val_auprc = -float("inf")
    best_epoch = -1

    training_history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )

        (
            val_y,
            val_prob,
            val_loss,
        ) = predict_dataset(
            model,
            val_loader,
        )

        val_auprc = safe_auprc(
            val_y,
            val_prob,
        )

        val_auroc = safe_auroc(
            val_y,
            val_prob,
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auprc": float(val_auprc),
            "val_auroc": float(val_auroc),
            "epoch_time_seconds": float(
                epoch_time
            ),
        }

        training_history.append(
            epoch_record
        )

        print(
            f"{exp_id} | "
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_loss:.6f} | "
            f"Val AUPRC {val_auprc:.6f} | "
            f"Val AUROC {val_auroc:.6f}"
        )

        # ----------------------------------------------------
        # Best checkpoint = highest validation AUPRC
        # ----------------------------------------------------

        if val_auprc > best_val_auprc:

            best_val_auprc = float(
                val_auprc
            )

            best_epoch = epoch

            torch.save(
                {
                    "experiment_id": exp_id,
                    "feature_set": FEATURE_SET,
                    "config": config,
                    "input_dim": input_dim,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auprc": best_val_auprc,
                    "random_seed": RANDOM_SEED,
                },
                checkpoint_path,
            )

            print(
                f"  -> New best checkpoint saved "
                f"(Val AUPRC={best_val_auprc:.6f})"
            )

    # --------------------------------------------------------
    # Load best validation checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Validation prediction using best model
    # --------------------------------------------------------

    (
        val_y,
        val_prob,
        val_loss,
    ) = predict_dataset(
        model,
        val_loader,
    )

    val_threshold, val_f1 = calculate_threshold(
        val_y,
        val_prob,
    )

    val_metrics = calculate_metrics(
        val_y,
        val_prob,
        val_threshold,
    )

    # --------------------------------------------------------
    # Frozen test evaluation
    # --------------------------------------------------------

    (
        test_y,
        test_prob,
        test_loss,
    ) = predict_dataset(
        model,
        test_loader,
    )

    test_metrics = calculate_metrics(
        test_y,
        test_prob,
        val_threshold,
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    np.savez_compressed(
        prediction_path,
        y_test=test_y.astype(
            np.float32
        ),
        test_probs=test_prob.astype(
            np.float32
        ),
        threshold=np.array(
            val_threshold,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    result = {
        "experiment_id": exp_id,
        "family": "causal_transformer",
        "feature_set": FEATURE_SET,

        "config": config,

        "input_dim": int(input_dim),

        "batch_size": BATCH_SIZE,
        "epochs_requested": NUM_EPOCHS,
        "best_epoch": int(best_epoch),

        "random_seed": RANDOM_SEED,

        "pos_weight": float(POS_WEIGHT),

        "selection_metric": "validation_auprc",

        "threshold_selection": (
            "validation_max_f1"
        ),

        "best_validation_auprc": float(
            best_val_auprc
        ),

        "validation": val_metrics,

        "test": test_metrics,

        "test_threshold_source": (
            "validation"
        ),

        "test_used_for_selection": False,

        "training_time_seconds": float(
            elapsed
        ),

        "checkpoint": str(
            checkpoint_path
        ),

        "test_predictions": str(
            prediction_path
        ),

        "training_history": training_history,
    }

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

    # --------------------------------------------------------
    # Completion marker
    # --------------------------------------------------------

    completion_record = {
        "experiment_id": exp_id,
        "status": "COMPLETE",
        "best_epoch": int(best_epoch),
        "validation_auprc": float(
            val_metrics["auprc"]
        ),
        "validation_auroc": float(
            val_metrics["auroc"]
        ),
        "validation_f1": float(
            val_metrics["f1"]
        ),
        "threshold": float(
            val_threshold
        ),
        "test_auprc": float(
            test_metrics["auprc"]
        ),
        "test_auroc": float(
            test_metrics["auroc"]
        ),
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

    print()
    print(
        f"{exp_id} COMPLETE"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Validation AUPRC: "
        f"{val_metrics['auprc']:.6f}"
    )

    print(
        f"Validation AUROC: "
        f"{val_metrics['auroc']:.6f}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['f1']:.6f}"
    )

    print(
        f"Frozen threshold: "
        f"{val_threshold:.4f}"
    )

    print(
        f"Test AUPRC: "
        f"{test_metrics['auprc']:.6f}"
    )

    print(
        f"Test AUROC: "
        f"{test_metrics['auroc']:.6f}"
    )

    return result


# ============================================================
# FAMILY SUMMARY
# ============================================================

def create_family_summary(
    results: Dict[str, Dict]
):

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["validation"][
            "auprc"
        ],
        reverse=True,
    )

    summary = {
        "family": "causal_transformer",
        "feature_set": FEATURE_SET,

        "selection_metric": (
            "validation_auprc"
        ),

        "experiments": results,

        "validation_ranking": [
            {
                "rank": rank,
                "experiment_id": exp_id,
                "auprc": result[
                    "validation"
                ]["auprc"],
                "auroc": result[
                    "validation"
                ]["auroc"],
                "f1": result[
                    "validation"
                ]["f1"],
                "sensitivity": result[
                    "validation"
                ]["sensitivity"],
                "specificity": result[
                    "validation"
                ]["specificity"],
            }
            for rank, (
                exp_id,
                result,
            ) in enumerate(
                ranking,
                start=1,
            )
        ],

        "selected_model": ranking[0][0],

        "selected_model_validation_auprc":
            ranking[0][1]["validation"]["auprc"],

        "selected_model_validation_auroc":
            ranking[0][1]["validation"]["auroc"],

        "selected_model_validation_f1":
            ranking[0][1]["validation"]["f1"],

        "selected_model_threshold":
            ranking[0][1]["validation"]["threshold"],

        "selected_model_test":
            ranking[0][1]["test"],
    }

    summary_path = (
        RESULT_ROOT
        / "transformer_family_summary.json"
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

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("STAGE 3.5 — CAUSAL TRANSFORMER FAMILY")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Feature set: {FEATURE_SET}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Training-only pos_weight: "
        f"{POS_WEIGHT:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("Loading datasets...")

    train_loader = create_dataloader(
        "train",
        shuffle=True,
    )

    val_loader = create_dataloader(
        "validation",
        shuffle=False,
    )

    test_loader = create_dataloader(
        "test",
        shuffle=False,
    )

    # Infer feature dimensionality
    sample_dataset = SepsisSequenceDataset(
        "train"
    )

    input_dim = int(
        sample_dataset.X.shape[1]
    )

    print(
        f"Input features: {input_dim}"
    )

    print(
        f"Train patients: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation patients: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test patients: "
        f"{len(test_loader.dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Run family
    # --------------------------------------------------------

    results = {}

    for exp_id, config in EXPERIMENTS.items():

        set_seed(
            RANDOM_SEED
        )

        if experiment_complete(
            exp_id
        ):

            print(
                f"{exp_id}: COMPLETE — SKIP"
            )

            results[exp_id] = (
                reconstruct_completed_result(
                    exp_id
                )
            )

            continue

        results[exp_id] = run_experiment(
            exp_id=exp_id,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_dim=input_dim,
        )

    # --------------------------------------------------------
    # Family summary
    # --------------------------------------------------------

    summary = create_family_summary(
        results
    )

    ranking = summary[
        "validation_ranking"
    ]

    print()
    print("=" * 70)
    print("TRANSFORMER FAMILY — VALIDATION RANKING")
    print("=" * 70)

    for row in ranking:

        print(
            f"{row['rank']}. "
            f"{row['experiment_id']} | "
            f"AUPRC {row['auprc']:.6f} | "
            f"AUROC {row['auroc']:.6f} | "
            f"F1 {row['f1']:.6f} | "
            f"Sensitivity {row['sensitivity']:.4f} | "
            f"Specificity {row['specificity']:.4f}"
        )

    print()
    print("=" * 70)
    print("TRANSFORMER FAMILY — TEST RESULTS")
    print("=" * 70)

    test_ranking = sorted(
        results.items(),
        key=lambda item: item[1]["test"][
            "auprc"
        ],
        reverse=True,
    )

    for rank, (
        exp_id,
        result,
    ) in enumerate(
        test_ranking,
        start=1,
    ):

        test = result["test"]

        print(
            f"{rank}. "
            f"{exp_id} | "
            f"AUPRC {test['auprc']:.6f} | "
            f"AUROC {test['auroc']:.6f} | "
            f"F1 {test['f1']:.6f} | "
            f"Sensitivity {test['sensitivity']:.4f} | "
            f"Specificity {test['specificity']:.4f}"
        )

    selected_id = summary[
        "selected_model"
    ]

    selected_result = results[
        selected_id
    ]

    print()
    print("=" * 70)
    print("VALIDATION-SELECTED TRANSFORMER")
    print("=" * 70)

    print(
        f"Selected model: {selected_id}"
    )

    print(
        f"Validation AUPRC: "
        f"{selected_result['validation']['auprc']:.6f}"
    )

    print(
        f"Validation AUROC: "
        f"{selected_result['validation']['auroc']:.6f}"
    )

    print(
        f"Validation F1: "
        f"{selected_result['validation']['f1']:.6f}"
    )

    print(
        f"Frozen threshold: "
        f"{selected_result['validation']['threshold']:.4f}"
    )

    print()
    print("=" * 70)
    print("TRANSFORMER FAMILY COMPLETE")
    print("=" * 70)

    print(
        f"Completed: "
        f"{len(results)}/{len(EXPERIMENTS)}"
    )

    print(
        f"Validation winner: "
        f"{selected_id}"
    )

    print(
        "Summary:"
    )

    print(
        RESULT_ROOT
        / "transformer_family_summary.json"
    )


if __name__ == "__main__":
    main()