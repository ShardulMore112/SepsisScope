"""
Stage 3.2.4 — Simple LSTM Causal Early-Warning Baseline

Same experimental setup as Stage 3.2.3 GRU.
Only the recurrent architecture changes.

Feature set:
    F0 = 80 features

Task:
    timestep-level causal sepsis prediction

Selection:
    best validation AUPRC -> checkpoint
    validation F1 -> threshold
    frozen threshold -> test
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "baseline_result"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints"
)

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_SET = "F0"

SEED = 42

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

BATCH_SIZE = 128

EPOCHS = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

USE_POS_WEIGHT = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split: str):

    root = PROCESSED_ROOT / FEATURE_SET / split

    X = np.load(
        root / "X.npy",
        mmap_mode="r",
    )

    y = np.load(
        root / "labels.npy",
        mmap_mode="r",
    )

    offsets = np.load(
        root / "patient_offsets.npy"
    )

    patient_keys = np.load(
        root / "patient_keys.npy"
    )

    hour_index = np.load(
        root / "hour_index.npy"
    )

    return (
        X,
        y,
        offsets,
        patient_keys,
        hour_index,
    )


# ============================================================
# DATASET
# ============================================================

class SepsisSequenceDataset:

    def __init__(
        self,
        X,
        y,
        offsets,
    ):

        self.X = X
        self.y = y
        self.offsets = offsets

    def __len__(self):

        return len(self.offsets) - 1

    def get_patient(self, index):

        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])

        X_patient = np.array(
            self.X[start:end],
            dtype=np.float32,
            copy=True,
        )

        y_patient = np.array(
            self.y[start:end],
            dtype=np.float32,
            copy=True,
        )

        return (
            torch.from_numpy(X_patient),
            torch.from_numpy(y_patient),
        )


# ============================================================
# COLLATE
# ============================================================

def collate_sequences(batch):

    batch_size = len(batch)

    lengths = [
        item[0].shape[0]
        for item in batch
    ]

    max_length = max(lengths)

    n_features = batch[0][0].shape[1]

    X = torch.zeros(
        batch_size,
        max_length,
        n_features,
        dtype=torch.float32,
    )

    y = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.float32,
    )

    padding_mask = torch.ones(
        batch_size,
        max_length,
        dtype=torch.bool,
    )

    for i, (X_patient, y_patient) in enumerate(batch):

        length = X_patient.shape[0]

        X[i, :length] = X_patient
        y[i, :length] = y_patient

        padding_mask[i, :length] = False

    lengths = torch.tensor(
        lengths,
        dtype=torch.long,
    )

    return X, y, padding_mask, lengths


# ============================================================
# DATALOADER
# ============================================================

class SequenceDataLoader:

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        seed,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):

        indices = np.arange(
            len(self.dataset)
        )

        if self.shuffle:

            rng = np.random.default_rng(
                self.seed
            )

            rng.shuffle(indices)

        for start in range(
            0,
            len(indices),
            self.batch_size,
        ):

            batch_indices = indices[
                start:start + self.batch_size
            ]

            batch = [
                self.dataset.get_patient(int(i))
                for i in batch_indices
            ]

            yield collate_sequences(batch)


# ============================================================
# SIMPLE LSTM
# ============================================================

class SimpleLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        X,
        padding_mask=None,
    ):

        # Standard LSTM recurrence is causal:
        # output at t depends only on x_1 ... x_t.

        hidden, _ = self.lstm(X)

        logits = self.output(
            hidden
        ).squeeze(-1)

        return logits


# ============================================================
# VALID TIMESTEPS
# ============================================================

def flatten_valid_timesteps(
    logits,
    y,
    padding_mask,
):

    valid = ~padding_mask

    return (
        logits[valid],
        y[valid],
    )


# ============================================================
# THRESHOLD
# ============================================================

def find_best_f1_threshold(
    probabilities,
    y_true,
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
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    probabilities,
    y_true,
    threshold,
):

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
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
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
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    loader,
    device,
):

    model.eval()

    all_probabilities = []
    all_labels = []

    for (
        X,
        y,
        padding_mask,
        lengths,
    ) in loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        padding_mask = padding_mask.to(
            device,
            non_blocking=True,
        )

        logits = model(
            X,
            padding_mask,
        )

        valid_logits, valid_y = (
            flatten_valid_timesteps(
                logits,
                y,
                padding_mask,
            )
        )

        probabilities = torch.sigmoid(
            valid_logits
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            valid_y.cpu().numpy()
        )

    return (
        np.concatenate(all_probabilities),
        np.concatenate(all_labels),
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    total_timesteps = 0

    for (
        X,
        y,
        padding_mask,
        lengths,
    ) in loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        padding_mask = padding_mask.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            X,
            padding_mask,
        )

        valid_logits, valid_y = (
            flatten_valid_timesteps(
                logits,
                y,
                padding_mask,
            )
        )

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

        count = valid_y.numel()

        total_loss += (
            loss.item() * count
        )

        total_timesteps += count

    return (
        total_loss / total_timesteps
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 70)
    print(
        "STAGE 3.2.4 — SIMPLE LSTM "
        "CAUSAL EARLY-WARNING BASELINE"
    )
    print("=" * 70)

    print(f"\nDevice: {device}")
    print(f"Feature set: {FEATURE_SET}")

    print("\nConfiguration:")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "\nLoading frozen Step-2 data..."
    )

    (
        X_train,
        y_train_raw,
        train_offsets,
        train_keys,
        train_hours,
    ) = load_split("train")

    (
        X_validation,
        y_validation_raw,
        validation_offsets,
        validation_keys,
        validation_hours,
    ) = load_split("validation")

    (
        X_test,
        y_test_raw,
        test_offsets,
        test_keys,
        test_hours,
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
    # DATASETS
    # --------------------------------------------------------

    train_dataset = SepsisSequenceDataset(
        X_train,
        y_train_raw,
        train_offsets,
    )

    validation_dataset = SepsisSequenceDataset(
        X_validation,
        y_validation_raw,
        validation_offsets,
    )

    test_dataset = SepsisSequenceDataset(
        X_test,
        y_test_raw,
        test_offsets,
    )

    train_loader = SequenceDataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    validation_loader = SequenceDataLoader(
        validation_dataset,
        BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    test_loader = SequenceDataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    input_size = X_train.shape[1]

    print(
        f"\nInput features: {input_size}"
    )

    if input_size != 80:

        raise ValueError(
            "F0 should contain exactly 80 features."
        )

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    train_labels = np.asarray(
        y_train_raw
    )

    positive_timesteps = int(
        train_labels.sum()
    )

    negative_timesteps = int(
        train_labels.size
        - positive_timesteps
    )

    pos_weight_value = (
        negative_timesteps
        / positive_timesteps
    )

    print(
        "\nTraining timestep imbalance:"
    )

    print(
        f"  Positive: "
        f"{positive_timesteps:,}"
    )

    print(
        f"  Negative: "
        f"{negative_timesteps:,}"
    )

    print(
        f"  pos_weight: "
        f"{pos_weight_value:.4f}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SimpleLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    if USE_POS_WEIGHT:

        pos_weight = torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

    else:

        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTraining LSTM..."
    )

    best_val_auprc = -np.inf
    best_epoch = 0
    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_probabilities,
            val_labels,
        ) = predict_dataset(
            model,
            validation_loader,
            device,
        )

        val_auroc = roc_auc_score(
            val_labels,
            val_probabilities,
        )

        val_auprc = average_precision_score(
            val_labels,
            val_probabilities,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.5f} | "
            f"Val AUROC {val_auroc:.5f} | "
            f"Val AUPRC {val_auprc:.5f}"
        )

        if val_auprc > best_val_auprc:

            best_val_auprc = float(
                val_auprc
            )

            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:

        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_ROOT
        / "baseline_lstm_f0.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "feature_set":
                FEATURE_SET,

            "seed":
                SEED,

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "best_epoch":
                best_epoch,

            "best_validation_auprc":
                best_val_auprc,
        },
        checkpoint_path,
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation AUPRC: "
        f"{best_val_auprc:.6f}"
    )

    print(
        f"Checkpoint saved:\n"
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nEvaluating validation set..."
    )

    (
        validation_probabilities,
        validation_labels,
    ) = predict_dataset(
        model,
        validation_loader,
        device,
    )

    validation_threshold, validation_f1 = (
        find_best_f1_threshold(
            validation_probabilities,
            validation_labels,
        )
    )

    validation_metrics = calculate_metrics(
        validation_probabilities,
        validation_labels,
        validation_threshold,
    )

    print(
        f"\nValidation-selected threshold: "
        f"{validation_threshold:.4f}"
    )

    print(
        f"Validation F1: "
        f"{validation_f1:.6f}"
    )

    print(
        "\nValidation metrics:"
    )

    for key, value in validation_metrics.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nEvaluating test set..."
    )

    (
        test_probabilities,
        test_labels,
    ) = predict_dataset(
        model,
        test_loader,
        device,
    )

    test_metrics = calculate_metrics(
        test_probabilities,
        test_labels,
        validation_threshold,
    )

    print(
        "\nTest metrics:"
    )

    for key, value in test_metrics.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_path = (
        RESULTS_ROOT
        / "baseline_lstm_f0_test_predictions.npz"
    )

    np.savez_compressed(
        prediction_path,
        probabilities=test_probabilities,
        labels=test_labels,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {

        "stage": "3.2.4",

        "model": "Simple LSTM",

        "task":
            "causal timestep-level sepsis prediction",

        "feature_set":
            FEATURE_SET,

        "seed":
            SEED,

        "input_features":
            int(input_size),

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "positive_weight":
            float(pos_weight_value),

        "best_epoch":
            int(best_epoch),

        "best_validation_AUPRC":
            float(best_val_auprc),

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "checkpoint":
            str(checkpoint_path),

        "test_predictions":
            str(prediction_path),

        "causal_constraint":
            "LSTM hidden state at timestep t "
            "uses only timesteps <= t",
    }

    summary_path = (
        RESULTS_ROOT
        / "baseline_lstm_summary.json"
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

    print(
        f"\nSummary saved:\n"
        f"{summary_path}"
    )

    print("\n" + "=" * 70)
    print(
        "STAGE 3.2.4 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()