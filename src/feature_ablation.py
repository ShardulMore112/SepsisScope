"""
X-FedSepsis
STAGE 3.6 — CAUSAL FEATURE ABLATION

Purpose
-------
Controlled feature-ablation study using the already-selected architectures:

    GRU-5       + F1/F2/F3
    LSTM-4      + F1/F2/F3
    TR-1        + F1/F2/F3

F0 is NOT retrained because the F0 results already exist from Stage 3.3-3.5.

Research question
-----------------
How does progressively adding causal temporal/clinical information
change the behavior of different temporal architectures?

Feature hierarchy
-----------------
F0 -> F1 -> F2 -> F3

F0 = Raw + Missingness
F1 = F0 + Causal temporal deltas
F2 = F1 + Causal rolling statistics
F3 = F2 + Clinical ratios/scores

Experimental rules
------------------
1. Patient-level frozen split.
2. No test-based model selection.
3. Best checkpoint selected using validation AUPRC.
4. Threshold selected using validation F1.
5. Test evaluated only after checkpoint and threshold are frozen.
6. Training-only pos_weight.
7. Causal Transformer attention.
8. Padding is ignored.
9. Processed data are READ ONLY.
10. Resume-safe experiment markers.

Experiments
-----------
GRU-5:
    GRU hidden=256, layers=2, dropout=0.2, lr=5e-4
    F1, F2, F3

LSTM-4:
    LSTM hidden=128, layers=2, dropout=0.2, lr=5e-4
    F1, F2, F3

TR-1:
    Transformer d_model=64, heads=4, layers=2,
    FFN=128, dropout=0.1, lr=1e-3
    F1, F2, F3

Usage
-----
From:
    D:\\CAPSTONE\\SEPSIS\\src

Run:
    python .\\feature_ablation.py
"""

from __future__ import annotations

import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\CAPSTONE\SEPSIS")

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "feature_ablation"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_result"
    / "feature_ablation"
)

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42

BATCH_SIZE = 128
EPOCHS = 5

POS_WEIGHT = 54.7340

GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_DIMS = {
    "F0": 80,
    "F1": 152,
    "F2": 296,
    "F3": 306,
}


# Only these 9 experiments are run.
EXPERIMENTS = [
    # GRU-5
    {
        "id": "GRU5-F1",
        "family": "GRU",
        "feature_set": "F1",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
    },
    {
        "id": "GRU5-F2",
        "family": "GRU",
        "feature_set": "F2",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
    },
    {
        "id": "GRU5-F3",
        "family": "GRU",
        "feature_set": "F3",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
    },

    # LSTM-4
    {
        "id": "LSTM4-F1",
        "family": "LSTM",
        "feature_set": "F1",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
    },
    {
        "id": "LSTM4-F2",
        "family": "LSTM",
        "feature_set": "F2",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
    },
    {
        "id": "LSTM4-F3",
        "family": "LSTM",
        "feature_set": "F3",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
    },

    # TR-1
    {
        "id": "TR1-F1",
        "family": "TRANSFORMER",
        "feature_set": "F1",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "lr": 1e-3,
    },
    {
        "id": "TR1-F2",
        "family": "TRANSFORMER",
        "feature_set": "F2",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "lr": 1e-3,
    },
    {
        "id": "TR1-F3",
        "family": "TRANSFORMER",
        "feature_set": "F3",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "lr": 1e-3,
    },
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SepsisTemporalDataset(Dataset):
    """
    Reads the already-created processed patient sequences.

    Expected structure:

        data/processed/Fx/train/X.npy
        data/processed/Fx/train/labels.npy
        data/processed/Fx/train/patient_offsets.npy

        data/processed/Fx/validation/...
        data/processed/Fx/test/...

    Each patient occupies one contiguous range in X/labels.
    """

    def __init__(
        self,
        feature_set: str,
        split: str,
    ):
        self.feature_set = feature_set
        self.split = split

        split_dir = (
            PROCESSED_ROOT
            / feature_set
            / split
        )

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Processed directory does not exist:\n{split_dir}"
            )

        self.X = np.load(
            split_dir / "X.npy",
            mmap_mode="r",
        )

        self.y = np.load(
            split_dir / "labels.npy",
            mmap_mode="r",
        )

        self.offsets = np.load(
            split_dir / "patient_offsets.npy",
        )

        if self.X.ndim != 2:
            raise ValueError(
                f"{feature_set}/{split}/X.npy must be 2D. "
                f"Found {self.X.shape}"
            )

        if self.X.shape[1] != FEATURE_DIMS[feature_set]:
            raise ValueError(
                f"{feature_set}/{split}: expected "
                f"{FEATURE_DIMS[feature_set]} features, "
                f"found {self.X.shape[1]}"
            )

        if len(self.offsets) < 2:
            raise ValueError(
                f"Invalid offsets for {feature_set}/{split}"
            )

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):

        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])

        x = np.asarray(
            self.X[start:end],
            dtype=np.float32,
        )

        y = np.asarray(
            self.y[start:end],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_batch(batch):

    xs, ys = zip(*batch)

    lengths = torch.tensor(
        [len(x) for x in xs],
        dtype=torch.long,
    )

    max_len = int(lengths.max())

    batch_size = len(xs)

    feature_dim = xs[0].shape[1]

    X = torch.zeros(
        batch_size,
        max_len,
        feature_dim,
        dtype=torch.float32,
    )

    Y = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.float32,
    )

    padding_mask = torch.ones(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    for i, (x, y) in enumerate(zip(xs, ys)):

        length = len(x)

        X[i, :length] = x
        Y[i, :length] = y

        padding_mask[i, :length] = False

    return X, Y, lengths, padding_mask


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(feature_set):

    train_ds = SepsisTemporalDataset(
        feature_set,
        "train",
    )

    val_ds = SepsisTemporalDataset(
        feature_set,
        "validation",
    )

    test_ds = SepsisTemporalDataset(
        feature_set,
        "test",
    )

    common = dict(
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **common,
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# MASK HELPERS
# ============================================================

def valid_mask_from_lengths(lengths, max_len):

    positions = torch.arange(
        max_len,
        device=lengths.device,
    ).unsqueeze(0)

    return positions < lengths.unsqueeze(1)


# ============================================================
# GRU
# ============================================================

class GRUModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden,
        layers,
        dropout,
    ):
        super().__init__()

        effective_dropout = (
            dropout if layers > 1 else 0.0
        )

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output = nn.Linear(
            hidden,
            1,
        )

    def forward(
        self,
        x,
        lengths,
    ):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.rnn(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )

        logits = self.output(out).squeeze(-1)

        return logits


# ============================================================
# LSTM
# ============================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden,
        layers,
        dropout,
    ):
        super().__init__()

        effective_dropout = (
            dropout if layers > 1 else 0.0
        )

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output = nn.Linear(
            hidden,
            1,
        )

    def forward(
        self,
        x,
        lengths,
    ):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.rnn(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )

        logits = self.output(out).squeeze(-1)

        return logits


# ============================================================
# SINUSOIDAL POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512,
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model,
        )

        position = torch.arange(
            0,
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

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


# ============================================================
# CAUSAL TRANSFORMER
# ============================================================

class CausalTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        self.position = PositionalEncoding(
            d_model=d_model,
            max_len=512,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output = nn.Linear(
            d_model,
            1,
        )

    def forward(
        self,
        x,
        padding_mask,
    ):

        x = self.input_projection(x)

        x = self.position(x)

        seq_len = x.size(1)

        # Strictly causal:
        # position t cannot attend to t+1, t+2, ...
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.output(x).squeeze(-1)

        return logits


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(config):

    feature_set = config["feature_set"]

    input_dim = FEATURE_DIMS[feature_set]

    family = config["family"]

    if family == "GRU":

        return GRUModel(
            input_dim=input_dim,
            hidden=config["hidden"],
            layers=config["layers"],
            dropout=config["dropout"],
        )

    if family == "LSTM":

        return LSTMModel(
            input_dim=input_dim,
            hidden=config["hidden"],
            layers=config["layers"],
            dropout=config["dropout"],
        )

    if family == "TRANSFORMER":

        return CausalTransformer(
            input_dim=input_dim,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
        )

    raise ValueError(
        f"Unknown family: {family}"
    )


# ============================================================
# LOSS
# ============================================================

def weighted_bce_loss(
    logits,
    targets,
    valid_mask,
):

    pos_weight = torch.tensor(
        POS_WEIGHT,
        device=logits.device,
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none",
    )

    losses = loss_fn(
        logits,
        targets,
    )

    losses = losses[valid_mask]

    return losses.mean()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
):

    model.train()

    total_loss = 0.0
    batches = 0

    for X, y, lengths, padding_mask in loader:

        X = X.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        lengths = lengths.to(
            DEVICE,
            non_blocking=True,
        )

        padding_mask = padding_mask.to(
            DEVICE,
            non_blocking=True,
        )

        valid_mask = ~padding_mask

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            X,
            lengths,
        ) if isinstance(
            model,
            (GRUModel, LSTMModel)
        ) else model(
            X,
            padding_mask,
        )

        # Packed RNN output can be shorter than the
        # padded input, so use the actual output length.
        current_len = logits.shape[1]

        valid_mask = valid_mask[:, :current_len]
        y = y[:, :current_len]

        loss = weighted_bce_loss(
            logits,
            y,
            valid_mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
):

    model.eval()

    all_probs = []
    all_targets = []

    total_loss = 0.0
    batches = 0

    for X, y, lengths, padding_mask in loader:

        X = X.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        lengths = lengths.to(
            DEVICE,
            non_blocking=True,
        )

        padding_mask = padding_mask.to(
            DEVICE,
            non_blocking=True,
        )

        valid_mask = ~padding_mask

        logits = model(
            X,
            lengths,
        ) if isinstance(
            model,
            (GRUModel, LSTMModel)
        ) else model(
            X,
            padding_mask,
        )

        current_len = logits.shape[1]

        valid_mask = valid_mask[:, :current_len]
        y = y[:, :current_len]

        loss = weighted_bce_loss(
            logits,
            y,
            valid_mask,
        )

        total_loss += loss.item()
        batches += 1

        probs = torch.sigmoid(
            logits
        )

        all_probs.append(
            probs[valid_mask]
            .detach()
            .cpu()
            .numpy()
        )

        all_targets.append(
            y[valid_mask]
            .detach()
            .cpu()
            .numpy()
        )

    probs = np.concatenate(
        all_probs
    )

    targets = np.concatenate(
        all_targets
    )

    return (
        probs,
        targets,
        total_loss / max(batches, 1),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    probs,
    threshold,
):

    preds = (
        probs >= threshold
    ).astype(np.int32)

    try:
        auroc = roc_auc_score(
            targets,
            probs,
        )
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(
            targets,
            probs,
        )
    except ValueError:
        auprc = float("nan")

    f1 = f1_score(
        targets,
        preds,
        zero_division=0,
    )

    precision = precision_score(
        targets,
        preds,
        zero_division=0,
    )

    sensitivity = recall_score(
        targets,
        preds,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        targets,
        preds,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Precision": float(precision),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# VALIDATION THRESHOLD
# ============================================================

def find_best_f1_threshold(
    targets,
    probs,
):

    best_threshold = 0.5
    best_f1 = -1.0

    # Same style of threshold search used by the
    # previous family experiments.
    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
    )

    for threshold in thresholds:

        preds = (
            probs >= threshold
        ).astype(np.int32)

        score = f1_score(
            targets,
            preds,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                round(threshold, 2)
            )

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(config):

    experiment_id = config["id"]
    feature_set = config["feature_set"]

    exp_dir = (
        RESULT_ROOT
        / experiment_id
    )

    ckpt_dir = (
        CHECKPOINT_ROOT
        / experiment_id
    )

    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        exp_dir
        / "result.json"
    )

    complete_file = (
        exp_dir
        / "COMPLETE"
    )

    checkpoint_file = (
        ckpt_dir
        / "best_model.pt"
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if (
        complete_file.exists()
        and result_file.exists()
    ):

        print(
            f"\n{experiment_id} already COMPLETE."
        )

        with open(
            result_file,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    print()
    print("=" * 70)
    print(f"STARTING {experiment_id}")
    print("=" * 70)

    print(
        f"Architecture : {config['family']}"
    )

    print(
        f"Feature set  : {feature_set}"
    )

    print(
        f"Input features: "
        f"{FEATURE_DIMS[feature_set]}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Configuration: {config}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        feature_set
    )

    print(
        f"Train patients: {len(train_ds)}"
    )

    print(
        f"Validation patients: {len(val_ds)}"
    )

    print(
        f"Test patients: {len(test_ds)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    set_seed(SEED)

    model = build_model(
        config
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_auprc = -np.inf
    best_epoch = None

    history = []

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        (
            val_probs,
            val_targets,
            val_loss,
        ) = predict(
            model,
            val_loader,
        )

        val_auprc = average_precision_score(
            val_targets,
            val_probs,
        )

        val_auroc = roc_auc_score(
            val_targets,
            val_probs,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_AUPRC": float(val_auprc),
                "val_AUROC": float(val_auroc),
            }
        )

        print(
            f"{experiment_id} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_loss:.6f} | "
            f"Val AUPRC {val_auprc:.6f} | "
            f"Val AUROC {val_auroc:.6f}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT = VALIDATION AUPRC
        # ----------------------------------------------------

        if val_auprc > best_val_auprc:

            best_val_auprc = val_auprc
            best_epoch = epoch

            torch.save(
                {
                    "experiment_id": experiment_id,
                    "config": config,
                    "feature_set": feature_set,
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_AUPRC": float(
                        val_auprc
                    ),
                    "val_AUROC": float(
                        val_auroc
                    ),
                    "seed": SEED,
                },
                checkpoint_file,
            )

            print(
                "  -> New best checkpoint saved "
                f"(Val AUPRC={val_auprc:.6f})"
            )

    # --------------------------------------------------------
    # LOAD BEST VALIDATION CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # FROZEN VALIDATION METRICS
    # --------------------------------------------------------

    (
        val_probs,
        val_targets,
        val_loss,
    ) = predict(
        model,
        val_loader,
    )

    val_auprc = average_precision_score(
        val_targets,
        val_probs,
    )

    val_auroc = roc_auc_score(
        val_targets,
        val_probs,
    )

    threshold, val_best_f1 = (
        find_best_f1_threshold(
            val_targets,
            val_probs,
        )
    )

    val_metrics = calculate_metrics(
        val_targets,
        val_probs,
        threshold,
    )

    # --------------------------------------------------------
    # TEST EVALUATION
    #
    # At this point:
    #   - checkpoint is frozen
    #   - threshold is frozen
    # Therefore test is now evaluated.
    # --------------------------------------------------------

    (
        test_probs,
        test_targets,
        test_loss,
    ) = predict(
        model,
        test_loader,
    )

    test_metrics = calculate_metrics(
        test_targets,
        test_probs,
        threshold,
    )

    elapsed = (
        time.time()
        - start_time
    )

    result = {
        "experiment_id": experiment_id,
        "family": config["family"],
        "feature_set": feature_set,
        "input_features": FEATURE_DIMS[
            feature_set
        ],

        "configuration": config,

        "seed": SEED,

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "pos_weight": POS_WEIGHT,

        "best_epoch": best_epoch,

        "validation": {
            "loss": float(val_loss),
            **val_metrics,
        },

        "test": {
            "loss": float(test_loss),
            **test_metrics,
        },

        "runtime_seconds": float(
            elapsed
        ),

        "history": history,

        "checkpoint": str(
            checkpoint_file
        ),
    }

    save_json(
        result_file,
        result,
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    np.savez_compressed(
        exp_dir
        / "validation_predictions.npz",
        probs=val_probs,
        targets=val_targets,
        threshold=np.array(
            [threshold],
            dtype=np.float32,
        ),
    )

    np.savez_compressed(
        exp_dir
        / "test_predictions.npz",
        probs=test_probs,
        targets=test_targets,
        threshold=np.array(
            [threshold],
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------------
    # COMPLETE MARKER
    # --------------------------------------------------------

    complete_file.write_text(
        "COMPLETE\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print(
        f"{experiment_id} COMPLETE"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Validation AUPRC: "
        f"{val_metrics['AUPRC']:.6f}"
    )

    print(
        f"Validation AUROC: "
        f"{val_metrics['AUROC']:.6f}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['F1']:.6f}"
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Test AUPRC: "
        f"{test_metrics['AUPRC']:.6f}"
    )

    print(
        f"Test AUROC: "
        f"{test_metrics['AUROC']:.6f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['F1']:.6f}"
    )

    return result


# ============================================================
# BUILD FINAL ABLATION SUMMARY
# ============================================================

def build_summary(results):

    # --------------------------------------------------------
    # Sort by validation AUPRC
    # --------------------------------------------------------

    validation_ranking = sorted(
        results,
        key=lambda x: x["validation"]["AUPRC"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Family-specific rankings
    # --------------------------------------------------------

    family_rankings = {}

    for family in [
        "GRU",
        "LSTM",
        "TRANSFORMER",
    ]:

        family_results = [
            r
            for r in results
            if r["family"] == family
        ]

        family_rankings[family] = sorted(
            family_results,
            key=lambda x: (
                x["validation"]["AUPRC"]
            ),
            reverse=True,
        )

    # --------------------------------------------------------
    # F0 BASELINES ALREADY COMPLETED
    #
    # These are recorded here only for comparison.
    # They are NOT retrained.
    # --------------------------------------------------------

    existing_f0 = {
        "GRU": {
            "model": "GRU-5",
            "feature_set": "F0",
            "validation_AUPRC": 0.082515,
            "test_AUPRC": 0.083072,
        },
        "LSTM": {
            "model": "LSTM-4",
            "feature_set": "F0",
            "validation_AUPRC": 0.076290,
            "test_AUPRC": 0.079506,
        },
        "TRANSFORMER": {
            "model": "TR-1",
            "feature_set": "F0",
            "validation_AUPRC": 0.086798,
            "test_AUPRC": 0.081869,
        },
    }

    summary = {
        "stage": "3.6",
        "description": (
            "Controlled causal feature ablation"
        ),

        "experiments_run": len(results),

        "experiments_expected": 9,

        "F0_retrained": False,

        "feature_hierarchy": [
            "F0",
            "F1",
            "F2",
            "F3",
        ],

        "existing_F0_baselines": existing_f0,

        "validation_ranking": [
            {
                "rank": i + 1,
                "experiment": r["experiment_id"],
                "family": r["family"],
                "feature_set": r["feature_set"],
                "AUPRC": r[
                    "validation"
                ]["AUPRC"],
                "AUROC": r[
                    "validation"
                ]["AUROC"],
                "F1": r[
                    "validation"
                ]["F1"],
            }
            for i, r in enumerate(
                validation_ranking
            )
        ],

        "family_rankings": {
            family: [
                {
                    "rank": i + 1,
                    "feature_set": r[
                        "feature_set"
                    ],
                    "AUPRC": r[
                        "validation"
                    ]["AUPRC"],
                    "AUROC": r[
                        "validation"
                    ]["AUROC"],
                    "F1": r[
                        "validation"
                    ]["F1"],
                }
                for i, r in enumerate(
                    ranked
                )
            ]
            for family, ranked
            in family_rankings.items()
        },

        "results": results,
    }

    summary_path = (
        RESULT_ROOT
        / "feature_ablation_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STAGE 3.6 — VALIDATION FEATURE ABLATION RANKING"
    )
    print("=" * 70)

    for i, r in enumerate(
        validation_ranking,
        start=1,
    ):

        print(
            f"{i}. "
            f"{r['experiment_id']} | "
            f"AUPRC "
            f"{r['validation']['AUPRC']:.6f} | "
            f"AUROC "
            f"{r['validation']['AUROC']:.6f} | "
            f"F1 "
            f"{r['validation']['F1']:.6f}"
        )

    print()
    print("=" * 70)
    print(
        "STAGE 3.6 — FAMILY FEATURE ABLATION"
    )
    print("=" * 70)

    for family, ranked in family_rankings.items():

        print()
        print(f"{family}")

        for i, r in enumerate(
            ranked,
            start=1,
        ):

            print(
                f"  {i}. "
                f"{r['feature_set']} | "
                f"AUPRC "
                f"{r['validation']['AUPRC']:.6f} | "
                f"AUROC "
                f"{r['validation']['AUROC']:.6f}"
            )

    print()
    print(
        "Summary saved:"
    )
    print(
        summary_path
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print(
        "STAGE 3.6 — CAUSAL FEATURE ABLATION"
    )
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Training-only pos_weight: "
        f"{POS_WEIGHT:.4f}"
    )

    print()
    print(
        "Experiments:"
    )

    for exp in EXPERIMENTS:

        print(
            f"  {exp['id']} -> "
            f"{exp['family']} + "
            f"{exp['feature_set']}"
        )

    print()
    print(
        "F0 will NOT be retrained."
    )

    print(
        "Total new training runs: 9"
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    results = []

    for config in EXPERIMENTS:

        result = run_experiment(
            config
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    build_summary(
        results
    )

    print()
    print("=" * 70)
    print(
        "STAGE 3.6 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Completed: "
        f"{len(results)}/9"
    )


if __name__ == "__main__":
    main()