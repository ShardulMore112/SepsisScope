"""
STAGE 3.4 — SYSTEMATIC LSTM FAMILY
===================================

Resume-safe LSTM family experiment runner.

Experiment matrix
-----------------
LSTM-1: hidden=64,  layers=1, dropout=0.0, lr=1e-3
LSTM-2: hidden=128, layers=1, dropout=0.0, lr=1e-3
LSTM-3: hidden=128, layers=2, dropout=0.1, lr=1e-3
LSTM-4: hidden=128, layers=2, dropout=0.2, lr=5e-4
LSTM-5: hidden=256, layers=2, dropout=0.2, lr=5e-4

Feature set:
    F0 = 80 features

Training:
    timestep-level weighted BCE
    training-only positive timestep weight
    AdamW
    gradient clipping

Selection:
    best validation AUPRC
    threshold selected on validation by F1

Test:
    evaluated using frozen validation threshold

Resume behavior
---------------
If checkpoint exists:
    -> DO NOT TRAIN
    -> load checkpoint
    -> run validation inference
    -> recover validation threshold
    -> use existing test predictions when available
    -> reconstruct standardized result

If checkpoint does not exist:
    -> train experiment
    -> save checkpoint
    -> save predictions
    -> save result
    -> save completion marker
"""

from __future__ import annotations

import json
import random
from datetime import datetime
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

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_result"
    / "lstm_family"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "lstm_family"
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_SET = "F0"

SEED = 42

BATCH_SIZE = 128

EPOCHS = 15

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0


# ============================================================
# EXPERIMENT MATRIX
# ============================================================

EXPERIMENTS = [

    {
        "id": "LSTM-1",
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
    },

    {
        "id": "LSTM-2",
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
    },

    {
        "id": "LSTM-3",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },

    {
        "id": "LSTM-4",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 5e-4,
    },

    {
        "id": "LSTM-5",
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 5e-4,
    },

]


# ============================================================
# UTILITY
# ============================================================

def now_string():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# PATHS FOR EACH EXPERIMENT
# ============================================================

def get_paths(experiment_id):

    name = experiment_id.lower()

    return {
        "checkpoint":
            CHECKPOINT_ROOT
            / f"{name}.pt",

        "result":
            RESULTS_ROOT
            / f"{name}_result.json",

        "completion":
            RESULTS_ROOT
            / f"{name}_complete.json",

        "predictions":
            RESULTS_ROOT
            / f"{name}_test_predictions.npz",
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_split(split):

    split_root = (
        PROCESSED_ROOT
        / FEATURE_SET
        / split
    )

    X = np.load(
        split_root / "X.npy",
        mmap_mode="r",
    )

    y = np.load(
        split_root / "labels.npy",
        mmap_mode="r",
    )

    offsets = np.load(
        split_root / "patient_offsets.npy"
    )

    patient_keys = np.load(
        split_root / "patient_keys.npy"
    )

    hour_index = np.load(
        split_root / "hour_index.npy"
    )

    return (
        X,
        y,
        offsets,
        patient_keys,
        hour_index,
    )


# ============================================================
# SEQUENCE DATASET
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

        start = int(
            self.offsets[index]
        )

        end = int(
            self.offsets[index + 1]
        )

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

    for i, (
        X_patient,
        y_patient,
    ) in enumerate(batch):

        length = X_patient.shape[0]

        X[
            i,
            :length
        ] = X_patient

        y[
            i,
            :length
        ] = y_patient

        padding_mask[
            i,
            :length
        ] = False

    return (
        X,
        y,
        padding_mask,
        torch.tensor(
            lengths,
            dtype=torch.long,
        ),
    )


# ============================================================
# SIMPLE DATA LOADER
# ============================================================

class SequenceDataLoader:

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=SEED,
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
                start:
                start + self.batch_size
            ]

            batch = [
                self.dataset.get_patient(
                    int(index)
                )
                for index
                in batch_indices
            ]

            yield collate_sequences(
                batch
            )


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
    ):

        super().__init__()

        effective_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
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

        hidden, _ = self.lstm(X)

        logits = self.output(
            hidden
        ).squeeze(-1)

        return logits


# ============================================================
# VALID TIMESTEPS
# ============================================================

def flatten_valid(
    logits,
    labels,
    padding_mask,
):

    valid = ~padding_mask

    return (
        logits[valid],
        labels[valid],
    )


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

    probabilities = []

    labels = []

    for (
        X,
        y,
        padding_mask,
        lengths,
    ) in loader:

        X = X.to(device)

        y = y.to(device)

        padding_mask = padding_mask.to(
            device
        )

        logits = model(
            X,
            padding_mask,
        )

        valid_logits, valid_labels = (
            flatten_valid(
                logits,
                y,
                padding_mask,
            )
        )

        probs = torch.sigmoid(
            valid_logits
        )

        probabilities.append(
            probs.cpu().numpy()
        )

        labels.append(
            valid_labels.cpu().numpy()
        )

    return (
        np.concatenate(
            probabilities
        ),
        np.concatenate(
            labels
        ),
    )


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def find_best_threshold(
    probabilities,
    labels,
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
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

        _, _, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="binary",
                zero_division=0,
            )
        )

        if f1 > best_f1:

            best_f1 = float(f1)

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

def calculate_metrics(
    probabilities,
    labels,
    threshold,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    auroc = roc_auc_score(
        labels,
        probabilities,
    )

    auprc = average_precision_score(
        labels,
        probabilities,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

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

        "AUROC":
            float(auroc),

        "AUPRC":
            float(auprc),

        "F1":
            float(f1),

        "Precision":
            float(precision),

        "Sensitivity":
            float(sensitivity),

        "Specificity":
            float(specificity),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "Threshold":
            float(threshold),
    }


# ============================================================
# NPZ HELPERS
# ============================================================

def find_npz_array(
    data,
    possible_names,
):

    available = set(
        data.files
    )

    for name in possible_names:

        if name in available:

            return np.asarray(
                data[name]
            )

    lower_map = {
        key.lower(): key
        for key in available
    }

    for name in possible_names:

        if (
            name.lower()
            in lower_map
        ):

            return np.asarray(
                data[
                    lower_map[
                        name.lower()
                    ]
                ]
            )

    return None


def load_existing_test_predictions(
    prediction_path
):

    if not prediction_path.exists():

        return None

    try:

        data = np.load(
            prediction_path
        )

        probabilities = find_npz_array(
            data,
            [
                "probabilities",
                "test_probabilities",
                "probs",
                "test_probs",
                "predictions",
                "test_predictions",
            ],
        )

        labels = find_npz_array(
            data,
            [
                "labels",
                "test_labels",
                "y_test",
                "targets",
            ],
        )

        if (
            probabilities is None
            or labels is None
        ):

            return None

        return (
            probabilities,
            labels,
        )

    except Exception:

        return None


def save_test_predictions(
    path,
    probabilities,
    labels,
):

    np.savez_compressed(
        path,
        probabilities=
            probabilities,
        labels=
            labels,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    device,
):

    try:

        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            checkpoint_path,
            map_location=device,
        )


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

def get_experiment_config(
    experiment_id
):

    for config in EXPERIMENTS:

        if (
            config["id"].lower()
            == experiment_id.lower()
        ):

            return config.copy()

    raise ValueError(
        f"Unknown experiment: "
        f"{experiment_id}"
    )


def get_model_config_from_checkpoint(
    checkpoint,
    experiment_id,
):

    config = checkpoint.get(
        "experiment"
    )

    if isinstance(
        config,
        dict,
    ):

        if all(
            key in config
            for key in [
                "hidden_size",
                "num_layers",
                "dropout",
            ]
        ):

            fallback = (
                get_experiment_config(
                    experiment_id
                )
            )

            return {

                "id":
                    experiment_id,

                "hidden_size":
                    int(
                        config[
                            "hidden_size"
                        ]
                    ),

                "num_layers":
                    int(
                        config[
                            "num_layers"
                        ]
                    ),

                "dropout":
                    float(
                        config[
                            "dropout"
                        ]
                    ),

                "learning_rate":
                    float(
                        config.get(
                            "learning_rate",
                            fallback[
                                "learning_rate"
                            ],
                        )
                    ),
            }

    return get_experiment_config(
        experiment_id
    )


# ============================================================
# RECONSTRUCT COMPLETED EXPERIMENT
# ============================================================

def reconstruct_completed_experiment(
    experiment_id,
    train_positive_weight,
    device,
):

    paths = get_paths(
        experiment_id
    )

    checkpoint_path = (
        paths["checkpoint"]
    )

    result_path = (
        paths["result"]
    )

    completion_path = (
        paths["completion"]
    )

    prediction_path = (
        paths["predictions"]
    )

    print("\n")
    print("-" * 70)

    print(
        f"{experiment_id}: "
        "LOADING EXISTING CHECKPOINT"
    )

    print(
        "NO TRAINING WILL BE PERFORMED."
    )

    print("-" * 70)

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    config = (
        get_model_config_from_checkpoint(
            checkpoint,
            experiment_id,
        )
    )

    print(
        f"Hidden size: "
        f"{config['hidden_size']}"
    )

    print(
        f"Layers: "
        f"{config['num_layers']}"
    )

    print(
        f"Dropout: "
        f"{config['dropout']}"
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
    ) = load_split(
        "validation"
    )

    input_size = X_val.shape[1]

    if input_size != 80:

        raise ValueError(
            f"Expected F0 to contain 80 features, "
            f"found {input_size}."
        )

    val_dataset = (
        SepsisSequenceDataset(
            X_val,
            y_val,
            val_offsets,
        )
    )

    val_loader = (
        SequenceDataLoader(
            val_dataset,
            BATCH_SIZE,
            shuffle=False,
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config[
            "hidden_size"
        ],
        num_layers=config[
            "num_layers"
        ],
        dropout=config[
            "dropout"
        ],
    ).to(device)

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        state_dict = checkpoint.get(
            "state_dict"
        )

    if state_dict is None:

        raise ValueError(
            f"{experiment_id}: "
            "Checkpoint contains no model state."
        )

    model.load_state_dict(
        state_dict
    )

    print(
        f"{experiment_id}: "
        "Checkpoint loaded."
    )

    # --------------------------------------------------------
    # Validation inference
    # --------------------------------------------------------

    (
        val_probabilities,
        val_labels,
    ) = predict_dataset(
        model,
        val_loader,
        device,
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    saved_threshold = (
        checkpoint.get(
            "validation_threshold"
        )
    )

    if saved_threshold is not None:

        threshold = float(
            saved_threshold
        )

        print(
            f"{experiment_id}: "
            f"Using saved threshold "
            f"{threshold:.4f}"
        )

    else:

        (
            threshold,
            _,
        ) = find_best_threshold(
            val_probabilities,
            val_labels,
        )

        print(
            f"{experiment_id}: "
            f"Recovered threshold "
            f"{threshold:.4f}"
        )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    validation_metrics = (
        calculate_metrics(
            val_probabilities,
            val_labels,
            threshold,
        )
    )

    # --------------------------------------------------------
    # Existing test predictions
    # --------------------------------------------------------

    existing_test = (
        load_existing_test_predictions(
            prediction_path
        )
    )

    if existing_test is not None:

        print(
            f"{experiment_id}: "
            "Using existing test predictions."
        )

        (
            test_probabilities,
            test_labels,
        ) = existing_test

    else:

        print(
            f"{experiment_id}: "
            "Generating test predictions "
            "from checkpoint."
        )

        (
            X_test,
            y_test,
            test_offsets,
            _,
            _,
        ) = load_split(
            "test"
        )

        test_dataset = (
            SepsisSequenceDataset(
                X_test,
                y_test,
                test_offsets,
            )
        )

        test_loader = (
            SequenceDataLoader(
                test_dataset,
                BATCH_SIZE,
                shuffle=False,
            )
        )

        (
            test_probabilities,
            test_labels,
        ) = predict_dataset(
            model,
            test_loader,
            device,
        )

        save_test_predictions(
            prediction_path,
            test_probabilities,
            test_labels,
        )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_metrics = calculate_metrics(
        test_probabilities,
        test_labels,
        threshold,
    )

    best_epoch = int(
        checkpoint.get(
            "best_epoch",
            0,
        )
    )

    best_val_auprc = float(
        checkpoint.get(
            "best_validation_auprc",
            validation_metrics[
                "AUPRC"
            ],
        )
    )

    # --------------------------------------------------------
    # Standard result
    # --------------------------------------------------------

    result = {

        "stage":
            "3.4",

        "family":
            "LSTM",

        "experiment":
            config,

        "feature_set":
            FEATURE_SET,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "positive_weight":
            float(
                checkpoint.get(
                    "positive_weight",
                    train_positive_weight,
                )
            ),

        "best_epoch":
            best_epoch,

        "best_validation_AUPRC":
            best_val_auprc,

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "test_predictions":
            str(
                prediction_path
            ),

        "reconstructed_without_training":
            True,

        "completed_at":
            now_string(),
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

    completion = {

        "experiment":
            experiment_id,

        "status":
            "COMPLETE",

        "completed_at":
            now_string(),

        "reconstructed_without_training":
            True,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "result_file":
            str(
                result_path
            ),

        "test_predictions":
            str(
                prediction_path
            ),
    }

    with open(
        completion_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            completion,
            f,
            indent=2,
        )

    print(
        f"{experiment_id}: "
        "RECONSTRUCTION COMPLETE"
    )

    print(
        f"  Validation AUPRC: "
        f"{validation_metrics['AUPRC']:.6f}"
    )

    print(
        f"  Validation AUROC: "
        f"{validation_metrics['AUROC']:.6f}"
    )

    print(
        f"  Validation F1: "
        f"{validation_metrics['F1']:.6f}"
    )

    print(
        f"  Test AUPRC: "
        f"{test_metrics['AUPRC']:.6f}"
    )

    print(
        f"  Test AUROC: "
        f"{test_metrics['AUROC']:.6f}"
    )

    print(
        f"  Test F1: "
        f"{test_metrics['F1']:.6f}"
    )

    return result


# ============================================================
# TRAIN EXPERIMENT
# ============================================================

def train_experiment(
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    input_size,
    pos_weight,
    device,
):

    experiment_id = config["id"]

    print("\n")
    print("=" * 70)

    print(
        f"TRAINING {experiment_id}"
    )

    print("=" * 70)

    set_seed(SEED)

    train_loader = (
        SequenceDataLoader(
            train_dataset,
            BATCH_SIZE,
            shuffle=True,
            seed=SEED,
        )
    )

    val_loader = (
        SequenceDataLoader(
            val_dataset,
            BATCH_SIZE,
            shuffle=False,
        )
    )

    test_loader = (
        SequenceDataLoader(
            test_dataset,
            BATCH_SIZE,
            shuffle=False,
        )
    )

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config[
            "hidden_size"
        ],
        num_layers=config[
            "num_layers"
        ],
        dropout=config[
            "dropout"
        ],
    ).to(device)

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config[
            "learning_rate"
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best_auprc = -np.inf

    best_epoch = 0

    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        total_count = 0

        for (
            X,
            y,
            padding_mask,
            _,
        ) in train_loader:

            X = X.to(device)

            y = y.to(device)

            padding_mask = (
                padding_mask.to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X,
                padding_mask,
            )

            valid_logits, valid_y = (
                flatten_valid(
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
                loss.item()
                * count
            )

            total_count += count

        train_loss = (
            total_loss
            / total_count
        )

        (
            val_probabilities,
            val_labels,
        ) = predict_dataset(
            model,
            val_loader,
            device,
        )

        val_auroc = roc_auc_score(
            val_labels,
            val_probabilities,
        )

        val_auprc = (
            average_precision_score(
                val_labels,
                val_probabilities,
            )
        )

        print(
            f"{experiment_id} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.5f} | "
            f"Val AUROC "
            f"{val_auroc:.5f} | "
            f"Val AUPRC "
            f"{val_auprc:.5f}"
        )

        if val_auprc > best_auprc:

            best_auprc = float(
                val_auprc
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }

    model.load_state_dict(
        best_state
    )

    (
        val_probabilities,
        val_labels,
    ) = predict_dataset(
        model,
        val_loader,
        device,
    )

    (
        threshold,
        _,
    ) = find_best_threshold(
        val_probabilities,
        val_labels,
    )

    validation_metrics = (
        calculate_metrics(
            val_probabilities,
            val_labels,
            threshold,
        )
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
        threshold,
    )

    paths = get_paths(
        experiment_id
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "experiment":
            config,

        "feature_set":
            FEATURE_SET,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "positive_weight":
            float(pos_weight),

        "best_epoch":
            int(best_epoch),

        "best_validation_auprc":
            float(best_auprc),

        "validation_threshold":
            float(threshold),
    }

    torch.save(
        checkpoint,
        paths["checkpoint"],
    )

    save_test_predictions(
        paths["predictions"],
        test_probabilities,
        test_labels,
    )

    result = {

        "stage":
            "3.4",

        "family":
            "LSTM",

        "experiment":
            config,

        "feature_set":
            FEATURE_SET,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "positive_weight":
            float(pos_weight),

        "best_epoch":
            int(best_epoch),

        "best_validation_AUPRC":
            float(best_auprc),

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "checkpoint":
            str(
                paths["checkpoint"]
            ),

        "test_predictions":
            str(
                paths["predictions"]
            ),

        "reconstructed_without_training":
            False,

        "completed_at":
            now_string(),
    }

    with open(
        paths["result"],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    completion = {

        "experiment":
            experiment_id,

        "status":
            "COMPLETE",

        "completed_at":
            now_string(),

        "reconstructed_without_training":
            False,

        "checkpoint":
            str(
                paths["checkpoint"]
            ),

        "result_file":
            str(
                paths["result"]
            ),

        "test_predictions":
            str(
                paths["predictions"]
            ),
    }

    with open(
        paths["completion"],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            completion,
            f,
            indent=2,
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 70)

    print(
        "STAGE 3.4 — SYSTEMATIC LSTM FAMILY"
    )

    print("=" * 70)

    print(
        f"\nStarted at: "
        f"{now_string()}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Feature set: {FEATURE_SET}"
    )

    print(
        f"Experiments: "
        f"{len(EXPERIMENTS)}"
    )

    print(
        "\nResume mode: ENABLED"
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "EXPERIMENT STATUS"
    )

    print("=" * 70)

    for config in EXPERIMENTS:

        paths = get_paths(
            config["id"]
        )

        if paths["checkpoint"].exists():

            print(
                f"{config['id']}: "
                "CHECKPOINT FOUND — "
                "NO RETRAINING"
            )

        else:

            print(
                f"{config['id']}: "
                "CHECKPOINT MISSING — "
                "WILL TRAIN"
            )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("\n")
    print(
        "Loading frozen Step-2 data..."
    )

    (
        X_train,
        y_train,
        train_offsets,
        train_keys,
        train_hours,
    ) = load_split(
        "train"
    )

    (
        X_val,
        y_val,
        val_offsets,
        val_keys,
        val_hours,
    ) = load_split(
        "validation"
    )

    (
        X_test,
        y_test,
        test_offsets,
        test_keys,
        test_hours,
    ) = load_split(
        "test"
    )

    input_size = X_train.shape[1]

    print(
        f"Train patients: "
        f"{len(train_offsets) - 1:,}"
    )

    print(
        f"Validation patients: "
        f"{len(val_offsets) - 1:,}"
    )

    print(
        f"Test patients: "
        f"{len(test_offsets) - 1:,}"
    )

    print(
        f"Input features: "
        f"{input_size}"
    )

    if input_size != 80:

        raise ValueError(
            "F0 must contain exactly 80 features."
        )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        SepsisSequenceDataset(
            X_train,
            y_train,
            train_offsets,
        )
    )

    val_dataset = (
        SepsisSequenceDataset(
            X_val,
            y_val,
            val_offsets,
        )
    )

    test_dataset = (
        SepsisSequenceDataset(
            X_test,
            y_test,
            test_offsets,
        )
    )

    # --------------------------------------------------------
    # POSITIVE WEIGHT
    # --------------------------------------------------------

    train_labels = np.asarray(
        y_train
    )

    positive_timesteps = int(
        train_labels.sum()
    )

    negative_timesteps = int(
        train_labels.size
        - positive_timesteps
    )

    pos_weight = (
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
        f"{pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    results = []

    for config in EXPERIMENTS:

        experiment_id = config["id"]

        paths = get_paths(
            experiment_id
        )

        print("\n")
        print("=" * 70)

        if paths["checkpoint"].exists():

            print(
                f"{experiment_id}: "
                "EXISTING CHECKPOINT DETECTED"
            )

            print(
                "Skipping training."
            )

            result = (
                reconstruct_completed_experiment(
                    experiment_id,
                    pos_weight,
                    device,
                )
            )

        else:

            print(
                f"{experiment_id}: "
                "NO CHECKPOINT FOUND"
            )

            print(
                "Training is required."
            )

            result = train_experiment(
                config,
                train_dataset,
                val_dataset,
                test_dataset,
                input_size,
                pos_weight,
                device,
            )

        results.append(
            result
        )

    # --------------------------------------------------------
    # VALIDATION RANKING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "LSTM FAMILY — VALIDATION RANKING"
    )

    print("=" * 70)

    validation_ranking = sorted(
        results,
        key=lambda result:
            result[
                "validation"
            ][
                "AUPRC"
            ],
        reverse=True,
    )

    for rank, result in enumerate(
        validation_ranking,
        start=1,
    ):

        experiment_id = (
            result[
                "experiment"
            ].get(
                "id",
                "UNKNOWN",
            )
        )

        metrics = result[
            "validation"
        ]

        print(
            f"{rank}. "
            f"{experiment_id} | "
            f"AUPRC "
            f"{metrics['AUPRC']:.6f} | "
            f"AUROC "
            f"{metrics['AUROC']:.6f} | "
            f"F1 "
            f"{metrics['F1']:.6f} | "
            f"Sensitivity "
            f"{metrics['Sensitivity']:.4f} | "
            f"Specificity "
            f"{metrics['Specificity']:.4f}"
        )

    # --------------------------------------------------------
    # TEST RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "LSTM FAMILY — TEST RESULTS"
    )

    print("=" * 70)

    test_ranking = sorted(
        results,
        key=lambda result:
            result[
                "test"
            ][
                "AUPRC"
            ],
        reverse=True,
    )

    for rank, result in enumerate(
        test_ranking,
        start=1,
    ):

        experiment_id = (
            result[
                "experiment"
            ].get(
                "id",
                "UNKNOWN",
            )
        )

        metrics = result[
            "test"
        ]

        print(
            f"{rank}. "
            f"{experiment_id} | "
            f"AUPRC "
            f"{metrics['AUPRC']:.6f} | "
            f"AUROC "
            f"{metrics['AUROC']:.6f} | "
            f"F1 "
            f"{metrics['F1']:.6f} | "
            f"Sensitivity "
            f"{metrics['Sensitivity']:.4f} | "
            f"Specificity "
            f"{metrics['Specificity']:.4f}"
        )

    # --------------------------------------------------------
    # VALIDATION WINNER
    # --------------------------------------------------------

    winner = validation_ranking[0]

    winner_id = (
        winner[
            "experiment"
        ]["id"]
    )

    print("\n")
    print("=" * 70)

    print(
        "VALIDATION-SELECTED LSTM"
    )

    print("=" * 70)

    print(
        f"Selected model: "
        f"{winner_id}"
    )

    print(
        f"Validation AUPRC: "
        f"{winner['validation']['AUPRC']:.6f}"
    )

    print(
        f"Validation AUROC: "
        f"{winner['validation']['AUROC']:.6f}"
    )

    print(
        f"Validation F1: "
        f"{winner['validation']['F1']:.6f}"
    )

    print(
        f"Frozen threshold: "
        f"{winner['validation']['Threshold']:.4f}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "stage":
            "3.4",

        "family":
            "LSTM",

        "feature_set":
            FEATURE_SET,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "positive_weight":
            float(pos_weight),

        "total_experiments":
            len(EXPERIMENTS),

        "completed_experiments":
            len(results),

        "validation_selected_model":
            winner_id,

        "validation_ranking":
            [
                item[
                    "experiment"
                ]["id"]
                for item
                in validation_ranking
            ],

        "experiments":
            results,

        "created_at":
            now_string(),
    }

    summary_path = (
        RESULTS_ROOT
        / "lstm_family_summary.json"
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

    progress_path = (
        RESULTS_ROOT
        / "lstm_family_progress.json"
    )

    with open(
        progress_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "LSTM FAMILY COMPLETE"
    )

    print("=" * 70)

    print(
        f"Completed: "
        f"{len(results)}/"
        f"{len(EXPERIMENTS)}"
    )

    print(
        f"Validation winner: "
        f"{winner_id}"
    )

    print(
        f"Summary:"
    )

    print(
        summary_path
    )

    print(
        f"\nFinished at: "
        f"{now_string()}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()