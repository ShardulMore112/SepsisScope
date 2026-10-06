"""
Stage 3.1 — Modeling Data Infrastructure
=========================================

Purpose:
    Provide a single, reusable PyTorch data pipeline for all downstream
    temporal models (GRU, LSTM, Causal Transformer).

Supported feature sets:
    F0, F1, F2, F3

Responsibilities:
    - Load frozen Step-2 processed arrays
    - Reconstruct variable-length patient sequences using patient offsets
    - Batch patients together
    - Dynamically pad sequences within each batch
    - Generate padding masks
    - Preserve labels and sequence lengths
    - Provide reproducible DataLoaders

IMPORTANT:
    This file contains NO predictive model.
    It does NOT modify the processed data.
    It does NOT use the test set for training or selection.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


# ============================================================
# REPRODUCIBILITY
# ============================================================

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducible data loading and experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# FEATURE SET INFORMATION
# ============================================================

EXPECTED_FEATURE_COUNTS = {
    "F0": 80,
    "F1": 152,
    "F2": 296,
    "F3": 306,
}


# ============================================================
# DATASET
# ============================================================

class SepsisSequenceDataset(Dataset):
    """
    Patient-level temporal dataset.

    Each item corresponds to ONE patient.

    Returns:
        {
            "x":         Tensor [T, F],
            "y":         Tensor [T],
            "length":    int,
            "patient_key": str,
            "hours":     Tensor [T]
        }

    X is stored as one continuous hourly matrix. Patient boundaries
    are recovered using patient_offsets.npy.
    """

    def __init__(
        self,
        split: str,
        feature_set: str = "F0",
        processed_root: Path | str = PROCESSED_ROOT,
    ):
        super().__init__()

        split = split.lower()
        feature_set = feature_set.upper()

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected: train, validation, or test."
            )

        if feature_set not in EXPECTED_FEATURE_COUNTS:
            raise ValueError(
                f"Invalid feature set '{feature_set}'. "
                f"Expected: F0, F1, F2, or F3."
            )

        self.split = split
        self.feature_set = feature_set

        self.root = (
            Path(processed_root)
            / feature_set
            / split
        )

        if not self.root.exists():
            raise FileNotFoundError(
                f"Processed directory does not exist:\n{self.root}"
            )

        # ----------------------------------------------------
        # Load arrays
        # ----------------------------------------------------

        self.X = np.load(
            self.root / "X.npy",
            mmap_mode="r"
        )

        self.labels = np.load(
            self.root / "labels.npy",
            mmap_mode="r"
        )

        self.patient_offsets = np.load(
            self.root / "patient_offsets.npy"
        )

        self.patient_keys = np.load(
            self.root / "patient_keys.npy"
        )

        self.hour_index = np.load(
            self.root / "hour_index.npy",
            mmap_mode="r"
        )

        # ----------------------------------------------------
        # Load feature names
        # ----------------------------------------------------

        with open(
            self.root / "feature_names.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.feature_names = json.load(f)

        # ----------------------------------------------------
        # Validate structure
        # ----------------------------------------------------

        self._validate()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate(self) -> None:
        """
        Validate that processed arrays are internally consistent.
        """

        expected_features = EXPECTED_FEATURE_COUNTS[self.feature_set]

        if self.X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {self.X.shape}"
            )

        if self.X.shape[1] != expected_features:
            raise ValueError(
                f"{self.feature_set} expected {expected_features} "
                f"features, got {self.X.shape[1]}"
            )

        if len(self.labels) != len(self.X):
            raise ValueError(
                "X and labels have different numbers of rows."
            )

        if len(self.hour_index) != len(self.X):
            raise ValueError(
                "X and hour_index have different numbers of rows."
            )

        if len(self.patient_offsets) < 2:
            raise ValueError(
                "patient_offsets must contain at least two values."
            )

        if len(self.patient_keys) != len(self.patient_offsets) - 1:
            raise ValueError(
                "Number of patient keys does not match number "
                "of patient sequences."
            )

        if self.patient_offsets[0] != 0:
            raise ValueError(
                "patient_offsets must start at zero."
            )

        if self.patient_offsets[-1] != len(self.X):
            raise ValueError(
                "Final patient offset must equal number of rows in X."
            )

        if not np.all(
            self.patient_offsets[1:] >= self.patient_offsets[:-1]
        ):
            raise ValueError(
                "patient_offsets must be monotonically increasing."
            )

        # Check labels are binary.
        unique_labels = np.unique(self.labels)

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"Labels must be binary. Found: {unique_labels}"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self) -> int:
        """
        Number of patients.
        """
        return len(self.patient_keys)

    # ========================================================
    # ITEM
    # ========================================================

    def __getitem__(
        self,
        index: int
    ) -> Dict:

        start = int(self.patient_offsets[index])
        end = int(self.patient_offsets[index + 1])

        x = np.asarray(
            self.X[start:end],
            dtype=np.float32
        )

        y = np.asarray(
            self.labels[start:end],
            dtype=np.int64
        )

        hours = np.asarray(
            self.hour_index[start:end],
            dtype=np.int64
        )

        patient_key = self.patient_keys[index]

        # Convert numpy scalar/bytes safely to string.
        if isinstance(patient_key, bytes):
            patient_key = patient_key.decode("utf-8")

        patient_key = str(patient_key)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "length": end - start,
            "patient_key": patient_key,
            "hours": torch.from_numpy(hours),
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def sepsis_collate_fn(
    batch: List[Dict]
) -> Dict:
    """
    Dynamically pad variable-length patient sequences.

    Input:
        list of patient dictionaries

    Output:
        x:
            [batch_size, max_sequence_length, num_features]

        y:
            [batch_size, max_sequence_length]

        padding_mask:
            [batch_size, max_sequence_length]

            True  = padding
            False = real timestep

        lengths:
            [batch_size]

        patient_keys:
            list[str]

        hours:
            [batch_size, max_sequence_length]

            Padded positions contain -1.
    """

    batch_size = len(batch)

    lengths = torch.tensor(
        [item["length"] for item in batch],
        dtype=torch.long
    )

    max_length = int(lengths.max().item())

    num_features = batch[0]["x"].shape[1]

    # --------------------------------------------------------
    # Allocate padded tensors
    # --------------------------------------------------------

    x = torch.zeros(
        batch_size,
        max_length,
        num_features,
        dtype=torch.float32
    )

    y = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.long
    )

    hours = torch.full(
        (batch_size, max_length),
        fill_value=-1,
        dtype=torch.long
    )

    # True means padding.
    padding_mask = torch.ones(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    # --------------------------------------------------------
    # Insert patient sequences
    # --------------------------------------------------------

    patient_keys = []

    for i, item in enumerate(batch):

        length = item["length"]

        x[i, :length] = item["x"]

        y[i, :length] = item["y"]

        hours[i, :length] = item["hours"]

        padding_mask[i, :length] = False

        patient_keys.append(
            item["patient_key"]
        )

    return {
        "x": x,
        "y": y,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "patient_keys": patient_keys,
        "hours": hours,
    }


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloader(
    split: str,
    feature_set: str = "F0",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = DEFAULT_SEED,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for a specific split and feature set.

    Parameters
    ----------
    split:
        train / validation / test

    feature_set:
        F0 / F1 / F2 / F3

    batch_size:
        Number of patients per batch.

    shuffle:
        Should normally be True only for training.

    num_workers:
        Number of PyTorch worker processes.

    seed:
        Reproducibility seed.

    pin_memory:
        Useful when transferring batches to CUDA.
    """

    dataset = SepsisSequenceDataset(
        split=split,
        feature_set=feature_set,
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sepsis_collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


# ============================================================
# COMPLETE LOADER SET
# ============================================================

def create_all_dataloaders(
    feature_set: str = "F0",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = DEFAULT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    IMPORTANT:
        Test loader is created only for later final evaluation.
        This function does not perform any training or selection.
    """

    train_loader = create_dataloader(
        split="train",
        feature_set=feature_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )

    validation_loader = create_dataloader(
        split="validation",
        feature_set=feature_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    test_loader = create_dataloader(
        split="test",
        feature_set=feature_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ============================================================
# INFRASTRUCTURE SELF-TEST
# ============================================================

def run_self_test() -> None:
    """
    Verify that the modeling data infrastructure works.

    This does NOT train a model.
    """

    print("=" * 70)
    print("STAGE 3.1 — MODELING DATA INFRASTRUCTURE SELF-TEST")
    print("=" * 70)

    set_seed(DEFAULT_SEED)

    for feature_set in ["F0", "F1", "F2", "F3"]:

        print(f"\nTesting {feature_set}...")

        loader = create_dataloader(
            split="train",
            feature_set=feature_set,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        x = batch["x"]
        y = batch["y"]
        padding_mask = batch["padding_mask"]
        lengths = batch["lengths"]

        print(f"  X shape:           {tuple(x.shape)}")
        print(f"  Y shape:           {tuple(y.shape)}")
        print(f"  Padding mask:      {tuple(padding_mask.shape)}")
        print(f"  Lengths:            {lengths.tolist()}")
        print(f"  Patients:           {batch['patient_keys']}")

        assert x.ndim == 3
        assert y.ndim == 2
        assert padding_mask.ndim == 2

        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == padding_mask.shape[0]

        assert x.shape[1] == y.shape[1]
        assert x.shape[1] == padding_mask.shape[1]

        assert x.shape[2] == EXPECTED_FEATURE_COUNTS[feature_set]

        for i, length in enumerate(lengths.tolist()):

            # Real timesteps must NOT be masked.
            assert not padding_mask[i, :length].any()

            # Padding timesteps MUST be masked.
            if length < x.shape[1]:
                assert padding_mask[i, length:].all()

    print("\n" + "=" * 70)
    print("STAGE 3.1 SELF-TEST PASSED")
    print("=" * 70)
    print("No models were trained.")
    print("No processed data was modified.")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_self_test()