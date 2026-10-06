# ============================================================
# X-FedSepsis - Streamlit Research Prototype
# ============================================================
#
# Upload one PhysioNet/CinC Sepsis Challenge .psv patient file
# and perform strictly causal hour-by-hour inference using:
#
#   1. GRU5
#   2. LSTM4
#   3. TR1 Causal Transformer
#
# Decision rule:
#   OR rule:
#   If ANY model crosses its frozen validation threshold,
#   raise a sepsis alert.
#
# IMPORTANT:
#   Research prototype only.
#   NOT a clinical diagnostic system.
#
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="X-FedSepsis",
    page_icon="🫀",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

GRU_CHECKPOINT = (
    CHECKPOINT_DIR
    / "gru_family"
    / "gru-5.pt"
)

LSTM_CHECKPOINT = (
    CHECKPOINT_DIR
    / "lstm_family"
    / "lstm-4.pt"
)

TRANSFORMER_CHECKPOINT = (
    CHECKPOINT_DIR
    / "transformer_family"
    / "TR-1.pt"
)

MEDIAN_FILE = (
    PROJECT_ROOT
    / "results"
    / "preprocessing_result"
    / "training_medians.csv"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# ORIGINAL 40 PHYSIONET FEATURES
# ============================================================

BASE_FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]


# ============================================================
# MODEL INPUT
# ============================================================

INPUT_SIZE = 80


# ============================================================
# FROZEN VALIDATION THRESHOLDS
# ============================================================

GRU_THRESHOLD = 0.86
LSTM_THRESHOLD = 0.84
TRANSFORMER_THRESHOLD = 0.80


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def clean_state_dict(state_dict):
    """
    Remove common DataParallel prefix if present.
    """

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(
    model,
    checkpoint_path,
):
    """
    Load a trained checkpoint and provide a useful error
    message if the architecture does not match.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # Handle common checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    state_dict = clean_state_dict(
        state_dict
    )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as e:

        raise RuntimeError(
            "\n\n"
            "Checkpoint architecture mismatch.\n\n"
            f"Checkpoint:\n{checkpoint_path}\n\n"
            f"Model class:\n"
            f"{model.__class__.__name__}\n\n"
            f"Original error:\n{e}\n"
        )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# GRU5
# ============================================================

class GRU5(nn.Module):
    """
    GRU5:

        input_size = 80
        hidden_size = 256
        num_layers = 2
        dropout = 0.2

    Checkpoint output layer:
        output.weight
        output.bias
    """

    def __init__(
        self,
        input_size=80,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.gru = nn.GRU(
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

    def forward(self, x):

        output, _ = self.gru(x)

        logits = self.output(
            output
        )

        logits = logits.squeeze(-1)

        return logits


# ============================================================
# LSTM4
# ============================================================

class LSTM4(nn.Module):
    """
    LSTM4:

        input_size = 80
        hidden_size = 128
        num_layers = 2
        dropout = 0.2

    Checkpoint output layer:
        output.weight
        output.bias
    """

    def __init__(
        self,
        input_size=80,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
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

    def forward(self, x):

        output, _ = self.lstm(x)

        logits = self.output(
            output
        )

        logits = logits.squeeze(-1)

        return logits


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    IMPORTANT:
    The TR-1 checkpoint contains:

        positional_encoding.pe

    and its tensor shape is:

        [1, 512, 64]

    Therefore max_len MUST be 512.
    """

    def __init__(
        self,
        d_model,
        max_len=512,
    ):

        super().__init__()

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
            * (
                -np.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            1,
            max_len,
            d_model,
        )

        pe[:, :, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, :, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        sequence_length = x.size(1)

        if sequence_length > self.pe.size(1):

            raise ValueError(
                f"Sequence length "
                f"{sequence_length} exceeds "
                f"maximum supported length "
                f"{self.pe.size(1)}."
            )

        return (
            x
            + self.pe[
                :,
                :sequence_length,
                :
            ]
        )


# ============================================================
# TR1 CAUSAL TRANSFORMER
# ============================================================

class CausalTransformer(nn.Module):
    """
    TR1:

        input_size = 80
        d_model = 64
        nhead = 4
        num_layers = 2
        dim_feedforward = 128
        dropout = 0.1
        max_len = 512

    Architecture names match checkpoint:

        input_projection
        positional_encoding
        encoder
        dropout
        output_layer

    Strict causal attention prevents future information
    from being used at the current timestep.
    """

    def __init__(
        self,
        input_size=80,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=512,
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                max_len=max_len,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.output_layer = nn.Linear(
            d_model,
            1,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        x = self.input_projection(x)

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        x = self.positional_encoding(x)

        # ----------------------------------------------------
        # Strict causal attention mask
        #
        # True means that position is masked.
        #
        # Future positions are masked:
        #
        # t cannot see t+1, t+2, ...
        # ----------------------------------------------------

        seq_len = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        x = self.encoder(
            x,
            mask=causal_mask,
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        x = self.dropout(x)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        logits = self.output_layer(
            x
        )

        logits = logits.squeeze(-1)

        return logits


# ============================================================
# LOAD TRAINING MEDIANS
# ============================================================

@st.cache_data
def load_training_medians():

    if not MEDIAN_FILE.exists():

        raise FileNotFoundError(
            "Training median file not found:\n\n"
            f"{MEDIAN_FILE}"
        )

    medians_df = pd.read_csv(
        MEDIAN_FILE
    )

    if "feature" not in medians_df.columns:

        raise ValueError(
            "training_medians.csv does not "
            "contain a 'feature' column."
        )

    possible_value_columns = [
        "training_median_after_causal_ffill",
        "median",
    ]

    value_column = None

    for column in possible_value_columns:

        if column in medians_df.columns:

            value_column = column
            break

    if value_column is None:

        raise ValueError(
            "Could not find median value column.\n\n"
            "Expected one of:\n"
            "training_median_after_causal_ffill\n"
            "median"
        )

    medians = (
        medians_df
        .set_index("feature")
        [value_column]
        .reindex(BASE_FEATURES)
    )

    missing_features = (
        medians[
            medians.isna()
        ]
        .index
        .tolist()
    )

    if missing_features:

        raise ValueError(
            "Training medians missing for:\n"
            + "\n".join(
                missing_features
            )
        )

    return medians.astype(
        float
    )


# ============================================================
# CAUSAL PREPROCESSING
# ============================================================

def preprocess_patient(
    raw_df,
    training_medians,
):
    """
    F0 representation:

        40 raw variables
        +
        40 missingness indicators
        =
        80 features

    Preprocessing:

        1. Select original 40 variables
        2. Convert to numeric
        3. Convert -1 -> NaN
        4. Create missingness indicators
        5. Causal forward fill
        6. Fill remaining missing values with
           training-set medians

    No future patient rows are used.
    """

    df = raw_df.copy()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        feature
        for feature in BASE_FEATURES
        if feature not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Uploaded PSV is missing required "
            "columns:\n\n"
            + ", ".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    values = df[
        BASE_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for feature in BASE_FEATURES:

        values[feature] = pd.to_numeric(
            values[feature],
            errors="coerce",
        )

    # --------------------------------------------------------
    # PhysioNet missing convention
    # --------------------------------------------------------

    values = values.replace(
        -1,
        np.nan,
    )

    # --------------------------------------------------------
    # Missingness indicators BEFORE imputation
    # --------------------------------------------------------

    missingness = (
        values
        .isna()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Causal forward fill
    # --------------------------------------------------------

    values = values.ffill()

    # --------------------------------------------------------
    # Training-only median imputation
    # --------------------------------------------------------

    values = values.fillna(
        training_medians
    )

    # --------------------------------------------------------
    # Ensure ordering
    # --------------------------------------------------------

    values = values[
        BASE_FEATURES
    ]

    missingness = missingness[
        BASE_FEATURES
    ]

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    values_array = values.to_numpy(
        dtype=np.float32
    )

    missingness_array = (
        missingness.to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # F0 = values + missingness
    # --------------------------------------------------------

    X = np.concatenate(
        [
            values_array,
            missingness_array,
        ],
        axis=1,
    )

    return X


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    # --------------------------------------------------------
    # GRU5
    # --------------------------------------------------------

    gru = GRU5(
        input_size=80,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    )

    gru = load_checkpoint(
        gru,
        GRU_CHECKPOINT,
    )

    models["GRU5"] = gru

    # --------------------------------------------------------
    # LSTM4
    # --------------------------------------------------------

    lstm = LSTM4(
        input_size=80,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    )

    lstm = load_checkpoint(
        lstm,
        LSTM_CHECKPOINT,
    )

    models["LSTM4"] = lstm

    # --------------------------------------------------------
    # TR1
    # --------------------------------------------------------

    transformer = CausalTransformer(
        input_size=80,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,

        # IMPORTANT:
        # Checkpoint has:
        # positional_encoding.pe
        # shape [1, 512, 64]
        max_len=512,
    )

    transformer = load_checkpoint(
        transformer,
        TRANSFORMER_CHECKPOINT,
    )

    models["TR1"] = transformer

    return models


# ============================================================
# CAUSAL INFERENCE
# ============================================================

def run_causal_inference(
    X,
    models,
):
    """
    At hour t, only rows 0:t are supplied.

    Future rows are never supplied to the model.

    The probability for hour t is the model's output
    at the final timestep t.
    """

    sequence_length = X.shape[0]

    results = {
        "GRU5": [],
        "LSTM4": [],
        "TR1": [],
    }

    progress = st.progress(
        0,
        text="Running causal inference...",
    )

    for t in range(
        sequence_length
    ):

        current_sequence = X[
            : t + 1
        ]

        tensor = torch.tensor(
            current_sequence,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            for model_name, model in (
                models.items()
            ):

                logits = model(
                    tensor
                )

                current_logit = logits[
                    0,
                    -1,
                ]

                probability = (
                    torch.sigmoid(
                        current_logit
                    )
                    .item()
                )

                results[
                    model_name
                ].append(
                    probability
                )

        progress.progress(
            (t + 1)
            / sequence_length,
            text=(
                "Running causal inference: "
                f"hour {t + 1}/"
                f"{sequence_length}"
            ),
        )

    progress.empty()

    return {
        key: np.asarray(
            value,
            dtype=np.float32,
        )
        for key, value in results.items()
    }


# ============================================================
# OR DECISION
# ============================================================

def calculate_or_decision(
    probabilities
):

    gru_positive = (
        probabilities["GRU5"]
        >= GRU_THRESHOLD
    )

    lstm_positive = (
        probabilities["LSTM4"]
        >= LSTM_THRESHOLD
    )

    transformer_positive = (
        probabilities["TR1"]
        >= TRANSFORMER_THRESHOLD
    )

    any_positive = (
        gru_positive
        | lstm_positive
        | transformer_positive
    )

    return {
        "GRU5": gru_positive,
        "LSTM4": lstm_positive,
        "TR1": transformer_positive,
        "OR": any_positive,
    }


# ============================================================
# FIRST ALERT
# ============================================================

def get_first_alert(
    decisions
):

    alert_indices = np.where(
        decisions["OR"]
    )[0]

    if len(alert_indices) == 0:

        return None

    return int(
        alert_indices[0]
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫀 X-FedSepsis"
)

st.subheader(
    "Causal Multi-Architecture Sepsis Prediction"
)

st.markdown(
    """
Upload a **single PhysioNet/CinC Sepsis Challenge `.psv`
patient file**.

The system processes the trajectory **hour by hour**
using three trained temporal architectures:

- **GRU5**
- **LSTM4**
- **TR1 Causal Transformer**

The demonstration uses an **OR decision rule**.
"""
)

st.warning(
    """
**Research prototype only.**

This system is not a medical device and must not be
used for clinical diagnosis or treatment decisions.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        f"Device: `{DEVICE}`"
    )

    st.write(
        f"Input features: `{INPUT_SIZE}`"
    )

    st.write(
        "Representation: `F0`"
    )

    st.write(
        "Inference: `Strictly causal`"
    )

    st.divider()

    st.subheader(
        "Frozen thresholds"
    )

    st.write(
        f"GRU5: `{GRU_THRESHOLD:.2f}`"
    )

    st.write(
        f"LSTM4: `{LSTM_THRESHOLD:.2f}`"
    )

    st.write(
        f"TR1: `{TRANSFORMER_THRESHOLD:.2f}`"
    )

    st.write(
        "Decision: `OR`"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a patient PSV file",
    type=["psv"],
)


if uploaded_file is None:

    st.info(
        "Upload a `.psv` file to begin."
    )

    st.stop()


# ============================================================
# READ PSV
# ============================================================

try:

    raw_df = pd.read_csv(
        uploaded_file,
        sep="|",
    )

except Exception as e:

    st.error(
        f"Could not read PSV file:\n\n{e}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(raw_df) == 0:

    st.error(
        "Uploaded PSV contains no rows."
    )

    st.stop()


# ============================================================
# DISPLAY DATA
# ============================================================

st.success(
    "Loaded patient trajectory with "
    f"**{len(raw_df)} hourly rows**."
)

with st.expander(
    "View uploaded patient data"
):

    st.dataframe(
        raw_df,
        use_container_width=True,
    )


# ============================================================
# LOAD MEDIANS
# ============================================================

try:

    training_medians = (
        load_training_medians()
    )

except Exception as e:

    st.error(
        "Failed to load training medians."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREPROCESS
# ============================================================

try:

    X = preprocess_patient(
        raw_df,
        training_medians,
    )

except Exception as e:

    st.error(
        "Preprocessing failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# INPUT CHECK
# ============================================================

if X.shape[1] != INPUT_SIZE:

    st.error(
        "Unexpected model input dimension.\n\n"
        f"Expected: {INPUT_SIZE}\n"
        f"Received: {X.shape[1]}"
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    with st.spinner(
        "Loading trained models..."
    ):

        models = load_models()

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


st.success(
    "All three trained models loaded successfully."
)


# ============================================================
# RUN BUTTON
# ============================================================

run_prediction = st.button(
    "🚀 Run Causal Sepsis Prediction",
    type="primary",
    use_container_width=True,
)


if not run_prediction:

    st.info(
        "Click **Run Causal Sepsis Prediction** "
        "to process the patient trajectory."
    )

    st.stop()


# ============================================================
# INFERENCE
# ============================================================

try:

    probabilities = (
        run_causal_inference(
            X,
            models,
        )
    )

except Exception as e:

    st.error(
        "Inference failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DECISIONS
# ============================================================

decisions = calculate_or_decision(
    probabilities
)

first_alert = get_first_alert(
    decisions
)


# ============================================================
# FINAL STATUS
# ============================================================

st.divider()

if first_alert is not None:

    st.error(
        "🚨 SEPSIS ALERT — "
        f"first alert at hour "
        f"**{first_alert + 1}**"
    )

    st.write(
        "At least one model crossed its frozen "
        "validation threshold."
    )

else:

    st.success(
        "✅ NO ALERT during the observed trajectory."
    )

    st.write(
        "None of the three models crossed its "
        "frozen validation threshold."
    )


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader(
    "Model Status"
)

latest_index = (
    len(raw_df) - 1
)

col1, col2, col3 = st.columns(
    3
)


# ------------------------------------------------------------
# GRU5
# ------------------------------------------------------------

with col1:

    latest_gru = probabilities[
        "GRU5"
    ][latest_index]

    st.metric(
        "GRU5 probability",
        f"{latest_gru:.3f}",
    )

    if latest_gru >= GRU_THRESHOLD:

        st.error(
            f"POSITIVE ≥ "
            f"{GRU_THRESHOLD:.2f}"
        )

    else:

        st.success(
            f"NEGATIVE < "
            f"{GRU_THRESHOLD:.2f}"
        )


# ------------------------------------------------------------
# LSTM4
# ------------------------------------------------------------

with col2:

    latest_lstm = probabilities[
        "LSTM4"
    ][latest_index]

    st.metric(
        "LSTM4 probability",
        f"{latest_lstm:.3f}",
    )

    if latest_lstm >= LSTM_THRESHOLD:

        st.error(
            f"POSITIVE ≥ "
            f"{LSTM_THRESHOLD:.2f}"
        )

    else:

        st.success(
            f"NEGATIVE < "
            f"{LSTM_THRESHOLD:.2f}"
        )


# ------------------------------------------------------------
# TR1
# ------------------------------------------------------------

with col3:

    latest_tr = probabilities[
        "TR1"
    ][latest_index]

    st.metric(
        "TR1 probability",
        f"{latest_tr:.3f}",
    )

    if latest_tr >= TRANSFORMER_THRESHOLD:

        st.error(
            f"POSITIVE ≥ "
            f"{TRANSFORMER_THRESHOLD:.2f}"
        )

    else:

        st.success(
            f"NEGATIVE < "
            f"{TRANSFORMER_THRESHOLD:.2f}"
        )


# ============================================================
# PROBABILITY TRAJECTORY
# ============================================================

st.divider()

st.subheader(
    "Causal Probability Trajectories"
)

hours = np.arange(
    1,
    len(raw_df) + 1,
)


fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    hours,
    probabilities["GRU5"],
    label="GRU5",
)

ax.plot(
    hours,
    probabilities["LSTM4"],
    label="LSTM4",
)

ax.plot(
    hours,
    probabilities["TR1"],
    label="TR1",
)

ax.axhline(
    GRU_THRESHOLD,
    linestyle="--",
    label="GRU5 threshold",
)

ax.axhline(
    LSTM_THRESHOLD,
    linestyle="--",
    label="LSTM4 threshold",
)

ax.axhline(
    TRANSFORMER_THRESHOLD,
    linestyle="--",
    label="TR1 threshold",
)

if first_alert is not None:

    ax.axvline(
        first_alert + 1,
        linestyle=":",
        linewidth=2,
        label="First OR alert",
    )

ax.set_xlabel(
    "Hour"
)

ax.set_ylabel(
    "Predicted sepsis probability"
)

ax.set_title(
    "Strictly Causal Model Predictions"
)

ax.set_ylim(
    0,
    1,
)

ax.grid(
    alpha=0.25
)

ax.legend()

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)


# ============================================================
# HOUR-BY-HOUR TABLE
# ============================================================

st.divider()

st.subheader(
    "Hour-by-Hour Decisions"
)

decision_df = pd.DataFrame(
    {
        "Hour": hours,

        "GRU5 Probability":
            probabilities["GRU5"],

        "LSTM4 Probability":
            probabilities["LSTM4"],

        "TR1 Probability":
            probabilities["TR1"],

        "GRU5 Positive":
            decisions["GRU5"],

        "LSTM4 Positive":
            decisions["LSTM4"],

        "TR1 Positive":
            decisions["TR1"],

        "OR Alert":
            decisions["OR"],
    }
)

st.dataframe(
    decision_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ALERT DETAILS
# ============================================================

st.divider()

st.subheader(
    "Alert Details"
)

if first_alert is not None:

    alert_hour = first_alert

    alerting_models = []

    if decisions["GRU5"][alert_hour]:
        alerting_models.append(
            "GRU5"
        )

    if decisions["LSTM4"][alert_hour]:
        alerting_models.append(
            "LSTM4"
        )

    if decisions["TR1"][alert_hour]:
        alerting_models.append(
            "TR1"
        )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        st.metric(
            "First alert hour",
            alert_hour + 1,
        )

    with c2:

        st.metric(
            "Alerting models",
            len(alerting_models),
        )

    with c3:

        st.write(
            "**Models:**"
        )

        for model_name in (
            alerting_models
        ):

            st.write(
                f"- {model_name}"
            )

else:

    st.info(
        "No model crossed its threshold."
    )


# ============================================================
# THRESHOLD TABLE
# ============================================================

st.divider()

st.subheader(
    "Frozen Model Thresholds"
)

threshold_df = pd.DataFrame(
    {
        "Model": [
            "GRU5",
            "LSTM4",
            "TR1",
        ],

        "Validation Threshold": [
            GRU_THRESHOLD,
            LSTM_THRESHOLD,
            TRANSFORMER_THRESHOLD,
        ],

        "Latest Probability": [
            probabilities["GRU5"][-1],
            probabilities["LSTM4"][-1],
            probabilities["TR1"][-1],
        ],

        "Latest Decision": [

            (
                "POSITIVE"
                if probabilities["GRU5"][-1]
                >= GRU_THRESHOLD
                else "NEGATIVE"
            ),

            (
                "POSITIVE"
                if probabilities["LSTM4"][-1]
                >= LSTM_THRESHOLD
                else "NEGATIVE"
            ),

            (
                "POSITIVE"
                if probabilities["TR1"][-1]
                >= TRANSFORMER_THRESHOLD
                else "NEGATIVE"
            ),
        ],
    }
)

st.dataframe(
    threshold_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "Technical details"
):

    st.markdown(
        f"""
### F0 Representation

- 40 original PhysioNet variables
- 40 missingness indicators
- Total: **80 features**

### Preprocessing

1. `-1` → missing
2. Missingness indicators generated
3. Within-patient causal forward fill
4. Remaining missing values filled using
   training-set medians

### GRU5

- Hidden size: 256
- Layers: 2
- Dropout: 0.2

### LSTM4

- Hidden size: 128
- Layers: 2
- Dropout: 0.2

### TR1

- d_model: 64
- Attention heads: 4
- Transformer layers: 2
- Feed-forward dimension: 128
- Dropout: 0.1
- Positional encoding length: **512**
- Strict causal attention mask

### Decision rule

GRU5 ≥ {GRU_THRESHOLD}

OR

LSTM4 ≥ {LSTM_THRESHOLD}

OR

TR1 ≥ {TRANSFORMER_THRESHOLD}

→ **SEPSIS ALERT**

### Causality

At hour `t`, only observations through hour `t`
are provided to the model.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "X-FedSepsis — Research Prototype | "
    "Not for clinical diagnosis"
)