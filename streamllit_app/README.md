# X-FedSepsis Streamlit Early-Warning Demo

## Install

From the regular project directory:

```powershell
cd D:\CAPSTONE\SEPSIS
pip install -r streamlit_app\requirements.txt
```

## Run

```powershell
streamlit run streamlit_app\app.py
```

## Expected project layout

The app expects the existing project to contain:

```text
SEPSIS/
├── checkpoints/
│   ├── gru_family/gru-5.pt
│   ├── lstm_family/lstm-4.pt
│   └── transformer_family/TR-1.pt
├── data/
│   └── processed/
│       └── training medians artifact
├── src/
└── streamlit_app/
    ├── app.py
    └── requirements.txt
```

The app searches for the training-only median artifact at:

- `data/processed/training_medians.npy`
- `data/processed/training_medians.npz`
- `data/processed/medians.npy`
- `data/processed/medians.npz`
- `results/preprocessing_result/training_medians.npy`
- `results/preprocessing_result/training_medians.npz`

If the actual project uses a different filename, update `locate_training_medians()` in `app.py` to point to that existing artifact. **Do not calculate medians from the uploaded patient.**

## Behavior

The uploaded `.psv` is processed causally. At hour `t`, only rows `0:t` are passed to the model.

The UI evaluates:

- GRU-5
- LSTM-4
- TR1 causal Transformer

and applies the frozen thresholds:

- GRU-5: 0.86
- LSTM-4: 0.84
- TR1: 0.80

The decision rule is OR: if any loaded model crosses its threshold at the current hour, the patient is flagged.

The first alert hour is shown, along with all model probability trajectories.

## Important research note

This is a demonstration system based on the project models. It is not clinically validated and must not be used to diagnose or treat a patient.
