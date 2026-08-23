"""
api.py

FastAPI backend for DeepShieldIDS.
Loads the trained HybridIDSNet model and predicts
BENIGN / BruteForce / DoS / Probe / WebAttack.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hybridids_best.pt"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_test_scaler.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_test_label_encoder.pkl"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] Using device: {DEVICE}")


# ============================================================
# LOAD SCALER AND LABEL ENCODER
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Label encoder not found: {ENCODER_PATH}"
    )


scaler = joblib.load(SCALER_PATH)

label_encoder = joblib.load(ENCODER_PATH)


CLASS_NAMES = list(
    label_encoder.classes_
)


print(
    f"[INFO] Classes: {CLASS_NAMES}"
)


# ============================================================
# MODEL
# ============================================================

# CIC-IDS2017 has 78 features
SEQ_LEN = 78

N_CLASSES = len(CLASS_NAMES)


model = build_model(
    seq_len=SEQ_LEN,
    n_classes=N_CLASSES,
    device=DEVICE
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model.eval()


print("[INFO] HybridIDSNet loaded successfully")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DeepShieldIDS API",
    description="Network Intrusion Detection API using HybridIDSNet",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    features: list[float]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "DeepShieldIDS API Running",
        "model": "HybridIDSNet",
        "classes": CLASS_NAMES
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        features = request.features


        # ----------------------------------------------------
        # Check feature count
        # ----------------------------------------------------

        if len(features) != SEQ_LEN:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected {SEQ_LEN} features, "
                    f"but received {len(features)}."
                )
            )


        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        X = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        X = scaler.transform(X)


        # ----------------------------------------------------
        # Shape for HybridIDSNet
        #
        # Model expects:
        # (batch, seq_len, 1)
        # ----------------------------------------------------

        X = np.expand_dims(
            X,
            axis=2
        )


        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(DEVICE)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                X_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]


        confidence = float(
            probabilities[0, predicted_index].item()
        )


        probability_dict = {}

        for i, class_name in enumerate(CLASS_NAMES):

            probability_dict[class_name] = float(
                probabilities[0, i].item()
            )


        return {

            "prediction": predicted_label,

            "confidence": round(
                confidence,
                4
            ),

            "probabilities": probability_dict

        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            f"[ERROR] Prediction failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):

    try:

        if len(requests) == 0:

            raise HTTPException(
                status_code=400,
                detail="No prediction data received."
            )


        if len(requests) > 1000:

            raise HTTPException(
                status_code=400,
                detail="Maximum 1000 records allowed per request."
            )


        all_features = []


        for request in requests:

            if len(request.features) != SEQ_LEN:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Each record must contain "
                        f"{SEQ_LEN} features."
                    )
                )

            all_features.append(
                request.features
            )


        X = np.array(
            all_features,
            dtype=np.float32
        )


        X = scaler.transform(X)


        X = np.expand_dims(
            X,
            axis=2
        )


        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(DEVICE)


        with torch.no_grad():

            logits = model(
                X_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )


        results = []


        for i in range(len(predictions)):

            predicted_index = int(
                predictions[i].item()
            )


            predicted_label = label_encoder.inverse_transform(
                [predicted_index]
            )[0]


            confidence = float(
                probabilities[
                    i,
                    predicted_index
                ].item()
            )


            results.append({

                "prediction": predicted_label,

                "confidence": round(
                    confidence,
                    4
                )

            })


        return {

            "count": len(results),

            "results": results

        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            f"[ERROR] Batch prediction failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )