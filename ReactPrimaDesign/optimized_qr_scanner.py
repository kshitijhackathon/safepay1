# 🟢 REPLIT-FRIENDLY QR SCAN SYSTEM v2
# 🚀 Optimized Initial Training & UPI Detection

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re
import time
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from diskcache import Cache
import json
import os

# 📦 Pre-Trained Model Setup
MODEL_FILE = "model.joblib"
DATA_FILE = "qr_data.json"

def load_data():
    """Load training data with initial safe examples"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {
        'X': [
            # Initial safe UPI examples
            {'length': 45, 'has_upi': 1, 'num_params': 2, 'urgent': 0, 'payment': 0, 'currency': 1},
            {'length': 50, 'has_upi': 1, 'num_params': 3, 'urgent': 0, 'payment': 0, 'currency': 1},
            # Common scam patterns
            {'length': 80, 'has_upi': 0, 'num_params': 5, 'urgent': 1, 'payment': 1, 'currency': 0},
        ],
        'y': [0, 0, 1]  # Corresponding labels (0 = safe, 1 = scam)
    }

# 💾 Data Storage Functions
def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

# Initialize model with pre-trained data
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = make_pipeline(
        DictVectorizer(),
        SGDClassifier(loss='log_loss', class_weight='balanced')
    )
    # Train on initial data immediately
    initial_data = load_data()
    if initial_data['X']:
        model.fit(initial_data['X'], initial_data['y'])
        joblib.dump(model, MODEL_FILE)
        save_data(initial_data)  # Save initial dataset

# 🚄 Enhanced Feature Extraction
def extract_features(qr_text: str) -> dict:
    qr_lower = qr_text.lower()
    return {
        'length': min(len(qr_text), 100),
        'has_upi': int(re.search(r'^upi://', qr_lower) is not None),
        'num_params': qr_text.count('&'),
        'urgent': int('urgent' in qr_lower),
        'payment': int('payment' in qr_lower),
        'currency': int('inr' in qr_lower)  # New feature
    }

app = FastAPI()
cache = Cache('./cache')

class QRRequest(BaseModel):
    qr_text: str

@app.post("/predict")
async def predict(request: QRRequest):
    start = time.time()
    
    # 🔍 Cache First
    if request.qr_text in cache:
        return {**cache[request.qr_text], "cached": True}
    
    # ⚡ Feature Extraction
    features = extract_features(request.qr_text)
    
    # 🧠 Model Prediction
    try:
        proba = model.predict_proba([features])[0][1]
    except Exception as e:
        proba = 0.5  # Fallback for empty model
    
    risk_score = round(proba * 100, 2)
    
    # 🛡️ Enhanced Security Rules
    upi_safe_boost = 0
    if features['has_upi']:
        upi_safe_boost = -25  # UPI links get safety boost
        if features['payment'] and not features['urgent']:
            upi_safe_boost = -15  # Payment UPI less reduction
    
    risk_score = max(0, min(100, risk_score + upi_safe_boost))
    
    result = {
        "risk_score": risk_score,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "features": features  # For debugging
    }
    
    cache.set(request.qr_text, result, expire=300)
    return result

@app.post("/feedback")
async def feedback(qr_text: str, is_scam: bool):
    features = extract_features(qr_text)
    data = load_data()
    
    data['X'].append(features)
    data['y'].append(int(is_scam))
    
    # Retrain with every 25 new samples (improved frequency)
    if len(data['y']) % 25 == 0:
        model.fit(data['X'], data['y'])
        joblib.dump(model, MODEL_FILE)
        save_data(data)
    
    return {"status": "feedback_received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)