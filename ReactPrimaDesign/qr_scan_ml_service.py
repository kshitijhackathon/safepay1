"""
REAL-TIME QR SCAN SYSTEM (<100ms Latency)
Replit-Optimized with River ML + FastAPI
"""

from fastapi import FastAPI, Request, middleware
from river import compose, linear_model, optim, preprocessing, metrics, drift
from pydantic import BaseModel
import uvicorn
import numpy as np
import re
import time
import json
import asyncio
import os
from collections import deque
from diskcache import Cache
from typing import Optional, Dict, List, Any

# Optimized regex patterns
UPI_REGEX = re.compile(r'^upi://[^\s/$.?#].[^\s]*$', re.IGNORECASE)

# Ultra-Fast Feature Engineering
feature_pipeline = compose.Pipeline(
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(
        optimizer=optim.SGD(0.01),
        loss=optim.losses.Log()
    )
)

# Online Learning Model
model = compose.Pipeline(
    preprocessing.TargetStandardScaler(
        regressor=linear_model.LinearRegression()
    ),
    feature_pipeline
)

# Caching System for Frequent Patterns
cache = Cache('./prediction_cache')

# Model Sharding
model_shards = {
    'in': linear_model.LogisticRegression(),
    'global': linear_model.PAClassifier()
}

app = FastAPI()

class QRRequest(BaseModel):
    qr_text: str

class BatchProcessor:
    def __init__(self, max_batch_size=32, max_delay=0.05):
        self.batch = deque()
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay
    
    async def process(self, qr_text):
        self.batch.append(qr_text)
        if len(self.batch) >= self.max_batch_size:
            return self.predict_batch()
        await asyncio.sleep(self.max_delay)
        return self.predict_batch()
    
    def predict_batch(self):
        # Process all QR codes in batch
        results = []
        while self.batch:
            qr_text = self.batch.popleft()
            features = extract_features(qr_text)
            prediction = model.predict_one(features)
            proba = model.predict_proba_one(features).get(1, 0.0)
            results.append({
                "qr_text": qr_text,
                "risk_score": round(proba * 100, 2),
                "prediction": prediction
            })
        return results

def is_valid_upi(qr_text):
    """Check if QR text contains valid UPI format"""
    return bool(UPI_REGEX.match(qr_text))

def route_model(qr_text):
    """Route to appropriate model shard based on QR content"""
    return 'in' if '.in/' in qr_text else 'global'

def extract_features(qr_text: str) -> dict:
    """Real-time feature extraction (<5ms)"""
    features = {
        'length': len(qr_text),
        'entropy': -sum((count/len(qr_text)) * np.log2(count/len(qr_text)) 
                     for count in [qr_text.count(c) for c in set(qr_text)] if count > 0),
        'has_upi': int('upi://' in qr_text),
        'num_params': qr_text.count('&'),
        'suspicious_keywords': sum(1 for kw in ['urgent', 'payment', 'offer'] 
                                if kw in qr_text.lower())
    }
    
    # Extract additional features for UPI QR codes
    if features['has_upi']:
        # Check for valid UPI format
        features['valid_upi_format'] = int(is_valid_upi(qr_text))
        
        # Extract payment parameters
        if 'pa=' in qr_text:
            try:
                pa_value = re.search(r'pa=([^&]+)', qr_text).group(1)
                features['pa_length'] = len(pa_value)
                features['has_valid_pa'] = int('@' in pa_value and '.' in pa_value)
            except:
                features['pa_length'] = 0
                features['has_valid_pa'] = 0
        else:
            features['pa_length'] = 0
            features['has_valid_pa'] = 0
            
        # Check for amount parameter
        features['has_amount'] = int('am=' in qr_text)
    
    return features

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to measure and log request processing time"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response

@app.get("/")
async def root():
    """Service health check endpoint"""
    return {"status": "online", "service": "QR ML Scanner"}

@app.post("/predict")
async def predict(request: QRRequest):
    """Predict risk score for a QR code"""
    start = time.time()
    
    # Check Cache First
    if request.qr_text in cache:
        cached_result = cache[request.qr_text]
        return {**cached_result, "cached": True, "latency_ms": (time.time()-start)*1000}
    
    # Feature Extraction
    features = extract_features(request.qr_text)
    
    # Model Selection
    shard = route_model(request.qr_text)
    
    # Model Prediction 
    prediction = model.predict_one(features)
    proba = model.predict_proba_one(features).get(1, 0.0)
    
    # Security Rules Enhancement
    if features['suspicious_keywords'] > 2:
        proba = min(proba + 0.3, 1.0)
    
    # Lower risk for valid UPI with proper parameters
    if features['has_upi'] and features['has_valid_pa'] and features['has_amount']:
        proba = max(proba - 0.1, 0.0)
    
    # Format result
    result = {
        "risk_score": round(proba * 100, 2),
        "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low",
        "features": features,
        "recommendation": "Block" if proba > 0.7 else "Verify" if proba > 0.3 else "Allow",
        "latency_ms": round((time.time() - start) * 1000, 2)
    }
    
    # Cache Frequent Queries (expire after 5 minutes)
    if features['has_upi']:
        cache.set(request.qr_text, result, expire=300)
    
    return result

@app.post("/feedback")
async def feedback(qr_text: str, is_scam: bool):
    """Process user feedback to improve the model"""
    features = extract_features(qr_text)
    
    # Update both the general model and the appropriate shard
    model.learn_one(features, int(is_scam))
    shard = route_model(qr_text)
    model_shards[shard].learn_one(features, int(is_scam))
    
    # Clear the cache for this QR code
    if qr_text in cache:
        del cache[qr_text]
    
    return {"status": "model_updated", "shard": shard}

@app.post("/batch_predict")
async def batch_predict(requests: List[QRRequest]):
    """Process a batch of QR codes together"""
    results = []
    start = time.time()
    
    for req in requests:
        # Check cache
        if req.qr_text in cache:
            results.append({**cache[req.qr_text], "cached": True})
            continue
        
        # Generate features
        features = extract_features(req.qr_text)
        proba = model.predict_proba_one(features).get(1, 0.0)
        
        # Apply business rules
        if features['suspicious_keywords'] > 2:
            proba = min(proba + 0.3, 1.0)
        
        result = {
            "qr_text": req.qr_text,
            "risk_score": round(proba * 100, 2),
            "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low",
            "features": features,
            "cached": False
        }
        
        # Cache result
        if features['has_upi']:
            cache.set(req.qr_text, result, expire=300)
            
        results.append(result)
    
    batch_time = (time.time() - start) * 1000
    average_time = batch_time / len(requests) if requests else 0
    
    return {
        "results": results,
        "batch_size": len(results),
        "total_time_ms": round(batch_time, 2),
        "average_time_ms": round(average_time, 2)
    }

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Fallback to rule-based risk assessment on model failure"""
    # Try to extract the QR text from the request
    try:
        body = await request.json()
        qr_text = body.get("qr_text", "")
    except:
        qr_text = ""
    
    # Simple rule-based assessment
    risk_score = 0
    if 'upi://' in qr_text:
        if not is_valid_upi(qr_text):
            risk_score = 80  # High risk for invalid UPI format
        else:
            # Check for suspicious signs
            suspicious_count = sum(1 for kw in ['urgent', 'payment', 'offer'] 
                                  if kw in qr_text.lower())
            risk_score = min(30 + suspicious_count * 20, 100)
    
    return {
        "error": "Model prediction failed, using fallback rules",
        "risk_score": risk_score,
        "risk_level": "High" if risk_score > 70 else "Medium" if risk_score > 30 else "Low",
        "fallback": True
    }

if __name__ == "__main__":
    port = int(os.environ.get("ML_QR_SERVICE_PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)