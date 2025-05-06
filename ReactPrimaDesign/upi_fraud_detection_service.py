"""
UPI Fraud Detection Service
FastAPI service that provides fraud risk assessment for UPI QR codes
"""

import os
import time
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn

from upi_fraud_detection_model import predict_fraud_risk, train_model

# Create FastAPI app
app = FastAPI(title="UPI Fraud Detection Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QRRequest(BaseModel):
    qr_text: str
    additional_info: Optional[Dict[str, Any]] = None

# In-memory cache for quick lookups
prediction_cache = {}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to measure and log request processing time"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "service": "UPI Fraud Detection Service",
        "model_ready": os.path.exists("upi_fraud_model.joblib")
    }

@app.post("/predict")
async def predict(request: QRRequest):
    """
    Predict fraud risk for a QR code
    
    Args:
        qr_text: The QR code text (must be a UPI QR code)
        additional_info: Optional additional information about the transaction
        
    Returns:
        Fraud risk assessment with detailed explanation
    """
    start_time = time.time()
    
    # Check cache for repeated requests
    if request.qr_text in prediction_cache:
        result = prediction_cache[request.qr_text].copy()
        result["from_cache"] = True
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    try:
        # Call prediction function
        result = predict_fraud_risk(request.qr_text)
        
        # Add metadata
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        result["from_cache"] = False
        
        # Cache result for future use (up to 100 items)
        if len(prediction_cache) < 100:
            prediction_cache[request.qr_text] = result.copy()
        
        return result
    
    except Exception as e:
        # Provide a fallback on error
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "Error processing UPI QR code",
                "risk_level": "Medium",  # Default to medium on error
                "risk_score": 50,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        )

@app.post("/train")
async def train():
    """
    Train or retrain the fraud detection model
    
    Returns:
        Training result status
    """
    try:
        train_model()
        # Clear cache after retraining
        prediction_cache.clear()
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error training model: {str(e)}"}
        )

@app.post("/feedback")
async def feedback(qr_text: str, is_fraud: bool, explanation: Optional[str] = None):
    """
    Submit feedback to improve the model
    
    Args:
        qr_text: The QR code text
        is_fraud: Whether the QR code is fraudulent
        explanation: Optional explanation for the feedback
        
    Returns:
        Feedback submission status
    """
    # In a production system, this would store feedback for later model retraining
    # For this demo, we'll just acknowledge the feedback
    return {
        "status": "success",
        "message": "Feedback recorded for future model improvement",
        "qr_text": qr_text[:50] + "..." if len(qr_text) > 50 else qr_text,
        "is_fraud": is_fraud
    }

@app.get("/stats")
async def stats():
    """
    Get service statistics
    
    Returns:
        Service statistics including cache size, model info, etc.
    """
    model_exists = os.path.exists("upi_fraud_model.joblib")
    model_size = os.path.getsize("upi_fraud_model.joblib") if model_exists else 0
    
    scaler_exists = os.path.exists("upi_scaler.joblib")
    scaler_size = os.path.getsize("upi_scaler.joblib") if scaler_exists else 0
    
    return {
        "cache_size": len(prediction_cache),
        "model_exists": model_exists,
        "model_size_bytes": model_size,
        "scaler_exists": scaler_exists,
        "scaler_size_bytes": scaler_size,
        "service_uptime": "Unknown",  # Would track this in a real service
    }

def start_server():
    """Start the FastAPI server"""
    # Get port from environment or use default
    port = int(os.environ.get("UPI_FRAUD_PORT", 5050))
    
    # Check if model exists, train if not
    if not os.path.exists("upi_fraud_model.joblib"):
        print("No model found. Training a new model...")
        train_model()
    else:
        print("UPI fraud detection model already exists")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_server()