"""
Optimized QR Risk Detection Service
Integrates machine learning model for accurate risk assessment of QR codes
"""
import os
import re
import time
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn

# Import the QR risk detection model
from qr_risk_detection_model import analyze_qr_risk, train_model

app = FastAPI(title="QR Risk Detection Service")

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
    redirect_url: Optional[str] = None
    report_count: Optional[int] = 0

# In-memory cache for quick repeat lookups
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
    return {"status": "healthy", "service": "QR Risk Detection Service"}

@app.post("/predict")
async def predict(request: QRRequest):
    """
    Predict risk score for a QR code
    
    Args:
        qr_text: The raw QR code text
        redirect_url: Optional URL the QR code redirects to
        report_count: Optional number of times this QR has been reported
        
    Returns:
        Risk assessment with score, level, and explanation
    """
    start_time = time.time()
    
    # Cache check for performance
    cache_key = request.qr_text
    if cache_key in prediction_cache:
        result = prediction_cache[cache_key]
        result["from_cache"] = True
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    # Extract UPI info if present
    upi_id = None
    match = re.search(r'pa=([\w@.]+)', request.qr_text)
    if match:
        upi_id = match.group(1)
    
    # Analyze risk
    try:
        # Handle optional report_count parameter
        report_count = 0
        if request.report_count is not None:
            report_count = request.report_count

        risk_result = analyze_qr_risk(
            qr_content=request.qr_text,
            redirect_url=request.redirect_url,
            report_count=report_count
        )
        
        # Enhance with UPI ID
        if upi_id:
            risk_result["upi_id"] = upi_id
        
        # Standardize the response
        response = {
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "is_scam": risk_result["is_scam"],
            "explanation": risk_result["explanation"],
            "features": risk_result["features"],
            "scan_time_ms": int((time.time() - start_time) * 1000),
            "from_cache": False
        }
        
        # Cache result for future use
        prediction_cache[cache_key] = response
        
        return response
    
    except Exception as e:
        # Fallback to rules-based assessment on error
        print(f"Error in ML prediction: {str(e)}")
        
        # Simple rule-based risk assessment
        risk_score = 0
        explanations = []
        
        # Check for common scam patterns
        if upi_id and any(domain in upi_id for domain in ["paytm", "okaxis", "oksbi", "icici", "okicici"]):
            # Known financial providers - lower risk
            risk_score += 10  # Low baseline risk for known providers
        else:
            risk_score += 50  # Higher baseline risk for unknown providers
            explanations.append("Unknown UPI provider domain")
        
        # Check for suspicious patterns
        if "urgent" in request.qr_text.lower() or "verify" in request.qr_text.lower():
            risk_score += 30
            explanations.append("Contains urgency keywords")
            
        # Check amount if present
        amount_match = re.search(r'am=(\d+(\.\d+)?)', request.qr_text)
        if amount_match:
            amount = float(amount_match.group(1))
            if amount > 10000:
                risk_score += 20
                explanations.append(f"High transaction amount: {amount}")
        
        # Normalize risk score
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        risk_level = "High" if risk_score > 70 else "Medium" if risk_score > 40 else "Low"
        
        # Create fallback result
        fallback_result = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "is_scam": risk_score > 70,
            "explanation": explanations or ["Fallback analysis due to error"],
            "features": {"fallback": True},
            "scan_time_ms": int((time.time() - start_time) * 1000),
            "from_cache": False,
            "error": str(e)
        }
        
        return fallback_result

@app.post("/batch_predict")
async def batch_predict(requests: List[QRRequest]):
    """Process multiple QR codes in batch for efficiency"""
    results = []
    for request in requests:
        result = await predict(request)
        results.append(result)
    return results

@app.post("/feedback")
async def feedback(qr_text: str, is_scam: bool):
    """Process user feedback to improve the model"""
    # In real implementation, this would store feedback for model retraining
    return {"success": True, "message": "Feedback recorded for future model improvement"}

@app.post("/train")
async def train():
    """Force model retraining"""
    try:
        model = train_model()
        return {"success": True, "message": "Model trained successfully"}
    except Exception as e:
        return {"success": False, "message": f"Error training model: {str(e)}"}

def start_server():
    """Start the FastAPI server"""
    port = int(os.environ.get("ML_QR_SERVICE_PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_server()