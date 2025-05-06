"""
Voice and Text Scam Detection Service
Provides FastAPI endpoints for ML-based detection of voice and text scams
"""

import os
import time
import json
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Import our ML models for voice and text scam detection
from voice_text_scam_model import (
    analyze_voice, analyze_text, batch_analyze_text, train_models
)

# Create the FastAPI app
app = FastAPI(title="Voice and Text Scam Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class VoiceAnalysisRequest(BaseModel):
    transcript: str
    audio_features: Optional[Dict[str, Any]] = None

class TextAnalysisRequest(BaseModel):
    text: str
    message_type: Optional[str] = "SMS"
    context: Optional[Dict[str, Any]] = None

class BatchTextAnalysisRequest(BaseModel):
    messages: List[Dict[str, Any]]

# Add processing time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to measure and report processing time"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Voice and Text Scam Detection API"}

@app.get("/status")
async def status():
    """Health check endpoint"""
    return {"status": "online", "service": "Voice and Text Scam Detection API"}

@app.post("/analyze-voice")
async def process_voice(request: VoiceAnalysisRequest):
    """
    Analyze voice transcript for scam detection
    
    Parameters:
    - transcript: The text transcript of the voice content
    - audio_features: Optional additional features extracted from audio
    
    Returns:
    - Scam analysis results including confidence score and indicators
    """
    try:
        start_time = time.time()
        
        # Check if transcript is empty
        if not request.transcript or len(request.transcript.strip()) == 0:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")
        
        # Process the voice transcript
        result = analyze_voice(request.transcript, request.audio_features)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result["processing_time_ms"] = processing_time
        
        return result
    except Exception as e:
        print(f"Error in voice analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")

@app.post("/analyze-text")
async def process_text(request: TextAnalysisRequest):
    """
    Analyze text message for scam detection
    
    Parameters:
    - text: The text content to analyze
    - message_type: Type of message (SMS, WhatsApp, Email)
    - context: Additional context information
    
    Returns:
    - Scam analysis results including confidence score and indicators
    """
    try:
        start_time = time.time()
        
        # Check if text is empty
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Process the text (ensure message_type is a string and not None)
        message_type = request.message_type if request.message_type is not None else 'SMS'
        result = analyze_text(request.text, message_type, request.context)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result["processing_time_ms"] = processing_time
        
        return result
    except Exception as e:
        print(f"Error in text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/batch-analyze-text")
async def process_batch_text(request: BatchTextAnalysisRequest):
    """
    Analyze multiple text messages in batch
    
    Parameters:
    - messages: List of messages with text content and message type
    
    Returns:
    - List of scam analysis results for each message
    """
    try:
        start_time = time.time()
        
        # Check if messages list is empty
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")
        
        # Extract texts and message types
        texts = [msg.get("text", "") for msg in request.messages]
        message_types = [msg.get("message_type", "SMS") for msg in request.messages]
        
        # Process the batch
        results = batch_analyze_text(texts, message_types)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "results": results,
            "count": len(results),
            "processing_time_ms": processing_time
        }
    except Exception as e:
        print(f"Error in batch text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch text analysis failed: {str(e)}")

@app.post("/train")
async def train_service_models():
    """
    Train or retrain the ML models
    
    Returns:
    - Training results including model accuracies
    """
    try:
        start_time = time.time()
        
        # Train the models
        results = train_models()
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        results["processing_time_ms"] = processing_time
        
        return results
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )

# Main function to start the server
def start_server():
    """Start the FastAPI server"""
    # Determine the port, with fallback to 8082 (same as in voice-text-ml.ts)
    port = int(os.getenv("ML_VOICE_TEXT_SERVICE_PORT", 8082))
    
    print(f"Starting Voice and Text Scam Detection API on port {port}...")
    uvicorn.run("voice_text_scam_service:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    # Create directory for model files
    os.makedirs("models", exist_ok=True)
    
    # Start the server
    start_server()