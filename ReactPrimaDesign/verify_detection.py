#!/usr/bin/env python3
"""
Simple verification script for the video detection model.
This script simply loads the models and verifies they can be used for prediction.
"""
import os
import sys
import joblib
import numpy as np
from pathlib import Path

def verify_models():
    """Verify that the models can be loaded and used for prediction"""
    # Define model paths
    models_dir = Path("./models")
    visual_model_path = models_dir / "visual_scam_detector.joblib"
    audio_model_path = models_dir / "audio_scam_detector.joblib"
    ensemble_model_path = models_dir / "ensemble_scam_detector.joblib"
    
    # Check if models exist
    models_exist = (
        os.path.exists(visual_model_path) and
        os.path.exists(audio_model_path) and
        os.path.exists(ensemble_model_path)
    )
    
    if not models_exist:
        print("Error: Model files not found. Run initialize_models.py first.")
        sys.exit(1)
    
    # Load models
    try:
        visual_model = joblib.load(visual_model_path)
        audio_model = joblib.load(audio_model_path)
        ensemble_model = joblib.load(ensemble_model_path)
        print("✓ Successfully loaded all models")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    # Test with sample inputs
    try:
        # Sample visual features
        visual_sample = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
        visual_pred = visual_model.predict_proba(visual_sample)[0][1]
        print(f"Visual model prediction: {visual_pred:.3f}")
        
        # Sample audio features
        audio_sample = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
        audio_pred = audio_model.predict_proba(audio_sample)[0][1]
        print(f"Audio model prediction: {audio_pred:.3f}")
        
        # Sample ensemble features
        ensemble_sample = np.array([[visual_pred, audio_pred, 0.5]])
        ensemble_pred = ensemble_model.predict_proba(ensemble_sample)[0][1]
        print(f"Ensemble model prediction: {ensemble_pred:.3f}")
        
        print("\n✓ All models working correctly!")
    except Exception as e:
        print(f"Error testing models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_models()