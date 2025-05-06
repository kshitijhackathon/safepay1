#!/usr/bin/env python3
"""
Initialize basic models for the video scam detection system.
This creates properly formatted model files with simple rule-based logic
until the full training with all datasets is complete.
"""
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def initialize_models():
    """Create basic model files that can be used for testing"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a basic visual model
    visual_model_path = os.path.join(models_dir, 'visual_scam_detector.joblib')
    if not os.path.exists(visual_model_path):
        print(f"Creating basic visual model at {visual_model_path}")
        # Create a trained Random Forest model with a simple dummy dataset
        # The features represent common visual patterns in scam vs legitimate videos
        # These are basic placeholder features until full training is complete
        X = np.array([
            # Scam examples (more face close-ups, poor lighting, hidden faces)
            [0.8, 0.2, 0.9, 0.7, 0.8],  # High face ratio, poor lighting
            [0.7, 0.3, 0.8, 0.6, 0.7],  # Similar pattern
            # Legitimate examples (more varied scenes, better lighting)
            [0.4, 0.7, 0.2, 0.3, 0.2],  # Lower face ratio, better lighting
            [0.3, 0.8, 0.3, 0.2, 0.3],  # Similar pattern
        ])
        y = np.array([1, 1, 0, 0])  # 1 = scam, 0 = legitimate
        
        # Create and fit the model
        visual_model = RandomForestClassifier(n_estimators=10, random_state=42)
        visual_model.fit(X, y)
        
        # Save the model
        joblib.dump(visual_model, visual_model_path)
        print("Visual model created and saved")
    else:
        print(f"Visual model already exists at {visual_model_path}")
    
    # Create a basic audio model
    audio_model_path = os.path.join(models_dir, 'audio_scam_detector.joblib')
    if not os.path.exists(audio_model_path):
        print(f"Creating basic audio model at {audio_model_path}")
        # Create a trained Random Forest model with a simple dummy dataset
        # The features represent common audio patterns in scam vs legitimate
        # These are basic placeholder features until full training is complete
        X = np.array([
            # Scam examples (more urgent tone, background noise)
            [0.9, 0.7, 0.8, 0.2, 0.9],  # Higher urgency, less natural
            [0.8, 0.6, 0.7, 0.3, 0.8],  # Similar pattern
            # Legitimate examples (more natural tone)
            [0.3, 0.2, 0.3, 0.7, 0.2],  # More natural speech
            [0.2, 0.3, 0.2, 0.8, 0.3],  # Similar pattern
        ])
        y = np.array([1, 1, 0, 0])  # 1 = scam, 0 = legitimate
        
        # Create and fit the model
        audio_model = RandomForestClassifier(n_estimators=10, random_state=42)
        audio_model.fit(X, y)
        
        # Save the model
        joblib.dump(audio_model, audio_model_path)
        print("Audio model created and saved")
    else:
        print(f"Audio model already exists at {audio_model_path}")
    
    # Create a basic ensemble model
    ensemble_model_path = os.path.join(models_dir, 'ensemble_scam_detector.joblib')
    if not os.path.exists(ensemble_model_path):
        print(f"Creating basic ensemble model at {ensemble_model_path}")
        # Create a trained Random Forest model with a simple dummy dataset
        # The features represent combined predictions from visual and audio models
        X = np.array([
            # Highly suspicious (both visual and audio)
            [0.9, 0.9, 0.8],  # High visual, high audio, high keyword match
            [0.8, 0.8, 0.7],  # Similar pattern
            # Mixed signals
            [0.8, 0.2, 0.5],  # High visual, low audio
            [0.2, 0.8, 0.5],  # Low visual, high audio
            # Legitimate (low on all signals)
            [0.2, 0.3, 0.2],  # Low on all metrics
            [0.3, 0.2, 0.3],  # Similar pattern
        ])
        y = np.array([1, 1, 0, 0, 0, 0])  # 1 = scam, 0 = legitimate
        
        # Create and fit the model
        ensemble_model = RandomForestClassifier(n_estimators=10, random_state=42)
        ensemble_model.fit(X, y)
        
        # Save the model
        joblib.dump(ensemble_model, ensemble_model_path)
        print("Ensemble model created and saved")
    else:
        print(f"Ensemble model already exists at {ensemble_model_path}")
        
    # Create a simple training summary
    summary_path = os.path.join(models_dir, 'training_summary.json')
    if not os.path.exists(summary_path):
        import json
        summary = {
            'total_videos': 4,
            'scam_videos': 2,
            'legitimate_videos': 2,
            'total_audio_files': 0,
            'scam_audio_files': 0,
            'legitimate_audio_files': 0,
            'visual_samples': 4,
            'audio_samples': 4
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Created training summary at {summary_path}")
    else:
        print(f"Training summary already exists at {summary_path}")
    
    print("\nModel initialization complete. These are basic models that will be")
    print("replaced when full training with all datasets completes.")

if __name__ == "__main__":
    initialize_models()