#!/usr/bin/env python3
"""
Quick training script for video scam detection model using the enhanced ScamVideoDetector class.
This script uses the fully integrated training functionality directly in the ScamVideoDetector class.
"""
import os
import sys
from pathlib import Path
import argparse

try:
    # Try specific import paths
    try:
        from server.services.video_detection import ScamVideoDetector, get_detector
        print("✓ Imported from server.services.video_detection")
    except ImportError:
        try:
            from services.video_detection import ScamVideoDetector, get_detector
            print("✓ Imported from services.video_detection")
        except ImportError:
            from video_detection import ScamVideoDetector, get_detector
            print("✓ Imported from video_detection")
except ImportError as e:
    print(f"Error: Could not import ScamVideoDetector. Details: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Quick train video scam detection models')
    parser.add_argument('--data-dir', default='./data', 
                        help='Directory containing training data (default: ./data)')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please make sure you have videos in the data directory.")
        sys.exit(1)
    
    # Initialize detector
    detector = get_detector()
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Train using the integrated method in ScamVideoDetector
    print(f"Training with data from: {data_dir}")
    result = detector.train_model(data_dir)
    
    if result['success']:
        print("\n✅ TRAINING SUCCESSFUL")
        print(f"Message: {result['message']}")
        print("\nModel details:")
        for key, value in result['details'].items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ TRAINING FAILED")
        print(f"Error: {result['message']}")

if __name__ == "__main__":
    main()