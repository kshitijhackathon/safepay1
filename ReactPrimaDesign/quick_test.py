#!/usr/bin/env python3
"""
Quick test script for the video detection functionality.
This script tests the video detection system with one sample video.
"""
import os
import sys
import json
from pathlib import Path

# Try to import the ScamVideoDetector class
try:
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

def quick_test():
    """Test the video detection functionality with a single sample video"""
    # Initialize the detector
    detector = get_detector()
    
    # Test with one of the sample videos from attached_assets
    test_video_path = os.path.join("attached_assets", "vs14.mp4")
    if not os.path.exists(test_video_path):
        print(f"Error: Test video not found at {test_video_path}")
        # Try an alternative path
        test_video_path = os.path.join("data", "scam", "vs14.mp4")
        if not os.path.exists(test_video_path):
            print(f"Error: Alternative test video not found at {test_video_path}")
            sys.exit(1)
    
    print(f"Testing video: {test_video_path}")
    
    # Analyze the video
    try:
        result = detector.analyze_video(test_video_path)
        
        # Print results
        print("\nRESULTS:")
        print(f"Is scam: {result['is_scam']}")
        print(f"Confidence: {result['confidence'] * 100:.1f}%")
        print(f"Visual analysis: {result.get('visual_confidence', 0) * 100:.1f}%")
        print(f"Audio analysis: {result.get('audio_confidence', 0) * 100:.1f}%")
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        for key, value in result.items():
            if key not in ['is_scam', 'confidence', 'visual_confidence', 'audio_confidence']:
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    quick_test()