#!/usr/bin/env python3
"""
Test script for the video detection functionality.
This script tests the video detection system with sample videos from the data directory.
"""
import os
import sys
import json
import random
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

def test_video_detection():
    """Test the video detection functionality with sample videos"""
    # Initialize the detector
    detector = get_detector()
    
    # Find sample videos to test
    data_dir = Path("./data")
    if not data_dir.exists():
        print("Error: Data directory not found.")
        sys.exit(1)
    
    scam_dir = data_dir / "scam"
    legitimate_dir = data_dir / "legitimate"
    
    sample_videos = []
    
    # Get scam videos
    if scam_dir.exists():
        for file in os.listdir(scam_dir):
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                sample_videos.append((os.path.join(scam_dir, file), True))  # (path, is_scam)
    
    # Get legitimate videos
    if legitimate_dir.exists():
        for file in os.listdir(legitimate_dir):
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                sample_videos.append((os.path.join(legitimate_dir, file), False))  # (path, is_scam)
    
    if not sample_videos:
        print("Error: No sample videos found in data directory.")
        sys.exit(1)
    
    # Select a random sample if we have many videos
    if len(sample_videos) > 4:
        selected_samples = random.sample(sample_videos, 4)
    else:
        selected_samples = sample_videos
    
    print(f"Testing with {len(selected_samples)} sample videos")
    
    # Test each video
    for i, (video_path, is_scam) in enumerate(selected_samples):
        filename = os.path.basename(video_path)
        expected_type = "SCAM" if is_scam else "LEGITIMATE"
        print(f"\nTesting video {i+1}/{len(selected_samples)}: {filename} (Expected: {expected_type})")
        
        # Analyze the video
        try:
            result = detector.analyze_video(video_path)
            
            # Print results
            verdict = "SCAM DETECTED" if result['is_scam'] else "LEGITIMATE"
            confidence = result['confidence'] * 100  # Convert to percentage
            
            print(f"Result: {verdict} (Confidence: {confidence:.1f}%)")
            print(f"Visual analysis: {result.get('visual_result', 'N/A')}")
            print(f"Audio analysis: {result.get('audio_result', 'N/A')}")
            print(f"Text analysis: {result.get('text_result', 'N/A')}")
            
            # Check if prediction matches expected label
            matches_expected = (result['is_scam'] == is_scam)
            print(f"Matches expected: {'✓' if matches_expected else '✗'}")
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_video_detection()