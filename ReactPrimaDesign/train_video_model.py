#!/usr/bin/env python3
"""
Train the video scam detection model using the downloaded data.
This script processes video files and trains the visual, audio, and text models.
"""
import os
import sys
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Add server directories to the path to import ScamVideoDetector
server_services_path = os.path.join(os.path.dirname(__file__), "server", "services")
server_path = os.path.join(os.path.dirname(__file__), "server")
sys.path.append(server_services_path)
sys.path.append(server_path)

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
    print("Available paths:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)

def find_training_videos(data_dir: str) -> Tuple[List[str], List[str]]:
    """Find all training videos in the data directory, splitting into scam and legitimate sets."""
    scam_videos = []
    legitimate_videos = []
    
    # Check for specific scam and legitimate directories
    scam_dir = os.path.join(data_dir, "scam")
    legitimate_dir = os.path.join(data_dir, "legitimate")
    
    # Get videos from the scam directory
    if os.path.exists(scam_dir):
        for file in os.listdir(scam_dir):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                full_path = os.path.join(scam_dir, file)
                scam_videos.append(full_path)
    
    # Get videos from the legitimate directory
    if os.path.exists(legitimate_dir):
        for file in os.listdir(legitimate_dir):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                full_path = os.path.join(legitimate_dir, file)
                legitimate_videos.append(full_path)
    
    # If no videos found in specific directories, try looking in the main data directory
    if not scam_videos and not legitimate_videos:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                    full_path = os.path.join(root, file)
                    
                    # Determine if this is a scam video based on filename
                    if file.startswith("vs"):
                        scam_videos.append(full_path)
                    elif file.startswith("v"):
                        legitimate_videos.append(full_path)
    
    return scam_videos, legitimate_videos

def find_audio_training_files(data_dir: str) -> Tuple[List[str], List[str]]:
    """Find all audio training files in the audio directory, splitting into scam and legitimate sets."""
    scam_audio_files = []
    legitimate_audio_files = []
    
    # Check for audio directories
    audio_base_dir = os.path.join(data_dir, "audio")
    scam_audio_dir = os.path.join(audio_base_dir, "scam")
    legitimate_audio_dir = os.path.join(audio_base_dir, "legitimate")
    
    # Get audio files from the scam directory
    if os.path.exists(scam_audio_dir):
        for file in os.listdir(scam_audio_dir):
            if file.lower().endswith(('.mp4')):  # We're using MP4 files as audio in our extraction
                full_path = os.path.join(scam_audio_dir, file)
                scam_audio_files.append(full_path)
    
    # Get audio files from the legitimate directory
    if os.path.exists(legitimate_audio_dir):
        for file in os.listdir(legitimate_audio_dir):
            if file.lower().endswith(('.mp4')):  # We're using MP4 files as audio in our extraction
                full_path = os.path.join(legitimate_audio_dir, file)
                legitimate_audio_files.append(full_path)
    
    print(f"Found {len(scam_audio_files)} scam audio files and {len(legitimate_audio_files)} legitimate audio files")
    return scam_audio_files, legitimate_audio_files

def process_videos(detector: ScamVideoDetector, video_paths: List[str], is_scam: bool):
    """Process videos and prepare features for training."""
    features = []
    labels = []
    
    for video_path in video_paths:
        print(f"Processing {'scam' if is_scam else 'legitimate'} video: {os.path.basename(video_path)}")
        
        try:
            # Extract visual features
            visual_features = detector.extract_visual_features(video_path)
            
            # Extract audio features
            audio_path = detector.extract_audio_from_video(video_path)
            if audio_path:
                audio_features = detector.extract_audio_features(audio_path)
                # Clean up temporary audio file if it's not in our data directory
                if os.path.exists(audio_path) and '/data/audio/' not in audio_path and 'data/audio/' not in audio_path:
                    os.unlink(audio_path)
            else:
                # Placeholder if audio extraction fails
                audio_features = None
            
            if visual_features is not None:
                features.append({
                    'visual': visual_features,
                    'audio': audio_features,
                    'path': video_path
                })
                labels.append(1 if is_scam else 0)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
    
    return features, labels

def process_audio_files(detector: ScamVideoDetector, audio_paths: List[str], is_scam: bool):
    """Process audio files and prepare features for training."""
    features = []
    labels = []
    
    for audio_path in audio_paths:
        print(f"Processing {'scam' if is_scam else 'legitimate'} audio: {os.path.basename(audio_path)}")
        
        try:
            # We only need audio features for these files
            audio_features = detector.extract_audio_features(audio_path)
            
            if audio_features is not None:
                features.append({
                    'visual': None,  # No visual features for audio-only files
                    'audio': audio_features,
                    'path': audio_path
                })
                labels.append(1 if is_scam else 0)
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
    
    return features, labels

def train_models(detector: ScamVideoDetector, scam_videos: List[str], legitimate_videos: List[str], 
              scam_audio_files: List[str] = [], legitimate_audio_files: List[str] = []):
    """
    Train the models using the processed videos and audio files.
    
    Args:
        detector: ScamVideoDetector instance
        scam_videos: List of scam video paths
        legitimate_videos: List of legitimate video paths
        scam_audio_files: Optional list of scam audio files
        legitimate_audio_files: Optional list of legitimate audio files
    """
    # Process scam videos
    scam_features, scam_labels = process_videos(detector, scam_videos, True)
    
    # Process legitimate videos
    legitimate_features, legitimate_labels = process_videos(detector, legitimate_videos, False)
    
    # Process additional audio files if provided
    scam_audio_features = []
    scam_audio_labels = []
    legitimate_audio_features = []
    legitimate_audio_labels = []
    
    if scam_audio_files:
        scam_audio_features, scam_audio_labels = process_audio_files(detector, scam_audio_files, True)
        print(f"Processed {len(scam_audio_features)} scam audio files")
    
    if legitimate_audio_files:
        legitimate_audio_features, legitimate_audio_labels = process_audio_files(detector, legitimate_audio_files, False)
        print(f"Processed {len(legitimate_audio_features)} legitimate audio files")
    
    # Combine all features and labels
    all_features = scam_features + legitimate_features + scam_audio_features + legitimate_audio_features
    all_labels = scam_labels + legitimate_labels + scam_audio_labels + legitimate_audio_labels
    
    # Shuffle the data (maintaining the feature-label correspondence)
    combined = list(zip(all_features, all_labels))
    random.shuffle(combined)
    all_features, all_labels = zip(*combined) if combined else ([], [])
    
    # Train visual model (using only video data, not audio-only files)
    visual_data = [(feature['visual'], label) for feature, label in zip(all_features, all_labels) 
                  if feature['visual'] is not None]
    
    if visual_data:
        print(f"\nTraining visual model with {len(visual_data)} samples...")
        X_visual, y_visual = zip(*visual_data)
        try:
            detector.visual_classifier.fit(X_visual, y_visual)
            # Save the model
            detector.save_visual_model()
            print("Visual model trained and saved!")
        except Exception as e:
            print(f"Error training visual model: {e}")
    
    # Train audio model (using all data that has audio features)
    audio_data = [(feature['audio'], label) for feature, label in zip(all_features, all_labels) 
                 if feature['audio'] is not None]
    
    if audio_data:
        print(f"\nTraining audio model with {len(audio_data)} samples...")
        X_audio, y_audio = zip(*audio_data)
        try:
            detector.audio_classifier.fit(X_audio, y_audio)
            # Save the model
            detector.save_audio_model()
            print("Audio model trained and saved!")
        except Exception as e:
            print(f"Error training audio model: {e}")
    
    # Save training summary
    summary = {
        'total_videos': len(scam_features) + len(legitimate_features),
        'scam_videos': len(scam_features),
        'legitimate_videos': len(legitimate_features),
        'total_audio_files': len(scam_audio_features) + len(legitimate_audio_features),
        'scam_audio_files': len(scam_audio_features),
        'legitimate_audio_files': len(legitimate_audio_features),
        'visual_samples': len(visual_data) if visual_data else 0,
        'audio_samples': len(audio_data) if audio_data else 0
    }
    
    with open(os.path.join(os.path.dirname(__file__), 'models', 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\nTraining summary:")
    print(f"Total videos processed: {summary['total_videos']}")
    print(f"Scam videos: {summary['scam_videos']}")
    print(f"Legitimate videos: {summary['legitimate_videos']}")
    print(f"Total audio files processed: {summary['total_audio_files']}")
    print(f"Scam audio files: {summary['scam_audio_files']}")
    print(f"Legitimate audio files: {summary['legitimate_audio_files']}")
    print(f"Visual samples used: {summary['visual_samples']}")
    print(f"Audio samples used: {summary['audio_samples']}")

def main():
    parser = argparse.ArgumentParser(description='Train video scam detection models')
    parser.add_argument('--data-dir', default='./data', help='Directory containing training data')
    parser.add_argument('--audio-only', action='store_true', help='Train only with audio files')
    parser.add_argument('--include-audio', action='store_true', help='Include audio files in training')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please run 'python download_training_data.py' first.")
        sys.exit(1)
    
    # Initialize detector
    detector = get_detector()
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Find training videos
    scam_videos, legitimate_videos = find_training_videos(data_dir)
    
    # Find audio training files
    scam_audio_files, legitimate_audio_files = find_audio_training_files(data_dir)
    
    # Basic validation for videos
    if not scam_videos and not args.audio_only:
        print("Warning: No scam videos found.")
    if not legitimate_videos and not args.audio_only:
        print("Warning: No legitimate videos found.")
    
    # Basic validation for audio files
    if not scam_audio_files and (args.audio_only or args.include_audio):
        print("Warning: No scam audio files found.")
    if not legitimate_audio_files and (args.audio_only or args.include_audio):
        print("Warning: No legitimate audio files found.")
    
    print(f"Found {len(scam_videos)} scam videos and {len(legitimate_videos)} legitimate videos.")
    
    # Train the models
    if args.audio_only:
        # Train using only audio files
        if scam_audio_files or legitimate_audio_files:
            print("Training using only audio files...")
            train_models(detector, [], [], scam_audio_files, legitimate_audio_files)
        else:
            print("No audio files found for training. Please check the data directory.")
    elif args.include_audio:
        # Train using both videos and audio
        if (scam_videos or legitimate_videos or scam_audio_files or legitimate_audio_files):
            print("Training using both videos and audio files...")
            train_models(detector, scam_videos, legitimate_videos, scam_audio_files, legitimate_audio_files)
        else:
            print("No training data found. Please check the data directory.")
    else:
        # Train using only videos (default)
        if scam_videos or legitimate_videos:
            print("Training using only video files...")
            train_models(detector, scam_videos, legitimate_videos)
        else:
            print("No videos found for training. Please check the data directory.")
            
    # Default to including audio if available
    if not args.audio_only and not args.include_audio and (scam_audio_files or legitimate_audio_files):
        print("\nNote: Audio files were found but not used for training.")
        print("Run with --include-audio to use both video and audio files.")
        print("Run with --audio-only to use only audio files.")

if __name__ == "__main__":
    main()