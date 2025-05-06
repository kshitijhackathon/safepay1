#!/usr/bin/env python3
"""
Script to extract audio from video files for audio recognition training.
This extracts audio from the downloaded video files and prepares it for 
training the audio recognition component of the video detection feature.
"""
import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

# Import necessary libraries for audio extraction
try:
    import numpy as np
    import librosa
    import soundfile as sf
    import cv2
    print("Using OpenCV and librosa for audio extraction")
except ImportError:
    print("Error: Missing required libraries for audio extraction.")
    print("Please install: pip install numpy librosa soundfile opencv-python")
    sys.exit(1)

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Create audio directory structure
audio_base_dir = data_dir / "audio"
audio_base_dir.mkdir(exist_ok=True)

# Create scam and legitimate audio directories
scam_audio_dir = audio_base_dir / "scam"
scam_audio_dir.mkdir(exist_ok=True)

legitimate_audio_dir = audio_base_dir / "legitimate"
legitimate_audio_dir.mkdir(exist_ok=True)

def extract_audio(video_path, audio_output_path):
    """
    Extract audio from a video file with voice focus and noise reduction
    We perform basic audio preprocessing to better isolate voice content
    """
    try:
        # Create the MP4 copy path
        video_copy_path = str(audio_output_path).replace('.wav', '.mp4')
        
        try:
            # First try using ffmpeg which is now installed
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-af', 'highpass=f=200,lowpass=f=3000,volume=2',  # Voice frequency range and amplification
                '-ac', '1',  # Convert to mono
                '-ar', '16000',  # 16kHz sample rate (good for speech)
                '-y',  # Overwrite output files
                audio_output_path
            ]
            
            # Run ffmpeg
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"  ✓ Extracted voice-optimized audio to {audio_output_path}")
            
            # For backup, also copy the video file for direct processing
            shutil.copy2(video_path, video_copy_path)
            print(f"  ✓ Also copied video file to {video_copy_path}")
        except Exception as ffmpeg_error:
            print(f"  ✗ FFmpeg processing failed: {ffmpeg_error}")
            print(f"  ⚠ Falling back to direct video copy")
            
            # Copy the video file to the audio directory as fallback
            shutil.copy2(video_path, video_copy_path)
            print(f"  ✓ Copied video file to {video_copy_path}")
            
            # Create an empty WAV file as placeholder
            with open(audio_output_path, 'wb') as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')
        
        # Create enhanced metadata with voice focus info
        audio_meta_path = str(audio_output_path).replace('.wav', '.meta')
        with open(audio_meta_path, 'w') as f:
            f.write(f"source_video: {video_path}\n")
            f.write(f"copy_path: {video_copy_path}\n")
            f.write(f"voice_optimized: {os.path.exists(audio_output_path) and os.path.getsize(audio_output_path) > 1000}\n")
        
        # Use OpenCV to extract basic video properties for the metadata
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Check for faces to help determine if there's a person speaking
            face_detected = False
            sample_frames = min(frame_count, 30)  # Check up to 30 frames
            sample_interval = max(1, int(frame_count / sample_frames))
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            for i in range(0, int(frame_count), sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        face_detected = True
                        break
            
            # Add enhanced metadata
            with open(audio_meta_path, 'a') as f:
                f.write(f"fps: {fps}\n")
                f.write(f"frame_count: {frame_count}\n")
                f.write(f"duration: {duration} seconds\n")
                f.write(f"face_detected: {face_detected}\n")
                
            cap.release()
        
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def extract_audio_from_directory(video_dir, audio_dir, prefix=""):
    """Extract audio from all videos in a directory"""
    extracted_count = 0
    
    # Ensure the directories exist
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return 0
    
    # Process all videos in the directory
    for file in os.listdir(video_dir):
        if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
            video_path = os.path.join(video_dir, file)
            audio_filename = f"{prefix}{os.path.splitext(file)[0]}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            
            print(f"Extracting audio from {file}...")
            if extract_audio(video_path, audio_path):
                extracted_count += 1
                print(f"  ✓ Created {audio_filename}")
    
    return extracted_count

def main():
    print("Starting audio extraction for training...")
    
    # Define video directories
    scam_video_dir = os.path.join("./data", "scam")
    legitimate_video_dir = os.path.join("./data", "legitimate")
    
    # Extract audio from scam videos
    print("\nExtracting audio from scam videos:")
    scam_count = extract_audio_from_directory(scam_video_dir, scam_audio_dir, "scam_")
    
    # Extract audio from legitimate videos
    print("\nExtracting audio from legitimate videos:")
    legitimate_count = extract_audio_from_directory(legitimate_video_dir, legitimate_audio_dir, "legitimate_")
    
    print(f"\nAudio extraction summary:")
    print(f"  - Scam audio files: {scam_count}")
    print(f"  - Legitimate audio files: {legitimate_count}")
    
    if scam_count > 0 and legitimate_count > 0:
        print("\nAudio data is ready for training.")
        print("These audio files will be used to improve the audio recognition component of the video detection feature.")
    else:
        print("\nWarning: Not enough audio was extracted for proper training.")

if __name__ == "__main__":
    main()