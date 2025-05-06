#!/usr/bin/env python3
"""
Script to apply voice focus to the existing videos without training the model.
This helps capture voice features better without the overhead of model training.
"""
import os
import sys
import subprocess
import cv2
import numpy as np

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def apply_voice_focus(video_path, output_dir, is_scam=False):
    """Apply voice focus to audio and generate metadata"""
    # Determine base filename
    basename = os.path.basename(video_path)
    name, ext = os.path.splitext(basename)
    
    # Construct paths
    prefix = "scam_" if is_scam else "legitimate_"
    output_audio = os.path.join(output_dir, f"{prefix}{name}.wav")
    output_video = os.path.join(output_dir, f"{prefix}{name}.mp4")
    output_meta = os.path.join(output_dir, f"{prefix}{name}.meta")
    
    print(f"Processing: {basename}")
    
    # Copy source video
    try:
        with open(video_path, 'rb') as src_file:
            with open(output_video, 'wb') as dst_file:
                dst_file.write(src_file.read())
        print(f"  ✓ Copied video to {output_video}")
    except Exception as e:
        print(f"  ✗ Error copying video: {e}")
        return False
    
    # Extract optimized audio with voice focus
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-af', 'highpass=f=200,lowpass=f=3000,volume=2',  # Voice frequency range
            '-ac', '1',  # Convert to mono
            '-ar', '16000',  # 16kHz sample rate for speech
            '-y',  # Overwrite
            output_audio
        ]
        
        # Execute ffmpeg
        proc = subprocess.run(command, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               timeout=120)  # 2 minute timeout
        
        if proc.returncode == 0:
            print(f"  ✓ Extracted voice-optimized audio to {output_audio}")
            voice_optimized = True
        else:
            print(f"  ✗ Audio extraction failed: {proc.stderr.decode()}")
            # Create placeholder file
            with open(output_audio, 'wb') as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')
            voice_optimized = False
            
    except Exception as e:
        print(f"  ✗ Error processing audio: {e}")
        # Create placeholder file
        with open(output_audio, 'wb') as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')
        voice_optimized = False
    
    # Generate enhanced metadata
    try:
        # Use OpenCV to extract video properties
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Check for faces to help determine if there's a person speaking
            face_detected = False
            sample_frames = min(frame_count, 30)  # Check up to 30 frames
            sample_interval = max(1, int(frame_count / sample_frames))
            
            # Use face cascade if available
            try:
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
            except Exception as e:
                print(f"  ⚠ Face detection error: {e}")
                
            cap.release()
            
            # Write metadata file
            with open(output_meta, 'w') as f:
                f.write(f"source_video: {video_path}\n")
                f.write(f"voice_optimized: {voice_optimized}\n")
                f.write(f"fps: {fps}\n")
                f.write(f"frame_count: {frame_count}\n")
                f.write(f"duration: {duration} seconds\n")
                f.write(f"face_detected: {face_detected}\n")
                f.write(f"is_scam: {is_scam}\n")
                
            print(f"  ✓ Generated enhanced metadata")
            print(f"    - Duration: {duration:.1f} seconds")
            print(f"    - Face detected: {'Yes' if face_detected else 'No'}")
            print(f"    - Voice optimized: {'Yes' if voice_optimized else 'No'}")
            
            return True
        else:
            print(f"  ✗ Could not open video file: {video_path}")
            return False
    except Exception as e:
        print(f"  ✗ Error generating metadata: {e}")
        return False

def process_directory(input_dir, output_dir, is_scam=False):
    """Process all videos in a directory"""
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return 0
        
    # Count processed files
    count = 0
    
    # Process all videos
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
            video_path = os.path.join(input_dir, file)
            if apply_voice_focus(video_path, output_dir, is_scam):
                count += 1
    
    return count

def main():
    """Main entry point"""
    # Define directories
    data_dir = "./data"
    scam_dir = os.path.join(data_dir, "scam")
    legitimate_dir = os.path.join(data_dir, "legitimate")
    
    # Create audio output directories
    audio_dir = os.path.join(data_dir, "audio")
    scam_audio_dir = os.path.join(audio_dir, "scam")
    legitimate_audio_dir = os.path.join(audio_dir, "legitimate")
    
    ensure_dir(audio_dir)
    ensure_dir(scam_audio_dir)
    ensure_dir(legitimate_audio_dir)
    
    # Process scam videos
    print("\nProcessing scam videos:")
    scam_count = process_directory(scam_dir, scam_audio_dir, is_scam=True)
    
    # Process legitimate videos
    print("\nProcessing legitimate videos:")
    legitimate_count = process_directory(legitimate_dir, legitimate_audio_dir, is_scam=False)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Processed {scam_count} scam videos")
    print(f"Processed {legitimate_count} legitimate videos")
    print("Audio data is ready for training with improved voice focus.")

if __name__ == "__main__":
    main()