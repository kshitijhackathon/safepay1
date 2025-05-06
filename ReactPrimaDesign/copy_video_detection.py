#!/usr/bin/env python3
"""
Helper script to copy the video_detection.py file to root directory for easier imports.
"""
import os
import shutil
import sys

def main():
    """Copy the video_detection.py file to root directory"""
    source_path = os.path.join("server", "services", "video_detection.py")
    target_path = "video_detection.py"
    
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} does not exist.")
        sys.exit(1)
    
    try:
        shutil.copy2(source_path, target_path)
        print(f"✓ Successfully copied {source_path} to {target_path}")
    except Exception as e:
        print(f"Error copying file: {e}")
        sys.exit(1)
    
    print("Module is now importable directly from the root directory.")

if __name__ == "__main__":
    main()