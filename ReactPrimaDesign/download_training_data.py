#!/usr/bin/env python3
"""
Script to download training data from Google Drive and prepare it for the video detection model.
Uses direct file IDs instead of folder ID for more reliable downloads.
"""
import os
import sys
import gdown
import shutil
from pathlib import Path

# Create required directories
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Make directories for scam and legitimate videos
scam_dir = data_dir / "scam"
scam_dir.mkdir(exist_ok=True)

legitimate_dir = data_dir / "legitimate"
legitimate_dir.mkdir(exist_ok=True)

print("Starting download of training data for video scam detection...")

# File IDs and names for scam videos (vs prefix)
scam_videos = [
    # Using video IDs directly from your attached assets
    ("1h782N-LhGTWy2WF5tsDIXLGtp88RMoz7", "vs1.mp4"),
    ("14oieb8-zkbQV4BDP16lyPeL04-jFtxRq", "vs2.mp4"),
    ("1uyjJSgl46wBpjhJvScYcvGDlIyGtesMa", "vs3.mp4"),
    ("1o0zf393eoh0AGPLTCLqsNBweT1F_SY1t", "vs4.mp4"),
    ("1JSvw9Tl0tMYYhh_ypKOOX4ECoc56QUU1", "vs5.mp4"),
    ("1062sjRlPAYzZNln7nN7XAWNXq8pUHLHa", "vs6.mp4")
]

# File IDs and names for legitimate videos (v prefix)
legitimate_videos = [
    # Using video IDs directly from your attached assets
    ("1ciA_ObObT9tp5VkfsPh5b8k7sK_iGUyY", "v1.mp4"),
    ("101TQtc4mO2NZpc5j_yBvpitd5zt136Hb", "v2.mp4"),
    ("1gtbEAjoHtgSav5fgqXHYbjnnMBy6N3kP", "v3.mp4"),
    ("1-fVJQtutT-DcgsZyB7j-fxCSEofcmx7H", "v4.mp4"),
    ("1wylqRXBXnIO0Se_n5M11puD8OoK0-PK_", "v5.mp4"),
    ("11EIJWPY5DrbhyAiMij7WrX623HyXXpgG", "v6.mp4")
]

# Function to download a single file
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")
        return False

# Download scam videos
print("\nDownloading scam videos:")
scam_count = 0
for file_id, filename in scam_videos:
    output_path = os.path.join(scam_dir, filename)
    print(f"Downloading {filename}...")
    if download_file(file_id, output_path):
        size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        print(f"  ✓ {filename} ({size:.2f} MB)")
        scam_count += 1

# Download legitimate videos
print("\nDownloading legitimate videos:")
legitimate_count = 0
for file_id, filename in legitimate_videos:
    output_path = os.path.join(legitimate_dir, filename)
    print(f"Downloading {filename}...")
    if download_file(file_id, output_path):
        size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        print(f"  ✓ {filename} ({size:.2f} MB)")
        legitimate_count += 1

print(f"\nDownload summary:")
print(f"  - Scam videos: {scam_count}/{len(scam_videos)}")
print(f"  - Legitimate videos: {legitimate_count}/{len(legitimate_videos)}")

if scam_count > 0 and legitimate_count > 0:
    print("\nData is ready for training the video detection model.")
    print("Run 'python train_video_model.py' to train the model with this data.")
else:
    print("\nWarning: Not enough videos were downloaded for proper training.")
    print("Please check your internet connection and try again.")

# Additionally download our sample video attachments to the data directory
print("\nCopying sample videos from the attached_assets folder...")
sample_videos_folder = Path("./attached_assets")
if sample_videos_folder.exists():
    video_count = 0
    for file in os.listdir(sample_videos_folder):
        if file.endswith(".mp4"):
            src = os.path.join(sample_videos_folder, file)
            # Determine if it's a scam video (vs prefix) or legitimate (v prefix)
            if file.startswith("vs"):
                dst = os.path.join(scam_dir, file)
            else:
                dst = os.path.join(legitimate_dir, file)
            try:
                shutil.copy(src, dst)
                video_count += 1
            except Exception as e:
                print(f"Error copying {file}: {e}")
    
    print(f"Copied {video_count} sample videos from attached_assets.")
else:
    print("No attached_assets folder found for sample videos.")