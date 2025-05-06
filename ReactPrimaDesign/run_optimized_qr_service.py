"""
Helper script to manually start the Optimized QR ML Service.
"""

import subprocess
import sys
import os

def main():
    print("Starting Optimized QR ML Service...")
    
    # Install dependencies if needed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_qr.txt"])
        print("Dependencies installed/verified.")
    except Exception as e:
        print(f"Warning: Could not verify dependencies: {e}")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
        print("Created cache directory")
    
    # Start the service in the foreground so you can see logs
    print("Starting service on port 8000...")
    subprocess.call([sys.executable, "optimized_qr_scanner.py"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nService stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
