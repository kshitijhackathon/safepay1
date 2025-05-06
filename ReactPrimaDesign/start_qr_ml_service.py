#!/usr/bin/env python
"""
Starter script for the QR ML Service.
Runs the FastAPI service on port 8000.
"""

import os
import subprocess
import sys

def main():
    print("Starting QR ML Service on port 8000...")
    try:
        # Start the FastAPI service
        subprocess.run([
            "python", "-m", "uvicorn", 
            "qr_scan_ml_service:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down QR ML Service...")
    except Exception as e:
        print(f"Error starting QR ML Service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()