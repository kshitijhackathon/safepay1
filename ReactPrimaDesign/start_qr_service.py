"""
QR Scam Detection Service Starter Script
"""

import subprocess
import os
import signal
import sys
import time

def start_service():
    """Start the QR scam detection service"""
    print("Starting QR Scam Detection Service...")
    
    # Create model first if it doesn't exist
    if not os.path.exists('scam_model.pkl'):
        print("Training initial model...")
        subprocess.run(['python', 'scam_model.py'])
    
    # Launch the service
    try:
        port = os.environ.get("ML_QR_SERVICE_PORT", "8081")
        process = subprocess.Popen(['python', 'qr_scam_service.py'])
        print(f"QR Scam Detection Service started on port {port} (PID: {process.pid})")
        
        # Keep the script running to maintain the service
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nShutting down QR Scam Detection Service...")
        process.terminate()
        
    except Exception as e:
        print(f"Error running QR service: {str(e)}")

if __name__ == '__main__':
    start_service()