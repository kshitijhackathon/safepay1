"""
Starter script for the Optimized QR ML Service.
Runs the FastAPI service on port 8000.
"""

import os
import subprocess
import sys

def main():
    print("Starting Optimized QR ML Service...")
    log_file = "qr_service.log"
    
    try:
        # Ensure dependencies are installed
        print("Checking dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_qr.txt"])
        
        # Create cache directory if it doesn't exist
        if not os.path.exists('./cache'):
            os.makedirs('./cache')
            print("Created cache directory")
            
        # Make sure the script is executable
        if os.name != 'nt':  # Not Windows
            subprocess.check_call(["chmod", "+x", "optimized_qr_scanner.py"])
        
        # Start the FastAPI server with logging
        print(f"Starting FastAPI server on port 8000 (logging to {log_file})...")
        
        # Open log file
        with open(log_file, "w") as f:
            # Run process and redirect output to log file
            process = subprocess.Popen(
                [sys.executable, "optimized_qr_risk_service.py"],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            print(f"QR Service started with PID: {process.pid}")
            print(f"You can monitor the logs with: tail -f {log_file}")
            
            # For non-background operation, uncomment this:
            # process.wait()
            # exit_code = process.returncode
            # print(f"Service exited with code: {exit_code}")
            
    except Exception as e:
        print(f"Error starting QR ML Service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()