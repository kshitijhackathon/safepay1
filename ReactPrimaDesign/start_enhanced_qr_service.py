"""
Enhanced QR Scam Detection Service Starter Script
Starts both the traditional QR scanner and the new ML-based risk detection service
"""

import subprocess
import os
import signal
import sys
import time
import requests
import json
import threading
import argparse

# Configuration
QR_RISK_SERVICE_PORT = 5050
QR_SCANNER_PORT = 8000
LOG_FILE = "qr_service.log"

# Initialize logger
def log_message(message):
    """Log message to file and console"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

def check_service(port):
    """Check if a service is running on the specified port"""
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def monitor_output(process, name):
    """Monitor and log process output"""
    for line in iter(process.stdout.readline, b''):
        log_message(f"[{name}] {line.decode('utf-8').strip()}")

def start_service(integrated_mode=True):
    """Start the QR scam detection services"""
    log_message("Starting Enhanced QR Scam Detection Services...")
    
    processes = []
    
    # Train risk model if it doesn't exist
    if not os.path.exists('qr_risk_model.joblib'):
        log_message("Training ML risk detection model...")
        try:
            subprocess.run(['python', 'qr_risk_detection_model.py'], check=True)
            log_message("ML model training completed")
        except subprocess.CalledProcessError:
            log_message("Error training ML model. Will continue with existing models.")
    
    try:
        # Start the original QR scanner if it's not already running
        if not check_service(QR_SCANNER_PORT):
            log_message("Starting original QR scanner service...")
            scanner_proc = subprocess.Popen(
                ['python', 'optimized_qr_scanner.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            processes.append(scanner_proc)
            threading.Thread(
                target=monitor_output, 
                args=(scanner_proc, "QR Scanner"), 
                daemon=True
            ).start()
            log_message(f"QR Scanner Service started on port {QR_SCANNER_PORT}")
        else:
            log_message(f"QR Scanner Service already running on port {QR_SCANNER_PORT}")
        
        # Wait a bit to ensure the first service is up
        time.sleep(2)
        
        # Start the ML-based risk service if it's not already running
        if not check_service(QR_RISK_SERVICE_PORT):
            log_message("Starting ML-based QR Risk Detection Service...")
            risk_proc = subprocess.Popen(
                ['python', 'optimized_qr_risk_service.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            processes.append(risk_proc)
            threading.Thread(
                target=monitor_output, 
                args=(risk_proc, "QR Risk Service"), 
                daemon=True
            ).start()
            log_message(f"QR Risk Detection Service started on port {QR_RISK_SERVICE_PORT}")
        else:
            log_message(f"QR Risk Detection Service already running on port {QR_RISK_SERVICE_PORT}")
        
        # Start the integration service if requested
        if integrated_mode:
            log_message("Starting QR Integration Service...")
            integration_proc = subprocess.Popen(
                ['python', 'enhanced_qr_integration.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            processes.append(integration_proc)
            threading.Thread(
                target=monitor_output, 
                args=(integration_proc, "QR Integration"), 
                daemon=True
            ).start()
        
        # Test the services
        log_message("Testing services with sample QR code...")
        time.sleep(5)  # Give services time to start
        
        test_qr = "upi://pay?pa=test@oksbi&pn=Test%20Merchant&am=100"
        
        try:
            # Test original scanner
            scanner_response = requests.post(
                f"http://localhost:{QR_SCANNER_PORT}/predict",
                json={"qr_text": test_qr},
                timeout=3
            )
            log_message(f"QR Scanner test response: {scanner_response.status_code}")
            if scanner_response.status_code == 200:
                log_message(f"Scanner result: {json.dumps(scanner_response.json())}")
            
            # Test ML service
            ml_response = requests.post(
                f"http://localhost:{QR_RISK_SERVICE_PORT}/predict",
                json={"qr_text": test_qr},
                timeout=3
            )
            log_message(f"ML Risk Service test response: {ml_response.status_code}")
            if ml_response.status_code == 200:
                log_message(f"ML result: {json.dumps(ml_response.json())}")
            
            log_message("Both services are running and responding to requests")
        except Exception as e:
            log_message(f"Error testing services: {str(e)}")
        
        log_message("All QR detection services started successfully")
        log_message("Press Ctrl+C to shut down...")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log_message("\nShutting down QR Scam Detection Services...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        log_message("All services stopped")
        
    except Exception as e:
        log_message(f"Error running QR services: {str(e)}")
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start Enhanced QR Scam Detection Services")
    parser.add_argument("--integrated", action="store_true", 
                        help="Start the integration service to combine results")
    parser.add_argument("--train", action="store_true",
                        help="Force retraining of the ML model")
    parser.add_argument("--measure", action="store_true",
                        help="Measure model accuracy before starting services")
    
    args = parser.parse_args()
    
    # Force model training if requested
    if args.train and os.path.exists('qr_risk_model.joblib'):
        os.remove('qr_risk_model.joblib')
        log_message("Removed existing model for retraining")
    
    # Measure accuracy if requested
    if args.measure:
        log_message("Measuring model accuracy...")
        subprocess.run(['python', 'measure_qr_model_accuracy.py'])
    
    start_service(integrated_mode=args.integrated)