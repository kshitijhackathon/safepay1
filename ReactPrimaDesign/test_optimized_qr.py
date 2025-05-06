"""
Test script for the Optimized QR Scanner service
"""

import subprocess
import time
import requests
import json
import sys

def main():
    # Try to start the QR service in the background
    print("Starting Optimized QR Service...")
    try:
        # Start the service in the background
        process = subprocess.Popen([sys.executable, "optimized_qr_scanner.py"], 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if service is running
        try:
            # Test a sample QR code
            sample_qr = "upi://pay?pa=example@upi&pn=Example%20Name&am=100.00"
            
            print(f"Testing QR code: {sample_qr}")
            response = requests.post("http://localhost:8000/predict", 
                                    json={"qr_text": sample_qr})
            
            if response.status_code == 200:
                result = response.json()
                print(f"Service is running!")
                print(f"Risk Score: {result['risk_score']}")
                print(f"Latency: {result['latency_ms']} ms")
                if result.get('cached'):
                    print("Result was cached")
                    
                # Test feedback endpoint
                print("\nTesting feedback submission...")
                feedback_response = requests.post("http://localhost:8000/feedback", 
                                                 json={"qr_text": sample_qr, "is_scam": False})
                
                if feedback_response.status_code == 200:
                    print("Feedback submitted successfully!")
                else:
                    print(f"Feedback error: {feedback_response.status_code}")
                    
            else:
                print(f"Service test failed with status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to the QR service. Make sure it's running on port 8000.")
        
        # Stop the process
        process.terminate()
        print("\nTest complete. Service stopped.")
        
    except Exception as e:
        print(f"Error testing QR service: {e}")

if __name__ == "__main__":
    main()
