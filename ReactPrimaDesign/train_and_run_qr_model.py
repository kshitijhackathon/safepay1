"""
Train QR Risk Model and Start Service
This script trains the QR risk detection model and starts the FastAPI service
"""
import os
import sys
import time
from qr_risk_detection_model import train_model, analyze_qr_risk

def main():
    """Train model and start service"""
    print("Starting QR Risk Detection Training and Service...")
    
    # Train model
    try:
        print("Training QR risk detection model...")
        model = train_model()
        print("Model training completed successfully.")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        sys.exit(1)
    
    # Test the model with sample data
    print("\nTesting model with sample data...")
    test_cases = [
        "upi://pay?pa=legit@oksbi&pn=Trusted%20Merchant&am=200",
        "upi://pay?pa=urgent@verify.com&pn=URGENT%20VERIFY&am=9999",
        "upi://pay?pa=random123@randomdomain.com&pn=Test&am=500",
    ]
    
    for qr in test_cases:
        print(f"\nTesting QR: {qr}")
        result = analyze_qr_risk(qr)
        print(f"  Risk Score: {result['risk_score']}%")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Is Scam: {result['is_scam']}")
        print(f"  Explanation: {result['explanation']}")
    
    # Start the service
    print("\nStarting QR Risk Detection service...")
    os.system("python optimized_qr_risk_service.py")

if __name__ == "__main__":
    main()