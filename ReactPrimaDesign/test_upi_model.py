"""
Test UPI Fraud Detection Model
This script tests the UPI fraud detection model with various QR codes
"""

import os
import sys
import time
from upi_fraud_detection_model import predict_fraud_risk, train_model

def main():
    """Test the UPI fraud detection model"""
    print("Testing UPI Fraud Detection Model")
    print("=" * 50)
    
    # Train model if it doesn't exist
    if not os.path.exists('upi_fraud_model.joblib'):
        print("No model found. Training a new model...")
        train_model()
    
    # Test with various QR codes
    test_qrs = [
        # Safe QR codes
        {
            "name": "Standard merchant payment (low amount)",
            "qr": "upi://pay?pa=merchant@oksbi&pn=Trusted%20Store&am=200"
        },
        {
            "name": "Standard merchant payment (medium amount)",
            "qr": "upi://pay?pa=shop@okaxis&pn=Retail%20Store&am=2000"
        },
        {
            "name": "Known bank UPI (high amount)",
            "qr": "upi://pay?pa=customer@okicici&pn=Customer%20Name&am=12000"
        },
        
        # Risky QR codes
        {
            "name": "Non-standard domain",
            "qr": "upi://pay?pa=payment@gmail.com&pn=Payment%20Service&am=5000"
        },
        {
            "name": "Urgent keyword",
            "qr": "upi://pay?pa=urgent@verification.com&pn=URGENT%20VERIFY&am=9999"
        },
        {
            "name": "Very high amount",
            "qr": "upi://pay?pa=transfer@ybl&pn=Money%20Transfer&am=50000"
        },
        
        # Suspicious patterns
        {
            "name": "Unusual UPI ID length",
            "qr": "upi://pay?pa=very.long.unusual.username.with.many.dots@domain.com&pn=Strange&am=1000"
        },
        {
            "name": "Round amount (common in scams)",
            "qr": "upi://pay?pa=refund@gmail.com&pn=Refund%20Processing&am=10000"
        },
        {
            "name": "Random characters in UPI ID",
            "qr": "upi://pay?pa=x7q9z@randomdomain.com&pn=Unknown&am=3499"
        }
    ]
    
    results = []
    print("\nRunning fraud detection on test QR codes:")
    
    for i, test in enumerate(test_qrs):
        print(f"\nTest {i+1}: {test['name']}")
        print(f"QR: {test['qr']}")
        
        start_time = time.time()
        result = predict_fraud_risk(test['qr'])
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        print(f"  Risk Score: {result['risk_score']}%")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Fraud: {'Yes' if result['is_fraud'] else 'No'}")
        print(f"  Explanation: {', '.join(result['explanation'])}")
        print(f"  Processing time: {elapsed_time:.2f} ms")
        
        # Store results for summary
        result['test_name'] = test['name']
        result['qr'] = test['qr']
        result['processing_time_ms'] = elapsed_time
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {len(results)}")
    print(f"Average processing time: {sum(r['processing_time_ms'] for r in results) / len(results):.2f} ms")
    
    # Count by risk level
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for r in results:
        risk_counts[r['risk_level']] += 1
    
    print("\nRisk level distribution:")
    for level, count in risk_counts.items():
        print(f"  {level}: {count} ({count/len(results)*100:.1f}%)")
    
    # Count fraud predictions
    fraud_count = sum(1 for r in results if r['is_fraud'])
    print(f"\nFraud predictions: {fraud_count} ({fraud_count/len(results)*100:.1f}%)")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()