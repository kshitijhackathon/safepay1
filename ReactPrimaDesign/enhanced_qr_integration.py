"""
Enhanced QR Integration
Integrates the new ML-based QR risk detection model with the existing scanner
"""
import os
import sys
import requests
import json
import time
import threading
import subprocess
from typing import Dict, Any, Optional

# Constants
QR_RISK_SERVICE_PORT = 5050
QR_SCANNER_PORT = 8000
INTEGRATION_LOG = "qr_integration.log"

def log_message(message: str):
    """Log message to file and console"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(INTEGRATION_LOG, "a") as f:
        f.write(log_line + "\n")

def check_service_running(port: int) -> bool:
    """Check if a service is running on the specified port"""
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def start_ml_service():
    """Start the ML-based QR risk detection service"""
    log_message("Starting ML-based QR risk detection service...")
    return subprocess.Popen(
        ["python", "optimized_qr_risk_service.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

def start_scanner_service():
    """Start the existing QR scanner service"""
    log_message("Starting QR scanner service...")
    return subprocess.Popen(
        ["python", "optimized_qr_scanner.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

def monitor_service_output(process, name):
    """Monitor and log service output"""
    for line in iter(process.stdout.readline, b''):
        log_message(f"[{name}] {line.decode('utf-8').strip()}")

class QRIntegrator:
    """Class to integrate both QR services"""
    def __init__(self):
        self.ml_service_proc = None
        self.scanner_service_proc = None
        self.is_running = False
    
    def start_services(self):
        """Start both services if they're not already running"""
        # Check if ML service is running
        if not check_service_running(QR_RISK_SERVICE_PORT):
            self.ml_service_proc = start_ml_service()
            threading.Thread(
                target=monitor_service_output, 
                args=(self.ml_service_proc, "ML QR Risk Service"), 
                daemon=True
            ).start()
        else:
            log_message("ML QR Risk Service already running on port 5050")
        
        # Wait a bit for the service to start
        time.sleep(2)
        
        # Check if QR scanner service is running
        if not check_service_running(QR_SCANNER_PORT):
            self.scanner_service_proc = start_scanner_service()
            threading.Thread(
                target=monitor_service_output, 
                args=(self.scanner_service_proc, "QR Scanner Service"), 
                daemon=True
            ).start()
        else:
            log_message("QR Scanner Service already running on port 8000")
            
        self.is_running = True
        log_message("Services started successfully")
    
    def analyze_qr(self, qr_text: str) -> Dict[str, Any]:
        """Analyze QR code using both services and combine results"""
        # Check scanner service
        scanner_result = self._get_scanner_analysis(qr_text)
        
        # Check ML service
        ml_result = self._get_ml_analysis(qr_text)
        
        # Combine results
        combined_result = self._combine_results(scanner_result, ml_result)
        
        log_message(f"Analysis complete for QR: {qr_text[:30]}... - " +
                   f"Risk: {combined_result.get('risk_score', 'N/A')}%")
        
        return combined_result
    
    def _get_scanner_analysis(self, qr_text: str) -> Dict[str, Any]:
        """Get analysis from the existing scanner service"""
        try:
            response = requests.post(
                f"http://localhost:{QR_SCANNER_PORT}/predict",
                json={"qr_text": qr_text},
                timeout=3
            )
            return response.json()
        except Exception as e:
            log_message(f"Error getting scanner analysis: {str(e)}")
            return {"error": str(e), "risk_score": 50, "features": {}}
    
    def _get_ml_analysis(self, qr_text: str) -> Dict[str, Any]:
        """Get analysis from the ML risk detection service"""
        try:
            response = requests.post(
                f"http://localhost:{QR_RISK_SERVICE_PORT}/predict",
                json={"qr_text": qr_text},
                timeout=3
            )
            return response.json()
        except Exception as e:
            log_message(f"Error getting ML analysis: {str(e)}")
            return {"error": str(e), "risk_score": 50, "risk_level": "Medium"}
    
    def _combine_results(self, scanner_result: Dict[str, Any], 
                         ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from both services with weighted averaging"""
        try:
            # Get risk scores from both services
            scanner_score = scanner_result.get("risk_score", 50)
            ml_score = ml_result.get("risk_score", 50)
            
            # Weighted average favoring ML service (60% ML, 40% scanner)
            combined_score = int((ml_score * 0.6) + (scanner_score * 0.4))
            
            # Determine risk level based on combined score
            risk_level = "High" if combined_score > 70 else \
                        "Medium" if combined_score > 40 else "Low"
            
            # Combine explanations
            explanations = ml_result.get("explanation", [])
            
            # Combine features
            combined_features = {
                **scanner_result.get("features", {}),
                **ml_result.get("features", {})
            }
            
            # Combine for final result
            combined_result = {
                "risk_score": combined_score,
                "risk_level": risk_level,
                "recommendation": "Block" if combined_score > 70 else 
                                "Caution" if combined_score > 40 else "Allow",
                "explanation": explanations,
                "features": combined_features,
                "services": {
                    "scanner": {
                        "risk_score": scanner_score,
                        "latency_ms": scanner_result.get("latency_ms", 0)
                    },
                    "ml": {
                        "risk_score": ml_score,
                        "risk_level": ml_result.get("risk_level", "Medium"),
                        "scan_time_ms": ml_result.get("scan_time_ms", 0)
                    }
                }
            }
            
            return combined_result
        
        except Exception as e:
            log_message(f"Error combining results: {str(e)}")
            # Fallback to ML result or scanner result
            if "risk_score" in ml_result:
                return ml_result
            return scanner_result
    
    def stop_services(self):
        """Stop all services"""
        if self.ml_service_proc:
            self.ml_service_proc.terminate()
            log_message("ML QR Risk Service stopped")
            
        if self.scanner_service_proc:
            self.scanner_service_proc.terminate()
            log_message("QR Scanner Service stopped")
            
        self.is_running = False

def main():
    """Main function to run integration"""
    log_message("Starting QR integration service...")
    
    integrator = QRIntegrator()
    
    try:
        # Start services
        integrator.start_services()
        
        # Test with sample QR codes
        test_qrs = [
            "upi://pay?pa=legit@oksbi&pn=Trusted%20Merchant&am=200",
            "upi://pay?pa=urgent@verify.com&pn=URGENT%20VERIFY&am=9999",
            "upi://pay?pa=random123@randomdomain.com&pn=Test&am=500"
        ]
        
        log_message("\nTesting integration with sample QR codes...")
        
        for qr in test_qrs:
            log_message(f"\nAnalyzing QR: {qr}")
            result = integrator.analyze_qr(qr)
            
            log_message(f"  Combined Risk Score: {result['risk_score']}%")
            log_message(f"  Risk Level: {result['risk_level']}")
            log_message(f"  Recommendation: {result['recommendation']}")
            if "explanation" in result:
                log_message(f"  Explanation: {result['explanation']}")
        
        log_message("\nIntegration test complete. Services are running.")
        log_message("Press Ctrl+C to stop services...")
        
        # Keep running until interrupted
        while integrator.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log_message("Stopping services due to user interrupt...")
    finally:
        integrator.stop_services()
        log_message("QR integration service stopped")

if __name__ == "__main__":
    main()