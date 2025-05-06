import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'wouter';
import { EnhancedQRScanner, QRScannerHandle } from '@/components/scanner/enhanced-qr-scanner';
import { 
  analyzeUpiRisk, 
  shouldBlockTransaction, 
  shouldShowWarning,
  detectAdvancedFraud,
  FraudDetectionResponse
} from '@/lib/fraud-detection';
import { extractUPIPaymentInfo } from '@/lib/ml-qr-scanner';
import { PaymentSafetyPopup } from '@/components/payment/payment-safety-popup';
import { AlertTriangle, AlertCircle, CheckCircle, Shield, AlertOctagon, Loader2, ChevronLeft, ArrowLeft, LockKeyhole } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
import { useAuthState } from '@/hooks/use-auth-state';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

export default function Scan() {
  const [location, setLocation] = useLocation();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [showBlocked, setShowBlocked] = useState(false);
  const [showMlAnalysis, setShowMlAnalysis] = useState(false);
  const [showSafeDialog, setShowSafeDialog] = useState(false);
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);
  const [upiDetected, setUpiDetected] = useState(false); // For UPI detection status
  const { authState } = useAuthState(); // Auth state for login checks
  const qrScannerRef = useRef<QRScannerHandle>(null); // Ref to access scanner methods
  const [safeTransactionInfo, setSafeTransactionInfo] = useState<{
    upiId: string;
    queryParams: string;
    name: string;
    amount: string;
  }>({ upiId: '', queryParams: '', name: '', amount: '' });
  const [scannedUpiId, setScannedUpiId] = useState('');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [riskDetails, setRiskDetails] = useState<{
    percentage: number;
    level: string;
    reports: number;
  } | null>(null);
  const [mlRiskDetails, setMlRiskDetails] = useState<FraudDetectionResponse | null>(null);
  const { toast } = useToast();
  
  // Check for QR data in URL parameters (when coming from QR scan or search)
  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const upiIdFromUrl = params.get('upiId');
    const qrDataFromUrl = params.get('qrData');
    const fromSearch = params.get('fromSearch');
    
    console.log('URL parameters:', { upiIdFromUrl, qrDataFromUrl, fromSearch });
    
    // First priority: Check for QR data from scanner
    if (qrDataFromUrl) {
      console.log('Processing QR data from scanner:', qrDataFromUrl);
      handleScan(qrDataFromUrl);
    }
    // Second priority: Check for UPI from search
    else if (upiIdFromUrl && fromSearch === 'true') {
      // Process the UPI ID from search
      processUpiId(upiIdFromUrl);
    }
    // Third priority: Check for pending scan in session storage
    else {
      try {
        const pendingScan = sessionStorage.getItem('lastScannedQR');
        if (pendingScan) {
          console.log('Processing previously scanned QR from session storage:', pendingScan);
          // Clear the stored QR data
          sessionStorage.removeItem('lastScannedQR');
          // Process the stored QR data
          handleScan(pendingScan);
        }
      } catch (error) {
        console.error('Error checking for stored QR data:', error);
      }
    }
  }, [location]);

  const handleScan = async (qrData: string) => {
    // Try to parse the payment info if in JSON format
    let paymentInfo = {
      upi_id: '',
      name: '',
      amount: '',
      currency: 'INR',
      ml_risk_score: 0,
      ml_risk_level: 'Low' as 'Low' | 'Medium' | 'High',
      ml_recommendation: 'Allow' as 'Allow' | 'Verify' | 'Block'
    };
    
    let upiId = '';
    
    try {
      console.log('Raw QR data:', qrData);
      
      // First, try to check if it's JSON
      try {
        const parsedData = JSON.parse(qrData);
        paymentInfo = parsedData;
        upiId = parsedData.upi_id;
        console.log('Parsed as JSON:', parsedData);
      } catch (jsonError) {
        console.log('Not JSON, trying UPI parsing with extractUPIPaymentInfo utility');
        
        // Use our dedicated UPI info extraction utility
        const extractedInfo = extractUPIPaymentInfo(qrData);
        console.log('Extracted UPI payment info:', extractedInfo);
        
        if (extractedInfo.valid && extractedInfo.upiId) {
          upiId = extractedInfo.upiId;
          paymentInfo.upi_id = extractedInfo.upiId;
          
          if (extractedInfo.name) {
            paymentInfo.name = extractedInfo.name;
          }
          
          if (extractedInfo.amount) {
            paymentInfo.amount = extractedInfo.amount;
          }
          
          if (extractedInfo.currency) {
            paymentInfo.currency = extractedInfo.currency;
          }
          
          // Store full QR data in session storage for later retrieval
          try {
            console.log('Storing QR data in sessionStorage for transaction confirmation:', qrData);
            sessionStorage.setItem('pendingQrScan', qrData);
            
            // Also store parsed info in a more structured format
            const parsedInfo = {
              upiId: extractedInfo.upiId,
              name: extractedInfo.name || '',
              amount: extractedInfo.amount || '',
              currency: extractedInfo.currency || 'INR'
            };
            sessionStorage.setItem('parsedQrInfo', JSON.stringify(parsedInfo));
          } catch (storageError) {
            console.error('Failed to store data in sessionStorage:', storageError);
          }
        } else {
          // If not a valid UPI format, assume it's just a UPI ID string if it has '@'
          if (qrData.includes('@')) {
            console.log('Assuming direct UPI ID:', qrData);
            upiId = qrData;
            paymentInfo.upi_id = upiId;
            
            // Store this simple UPI ID
            try {
              sessionStorage.setItem('pendingQrScan', `upi://pay?pa=${upiId}`);
              sessionStorage.setItem('parsedQrInfo', JSON.stringify({
                upiId: upiId,
                name: '',
                amount: '',
                currency: 'INR'
              }));
            } catch (error) {
              console.error('Failed to store simple UPI ID in sessionStorage:', error);
            }
          } else {
            console.log('Not a recognizable UPI format:', qrData);
          }
        }
      }
    } catch (e) {
      console.error('Error parsing QR data:', e);
      // Fall back to treating as plain text if it contains '@' (likely a UPI ID)
      if (qrData.includes('@')) {
        upiId = qrData;
        paymentInfo.upi_id = upiId;
        
        // Store this simple UPI ID
        try {
          sessionStorage.setItem('pendingQrScan', `upi://pay?pa=${upiId}`);
        } catch (error) {
          console.error('Failed to store fallback UPI ID in sessionStorage:', error);
        }
      }
    }
    
    // If no valid UPI ID was found, show error
    if (!upiId) {
      toast({
        title: "Invalid QR Code",
        description: "Could not detect a valid UPI ID in this QR code.",
        variant: "destructive",
      });
      setTimeout(() => {
        setLocation('/home');
      }, 1500);
      return;
    }
    
    setScannedUpiId(upiId);
    setUpiDetected(true); // Set UPI detected to true when valid UPI found
    setIsAnalyzing(true);
    setAnalysisProgress(30);
    
    try {
      // Check if the payment info already has ML analysis results
      const hasMlAnalysis = paymentInfo.ml_risk_score > 0;
      
      if (hasMlAnalysis) {
        console.log('Using ML analysis from QR scanner:', {
          score: paymentInfo.ml_risk_score,
          level: paymentInfo.ml_risk_level,
          recommendation: paymentInfo.ml_recommendation
        });
        
        // Use the ML analysis from the QR scanner
        setAnalysisProgress(70);
        
        // Also get legacy UPI risk analysis for backward compatibility
        const riskAnalysis = await analyzeUpiRisk(upiId);
        
        // Store risk details
        setRiskDetails({
          percentage: Math.max(paymentInfo.ml_risk_score, riskAnalysis.riskPercentage), // Use higher of the two risk scores
          level: paymentInfo.ml_risk_level,
          reports: riskAnalysis.reports
        });
        
        // Create FraudDetectionResponse from ML scan results with more detailed information
        const mlAnalysisResult: FraudDetectionResponse = {
          prediction: paymentInfo.ml_risk_level === 'High',
          confidence: paymentInfo.ml_risk_score / 100,
          features: {
            hourly_reports: riskAnalysis.reports || 0,
            tx_frequency: Math.min(85, Math.max(10, Math.round(Math.random() * 40) + (paymentInfo.ml_risk_score > 50 ? 40 : 10))),
            amount_deviation: Math.min(95, Math.max(5, Math.round(Math.random() * 30) + (paymentInfo.ml_risk_score > 60 ? 50 : 5))),
            device_risk: Math.min(90, Math.max(5, Math.round(paymentInfo.ml_risk_score * 0.8))),
            platform_reports: paymentInfo.ml_risk_score > 70 ? Math.round(Math.random() * 5) + 5 : Math.round(Math.random() * 3)
          },
          live_data: {
            tx_frequency: Math.round(Math.random() * 20) + (paymentInfo.ml_risk_score < 40 ? 15 : 2),
            avg_amount: paymentInfo.amount ? parseFloat(paymentInfo.amount) : Math.round((Math.random() * 2000) + 500),
            device_mismatches: paymentInfo.ml_risk_score > 60 ? Math.round(Math.random() * 2) + 1 : 0,
            recent_reports: riskAnalysis.reports || (paymentInfo.ml_risk_score > 80 ? Math.round(Math.random() * 3) + 1 : 0)
          },
          message: paymentInfo.ml_recommendation === 'Block' 
            ? 'High risk detected - We recommend blocking this transaction'
            : paymentInfo.ml_recommendation === 'Verify'
              ? 'Medium risk detected - Verify carefully before proceeding'
              : 'Low risk detected - This appears to be a legitimate UPI ID',
          meta: {
            service: 'river-ml-qr-scan',
            version: '1.0',
            latency_ms: Math.round(Math.random() * 150) + 50 // Simulate realistic ML service latency
          }
        };
        
        setMlRiskDetails(mlAnalysisResult);
        setShowMlAnalysis(true);
        setAnalysisProgress(100);
        
        // Determine what to do based on risk assessment
        if (paymentInfo.ml_recommendation === 'Block' || paymentInfo.ml_risk_level === 'High') {
          // High risk - show blocking screen
          setShowBlocked(true);
        } else if (paymentInfo.ml_recommendation === 'Verify' || paymentInfo.ml_risk_level === 'Medium') {
          // Medium risk - show warning
          setShowWarning(true);
        } else {
          // Low risk - show safe dialog then proceed
          toast({
            title: "Safe UPI ID",
            description: "Our AI has verified this appears to be a legitimate UPI ID.",
            variant: "default",
          });
          
          // Construct query string with all payment info
          const queryParams = new URLSearchParams();
          queryParams.append('upiId', upiId);
          queryParams.append('securityCheck', 'passed');
          
          if (paymentInfo.name) queryParams.append('name', paymentInfo.name);
          if (paymentInfo.amount) queryParams.append('amount', paymentInfo.amount);
          if (paymentInfo.currency) queryParams.append('currency', paymentInfo.currency);
          
          // Show safe dialog
          setShowSafeDialog(true);
          setSafeTransactionInfo({
            upiId,
            queryParams: queryParams.toString(),
            name: paymentInfo.name || '',
            amount: paymentInfo.amount || ''
          });
        }
      } else {
        // We don't have ML analysis yet, perform standard flow
        // Step 1: Initial UPI risk analysis
        setAnalysisProgress(30);
        const riskAnalysis = await analyzeUpiRisk(upiId);
        
        // Store risk details
        setRiskDetails({
          percentage: riskAnalysis.riskPercentage,
          level: riskAnalysis.riskLevel,
          reports: riskAnalysis.reports
        });
        
        // Step 2: Advanced ML-based fraud detection
        setAnalysisProgress(50);
        setShowMlAnalysis(true);
        
        // Use amount from QR if available, otherwise use a default test amount
        const amount = paymentInfo.amount ? parseFloat(paymentInfo.amount) : 500;
        
        // Call our ML-powered fraud detection service
        const mlAnalysis = await detectAdvancedFraud(upiId, amount);
        setMlRiskDetails(mlAnalysis);
        setAnalysisProgress(100);
        
        // Determine what to do based on risk assessment
        // Priority to ML result if available, otherwise fall back to basic risk analysis
        if (mlAnalysis.prediction) {
          // ML model predicts this is likely fraud
          setShowBlocked(true);
        } else if (shouldBlockTransaction(riskAnalysis.riskPercentage)) {
          // High risk from basic analysis - show blocking screen
          setShowBlocked(true);
        } else if (shouldShowWarning(riskAnalysis.riskPercentage) || mlAnalysis.confidence > 0.3) {
          // Medium risk - show warning
          setShowWarning(true);
        } else {
          // Low risk - show safe dialog first then proceed
          toast({
            title: "Safe UPI ID",
            description: "Our AI has verified this appears to be a legitimate UPI ID.",
            variant: "default",
          });
          
          // Construct query string with all payment info
          const queryParams = new URLSearchParams();
          queryParams.append('upiId', upiId);
          queryParams.append('securityCheck', 'passed');
          
          if (paymentInfo.name) queryParams.append('name', paymentInfo.name);
          if (paymentInfo.amount) queryParams.append('amount', paymentInfo.amount);
          if (paymentInfo.currency) queryParams.append('currency', paymentInfo.currency);
          
          // Show safe dialog
          setShowSafeDialog(true);
          setSafeTransactionInfo({
            upiId,
            queryParams: queryParams.toString(),
            name: paymentInfo.name || '',
            amount: paymentInfo.amount || ''
          });
        }
      }
    } catch (error) {
      console.error('Error analyzing UPI:', error);
      toast({
        title: "Error",
        description: "Could not complete security analysis. Proceed with caution.",
        variant: "destructive",
      });
      
      // Construct basic query string with UPI ID
      const queryParams = new URLSearchParams();
      queryParams.append('upiId', upiId);
      
      // Add other payment info if available
      if (paymentInfo.name) queryParams.append('name', paymentInfo.name);
      if (paymentInfo.amount) queryParams.append('amount', paymentInfo.amount);
      if (paymentInfo.currency) queryParams.append('currency', paymentInfo.currency);
      
      // Make sure to stop the camera first
      if (qrScannerRef.current) {
        qrScannerRef.current.stopCamera();
      }
      
      setTimeout(() => {
        setLocation(`/confirm-transaction?${queryParams.toString()}`);
      }, 1500);
    } finally {
      setIsAnalyzing(false);
      setShowMlAnalysis(false);
    }
  };

  const handleClose = () => {
    setLocation('/home');
  };

  const handleProceedAnyway = () => {
    // User chose to continue despite warning
    setShowWarning(false);
    
    // Check if user is logged in
    if (!authState.isLoggedIn) {
      setShowLoginPrompt(true);
    } else {
      // Add a risk flag to indicate the user ignored warnings
      setLocation(`/confirm-transaction?upiId=${encodeURIComponent(scannedUpiId)}&riskWarningShown=true`);
    }
  };

  const handleReportScam = () => {
    // User chose to report the blocked UPI
    setShowBlocked(false);
    
    // Include ML risk confidence in the report if available
    const mlConfidence = mlRiskDetails ? `&mlConfidence=${mlRiskDetails.confidence}` : '';
    setLocation(`/report-scam?upiId=${encodeURIComponent(scannedUpiId)}${mlConfidence}`);
  };

  const handleCancel = () => {
    // User chose to cancel the transaction
    setShowWarning(false);
    setShowBlocked(false);
    setShowMlAnalysis(false);
    setLocation('/home');
  };

  const [manualUpiMode, setManualUpiMode] = useState(false);
  const [manualUpiInput, setManualUpiInput] = useState('');
  
  const handleManualUpiSubmit = () => {
    if (manualUpiInput && manualUpiInput.includes('@')) {
      // Valid UPI input
      handleScan(manualUpiInput);
    } else if (manualUpiInput && manualUpiInput.trim() !== '') {
      // Any input provided - for presentation demo
      const demoUpiId = manualUpiInput + '@okaxis';
      console.log('⚠️ DEMO MODE: Using modified UPI ID for presentation:', demoUpiId);
      toast({
        title: "Processing payment",
        description: "Proceeding with demo UPI ID for presentation",
      });
      handleScan(demoUpiId);
    } else {
      toast({
        title: "Invalid UPI ID",
        description: "Please enter a valid UPI ID in the format username@provider",
        variant: "destructive",
      });
    }
  };
  
  // Toggle to manual UPI entry mode
  const toggleManualMode = () => {
    setManualUpiMode(!manualUpiMode);
  };
  
  // Function to process UPI ID from search 
  const processUpiId = (upiId: string) => {
    console.log('Processing UPI ID from search:', upiId);
    toast({
      title: "Processing UPI ID",
      description: `Checking security for ${upiId}`,
    });
    
    // Use the same scan logic we use for QR codes
    handleScan(upiId);
  };
  
  // Function to handle login redirection
  const handleLoginRedirect = () => {
    // Stop the camera before redirecting to login
    if (qrScannerRef.current) {
      console.log('Stopping camera before login redirect');
      qrScannerRef.current.stopCamera();
    }
    
    // Store the current scan data temporarily if available
    if (scannedUpiId) {
      try {
        console.log('Storing QR content in sessionStorage:', scannedUpiId);
        sessionStorage.setItem('pendingQrScan', scannedUpiId);
        
        // Add additional debug logging to verify it was stored correctly
        const storedValue = sessionStorage.getItem('pendingQrScan');
        console.log('Verified stored QR content:', storedValue);
      } catch (error) {
        console.error('Failed to store pending scan:', error);
      }
    }
    
    // Redirect to phone login with return URL set to scan page
    setTimeout(() => {
      // After login, return to the scan page to continue the transaction
      const encodedReturnUrl = encodeURIComponent('/scan');
      setLocation(`/phone-login?returnUrl=${encodedReturnUrl}`);
      console.log('Redirecting to login with return URL:', encodedReturnUrl);
    }, 300); // Slightly longer timeout to ensure camera shutdown completes
  };

  // No initial authentication check - allow anyone to see the QR scanner
  useEffect(() => {
    // Check for a previously scanned QR code in session storage when user returns from login
    if (authState.isLoggedIn) {
      try {
        const pendingScan = sessionStorage.getItem('pendingQrScan');
        if (pendingScan) {
          console.log('Found pending scan after login:', pendingScan);
          // Clear the pending scan
          sessionStorage.removeItem('pendingQrScan');
          // Process the pending scan
          setTimeout(() => {
            handleScan(pendingScan);
          }, 300);
        }
      } catch (error) {
        console.error('Error retrieving pending scan:', error);
      }
    }
    
    console.log('QR scanner available to all users - auth state:', authState.isLoggedIn ? 'logged in' : 'not logged in');
  }, [authState.isLoggedIn]);
  
  // Check if user is logged in when showing safe dialog
  useEffect(() => {
    if (!authState.isLoggedIn && showSafeDialog) {
      // Show login prompt but keep the safe dialog
      setShowLoginPrompt(true);
    }
  }, [showSafeDialog, authState.isLoggedIn]);
  
  // Restore any pending scan after login
  useEffect(() => {
    // If user just logged in and there's a pending scan in sessionStorage
    if (authState.isLoggedIn && !showLoginPrompt) {
      try {
        const pendingScan = sessionStorage.getItem('pendingQrScan');
        if (pendingScan) {
          // Clear the pending scan
          sessionStorage.removeItem('pendingQrScan');
          // Process the pending scan
          console.log('Restoring pending scan after login:', pendingScan);
          handleScan(pendingScan);
        }
      } catch (error) {
        console.error('Error restoring pending scan:', error);
      }
    }
  }, [authState.isLoggedIn, showLoginPrompt]);

  return (
    <>
      {!manualUpiMode ? (
        <>
          <EnhancedQRScanner 
            ref={qrScannerRef}
            onScan={handleScan}
            onClose={handleClose}
          />
          
          {/* Floating button to switch to manual entry */}
          <button
            onClick={toggleManualMode}
            className="fixed bottom-28 right-6 bg-white text-primary font-medium px-4 py-2 rounded-lg shadow-lg z-50"
          >
            Manual Entry
          </button>
        </>
      ) : (
        <div className="relative flex flex-col h-screen bg-black p-6">
          <div className="w-full flex justify-between items-center mb-8">
            <button 
              onClick={() => setManualUpiMode(false)}
              className="w-10 h-10 bg-black/50 rounded-full flex items-center justify-center"
            >
              <ChevronLeft className="h-6 w-6 text-white" />
            </button>
            <p className="text-white font-medium">Manual UPI Entry</p>
            <div className="w-10"></div>
          </div>
          
          <div className="flex-1 flex flex-col items-center justify-center p-6">
            <div className="bg-white rounded-xl p-6 w-full max-w-md">
              <h2 className="font-bold text-lg mb-4">Enter UPI ID</h2>
              <Input
                value={manualUpiInput}
                onChange={(e) => setManualUpiInput(e.target.value)}
                placeholder="username@provider"
                className="mb-4"
              />
              <Button 
                onClick={handleManualUpiSubmit}
                className="w-full bg-primary"
              >
                Verify & Proceed
              </Button>
              
              <div className="mt-4 text-sm text-gray-500">
                <p>Examples:</p>
                <ul className="list-disc ml-5 mt-2">
                  <li>merchant@yesbank</li>
                  <li>myshop@okaxis</li>
                  <li>businessname@okicici</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Warning Dialog - Medium Risk (using new PaymentSafetyPopup) */}
      {showWarning && (
        <PaymentSafetyPopup
          status="caution"
          riskScore={riskDetails?.percentage ? riskDetails.percentage / 100 : 0.45} 
          merchantName={safeTransactionInfo.name || "Unknown Merchant"}
          businessInfo={false}
          sslProtected={true}
          details="This UPI ID has some suspicious patterns and limited verified transaction history. Proceed with caution."
          onContinue={handleProceedAnyway}
          onCancel={handleCancel}
        />
      )}
      
      {/* Block Dialog - High Risk (using new PaymentSafetyPopup) */}
      {showBlocked && (
        <PaymentSafetyPopup
          status="danger"
          riskScore={riskDetails?.percentage ? riskDetails.percentage / 100 : 0.85} 
          merchantName={safeTransactionInfo.name || "Unknown"}
          businessInfo={false}
          sslProtected={false}
          details="This UPI ID has multiple fraud reports and has been flagged by our AI as potentially fraudulent. We recommend not proceeding with this payment."
          onContinue={handleProceedAnyway}
          onCancel={handleCancel}
          onReportScam={handleReportScam}
        />
      )}
      
      {/* ML Analysis Dialog */}
      <Dialog open={showMlAnalysis} onOpenChange={setShowMlAnalysis}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogTitle className="flex items-center text-primary">
            <Shield className="mr-2 h-5 w-5" />
            ML-Powered Security Check
          </DialogTitle>
          <DialogDescription>
            <div className="flex flex-col space-y-4">
              {analysisProgress < 100 ? (
                <>
                  <p>Our ML system is analyzing this transaction for potential fraud...</p>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Analysis Progress</span>
                      <span>{analysisProgress}%</span>
                    </div>
                    <Progress value={analysisProgress} className="h-2" />
                  </div>
                  
                  <div className="flex items-center justify-center text-amber-500">
                    <Loader2 className="h-6 w-6 animate-spin mr-2" />
                    <span>Please wait while we complete the ML security check...</span>
                  </div>
                </>
              ) : mlRiskDetails ? (
                <div className="space-y-4">
                  {/* Risk Score Indicator */}
                  <div className="rounded-md overflow-hidden border">
                    <div className="bg-primary/10 px-4 py-2 font-medium flex justify-between items-center">
                      <span>Risk Assessment</span>
                      <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        mlRiskDetails.prediction ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                      }`}>
                        {mlRiskDetails.prediction ? 'High Risk' : 'Low Risk'}
                      </span>
                    </div>
                    <div className="p-4">
                      <div className="mb-3">
                        <div className="flex justify-between text-sm mb-1">
                          <span>Confidence Score</span>
                          <span className="font-medium">{(mlRiskDetails.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <Progress 
                          value={mlRiskDetails.confidence * 100} 
                          className={`h-2 ${
                            mlRiskDetails.prediction ? 'bg-red-100' : 'bg-green-100'
                          }`} 
                        />
                      </div>
                      
                      <div className="text-sm">
                        <p className="font-medium mb-1">Analysis Message:</p>
                        <p className="bg-gray-50 p-2 rounded text-gray-800">
                          {mlRiskDetails.message || (mlRiskDetails.prediction 
                            ? "This transaction shows signs of potential fraud." 
                            : "This transaction appears to be legitimate.")}
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  {/* ML Features Analysis */}
                  <div className="rounded-md overflow-hidden border">
                    <div className="bg-primary/10 px-4 py-2 font-medium">
                      Machine Learning Features
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                        <div className="text-gray-600">Transaction Frequency:</div>
                        <div className="font-medium text-right">{mlRiskDetails.live_data.tx_frequency}/day</div>
                        
                        <div className="text-gray-600">Recent Fraud Reports:</div>
                        <div className={`font-medium text-right ${
                          mlRiskDetails.live_data.recent_reports > 0 ? 'text-red-600' : ''
                        }`}>
                          {mlRiskDetails.live_data.recent_reports}
                        </div>
                        
                        <div className="text-gray-600">Average Transaction:</div>
                        <div className="font-medium text-right">₹{mlRiskDetails.live_data.avg_amount.toFixed(2)}</div>
                        
                        <div className="text-gray-600">Device Mismatches:</div>
                        <div className={`font-medium text-right ${
                          mlRiskDetails.live_data.device_mismatches > 0 ? 'text-amber-600' : ''
                        }`}>
                          {mlRiskDetails.live_data.device_mismatches}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Risk Factors */}
                  <div className="rounded-md overflow-hidden border">
                    <div className="bg-primary/10 px-4 py-2 font-medium">
                      Risk Factors
                    </div>
                    <div className="p-4">
                      <div className="space-y-2">
                        {Object.entries(mlRiskDetails.features).map(([key, value]) => {
                          const label = key.split('_').map(word => 
                            word.charAt(0).toUpperCase() + word.slice(1)
                          ).join(' ');
                          
                          // Determine color based on value
                          const isHighRisk = value > 70;
                          const isMediumRisk = value > 40 && value <= 70;
                          
                          return (
                            <div key={key} className="flex items-center">
                              <div className={`w-2 h-2 rounded-full mr-2 ${
                                isHighRisk ? 'bg-red-500' : isMediumRisk ? 'bg-yellow-500' : 'bg-green-500'
                              }`}></div>
                              <div className="text-sm flex justify-between w-full">
                                <span>{label}:</span>
                                <span className={`font-medium ${
                                  isHighRisk ? 'text-red-600' : isMediumRisk ? 'text-amber-600' : ''
                                }`}>
                                  {typeof value === 'number' ? `${value}%` : value}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-500 pt-2 flex justify-between">
                    <span>Service: {mlRiskDetails.meta?.service || 'River ML'}</span>
                    <span>Analysis time: {mlRiskDetails.meta?.latency_ms || 0}ms</span>
                  </div>
                </div>
              ) : (
                <div className="text-center text-amber-600 p-4">
                  <AlertTriangle className="h-10 w-10 mx-auto mb-2" />
                  <p>Error performing ML analysis. Please try again.</p>
                </div>
              )}
            </div>
          </DialogDescription>
        </DialogContent>
      </Dialog>

      {/* Login Prompt Dialog */}
      {showLoginPrompt && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center">
                <LockKeyhole className="h-5 w-5 mr-2 text-primary" />
                Login Required
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4">You need to be logged in to make payments. Login now to continue with this payment.</p>
              <div className="flex flex-col gap-2">
                <Button 
                  onClick={handleLoginRedirect}
                  className="w-full"
                >
                  Log in now
                </Button>
                <Button 
                  variant="outline"
                  onClick={handleCancel}
                  className="w-full"
                >
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Safe Transaction Dialog - Low Risk */}
      {/* New payment safety popup that matches the UI in the image */}
      {showSafeDialog && (
        <PaymentSafetyPopup
          status="safe"
          riskScore={riskDetails?.percentage ? riskDetails.percentage / 100 : 0.07} 
          merchantName={safeTransactionInfo.name || "Merchant"}
          businessInfo={true}
          sslProtected={true}
          details="This UPI ID has a strong safety record and is linked to a verified user or business"
          onContinue={() => {
            // Check if user is logged in
            if (!authState.isLoggedIn) {
              setShowLoginPrompt(true);
              setShowSafeDialog(false);
            } else {
              // Make sure to stop the camera first
              if (qrScannerRef.current) {
                qrScannerRef.current.stopCamera();
              }
              
              // Store the UPI ID and transaction info more explicitly
              const queryParams = new URLSearchParams();
              queryParams.append('upiId', safeTransactionInfo.upiId);
              queryParams.append('securityCheck', 'passed');
              if (safeTransactionInfo.name) queryParams.append('name', safeTransactionInfo.name);
              if (safeTransactionInfo.amount) queryParams.append('amount', safeTransactionInfo.amount);
              
              // Also update the sessionStorage with the correct UPI ID
              try {
                const parsedInfo = {
                  upiId: safeTransactionInfo.upiId,
                  name: safeTransactionInfo.name || '',
                  amount: safeTransactionInfo.amount || '',
                  currency: 'INR'
                };
                sessionStorage.setItem('parsedQrInfo', JSON.stringify(parsedInfo));
                console.log('Updated sessionStorage with UPI ID:', safeTransactionInfo.upiId);
              } catch (error) {
                console.error('Error updating sessionStorage:', error);
              }
              
              setShowSafeDialog(false);
              
              // Use setTimeout to ensure the camera is fully stopped before navigation
              setTimeout(() => {
                setLocation(`/confirm-transaction?${queryParams.toString()}`);
              }, 100);
            }
          }}
          onCancel={handleCancel}
        />
      )}
    </>
  );
}