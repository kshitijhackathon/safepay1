import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { X, ArrowLeft, Check } from 'lucide-react';
import { analyzeQRWithOptimizedML } from '@/lib/enhanced-optimized-qr-scanner';

export default function ManualUpiEntry() {
  const [, setLocation] = useLocation();
  const [upiId, setUpiId] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const handleClose = () => {
    setLocation('/home');
  };
  
  const handleSubmit = async () => {
    // Simple UPI validation
    if (!upiId.trim()) {
      setError('Please enter a UPI ID');
      return;
    }
    
    // Add default domain if missing
    let processedUpiId = upiId;
    if (!processedUpiId.includes('@')) {
      processedUpiId += '@okaxis'; // Add a default bank
    }
    
    setIsProcessing(true);
    
    // Extract merchant name from UPI ID
    const merchantFromUpi = processedUpiId.split('@')[0];
    let merchantName = 'Merchant';
    
    if (merchantFromUpi) {
      // Convert camelCase or snake_case to Title Case with spaces
      merchantName = merchantFromUpi
        .replace(/([A-Z])/g, ' $1') // Add space before capital letters
        .replace(/_/g, ' ') // Replace underscores with spaces
        .replace(/^\w/, (c) => c.toUpperCase()) // Capitalize first letter
        .trim(); // Remove leading/trailing spaces
    }
    
    // Create a payment info object
    let paymentInfo = {
      upi_id: processedUpiId,
      name: merchantName,
      amount: '',  // Let user fill this on the payment page
      currency: 'INR',
      ml_risk_score: 0,
      ml_risk_level: 'Low' as 'Low' | 'Medium' | 'High',
      ml_recommendation: 'Allow' as 'Allow' | 'Verify' | 'Block'
    };
    
    // Construct a UPI URL for ML analysis
    const upiUrl = `upi://pay?pa=${processedUpiId}&pn=Demo%20Merchant&am=100&cu=INR&tn=Payment`;
    
    try {
      // Analyze the UPI with ML
      const mlResult = await analyzeQRWithOptimizedML(upiUrl);
      
      // Update payment info with ML risk details
      paymentInfo.ml_risk_score = mlResult.risk_score;
      paymentInfo.ml_risk_level = mlResult.risk_level;
      paymentInfo.ml_recommendation = mlResult.recommendation;
      
      // Store the payment info
      try {
        sessionStorage.setItem('lastScannedQR', JSON.stringify(paymentInfo));
      } catch (error) {
        console.error('Error storing payment info:', error);
      }
      
      // Navigate to scan page
      setLocation('/scan?qrData=' + encodeURIComponent(JSON.stringify(paymentInfo)));
    } catch (error) {
      console.error('Error analyzing UPI:', error);
      setError('Error analyzing UPI. Please try again.');
      setIsProcessing(false);
    }
  };
  
  return (
    <div className="h-screen w-full bg-primary flex flex-col">
      {/* Header */}
      <div className="p-4 flex items-center justify-between bg-primary text-white">
        <button onClick={handleClose} className="p-2">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <p className="text-white font-medium">Enter UPI ID</p>
        <div className="w-10"></div>
      </div>
      
      {/* Content */}
      <div className="flex-1 flex flex-col p-6">
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="w-full max-w-md">
            <div className="mb-6">
              <label className="block text-white text-sm font-medium mb-2">Enter UPI ID</label>
              <div className="relative">
                <input
                  type="text"
                  value={upiId}
                  onChange={(e) => setUpiId(e.target.value)}
                  placeholder="username@bank"
                  className="bg-white/10 border border-white/40 text-white w-full p-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                {upiId && (
                  <button
                    onClick={() => setUpiId('')}
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 text-white/60"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>
              <p className="text-white/60 text-xs mt-2">
                Example: johndoe@okicici or 9876543210@paytm
              </p>
              
              {error && (
                <div className="bg-red-500/70 text-white px-4 py-2 rounded-lg text-center mt-2">
                  {error}
                </div>
              )}
            </div>
            
            <button
              onClick={handleSubmit}
              disabled={isProcessing}
              className="w-full bg-white text-primary hover:bg-blue-100 px-6 py-4 rounded-lg text-lg font-medium flex items-center justify-center"
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin mr-2 h-5 w-5 border-2 border-primary border-t-transparent rounded-full"></div>
                  Processing...
                </>
              ) : (
                <>Verify UPI</>
              )}
            </button>
            
            <button
              onClick={() => setLocation('/qr-scan')}
              className="text-white flex items-center justify-center gap-2 mt-4"
            >
              <ArrowLeft size={16} />
              Back to QR Scanner
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}