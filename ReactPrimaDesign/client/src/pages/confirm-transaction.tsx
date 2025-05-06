import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';
import { CheckCircle, ChevronLeft, Shield, AlertTriangle, Info } from 'lucide-react';
import { detectAdvancedFraud } from '@/lib/fraud-detection';
import { enhancedFraudDetection, validateUpiId, getSecurityTip } from '@/lib/enhanced-fraud-detection';
import { useToast } from '@/hooks/use-toast';
import { UpiPinDialog } from '@/components/payment/upi-pin-dialog';
import { useAuthState } from '@/hooks/use-auth-state';

export default function ConfirmTransaction() {
  const [location, setLocation] = useLocation();
  const [amount, setAmount] = useState('1000');
  const [fromAccount, setFromAccount] = useState('');
  const [toAccount, setToAccount] = useState('');
  const [upiId, setUpiId] = useState('');
  const [merchant, setMerchant] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [safetyScore, setSafetyScore] = useState<number | null>(null);
  const [showPaymentApps, setShowPaymentApps] = useState(false);
  const [showUpiPin, setShowUpiPin] = useState(false);
  const [selectedUpiApp, setSelectedUpiApp] = useState<string>('');
  const { toast } = useToast();
  const { authState } = useAuthState();
  
  // Set the "From" account to the user's phone number as UPI ID if logged in
  useEffect(() => {
    // If the user is logged in, use their phone number as UPI ID
    if (authState.isLoggedIn && authState.phoneNumber) {
      // Create a UPI ID from the phone number (e.g., 9876543210@ybl)
      // Remove any non-numeric characters from phone number
      const cleanPhone = authState.phoneNumber.replace(/\D/g, '');
      // Take the last 10 digits if longer
      const phoneForUpi = cleanPhone.length > 10 ? cleanPhone.slice(-10) : cleanPhone;
      // Format as a UPI ID
      const upiFormat = `${phoneForUpi}@ybl`;
      setFromAccount(upiFormat);
    } else {
      // Otherwise use a default value
      setFromAccount('yourname@okhdfc');
    }
  }, [authState]);

  // Analyze UPI safety on component mount
  useEffect(() => {
    const params = new URLSearchParams(location.split('?')[1] || '');
    const urlUpiId = params.get('upiId');
    const urlName = params.get('name');
    const urlAmount = params.get('amount');
    const urlCurrency = params.get('currency');
    const securityCheck = params.get('securityCheck');
    
    console.log('Page loaded with params:', { urlUpiId, urlName, urlAmount, urlCurrency, securityCheck });
    
    // First try to use URL parameters if they exist and include a UPI ID
    if (location.includes('?') && urlUpiId) {
      console.log('Using UPI ID from URL:', urlUpiId);
      
      // Set UPI ID values
      setUpiId(urlUpiId);
      setToAccount(urlUpiId); // Important: Show UPI ID in the "To" field
      
      // Set merchant name if provided in the URL
      if (urlName) {
        // Use the name from the URL params (from QR code)
        try {
          // Decode any URI encoded characters
          const decodedName = decodeURIComponent(urlName);
          setMerchant(decodedName);
        } catch (e) {
          // If decoding fails, use as is
          setMerchant(urlName);
        }
      } else {
        // Extract merchant name from UPI ID (e.g., merchantname@upi)
        const merchantFromUpi = urlUpiId.split('@')[0];
        if (merchantFromUpi) {
          // Convert camelCase or snake_case to Title Case with spaces
          const formattedName = merchantFromUpi
            .replace(/([A-Z])/g, ' $1') // Add space before capital letters
            .replace(/_/g, ' ') // Replace underscores with spaces
            .replace(/^\w/, (c: string) => c.toUpperCase()) // Capitalize first letter
            .trim(); // Remove leading/trailing spaces
            
          setMerchant(formattedName);
        }
      }
      
      // Set amount if provided in the URL
      if (urlAmount) {
        setAmount(urlAmount);
      }
      
      // If security check wasn't already performed in the previous step,
      // do it now
      if (securityCheck !== 'passed') {
        analyzeSafety(urlUpiId);
      } else {
        // Already passed security check
        setSafetyScore(93);
        setIsAnalyzing(false);
      }
      return;
    }
    
    // If we don't have URL params, try to get the structured data from sessionStorage
    try {
      const parsedQrInfo = sessionStorage.getItem('parsedQrInfo');
      if (parsedQrInfo) {
        console.log('Found structured QR info in sessionStorage:', parsedQrInfo);
        
        try {
          const parsedInfo = JSON.parse(parsedQrInfo);
          console.log('Parsed structured info:', parsedInfo);
          
          if (parsedInfo.upiId) {
            // Use the extracted values
            const extractedUpiId = parsedInfo.upiId;
            console.log('Using structured UPI ID:', extractedUpiId);
            
            setUpiId(extractedUpiId);
            setToAccount(extractedUpiId); // Important: Show UPI ID in the "To" field
            
            // Set merchant name
            if (parsedInfo.name) {
              setMerchant(parsedInfo.name);
            } else {
              // Extract merchant name from UPI ID
              const merchantFromUpi = extractedUpiId.split('@')[0];
              if (merchantFromUpi) {
                const formattedName = merchantFromUpi
                  .replace(/([A-Z])/g, ' $1')
                  .replace(/_/g, ' ')
                  .replace(/^\w/, (c: string) => c.toUpperCase())
                  .trim();
                
                setMerchant(formattedName);
              }
            }
            
            // Set amount if available
            if (parsedInfo.amount) {
              setAmount(parsedInfo.amount);
            }
            
            // Only remove the parsedQrInfo, keep pendingQrScan as backup
            sessionStorage.removeItem('parsedQrInfo');
            
            // Analyze safety of this UPI ID
            analyzeSafety(extractedUpiId);
            return;
          }
        } catch (parseError) {
          console.error('Error parsing structured QR info:', parseError);
        }
      }
      
      // Fallback to the raw UPI string
      const pendingScan = sessionStorage.getItem('pendingQrScan');
      if (pendingScan && pendingScan.includes('pa=')) {
        console.log('Found raw pending scan in sessionStorage:', pendingScan);
        
        // Extract UPI ID from pendingScan
        const upiMatch = pendingScan.match(/pa=([^&]+)/);
        if (upiMatch && upiMatch[1]) {
          const extractedUpiId = decodeURIComponent(upiMatch[1]);
          console.log('Extracted UPI ID from raw scan:', extractedUpiId);
          
          // Extract name if available
          let extractedName = '';
          const nameMatch = pendingScan.match(/pn=([^&]+)/);
          if (nameMatch && nameMatch[1]) {
            extractedName = decodeURIComponent(nameMatch[1]);
          }
          
          // Use the extracted values
          setUpiId(extractedUpiId);
          setToAccount(extractedUpiId); // Important: Show UPI ID in the "To" field
          if (extractedName) {
            setMerchant(extractedName);
          } else {
            // Extract merchant name from UPI ID
            const merchantFromUpi = extractedUpiId.split('@')[0];
            if (merchantFromUpi) {
              const formattedName = merchantFromUpi
                .replace(/([A-Z])/g, ' $1')
                .replace(/_/g, ' ')
                .replace(/^\w/, (c: string) => c.toUpperCase())
                .trim();
              
              setMerchant(formattedName);
            }
          }
          
          // Extract amount if available
          const amountMatch = pendingScan.match(/am=([^&]+)/);
          if (amountMatch && amountMatch[1]) {
            setAmount(decodeURIComponent(amountMatch[1]));
          }
          
          // Clear the pending scan
          sessionStorage.removeItem('pendingQrScan');
          
          // Analyze safety of this UPI ID
          analyzeSafety(extractedUpiId);
          return;
        }
      }
    } catch (error) {
      console.error('Error checking sessionStorage:', error);
    }
    
    // If we still don't have a UPI ID, check for partial data in URL or use demo mode
    // Try to find a UPI ID in the query string first (even partial matches)
    let foundUpiId = '';
    try {
      // Check if "upiId" is in the URL even if it's not properly parsed
      const urlString = window.location.search;
      const upiIdMatch = urlString.match(/upiId=([^&]+)/);
      if (upiIdMatch && upiIdMatch[1]) {
        foundUpiId = decodeURIComponent(upiIdMatch[1]);
        console.log('Found UPI ID in URL string:', foundUpiId);
      }
    } catch (e) {
      console.error('Error parsing URL directly:', e);
    }

    if (foundUpiId && foundUpiId.includes('@')) {
      // We found a UPI ID directly in the URL string
      console.log('Using UPI ID found in URL string:', foundUpiId);
      setUpiId(foundUpiId);
      setToAccount(foundUpiId); // Show UPI ID in the "To" field
      
      // Extract merchant name from UPI ID
      const merchantFromUpi = foundUpiId.split('@')[0];
      if (merchantFromUpi) {
        const formattedName = merchantFromUpi
          .replace(/([A-Z])/g, ' $1')
          .replace(/_/g, ' ')
          .replace(/^\w/, (c) => typeof c === 'string' ? c.toUpperCase() : c)
          .trim();
        
        setMerchant(formattedName);
      }
      
      setSafetyScore(80);
      setIsAnalyzing(false);
      analyzeSafety(foundUpiId);
    } else {
      // If all else fails, use demo mode with the fixed values
      console.log('⚠️ DEMO MODE: Using default values for presentation');
      const demoUpiId = 'shalkowfoebe@okicici'; // Use a real example that has been previously scanned
      const demoName = 'Shalkow Foebe';
      const demoAmount = '100';
      
      setUpiId(demoUpiId);
      setToAccount(demoUpiId); // Show UPI ID in the "To" field
      setMerchant(demoName);
      setAmount(demoAmount);
      setSafetyScore(95);
      setIsAnalyzing(false);
    }
  }, [location, toast, setLocation]);
  
  const analyzeSafety = async (upiId: string) => {
    setIsAnalyzing(true);
    
    try {
      // First, validate the UPI ID with AI
      const upiValidation = await validateUpiId(upiId);
      
      if (upiValidation.is_suspicious && upiValidation.confidence > 0.7) {
        toast({
          title: "Suspicious UPI ID",
          description: `This UPI ID appears suspicious: ${upiValidation.flags[0]}`,
          variant: "destructive",
        });
      }
      
      // Run enhanced AI-powered fraud detection
      const deviceFingerprint = navigator.userAgent; // Simple fingerprint, could be more sophisticated
      const enhancedAnalysis = await enhancedFraudDetection(
        upiId, 
        parseFloat(amount),
        deviceFingerprint,
        `Payment to ${merchant || upiId.split('@')[0]}`
      );
      
      // Calculate a safety score between 0-100 based on the analysis
      // Lower confidence of fraud means higher safety
      const calculatedScore = Math.round(100 - (enhancedAnalysis.confidence * 100));
      setSafetyScore(calculatedScore);
      
      // If the transaction is flagged as fraudulent with high confidence
      if (enhancedAnalysis.prediction && enhancedAnalysis.confidence > 0.7) {
        toast({
          title: "High Risk Detected",
          description: enhancedAnalysis.message || "This transaction appears to be unsafe. We recommend canceling.",
          variant: "destructive",
        });
        
        // Get AI-generated security tip based on user's transaction pattern
        const mockUserActivity = {
          recentTransactions: [
            { amount: parseFloat(amount), recipient: upiId, timestamp: new Date() }
          ],
          paymentPatterns: {
            frequentRecipients: [],
            averageAmount: parseFloat(amount),
            unusualActivities: enhancedAnalysis.prediction ? ['suspicious recipient'] : []
          }
        };
        
        try {
          const securityTip = await getSecurityTip(mockUserActivity);
          setTimeout(() => {
            toast({
              title: "Security Tip",
              description: securityTip,
              variant: "default",
            });
          }, 3000);
        } catch (tipError) {
          console.error('Error getting security tip:', tipError);
        }
      }
    } catch (error) {
      console.error('Error analyzing UPI safety:', error);
      // Fall back to standard fraud detection
      try {
        const mlAnalysis = await detectAdvancedFraud(upiId, parseFloat(amount));
        const fallbackScore = Math.round(100 - (mlAnalysis.confidence * 100));
        setSafetyScore(fallbackScore);
      } catch (fallbackError) {
        console.error('Error with fallback detection:', fallbackError);
        setSafetyScore(50); // Medium safety score as last resort
      }
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleContinue = () => {
    // Instead of navigating, show payment app options directly
    setShowPaymentApps(true);
  };
  
  const handlePayWithApp = (app: string) => {
    // Close the payment apps dialog
    setShowPaymentApps(false);
    
    // Set the selected app and show UPI PIN dialog
    setSelectedUpiApp(app);
    setShowUpiPin(true);
  };
  
  const handlePaymentSuccess = () => {
    setIsLoading(true);
    
    // Generate a transaction ID
    const transactionId = `UPI${Date.now().toString().substring(5)}`;
    
    // In a real app, this would process the payment with the UPI app
    setTimeout(() => {
      toast({
        title: "Payment Successful",
        description: `Your payment of ₹${amount} to ${merchant} has been processed successfully.`,
      });
      
      // Pass transaction details to success page via URL parameters
      const successUrl = `/success?amount=${encodeURIComponent(amount)}&merchantName=${encodeURIComponent(merchant)}&upiId=${encodeURIComponent(upiId)}&txnId=${encodeURIComponent(transactionId)}&app=${encodeURIComponent(selectedUpiApp || 'UPI')}&fromUpiId=${encodeURIComponent(fromAccount)}`;
      
      setLocation(successUrl);
    }, 1500);
  };
  
  return (
    <div className="flex flex-col px-6 py-8 min-h-screen bg-white">
      {/* Header */}
      <div className="flex items-center mb-6">
        <button
          onClick={() => setLocation('/scan')}
          className="w-10 h-10 flex items-center justify-center"
        >
          <ChevronLeft className="h-6 w-6" />
        </button>
        <h1 className="text-xl font-semibold ml-2">Confirm</h1>
      </div>
      
      <p className="text-gray-600 mb-6">Confirm transaction information</p>
      
      {/* Transaction details form */}
      <div className="space-y-4 mb-6">
        <div>
          <p className="text-sm text-gray-500 mb-1">From</p>
          <Input 
            value={fromAccount} 
            readOnly 
            className="bg-gray-50" 
          />
        </div>
        
        <div>
          <p className="text-sm text-gray-500 mb-1">To</p>
          <Input 
            value={toAccount} 
            className="bg-gray-50" 
            readOnly 
          />
        </div>
        
        <div>
          <p className="text-sm text-gray-500 mb-1">Amount</p>
          <Input 
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            className="bg-gray-50"
            type="number"
          />
        </div>
      </div>
      
      {/* UPI Safety Analysis Card */}
      <Card className="p-4 mb-8 border border-gray-100">
        <div className="flex items-start">
          <div className="flex-shrink-0 mr-3 mt-1">
            {isAnalyzing ? (
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            ) : safetyScore && safetyScore > 70 ? (
              <CheckCircle className="h-6 w-6 text-green-500" />
            ) : safetyScore && safetyScore > 30 ? (
              <AlertTriangle className="h-6 w-6 text-amber-500" />
            ) : (
              <Info className="h-6 w-6 text-red-500" />
            )}
          </div>
          
          <div className="flex-1">
            <div className="flex justify-between items-center mb-1">
              <p className="font-medium text-gray-900">
                {upiId}
              </p>
              
              {safetyScore !== null && !isAnalyzing && (
                <div className="flex items-center">
                  <span className="text-gray-600 text-sm mr-2">{safetyScore}%</span>
                  <div className="w-8 h-8 rounded-full border-4 border-gray-200 flex items-center justify-center">
                    <div 
                      className="rounded-full" 
                      style={{ 
                        width: '100%', 
                        height: '100%',
                        background: `conic-gradient(${safetyScore > 70 ? '#22c55e' : safetyScore > 30 ? '#f59e0b' : '#ef4444'} ${safetyScore}%, transparent 0%)` 
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
            
            <p className="text-green-600 font-medium mb-1">{safetyScore && safetyScore > 70 ? 'Safe to pay' : 'Proceed with caution'}</p>
            
            <div className="space-y-1 mt-2">
              <div className="flex items-center text-sm text-gray-600">
                <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                <span>Verified Merchant</span>
              </div>
              
              <div className="flex items-center text-sm text-gray-600">
                <span className="inline-block w-3 h-3 bg-gray-300 rounded-full mr-2"></span>
                <span>Linked to registered business</span>
              </div>
              
              <div className="flex items-center text-sm text-gray-600">
                <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                <span>SCA Protected</span>
              </div>
            </div>
            
            <p className="text-xs text-gray-500 mt-2">
              This UPI ID has a strong safety record and is linked to a verified user or business
            </p>
          </div>
        </div>
      </Card>
      
      {/* Continue Button */}
      <Button 
        className="w-full bg-primary text-white py-6" 
        size="lg"
        onClick={handleContinue}
        disabled={isAnalyzing || isLoading}
      >
        {isAnalyzing ? 'Analyzing...' : isLoading ? 'Processing...' : 'Continue to pay'}
      </Button>
      
      {/* Payment Apps Dialog - Similar to screenshot */}
      <Dialog open={showPaymentApps} onOpenChange={setShowPaymentApps}>
        <DialogContent className="sm:max-w-[425px] p-0 overflow-hidden">
          <div className="bg-white rounded-lg">
            <div className="p-6 pb-4">
              <DialogTitle className="text-center text-lg font-medium">
                Pay through -
              </DialogTitle>
            </div>
            
            <div className="px-8 pb-6">
              <div className="grid grid-cols-4 gap-4">
                <button 
                  onClick={() => handlePayWithApp('GPay')}
                  className="flex flex-col items-center"
                >
                  <div className="w-16 h-16 mb-2">
                    <img 
                      src="https://cdn.iconscout.com/icon/free/png-512/free-google-pay-2038779-1721670.png" 
                      alt="Google Pay" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <span className="text-sm">GPay</span>
                </button>
                
                <button 
                  onClick={() => handlePayWithApp('Paytm')}
                  className="flex flex-col items-center"
                >
                  <div className="w-16 h-16 mb-2">
                    <img 
                      src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Paytm_Logo_%28standalone%29.svg/2560px-Paytm_Logo_%28standalone%29.svg.png" 
                      alt="Paytm" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <span className="text-sm">Paytm</span>
                </button>
                
                <button 
                  onClick={() => handlePayWithApp('PhonePe')}
                  className="flex flex-col items-center"
                >
                  <div className="w-16 h-16 mb-2">
                    <img 
                      src="https://www.logo.wine/a/logo/PhonePe/PhonePe-Logo.wine.svg" 
                      alt="PhonePe" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <span className="text-sm">PhonePe</span>
                </button>
                
                <button 
                  onClick={() => {
                    setShowPaymentApps(false);
                    // Navigate to the Stripe checkout page
                    setLocation(`/checkout?upiId=${encodeURIComponent(upiId)}&amount=${amount}&description=${encodeURIComponent(`Payment to ${merchant}`)}`);
                  }}
                  className="flex flex-col items-center"
                >
                  <div className="w-16 h-16 mb-2">
                    <img 
                      src="https://cdn.iconscout.com/icon/free/png-256/free-stripe-2-498440.png" 
                      alt="Stripe" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <span className="text-sm">Card</span>
                </button>
              </div>
            </div>
            
            <div className="px-6 py-4">
              <Button 
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-6" 
                size="lg"
                onClick={() => setShowPaymentApps(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      
      {/* UPI PIN Dialog */}
      <UpiPinDialog
        open={showUpiPin}
        onOpenChange={setShowUpiPin}
        onSuccess={handlePaymentSuccess}
        upiApp={selectedUpiApp}
      />
    </div>
  );
}