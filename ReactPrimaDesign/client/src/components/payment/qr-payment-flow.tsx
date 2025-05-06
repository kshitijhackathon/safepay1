import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Smartphone } from "lucide-react";
import { 
  RadioGroup, 
  RadioGroupItem 
} from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { SafetyVerification } from "./safety-verification";
import { TransactionConfirmation } from "./transaction-confirmation";

// Import Instascan when it's available in the window object
declare global {
  interface Window {
    Instascan: any;
  }
}

interface QRPaymentFlowProps {
  onProcessQR: (qrData: any) => void;
  onCancel: () => void;
}

export function QRPaymentFlow({ onProcessQR, onCancel }: QRPaymentFlowProps) {
  const [scannerActive, setScannerActive] = useState(false);
  const [scanner, setScanner] = useState<any>(null);
  const [paymentData, setPaymentData] = useState<any>(null);
  const [amount, setAmount] = useState("");
  const [selectedApp, setSelectedApp] = useState("gpay");
  const [showAppSelection, setShowAppSelection] = useState(false);
  const [showSafetyVerification, setShowSafetyVerification] = useState(false);
  const [showTransactionConfirmation, setShowTransactionConfirmation] = useState(false);
  const [merchantRiskScore, setMerchantRiskScore] = useState(85); // Default to medium-high safety
  const [isRegisteredBusiness, setIsRegisteredBusiness] = useState(true);
  const { toast } = useToast();

  const initScanner = () => {
    setScannerActive(true);
    
    // Debug log to check what scanner libraries are available
    // Declare global libraries for TypeScript
    const w = window as any;
    
    console.log("Scanner libraries:", {
      Instascan: !!w.Instascan,
      ZXing: typeof w.ZXing !== 'undefined' ? 'Available' : 'Not available',
      jsQR: typeof w.jsQR !== 'undefined' ? 'Available' : 'Not available',
    });
    
    if (!w.Instascan) {
      toast({
        title: "Scanner Not Available",
        description: "QR scanner libraries not loaded. Using demo mode.",
        variant: "destructive",
      });
      
      // Auto switch to mock data if scanner not available
      setTimeout(() => {
        useMockQRData();
      }, 1500);
      
      setScannerActive(false);
      return;
    }
    
    let newScanner = new w.Instascan.Scanner({ 
      video: document.getElementById('scanner'),
      mirror: false
    });

    setScanner(newScanner);

    w.Instascan.Camera.getCameras()
      .then((cameras: any[]) => {
        if (cameras.length > 0) {
          newScanner.start(cameras[0]);
        } else {
          toast({
            title: "Camera Error",
            description: "No cameras found on your device",
            variant: "destructive",
          });
          resetScanner();
        }
      })
      .catch((err: any) => {
        console.error('Camera error:', err);
        toast({
          title: "Camera Error",
          description: "Unable to access camera. Please check permissions.",
          variant: "destructive",
        });
        resetScanner();
      });

    newScanner.addListener('scan', (content: string) => {
      if (validateUPIQr(content)) {
        processValidQR(content);
        newScanner.stop();
      } else {
        toast({
          title: "Invalid QR Code",
          description: "Please scan a valid UPI payment QR code",
          variant: "destructive",
        });
      }
    });
  };

  const validateUPIQr = (content: string): boolean => {
    // Basic UPI QR validation
    return content.startsWith('upi://pay?') || 
           content.startsWith('https://upi://pay?');
  };

  const processValidQR = (content: string) => {
    const params = new URLSearchParams(content.split('?')[1]);
    
    // Store payment details
    const qrData = {
      upiId: params.get('pa') || 'unknown@upi',
      name: params.get('pn') || 'Merchant',
      amount: params.get('am') || ''
    };
    
    setPaymentData(qrData);
    setScannerActive(false);
    
    // Get UPI risk data for this merchant
    fetchUpiRiskData(qrData.upiId);
  };
  
  // Fetch UPI risk data from API
  const fetchUpiRiskData = async (upiId: string) => {
    try {
      const encodedUpiId = encodeURIComponent(upiId);
      const response = await fetch(`/api/upi/check/${encodedUpiId}`);
      
      if (response.ok) {
        const data = await response.json();
        // Update risk score from API
        setMerchantRiskScore(data.riskPercentage || 85);
        // Check if this is a registered business
        setIsRegisteredBusiness(data.status === 'SAFE' || data.status === 'VERIFIED');
      } else {
        // Use default values if API fails
        console.error('Failed to fetch UPI risk data');
      }
    } catch (error) {
      console.error('Error fetching UPI risk data:', error);
    }
  };

  const resetScanner = () => {
    if (scanner) {
      scanner.stop();
    }
    setScannerActive(false);
    setPaymentData(null);
  };

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Only allow numbers and a single decimal point
    const value = e.target.value.replace(/[^0-9.]/g, '');
    const parts = value.split('.');
    if (parts.length > 2) {
      // Don't allow multiple decimal points
      return;
    }
    setAmount(value);
  };

  // Redirects to UPI payment app with deep link
  const redirectToPaymentApp = () => {
    if (!paymentData || !paymentData.upiId || !amount) {
      toast({
        title: "Invalid Payment",
        description: "Missing UPI ID or amount",
        variant: "destructive",
      });
      return;
    }
    
    // Create UPI deep link
    const upiId = paymentData.upiId;
    const payeeName = paymentData.name || "Merchant";
    const amountValue = amount;
    const transactionNote = "Payment via SafePay";
    
    // Format the UPI link
    const upiLink = `upi://pay?pa=${upiId}&pn=${encodeURIComponent(payeeName)}&am=${amountValue}&tn=${encodeURIComponent(transactionNote)}`;
    
    // Log the link (for debugging)
    console.log("UPI Link:", upiLink);
    
    // Redirect to UPI app
    window.location.href = upiLink;
    
    // Fallback if UPI app not installed
    setTimeout(() => {
      if (!document.hidden) {
        toast({
          title: "No UPI App Found",
          description: `Install ${selectedApp === 'gpay' ? 'Google Pay' : selectedApp === 'phonepe' ? 'PhonePe' : 'Paytm'} to complete this payment`,
          variant: "destructive",
        });
      }
    }, 2000);
  };

  const handleProceed = () => {
    if (!amount || isNaN(parseFloat(amount)) || parseFloat(amount) <= 0) {
      toast({
        title: "Invalid Amount",
        description: "Please enter a valid payment amount",
        variant: "destructive",
      });
      return;
    }

    // Show safety verification first
    setShowSafetyVerification(true);
  };
  
  const handleContinueToTransaction = () => {
    // Show transaction confirmation screen
    setShowSafetyVerification(false);
    setShowTransactionConfirmation(true);
  };
  
  const handleBackToSafety = () => {
    // Go back to safety verification
    setShowTransactionConfirmation(false);
    setShowSafetyVerification(true);
  };
  
  const handleConfirmTransaction = () => {
    // Move to payment app selection
    setShowTransactionConfirmation(false);
    setShowAppSelection(true);
  };
  
  const handlePaymentAppSelected = () => {
    // Redirect to payment app
    redirectToPaymentApp();
    
    // Also notify parent component
    const finalData = {
      ...paymentData,
      amount: amount,
      paymentApp: selectedApp
    };
    
    onProcessQR(finalData);
  };

  // Clean up scanner on unmount
  useEffect(() => {
    return () => {
      if (scanner) {
        scanner.stop();
      }
    };
  }, [scanner]);

  // Mock QR data for demo purposes
  const useMockQRData = () => {
    // Create a mock UPI QR for demo
    const mockQRData = {
      upiId: "demostore@oksbi",
      name: "Demo Merchant",
      amount: ""
    };
    
    setPaymentData(mockQRData);
    setScannerActive(false);
    
    // Set default risk values for demo
    setMerchantRiskScore(85);
    setIsRegisteredBusiness(true);
    
    toast({
      title: "Demo Mode",
      description: "Using mock UPI data for demo purposes",
    });
  };

  return (
    <div className="qr-payment-flow">
      {!scannerActive && !paymentData && (
        <div className="text-center space-y-3">
          <Button 
            onClick={initScanner} 
            className="w-full"
          >
            Scan QR Code
          </Button>
          <Button 
            onClick={useMockQRData} 
            variant="outline"
            className="w-full"
          >
            Use Demo Data (for testing)
          </Button>
        </div>
      )}
      
      {scannerActive && (
        <Card className="mb-4">
          <CardContent className="p-4">
            <div id="scanner-container" className="relative">
              <video id="scanner" className="w-full h-auto rounded-md bg-black"></video>
              <Button 
                onClick={resetScanner}
                variant="outline" 
                className="mt-4 w-full"
              >
                Cancel Scan
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
      
      {paymentData && !showSafetyVerification && !showTransactionConfirmation && !showAppSelection && (
        <Card className="mb-4">
          <CardContent className="p-4">
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-lg">Payment Details</h3>
                <p className="text-sm text-muted-foreground">Confirm payment information</p>
              </div>
              
              <div className="space-y-4">
                <div className="rounded-lg border p-3 space-y-1">
                  <div className="text-sm text-gray-500 dark:text-gray-400">To</div>
                  <div className="font-medium">
                    {paymentData.name}
                    <div className="text-sm text-muted-foreground">{paymentData.upiId}</div>
                  </div>
                </div>
                
                {paymentData.amount && (
                  <div className="text-sm text-muted-foreground mt-2">
                    <span>Suggested Amount: </span>
                    <span className="font-medium">₹{paymentData.amount}</span>
                  </div>
                )}
              </div>
              
              <div className="space-y-2 mt-4">
                <label className="text-sm font-medium">
                  Payment Amount
                </label>
                <div className="flex items-center">
                  <span className="mr-2">₹</span>
                  <Input
                    type="text"
                    value={amount}
                    onChange={handleAmountChange}
                    placeholder={paymentData.amount || "Enter amount"}
                    className="flex-1"
                  />
                </div>
              </div>
              
              <div className="flex gap-3">
                <Button 
                  onClick={handleProceed}
                  className="flex-1"
                  disabled={!amount}
                >
                  Proceed
                </Button>
                <Button 
                  onClick={resetScanner}
                  variant="outline" 
                  className="flex-1"
                >
                  Cancel
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Safety Verification Screen */}
      {paymentData && showSafetyVerification && (
        <SafetyVerification
          merchantName={paymentData.name}
          isRegisteredBusiness={isRegisteredBusiness}
          riskScore={merchantRiskScore}
          upiId={paymentData.upiId}
          onContinue={handleContinueToTransaction}
          onCancel={() => {
            setShowSafetyVerification(false);
          }}
        />
      )}
      
      {/* Transaction Confirmation Screen */}
      {paymentData && showTransactionConfirmation && (
        <TransactionConfirmation
          userName="You" // Could be updated with user's info from context
          merchantName={paymentData.name}
          merchantUpiId={paymentData.upiId}
          amount={parseFloat(amount)}
          trustScore={merchantRiskScore}
          onConfirm={handleConfirmTransaction}
          onBack={handleBackToSafety}
        />
      )}
      
      {/* Payment App Selection Screen */}
      {paymentData && showAppSelection && (
        <Card className="mb-4">
          <CardContent className="p-4">
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-lg">Select Payment App</h3>
                <p className="text-sm text-muted-foreground">Choose an app to complete your payment</p>
              </div>
              
              <div className="space-y-4">
                <div className="rounded-lg border p-3 space-y-1">
                  <div className="text-sm text-gray-500 dark:text-gray-400">To</div>
                  <div className="font-medium">
                    {paymentData.name}
                    <div className="text-sm text-muted-foreground">{paymentData.upiId}</div>
                  </div>
                </div>
                
                <div className="rounded-lg border p-3 space-y-1">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Amount</div>
                  <div className="text-xl font-bold text-primary">₹{amount}</div>
                </div>
              </div>
              
              <RadioGroup 
                value={selectedApp} 
                onValueChange={setSelectedApp}
                className="mt-2 grid grid-cols-1 gap-3"
              >
                <div className={`border rounded-md p-3 ${selectedApp === 'gpay' ? 'border-primary bg-primary/5' : 'border-muted'}`}>
                  <RadioGroupItem value="gpay" id="gpay" className="sr-only" />
                  <Label htmlFor="gpay" className="flex items-center justify-between cursor-pointer">
                    <div className="flex items-center space-x-2">
                      <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                        <span className="text-green-600 font-bold">G</span>
                      </div>
                      <div>
                        <div className="font-medium">Google Pay</div>
                        <div className="text-sm text-muted-foreground">UPI Payments</div>
                      </div>
                    </div>
                    {selectedApp === 'gpay' && (
                      <div className="w-4 h-4 border-2 border-primary rounded-full bg-primary"></div>
                    )}
                  </Label>
                </div>
                
                <div className={`border rounded-md p-3 ${selectedApp === 'phonepe' ? 'border-primary bg-primary/5' : 'border-muted'}`}>
                  <RadioGroupItem value="phonepe" id="phonepe" className="sr-only" />
                  <Label htmlFor="phonepe" className="flex items-center justify-between cursor-pointer">
                    <div className="flex items-center space-x-2">
                      <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center">
                        <span className="text-indigo-600 font-bold">P</span>
                      </div>
                      <div>
                        <div className="font-medium">PhonePe</div>
                        <div className="text-sm text-muted-foreground">UPI Payments</div>
                      </div>
                    </div>
                    {selectedApp === 'phonepe' && (
                      <div className="w-4 h-4 border-2 border-primary rounded-full bg-primary"></div>
                    )}
                  </Label>
                </div>
                
                <div className={`border rounded-md p-3 ${selectedApp === 'paytm' ? 'border-primary bg-primary/5' : 'border-muted'}`}>
                  <RadioGroupItem value="paytm" id="paytm" className="sr-only" />
                  <Label htmlFor="paytm" className="flex items-center justify-between cursor-pointer">
                    <div className="flex items-center space-x-2">
                      <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                        <span className="text-blue-600 font-bold">P</span>
                      </div>
                      <div>
                        <div className="font-medium">Paytm</div>
                        <div className="text-sm text-muted-foreground">UPI Payments</div>
                      </div>
                    </div>
                    {selectedApp === 'paytm' && (
                      <div className="w-4 h-4 border-2 border-primary rounded-full bg-primary"></div>
                    )}
                  </Label>
                </div>
              </RadioGroup>
              
              <div className="flex gap-3 mt-4">
                <Button 
                  onClick={handlePaymentAppSelected}
                  className="flex-1"
                >
                  <Smartphone className="h-4 w-4 mr-2" />
                  Pay Now
                </Button>
                <Button 
                  onClick={() => {
                    setShowAppSelection(false);
                    setShowTransactionConfirmation(true);
                  }}
                  variant="outline" 
                  className="flex-1"
                >
                  Go Back
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}