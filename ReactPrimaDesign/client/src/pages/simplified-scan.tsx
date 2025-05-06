import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { QRPaymentFlow } from '@/components/payment/qr-payment-flow';
import { Loader2, ArrowLeft, Smartphone } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Payment app options
const paymentApps = [
  { id: 'gpay', name: 'Google Pay', color: 'bg-white', deepLink: 'gpay://' },
  { id: 'phonepe', name: 'PhonePe', color: 'bg-[#5f259f]', deepLink: 'phonepe://' },
  { id: 'paytm', name: 'Paytm', color: 'bg-[#00baf2]', deepLink: 'paytm://' },
];

export default function SimplifiedScan() {
  const [, setLocation] = useLocation();
  const [isLoading, setIsLoading] = useState(false);
  const [showPaymentConfirm, setShowPaymentConfirm] = useState(false);
  const [paymentData, setPaymentData] = useState<any>(null);
  const { toast } = useToast();
  
  // Load Instascan library when component mounts
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://rawgit.com/schmich/instascan-builds/master/instascan.min.js';
    script.async = true;
    document.body.appendChild(script);
    
    return () => {
      document.body.removeChild(script);
    };
  }, []);
  
  const handleProcessQR = (qrData: any) => {
    console.log('Processing QR data:', qrData);
    setPaymentData(qrData);
    setShowPaymentConfirm(true);
    
    // Show toast with merchant info
    toast({
      title: "QR Scanned Successfully",
      description: `Payment to ${qrData.name} (${qrData.upiId})`,
    });
  };
  
  const handleBack = () => {
    if (showPaymentConfirm) {
      setShowPaymentConfirm(false);
      setPaymentData(null);
    } else {
      setLocation('/home');
    }
  };
  
  const handlePaymentApp = (appId: string) => {
    setIsLoading(true);
    
    // Find the selected payment app
    const app = paymentApps.find(a => a.id === appId);
    if (!app) return;
    
    // In a real app, we would create a deep link to the selected payment app
    const { upiId, name, amount } = paymentData;
    const upiDeepLink = `upi://pay?pa=${upiId}&pn=${encodeURIComponent(name)}&am=${amount}&cu=INR`;
    
    // For demo, we'll show a toast and redirect to success
    toast({
      title: `Redirecting to ${app.name}`,
      description: `Opening ${app.name} for payment of ₹${amount}`,
    });
    
    // Simulate app opening delay
    setTimeout(() => {
      // In a real app, we would open the deep link
      // window.location.href = upiDeepLink;
      
      // For demo, redirect to success page
      const successParams = new URLSearchParams();
      successParams.append('amount', amount);
      successParams.append('app', appId);
      successParams.append('merchantName', name);
      successParams.append('upiId', upiId);
      
      setLocation(`/success?${successParams.toString()}`);
    }, 1500);
  };
  
  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white p-4 flex items-center border-b">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleBack}
          className="mr-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-bold">
          {showPaymentConfirm ? 'Payment Options' : 'Scan QR Code'}
        </h1>
      </div>
      
      <div className="flex-1 p-4">
        <div className="max-w-md mx-auto">
          {!showPaymentConfirm ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-center">Scan UPI QR Code</CardTitle>
              </CardHeader>
              <CardContent>
                <QRPaymentFlow 
                  onProcessQR={handleProcessQR}
                  onCancel={handleBack}
                />
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-center">Select Payment App</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-500">Merchant</span>
                    <span className="font-medium">{paymentData.name}</span>
                  </div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-500">UPI ID</span>
                    <span>{paymentData.upiId}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Amount</span>
                    <span className="font-medium">₹{paymentData.amount}</span>
                  </div>
                </div>
                
                <div className="space-y-3">
                  {paymentApps.map((app) => (
                    <button
                      key={app.id}
                      onClick={() => handlePaymentApp(app.id)}
                      className="flex items-center w-full p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
                      disabled={isLoading}
                    >
                      <div className={`w-10 h-10 ${app.color} rounded-full flex items-center justify-center mr-3 border`}>
                        <div className="text-center font-bold text-sm">{app.name.charAt(0)}</div>
                      </div>
                      <div className="flex-1 text-left">
                        <div className="font-medium">{app.name}</div>
                      </div>
                      {isLoading && app.id === 'gpay' && (
                        <Loader2 className="h-5 w-5 animate-spin" />
                      )}
                    </button>
                  ))}
                </div>
                
                <div className="mt-4 text-center text-sm text-gray-500">
                  <p>You'll be redirected to complete the payment</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}