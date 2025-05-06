import React, { useEffect, useState } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckCircle, Copy, Home, Share } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface PaymentDetails {
  upiId: string;
  merchantName?: string;
  amount: string;
  note?: string;
  app: string;
  safetyScore?: string;
  timestamp: string;
}

export default function PaymentSuccess() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [paymentDetails, setPaymentDetails] = useState<PaymentDetails | null>(null);
  
  // Extract payment details from URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const amount = urlParams.get('amount') || '';
  const app = urlParams.get('app') || '';
  const merchantName = urlParams.get('merchantName') || '';
  const upiId = urlParams.get('upiId') || '';
  
  // Load payment details from session storage
  useEffect(() => {
    try {
      const details = sessionStorage.getItem('lastPayment');
      if (details) {
        setPaymentDetails(JSON.parse(details));
      }
    } catch (error) {
      console.error('Failed to load payment details:', error);
    }
  }, []);
  
  // Generate a random transaction ID for demo purposes
  const transactionId = React.useMemo(() => {
    const randomPart = Math.random().toString(36).substring(2, 10).toUpperCase();
    return `TXN${randomPart}`;
  }, []);
  
  // Format timestamp for display
  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString('en-IN', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch (error) {
      return new Date().toLocaleString('en-IN');
    }
  };
  
  // Get payment app name for display
  const getAppName = (appId: string) => {
    switch (appId) {
      case 'gpay': return 'Google Pay';
      case 'phonepe': return 'PhonePe';
      case 'paytm': return 'Paytm';
      default: return appId;
    }
  };
  
  const copyTransactionId = () => {
    navigator.clipboard.writeText(transactionId).then(() => {
      toast({
        title: 'Copied',
        description: 'Transaction ID copied to clipboard',
      });
    });
  };
  
  const handleShare = () => {
    // Get merchant name for share text
    const displayName = paymentDetails?.merchantName || merchantName || 'merchant';
    const displayUpi = paymentDetails?.upiId || upiId || '';
    
    if (navigator.share) {
      navigator.share({
        title: 'Payment Receipt',
        text: `I've sent ₹${amount} to ${displayName} (${displayUpi}) via UPI. Transaction ID: ${transactionId}`,
      }).catch((error) => {
        console.log('Error sharing:', error);
      });
    } else {
      toast({
        title: 'Share not supported',
        description: 'Your browser does not support sharing',
      });
    }
  };
  
  const goHome = () => {
    setLocation('/home');
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <div className="flex-1 py-10 px-4">
        <div className="max-w-md mx-auto">
          {/* Success Animation */}
          <div className="flex flex-col items-center justify-center mb-8">
            <div className="w-20 h-20 rounded-full bg-green-100 flex items-center justify-center mb-4">
              <CheckCircle className="h-12 w-12 text-green-500" />
            </div>
            <h1 className="text-2xl font-bold text-center">Payment Successful!</h1>
            <p className="text-gray-500 text-center mt-1">
              Your payment has been processed successfully
            </p>
          </div>
          
          {/* Transaction Details Card */}
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Transaction Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between">
                <div className="text-gray-500">Amount</div>
                <div className="font-bold">₹{amount}</div>
              </div>
              
              <div className="flex justify-between">
                <div className="text-gray-500">Paid to</div>
                <div>
                  {paymentDetails?.merchantName || merchantName || 'Unknown Merchant'}
                  {(paymentDetails?.upiId || upiId) && (
                    <div className="text-xs text-gray-500">
                      {paymentDetails?.upiId || upiId}
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex justify-between">
                <div className="text-gray-500">Payment method</div>
                <div>{getAppName(app || paymentDetails?.app || '')}</div>
              </div>
              
              {paymentDetails?.note && (
                <div className="flex justify-between">
                  <div className="text-gray-500">Note</div>
                  <div>{paymentDetails.note}</div>
                </div>
              )}
              
              <div className="flex justify-between">
                <div className="text-gray-500">Date & Time</div>
                <div>{paymentDetails ? formatTime(paymentDetails.timestamp) : formatTime(new Date().toISOString())}</div>
              </div>
              
              <div className="pt-2 border-t">
                <div className="flex justify-between items-center">
                  <div className="text-gray-500">Transaction ID</div>
                  <div className="flex items-center">
                    <span className="mr-2 text-sm font-medium">{transactionId}</span>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="h-8 w-8" 
                      onClick={copyTransactionId}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Action Buttons */}
          <div className="flex gap-4">
            <Button 
              variant="outline" 
              className="flex-1" 
              onClick={handleShare}
            >
              <Share className="mr-2 h-4 w-4" />
              Share Receipt
            </Button>
            
            <Button 
              className="flex-1"
              onClick={goHome}
            >
              <Home className="mr-2 h-4 w-4" />
              Return Home
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}