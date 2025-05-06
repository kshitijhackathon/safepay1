import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { ArrowLeft, ChevronRight, IndianRupee, LockKeyhole, CreditCard } from 'lucide-react';
import { useAuthState } from '@/hooks/use-auth-state';

export default function Payment() {
  const [, setLocation] = useLocation();
  const [location] = useLocation();
  const { toast } = useToast();
  const { authState } = useAuthState();
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);
  
  // Check if user is logged in
  useEffect(() => {
    if (!authState.isLoggedIn && !showLoginPrompt) {
      setShowLoginPrompt(true);
    }
  }, [authState.isLoggedIn, showLoginPrompt]);
  
  // Extract payment details from URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const upiId = urlParams.get('upiId') || '';
  const merchantName = urlParams.get('name') || '';
  const urlAmount = urlParams.get('amount');
  const securityCheck = urlParams.get('securityCheck') === 'passed';
  const safetyScore = urlParams.get('safetyScore') || '90';
  
  // Use the amount from URL if available, otherwise empty string
  const [amount, setAmount] = useState<string>(urlAmount || '');
  const [note, setNote] = useState<string>('');
  const [selectedApp, setSelectedApp] = useState<string | null>(null);
  
  // Payment app options with icons
  const paymentApps = [
    { id: 'gpay', name: 'Google Pay', icon: '../../googlepay_icon.svg', color: 'bg-white' },
    { id: 'phonepe', name: 'PhonePe', icon: '../../phonepe_icon.svg', color: 'bg-[#5f259f]' },
    { id: 'paytm', name: 'Paytm', icon: '../../paytm_icon.svg', color: 'bg-[#00baf2]' },
    { id: 'card', name: 'Credit/Debit Card', icon: null, color: 'bg-[#5164BF]' },
  ];
  
  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Only allow numeric input
    const value = e.target.value.replace(/[^0-9]/g, '');
    setAmount(value);
  };
  
  const handleNoteChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNote(e.target.value);
  };
  
  const selectPaymentApp = (appId: string) => {
    setSelectedApp(appId);
  };
  
  const handleProceed = () => {
    if (!amount || parseFloat(amount) <= 0) {
      toast({
        title: 'Invalid amount',
        description: 'Please enter a valid payment amount',
        variant: 'destructive',
      });
      return;
    }
    
    if (!selectedApp) {
      toast({
        title: 'Select payment app',
        description: 'Please select a payment app to proceed',
        variant: 'destructive',
      });
      return;
    }
    
    // If card payment is selected, redirect to Stripe checkout
    if (selectedApp === 'card') {
      // Create URL params for the checkout page
      const checkoutParams = new URLSearchParams();
      checkoutParams.append('amount', amount);
      checkoutParams.append('upiId', upiId);
      checkoutParams.append('description', `Payment to ${merchantName || upiId}${note ? ` - ${note}` : ''}`);
      
      // Navigate to Stripe checkout page
      setLocation(`/checkout?${checkoutParams.toString()}`);
      return;
    }
    
    // For UPI payment apps, we would prepare deep links in a real app
    // For demo, we'll just navigate to the transaction success screen
    
    // Generate a unique transaction ID
    const transactionId = `STRIPE-${Math.random().toString(36).substring(2, 10).toUpperCase()}-${Math.random().toString(36).substring(2, 10).toUpperCase()}`;
    
    // Format current date
    const formattedDate = new Date().toLocaleString('en-US', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
    
    // Create complete payment details
    const paymentDetails = {
      upiId,
      merchantName,
      amount,
      note,
      app: selectedApp,
      safetyScore,
      timestamp: new Date().toISOString(),
      transactionId: transactionId,
      date: formattedDate,
      fromUpiId: authState.phoneNumber ? `${authState.phoneNumber.replace(/\D/g, '').slice(-10)}@ybl` : 'yourname@okhdfc'
    };
    
    // Store complete payment details in session storage for the success screen
    sessionStorage.setItem('lastPayment', JSON.stringify(paymentDetails));
    
    // Navigate to success screen without query parameters
    // We'll rely solely on session storage to avoid URL parameter issues
    setLocation(`/success`);
  };
  
  const handleBack = () => {
    setLocation(`/confirm-transaction?upiId=${encodeURIComponent(upiId)}`);
  };

  // Function to handle login redirection
  const handleLoginRedirect = () => {
    // Use a fixed return URL instead of the current URL to avoid encoding issues
    // After login, always go back to the home page which is safer
    setTimeout(() => {
      setLocation('/phone-login?returnUrl=/home');
    }, 10);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white border-b p-4 flex items-center">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleBack}
          className="mr-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-bold">Complete Payment</h1>
      </div>
      
      {/* Login Prompt Overlay */}
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
              <p className="mb-4">You need to be logged in to make payments. Login now to continue with your payment.</p>
              <div className="flex flex-col gap-2">
                <Button 
                  onClick={handleLoginRedirect}
                  className="w-full"
                >
                  Log in now
                </Button>
                <Button 
                  variant="outline"
                  onClick={handleBack}
                  className="w-full"
                >
                  Go back
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      <div className="flex-1 p-4">
        <div className="max-w-md mx-auto space-y-6">
          {/* UPI ID Display */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Merchant UPI ID</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-medium">{upiId}</div>
                  {securityCheck && (
                    <div className="text-xs px-2 py-1 bg-green-50 text-green-600 rounded-full">
                      Verified
                    </div>
                  )}
                </div>
                
                {merchantName && (
                  <div className="text-sm text-gray-600">
                    <span className="font-medium">Merchant: </span>
                    {merchantName}
                  </div>
                )}
                
                <div className="text-xs text-gray-500 flex items-center">
                  <span className="inline-block w-2 h-2 rounded-full mr-1" 
                    style={{ 
                      backgroundColor: parseInt(safetyScore) > 70 
                        ? '#10b981' 
                        : parseInt(safetyScore) > 30 
                          ? '#f59e0b' 
                          : '#ef4444'
                    }}></span>
                  Safety Score: {safetyScore}%
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Amount Entry */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Payment Amount</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <div className="flex-none w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center mr-3">
                  <IndianRupee className="h-5 w-5 text-gray-500" />
                </div>
                <Input
                  type="text"
                  value={amount}
                  onChange={handleAmountChange}
                  placeholder="Enter amount"
                  className="text-lg font-semibold"
                />
              </div>
              
              <div className="mt-4">
                <Input
                  type="text"
                  value={note}
                  onChange={handleNoteChange}
                  placeholder="Add a note (optional)"
                  className="text-sm"
                />
              </div>
            </CardContent>
          </Card>
          
          {/* Payment App Selection */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Select Payment App</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {paymentApps.map((app) => (
                <div
                  key={app.id}
                  onClick={() => selectPaymentApp(app.id)}
                  className={`flex items-center p-3 rounded-lg border cursor-pointer transition-colors
                    ${selectedApp === app.id ? 'border-primary bg-primary/5' : 'border-gray-200 hover:bg-gray-50'}`}
                >
                  <div className={`w-10 h-10 ${app.color} rounded-full flex items-center justify-center mr-3`}>
                    {app.id === 'card' ? (
                      <CreditCard className="h-5 w-5 text-white" />
                    ) : (
                      <div className="text-center font-bold text-sm">{app.name.charAt(0)}</div>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium">{app.name}</div>
                  </div>
                  <ChevronRight className={`h-5 w-5 ${selectedApp === app.id ? 'text-primary' : 'text-gray-400'}`} />
                </div>
              ))}
            </CardContent>
            <CardFooter>
              <Button 
                className="w-full" 
                size="lg"
                onClick={handleProceed}
                disabled={!amount || !selectedApp}
              >
                Proceed to Pay ₹{amount || '0'}
              </Button>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
}