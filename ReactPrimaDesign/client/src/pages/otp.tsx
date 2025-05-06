import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { OtpInput } from '@/components/ui/otp-input';

export default function OTP() {
  const [location, setLocation] = useLocation();
  const [timeLeft, setTimeLeft] = useState(30);
  const [isResending, setIsResending] = useState(false);

  // Set up timer for OTP resend
  useEffect(() => {
    if (timeLeft <= 0) return;
    
    const timer = setTimeout(() => {
      setTimeLeft(prev => prev - 1);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [timeLeft]);
  
  const handleOtpComplete = (otp: string) => {
    // In a real app, we would validate the OTP with the backend
    setLocation('/pin');
  };
  
  const handleResendOtp = () => {
    setIsResending(true);
    
    // Simulate OTP resend
    setTimeout(() => {
      setTimeLeft(30);
      setIsResending(false);
    }, 1000);
  };

  return (
    <div className="flex flex-col px-6 py-12 min-h-screen">
      <div className="mt-16 mb-12">
        <h1 className="text-3xl font-bold mb-2">Enter OTP</h1>
        <p className="text-gray-500">We've sent a verification code to +91 9876543210</p>
      </div>
      
      <OtpInput 
        length={6}
        onComplete={handleOtpComplete}
        className="mb-8"
      />
      
      <Button 
        onClick={() => setLocation('/pin')}
        className="w-full bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md"
      >
        Verify
      </Button>
      
      <p className="text-center text-sm text-gray-500 mt-8">
        Didn't receive the code?{' '}
        {timeLeft > 0 ? (
          <span>Resend OTP in {timeLeft}s</span>
        ) : (
          <button 
            onClick={handleResendOtp}
            disabled={isResending}
            className="text-primary font-medium disabled:opacity-50"
          >
            {isResending ? 'Sending...' : 'Resend OTP'}
          </button>
        )}
      </p>
    </div>
  );
}
