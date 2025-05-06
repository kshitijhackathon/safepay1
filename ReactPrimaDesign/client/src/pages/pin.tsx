import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { PinPad } from '@/components/ui/pin-pad';

export default function PIN() {
  const [location, setLocation] = useLocation();

  const handlePinComplete = (pin: string) => {
    // In a real app, we would save the PIN securely
    setTimeout(() => {
      setLocation('/home');
    }, 500);
  };

  return (
    <div className="flex flex-col px-6 py-12 min-h-screen">
      <div className="mt-16 mb-12 text-center">
        <h1 className="text-3xl font-bold mb-2">Set PIN</h1>
        <p className="text-gray-500">Create a 4-digit secure PIN for transactions</p>
      </div>
      
      <PinPad 
        length={4}
        onComplete={handlePinComplete}
        className="mt-auto mb-8"
      />
    </div>
  );
}
