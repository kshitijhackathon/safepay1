import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { isValidPhoneNumber } from '@/lib/upi';

export default function Login() {
  const [phone, setPhone] = useState('');
  const [location, setLocation] = useLocation();
  const [error, setError] = useState('');

  const handlePhoneChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.replace(/\D/g, '');
    setPhone(value);
    
    if (value && !isValidPhoneNumber(value) && value.length === 10) {
      setError('Please enter a valid 10-digit phone number');
    } else {
      setError('');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!phone) {
      setError('Phone number is required');
      return;
    }
    
    if (!isValidPhoneNumber(phone)) {
      setError('Please enter a valid 10-digit phone number');
      return;
    }
    
    // Navigate to OTP screen
    setLocation('/otp');
  };

  return (
    <div className="flex flex-col px-6 py-12 min-h-screen">
      <div className="mt-16 mb-12">
        <h1 className="text-3xl font-bold mb-2">Sign Up</h1>
        <p className="text-gray-500">Enter your phone number to continue</p>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-8">
          <Label htmlFor="phone" className="block text-sm font-medium text-gray-500 mb-2">
            Phone Number
          </Label>
          <div className="flex items-center bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm">
            <span className="text-gray-900 font-medium mr-2">+91</span>
            <Input
              type="tel"
              id="phone"
              maxLength={10}
              value={phone}
              onChange={handlePhoneChange}
              className="flex-1 bg-transparent border-none focus:ring-0 text-lg"
              placeholder="Enter phone number"
            />
          </div>
          {error && <p className="text-error text-sm mt-1">{error}</p>}
        </div>
        
        <Button 
          type="submit" 
          className="w-full bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md"
        >
          Sign Up
        </Button>
      </form>
      
      <p className="text-center text-sm text-gray-500 mt-8">
        By continuing, you agree to our{' '}
        <a href="#" className="text-primary">Terms of Service</a> and{' '}
        <a href="#" className="text-primary">Privacy Policy</a>
      </p>
    </div>
  );
}
