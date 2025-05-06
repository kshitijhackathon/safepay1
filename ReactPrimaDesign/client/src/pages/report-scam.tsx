import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScamType } from '@/lib/scam-detection';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';
import { useAuthState } from '@/hooks/use-auth-state';

export default function ReportScam() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { authState } = useAuthState();
  
  const [upiId, setUpiId] = useState('');
  const [scamType, setScamType] = useState<ScamType>(ScamType.Unknown);
  const [amountLost, setAmountLost] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Allow numbers, commas, and decimals
    const value = e.target.value.replace(/[^0-9,.]/g, '');
    setAmountLost(value);
  };

  const handleSubmit = async () => {
    // Validate inputs
    if (!upiId) {
      toast({
        title: "UPI ID Required",
        description: "Please enter the UPI ID of the scammer",
        variant: "destructive"
      });
      return;
    }

    if (!authState.isLoggedIn || !authState.userId) {
      toast({
        title: "Login Required",
        description: "Please login to submit a scam report",
        variant: "destructive"
      });
      setLocation('/login');
      return;
    }
    
    setIsSubmitting(true);
    
    // Convert amount to numeric value (paise/cents)
    const amountLostNumeric = amountLost ? 
      Math.round(parseFloat(amountLost.replace(/,/g, '')) * 100) : 
      null;

    try {
      const response = await apiRequest('POST', '/api/scam-reports', {
        userId: parseInt(authState.userId),
        upiId,
        scamType,
        amountLost: amountLostNumeric,
        description: description || null
      });

      if (response.ok) {
        toast({
          title: "Report Submitted",
          description: "Thank you for helping make UPI safer for everyone.",
          variant: "default"
        });
        setLocation('/home');
      } else {
        const error = await response.json();
        throw new Error(error.message || 'Failed to submit report');
      }
    } catch (err) {
      const error = err as Error;
      toast({
        title: "Submission Failed",
        description: error.message || "An error occurred. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col px-6 py-8">
      <div className="flex justify-between items-center mb-6">
        <button
          onClick={() => setLocation('/home')}
          className="w-10 h-10 bg-[#F5F6FA] rounded-full flex items-center justify-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-6 h-6"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15.75 19.5L8.25 12l7.5-7.5"
            />
          </svg>
        </button>
        <h1 className="text-xl font-bold">Report Scam</h1>
        <div className="w-10"></div>
      </div>
      
      <Card className="bg-white rounded-2xl p-6 shadow-md border border-gray-100 mb-6">
        <Label htmlFor="upi-id" className="block text-sm text-gray-500 mb-2">UPI ID</Label>
        <Input
          id="upi-id"
          type="text"
          value={upiId}
          onChange={(e) => setUpiId(e.target.value)}
          className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 border-none focus:ring-0"
        />
        
        <Label htmlFor="scam-type" className="block text-sm text-gray-500 mt-4 mb-2">Scam Type</Label>
        <Select value={scamType} onValueChange={(value) => setScamType(value as ScamType)}>
          <SelectTrigger className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 border-none focus:ring-0">
            <SelectValue placeholder="Select scam type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ScamType.Unknown}>Fake Products</SelectItem>
            <SelectItem value={ScamType.Phishing}>Phishing</SelectItem>
            <SelectItem value={ScamType.Banking}>Banking Fraud</SelectItem>
            <SelectItem value={ScamType.KYC}>KYC Scam</SelectItem>
            <SelectItem value={ScamType.Lottery}>Lottery Scam</SelectItem>
            <SelectItem value={ScamType.Refund}>Refund Scam</SelectItem>
            <SelectItem value={ScamType.Reward}>Reward Scam</SelectItem>
          </SelectContent>
        </Select>
        
        <Label htmlFor="amount-lost" className="block text-sm text-gray-500 mt-4 mb-2">Amount Lost (if any)</Label>
        <div className="flex items-center bg-[#F5F6FA] rounded-xl px-4 py-3">
          <span className="text-gray-900 mr-2">₹</span>
          <Input
            id="amount-lost"
            type="text"
            value={amountLost}
            onChange={handleAmountChange}
            className="flex-1 bg-transparent border-none focus:ring-0"
          />
        </div>
        
        <Label htmlFor="description" className="block text-sm text-gray-500 mt-4 mb-2">Description (optional)</Label>
        <Textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 h-24 border-none focus:ring-0"
          placeholder="Describe what happened..."
        />
      </Card>
      
      <Card className="bg-white rounded-2xl p-6 shadow-md border border-gray-100 mb-6">
        <Label className="block text-sm text-gray-500 mb-4">Add Screenshots (optional)</Label>
        <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 flex flex-col items-center justify-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-10 h-10 text-gray-500 mb-3"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
          <p className="text-gray-500 text-sm text-center">Tap to upload screenshots</p>
        </div>
      </Card>
      
      <Button
        onClick={handleSubmit}
        disabled={isSubmitting}
        className="bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md text-center"
      >
        {isSubmitting ? "Submitting..." : "Submit Report"}
      </Button>
    </div>
  );
}
