import React, { useState, useEffect } from 'react';
import QRCode from 'qrcode';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

// Test UPI IDs based on different risk levels
const TEST_UPI_IDS = [
  { 
    id: 'shallowfoebe@okicici', 
    name: 'Shallow Foebe', 
    description: 'Sample QR for ICICI Bank', 
    risk: 'low'
  },
  { 
    id: 'quickpay@okaxis', 
    name: 'Quick Pay Services', 
    description: 'Axis Bank merchant account', 
    risk: 'low'
  },
  { 
    id: 'mobikwik@ybl', 
    name: 'MobiKwik Wallet', 
    description: 'Digital wallet payment', 
    risk: 'medium'
  },
  { 
    id: 'fastcash.refund@hdfcbank', 
    name: 'FastCash Refunds', 
    description: 'HDFC Bank business account', 
    risk: 'medium'
  },
  { 
    id: 'instant-verify@oksbi', 
    name: 'Instant Verify', 
    description: 'SBI verification service (suspicious)', 
    risk: 'high'
  },
  { 
    id: 'support.kyc@yesbank', 
    name: 'KYC Support', 
    description: 'Yes Bank KYC (suspicious naming)', 
    risk: 'high'
  }
];

export const QRTestGenerator = () => {
  const [selectedId, setSelectedId] = useState(TEST_UPI_IDS[0].id);
  const [qrDataURL, setQrDataURL] = useState<string>('');
  const [customUpiId, setCustomUpiId] = useState('');
  const [usingCustom, setUsingCustom] = useState(false);
  const [amount, setAmount] = useState('');
  
  // Generate QR code when selected ID changes
  useEffect(() => {
    const generateQR = async () => {
      try {
        // Make sure we have a valid UPI ID
        const upiId = usingCustom ? (customUpiId || 'default@upi') : selectedId;
        
        // Format the payment string
        const paymentString = amount && !isNaN(Number(amount)) ? 
          `upi://pay?pa=${upiId}&am=${amount}&cu=INR` : 
          `upi://pay?pa=${upiId}&cu=INR`;
        
        console.log('Generating QR code for:', paymentString);
        
        // Generate QR code
        const url = await QRCode.toDataURL(paymentString, {
          width: 300,
          margin: 2,
          color: {
            dark: '#000000',
            light: '#FFFFFF'
          }
        });
        
        // Update state with the generated URL
        setQrDataURL(url);
      } catch (error) {
        console.error('Error generating QR code:', error);
      }
    };
    
    // Only generate if we have a valid ID
    if ((usingCustom && customUpiId) || (!usingCustom && selectedId)) {
      generateQR();
    }
  }, [selectedId, customUpiId, usingCustom, amount]);
  
  const selectedUpi = TEST_UPI_IDS.find(u => u.id === selectedId);
  
  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>QR Code Test Generator</CardTitle>
        <CardDescription>
          Create test QR codes for scanner testing
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col items-center space-y-4">
          {qrDataURL && (
            <div className="border-2 border-primary rounded-lg p-2 bg-white">
              <img src={qrDataURL} alt="QR Code" className="w-[250px] h-[250px]" />
            </div>
          )}
          
          <div className="text-center">
            <p className="text-lg font-semibold">
              UPI ID: {usingCustom ? customUpiId : selectedId}
            </p>
            {!usingCustom && selectedUpi && (
              <>
                <p className="text-sm text-muted-foreground">{selectedUpi.description}</p>
                <div className="mt-1">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                    ${selectedUpi.risk === 'low' ? 'bg-green-100 text-green-800' : 
                    selectedUpi.risk === 'medium' ? 'bg-yellow-100 text-yellow-800' : 
                    'bg-red-100 text-red-800'}`}>
                    {selectedUpi.risk === 'low' ? 'Low Risk' : 
                     selectedUpi.risk === 'medium' ? 'Medium Risk' : 'High Risk'}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
        
        <div className="space-y-3 pt-2">
          <div className="flex items-center space-x-2">
            <input 
              type="checkbox" 
              id="useCustom" 
              checked={usingCustom} 
              onChange={() => setUsingCustom(!usingCustom)} 
              className="rounded border-gray-300 text-primary focus:ring-primary"
            />
            <Label htmlFor="useCustom">Use custom UPI ID</Label>
          </div>
          
          {usingCustom ? (
            <div className="space-y-2">
              <Label htmlFor="customUpi">Custom UPI ID</Label>
              <Input 
                id="customUpi"
                placeholder="yourname@bankid" 
                value={customUpiId} 
                onChange={(e) => setCustomUpiId(e.target.value)} 
              />
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="upiSelect">Select Test UPI ID</Label>
              <Select value={selectedId} onValueChange={setSelectedId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select UPI ID" />
                </SelectTrigger>
                <SelectContent>
                  {TEST_UPI_IDS.map((upi) => (
                    <SelectItem key={upi.id} value={upi.id}>
                      {upi.name} ({upi.id})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          
          <div className="space-y-2">
            <Label htmlFor="amount">Amount (Optional)</Label>
            <Input 
              id="amount"
              type="number" 
              placeholder="Enter amount" 
              value={amount} 
              onChange={(e) => setAmount(e.target.value)} 
            />
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-center">
        <p className="text-xs text-muted-foreground text-center">
          Display this QR code on another device and scan it with the app to test the scanner
        </p>
      </CardFooter>
    </Card>
  );
};