import React, { useState } from 'react';
import { Dialog, DialogContent, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { LoaderCircle } from 'lucide-react';

interface UpiPinDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
  onRedirectToApp?: () => void; // Function to retry UPI deep link redirection
  upiData?: {
    upiId: string;
    amount: string;
    name: string;
    paymentApp?: string;
  }; // UPI payment data for the transaction
  upiApp?: string; // The selected UPI app name
}

export function UpiPinDialog({
  open,
  onOpenChange,
  onSuccess,
  onRedirectToApp,
  upiData,
  upiApp = 'UPI'
}: UpiPinDialogProps) {
  const [upiPin, setUpiPin] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [showAppNotFound, setShowAppNotFound] = useState(!!onRedirectToApp);
  const { toast } = useToast();
  
  // Demo PIN for testing
  const DEMO_PIN = '979480';
  
  const handleVerifyPin = () => {
    if (upiPin.length !== 6) {
      toast({
        title: "Invalid PIN",
        description: "Please enter a 6-digit UPI PIN",
        variant: "destructive",
      });
      return;
    }
    
    setIsVerifying(true);
    
    // Simulate PIN verification
    setTimeout(() => {
      if (upiPin === DEMO_PIN) {
        toast({
          title: "Payment Successful",
          description: "Your transaction has been processed successfully",
        });
        onSuccess();
        onOpenChange(false);
      } else {
        toast({
          title: "Incorrect PIN",
          description: "The UPI PIN you entered is incorrect",
          variant: "destructive",
        });
      }
      setIsVerifying(false);
    }, 1500);
  };
  
  const handlePinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    // Only allow digits and limit to 6 characters
    if (/^\d*$/.test(value) && value.length <= 6) {
      setUpiPin(value);
    }
  };
  
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        {showAppNotFound && onRedirectToApp ? (
          <>
            <DialogTitle className="text-center">UPI App Not Found</DialogTitle>
            <DialogDescription className="text-center">
              {upiApp === 'gpay' ? 'Google Pay' : upiApp === 'phonepe' ? 'PhonePe' : upiApp === 'paytm' ? 'Paytm' : 'UPI'} 
              app was not detected on your device
            </DialogDescription>
            
            <div className="flex flex-col items-center gap-4 py-4">
              {upiData && (
                <div className="w-full bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Payment to:</span>
                    <span>{upiData.name}</span>
                  </div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">UPI ID:</span>
                    <span className="font-mono text-xs">{upiData.upiId}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Amount:</span>
                    <span className="text-primary font-bold">₹{upiData.amount}</span>
                  </div>
                </div>
              )}
              
              <div className="text-center space-y-2">
                <p className="text-sm">
                  For the demo, you can either:
                </p>
                <ul className="text-sm text-left list-disc list-inside space-y-1">
                  <li>Install the selected UPI app and try again</li>
                  <li>Use the SafePay PIN verification method</li>
                  <li>Try with a different UPI app</li>
                </ul>
              </div>
            </div>
            
            <DialogFooter className="flex flex-col">
              <Button 
                onClick={onRedirectToApp}
                className="w-full"
              >
                Try Again with {upiApp === 'gpay' ? 'Google Pay' : upiApp === 'phonepe' ? 'PhonePe' : upiApp === 'paytm' ? 'Paytm' : 'UPI App'}
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full mt-2"
                onClick={() => setShowAppNotFound(false)}
              >
                Use SafePay PIN Verification
              </Button>
              
              <Button 
                variant="ghost" 
                className="w-full mt-2"
                onClick={() => onOpenChange(false)}
              >
                Cancel Payment
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogTitle className="text-center">Enter UPI PIN</DialogTitle>
            <DialogDescription className="text-center">
              Please enter your 6-digit {upiApp === 'gpay' ? 'Google Pay' : upiApp === 'phonepe' ? 'PhonePe' : upiApp === 'paytm' ? 'Paytm' : 'UPI'} PIN to authenticate this payment
            </DialogDescription>
            
            <div className="flex flex-col items-center gap-4 py-4">
              {upiData && (
                <div className="w-full bg-gray-50 dark:bg-gray-800 p-3 rounded-md text-sm mb-2">
                  <div className="flex justify-between">
                    <span className="font-medium">Amount:</span>
                    <span className="text-primary font-bold">₹{upiData.amount}</span>
                  </div>
                </div>
              )}
              
              <div className="w-full max-w-[200px]">
                <Input
                  type="password"
                  inputMode="numeric"
                  value={upiPin}
                  onChange={handlePinChange}
                  className="text-center text-lg tracking-widest h-12"
                  placeholder="******"
                  autoFocus
                />
              </div>
              
              <p className="text-sm font-medium text-blue-600">
                DEMO PIN: 979480 (Use this for testing)
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Enter the 6-digit PIN linked to your UPI account
              </p>
            </div>
            
            <DialogFooter className="flex flex-col">
              <Button 
                onClick={handleVerifyPin} 
                className="w-full bg-blue-600 hover:bg-blue-700"
                disabled={isVerifying || upiPin.length !== 6}
              >
                {isVerifying ? (
                  <>
                    <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                    Verifying...
                  </>
                ) : (
                  'Pay Now'
                )}
              </Button>
              
              {onRedirectToApp && (
                <Button 
                  variant="outline" 
                  className="w-full mt-2"
                  onClick={() => setShowAppNotFound(true)}
                  disabled={isVerifying}
                >
                  Try With UPI App Again
                </Button>
              )}
              
              <Button 
                variant="ghost" 
                className="w-full mt-2"
                onClick={() => onOpenChange(false)}
                disabled={isVerifying}
              >
                Cancel
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}