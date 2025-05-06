import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield, ArrowRight, ChevronLeft } from 'lucide-react';
import { formatCurrency } from '@/lib/utils';

interface TransactionConfirmationProps {
  userName: string;
  userUpiId?: string;
  merchantName: string;
  merchantUpiId: string;
  amount: number;
  trustScore: number;
  onConfirm: () => void;
  onBack: () => void;
}

export function TransactionConfirmation({
  userName,
  userUpiId,
  merchantName,
  merchantUpiId,
  amount,
  trustScore,
  onConfirm,
  onBack
}: TransactionConfirmationProps) {
  // Function to get trust indicator color
  const getTrustColor = (score: number): string => {
    if (score >= 80) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-bold">
          Confirm transaction information
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-5 pt-2">
        <div className="space-y-4">
          {/* From Field */}
          <div className="rounded-lg border p-3 space-y-1">
            <div className="text-sm text-gray-500 dark:text-gray-400">From</div>
            <div className="font-medium">
              {userUpiId 
                ? `${userUpiId.slice(0, 4)}****${userUpiId.slice(userUpiId.indexOf('@'))}` 
                : `Paying as: ${userName}`
              }
            </div>
          </div>

          {/* To Field with Trust Indicator */}
          <div className="rounded-lg border p-3 space-y-1 relative">
            <div className="text-sm text-gray-500 dark:text-gray-400">To</div>
            <div className="font-medium flex items-center justify-between">
              <div>
                <div>{merchantName}</div>
                <div className="text-sm text-gray-500">{merchantUpiId}</div>
              </div>
              <div className={`flex items-center ${getTrustColor(trustScore)}`}>
                <Shield className="h-4 w-4 mr-1" />
                <span className="text-sm font-semibold">{trustScore}% Safe</span>
              </div>
            </div>
          </div>

          {/* Amount Field */}
          <div className="rounded-lg border p-3 space-y-1">
            <div className="text-sm text-gray-500 dark:text-gray-400">Amount</div>
            <div className="text-2xl font-bold">
              {formatCurrency(amount)}
            </div>
          </div>
        </div>

        {/* Merchant UPI ID section */}
        <div className="pt-3 border-t">
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Merchant UPI ID
          </div>
          <div className="font-medium break-all">
            {merchantUpiId}
          </div>
        </div>
      </CardContent>

      <CardFooter className="flex gap-2 pt-2">
        <Button 
          variant="outline" 
          className="flex-1"
          onClick={onBack}
        >
          <ChevronLeft className="h-4 w-4 mr-1" /> Back
        </Button>
        <Button 
          className="flex-1"
          onClick={onConfirm}
        >
          Confirm Pay <ArrowRight className="h-4 w-4 ml-1" />
        </Button>
      </CardFooter>
    </Card>
  );
}