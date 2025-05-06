import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { ShieldCheck, CheckCircle, AlertCircle } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

interface SafetyVerificationProps {
  merchantName: string;
  isRegisteredBusiness: boolean;
  riskScore: number;
  upiId: string;
  onContinue: () => void;
  onCancel: () => void;
}

export function SafetyVerification({
  merchantName,
  isRegisteredBusiness,
  riskScore,
  upiId,
  onContinue,
  onCancel
}: SafetyVerificationProps) {
  // Determine risk level and color based on risk score
  const getRiskColor = (score: number): string => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 50) return 'bg-yellow-500';
    return 'bg-red-500';
  };
  
  const getRiskColorCSS = (score: number): string => {
    if (score >= 80) return 'green-500';
    if (score >= 50) return 'yellow-500';
    return 'red-500';
  };

  const getRiskLevelText = (score: number): string => {
    if (score >= 80) return 'Low Risk';
    if (score >= 50) return 'Medium Risk';
    return 'High Risk';
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="space-y-1 text-center pb-2">
        <CardTitle className="text-2xl font-bold flex items-center justify-center gap-2">
          <ShieldCheck className="h-6 w-6 text-green-500" />
          Safe to pay
        </CardTitle>
        <CardDescription>
          We've verified this payment's safety
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4 pt-2">
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-medium">Verified {merchantName}</p>
              <p className="text-sm text-muted-foreground">{upiId}</p>
            </div>
          </div>

          {isRegisteredBusiness && (
            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
              <p className="font-medium">Linked to a registered business</p>
            </div>
          )}

          <div className="flex items-start gap-3">
            <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
            <p className="font-medium">SSL Protected</p>
          </div>
        </div>
        
        <div className="space-y-2 pt-2">
          <div className="flex justify-between items-center">
            <h3 className="font-medium">Risk Score</h3>
            <span className={`font-bold ${riskScore >= 80 ? 'text-green-600' : riskScore >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
              {riskScore}%
            </span>
          </div>
          <div className="relative">
            <div className={cn("relative h-2.5 w-full overflow-hidden rounded-full bg-gray-200")}>
              <div 
                className={cn("h-full w-full flex-1 transition-all", getRiskColor(riskScore))} 
                style={{ 
                  transform: `translateX(-${100 - riskScore}%)`
                }}
              />
            </div>
            <div className="absolute -bottom-5 left-0 right-0 flex justify-between text-xs text-gray-500">
              <span>Risky</span>
              <span>Safe</span>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg mt-4 border border-blue-100 dark:border-blue-800">
          <h3 className="font-medium text-blue-700 dark:text-blue-300 flex items-center">
            <AlertCircle className="h-4 w-4 mr-1" />
            Safety Note
          </h3>
          <p className="text-sm text-blue-600 dark:text-blue-200 mt-1">
            This UPI ID has a strong safety record and is linked to a verified user or business.
          </p>
        </div>
      </CardContent>

      <CardFooter className="flex gap-2 pt-2">
        <Button 
          variant="outline" 
          className="flex-1"
          onClick={onCancel}
        >
          Cancel
        </Button>
        <Button 
          className="flex-1"
          onClick={onContinue}
        >
          Continue to pay
        </Button>
      </CardFooter>
    </Card>
  );
}