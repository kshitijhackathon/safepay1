import React from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { CheckIcon } from 'lucide-react';

interface PaymentSafetyPopupProps {
  status: 'safe' | 'caution' | 'danger';
  riskScore: number;
  merchantName?: string;
  businessInfo?: boolean;
  sslProtected?: boolean;
  details?: string;
  onContinue: () => void;
  onCancel: () => void;
  onReportScam?: () => void;
}

export function PaymentSafetyPopup({
  status,
  riskScore,
  merchantName = 'Merchant',
  businessInfo = true,
  sslProtected = true,
  details = 'This UPI ID has a strong safety record and is linked to a verified user or business',
  onContinue,
  onCancel,
  onReportScam
}: PaymentSafetyPopupProps) {
  const getRiskColor = () => {
    if (status === 'safe') return 'bg-green-500';
    if (status === 'caution') return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getStatusText = () => {
    if (status === 'safe') return 'Safe to pay';
    if (status === 'caution') return 'Proceed with caution';
    return 'Potentially unsafe';
  };

  const getRiskPercentage = () => {
    return Math.min(100, Math.max(0, riskScore * 100));
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-sm w-full p-6 flex flex-col items-center">
        <div className={`${getRiskColor()} h-16 w-16 rounded-full flex items-center justify-center mb-2`}>
          {status === 'safe' && <CheckIcon className="h-8 w-8 text-white" />}
          {status === 'caution' && <span className="text-white text-2xl font-bold">!</span>}
          {status === 'danger' && <span className="text-white text-2xl font-bold">×</span>}
        </div>
        
        <h2 className="text-lg font-semibold mb-4">{getStatusText()}</h2>
        
        <div className="w-full space-y-2 mb-4">
          <div className="flex items-start">
            <div className="flex-shrink-0 mt-0.5">
              {(status === 'safe' || status === 'caution') && (
                <div className="h-4 w-4 rounded-sm bg-green-500 flex items-center justify-center">
                  <CheckIcon className="h-3 w-3 text-white" />
                </div>
              )}
            </div>
            <div className="ml-2 text-sm">
              <p>Verified {merchantName}</p>
            </div>
          </div>
          
          {businessInfo && (
            <div className="flex items-start">
              <div className="flex-shrink-0 mt-0.5">
                <div className="h-4 w-4 rounded-sm bg-green-500 flex items-center justify-center">
                  <CheckIcon className="h-3 w-3 text-white" />
                </div>
              </div>
              <div className="ml-2 text-sm">
                <p>Linked to a registered business</p>
              </div>
            </div>
          )}
          
          {sslProtected && (
            <div className="flex items-start">
              <div className="flex-shrink-0 mt-0.5">
                <div className="h-4 w-4 rounded-sm bg-green-500 flex items-center justify-center">
                  <CheckIcon className="h-3 w-3 text-white" />
                </div>
              </div>
              <div className="ml-2 text-sm">
                <p>SSL Protected</p>
              </div>
            </div>
          )}
        </div>
        
        <div className="w-full mb-4">
          <div className="flex justify-between mb-1">
            <span className="text-xs text-gray-500">Risk Score</span>
            <span className="text-xs font-semibold">{getRiskPercentage().toFixed(0)}%</span>
          </div>
          <div className="flex items-center">
            <Progress 
              value={100 - getRiskPercentage()} 
              className={`h-2.5 flex-1 ${status === 'safe' ? 'bg-green-500' : status === 'caution' ? 'bg-yellow-500' : 'bg-red-500'}`}
            />
            <div className="ml-2 w-8 h-8 rounded-full border-2 border-gray-200 flex items-center justify-center text-xs font-medium">
              {getRiskPercentage().toFixed(0)}%
            </div>
          </div>
        </div>
        
        <p className="text-sm text-gray-600 mb-6 text-center">
          {details}
        </p>
        
        {status === 'danger' && onReportScam ? (
          <>
            <Button 
              variant="destructive" 
              className="w-full mb-3" 
              onClick={onReportScam}
            >
              Report Scam
            </Button>
            
            <Button 
              className="w-full bg-gray-700 hover:bg-gray-800 mb-3" 
              onClick={onContinue}
            >
              Continue anyway
            </Button>
            
            <Button 
              variant="ghost" 
              className="w-full text-gray-600" 
              onClick={onCancel}
            >
              Cancel
            </Button>
          </>
        ) : (
          <>
            <Button 
              className={`w-full ${status === 'safe' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-amber-500 hover:bg-amber-600'} mb-3`}
              onClick={onContinue}
            >
              Continue to pay
            </Button>
            
            <Button 
              variant="ghost" 
              className="w-full text-gray-600" 
              onClick={onCancel}
            >
              Cancel
            </Button>
          </>
        )}
      </div>
    </div>
  );
}