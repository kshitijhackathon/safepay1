import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { Button } from '@/components/ui/button';
import { ChevronLeft, AlertTriangle, Shield, ShieldCheck, Phone, Headphones, Info, Volume2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';

type CallResult = {
  risk_level: 'low' | 'medium' | 'high';
  risk_score: number;
  is_scam: boolean;
  scam_type?: string;
  explanation: string;
  caller_patterns?: string[];
};

const CallCheck: React.FC = () => {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isChecking, setIsChecking] = useState(false);
  const [checkProgress, setCheckProgress] = useState(0);
  const [callResult, setCallResult] = useState<CallResult | null>(null);
  const [historyExpanded, setHistoryExpanded] = useState(false);
  
  // Recent call history (simulated)
  const recentCalls = [
    { number: '9876543210', date: '22 Apr, 11:30 AM', duration: '2m 15s' },
    { number: '8765432109', date: '20 Apr, 3:45 PM', duration: '5m 22s' },
    { number: '7654321098', date: '19 Apr, 9:12 AM', duration: '1m 08s' },
  ];

  // Common scam call patterns (for educational purposes)
  const commonScamPatterns = [
    'Caller claims to be from your bank or government agency',
    'Threatens immediate account suspension or legal action',
    'Asks for OTP, UPI PIN or card CVV',
    'Creates urgency to prevent you from thinking',
    'Unusual calling hours (very early or late)',
    'International numbers or unknown country codes'
  ];
  
  // Handle phone number check
  const handlePhoneCheck = () => {
    if (!phoneNumber.trim()) {
      toast({
        title: "Empty Input",
        description: "Please enter a phone number to check",
        variant: "destructive",
      });
      return;
    }
    
    // Simple validation for demo
    if (phoneNumber.trim().length < 10) {
      toast({
        title: "Invalid Number",
        description: "Please enter a valid phone number",
        variant: "destructive",
      });
      return;
    }
    
    // Start checking
    setIsChecking(true);
    setCheckProgress(0);
    
    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      setCheckProgress(Math.min(progress, 100));
      
      if (progress >= 100) {
        clearInterval(interval);
        
        // Simulate analysis result - in a real app, this would come from the backend
        setTimeout(() => {
          // Randomly determine if it's a scam or not
          const isScam = Math.random() > 0.5;
          const riskScore = isScam ? 0.7 + (Math.random() * 0.3) : Math.random() * 0.4;
          
          let riskLevel: 'low' | 'medium' | 'high';
          if (riskScore < 0.3) riskLevel = 'low';
          else if (riskScore < 0.7) riskLevel = 'medium';
          else riskLevel = 'high';
          
          const scamTypes = [
            'Banking Fraud Call',
            'KYC Verification Scam',
            'Tax/Legal Threat Scam',
            'Investment Scheme',
            'OTP Harvesting'
          ];
          
          const result: CallResult = {
            risk_level: riskLevel,
            risk_score: riskScore,
            is_scam: isScam,
            explanation: isScam 
              ? "This number has been reported for suspicious activities matching known scam patterns." 
              : "No significant risk factors detected for this number.",
            caller_patterns: isScam ? [
              'Reported by multiple users',
              'Known for requesting financial information',
              'Uses urgency tactics'
            ] : undefined
          };
          
          if (isScam) {
            result.scam_type = scamTypes[Math.floor(Math.random() * scamTypes.length)];
          }
          
          setCallResult(result);
          setIsChecking(false);
        }, 500);
      }
    }, 50);
  };
  
  // Risk color and icon helpers
  const getRiskColor = (level: 'low' | 'medium' | 'high'): string => {
    switch (level) {
      case 'low': return 'bg-green-500';
      case 'medium': return 'bg-yellow-500';
      case 'high': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getRiskIcon = (level: 'low' | 'medium' | 'high') => {
    switch (level) {
      case 'low': return <ShieldCheck className="w-6 h-6 text-green-500" />;
      case 'medium': return <Shield className="w-6 h-6 text-yellow-500" />;
      case 'high': return <AlertTriangle className="w-6 h-6 text-red-500" />;
      default: return null;
    }
  };
  
  // Format phone number for display
  const formatPhoneNumber = (number: string): string => {
    if (number.length === 10) {
      return `+91 ${number.substring(0, 5)} ${number.substring(5)}`;
    }
    return number;
  };
  
  return (
    <div className="dark-bg-secondary min-h-screen flex flex-col">
      {/* Top bar */}
      <div className="bg-white dark:bg-gray-900 p-4 flex items-center shadow-sm">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => setLocation('/home')}
          className="mr-2"
        >
          <ChevronLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-medium dark-text-primary">Call Scam Check</h1>
      </div>
      
      {/* Content area */}
      <div className="flex-1 overflow-y-auto p-4 pb-20">
        <div className="mb-6">
          <p className="text-sm dark-text-secondary mb-4">
            Check if a phone number has been reported for scam calls or suspicious activities.
          </p>
          
          {/* Phone number input */}
          <div className="flex gap-2 mb-6">
            <div className="flex-1">
              <Input
                type="tel"
                placeholder="Enter phone number"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                className="w-full"
                maxLength={10}
              />
            </div>
            <Button 
              onClick={handlePhoneCheck}
              disabled={isChecking}
              className="whitespace-nowrap"
            >
              <Phone className="h-4 w-4 mr-2" />
              Check Number
            </Button>
          </div>
          
          {/* Recent calls from phone */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-medium">Recent Calls</h3>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => setHistoryExpanded(!historyExpanded)}
                className="text-xs"
              >
                {historyExpanded ? 'Show Less' : 'Show More'}
              </Button>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg border overflow-hidden">
              <div className="divide-y">
                {recentCalls.slice(0, historyExpanded ? recentCalls.length : 2).map((call, index) => (
                  <div key={index} className="flex items-center justify-between p-3">
                    <div className="flex items-center">
                      <div className="bg-blue-100 dark:bg-blue-900/30 w-8 h-8 rounded-full flex items-center justify-center mr-3">
                        <Phone className="h-4 w-4 text-blue-500" />
                      </div>
                      <div>
                        <p className="font-medium text-sm">{formatPhoneNumber(call.number)}</p>
                        <p className="text-xs text-gray-500">{call.date} • {call.duration}</p>
                      </div>
                    </div>
                    <Button 
                      size="sm" 
                      variant="ghost" 
                      className="h-8 w-8 p-0"
                      onClick={() => {
                        setPhoneNumber(call.number);
                        setCallResult(null);
                      }}
                    >
                      <Info className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Checking progress */}
          {isChecking && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Checking number...</span>
                <span className="text-sm">{checkProgress}%</span>
              </div>
              <Progress value={checkProgress} className="w-full" />
            </div>
          )}
          
          {/* Call check result */}
          {callResult && (
            <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded-lg border">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-medium">Check Result for {formatPhoneNumber(phoneNumber)}</h3>
                <div className={`px-3 py-1 rounded-full text-white text-xs ${getRiskColor(callResult.risk_level)}`}>
                  {callResult.risk_level.toUpperCase()} RISK
                </div>
              </div>
              
              <div className="flex items-center mb-4">
                {getRiskIcon(callResult.risk_level)}
                <div className="ml-3">
                  <p className="font-medium">
                    {callResult.is_scam ? 'Potential Scam Number!' : 'No Known Scam Reports'}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Risk Score: {Math.round(callResult.risk_score * 100)}%
                  </p>
                </div>
              </div>
              
              {callResult.is_scam && callResult.scam_type && (
                <div className="mb-4">
                  <p className="text-sm font-medium mb-1">Reported Scam Type:</p>
                  <p className="text-sm">{callResult.scam_type}</p>
                </div>
              )}
              
              <div className="mb-4">
                <p className="text-sm font-medium mb-1">Analysis:</p>
                <p className="text-sm">{callResult.explanation}</p>
              </div>
              
              {callResult.caller_patterns && (
                <div className="mb-4">
                  <p className="text-sm font-medium mb-1">Detected Patterns:</p>
                  <ul className="text-sm list-disc list-inside">
                    {callResult.caller_patterns.map((pattern, index) => (
                      <li key={index}>{pattern}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {callResult.is_scam && (
                <div className="mt-4 pt-4 border-t">
                  <Button 
                    variant="destructive" 
                    className="w-full"
                  >
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    Report & Block Number
                  </Button>
                </div>
              )}
            </div>
          )}
          
          {/* Educational section */}
          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="flex items-start">
                <Volume2 className="h-5 w-5 text-blue-500 mr-2 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-medium mb-2">Common Scam Call Patterns</h3>
                  <ul className="text-sm list-disc list-inside space-y-1">
                    {commonScamPatterns.map((pattern, index) => (
                      <li key={index}>{pattern}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <div className="flex items-start">
                <Headphones className="h-5 w-5 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-medium mb-2">Stay Safe from Call Scams</h3>
                  <ul className="text-sm list-disc list-inside space-y-1">
                    <li>Never share OTP, PIN or banking details over phone</li>
                    <li>Hang up if pressured or threatened</li>
                    <li>Verify by calling official numbers from bank website</li>
                    <li>Report suspicious calls to your bank and police</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
};

export default CallCheck;