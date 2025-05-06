import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { BarChart4, ShieldAlert, ShieldCheck, AlertTriangle, Info, AlertCircle, BadgeCheck, Skull, Zap } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { 
  calculateTransactionRiskScore, 
  getRiskColor, 
  getRiskBgColor, 
  getRiskTextColor,
  RiskLevel,
  analyzeDeviceFingerprint,
  analyzeBehavioralBiometrics
} from '@/lib/risk-scoring';

const RiskScoreDemo: React.FC = () => {
  const [upiId, setUpiId] = useState('');
  const [amount, setAmount] = useState<number>(0);
  const [note, setNote] = useState('');
  const [recipientName, setRecipientName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [riskResult, setRiskResult] = useState<any>(null);
  const [deviceInfo, setDeviceInfo] = useState<any>(null);
  const [behavioralData, setBehavioralData] = useState<any>(null);
  
  const handleAnalyze = async () => {
    if (!upiId || amount <= 0) {
      alert('Please enter a valid UPI ID and amount');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Collect device and behavioral data
      const deviceAnalysis = analyzeDeviceFingerprint();
      setDeviceInfo(deviceAnalysis);
      
      // Simulate collecting behavioral data
      const behavior = analyzeBehavioralBiometrics({
        typingSpeed: Math.random() > 0.9 ? 35 : 120 + Math.random() * 80, // Occasionally simulate bot-like typing
        touchPressure: Array(5).fill(0).map(() => 0.3 + Math.random() * 0.4),
        motionData: Array(15).fill(0).map(() => [
          Math.random() * 0.1 - 0.05,
          Math.random() * 0.1 - 0.05,
          Math.random() * 0.1 - 0.05
        ])
      });
      setBehavioralData(behavior);
      
      // Get transaction risk score
      const result = await calculateTransactionRiskScore({
        upiId,
        amount,
        note,
        recipient: recipientName,
        deviceInfo: deviceAnalysis
      });
      
      setRiskResult(result);
    } catch (error) {
      console.error('Error analyzing transaction:', error);
      alert('An error occurred while analyzing the transaction');
    } finally {
      setIsLoading(false);
    }
  };
  
  const getRiskLevelIcon = (level: RiskLevel) => {
    switch (level) {
      case RiskLevel.Low:
        return <ShieldCheck className="h-8 w-8 text-green-500" />;
      case RiskLevel.Medium:
        return <AlertCircle className="h-8 w-8 text-yellow-500" />;
      case RiskLevel.High:
        return <AlertTriangle className="h-8 w-8 text-orange-500" />;
      case RiskLevel.Critical:
        return <Skull className="h-8 w-8 text-red-500" />;
      default:
        return <Info className="h-8 w-8 text-gray-500" />;
    }
  };
  
  return (
    <div className="container mx-auto py-6 max-w-4xl">
      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <BarChart4 className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-2xl font-bold">Real-Time Risk Scoring Engine</h1>
            <p className="text-muted-foreground">
              Analyze transactions for fraud risk using ML-powered scoring
            </p>
          </div>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>Transaction Details</CardTitle>
            <CardDescription>
              Enter details of the transaction you want to analyze
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label htmlFor="upiId" className="text-sm font-medium">UPI ID</label>
                  <Input
                    id="upiId"
                    placeholder="e.g. john@oksbi"
                    value={upiId}
                    onChange={(e) => setUpiId(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="amount" className="text-sm font-medium">Amount (₹)</label>
                  <Input
                    id="amount"
                    type="number"
                    placeholder="e.g. 5000"
                    value={amount || ''}
                    onChange={(e) => setAmount(Number(e.target.value))}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label htmlFor="recipient" className="text-sm font-medium">Recipient Name (Optional)</label>
                  <Input
                    id="recipient"
                    placeholder="e.g. John Doe"
                    value={recipientName}
                    onChange={(e) => setRecipientName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="note" className="text-sm font-medium">Transaction Note (Optional)</label>
                  <Input
                    id="note"
                    placeholder="e.g. Rent payment"
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                  />
                </div>
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={handleAnalyze} 
              disabled={isLoading}
              className="w-full"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full mr-2"></div>
                  Analyzing Risk...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Analyze Transaction Risk
                </>
              )}
            </Button>
          </CardFooter>
        </Card>
        
        {riskResult && (
          <Card className={getRiskBgColor(riskResult.level)}>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                {getRiskLevelIcon(riskResult.level)}
                <span>Risk Analysis Result</span>
              </CardTitle>
              <CardDescription>
                Comprehensive risk assessment using ML analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Risk score visualization */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Risk Score</span>
                  <span className={`font-bold ${getRiskTextColor(riskResult.level)}`}>
                    {(riskResult.score * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress 
                  value={riskResult.score * 100} 
                  className={`h-3 ${getRiskColor(riskResult.level)}`}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Low Risk</span>
                  <span>Medium Risk</span>
                  <span>High Risk</span>
                  <span>Critical</span>
                </div>
              </div>
              
              {/* Risk level */}
              <div className="p-4 rounded-lg border bg-white">
                <div className="flex items-center gap-3 mb-2">
                  <Badge variant="outline" className={`${getRiskBgColor(riskResult.level)} px-2 py-1`}>
                    {riskResult.level.toUpperCase()}
                  </Badge>
                  <span className="font-medium">Risk Level</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  {riskResult.recommendation}
                </p>
              </div>
              
              {/* Risk factors */}
              <div>
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Risk Factors Detected
                </h3>
                
                {riskResult.factors.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No significant risk factors detected.</p>
                ) : (
                  <div className="space-y-2">
                    {riskResult.factors.map((factor: any, i: number) => (
                      <div key={i} className="flex items-center gap-3 p-2 rounded-lg border bg-white">
                        <div className="w-1 h-12 rounded-full" style={{ backgroundColor: `rgba(234, 84, 85, ${factor.impact})` }}></div>
                        <div>
                          <p className="text-sm font-medium">{factor.description}</p>
                          <p className="text-xs text-muted-foreground">Type: {factor.type}</p>
                        </div>
                        <div className="ml-auto">
                          <Badge variant="outline">
                            {Math.round(factor.impact * 100)}% impact
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Advanced metrics */}
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="advanced-metrics" className="border-none">
                  <AccordionTrigger className="py-2">
                    <span className="text-sm font-medium">Advanced Metrics</span>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-4 pt-2">
                      {/* Device fingerprinting */}
                      {deviceInfo && (
                        <div className="p-3 rounded-lg border">
                          <h4 className="font-medium mb-2 text-sm">Device Analysis</h4>
                          {deviceInfo.suspicious ? (
                            <div className="space-y-1">
                              <Badge variant="outline" className="bg-orange-50">SUSPICIOUS DEVICE</Badge>
                              <ul className="text-sm list-disc list-inside mt-1">
                                {deviceInfo.reasons.map((reason: string, i: number) => (
                                  <li key={i} className="text-orange-700">{reason}</li>
                                ))}
                              </ul>
                            </div>
                          ) : (
                            <div className="text-sm flex items-center gap-2">
                              <BadgeCheck className="text-green-500 w-4 h-4" />
                              <span>Trusted device detected</span>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Behavioral biometrics */}
                      {behavioralData && (
                        <div className="p-3 rounded-lg border">
                          <h4 className="font-medium mb-2 text-sm">Behavioral Biometrics</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-xs">Bot Probability</span>
                              <span className="text-xs font-medium">
                                {(behavioralData.botProbability * 100).toFixed(0)}%
                              </span>
                            </div>
                            <Progress 
                              value={behavioralData.botProbability * 100} 
                              className={`h-2 ${behavioralData.botProbability > 0.5 ? 'bg-red-500' : 'bg-green-500'}`}
                            />
                            
                            {behavioralData.indicators.length > 0 && (
                              <div className="mt-2">
                                <p className="text-xs font-medium mb-1">Suspicious Indicators:</p>
                                <ul className="text-xs list-disc list-inside">
                                  {behavioralData.indicators.map((indicator: string, i: number) => (
                                    <li key={i} className="text-red-700">{indicator}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {/* Network analysis simulation */}
                      <div className="p-3 rounded-lg border">
                        <h4 className="font-medium mb-2 text-sm">Network Analysis</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span>UPI Domain Trust Score</span>
                            <span>{riskResult.level === RiskLevel.Low ? '85%' : '45%'}</span>
                          </div>
                          <Progress 
                            value={riskResult.level === RiskLevel.Low ? 85 : 45} 
                            className="h-2"
                          />
                          
                          <div className="flex justify-between text-xs mt-2">
                            <span>Transaction Graph Analysis</span>
                            <span>{riskResult.level === RiskLevel.Low ? 'Normal Pattern' : 'Unusual Pattern'}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
              
              <Separator />
              
              {/* Action buttons */}
              <div className="flex flex-col sm:flex-row gap-2">
                {riskResult.level === RiskLevel.Critical || riskResult.level === RiskLevel.High ? (
                  <>
                    <Button variant="destructive" className="flex-1">
                      Block Transaction
                    </Button>
                    <Button variant="outline" className="flex-1">
                      Proceed with Additional Verification
                    </Button>
                  </>
                ) : (
                  <>
                    <Button variant="default" className="flex-1">
                      Proceed with Transaction
                    </Button>
                    <Button variant="outline" className="flex-1">
                      Verify Details First
                    </Button>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* AI-powered learning section */}
        <Card className="bg-slate-50">
          <CardHeader>
            <CardTitle className="text-base">How Our Risk Scoring Works</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm space-y-4">
              <p>
                Our advanced risk scoring engine combines multiple machine learning models:
              </p>
              <ul className="ml-6 list-disc space-y-2">
                <li>
                  <span className="font-medium">Ensemble Machine Learning</span>: Combines supervised models (XGBoost) 
                  with unsupervised anomaly detection (Isolation Forest)
                </li>
                <li>
                  <span className="font-medium">Behavioral Biometrics</span>: Analyzes typing patterns, device tilt, 
                  and touch pressure to distinguish humans from bots
                </li>
                <li>
                  <span className="font-medium">Graph Neural Networks</span>: Maps transaction networks to identify 
                  suspicious patterns like mule accounts and money laundering cycles
                </li>
                <li>
                  <span className="font-medium">Device Fingerprinting</span>: Identifies devices to detect unauthorized 
                  access attempts
                </li>
              </ul>
              <p className="italic text-muted-foreground">
                The system continuously learns from new fraud patterns to stay ahead of scammers.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default RiskScoreDemo;