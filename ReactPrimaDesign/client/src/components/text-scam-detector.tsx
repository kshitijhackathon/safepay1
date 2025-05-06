import React, { useState } from 'react';
import { MessageSquare, AlertTriangle, CheckCircle, Loader2, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { apiRequest } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';

interface TextScamDetectorProps {
  onAnalysisComplete?: (result: any) => void;
  fullPage?: boolean;
}

export default function TextScamDetector({
  onAnalysisComplete,
  fullPage = false
}: TextScamDetectorProps) {
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('SMS');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  // Analyze text message for scams
  const analyzeMessage = async () => {
    if (!message.trim()) {
      toast({
        title: 'Empty message',
        description: 'Please enter a message to analyze',
        variant: 'destructive',
      });
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Send to ML endpoint for analysis
      const response = await apiRequest('POST', '/api/ml-analyze-text', {
        text: message,
        message_type: messageType
      });
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setAnalysisResult(result);
        
        // Call callback if provided
        if (onAnalysisComplete) {
          onAnalysisComplete(result);
        }
        
        toast({
          title: 'Analysis complete',
          description: result.is_scam 
            ? 'Potential scam detected. Check the details.' 
            : 'Message appears safe.',
          variant: result.is_scam ? 'destructive' : 'default',
        });
      } else {
        setError(result.message || 'Analysis failed');
        toast({
          title: 'Analysis failed',
          description: result.message || 'Could not analyze message',
          variant: 'destructive',
        });
      }
    } catch (err: any) {
      console.error('Error analyzing message:', err);
      setError('Failed to analyze message. Please try again.');
      toast({
        title: 'Analysis error',
        description: 'Failed to analyze message. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Format risk score to percentage
  const formatRiskScore = (score: number) => {
    return Math.round(score) + '%';
  };

  // Determine risk color based on score
  const getRiskColor = (score: number) => {
    if (score < 30) return 'bg-green-500';
    if (score < 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  // Determine risk level based on score
  const getRiskLevel = (score: number) => {
    if (score < 30) return 'Low Risk';
    if (score < 70) return 'Medium Risk';
    return 'High Risk';
  };

  // Reset everything
  const handleReset = () => {
    setMessage('');
    setAnalysisResult(null);
    setError(null);
  };

  // Main component render
  return (
    <div className={`${fullPage ? 'h-full' : ''}`}>
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center">
            <MessageSquare className="mr-2 h-5 w-5" />
            Text Scam Detector
          </CardTitle>
          <CardDescription>
            Analyze text messages or WhatsApp content for potential scams
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          {/* Message Input */}
          <div className="space-y-4">
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium">Message Type:</label>
                <Select 
                  value={messageType} 
                  onValueChange={setMessageType}
                >
                  <SelectTrigger className="w-[140px]">
                    <SelectValue placeholder="Message Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SMS">SMS</SelectItem>
                    <SelectItem value="WhatsApp">WhatsApp</SelectItem>
                    <SelectItem value="Email">Email</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <Textarea
                placeholder="Enter or paste the message to analyze..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={5}
                className="resize-none"
              />
              
              <div className="flex justify-end gap-2">
                {analysisResult && (
                  <Button
                    variant="outline"
                    onClick={handleReset}
                    disabled={isAnalyzing}
                  >
                    Reset
                  </Button>
                )}
                
                <Button
                  onClick={analyzeMessage}
                  disabled={!message.trim() || isAnalyzing}
                  className="flex items-center gap-2"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4" />
                      Analyze Message
                    </>
                  )}
                </Button>
              </div>
            </div>
            
            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            {/* Loading State */}
            {isAnalyzing && (
              <div className="py-4">
                <p className="text-sm text-gray-500 mb-1">Analyzing message content...</p>
                <Progress value={65} className="w-full" />
              </div>
            )}
          </div>
          
          {/* Analysis Results */}
          {analysisResult && (
            <Tabs defaultValue="summary" className="w-full mt-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="details">Details</TabsTrigger>
              </TabsList>
              
              <TabsContent value="summary" className="space-y-4 pt-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Risk Assessment:</h3>
                  <Badge 
                    variant={analysisResult.is_scam ? "destructive" : "outline"}
                    className="ml-2"
                  >
                    {analysisResult.is_scam ? 'Potential Scam' : 'Likely Safe'}
                  </Badge>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Risk Score</span>
                    <span className="font-medium">{formatRiskScore(analysisResult.risk_score || 0)}</span>
                  </div>
                  <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${getRiskColor(analysisResult.risk_score || 0)} transition-all duration-500`}
                      style={{ width: `${analysisResult.risk_score || 0}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-right">{getRiskLevel(analysisResult.risk_score || 0)}</p>
                </div>
                
                {analysisResult.recommendation && (
                  <div className="mt-4 p-3 bg-gray-50 rounded-md text-sm">
                    <p className="font-medium mb-1">Recommendation:</p>
                    <p>{analysisResult.recommendation}</p>
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="details" className="space-y-4 pt-4">
                {analysisResult.scam_type && (
                  <div>
                    <h3 className="font-semibold mb-1">Scam Type:</h3>
                    <Badge variant="outline">{analysisResult.scam_type}</Badge>
                  </div>
                )}
                
                {analysisResult.scam_indicators && analysisResult.scam_indicators.length > 0 && (
                  <div className="mt-4">
                    <h3 className="font-semibold mb-1">Warning Signs:</h3>
                    <ul className="list-disc list-inside space-y-1">
                      {analysisResult.scam_indicators.map((indicator: string, index: number) => (
                        <li key={index} className="text-sm">{indicator}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {analysisResult.unsafe_elements && analysisResult.unsafe_elements.length > 0 && (
                  <div className="mt-4">
                    <h3 className="font-semibold mb-1 text-red-600">Unsafe Elements Detected:</h3>
                    <div className="p-2 bg-red-50 rounded-md">
                      <ul className="list-disc list-inside space-y-1">
                        {analysisResult.unsafe_elements.map((element: string, index: number) => (
                          <li key={index} className="text-sm text-red-700">{element}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
                
                <div className="flex items-center justify-between text-xs text-gray-500 mt-4">
                  <span>Confidence: {Math.round((analysisResult.confidence || 0) * 100)}%</span>
                  <span>Analysis: {analysisResult.analysis_method || 'ML Model'}</span>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
        
        <CardFooter className="flex justify-between text-xs text-gray-500">
          <span>Text analysis powered by ML</span>
          {analysisResult && (
            <span>
              {analysisResult.is_scam 
                ? <span className="flex items-center text-red-500"><AlertTriangle className="h-3 w-3 mr-1" /> Caution advised</span>
                : <span className="flex items-center text-green-500"><CheckCircle className="h-3 w-3 mr-1" /> Appears safe</span>
              }
            </span>
          )}
        </CardFooter>
      </Card>
    </div>
  );
}