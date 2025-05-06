import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ShieldAlert, ArrowLeft, Mic, MessageSquare, InfoIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import VoiceScamDetector from '@/components/voice-scam-detector';
import TextScamDetector from '@/components/text-scam-detector';
import { useLocation } from 'wouter';
import { useAuthState } from '@/hooks/use-auth-state';

export default function ScamDetectionPage() {
  const [location, setLocation] = useLocation();
  const [activeTab, setActiveTab] = useState<string>('voice');
  const [lastAnalysisResult, setLastAnalysisResult] = useState<any>(null);
  const { authState } = useAuthState();
  const isAuthenticated = authState.isLoggedIn;
  
  // Check for authentication and redirect if needed
  useEffect(() => {
    if (!isAuthenticated) {
      // Store that we're trying to access the scam detection page
      try {
        sessionStorage.setItem('pendingScamDetection', 'true');
      } catch (error) {
        console.error('Failed to store pending detection:', error);
      }
      
      // Redirect to phone login with return URL
      const returnUrl = encodeURIComponent('/scam-detection');
      setLocation(`/phone-login?returnUrl=${returnUrl}`);
    }
  }, [isAuthenticated, setLocation]);

  // Stop rendering the rest of the page if not authenticated
  if (!isAuthenticated) {
    return null;
  }

  const handleGoBack = () => {
    setLocation('/home');
  };

  const handleVoiceAnalysisComplete = (result: any) => {
    setLastAnalysisResult({
      type: 'voice',
      is_scam: result.analysis?.is_scam || false,
      risk_score: result.analysis?.risk_score || 0,
      timestamp: new Date().toISOString()
    });
  };

  const handleTextAnalysisComplete = (result: any) => {
    setLastAnalysisResult({
      type: 'text',
      is_scam: result.is_scam || false,
      risk_score: result.risk_score || 0,
      timestamp: new Date().toISOString()
    });
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white border-b p-4 flex items-center">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleGoBack}
          className="mr-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div className="flex-1">
          <h1 className="text-lg font-bold flex items-center">
            <ShieldAlert className="h-5 w-5 mr-2 text-primary" />
            Scam Detection
          </h1>
          <p className="text-sm text-gray-500">Analyze voice calls and text messages for scams</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-4">
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Information Card */}
          <Card>
            <CardContent className="pt-6">
              <Alert className="bg-primary/10 border-primary/20">
                <InfoIcon className="h-4 w-4 text-primary" />
                <AlertTitle>Enhanced AI Detection</AlertTitle>
                <AlertDescription>
                  This feature uses advanced machine learning to analyze voice and text content for potential scams.
                  Recordings are processed locally and not stored permanently.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          {/* Detection Tools Tabs */}
          <Tabs defaultValue="voice" value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="voice" className="flex items-center gap-2">
                <Mic className="h-4 w-4" />
                Voice Detection
              </TabsTrigger>
              <TabsTrigger value="text" className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                Text Detection
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="voice" className="mt-4">
              <VoiceScamDetector onAnalysisComplete={handleVoiceAnalysisComplete} />
            </TabsContent>
            
            <TabsContent value="text" className="mt-4">
              <TextScamDetector onAnalysisComplete={handleTextAnalysisComplete} />
            </TabsContent>
          </Tabs>

          {/* Recent Analysis */}
          {lastAnalysisResult && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Last Analysis Result</CardTitle>
                <CardDescription>
                  {new Date(lastAnalysisResult.timestamp).toLocaleString()}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between items-center">
                  <div className="flex items-center">
                    {lastAnalysisResult.type === 'voice' ? (
                      <Mic className="h-4 w-4 mr-2 text-gray-500" />
                    ) : (
                      <MessageSquare className="h-4 w-4 mr-2 text-gray-500" />
                    )}
                    <span className="text-sm">{lastAnalysisResult.type === 'voice' ? 'Voice' : 'Text'} Analysis</span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-sm">Risk Score: {Math.round(lastAnalysisResult.risk_score)}%</span>
                    <Badge variant={lastAnalysisResult.is_scam ? "destructive" : "outline"}>
                      {lastAnalysisResult.is_scam ? 'Potential Scam' : 'Likely Safe'}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Usage Tips */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Quick Tips</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-sm space-y-2">
                <li className="flex items-start">
                  <Badge variant="outline" className="mr-2 mt-0.5">Voice</Badge>
                  <span>Record calls or voice messages to detect scam patterns.</span>
                </li>
                <li className="flex items-start">
                  <Badge variant="outline" className="mr-2 mt-0.5">Text</Badge>
                  <span>Paste suspicious SMS, WhatsApp messages, or emails for analysis.</span>
                </li>
                <li className="flex items-start">
                  <Badge variant="outline" className="mr-2 mt-0.5">Security</Badge>
                  <span>Analysis is performed with highest privacy standards using AI and ML.</span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}