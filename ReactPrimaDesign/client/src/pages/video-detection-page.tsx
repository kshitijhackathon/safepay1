import React, { useState } from 'react';
import { Link, useLocation } from 'wouter';
import { AlertCircle, Camera, CheckCircle, Download, FileVideo, Info, Upload, IndianRupee } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';
import { CameraRecorder } from '@/components/video-detection/camera-recorder';
import { useToast } from '@/hooks/use-toast';
import { PageLayout } from '@/layouts/page-layout';
import { UpiPinDialog } from '@/components/payment/upi-pin-dialog';

type AnalysisResult = {
  is_scam: boolean;
  confidence: number;
  visual_confidence?: number;
  audio_confidence?: number;
  text_confidence?: number;
  reason?: string;
};

export default function VideoDetectionPage() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('live');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<AnalysisResult | null>(null);
  const [liveResult, setLiveResult] = useState<AnalysisResult | null>(null);
  
  // Payment flow state
  const [showPaymentOptions, setShowPaymentOptions] = useState(false);
  const [showUpiPin, setShowUpiPin] = useState(false);
  const [selectedUpiApp, setSelectedUpiApp] = useState('');
  const [paymentAmount, setPaymentAmount] = useState('');
  const [isProcessingPayment, setIsProcessingPayment] = useState(false);
  
  // Payment app options
  const paymentApps = [
    { id: 'gpay', name: 'Google Pay', color: 'bg-white', deepLink: 'gpay://' },
    { id: 'phonepe', name: 'PhonePe', color: 'bg-[#5f259f]', deepLink: 'phonepe://' },
    { id: 'paytm', name: 'Paytm', color: 'bg-[#00baf2]', deepLink: 'paytm://' },
  ];
  
  // Handle file upload and analysis
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setIsUploading(true);
    setUploadResult(null);
    
    try {
      // Create form data for the file upload
      const formData = new FormData();
      formData.append('video', file);
      
      // Send the file to the server for analysis
      const response = await fetch('/api/video-detection/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      setUploadResult(result);
      
      // Show toast notification with result
      if (result.is_scam) {
        toast({
          title: 'Scam Detected',
          description: `Analysis indicates ${Math.round(result.confidence * 100)}% likelihood of scam content.`,
          variant: 'destructive',
        });
      } else {
        toast({
          title: 'Analysis Complete',
          description: 'No scam detected in this video.',
          variant: 'default',
        });
      }
    } catch (error) {
      console.error('Error analyzing video:', error);
      toast({
        title: 'Analysis Failed',
        description: error instanceof Error ? error.message : 'Failed to analyze video',
        variant: 'destructive',
      });
    } finally {
      setIsUploading(false);
    }
  };
  
  // Handle live detection result
  const handleLiveResult = (result: AnalysisResult) => {
    setLiveResult(result);
  };
  
  // Handle amount change
  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    // Only allow numbers and up to 2 decimal places
    if (/^\d*\.?\d{0,2}$/.test(value)) {
      setPaymentAmount(value);
    }
  };
  
  // Handle payment app selection
  const handlePaymentApp = (appId: string) => {
    setIsProcessingPayment(true);
    setSelectedUpiApp(appId);
    
    // Find the selected payment app
    const app = paymentApps.find(a => a.id === appId);
    if (!app) return;
    
    // Show UPI PIN dialog
    setShowPaymentOptions(false);
    setTimeout(() => {
      setShowUpiPin(true);
      setIsProcessingPayment(false);
    }, 500);
  };
  
  // Handle payment initiation from both live detection and uploaded video
  const handleInitiatePayment = () => {
    // Check which tab is active and use the appropriate result
    const result = activeTab === 'live' ? liveResult : uploadResult;
    
    if (!result || result.is_scam || result.confidence >= 0.5) return;
    setShowPaymentOptions(true);
  };
  
  // Handle payment success
  const handlePaymentSuccess = () => {
    toast({
      title: "Payment Successful",
      description: `Payment of ₹${paymentAmount} has been processed successfully.`,
    });
    
    // Reset state and navigate to success page
    setTimeout(() => {
      const successParams = new URLSearchParams();
      successParams.append('amount', paymentAmount);
      successParams.append('app', selectedUpiApp);
      successParams.append('source', 'video');
      
      setLocation(`/success?${successParams.toString()}`);
    }, 1000);
  };
  
  return (
    <PageLayout 
      title="Video Scan" 
      description="Analyze videos for potential fraud with advanced AI detection"
      icon={<Camera className="h-6 w-6" />}
    >
      <div className="max-w-4xl mx-auto">
        <Alert className="mb-6">
          <Info className="h-4 w-4" />
          <AlertTitle>Enhanced Video Protection</AlertTitle>
          <AlertDescription>
            Our AI technology scans videos to detect common scam patterns including visual cues, audio markers, 
            and suspicious speech content to protect you from fraud.
          </AlertDescription>
        </Alert>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="live" className="flex items-center gap-2">
              <Camera className="h-4 w-4" />
              <span>Live Detection</span>
            </TabsTrigger>
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              <span>Upload Video</span>
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="live" className="w-full">
            <div className="flex flex-col md:flex-row gap-6">
              <div className="w-full md:w-7/12">
                <CameraRecorder onResult={handleLiveResult} />
              </div>
              
              <div className="w-full md:w-5/12 space-y-4">
                <div className="bg-card rounded-lg border p-4">
                  <h3 className="text-lg font-medium mb-2">Detection Dashboard</h3>
                  
                  {liveResult ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Overall Risk:</span>
                        <span className={`font-medium ${liveResult.is_scam ? 'text-destructive' : 'text-success'}`}>
                          {liveResult.is_scam ? 'High Risk' : 'Low Risk'}
                        </span>
                      </div>
                      
                      <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${liveResult.is_scam ? 'bg-destructive' : 'bg-success'}`} 
                          style={{ width: `${Math.round(liveResult.confidence * 100)}%` }} 
                        />
                      </div>
                      
                      <div className="pt-2 space-y-2">
                        {liveResult.visual_confidence !== undefined && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Visual Analysis:</span>
                            <span className="text-sm font-medium">{Math.round(liveResult.visual_confidence * 100)}%</span>
                          </div>
                        )}
                        
                        {liveResult.audio_confidence !== undefined && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Audio Analysis:</span>
                            <span className="text-sm font-medium">{Math.round(liveResult.audio_confidence * 100)}%</span>
                          </div>
                        )}
                        
                        {liveResult.text_confidence !== undefined && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Text Analysis:</span>
                            <span className="text-sm font-medium">{Math.round(liveResult.text_confidence * 100)}%</span>
                          </div>
                        )}
                      </div>
                      
                      {liveResult.reason && (
                        <div className="pt-2 border-t">
                          <span className="text-sm text-muted-foreground">Detection Notes:</span>
                          <p className="text-sm mt-1">{liveResult.reason}</p>
                        </div>
                      )}
                      
                      {/* Show payment button if risk is low */}
                      {!liveResult.is_scam && liveResult.confidence < 0.5 && (
                        <div className="pt-4 border-t mt-4">
                          <Button 
                            onClick={handleInitiatePayment} 
                            className="w-full flex items-center justify-center gap-2"
                            variant="default"
                          >
                            <IndianRupee className="h-4 w-4" />
                            <span>Proceed to Payment</span>
                          </Button>
                          <p className="text-xs text-muted-foreground mt-2 text-center">
                            Low risk detected. You can safely proceed with payment.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-6 text-center text-muted-foreground text-sm">
                      Start the live detection to view analysis results
                    </div>
                  )}
                </div>
                
                <div className="bg-card rounded-lg border p-4">
                  <h3 className="text-lg font-medium mb-2">How Video Detection Works</h3>
                  <ul className="space-y-2 list-disc list-inside text-sm">
                    <li>Our AI analyzes visual patterns for known scam indicators</li>
                    <li>Audio processing detects suspicious speech patterns</li>
                    <li>Voice stress analysis identifies potential deception</li>
                    <li>Face detection monitors suspicious behaviors</li>
                  </ul>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="upload" className="w-full">
            <div className="flex flex-col md:flex-row gap-6">
              <div className="w-full md:w-7/12">
                <div className="bg-card rounded-lg border p-6 text-center">
                  <FileVideo className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-medium mb-2">Upload Video for Analysis</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Upload a video file to analyze it for potential scam indicators
                  </p>
                  
                  <label htmlFor="video-upload">
                    <div className="border-2 border-dashed border-muted rounded-lg p-8 cursor-pointer hover:bg-muted/20 transition-colors">
                      <input
                        id="video-upload"
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={handleFileUpload}
                        disabled={isUploading}
                      />
                      
                      {isUploading ? (
                        <div className="flex flex-col items-center">
                          <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full mb-2" />
                          <span className="text-sm font-medium">Analyzing video...</span>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center">
                          <Upload className="h-8 w-8 mb-2 text-muted-foreground" />
                          <span className="text-sm font-medium">Click to select video</span>
                          <span className="text-xs text-muted-foreground mt-1">
                            Supports MP4, WebM, MOV (max 100MB)
                          </span>
                        </div>
                      )}
                    </div>
                  </label>
                  
                  {uploadResult && !isUploading && (
                    <div className={`mt-4 p-3 rounded-lg flex items-center ${uploadResult.is_scam ? 'bg-destructive/10 text-destructive' : 'bg-success/10 text-success'}`}>
                      {uploadResult.is_scam ? (
                        <AlertCircle className="h-5 w-5 mr-2 flex-shrink-0" />
                      ) : (
                        <CheckCircle className="h-5 w-5 mr-2 flex-shrink-0" />
                      )}
                      <span className="text-sm font-medium">
                        {uploadResult.is_scam 
                          ? `Scam detected with ${Math.round(uploadResult.confidence * 100)}% confidence` 
                          : 'No scam detected in this video'}
                      </span>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="w-full md:w-5/12 space-y-4">
                <div className="bg-card rounded-lg border p-4">
                  <h3 className="text-lg font-medium mb-2">Analysis Results</h3>
                  
                  {uploadResult ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Overall Risk:</span>
                        <span className={`font-medium ${uploadResult.is_scam ? 'text-destructive' : 'text-success'}`}>
                          {uploadResult.is_scam ? 'High Risk' : 'Low Risk'}
                        </span>
                      </div>
                      
                      <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${uploadResult.is_scam ? 'bg-destructive' : 'bg-success'}`} 
                          style={{ width: `${Math.round(uploadResult.confidence * 100)}%` }} 
                        />
                      </div>
                      
                      <div className="pt-2 space-y-2">
                        {uploadResult.visual_confidence !== undefined && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Visual Analysis:</span>
                            <span className="text-sm font-medium">{Math.round(uploadResult.visual_confidence * 100)}%</span>
                          </div>
                        )}
                        
                        {uploadResult.audio_confidence !== undefined && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Audio Analysis:</span>
                            <span className="text-sm font-medium">{Math.round(uploadResult.audio_confidence * 100)}%</span>
                          </div>
                        )}
                        
                        {uploadResult.text_confidence !== undefined && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Text Analysis:</span>
                            <span className="text-sm font-medium">{Math.round(uploadResult.text_confidence * 100)}%</span>
                          </div>
                        )}
                      </div>
                      
                      {uploadResult.reason && (
                        <div className="pt-2 border-t">
                          <span className="text-sm text-muted-foreground">Detection Notes:</span>
                          <p className="text-sm mt-1">{uploadResult.reason}</p>
                        </div>
                      )}
                      
                      {/* Show payment button if risk is low - using the same component for consistency */}
                      {!uploadResult.is_scam && uploadResult.confidence < 0.5 && (
                        <div className="pt-4 border-t mt-4">
                          <Button 
                            onClick={handleInitiatePayment} 
                            className="w-full flex items-center justify-center gap-2"
                            variant="default"
                          >
                            <IndianRupee className="h-4 w-4" />
                            <span>Proceed to Payment</span>
                          </Button>
                          <p className="text-xs text-muted-foreground mt-2 text-center">
                            Low risk detected. You can safely proceed with payment.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-6 text-center text-muted-foreground text-sm">
                      Upload a video to view analysis results
                    </div>
                  )}
                </div>
                
                <div className="bg-card rounded-lg border p-4">
                  <h3 className="text-lg font-medium mb-2">Video Analysis Features</h3>
                  <ul className="space-y-2 list-disc list-inside text-sm">
                    <li>UPI Fraud Detection</li>
                    <li>Deepfake Recognition</li>
                    <li>Social Engineering Detection</li>
                    <li>Investment Scam Indicators</li>
                    <li>Multi-modal Analysis (visual, audio, text)</li>
                  </ul>
                  
                  <div className="mt-4">
                    <Button variant="outline" size="sm" className="w-full flex items-center gap-2">
                      <Download className="h-4 w-4" />
                      <span>Download Analysis Report</span>
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Payment Options Dialog */}
      <Dialog open={showPaymentOptions} onOpenChange={setShowPaymentOptions}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogTitle className="text-center">Payment Details</DialogTitle>
          
          <div className="space-y-6 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Enter Payment Amount
                </label>
                <div className="flex items-center">
                  <span className="mr-2">₹</span>
                  <Input
                    type="text"
                    value={paymentAmount}
                    onChange={handleAmountChange}
                    placeholder="Enter amount"
                    className="flex-1"
                  />
                </div>
              </div>
              
              <div>
                <label className="text-sm font-medium block mb-2">
                  Select Payment App
                </label>
                <div className="space-y-3">
                  {paymentApps.map((app) => (
                    <button
                      key={app.id}
                      onClick={() => handlePaymentApp(app.id)}
                      className="flex items-center w-full p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
                      disabled={isProcessingPayment}
                    >
                      <div className={`w-10 h-10 ${app.color} rounded-full flex items-center justify-center mr-3 border`}>
                        <div className="text-center font-bold text-sm">{app.name.charAt(0)}</div>
                      </div>
                      <div className="flex-1 text-left">
                        <div className="font-medium">{app.name}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="flex justify-end">
              <Button 
                variant="outline" 
                onClick={() => setShowPaymentOptions(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      
      {/* UPI PIN Dialog */}
      <UpiPinDialog
        open={showUpiPin}
        onOpenChange={setShowUpiPin}
        onSuccess={handlePaymentSuccess}
        upiApp={selectedUpiApp}
      />
    </PageLayout>
  );
}