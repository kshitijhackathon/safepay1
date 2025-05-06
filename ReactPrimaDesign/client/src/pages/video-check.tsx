import React, { useState, useRef } from 'react';
import { useLocation } from 'wouter';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { Button } from '@/components/ui/button';
import { ChevronLeft, Upload, Video, AlertTriangle, Shield, ShieldCheck, IndianRupee } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Progress } from '@/components/ui/progress';

type VideoAnalysisResult = {
  is_scam: boolean;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high';
  explanation: string;
  scam_type?: string;
  visual_confidence?: number;
  audio_confidence?: number;
  text_confidence?: number;
};

const VideoCheck: React.FC = () => {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<'record' | 'upload'>('record');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);
  const recordingTimer = useRef<NodeJS.Timeout | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoUploadRef = useRef<HTMLInputElement>(null);
  
  // Start video recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      videoChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          videoChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const videoBlob = new Blob(videoChunksRef.current, { type: 'video/webm' });
        setVideoBlob(videoBlob);
        
        // Stop all tracks of the stream
        if (videoRef.current && videoRef.current.srcObject) {
          const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
          tracks.forEach(track => track.stop());
          videoRef.current.srcObject = null;
        }
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      
      // Set up timer to track recording duration
      let seconds = 0;
      recordingTimer.current = setInterval(() => {
        seconds++;
        setRecordingTime(seconds);
        
        // Auto-stop recording after 30 seconds
        if (seconds >= 30) {
          stopRecording();
        }
      }, 1000);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      toast({
        title: "Camera Access Error",
        description: "Could not access camera. Please check permissions.",
        variant: "destructive",
      });
    }
  };
  
  // Stop video recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
        recordingTimer.current = null;
      }
    }
  };
  
  // Handle file upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      
      // Check if file is a video
      if (!file.type.startsWith('video/')) {
        toast({
          title: "Invalid File",
          description: "Please upload a video file",
          variant: "destructive",
        });
        return;
      }
      
      // Check if file is too large (>25MB)
      if (file.size > 25 * 1024 * 1024) {
        toast({
          title: "File Too Large",
          description: "Please upload a video smaller than 25MB",
          variant: "destructive",
        });
        return;
      }
      
      setUploadedFile(file);
      
      // Create object URL for preview
      if (videoRef.current) {
        videoRef.current.src = URL.createObjectURL(file);
      }
    }
  };
  
  // Analyze video (simulated for now)
  const analyzeVideo = () => {
    setIsProcessing(true);
    
    // Simulate upload progress
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 5;
      setUploadProgress(Math.min(progress, 100));
      
      if (progress >= 100) {
        clearInterval(progressInterval);
        
        // Simulate analysis delay
        setTimeout(() => {
          // Simulate analysis result - in a real app, this would come from the backend
          const simulatedResult: VideoAnalysisResult = {
            is_scam: Math.random() > 0.5,
            confidence: Math.random(),
            risk_level: Math.random() < 0.33 ? 'low' : Math.random() < 0.67 ? 'medium' : 'high',
            explanation: "Our AI system analyzed visual, audio, and textual content of your video.",
            scam_type: Math.random() > 0.5 ? "UPI Verification Scam" : "Banking Fraud",
            visual_confidence: Math.random(),
            audio_confidence: Math.random(),
            text_confidence: Math.random()
          };
          
          setAnalysisResult(simulatedResult);
          setIsProcessing(false);
        }, 2000);
      }
    }, 100);
  };
  
  // Handle proceeding to payment
  const handleInitiatePayment = () => {
    // In a real app, this would navigate to payment flow or connect to a UPI app
    toast({
      title: "Payment Initiated",
      description: "Redirecting to payment options...",
    });
    
    setTimeout(() => {
      setLocation('/confirm-transaction');
    }, 1500);
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
        <h1 className="text-lg font-medium dark-text-primary">Video Scam Check</h1>
      </div>
      
      {/* Content area */}
      <div className="flex-1 overflow-y-auto p-4 pb-20">
        <div className="mb-6">
          <p className="text-sm dark-text-secondary mb-4">
            Record or upload a video to check for potential scams. Our AI will analyze visual, 
            audio, and text content to identify common scam patterns.
          </p>
          
          {/* Tab navigation */}
          <div className="flex border-b mb-4">
            <button
              className={`flex-1 py-2 text-center text-sm ${activeTab === 'record' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500'}`}
              onClick={() => setActiveTab('record')}
            >
              Record Video
            </button>
            <button
              className={`flex-1 py-2 text-center text-sm ${activeTab === 'upload' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500'}`}
              onClick={() => setActiveTab('upload')}
            >
              Upload Video
            </button>
          </div>
          
          {/* Video preview */}
          <div className="aspect-w-16 aspect-h-9 mb-4 bg-gray-900 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay={isRecording}
              muted={isRecording}
              controls={!isRecording && (videoBlob !== null || uploadedFile !== null)}
            />
            
            {!isRecording && !videoBlob && !uploadedFile && (
              <div className="flex items-center justify-center h-full bg-gray-800">
                <Video className="h-12 w-12 text-gray-400" />
              </div>
            )}
          </div>
          
          {/* Recording timer */}
          {isRecording && (
            <div className="mb-4 flex items-center justify-center">
              <div className="bg-red-500 rounded-full h-3 w-3 mr-2 animate-pulse"></div>
              <span className="text-sm">Recording: {Math.floor(recordingTime / 60).toString().padStart(2, '0')}:{(recordingTime % 60).toString().padStart(2, '0')}</span>
            </div>
          )}
          
          {/* Tab content */}
          {activeTab === 'record' ? (
            <div className="flex justify-center gap-4">
              {!isRecording ? (
                <Button 
                  onClick={startRecording} 
                  className="flex items-center gap-2"
                  disabled={videoBlob !== null}
                >
                  <Video className="h-4 w-4" />
                  Start Recording
                </Button>
              ) : (
                <Button 
                  variant="destructive" 
                  onClick={stopRecording}
                  className="flex items-center gap-2"
                >
                  <span className="h-3 w-3 bg-white rounded-sm"></span>
                  Stop Recording
                </Button>
              )}
              
              {videoBlob && !isProcessing && !analysisResult && (
                <Button 
                  variant="default" 
                  onClick={analyzeVideo}
                  className="flex items-center gap-2"
                >
                  <Shield className="h-4 w-4" />
                  Analyze Video
                </Button>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <input 
                type="file" 
                ref={videoUploadRef}
                onChange={handleFileChange}
                accept="video/*"
                className="hidden"
              />
              
              <Button 
                onClick={() => videoUploadRef.current?.click()}
                className="flex items-center gap-2"
                disabled={uploadedFile !== null}
              >
                <Upload className="h-4 w-4" />
                Select Video
              </Button>
              
              {uploadedFile && !isProcessing && !analysisResult && (
                <div className="w-full text-center">
                  <p className="text-sm mb-2">{uploadedFile.name}</p>
                  <Button 
                    variant="default" 
                    onClick={analyzeVideo}
                    className="flex items-center gap-2 mx-auto"
                  >
                    <Shield className="h-4 w-4" />
                    Analyze Video
                  </Button>
                </div>
              )}
            </div>
          )}
          
          {/* Processing state */}
          {isProcessing && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Analyzing video...</span>
                <span className="text-sm">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="w-full" />
            </div>
          )}
          
          {/* Analysis results */}
          {analysisResult && (
            <div className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-lg border">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium dark-text-primary">Analysis Result</h3>
                <div className={`px-3 py-1 rounded-full text-white text-xs ${getRiskColor(analysisResult.risk_level)}`}>
                  {analysisResult.risk_level.toUpperCase()} RISK
                </div>
              </div>
              
              <div className="flex items-center mb-4">
                {getRiskIcon(analysisResult.risk_level)}
                <div className="ml-3">
                  <p className="font-medium">
                    {analysisResult.is_scam ? 'Potential Scam Detected!' : 'No Scam Detected'}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Confidence: {Math.round(analysisResult.confidence * 100)}%
                  </p>
                </div>
              </div>
              
              {analysisResult.is_scam && analysisResult.scam_type && (
                <div className="mb-4">
                  <p className="text-sm font-medium mb-1">Detected Scam Type:</p>
                  <p className="text-sm">{analysisResult.scam_type}</p>
                </div>
              )}
              
              <div className="mb-4">
                <p className="text-sm font-medium mb-1">Analysis Explanation:</p>
                <p className="text-sm">{analysisResult.explanation}</p>
              </div>
              
              {/* Confidence breakdown */}
              {(analysisResult.visual_confidence !== undefined || 
                analysisResult.audio_confidence !== undefined || 
                analysisResult.text_confidence !== undefined) && (
                <div className="mb-4">
                  <p className="text-sm font-medium mb-1">Analysis Breakdown:</p>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    {analysisResult.visual_confidence !== undefined && (
                      <div>
                        <p className="text-xs text-gray-500">Visual</p>
                        <p className="font-medium">{Math.round(analysisResult.visual_confidence * 100)}%</p>
                      </div>
                    )}
                    {analysisResult.audio_confidence !== undefined && (
                      <div>
                        <p className="text-xs text-gray-500">Audio</p>
                        <p className="font-medium">{Math.round(analysisResult.audio_confidence * 100)}%</p>
                      </div>
                    )}
                    {analysisResult.text_confidence !== undefined && (
                      <div>
                        <p className="text-xs text-gray-500">Text</p>
                        <p className="font-medium">{Math.round(analysisResult.text_confidence * 100)}%</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Show payment button if risk is low */}
              {!analysisResult.is_scam && analysisResult.confidence < 0.5 && (
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
          )}
        </div>
        
        {/* Info cards */}
        <div className="space-y-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h3 className="font-medium mb-2">What We Check For</h3>
            <ul className="text-sm list-disc list-inside space-y-1">
              <li>Suspicious visual elements like fake logos or altered interfaces</li>
              <li>Voice characteristics associated with scam calls</li>
              <li>Common scam phrases and urgency language</li>
              <li>Suspicious payment instructions</li>
            </ul>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-medium mb-2">Remember</h3>
            <p className="text-sm">
              Any legitimate bank or service provider will never ask for your PIN, OTP,
              or full card details over a call or in a video.
            </p>
          </div>
        </div>
      </div>
      
      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
};

export default VideoCheck;