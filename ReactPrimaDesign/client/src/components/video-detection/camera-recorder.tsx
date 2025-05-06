import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, Camera, CheckCircle, Info, Loader2, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

type VideoDetectionResult = {
  is_scam: boolean;
  confidence: number;
  visual_confidence?: number;
  audio_confidence?: number;
  text_confidence?: number;
  reason?: string;
  features?: {
    face_count: number;
    face_ratio: number;
    eye_contact: number;
    edge_density: number;
  };
};

type CameraRecorderProps = {
  onResult?: (result: VideoDetectionResult) => void;
  onError?: (error: Error) => void;
  className?: string;
};

export function CameraRecorder({ onResult, onError, className }: CameraRecorderProps) {
  // Get toast hook
  const { toast } = useToast();
  
  // Refs for video and canvas elements
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // State for recording and analysis
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VideoDetectionResult | null>(null);
  const [usingFallback, setUsingFallback] = useState(false);
  
  // Refs for WebSocket connection and analysis
  const socketRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mockAnalysisCountRef = useRef(0);
  
  // Function to generate mock analysis results when WebSocket fails
  const generateMockAnalysisResult = (): VideoDetectionResult => {
    // Use a time-based seed to create a deterministic pattern that changes every 3 seconds
    const currentTime = Math.floor(Date.now() / 3000); // Changes every 3 seconds
    const cycle = currentTime % 5; // 5 different analysis states
    
    // Base risk on the cycle to create an observable pattern
    let baseRisk = 0;
    let reason = '';
    let faceCount = 1;
    let eyeContact = 0.5;
    
    switch(cycle) {
      case 0: 
        baseRisk = 0.15; 
        reason = 'Low risk: Normal video patterns detected';
        faceCount = 1;
        eyeContact = 0.8;
        break;
      case 1: 
        baseRisk = 0.35; 
        reason = 'Slight risk: Some unusual patterns in audio';
        faceCount = 1;
        eyeContact = 0.6;
        break;
      case 2: 
        baseRisk = 0.55; 
        reason = 'Medium risk: Unusual video angle and reduced eye contact';
        faceCount = 1;
        eyeContact = 0.3;
        break;
      case 3: 
        baseRisk = 0.75; 
        reason = 'High risk: Multiple scam indicators detected';
        faceCount = 2;
        eyeContact = 0.1;
        break;
      case 4: 
        baseRisk = 0.9; 
        reason = 'Very high risk: Multiple strong scam indicators';
        faceCount = 0;
        eyeContact = 0;
        break;
    }
    
    // Add slight randomness for visual effect
    const randomFactor = Math.random() * 0.1 - 0.05; // -0.05 to 0.05
    const confidence = Math.max(0, Math.min(1, baseRisk + randomFactor));
    
    // Create varied but consistent data for visualization
    const visual = Math.max(0.1, Math.min(0.95, confidence - 0.1 + Math.random() * 0.2));
    const audio = Math.max(0.1, Math.min(0.95, confidence + 0.05 + Math.random() * 0.15));
    const text = Math.max(0.1, Math.min(0.95, confidence - 0.05 + Math.random() * 0.1));
    
    return {
      is_scam: confidence > 0.5,
      confidence: confidence,
      visual_confidence: visual,
      audio_confidence: audio,
      text_confidence: text,
      reason: reason,
      features: {
        face_count: faceCount,
        face_ratio: 0.3 + Math.random() * 0.4,
        eye_contact: eyeContact,
        edge_density: 0.2 + confidence * 0.6 // Higher risk = higher edge density
      }
    };
  };
  
  // Connect to WebSocket server with fallback
  const connectWebSocket = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }
    
    setConnecting(true);
    setError(null);
    setUsingFallback(false);
    
    try {
      // Create WebSocket connection - try multiple potential URLs
      // Use the same port as the current connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/video-detection`;
      
      console.log('Connecting to WebSocket:', wsUrl);
      const socket = new WebSocket(wsUrl);
      
      // Set connection timeout
      const connectionTimeout = setTimeout(() => {
        if (socketRef.current?.readyState !== WebSocket.OPEN) {
          console.log('WebSocket connection timed out, using fallback mode');
          socket.close();
          enableFallbackMode();
        }
      }, 3000);
      
      socket.onopen = () => {
        console.log('WebSocket connected');
        clearTimeout(connectionTimeout);
        setConnecting(false);
        setConnected(true);
        socketRef.current = socket;
      };
      
      socket.onclose = () => {
        console.log('WebSocket disconnected');
        clearTimeout(connectionTimeout);
        
        if (!usingFallback) {
          setConnected(false);
          setIsRecording(false);
          socketRef.current = null;
        }
      };
      
      socket.onerror = (event) => {
        console.error('WebSocket error:', event);
        clearTimeout(connectionTimeout);
        enableFallbackMode();
      };
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'connected') {
            console.log('Connection confirmed:', data.message);
          }
          else if (data.type === 'result') {
            // Update the result state
            setResult({
              is_scam: data.is_scam,
              confidence: data.confidence,
              visual_confidence: data.visual_confidence,
              features: data.features
            });
            
            // Call the onResult callback if provided
            if (onResult) {
              onResult({
                is_scam: data.is_scam,
                confidence: data.confidence,
                visual_confidence: data.visual_confidence,
                features: data.features
              });
            }
          }
          else if (data.type === 'error') {
            console.error('Analysis error:', data.message);
            setError(data.message || 'Error analyzing video');
            setAnalyzing(false);
            
            if (onError) {
              onError(new Error(data.message || 'Error analyzing video'));
            }
          }
        } catch (error) {
          console.error('Error handling WebSocket message:', error);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      enableFallbackMode();
    }
  };
  
  // Enable fallback mode when WebSocket connection fails
  const enableFallbackMode = () => {
    console.log('Enabling fallback mode for video analysis');
    setUsingFallback(true);
    setConnecting(false);
    setConnected(true);
    setError(null);
  };
  
  // Request camera access
  const startCamera = async () => {
    try {
      setError(null);
      
      // Connect to WebSocket first if not connected
      if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
        // Try to connect but allow a fallback
        try {
          connectWebSocket();
          
          // If after 500ms we still don't have a connection, enable fallback mode
          setTimeout(() => {
            if (!connected && !connecting && !usingFallback) {
              console.log('Auto-enabling fallback mode after delayed connection attempt');
              enableFallbackMode();
            }
          }, 500);
        } catch (wsError) {
          console.error('WebSocket connection failed:', wsError);
          enableFallbackMode();
        }
      }
      
      // Get camera stream (this happens in parallel with WebSocket connection)
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });
      
      // Set stream to video element
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
      }
    } catch (error) {
      console.error('Failed to access camera:', error);
      setError('Failed to access camera: ' + (error instanceof Error ? error.message : 'Unknown error'));
      
      if (onError) {
        onError(error instanceof Error ? error : new Error('Failed to access camera'));
      }
    }
  };
  
  // Start recording and sending frames for analysis
  const startRecording = () => {
    if (!stream) {
      setError('Camera not ready');
      return;
    }
    
    // In fallback mode, we don't need a WebSocket connection
    if (!usingFallback && (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN)) {
      setError('Video analysis service not ready');
      return;
    }
    
    setIsRecording(true);
    setAnalyzing(true);
    setError(null);
    frameCountRef.current = 0;
    mockAnalysisCountRef.current = 0;
    
    // If using WebSocket, send start message
    if (!usingFallback && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'start' }));
      
      // Start sending frames to WebSocket
      sendFrames();
    } else {
      // Using fallback mode, start mock analysis
      startMockAnalysis();
    }
  };
  
  // Stop recording
  const stopRecording = () => {
    setIsRecording(false);
    setAnalyzing(false);
    
    // Stop any existing frame interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // If using WebSocket, send stop message
    if (!usingFallback && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'stop' }));
    }
  };
  
  // Stop camera and clear stream
  const stopCamera = () => {
    if (stream) {
      console.log('Stopping camera');
      
      // Stop all tracks
      stream.getTracks().forEach(track => {
        console.log('Stopping track:', track.kind);
        track.stop();
      });
      
      // Clear video source
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      
      // Reset state
      setStream(null);
      setConnected(false);
      setResult(null);
    }
  };
  
  // Start mock analysis for fallback mode
  const startMockAnalysis = () => {
    console.log('Starting mock analysis with isRecording:', isRecording);
    
    // Generate first result immediately
    const firstResult = generateMockAnalysisResult();
    setResult(firstResult);
    if (onResult) {
      onResult(firstResult);
    }
    
    // Set a notification toast for clear visibility
    toast({
      title: "Analysis Started",
      description: "Analysis will update every 3 seconds in demo mode",
      duration: 3000,
    });
    
    // Clear any existing interval to avoid multiple intervals
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // Create a new interval to generate mock results every 3 seconds
    const intervalId = setInterval(() => {
      // Check recording state at each interval execution time
      console.log('Interval executing, isRecording:', isRecording);
      
      if (!isRecording) {
        console.log('Stopping interval because isRecording is false');
        clearInterval(intervalId);
        return;
      }
      
      // Generate mock analysis result with more variation
      const mockResult = generateMockAnalysisResult();
      console.log('Generated new mock result with confidence:', mockResult.confidence);
      
      // Update state and call callbacks
      setResult(mockResult);
      if (onResult) {
        onResult(mockResult);
      }
      
      // Add a visual indicator that analysis has been updated
      toast({
        title: "Analysis Updated",
        description: `Result: ${mockResult.is_scam ? 'Potential scam detected' : 'No scam detected'} (${Math.round(mockResult.confidence * 100)}% confidence)`,
        duration: 2000,
      });
    }, 3000); // Update exactly every 3 seconds as requested
    
    // Store the interval ID for cleanup
    frameIntervalRef.current = intervalId;
  };
  
  // Capture and send frames to WebSocket
  const sendFrames = () => {
    if (!isRecording || !videoRef.current || !canvasRef.current) {
      return;
    }
    
    // For fallback mode, we don't need to check WebSocket
    if (!usingFallback && (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN)) {
      return;
    }
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) return;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    try {
      if (!usingFallback) {
        // Convert canvas to base64 image for WebSocket
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send frame to WebSocket
        socketRef.current!.send(JSON.stringify({
          type: 'frame',
          frame: frameCountRef.current++,
          data: frameData.split(',')[1] // Remove data:image/jpeg;base64, prefix
        }));
      }
      
      // Schedule next frame
      if (isRecording) {
        setTimeout(sendFrames, 100); // Send 10 frames per second
      }
    } catch (error) {
      console.error('Error sending frame:', error);
      
      // If WebSocket fails, switch to fallback mode
      if (!usingFallback) {
        console.log('Switching to fallback mode after frame error');
        enableFallbackMode();
        startMockAnalysis();
      } else {
        setError('Error analyzing video: ' + (error instanceof Error ? error.message : 'Unknown error'));
        setIsRecording(false);
        setAnalyzing(false);
      }
    }
  };
  
  // Connect WebSocket when component mounts
  useEffect(() => {
    connectWebSocket();
    
    // Clean up on unmount
    return () => {
      // Stop recording if active
      if (isRecording) {
        stopRecording();
      }
      
      // Close WebSocket connection
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []); // Only run once on mount
  
  // Handle stream cleanup separately to ensure it happens
  useEffect(() => {
    // Return cleanup function to stop camera when component unmounts or stream changes
    return () => {
      if (stream) {
        console.log('Cleaning up camera streams');
        stream.getTracks().forEach(track => {
          console.log('Stopping track:', track.kind);
          track.stop();
        });
        
        // If video ref still exists, clear it
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        
        setStream(null);
      }
    };
  }, [stream]);
  
  return (
    <Card className={cn("w-full max-w-md mx-auto", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Camera className="h-5 w-5" />
          Live Video Analysis
        </CardTitle>
        <CardDescription>
          Scan for potential scams in real-time with AI-powered detection
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Video preview */}
        <div className="relative rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800 aspect-video">
          {!stream && (
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <Camera className="h-12 w-12 text-muted-foreground mb-2" />
              <span className="text-sm text-muted-foreground">Camera preview will appear here</span>
            </div>
          )}
          
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={cn(
              "w-full h-full object-cover",
              !stream && "hidden"
            )}
          />
          
          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} className="hidden" />
          
          {/* Status overlay */}
          {connecting && (
            <div className="absolute inset-0 bg-background/80 flex flex-col items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
              <span className="text-sm font-medium">Connecting to analysis service...</span>
            </div>
          )}
          
          {/* Analysis result indicator */}
          {result && !analyzing && (
            <div className="absolute bottom-2 right-2 bg-background/90 rounded-lg p-2 flex items-center gap-2">
              {result.is_scam ? (
                <XCircle className="h-5 w-5 text-destructive" />
              ) : (
                <CheckCircle className="h-5 w-5 text-success" />
              )}
              <span className={cn(
                "text-sm font-medium",
                result.is_scam ? "text-destructive" : "text-success"
              )}>
                {result.is_scam 
                  ? `${Math.round(result.confidence * 100)}% Scam Risk` 
                  : 'No Scam Detected'}
              </span>
            </div>
          )}
        </div>
        
        {/* Analysis progress */}
        {analyzing && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">
                {usingFallback 
                  ? "Running local analysis (demo mode)" 
                  : "Analyzing video stream"}
              </span>
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
            </div>
            <Progress value={result ? Math.round(result.confidence * 100) : 0} className="h-2" />
          </div>
        )}
        
        {/* Fallback mode indicator */}
        {usingFallback && connected && !error && (
          <div className="bg-muted/50 rounded-lg p-2 flex items-center gap-2 text-muted-foreground">
            <Info className="h-4 w-4 flex-shrink-0" />
            <span className="text-xs">
              Running in demo mode. Analysis will be simulated for demonstration purposes.
            </span>
          </div>
        )}
        
        {/* Error message */}
        {error && (
          <div className="bg-destructive/10 text-destructive rounded-lg p-3 flex items-start gap-2">
            <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <span className="text-sm">{error}</span>
          </div>
        )}
      </CardContent>
      
      <CardFooter className="flex flex-col gap-2">
        {!stream ? (
          <Button onClick={startCamera} className="w-full">
            <Camera className="h-4 w-4 mr-2" />
            Access Camera
          </Button>
        ) : !isRecording ? (
          <div className="w-full flex gap-2">
            <Button onClick={startRecording} className="flex-1" disabled={!connected}>
              Start Analysis
            </Button>
            <Button onClick={stopCamera} variant="outline" className="flex-none">
              Close Camera
            </Button>
          </div>
        ) : (
          <Button onClick={stopRecording} variant="secondary" className="w-full">
            Stop Analysis
          </Button>
        )}
        
        {/* Show close camera button after analysis is complete if we have a result */}
        {!isRecording && stream && result && (
          <Button onClick={stopCamera} variant="outline" className="w-full mt-2">
            Close Camera
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}