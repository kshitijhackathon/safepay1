import React, { useState, useRef, useEffect } from 'react';
import { Mic, StopCircle, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { apiRequest } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';

interface VoiceScamDetectorProps {
  onAnalysisComplete?: (result: any) => void;
  showVisualizer?: boolean;
  fullPage?: boolean;
}

export default function VoiceScamDetector({ 
  onAnalysisComplete, 
  showVisualizer = true,
  fullPage = false
}: VoiceScamDetectorProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [transcript, setTranscript] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [visualizerValues, setVisualizerValues] = useState<number[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const { toast } = useToast();

  // Initialize audio context for visualization
  useEffect(() => {
    return () => {
      // Clean up animation frame on unmount
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      // Clean up audio context
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Set up audio visualization
  const setupAudioVisualization = async (stream: MediaStream) => {
    if (!showVisualizer) return;
    
    try {
      // Create audio context
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);
      
      // Set up data array for visualization
      const bufferLength = analyserRef.current.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);
      
      // Start visualization
      visualize();
    } catch (err) {
      console.error('Error setting up audio visualization:', err);
    }
  };

  // Visualize audio input
  const visualize = () => {
    if (!analyserRef.current || !dataArrayRef.current) return;
    
    analyserRef.current.getByteFrequencyData(dataArrayRef.current);
    
    // Calculate average levels for visualization
    const values = [];
    const numBars = 10;
    const segmentLength = Math.floor(dataArrayRef.current.length / numBars);
    
    for (let i = 0; i < numBars; i++) {
      let sum = 0;
      for (let j = 0; j < segmentLength; j++) {
        sum += dataArrayRef.current[i * segmentLength + j];
      }
      const average = sum / segmentLength;
      values.push(average);
    }
    
    setVisualizerValues(values);
    
    // Continue animation
    animationFrameRef.current = requestAnimationFrame(visualize);
  };

  // Start recording
  const startRecording = async () => {
    try {
      setError(null);
      setTranscript('');
      setAnalysisResult(null);
      audioChunksRef.current = [];
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Set up audio visualization
      setupAudioVisualization(stream);
      
      // Create and start media recorder
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(audioBlob);
        
        // Stop all tracks on the stream
        stream.getTracks().forEach(track => track.stop());
        
        // Stop visualization
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        
        setVisualizerValues([]);
        
        // Automatically analyze the recording
        analyzeAudio(audioBlob);
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      toast({
        title: 'Recording started',
        description: 'Speak clearly to analyze for potential scams.',
      });
    } catch (err: any) {
      console.error('Error starting recording:', err);
      setError('Microphone access denied or not available.');
      toast({
        title: 'Recording failed',
        description: 'Could not access microphone. Please check permissions.',
        variant: 'destructive',
      });
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      toast({
        title: 'Recording stopped',
        description: 'Processing your audio...',
      });
    }
  };

  // Analyze audio for scams
  const analyzeAudio = async (blob: Blob) => {
    setIsAnalyzing(true);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');
      
      // Send to ML endpoint for analysis
      const response = await apiRequest('POST', '/api/ml-process-audio', formData, true); // true for form data
      const result = await response.json();
      
      if (result.status === 'success') {
        setTranscript(result.transcript || 'Could not transcribe audio');
        setAnalysisResult(result.analysis);
        
        // Call callback if provided
        if (onAnalysisComplete) {
          onAnalysisComplete(result);
        }
      } else {
        setError(result.message || 'Analysis failed');
        toast({
          title: 'Analysis failed',
          description: result.message || 'Could not analyze audio',
          variant: 'destructive',
        });
      }
    } catch (err: any) {
      console.error('Error analyzing audio:', err);
      setError('Failed to analyze audio. Please try again.');
      toast({
        title: 'Analysis error',
        description: 'Failed to analyze audio. Please try again.',
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
    setAudioBlob(null);
    setTranscript('');
    setAnalysisResult(null);
    setError(null);
    setVisualizerValues([]);
    
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Render voice visualizer
  const renderVisualizer = () => {
    if (!showVisualizer) return null;
    
    return (
      <div className="flex items-end justify-center h-16 gap-1 my-4">
        {visualizerValues.length > 0 ? (
          visualizerValues.map((value, index) => (
            <div
              key={index}
              className="w-2 bg-primary rounded-t transition-all duration-100 ease-out"
              style={{ 
                height: `${Math.max(4, value / 2)}%`,
                opacity: value / 256 + 0.2
              }}
            />
          ))
        ) : (
          // Placeholder bars when not recording
          Array.from({ length: 10 }).map((_, index) => (
            <div
              key={index}
              className="w-2 bg-gray-200 rounded-t"
              style={{ height: '4%' }}
            />
          ))
        )}
      </div>
    );
  };

  // Main component render
  return (
    <div className={`${fullPage ? 'h-full' : ''}`}>
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Mic className="mr-2 h-5 w-5" />
            Voice Scam Detector
          </CardTitle>
          <CardDescription>
            Record your voice to analyze potential scam calls or messages
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          {/* Voice Visualizer */}
          {renderVisualizer()}
          
          {/* Recording Controls */}
          <div className="flex justify-center gap-4 mb-4">
            {!isRecording ? (
              <Button
                variant="default"
                size="lg"
                onClick={startRecording}
                disabled={isRecording || isAnalyzing}
                className="flex items-center gap-2 px-6"
              >
                <Mic className="h-5 w-5" />
                Start Recording
              </Button>
            ) : (
              <Button
                variant="destructive"
                size="lg"
                onClick={stopRecording}
                disabled={!isRecording || isAnalyzing}
                className="flex items-center gap-2 px-6"
              >
                <StopCircle className="h-5 w-5" />
                Stop Recording
              </Button>
            )}
            
            {audioBlob && !isRecording && !isAnalyzing && (
              <Button
                variant="outline"
                onClick={handleReset}
                className="flex items-center gap-2"
              >
                Reset
              </Button>
            )}
          </div>
          
          {/* Error Display */}
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {/* Loading State */}
          {isAnalyzing && (
            <div className="flex flex-col items-center justify-center py-4">
              <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
              <p className="text-sm text-gray-500">Analyzing audio content...</p>
              <Progress value={45} className="w-full mt-2" />
            </div>
          )}
          
          {/* Analysis Results */}
          {analysisResult && (
            <Tabs defaultValue="summary" className="w-full mt-4">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="details">Details</TabsTrigger>
              </TabsList>
              
              <TabsContent value="summary" className="space-y-4">
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
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
                </div>
              </TabsContent>
              
              <TabsContent value="details" className="space-y-4">
                {transcript && (
                  <div className="mt-2">
                    <h3 className="font-semibold mb-1">Transcript:</h3>
                    <div className="p-3 bg-gray-50 rounded-md text-sm">
                      <p>{transcript}</p>
                    </div>
                  </div>
                )}
                
                {analysisResult.scam_type && (
                  <div className="mt-2">
                    <h3 className="font-semibold mb-1">Scam Type:</h3>
                    <Badge variant="outline">{analysisResult.scam_type}</Badge>
                  </div>
                )}
                
                {analysisResult.scam_indicators && analysisResult.scam_indicators.length > 0 && (
                  <div className="mt-2">
                    <h3 className="font-semibold mb-1">Warning Signs:</h3>
                    <ul className="list-disc list-inside space-y-1">
                      {analysisResult.scam_indicators.map((indicator: string, index: number) => (
                        <li key={index} className="text-sm">{indicator}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div className="flex items-center justify-between text-xs text-gray-500 mt-4">
                  <span>Confidence: {Math.round(analysisResult.confidence * 100)}%</span>
                  <span>Analysis: {analysisResult.analysis_method || 'ML Model'}</span>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
        
        <CardFooter className="flex justify-between text-xs text-gray-500">
          <span>Voice analysis powered by ML</span>
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