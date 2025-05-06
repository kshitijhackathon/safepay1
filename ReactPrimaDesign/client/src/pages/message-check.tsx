import React, { useState, useRef } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { MessagesSquare, ShieldAlert, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { MessageAnalysisResult, analyzeMessageForScam } from '@/lib/scam-detection';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';

const MessageCheckPage: React.FC = () => {
  const [, setLocation] = useLocation();
  const [message, setMessage] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<MessageAnalysisResult | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    // Clear previous results when text changes
    setResult(null);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedImage(file);
      
      // Create a preview URL
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      
      // Clear previous results
      setResult(null);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const analyzeMessage = () => {
    if (!message.trim()) {
      alert('Please enter a message to analyze');
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate delay for analysis (would be real API call in production)
    setTimeout(() => {
      const analysisResult = analyzeMessageForScam(message);
      setResult(analysisResult);
      setIsAnalyzing(false);
    }, 1000);
  };

  const analyzeImage = () => {
    if (!selectedImage) {
      alert('Please select an image to analyze');
      return;
    }

    setIsAnalyzing(true);
    
    // In a real app, we'd upload the image for OCR processing
    // Here we're simulating API call for OCR and analysis
    setTimeout(() => {
      // This is a simulated OCR result - in a real app, this would come from the backend
      const extractedText = "Dear Customer, Your SBI account will be blocked. Please complete KYC verification urgently by clicking this link: http://bit.ly/update-kyc";
      
      const analysisResult = analyzeMessageForScam(extractedText);
      setResult({
        ...analysisResult,
        message: extractedText
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  const getRiskLevel = (confidence: number): 'low' | 'medium' | 'high' => {
    if (confidence < 0.3) return 'low';
    if (confidence < 0.7) return 'medium';
    return 'high';
  };

  const getRiskColor = (confidence: number): string => {
    const level = getRiskLevel(confidence);
    switch (level) {
      case 'low': return 'text-green-500 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-500 bg-yellow-50 border-yellow-200';
      case 'high': return 'text-red-500 bg-red-50 border-red-200';
      default: return 'text-gray-500 bg-gray-50 border-gray-200';
    }
  };

  const getRiskIcon = (confidence: number) => {
    const level = getRiskLevel(confidence);
    switch (level) {
      case 'low': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'medium': return <AlertCircle className="h-5 w-5 text-yellow-500" />;
      case 'high': return <ShieldAlert className="h-5 w-5 text-red-500" />;
      default: return null;
    }
  };

  const clearAll = () => {
    setMessage('');
    setSelectedImage(null);
    setImagePreview(null);
    setResult(null);
  };

  return (
    <div className="container mx-auto py-6 px-4 max-w-2xl">
      <h1 className="text-2xl font-bold mb-6 text-center">Message Fraud Detection</h1>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Analyze Text Message</CardTitle>
          <CardDescription>Enter the message to check for potential scams</CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            value={message}
            onChange={handleTextChange}
            placeholder="Paste the suspicious message here..."
            className="mb-4 min-h-[100px]"
          />
          <div className="flex justify-between">
            <Button variant="outline" onClick={clearAll}>
              Clear
            </Button>
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
                  <MessagesSquare className="h-4 w-4" />
                  Analyze Message
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Upload Screenshot</CardTitle>
          <CardDescription>Upload a screenshot of the suspicious message</CardDescription>
        </CardHeader>
        <CardContent>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleImageUpload}
            accept="image/*"
            className="hidden"
          />
          {imagePreview ? (
            <div className="mb-4">
              <img 
                src={imagePreview} 
                alt="Message Screenshot" 
                className="max-h-[200px] mx-auto rounded-md border"
              />
            </div>
          ) : (
            <div 
              onClick={triggerFileInput}
              className="border-2 border-dashed rounded-md p-8 mb-4 text-center cursor-pointer hover:bg-gray-50"
            >
              <FileText className="h-8 w-8 mx-auto mb-2 text-gray-400" />
              <p className="text-sm text-gray-500">Click to upload an image</p>
            </div>
          )}
          <div className="flex justify-between">
            <Button variant="outline" onClick={clearAll}>
              Clear
            </Button>
            <Button 
              onClick={analyzeImage} 
              disabled={!selectedImage || isAnalyzing}
              className="flex items-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing Image...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4" />
                  Extract & Analyze
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {isAnalyzing && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="text-center mb-2">
              <Loader2 className="h-8 w-8 mx-auto animate-spin text-primary mb-2" />
              <p>Analyzing message for fraud indicators...</p>
            </div>
            <Progress value={65} className="mb-2" />
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getRiskIcon(result.confidence)}
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium mb-2">Message</h3>
                <div className="p-3 bg-muted rounded-md text-sm">{result.message}</div>
              </div>

              <Separator />

              <div>
                <h3 className="font-medium mb-2">Risk Assessment</h3>
                <div className={`p-3 rounded-md border ${getRiskColor(result.confidence)}`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getRiskIcon(result.confidence)}
                      <span className="font-semibold">
                        {getRiskLevel(result.confidence) === 'low' ? 'Low Risk' : 
                         getRiskLevel(result.confidence) === 'medium' ? 'Medium Risk' : 'High Risk'}
                      </span>
                    </div>
                    <span className="text-sm">
                      {Math.round(result.confidence * 100)}% confidence
                    </span>
                  </div>
                  {result.scamType && (
                    <div className="text-sm">
                      <span className="font-medium">Detected type:</span> {result.scamType}
                    </div>
                  )}
                </div>
              </div>

              {result.scamIndicators.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <h3 className="font-medium mb-2">Detected Indicators</h3>
                    <ul className="list-disc pl-5 space-y-1">
                      {result.scamIndicators.map((indicator, index) => (
                        <li key={index} className="text-sm">{indicator}</li>
                      ))}
                    </ul>
                  </div>
                </>
              )}

              {result.isScam && (
                <>
                  <Separator />
                  <Alert variant="destructive">
                    <ShieldAlert className="h-4 w-4" />
                    <AlertTitle>Warning: Potential Scam Detected</AlertTitle>
                    <AlertDescription>
                      This message contains several indicators of a scam. Do not share any personal information, 
                      click on links, or send money/UPI payments in response to this message.
                    </AlertDescription>
                  </Alert>
                  <div className="flex justify-end">
                    <Button 
                      onClick={() => setLocation('/report-scam')}
                      className="flex items-center gap-2"
                    >
                      Report This Scam
                    </Button>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default MessageCheckPage;