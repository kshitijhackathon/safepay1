import { useState, useRef } from "react";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { AlertCircle, Upload, Check, Loader2, AlertTriangle } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import MainLayout from "@/layouts/main-layout";

interface KeywordAnalysis {
  score: number;
  match_count: number;
  categories: Record<string, number>;
}

interface AIAnalysis {
  is_scam: boolean;
  confidence: number;
  scam_type: string | null;
}

interface WhatsAppAnalysisResult {
  is_scam: boolean;
  confidence: number;
  status: "Scam Likely" | "Suspicious Message" | "Safe Message";
  scam_type: string | null;
  scam_indicators: string[];
  ocr_text: string | null;
  keyword_analysis: KeywordAnalysis;
  ai_analysis: AIAnalysis;
}

const WhatsAppCheck = () => {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [description, setDescription] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<WhatsAppAnalysisResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setImage(selectedFile);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (event) => {
        setImagePreview(event.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDescriptionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDescription(e.target.value);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleSubmit = async () => {
    if (!image && !description) {
      toast({
        title: "Input required",
        description: "Please provide a screenshot or message description",
        variant: "destructive",
      });
      return;
    }

    setAnalyzing(true);
    setResult(null);

    try {
      // Create form data for file upload
      const formData = new FormData();
      if (image) {
        formData.append("image", image);
      }
      formData.append("description", description);

      const response = await fetch("/api/analyze-whatsapp", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Error analyzing message");
      }

      const data = await response.json();
      setResult(data);
      
      // Show toast based on result
      if (data.is_scam) {
        toast({
          title: "Warning: Potential Scam Detected",
          description: `Confidence: ${Math.round(data.confidence * 100)}%`,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Message appears legitimate",
          description: `Confidence: ${Math.round(data.confidence * 100)}%`,
          variant: "default",
        });
      }
    } catch (error) {
      console.error("Error:", error);
      toast({
        title: "Analysis failed",
        description: "An error occurred while analyzing the message",
        variant: "destructive",
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const resetForm = () => {
    setImage(null);
    setImagePreview(null);
    setDescription("");
    setResult(null);
  };

  return (
    <MainLayout>
      <div className="container mx-auto p-4 max-w-md">
        <h1 className="text-2xl font-bold text-center mb-6">WhatsApp Message Check</h1>
        
        <Card>
          <CardHeader>
            <CardTitle>WhatsApp Scam Detector</CardTitle>
            <CardDescription>
              Upload a WhatsApp screenshot or paste message text to analyze for potential scams. Our system combines AI analysis, text extraction, and scam keyword matching to provide comprehensive protection.
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="upload">Upload Screenshot (Optional)</Label>
              <input
                type="file"
                id="upload"
                accept="image/*"
                onChange={handleImageChange}
                className="hidden"
                ref={fileInputRef}
              />
              
              <div 
                onClick={triggerFileInput}
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50 transition-colors"
              >
                {imagePreview ? (
                  <div className="space-y-2">
                    <img 
                      src={imagePreview} 
                      alt="WhatsApp message preview" 
                      className="max-h-40 mx-auto rounded"
                    />
                    <p className="text-sm text-muted-foreground">Click to change</p>
                  </div>
                ) : (
                  <div className="py-4">
                    <Upload className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                    <p>Click to upload a screenshot</p>
                  </div>
                )}
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="description">Message Text (Optional)</Label>
              <Textarea
                id="description"
                placeholder="Paste the message content or describe what it says..."
                value={description}
                onChange={handleDescriptionChange}
                rows={4}
              />
            </div>
          </CardContent>
          
          <CardFooter className="flex flex-col space-y-3">
            <div className="flex w-full space-x-2">
              <Button variant="outline" onClick={resetForm} className="flex-1">
                Reset
              </Button>
              <Button 
                onClick={handleSubmit} 
                disabled={analyzing || (!image && !description)}
                className="flex-1 bg-primary"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Analyze Message"
                )}
              </Button>
            </div>
            
            {result && (
              <div className={`w-full p-4 rounded-lg ${
                result.status === "Scam Likely" ? 'bg-red-50 dark:bg-red-900/30' : 
                result.status === "Suspicious Message" ? 'bg-amber-50 dark:bg-amber-900/30' : 
                'bg-green-50 dark:bg-green-900/30'
              }`}>
                <div className="flex items-start">
                  {result.status === "Scam Likely" ? (
                    <AlertTriangle className="h-6 w-6 text-red-500 dark:text-red-400 mr-3 mt-0.5 flex-shrink-0" />
                  ) : result.status === "Suspicious Message" ? (
                    <AlertTriangle className="h-6 w-6 text-amber-500 dark:text-amber-400 mr-3 mt-0.5 flex-shrink-0" />
                  ) : (
                    <Check className="h-6 w-6 text-green-500 dark:text-green-400 mr-3 mt-0.5 flex-shrink-0" />
                  )}
                  
                  <div className="flex-1">
                    <h3 className={`text-lg font-medium ${
                      result.status === "Scam Likely" ? 'text-red-800 dark:text-red-300' : 
                      result.status === "Suspicious Message" ? 'text-amber-800 dark:text-amber-300' : 
                      'text-green-800 dark:text-green-300'
                    }`}>
                      {result.status}
                    </h3>
                    
                    <div className="flex items-center mt-1 mb-2">
                      <div className="flex-1">
                        <div className="h-2 relative rounded-full overflow-hidden bg-gray-200 dark:bg-gray-700">
                          <div 
                            className={`h-full ${
                              result.status === "Scam Likely" ? 'bg-red-500' : 
                              result.status === "Suspicious Message" ? 'bg-amber-500' : 
                              'bg-green-500'
                            }`}
                            style={{ width: `${Math.round(result.confidence * 100)}%` }}
                          />
                        </div>
                      </div>
                      <span className="ml-2 text-xs font-medium">
                        {Math.round(result.confidence * 100)}% confidence
                      </span>
                    </div>
                    
                    {result.scam_type && (
                      <div className="mb-3">
                        <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full ${
                          result.status === "Scam Likely" ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 
                          'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200'
                        }`}>
                          {result.scam_type}
                        </span>
                      </div>
                    )}
                    
                    {result.ocr_text && (
                      <div className="mb-3 p-2 bg-gray-100 dark:bg-gray-800 rounded-md">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Extracted text from image:</p>
                        <p className="text-xs line-clamp-3">{result.ocr_text}</p>
                      </div>
                    )}
                    
                    {result.scam_indicators && result.scam_indicators.length > 0 && (
                      <div className="mt-3">
                        <p className={`text-sm font-medium mb-1 ${
                          result.status === "Scam Likely" ? 'text-red-700 dark:text-red-300' : 
                          result.status === "Suspicious Message" ? 'text-amber-700 dark:text-amber-300' : 
                          'text-green-700 dark:text-green-300'
                        }`}>
                          Warning signs detected:
                        </p>
                        <ul className="text-xs space-y-1 mt-2">
                          {result.scam_indicators.map((indicator: string, idx: number) => (
                            <li key={idx} className="flex items-start">
                              <span className="mr-1.5 mt-0.5">•</span>
                              <span>{indicator}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {result.keyword_analysis && result.keyword_analysis.match_count > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                          Categories of suspicious content detected:
                        </p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {Object.entries(result.keyword_analysis.categories || {}).map(([category, count], idx) => (
                            <span 
                              key={idx} 
                              className="inline-block px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-800 rounded-full"
                            >
                              {category} ({String(count)})
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {result.status === "Scam Likely" && (
                      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <p className="text-sm font-medium text-red-700 dark:text-red-300">Recommended actions:</p>
                        <ul className="text-xs mt-2 space-y-1">
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>Do not click on any links in this message</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>Do not share personal information or credentials</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>Block and report the sender</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>If you've already responded, secure your accounts immediately</span>
                          </li>
                        </ul>
                      </div>
                    )}
                    
                    {result.status === "Suspicious Message" && (
                      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <p className="text-sm font-medium text-amber-700 dark:text-amber-300">Proceed with caution:</p>
                        <ul className="text-xs mt-2 space-y-1">
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>Verify the sender through another communication channel</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>Do not share sensitive information without verification</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-1.5 mt-0.5">•</span>
                            <span>Be cautious of any unusual requests</span>
                          </li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </CardFooter>
        </Card>
      </div>
    </MainLayout>
  );
};

export default WhatsAppCheck;