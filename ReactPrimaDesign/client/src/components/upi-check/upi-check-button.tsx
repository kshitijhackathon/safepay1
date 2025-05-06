import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { apiRequest } from "@/lib/queryClient";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

// Types for UPI check response
interface DomainAnalysis {
  domain: string;
  count: number;
}

interface UpiCheckResponse {
  upiId: string;
  status: 'SAFE' | 'SUSPICIOUS' | 'SCAM';
  riskPercentage: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  reports: number;
  reason: string;
  confidence_score: number;
  risk_factors?: string[];
  recommendations?: string[];
  age?: string;
  reportedFor?: string;
  safety_score?: number; // AI-generated safety score
  ai_analysis?: string; // AI-generated explanation
}

export default function UpiCheckButton() {
  const [upiId, setUpiId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<UpiCheckResponse | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();

  const handleUpiIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUpiId(e.target.value);
  };

  const checkUpi = async () => {
    if (!upiId) {
      toast({
        title: "UPI ID Required",
        description: "Please enter a UPI ID to check",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    try {
      // Use the enhanced /check-scam endpoint
      const response = await apiRequest('POST', '/api/check-scam', { upiId });
      const data = await response.json();
      
      if (response.ok) {
        // Process the enhanced response format with all needed details
        const riskPercentage = data.status === 'SCAM' 
          ? 90  // High risk for scam
          : data.status === 'SUSPICIOUS' 
            ? 60  // Medium risk for suspicious
            : data.safety_score 
              ? 100 - data.safety_score // Invert safety score for risk percentage
              : Math.round((1 - data.confidence_score) * 30); // Default calculation
              
        setResults({
          upiId: upiId,
          status: data.status,
          riskPercentage: riskPercentage,
          riskLevel: data.status === 'SCAM' ? 'High' : 
                    data.status === 'SUSPICIOUS' ? 'Medium' : 'Low',
          reports: data.reports || 0,
          reason: data.reason,
          confidence_score: data.confidence_score,
          risk_factors: data.risk_factors || [],
          recommendations: data.recommendations || [],
          reportedFor: data.category || 'N/A',
          // Include new AI analysis fields
          safety_score: data.safety_score,
          ai_analysis: data.ai_analysis
        });
        
        // Show toast with the result
        const statusMap = {
          'SAFE': { title: 'Safe UPI ID', variant: 'default' },
          'SUSPICIOUS': { title: 'Suspicious UPI ID', variant: 'warning' },
          'SCAM': { title: 'Scam Alert!', variant: 'destructive' }
        };
        
        const status = data.status as keyof typeof statusMap;
        
        toast({
          title: statusMap[status].title,
          description: data.reason,
          variant: statusMap[status].variant as any
        });
      } else {
        toast({
          title: "Check Failed",
          description: data.message || "Failed to analyze UPI ID",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('UPI check error:', error);
      toast({
        title: "Check Failed",
        description: "An error occurred while checking the UPI ID",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'bg-destructive text-destructive-foreground';
      case 'medium':
        return 'bg-amber-500 text-white';
      case 'low':
        return 'bg-green-500 text-white';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getProgressColor = (score: number) => {
    if (score >= 70) return 'bg-destructive';
    if (score >= 40) return 'bg-amber-500';
    return 'bg-green-500';
  };

  const getRiskIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return <AlertTriangle className="h-6 w-6 text-destructive" />;
      case 'medium':
        return <AlertTriangle className="h-6 w-6 text-amber-500" />;
      case 'low':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      default:
        return <Shield className="h-6 w-6 text-muted-foreground" />;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button 
          className="w-full flex items-center gap-2 bg-primary text-white" 
          size="lg"
        >
          <Shield className="h-5 w-5" />
          UPI Scam Check
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Check UPI ID For Scams</DialogTitle>
          <DialogDescription>
            Enter a UPI ID to check its safety and fraud risk
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center space-x-2">
          <Input
            placeholder="Enter UPI ID (e.g., name@upi)"
            value={upiId}
            onChange={handleUpiIdChange}
            className="flex-1"
          />
          <Button 
            onClick={checkUpi} 
            disabled={isLoading}
          >
            {isLoading ? "Checking..." : "Check"}
          </Button>
        </div>

        {results && (
          <div className="mt-4 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center justify-between">
                  <span>Risk Assessment</span>
                  <span 
                    className={`px-2 py-1 text-xs rounded-full ${getRiskColor(results.riskLevel)}`}
                  >
                    {results.status}
                  </span>
                </CardTitle>
                <CardDescription>
                  {results.upiId}
                </CardDescription>
              </CardHeader>
              <CardContent className="pb-2">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Risk Score</span>
                    <span className="font-medium">{results.riskPercentage}%</span>
                  </div>
                  <Progress 
                    value={results.riskPercentage} 
                    className={getProgressColor(results.riskPercentage)}
                  />
                  
                  <div className="bg-muted p-3 rounded text-sm mt-4">
                    <span className="font-medium">Reason: </span>{results.reason}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 mt-4">
                    <div className="bg-muted p-2 rounded text-center">
                      <div className="text-sm text-muted-foreground">Reports</div>
                      <div className="text-lg font-medium">{results.reports}</div>
                    </div>
                    <div className="bg-muted p-2 rounded text-center">
                      <div className="text-sm text-muted-foreground">Confidence</div>
                      <div className="text-lg font-medium">{Math.round(results.confidence_score * 100)}%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {results.risk_factors && results.risk_factors.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Risk Factors</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-1 text-sm">
                    {results.risk_factors.map((factor: string, index: number) => (
                      <li key={index}>{factor}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {results.recommendations && results.recommendations.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Recommended Actions</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-1 text-sm">
                    {results.recommendations.map((action: string, index: number) => (
                      <li key={index}>{action}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {/* AI Analysis Card */}
            {results.safety_score !== undefined && results.ai_analysis && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <span>AI Safety Analysis</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      results.safety_score > 70 ? 'bg-green-500 text-white' :
                      results.safety_score > 40 ? 'bg-amber-500 text-white' :
                      'bg-destructive text-destructive-foreground'
                    }`}>
                      {results.safety_score}/100
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Safety Score</span>
                      <span className="font-medium">{results.safety_score}%</span>
                    </div>
                    <Progress 
                      value={results.safety_score} 
                      className={results.safety_score > 70 ? 'bg-green-500' : 
                                results.safety_score > 40 ? 'bg-amber-500' : 'bg-destructive'}
                    />
                    
                    <div className="bg-blue-50 p-3 rounded text-sm mt-2 text-blue-800">
                      {results.ai_analysis}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {results.reportedFor && results.reportedFor !== 'N/A' && (
              <div className="text-sm text-muted-foreground">
                <span className="font-medium">Reported For:</span> {results.reportedFor}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}