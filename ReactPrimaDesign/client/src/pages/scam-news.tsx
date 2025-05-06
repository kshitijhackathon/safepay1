import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { 
  Card, 
  CardContent, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { 
  AlertOctagon, 
  AlertTriangle, 
  Shield, 
  Activity, 
  MapPin, 
  ChevronRight, 
  ArrowUpRight, 
  Search, 
  Loader2, 
  CheckCircle,
  Info,
  ArrowLeft
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { NewsDetail } from '@/components/scam-news/news-detail';
import { Dialog, DialogContent } from '@/components/ui/dialog';

interface ScamAlert {
  title: string;
  type: string;
  description: string;
  affected_areas: string[];
  risk_level: 'High' | 'Medium' | 'Low';
  date_reported: string;
  verification_status: 'Verified' | 'Investigating' | 'Unverified';
}

interface HeatmapPoint {
  city: string;
  coordinates: [number, number];
  intensity: number;
  scam_type: string;
  trend: 'Increasing' | 'Stable' | 'Decreasing';
}

interface PreventionTip {
  tip: string;
  category: string;
}

interface ReportsSummary {
  total_reports: number;
  most_reported: string[];
  financial_loss: string;
  emerging_patterns: string[];
  hotspot_areas: string[];
}

interface UpiAnalysis {
  risk_level: string;
  confidence?: number;
  analysis: string;
  flags?: string[];
  recommendations?: string[];
}

interface ScamNewsData {
  alerts: ScamAlert[];
  geo_spread: HeatmapPoint[];
  prevention_tips: PreventionTip[];
  reports_summary: ReportsSummary;
  upi_analysis: UpiAnalysis | null;
  trust_score: number;
  last_updated: string;
}

export default function ScamNews() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  const [data, setData] = useState<ScamNewsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('alerts');
  const [upiSearch, setUpiSearch] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<ScamAlert | null>(null);
  const [showDetailDialog, setShowDetailDialog] = useState(false);
  
  // Check if we have a saved alert from the home screen
  useEffect(() => {
    const savedAlert = localStorage.getItem('selectedScamAlert');
    if (savedAlert) {
      try {
        const alertData = JSON.parse(savedAlert);
        setSelectedAlert(alertData);
        setShowDetailDialog(true);
        // Clear the stored data after using it
        localStorage.removeItem('selectedScamAlert');
      } catch (error) {
        console.error('Error parsing saved alert:', error);
      }
    }
  }, []);
  
  // Fetch scam news data
  const fetchScamNews = async () => {
    if (isLoading) return;
    
    setIsLoading(true);
    
    try {
      // Attempt to get geolocation
      let geolocation = 'India'; // Default fallback
      
      try {
        // This is just a mock - in a real app you'd use the browser's geolocation API
        // and then do reverse geocoding to get the city/state
        const cities = ['Mumbai, Maharashtra', 'Delhi, NCR', 'Bengaluru, Karnataka', 
                        'Chennai, Tamil Nadu', 'Hyderabad, Telangana', 'Pune, Maharashtra'];
        // Use a random city to ensure we get different data each time
        geolocation = cities[Math.floor(Math.random() * cities.length)];
        console.log('Using location:', geolocation);
      } catch (geoError) {
        console.error('Error getting geolocation:', geoError);
      }
      
      // Setup a timeout to avoid waiting too long for scam news
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      let result;
      
      try {
        const response = await fetch('/api/scam-news', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            trigger_source: 'user_button',
            geo_location: geolocation,
            refresh_token: Date.now(), // Add timestamp to force fresh data generation
            use_fallback: true // Tell server to use fallback data if API call takes too long
          }),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId); // Clear the timeout if request completes
        
        if (!response.ok) {
          throw new Error('Failed to fetch scam news');
        }
        
        result = await response.json();
        setData(result);
      } catch (error) {
        const fetchError = error as Error;
        if (fetchError.name === 'AbortError') {
          console.warn('Scam news fetch timed out, using fallback data');
          // Try to get fallback data directly without waiting for Groq
          try {
            const fallbackResponse = await fetch('/api/scam-news/fallback', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ geo_location: geolocation })
            });
            
            if (fallbackResponse.ok) {
              result = await fallbackResponse.json();
              setData(result);
            } else {
              throw new Error('Failed to fetch fallback scam news');
            }
          } catch (fallbackError) {
            console.error('Error fetching fallback data:', fallbackError);
            throw new Error('Failed to fetch scam data');
          }
        } else {
          throw fetchError;
        }
      }
      
      // Only show toast if we have data
      if (result) {
        toast({
          title: 'Scam news updated',
          description: `Found ${result.alerts.length} new alerts for ${geolocation}`,
        });
      }
    } catch (error) {
      console.error('Error fetching scam news:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch scam alerts. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Search UPI ID for potential scams
  const searchUpiId = async () => {
    if (!upiSearch.trim() || isSearching) return;
    
    setIsSearching(true);
    
    try {
      const response = await fetch('/api/scam-news/analyze-upi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ upi_id: upiSearch.trim() })
      });
      
      if (!response.ok) {
        throw new Error('Failed to analyze UPI ID');
      }
      
      const result = await response.json();
      
      // Update the UPI analysis part of the data
      setData(prevData => prevData ? {
        ...prevData,
        upi_analysis: result.upi_analysis
      } : null);
      
      // Switch to UPI tab to show results
      setActiveTab('upi');
    } catch (error) {
      console.error('Error analyzing UPI ID:', error);
      toast({
        title: 'Analysis Error',
        description: 'Failed to analyze UPI ID. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsSearching(false);
    }
  };
  
  // Fetch data on component mount
  useEffect(() => {
    fetchScamNews();
    
    // Set up auto refresh every 24 hours
    const refreshInterval = setInterval(() => {
      fetchScamNews();
    }, 24 * 60 * 60 * 1000); // 24 hours
    
    return () => clearInterval(refreshInterval);
  }, []);
  
  // Handle UPI search form submission
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    searchUpiId();
  };
  
  // Handle alert selection for detail view
  const handleAlertSelect = (alert: ScamAlert) => {
    setSelectedAlert(alert);
    setShowDetailDialog(true);
  };
  
  // Handle report similar scam
  const handleReportSimilar = () => {
    setShowDetailDialog(false);
    setLocation('/report-scam');
  };
  
  // Render risk level badge with appropriate color
  const renderRiskBadge = (level: string) => {
    let variant: 
      | 'default'
      | 'secondary'
      | 'destructive'
      | 'outline' = 'default';
    let icon = null;
    
    if (level.toLowerCase() === 'high') {
      variant = 'destructive';
      icon = <AlertOctagon className="h-3 w-3 mr-1" />;
    } else if (level.toLowerCase() === 'medium') {
      variant = 'default';
      icon = <AlertTriangle className="h-3 w-3 mr-1" />;
    } else {
      variant = 'secondary';
      icon = <Shield className="h-3 w-3 mr-1" />;
    }
    
    return (
      <Badge variant={variant} className="flex items-center">
        {icon}
        {level}
      </Badge>
    );
  };
  
  // Render verification status badge
  const renderVerificationBadge = (status: string) => {
    if (status.toLowerCase() === 'verified') {
      return <Badge variant="outline" className="bg-green-50 border-green-200 text-green-600 flex items-center">
        <CheckCircle className="h-3 w-3 mr-1" />
        Verified
      </Badge>;
    } else if (status.toLowerCase() === 'investigating') {
      return <Badge variant="outline" className="bg-orange-50 border-orange-200 text-orange-600 flex items-center">
        <Activity className="h-3 w-3 mr-1" />
        Investigating
      </Badge>;
    } else {
      return <Badge variant="outline" className="bg-gray-50 border-gray-200 text-gray-600 flex items-center">
        <Info className="h-3 w-3 mr-1" />
        Unverified
      </Badge>;
    }
  };
  
  return (
    <div className="flex flex-col p-6 space-y-6">
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setLocation('/home')}
          className="w-9 h-9 bg-slate-100 dark:bg-gray-700 rounded-full flex items-center justify-center"
        >
          <ArrowLeft className="w-5 h-5 text-gray-700 dark:text-gray-300" />
        </button>
        <h2 className="text-xl font-bold">Scam News</h2>
        <div className="w-9"></div> {/* Empty div to maintain centering */}
      </div>
      
      <p className="text-gray-500 text-sm">Latest UPI scams and safety alerts in your area</p>
      
      {/* UPI ID search */}
      <Card>
        <CardContent className="pt-6">
          <form onSubmit={handleSearchSubmit} className="flex space-x-2 w-full">
            <Input
              placeholder="Enter UPI ID to analyze..."
              value={upiSearch}
              onChange={(e) => setUpiSearch(e.target.value)}
              className="flex-1"
            />
            <Button 
              type="submit" 
              disabled={!upiSearch.trim() || isSearching}
            >
              {isSearching ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Search className="h-4 w-4 mr-2" />
              )}
              Analyze
            </Button>
          </form>
        </CardContent>
      </Card>
      
      {/* News detail dialog */}
      <Dialog open={showDetailDialog} onOpenChange={setShowDetailDialog}>
        <DialogContent className="sm:max-w-xl max-h-[90vh] overflow-y-auto">
          {selectedAlert && (
            <NewsDetail 
              alert={selectedAlert} 
              onClose={() => setShowDetailDialog(false)} 
              onReportSimilar={handleReportSimilar}
            />
          )}
        </DialogContent>
      </Dialog>
      
      {/* Main content tabs */}
      <Tabs 
        value={activeTab} 
        onValueChange={setActiveTab}
        className="w-full"
      >
        <TabsList className="grid grid-cols-3">
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="heatmap">Reports</TabsTrigger>
          <TabsTrigger value="upi">UPI Analysis</TabsTrigger>
        </TabsList>
        
        {/* Alerts Tab */}
        <TabsContent value="alerts">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle>Recent Scam Alerts</CardTitle>
              {data && (
                <div className="text-sm text-gray-500">
                  Last updated: {new Date(data.last_updated).toLocaleString()}
                </div>
              )}
            </CardHeader>
            
            <CardContent className="pt-0">
              {isLoading && !data ? (
                <div className="h-60 flex items-center justify-center">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : data && data.alerts.length > 0 ? (
                <ScrollArea className="h-[60vh] pr-4">
                  <div className="space-y-4">
                    {data.alerts.map((alert, index) => (
                      <Card 
                        key={index} 
                        className="overflow-hidden cursor-pointer hover:shadow-md transition-shadow"
                        onClick={() => handleAlertSelect(alert)}
                      >
                        <div className="flex flex-col">
                          <div className="p-4">
                            <div className="flex justify-between items-start mb-2">
                              <h3 className="font-bold text-lg">{alert.title}</h3>
                              {renderRiskBadge(alert.risk_level)}
                            </div>
                            
                            <div className="flex items-center mb-3 text-sm space-x-2">
                              <Badge variant="outline" className="bg-blue-50 border-blue-200 text-blue-600 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-400">
                                {alert.type}
                              </Badge>
                              {renderVerificationBadge(alert.verification_status)}
                            </div>
                            
                            <p className="text-gray-700 dark:text-gray-300 mb-3">{alert.description}</p>
                            
                            <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                              <MapPin className="h-3 w-3 mr-1" />
                              {alert.affected_areas.join(', ')}
                            </div>
                          </div>
                          
                          <div className="bg-gray-50 dark:bg-gray-800 px-4 py-2 text-xs text-gray-500 dark:text-gray-400 flex justify-between items-center">
                            <span>Reported on {new Date(alert.date_reported).toLocaleDateString()}</span>
                            <ChevronRight className="h-4 w-4" />
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="h-40 flex items-center justify-center">
                  <div className="text-center">
                    <Shield className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                    <p>No scam alerts found for your area</p>
                  </div>
                </div>
              )}
            </CardContent>
            
            <CardFooter>
              <Button 
                onClick={fetchScamNews} 
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Updating...
                  </>
                ) : (
                  <>
                    <ArrowUpRight className="h-4 w-4 mr-2" />
                    Refresh Alerts
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Reports Tab */}
        <TabsContent value="heatmap">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle>Scam Reports Summary</CardTitle>
            </CardHeader>
            
            <CardContent className="pt-0">
              {isLoading && !data ? (
                <div className="h-60 flex items-center justify-center">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : data ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <Card className="p-4">
                      <div className="text-3xl font-bold mb-1">{data.reports_summary.total_reports}</div>
                      <div className="text-sm text-gray-500">Total reports this week</div>
                    </Card>
                    
                    <Card className="p-4">
                      <div className="text-lg font-medium mb-1">{data.reports_summary.financial_loss}</div>
                      <div className="text-sm text-gray-500">Average financial loss</div>
                    </Card>
                  </div>
                  
                  <div>
                    <h3 className="font-medium mb-2">Most Reported Scams</h3>
                    <div className="space-y-2">
                      {data.reports_summary.most_reported.map((scam, index) => (
                        <div key={index} className="flex justify-between items-center">
                          <span>{scam}</span>
                          <Badge variant="outline">{Math.round(100 - index * 15)}%</Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h3 className="font-medium mb-3">Prevention Tips</h3>
                    <div className="space-y-3">
                      {data.prevention_tips.map((tip, index) => (
                        <div key={index} className="flex space-x-2">
                          <CheckCircle className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
                          <div>
                            <p className="text-gray-800">{tip.tip}</p>
                            <Badge className="mt-1">{tip.category}</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h3 className="font-medium mb-2">Emerging Scam Patterns</h3>
                    <ul className="list-disc pl-5 space-y-1">
                      {data.reports_summary.emerging_patterns.map((pattern, index) => (
                        <li key={index} className="text-gray-700">{pattern}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h3 className="font-medium mb-2">Hotspot Areas</h3>
                    <div className="grid grid-cols-2 gap-2">
                      {data.reports_summary.hotspot_areas.map((area, index) => (
                        <Badge key={index} variant="outline" className="justify-start">
                          <MapPin className="h-3 w-3 mr-1" />
                          {area}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-40 flex items-center justify-center">
                  <div className="text-center">
                    <Activity className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                    <p>No report data available</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* UPI Analysis Tab */}
        <TabsContent value="upi">
          <Card>
            <CardHeader>
              <CardTitle>UPI ID Risk Analysis</CardTitle>
            </CardHeader>
            
            <CardContent>
              {isSearching ? (
                <div className="h-60 flex items-center justify-center">
                  <div className="text-center">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-3 text-primary" />
                    <p>Analyzing UPI ID...</p>
                  </div>
                </div>
              ) : data && data.upi_analysis ? (
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-lg">{upiSearch}</span>
                      <Badge 
                        variant={
                          data.upi_analysis.risk_level.toLowerCase() === 'high' ? 'destructive' : 
                          data.upi_analysis.risk_level.toLowerCase() === 'medium' ? 'default' : 
                          'secondary'
                        }
                      >
                        {data.upi_analysis.risk_level} Risk
                      </Badge>
                    </div>
                    
                    {data.upi_analysis.confidence && (
                      <div className="text-sm text-gray-500">
                        Confidence: {Math.round(data.upi_analysis.confidence * 100)}%
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <h3 className="font-medium mb-2">Analysis</h3>
                    <p className="text-gray-700">{data.upi_analysis.analysis}</p>
                  </div>
                  
                  {data.upi_analysis.flags && data.upi_analysis.flags.length > 0 && (
                    <div>
                      <h3 className="font-medium mb-2">Risk Flags</h3>
                      <ul className="list-disc pl-5 space-y-1">
                        {data.upi_analysis.flags.map((flag, index) => (
                          <li key={index} className="text-red-600">{flag}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {data.upi_analysis.recommendations && data.upi_analysis.recommendations.length > 0 && (
                    <div>
                      <h3 className="font-medium mb-2">Recommendations</h3>
                      <ul className="list-disc pl-5 space-y-1">
                        {data.upi_analysis.recommendations.map((rec, index) => (
                          <li key={index} className="text-gray-700">{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <div className="pt-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Trust Score</span>
                      <span className="text-sm text-gray-500">
                        {data.trust_score}%
                      </span>
                    </div>
                    <Progress value={data.trust_score} className="h-2" />
                  </div>
                </div>
              ) : (
                <div className="h-60 flex items-center justify-center">
                  <div className="text-center">
                    <Search className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                    <p>Enter a UPI ID above to analyze its risk</p>
                  </div>
                </div>
              )}
            </CardContent>
            
            {data && data.upi_analysis && (
              <CardFooter>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => setLocation('/report-scam')}
                >
                  <AlertOctagon className="h-4 w-4 mr-2" />
                  Report This UPI ID
                </Button>
              </CardFooter>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}