import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { 
  Card, 
  CardContent, 
  CardFooter, 
  CardHeader, 
  CardTitle,
  CardDescription
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { 
  AlertOctagon, 
  ChevronRight, 
  Loader2, 
  Clock, 
  RefreshCw
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ScamAlert {
  title: string;
  type: string;
  description: string;
  affected_areas: string[];
  risk_level: 'High' | 'Medium' | 'Low';
  date_reported: string;
  verification_status: 'Verified' | 'Investigating' | 'Unverified';
}

interface ScamNewsData {
  alerts: ScamAlert[];
  last_updated: string;
}

export function ScamNewsCard() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [news, setNews] = useState<ScamAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [refreshing, setRefreshing] = useState(false);
  
  // Function to fetch latest scam news
  const fetchLatestNews = async (showToast = false) => {
    try {
      if (refreshing) return;
      
      setRefreshing(true);
      
      // Attempt to get geolocation
      let geolocation = 'India'; // Default fallback
      
      try {
        // In a real app, this would use the browser's geolocation API
        geolocation = 'Mumbai, Maharashtra'; 
      } catch (geoError) {
        console.error('Error getting geolocation:', geoError);
      }
      
      const response = await fetch('/api/scam-news', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trigger_source: 'auto_refresh',
          geo_location: geolocation
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch scam news');
      }
      
      const result = await response.json();
      
      // Store only the latest 5 alerts for the card
      setNews(result.alerts.slice(0, 5));
      setLastUpdated(result.last_updated);
      
      if (showToast) {
        toast({
          title: 'Scam news updated',
          description: `Latest alerts loaded`,
        });
      }
    } catch (error) {
      console.error('Error fetching scam news:', error);
      if (showToast) {
        toast({
          title: 'Error',
          description: 'Could not refresh scam alerts',
          variant: 'destructive',
        });
      }
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };
  
  // Initially fetch data
  useEffect(() => {
    fetchLatestNews();
    
    // Set up a timer to refresh the news every 24 hours
    const refreshInterval = setInterval(() => {
      fetchLatestNews();
    }, 24 * 60 * 60 * 1000); // 24 hours
    
    return () => clearInterval(refreshInterval);
  }, []);
  
  // Handle click on a news item
  const handleNewsClick = (alert: ScamAlert) => {
    // Store the selected alert in localStorage for the detail page
    localStorage.setItem('selectedScamAlert', JSON.stringify(alert));
    setLocation('/scam-news');
  };
  
  // Format the relative time
  const getRelativeTime = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 24) {
      return `${diffInHours} hour${diffInHours !== 1 ? 's' : ''} ago`;
    } else {
      const diffInDays = Math.floor(diffInHours / 24);
      return `${diffInDays} day${diffInDays !== 1 ? 's' : ''} ago`;
    }
  };
  
  return (
    <Card className="border-none shadow-sm bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">Latest Scam Alerts</CardTitle>
          {lastUpdated && (
            <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
              <Clock className="h-3 w-3 mr-1" />
              Updated {getRelativeTime(lastUpdated)}
            </div>
          )}
        </div>
        <CardDescription>Stay informed about recent financial scams</CardDescription>
      </CardHeader>
      
      <CardContent className="pb-2">
        {isLoading ? (
          <div className="h-32 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : news.length > 0 ? (
          <div className="space-y-3">
            {news.map((alert, index) => (
              <div key={index}>
                <div 
                  className="cursor-pointer hover:bg-blue-100/50 dark:hover:bg-blue-900/30 p-2 rounded-lg transition-colors"
                  onClick={() => handleNewsClick(alert)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h3 className="font-medium text-sm line-clamp-1">{alert.title}</h3>
                      <p className="text-xs text-gray-500 dark:text-gray-400 line-clamp-2 mt-1">
                        {alert.description}
                      </p>
                    </div>
                    <Badge 
                      variant={alert.risk_level.toLowerCase() === 'high' ? 'destructive' : 'outline'}
                      className="ml-2 shrink-0 text-xs"
                    >
                      {alert.risk_level.toLowerCase() === 'high' && (
                        <AlertOctagon className="h-3 w-3 mr-1" />
                      )}
                      {alert.risk_level}
                    </Badge>
                  </div>
                </div>
                {index < news.length - 1 && <Separator className="my-2" />}
              </div>
            ))}
          </div>
        ) : (
          <div className="h-32 flex items-center justify-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">No recent alerts available</p>
          </div>
        )}
      </CardContent>
      
      <CardFooter className="flex justify-between">
        <Button 
          variant="ghost" 
          size="sm" 
          className="text-xs flex items-center"
          onClick={() => fetchLatestNews(true)}
          disabled={refreshing}
        >
          {refreshing ? (
            <Loader2 className="h-3 w-3 animate-spin mr-1" />
          ) : (
            <RefreshCw className="h-3 w-3 mr-1" />
          )}
          Refresh
        </Button>
        
        <Button 
          variant="ghost" 
          size="sm" 
          className="text-xs flex items-center"
          onClick={() => setLocation('/scam-news')}
        >
          View All
          <ChevronRight className="h-3 w-3 ml-1" />
        </Button>
      </CardFooter>
    </Card>
  );
}