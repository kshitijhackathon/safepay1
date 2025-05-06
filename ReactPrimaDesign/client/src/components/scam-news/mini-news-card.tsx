import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Newspaper, ArrowRight, Loader2, AlertOctagon } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface ScamAlert {
  title: string;
  type: string;
  description: string;
  affected_areas: string[];
  risk_level: 'High' | 'Medium' | 'Low';
  date_reported: string;
  verification_status: 'Verified' | 'Investigating' | 'Unverified';
}

export function MiniNewsCard() {
  const [, setLocation] = useLocation();
  const [topAlert, setTopAlert] = useState<ScamAlert | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch the latest scam news alert
  useEffect(() => {
    async function fetchLatestNews() {
      try {
        setIsLoading(true);
        // Default location for demo
        const geolocation = 'Mumbai, Maharashtra';
        
        const response = await fetch('/api/scam-news', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            trigger_source: 'home_card',
            geo_location: geolocation
          })
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch scam news');
        }
        
        const result = await response.json();
        
        if (result.alerts && result.alerts.length > 0) {
          // Get the first (most recent/important) alert
          setTopAlert(result.alerts[0]);
        }
      } catch (error) {
        console.error('Error fetching mini scam news:', error);
      } finally {
        setIsLoading(false);
      }
    }
    
    fetchLatestNews();
  }, []);
  
  // Navigate to scam news page
  const handleClick = () => {
    if (topAlert) {
      // Store the selected alert for the detail view
      localStorage.setItem('selectedScamAlert', JSON.stringify(topAlert));
    }
    setLocation('/scam-news');
  };
  
  return (
    <div 
      className="w-full cursor-pointer"
      onClick={handleClick}
    >
      <div className="flex flex-col items-center">
        <div className="w-12 h-12 bg-blue-50 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-1 transition-colors duration-300 relative">
          <Newspaper className="w-5 h-5 text-blue-500 dark:text-blue-400" />
          
          {topAlert && topAlert.risk_level === 'High' && (
            <Badge 
              variant="destructive" 
              className="absolute -top-1 -right-1 h-4 w-4 p-0 flex items-center justify-center rounded-full"
            >
              <AlertOctagon className="h-3 w-3" />
            </Badge>
          )}
        </div>
        <span className="text-[10px] text-center dark:text-gray-300">Scam News</span>
      </div>
    </div>
  );
}