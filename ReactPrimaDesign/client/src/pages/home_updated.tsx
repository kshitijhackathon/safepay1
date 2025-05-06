import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { Bell, ShieldAlert, Search, ArrowRight, MoonIcon, SunIcon, Video } from 'lucide-react';
import { NotificationBar } from '@/components/ui/notification-bar';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from '@/hooks/useTheme';
import { useAuthState } from '@/hooks/use-auth-state';
import { VideoDetectionHomeButton } from '@/components/ui/video-detection-home-button';

export default function Home() {
  const [, setLocation] = useLocation();
  const [showNotification, setShowNotification] = useState(false);
  const [upiInput, setUpiInput] = useState('');
  const { toast } = useToast();
  const { isDark, setTheme } = useTheme();
  const { authState } = useAuthState();

  const handleAlertClick = () => {
    setShowNotification(true);
  };
  
  const handleUpiSearch = () => {
    if (!upiInput.trim()) {
      toast({
        title: "Empty Input",
        description: "Please enter a UPI ID to search",
        variant: "destructive",
      });
      return;
    }
    
    // If UPI format is valid (contains @), process it directly
    if (upiInput.includes('@')) {
      // Process the UPI ID - use the same route as scan.tsx
      const queryParams = new URLSearchParams();
      queryParams.append('upiId', upiInput);
      queryParams.append('fromSearch', 'true');
      
      setLocation(`/scan?${queryParams.toString()}`);
    } else {
      // Not in UPI format, add a default provider for demo
      const demoUpiId = upiInput + '@okaxis';
      toast({
        title: "Processing",
        description: `Using demo format: ${demoUpiId}`,
      });
      
      const queryParams = new URLSearchParams();
      queryParams.append('upiId', demoUpiId);
      queryParams.append('fromSearch', 'true');
      
      setLocation(`/scan?${queryParams.toString()}`);
    }
  };

  return (
    <div className="dark-bg-secondary h-screen overflow-hidden fixed inset-0 flex flex-col">
      {/* Top bar with search */}
      <div className="p-4 dark-bg-primary z-10 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-slate-100 dark:bg-gray-700 rounded-full px-3 py-1.5 flex items-center transition-colors duration-300">
            <Search className="w-4 h-4 dark-text-tertiary mr-2 flex-shrink-0" />
            <Input 
              className="border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-slate-500 dark:placeholder:text-gray-400 text-sm w-full h-8 dark-text-primary"
              placeholder="Enter UPI ID to check..."
              value={upiInput}
              onChange={(e) => setUpiInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleUpiSearch();
                }
              }}
            />
            {upiInput && (
              <Button 
                size="sm" 
                className="rounded-full h-7 w-7 p-0 flex items-center justify-center bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800"
                onClick={handleUpiSearch}
              >
                <ArrowRight className="w-4 h-4" />
              </Button>
            )}
          </div>
          <button 
            onClick={() => setTheme(isDark ? 'light' : 'dark')}
            className="w-8 h-8 rounded-full flex items-center justify-center bg-slate-100 dark:bg-gray-700 text-slate-500 dark:text-gray-300 hover:bg-slate-200 dark:hover:bg-gray-600 transition-colors duration-300"
          >
            {isDark ? <SunIcon size={16} /> : <MoonIcon size={16} />}
          </button>
        </div>
      </div>
      
      {/* Main content area - fixed height and scrollable if needed */}
      <div className="flex-1 overflow-y-auto pb-16">
        {/* Alert button */}
        <div className="px-4 py-6 flex justify-center">
          <Button 
            className="w-16 h-16 rounded-full bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-800 text-white transition-colors duration-300"
            onClick={handleAlertClick}
          >
            <ShieldAlert className="w-6 h-6" />
          </Button>
        </div>
        
        {/* Menu items - symmetrically arranged grid */}
        <div className="px-4">
          {/* Row 1 - 3 items per row */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <button 
              onClick={() => setLocation('/scam-news')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 bg-blue-50 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-1 transition-colors duration-300 relative">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 01-2.25 2.25M16.5 7.5V18a2.25 2.25 0 002.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 002.25 2.25h13.5M6 7.5h3v3H6v-3z"
                  />
                </svg>
                <div className="absolute -top-1 -right-1 h-4 w-4 bg-red-500 rounded-full flex items-center justify-center text-white text-[8px] font-bold">
                  !
                </div>
              </div>
              <span className="text-[10px] text-center dark:text-gray-300">Scam News</span>
            </button>
            
            <button 
              onClick={() => setLocation('/report-scam')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Report Scams</span>
            </button>
            
            <button 
              onClick={() => setLocation('/history')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">History</span>
            </button>
          </div>
          
          {/* Row 2 - 3 items per row */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <button 
              onClick={() => setLocation('/voice-check')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Voice Scam</span>
            </button>
            
            <button 
              onClick={() => setLocation('/message-check')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M2.25 12.76c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 011.037-.443 48.282 48.282 0 005.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Message Check</span>
            </button>
            
            <button 
              onClick={() => setLocation('/whatsapp-check')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">WhatsApp Check</span>
            </button>
          </div>
          
          {/* Row 3 - 3 items per row */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <VideoDetectionHomeButton />
            
            <button 
              onClick={() => setLocation('/risk-score-demo')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Risk Score Demo</span>
            </button>
            
            <button 
              onClick={() => setLocation('/upi-check')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 01-1.043 3.296 3.745 3.745 0 01-3.296 1.043A3.745 3.745 0 0112 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 01-3.296-1.043 3.745 3.745 0 01-1.043-3.296A3.745 3.745 0 013 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 011.043-3.296 3.746 3.746 0 013.296-1.043A3.746 3.746 0 0112 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 013.296 1.043 3.746 3.746 0 011.043 3.296A3.745 3.745 0 0121 12z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">UPI Check</span>
            </button>
          </div>
          
          {/* Row 4 - 3 items per row */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <button 
              onClick={() => setLocation('/my-reports')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M11.35 3.836c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m8.9-4.414c.376.023.75.05 1.124.08 1.131.094 1.976 1.057 1.976 2.192V16.5A2.25 2.25 0 0118 18.75h-2.25m-7.5-10.5H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V18.75m-7.5-10.5h6.375c.621 0 1.125.504 1.125 1.125v9.375m-8.25-3l1.5 1.5 3-3.75" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">My Reports</span>
            </button>
            
            <button 
              onClick={() => setLocation('/legal-help')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0012 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.988 5.988 0 01-2.031.352a5.988 5.988 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L18.75 4.971zm-16.5.52c.99-.203 1.99-.377 3-.52m0 0l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.989 5.989 0 01-2.031.352 5.989 5.989 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L5.25 4.971z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Legal Help</span>
            </button>
            
            <button 
              onClick={() => setLocation('/chat-support')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" 
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Help & Support</span>
            </button>
          </div>
          
          {/* Row 5 - Just one item with Security */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <button 
              onClick={() => setLocation('/security-settings')}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-5 h-5 text-blue-500 dark:text-blue-400"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z"
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center dark-text-secondary">Security</span>
            </button>
            {/* Empty space for symmetry */}
            <div></div>
            {/* Empty space for symmetry */}
            <div></div>
          </div>
        </div>
      </div>
      
      {/* Notification Bar */}
      {showNotification && (
        <NotificationBar
          title="Security Alert"
          description="Recent suspicious activity has been detected in your area. Please be vigilant with unknown UPI requests."
          onClose={() => setShowNotification(false)}
        />
      )}
      
      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
}