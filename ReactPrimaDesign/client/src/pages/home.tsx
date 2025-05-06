import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { 
  Bell, ShieldAlert, Search, ArrowRight, MoonIcon, SunIcon, Video, 
  Zap, Calculator, CreditCard, FileText, Gauge, Settings, ChevronRight,
  Gift, HelpCircle, Lock, MessageSquare, Phone, Users, AlertTriangle
} from 'lucide-react';
import { NotificationBar } from '@/components/ui/notification-bar';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from '@/hooks/useTheme';
import { useAuthState } from '@/hooks/use-auth-state';
// import { VideoDetectionHomeButton } from '@/components/ui/video-detection-home-button';

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

          
          {/* Grid layout with 4 columns and 2 rows (8 items total) */}
          <div className="grid grid-cols-4 gap-3 mb-6">
            {/* Row 1 - 4 items */}
            {/* Item 1: Scan and Pay */}
            <button 
              onClick={() => setLocation('/qr-scan')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  className="w-6 h-6 text-blue-500"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <rect x="7" y="7" width="3" height="3"></rect>
                  <rect x="14" y="7" width="3" height="3"></rect>
                  <rect x="7" y="14" width="3" height="3"></rect>
                  <rect x="14" y="14" width="3" height="3"></rect>
                </svg>
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Scan & Pay</span>
            </button>
            
            {/* Item 2: Scam News */}
            <button 
              onClick={() => setLocation('/scam-news')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-6 h-6 text-blue-500"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 01-2.25 2.25M16.5 7.5V18a2.25 2.25 0 002.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 002.25 2.25h13.5M6 7.5h3v3H6v-3z"
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Scam News</span>
            </button>
            
            {/* Item 3: Voice Check */}
            <button 
              onClick={() => setLocation('/voice-check')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <Phone className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Voice Check</span>
            </button>
            
            {/* Item 4: Report Scam */}
            <button 
              onClick={() => setLocation('/report-scam')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <AlertTriangle className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Report Scam</span>
            </button>
            
            {/* Row 2 - 4 items */}
            {/* Item 5: Security */}
            <button 
              onClick={() => setLocation('/security-settings')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <Lock className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Security</span>
            </button>
            
            {/* Item 6: Help */}
            <button 
              onClick={() => setLocation('/chat-support')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <HelpCircle className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Help</span>
            </button>
            
            {/* Item 7: WhatsApp Check */}
            <button 
              onClick={() => setLocation('/whatsapp-check')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <MessageSquare className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">WhatsApp</span>
            </button>
            
            {/* Item 8: View All Services */}
            <button 
              onClick={() => setLocation('/all-services')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <ChevronRight className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">View All</span>
            </button>
          </div>
        </div>
      </div>
      
      {/* Notification Bar */}
      {showNotification && (
        <NotificationBar
          message="Recent suspicious activity has been detected in your area. Please be vigilant with unknown UPI requests."
          onClose={() => setShowNotification(false)}
        />
      )}
      
      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
}