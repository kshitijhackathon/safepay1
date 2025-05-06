import React from 'react';
import { useLocation } from 'wouter';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { ChevronLeft, CreditCard, FileText, Gift, Headphones, Monitor, ChevronsUpDown, 
  Landmark, BarChart, HelpCircle, Info, MessageSquare, Phone, Users, Video, 
  History, AlertTriangle, BookOpen, MapPin } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function AllServices() {
  const [, setLocation] = useLocation();

  // Service categories with their respective items
  const serviceCategories = [
    {
      title: "Scam Detection",
      items: [
        { name: "UPI Check", icon: <CreditCard className="w-5 h-5 text-blue-500" />, path: "/upi-check" },
        { name: "Voice Check", icon: <Phone className="w-5 h-5 text-blue-500" />, path: "/voice-check" },
        { name: "Video Check", icon: <Video className="w-5 h-5 text-blue-500" />, path: "/video-check" },
        { name: "Message Check", icon: <MessageSquare className="w-5 h-5 text-blue-500" />, path: "/message-check" },
        { name: "WhatsApp Check", icon: <MessageSquare className="w-5 h-5 text-blue-500" />, path: "/whatsapp-check" },
        { name: "Call Check", icon: <Headphones className="w-5 h-5 text-blue-500" />, path: "/call-check" },
      ]
    },
    {
      title: "Reports & Analysis",
      items: [
        { name: "Risk Score", icon: <BarChart className="w-5 h-5 text-blue-500" />, path: "/risk-score-demo" },
        { name: "My Reports", icon: <FileText className="w-5 h-5 text-blue-500" />, path: "/my-reports" },
        { name: "History", icon: <History className="w-5 h-5 text-blue-500" />, path: "/history" },
        { name: "Report Scam", icon: <AlertTriangle className="w-5 h-5 text-blue-500" />, path: "/report-scam" },
        { name: "3D Fraud Map", icon: <MapPin className="w-5 h-5 text-blue-500" />, path: "/fraud-map" },

      ]
    },
    {
      title: "Support & Help",
      items: [
        { name: "Security", icon: <Monitor className="w-5 h-5 text-blue-500" />, path: "/security-settings" },
        { name: "Legal Help", icon: <Landmark className="w-5 h-5 text-blue-500" />, path: "/legal-help" },
        { name: "Chat Support", icon: <HelpCircle className="w-5 h-5 text-blue-500" />, path: "/chat-support" },
        { name: "Scam News", icon: <BookOpen className="w-5 h-5 text-blue-500" />, path: "/scam-news" },
        { name: "About Us", icon: <Info className="w-5 h-5 text-blue-500" />, path: "/about" },
        { name: "Community", icon: <Users className="w-5 h-5 text-blue-500" />, path: "/community" },
      ]
    },
  ];

  return (
    <div className="dark-bg-secondary min-h-screen flex flex-col">
      {/* Top bar */}
      <div className="bg-white dark:bg-gray-900 p-4 flex items-center shadow-sm">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => setLocation('/home')}
          className="mr-2"
        >
          <ChevronLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-medium dark-text-primary">All Services</h1>
      </div>
      
      {/* Content area */}
      <div className="flex-1 overflow-y-auto pb-16 p-4">
        {serviceCategories.map((category, index) => (
          <div key={index} className="mb-8">
            <h2 className="text-sm font-semibold mb-4 dark-text-primary">{category.title}</h2>
            <div className="grid grid-cols-3 gap-4">
              {category.items.map((item, itemIndex) => (
                <button 
                  key={itemIndex}
                  onClick={() => setLocation(item.path)}
                  className="flex flex-col items-center"
                >
                  <div className="w-16 h-16 bg-blue-50 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-1">
                    {item.icon}
                  </div>
                  <span className="text-[10px] text-center dark-text-secondary">{item.name}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
}