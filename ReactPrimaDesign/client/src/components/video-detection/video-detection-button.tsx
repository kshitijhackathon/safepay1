import React from 'react';
import { Button } from "@/components/ui/button";
import { Video, Play, AlertTriangle } from "lucide-react";
import { useLocation } from 'wouter';

export function VideoDetectionButton() {
  const [_, setLocation] = useLocation();
  
  const handleClick = () => {
    setLocation('/video-detection');
  };
  
  return (
    <Button 
      onClick={handleClick}
      className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-md"
    >
      <Video className="h-5 w-5" />
      <span>Video Scam Detection</span>
    </Button>
  );
}