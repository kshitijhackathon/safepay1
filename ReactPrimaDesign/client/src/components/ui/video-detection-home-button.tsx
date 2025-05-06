import React from 'react';
import { useLocation } from "wouter";
import { Video } from "lucide-react";

export function VideoDetectionHomeButton() {
  const [, setLocation] = useLocation();

  return (
    <button 
      onClick={() => setLocation('/video-detection')}
      className="flex flex-col items-center"
    >
      <div className="w-12 h-12 dark-bg-tertiary rounded-lg flex items-center justify-center mb-1">
        <Video className="w-5 h-5 text-blue-500 dark:text-blue-400" />
      </div>
      <span className="text-[10px] text-center dark-text-secondary">Video Scan</span>
    </button>
  );
}